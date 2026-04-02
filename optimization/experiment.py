"""
Experiment runner — orchestrates model x params x horizon evaluation.

For each model and horizon:
1. Optimize parameters via ParameterSearch
2. Run walk-forward backtest with best params
3. Persist all granular data (per-trial, per-window) to results/
4. Collect results for leaderboard
"""
import json
import numpy as np
import pandas as pd
import yaml
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from models.base import BaseModel, SimulationResult
from models.registry import get_model, list_models
from validation.scorer import Scorer, ScoreCard
from validation.backtest import WalkForwardBacktest
from .search import ParameterSearch
from .sensitivity import SensitivitySearch


console = Console()


# Default run mode settings
RUN_MODE_DEFAULTS = {
    "quick": {
        "n_simulations_search": 500,
        "n_simulations_final": 2000,
        "search_phase": "sobol_only",
        "trial_budgets": {"gbm": 20, "garch_1_1": 15, "regime_block_bootstrap": 30},
        "horizons": ["180d", "365d", "730d", "1460d"],
        "seeds": [42],
    },
    "standard": {
        "n_simulations_search": 1000,
        "n_simulations_final": 5000,
        "search_phase": "sobol_then_tpe",
        "trial_budgets": {"gbm": 30, "garch_1_1": 80, "regime_block_bootstrap": 150},
        "horizons": ["30d", "90d", "180d", "365d", "1460d"],
        "seeds": [42],
    },
    "full": {
        "n_simulations_search": 2000,
        "n_simulations_final": 10000,
        "search_phase": "sobol_then_tpe",
        "trial_budgets": {"gbm": 50, "garch_1_1": 150, "regime_block_bootstrap": 300},
        "horizons": ["30d", "90d", "180d", "365d", "1460d"],
        "seeds": [42, 123, 456, 789, 1024],
    },
}


def parse_horizon(horizon_str: str) -> int:
    """Convert horizon string like '30d' to integer days."""
    return int(horizon_str.replace("d", ""))


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


class ExperimentRunner:
    """
    Full experiment orchestrator.

    Runs model x horizon grid with parameter optimization and
    walk-forward backtesting. Persists all granular data for
    dashboard consumption.

    Parameters
    ----------
    config : dict
        Experiment configuration (from YAML or constructed directly).
    train_returns : np.ndarray
        Training log returns (1D).
    train_prices : np.ndarray
        Training prices (1D).
    test_prices : np.ndarray
        Out-of-sample prices (1D).
    test_returns : np.ndarray
        Out-of-sample log returns (1D).
    dates : np.ndarray, optional
        Full date array aligned with concatenated prices.
    train_end_idx : int, optional
        Index where training ends in the full series.
    """

    def __init__(
        self,
        config: dict,
        train_returns: np.ndarray,
        train_prices: np.ndarray,
        test_prices: np.ndarray,
        test_returns: np.ndarray,
        dates: np.ndarray | None = None,
        train_end_idx: int | None = None,
    ):
        self.config = config
        self.train_returns = train_returns
        self.train_prices = train_prices
        self.test_prices = test_prices
        self.test_returns = test_returns
        self.dates = dates
        self.train_end_idx = train_end_idx or len(train_prices)

        # Full series for walk-forward
        self.full_prices = np.concatenate([train_prices, test_prices[1:]])
        self.full_returns = np.concatenate([train_returns, test_returns])

        # Resolve run mode
        run_mode = config.get("experiment", {}).get("run_mode", "quick")
        self.run_mode = run_mode
        self.mode_defaults = RUN_MODE_DEFAULTS[run_mode]

        # Merge config over defaults
        sim_config = config.get("simulation", {})
        self.n_sims_search = sim_config.get(
            "n_simulations_search", self.mode_defaults["n_simulations_search"]
        )
        self.n_sims_final = sim_config.get(
            "n_simulations_final", self.mode_defaults["n_simulations_final"]
        )
        self.search_phase = sim_config.get(
            "search_phase", self.mode_defaults["search_phase"]
        )
        self.trial_budgets = sim_config.get(
            "trial_budgets", self.mode_defaults["trial_budgets"]
        )
        self.seeds = sim_config.get("seeds", self.mode_defaults["seeds"])

        # Walk-forward config
        wf_config = config.get("walk_forward", {})
        self.wf_step_size = wf_config.get("step_size_days", 180)
        self.wf_min_train = wf_config.get("min_training_days", 730)
        self.wf_recency = wf_config.get("recency_weighting", "exponential")
        self.wf_half_life = wf_config.get("recency_half_life_days", 730)

        # Models and horizons from config
        self.model_names = config.get("models", ["gbm"])
        self.horizons = config.get("horizons", self.mode_defaults["horizons"])

        # Scoring
        weights = config.get("scoring", {}).get("weights", None)
        self.scorer = Scorer(weights=weights)

        # Base seed
        self.base_seed = config.get("experiment", {}).get("seed", 42)

        # Results storage
        self.results: list[ScoreCard] = []
        self.granular_data: dict[str, Any] = {
            "experiment_name": config.get("experiment", {}).get("name", "unnamed"),
            "run_mode": run_mode,
            "timestamp": datetime.now().isoformat(),
            "config": _make_serializable(config),
            "data_info": {
                "train_days": len(train_returns),
                "test_days": len(test_returns),
                "train_start": str(dates[0]) if dates is not None else None,
                "train_end": str(dates[self.train_end_idx - 1]) if dates is not None else None,
                "test_start": str(dates[self.train_end_idx]) if dates is not None else None,
                "test_end": str(dates[-1]) if dates is not None else None,
            },
            "runs": [],
        }

    def _idx_to_date(self, idx: int) -> str | None:
        """Convert array index to date string."""
        if self.dates is not None and 0 <= idx < len(self.dates):
            return str(self.dates[idx])
        return None

    def _make_objective(
        self, model: BaseModel, horizon_days: int
    ) -> callable:
        """Create objective function for parameter search."""
        def objective(m: BaseModel) -> float:
            m.fit(self.train_returns)
            initial_price = self.train_prices[-1]

            result = m.simulate(
                n_simulations=self.n_sims_search,
                n_steps=horizon_days,
                initial_price=initial_price,
                seed=self.base_seed,
            )

            n_oos = min(horizon_days + 1, len(self.test_prices))
            oos_prices = self.test_prices[:n_oos]
            oos_returns = self.test_returns[:min(horizon_days, len(self.test_returns))]

            return self.scorer.score_quick(result, oos_prices, oos_returns)

        return objective

    def _make_detailed_objective(
        self, model: BaseModel, horizon_days: int
    ) -> callable:
        """Create objective that returns composite + per-metric scores."""
        def objective(m: BaseModel) -> dict:
            m.fit(self.train_returns)
            initial_price = self.train_prices[-1]

            result = m.simulate(
                n_simulations=self.n_sims_search,
                n_steps=horizon_days,
                initial_price=initial_price,
                seed=self.base_seed,
            )

            n_oos = min(horizon_days + 1, len(self.test_prices))
            oos_prices = self.test_prices[:n_oos]
            oos_returns = self.test_returns[:min(horizon_days, len(self.test_returns))]

            from validation.metrics import ScoringMetrics
            scores = ScoringMetrics.compute_all(
                result.paths, result.log_returns, oos_prices, oos_returns,
            )
            composite = sum(
                self.scorer.weights.get(k, 0) * max(0.0, min(1.0, v))
                for k, v in scores.items()
            )
            return {"composite": float(composite), "metrics": scores}

        return objective

    def run(self) -> list[ScoreCard]:
        """
        Execute the full experiment grid.

        Returns
        -------
        list[ScoreCard]
            All scored results, sorted by composite score descending.
        """
        self.results = []
        total_combos = len(self.model_names) * len(self.horizons)

        console.print(f"\n[bold]Experiment: {self.granular_data['experiment_name']}[/bold]")
        console.print(f"Models: {self.model_names}")
        console.print(f"Horizons: {self.horizons}")
        console.print(f"Search phase: {self.search_phase}")
        console.print(f"Simulations: {self.n_sims_search} (search) / {self.n_sims_final} (final)\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running experiments...", total=total_combos)

            for model_name in self.model_names:
                for horizon_str in self.horizons:
                    horizon_days = parse_horizon(horizon_str)
                    progress.update(task, description=f"{model_name} @ {horizon_str}")

                    run_data = self._run_single(model_name, horizon_str, horizon_days)
                    self.granular_data["runs"].append(run_data)

                    # Build ScoreCard from walk-forward weighted result
                    card = run_data["_scorecard"]
                    self.results.append(card)
                    progress.advance(task)

        # Sort by composite score descending
        self.results.sort(key=lambda c: c.composite_score, reverse=True)
        for i, card in enumerate(self.results):
            card.rank = i + 1

        # ── Cross-Horizon Aggregation ───────────────────────────────
        HZ_WEIGHTS = {"180d": 0.10, "365d": 0.20, "730d": 0.35, "1460d": 0.35}
        cross_hz = {}
        for model_name in self.model_names:
            per_hz = {}
            for run in self.granular_data["runs"]:
                if run["model"] == model_name:
                    per_hz[run["horizon"]] = run["walk_forward"]["weighted_composite"]

            if len(per_hz) >= 2:
                vals = [per_hz.get(k, 0) for k in HZ_WEIGHTS]
                composite = sum(per_hz.get(k, 0) * w for k, w in HZ_WEIGHTS.items())
                std = float(np.std(vals))
                penalty = std * 0.1
                cross_hz[model_name] = {
                    "composite": round(composite, 4),
                    "stability_penalty": round(penalty, 4),
                    "final_score": round(composite - penalty, 4),
                    "per_horizon": per_hz,
                    "horizon_std": round(std, 4),
                    "horizon_weights": HZ_WEIGHTS,
                }

        if cross_hz:
            ranked = sorted(cross_hz.items(), key=lambda x: x[1]["final_score"], reverse=True)
            console.print("\n[bold]Cross-Horizon Final Scores:[/bold]")
            for model_name, ch in ranked:
                console.print(
                    f"  {model_name:30s}  final={ch['final_score']:.4f}  "
                    f"(composite={ch['composite']:.4f} - penalty={ch['stability_penalty']:.4f})"
                )
            self.granular_data["cross_horizon"] = cross_hz

        return self.results

    def _run_single(
        self, model_name: str, horizon_str: str, horizon_days: int
    ) -> dict[str, Any]:
        """Run optimization + walk-forward for one model+horizon."""
        model = get_model(model_name)
        has_param_space = bool(model.get_param_space())

        # ── Phase 1: Parameter optimization ─────────────────────────
        # Models that use univariate sensitivity analysis
        SENSITIVITY_MODELS = {"regime_block_bootstrap"}

        if not has_param_space:
            # No hyperparameters to optimize (e.g. GBM).
            search_result = {
                "best_params": {},
                "best_score": None,
                "n_trials_completed": 0,
                "all_trials": [],
            }
            best_params = {}
        elif model_name in SENSITIVITY_MODELS:
            # Univariate sensitivity analysis with joint confirmation
            # Quick mode: reduce MSM em_iter for faster convergence
            if self.run_mode == "quick" and hasattr(model, "em_iter"):
                model.em_iter = 200
            objective = self._make_detailed_objective(model, horizon_days)
            search = SensitivitySearch(
                model=model,
                objective_fn=objective,
                seed=self.base_seed,
            )
            search_result = search.run()
            best_params = search_result["best_params"]
        else:
            # Standard Optuna search (e.g. GARCH)
            n_trials = self.trial_budgets.get(model_name, 10)
            objective = self._make_objective(model, horizon_days)
            search = ParameterSearch(
                model=model,
                objective_fn=objective,
                n_trials=n_trials,
                search_phase=self.search_phase,
                sobol_ratio=self.config.get("simulation", {}).get("sobol_ratio", 0.4),
                seed=self.base_seed,
            )
            search_result = search.run()
            best_params = search_result["best_params"]

        # ── Phase 2: Walk-forward backtest ──────────────────────────
        backtest = WalkForwardBacktest(
            prices=self.full_prices,
            returns=self.full_returns,
            min_training_days=self.wf_min_train,
            step_size_days=self.wf_step_size,
            recency_weighting=self.wf_recency,
            recency_half_life_days=self.wf_half_life,
            dates=self.dates,
        )

        wf_model = get_model(model_name)
        if best_params:
            wf_model.set_params(**best_params)

        wf_result = backtest.run(
            model=wf_model,
            horizon_days=horizon_days,
            n_simulations=self.n_sims_final,
            scorer=self.scorer,
            seed=self.base_seed,
        )

        # ── Phase 3: Single OOS scorecard (for leaderboard display) ─
        final_model = get_model(model_name)
        if best_params:
            final_model.set_params(**best_params)
        final_model.fit(self.train_returns)

        initial_price = self.train_prices[-1]
        sim_result = final_model.simulate(
            n_simulations=self.n_sims_final,
            n_steps=horizon_days,
            initial_price=initial_price,
            seed=self.base_seed,
        )

        n_oos = min(horizon_days + 1, len(self.test_prices))
        oos_prices = self.test_prices[:n_oos]
        oos_returns = self.test_returns[:min(horizon_days, len(self.test_returns))]

        final_card = self.scorer.score(
            sim_result=sim_result,
            hist_prices=oos_prices,
            hist_returns=oos_returns,
            horizon=horizon_str,
            train_prices=self.train_prices,
        )

        # Use walk-forward weighted composite as the leaderboard score
        if wf_result["n_windows"] > 0:
            final_card.composite_score = wf_result["weighted_composite"]

        params_display = best_params if best_params else final_card.params
        console.print(
            f"  {model_name:30s} {horizon_str:>6s}  "
            f"composite={final_card.composite_score:.4f}  "
            f"wf_windows={wf_result['n_windows']}  "
            f"params={params_display}"
        )

        # ── Build granular data record ──────────────────────────────
        window_records = []
        for i, wcard in enumerate(wf_result.get("window_scores", [])):
            w_start_idx = wf_result.get("window_positions", [0])[i] if "window_positions" in wf_result else None
            window_records.append({
                "window_num": i,
                "train_start_idx": 0,
                "train_end_idx": wcard._train_end_idx if hasattr(wcard, '_train_end_idx') else None,
                "test_start_idx": wcard._test_start_idx if hasattr(wcard, '_test_start_idx') else None,
                "test_end_idx": wcard._test_end_idx if hasattr(wcard, '_test_end_idx') else None,
                "train_start_date": wcard._train_start_date if hasattr(wcard, '_train_start_date') else None,
                "train_end_date": wcard._train_end_date if hasattr(wcard, '_train_end_date') else None,
                "test_start_date": wcard._test_start_date if hasattr(wcard, '_test_start_date') else None,
                "test_end_date": wcard._test_end_date if hasattr(wcard, '_test_end_date') else None,
                "weight": float(wf_result["weights"][i]) if i < len(wf_result["weights"]) else None,
                "fitted_params": wcard._fitted_params if hasattr(wcard, '_fitted_params') else wcard.params,
                "model_metadata": wcard._model_metadata if hasattr(wcard, '_model_metadata') else {},
                "composite_score": wcard.composite_score,
                "scoring_metrics": wcard.scoring_metrics,
                "raw_metrics": wcard.raw_metrics,
                "diagnostic_metrics": wcard.diagnostic_metrics,
                "sim_percentiles": wcard._sim_percentiles if hasattr(wcard, '_sim_percentiles') else None,
                "realized_prices": wcard._realized_prices if hasattr(wcard, '_realized_prices') else None,
            })

        run_record = {
            "model": model_name,
            "horizon": horizon_str,
            "horizon_days": horizon_days,
            "optimization": {
                "best_params": search_result["best_params"],
                "best_score": search_result["best_score"],
                "n_trials": search_result["n_trials_completed"],
                "search_phase": (
                    "mle" if not has_param_space
                    else "sensitivity" if model_name in SENSITIVITY_MODELS
                    else self.search_phase
                ),
                "trials": search_result["all_trials"],
            },
            "walk_forward": {
                "weighted_composite": wf_result["weighted_composite"],
                "n_windows": wf_result["n_windows"],
                "recency_weighting": self.wf_recency,
                "recency_half_life_days": self.wf_half_life,
                "step_size_days": self.wf_step_size,
                "min_training_days": self.wf_min_train,
                "windows": window_records,
            },
            "final_scorecard": {
                "composite_score": final_card.composite_score,
                "scoring_metrics": final_card.scoring_metrics,
                "raw_metrics": final_card.raw_metrics,
                "diagnostic_metrics": final_card.diagnostic_metrics,
                "distributions": final_card.distributions,
                "params": final_card.params,
            },
        }

        # Attach scorecard for leaderboard (not serialized)
        run_record["_scorecard"] = final_card

        return run_record

    def save_granular_results(self, output_dir: str = "results") -> Path:
        """
        Save all granular experiment data to JSON for dashboard.

        Parameters
        ----------
        output_dir : str
            Directory to write to.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Strip non-serializable objects
        data = _make_serializable(self.granular_data)

        # Remove internal _scorecard references
        for run in data.get("runs", []):
            run.pop("_scorecard", None)

        filepath = out_path / "experiment_data.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        console.print(f"\nGranular data saved to {filepath}")
        return filepath

    @classmethod
    def from_yaml(
        cls,
        config_path: str,
        train_returns: np.ndarray,
        train_prices: np.ndarray,
        test_prices: np.ndarray,
        test_returns: np.ndarray,
        dates: np.ndarray | None = None,
        train_end_idx: int | None = None,
    ) -> "ExperimentRunner":
        """Create ExperimentRunner from a YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config, train_returns, train_prices, test_prices, test_returns,
                   dates=dates, train_end_idx=train_end_idx)
