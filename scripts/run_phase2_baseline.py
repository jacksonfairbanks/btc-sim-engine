#!/usr/bin/env python
"""
Phase 2 Config 1 — Establish the true walk-forward baseline.

Runs RBB with locked params (geometric bl=30, regime_enabled=False) and GBM
through the full walk-forward pipeline across all four horizons. No parameter
search — params are fixed. This produces the definitive baseline score that
all Phase 2 regime comparisons will be measured against.

Usage:
    python scripts/run_phase2_baseline.py --mode quick    # validate pipeline
    python scripts/run_phase2_baseline.py --mode standard  # real scores
"""
import json
import time
import click
import numpy as np
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table

from data.loader import BTCDataLoader, get_price_array
from models.registry import get_model
from validation.scorer import Scorer
from validation.backtest import WalkForwardBacktest


console = Console()

# ── Phase 2 Config Definitions ────────────────────────────────────
PHASE2_CONFIGS = {
    1: {
        "name": "Config 1: No regime (baseline)",
        "short": "no_regime_baseline",
        "params": {
            "block_length_sampling": "geometric",
            "mean_block_length": 30,
            "min_block_length": 5,
            "block_stride": 5,
            "min_pool_size": 8,
            "regime_enabled": False,
            "transition_matrix_method": "fitted",
            "msm_variance_switching": True,
            "msm_frequency": "weekly",
        },
    },
    2: {
        "name": "Config 2: Regime ON, fitted transition, variance switching ON",
        "short": "regime_fitted_vsON",
        "params": {
            "block_length_sampling": "geometric",
            "mean_block_length": 30,
            "min_block_length": 5,
            "block_stride": 5,
            "min_pool_size": 8,
            "regime_enabled": True,
            "transition_matrix_method": "fitted",
            "msm_variance_switching": True,
            "msm_frequency": "weekly",
        },
    },
    3: {
        "name": "Config 3: Regime ON, fitted transition, variance switching OFF",
        "short": "regime_fitted_vsOFF",
        "params": {
            "block_length_sampling": "geometric",
            "mean_block_length": 30,
            "min_block_length": 5,
            "block_stride": 5,
            "min_pool_size": 8,
            "regime_enabled": True,
            "transition_matrix_method": "fitted",
            "msm_variance_switching": False,
            "msm_frequency": "weekly",
        },
    },
    4: {
        "name": "Config 4: Regime ON, empirical transition, variance switching ON",
        "short": "regime_empirical_vsON",
        "params": {
            "block_length_sampling": "geometric",
            "mean_block_length": 30,
            "min_block_length": 5,
            "block_stride": 5,
            "min_pool_size": 8,
            "regime_enabled": True,
            "transition_matrix_method": "empirical",
            "msm_variance_switching": True,
            "msm_frequency": "weekly",
        },
    },
    5: {
        "name": "Config 5: Regime ON, empirical transition, variance switching OFF",
        "short": "regime_empirical_vsOFF",
        "params": {
            "block_length_sampling": "geometric",
            "mean_block_length": 30,
            "min_block_length": 5,
            "block_stride": 5,
            "min_pool_size": 8,
            "regime_enabled": True,
            "transition_matrix_method": "empirical",
            "msm_variance_switching": False,
            "msm_frequency": "weekly",
        },
    },
    6: {
        "name": "Config 6: 3-state Baum-Welch HMM, fitted transition, variance switching ON",
        "short": "hmm3_fitted_vsON",
        "params": {
            "block_length_sampling": "geometric",
            "mean_block_length": 30,
            "min_block_length": 5,
            "block_stride": 5,
            "min_pool_size": 8,
            "regime_enabled": True,
            "regime_method": "hmm_baum_welch",
            "n_regimes": 3,
            "transition_matrix_method": "fitted",
            "msm_variance_switching": True,
            "msm_frequency": "weekly",
        },
    },
    # ── Phase 3: GARCH configs ─────────────────────────────────────
    10: {
        "name": "Config 10: GARCH(1,1) t-dist baseline",
        "short": "garch_t_baseline",
        "model": "garch_1_1",
        "params": {
            "p": 1,
            "q": 1,
            "dist": "t",
            "mean_model": "Constant",
        },
    },
    11: {
        "name": "Config 11: GARCH(2,2) t-dist Constant",
        "short": "garch22_t_const",
        "model": "garch_1_1",
        "params": {
            "p": 2,
            "q": 2,
            "dist": "t",
            "mean_model": "Constant",
        },
    },
    12: {
        "name": "Config 12: GARCH(1,1) t-dist Constant",
        "short": "garch11_t_const",
        "model": "garch_1_1",
        "params": {
            "p": 1,
            "q": 1,
            "dist": "t",
            "mean_model": "Constant",
        },
    },
}

HORIZONS = ["180d", "365d", "730d", "1460d"]
HZ_WEIGHTS = {"180d": 0.10, "365d": 0.20, "730d": 0.35, "1460d": 0.35}

MODE_SETTINGS = {
    "quick": {
        "n_simulations": 2000,
        "step_size_days": 365,
    },
    "standard": {
        "n_simulations": 5000,
        "step_size_days": 90,
    },
}


def run_walk_forward(
    model_name: str,
    locked_params: dict | None,
    horizons: list[str],
    full_prices: np.ndarray,
    full_returns: np.ndarray,
    dates: np.ndarray,
    n_simulations: int,
    step_size_days: int,
    train_returns: np.ndarray | None = None,
    train_prices: np.ndarray | None = None,
    test_prices: np.ndarray | None = None,
    test_returns: np.ndarray | None = None,
    seed: int = 42,
) -> dict:
    """
    Run walk-forward backtest for a model with locked params across all horizons.

    Returns dict with per-horizon results and cross-horizon aggregation.
    """
    scorer = Scorer()
    per_horizon = {}

    for hz_str in horizons:
        hz_days = int(hz_str.replace("d", ""))
        console.print(f"\n  [bold]{model_name} @ {hz_str}[/bold] ({hz_days}d, {n_simulations} sims)")

        model = get_model(model_name)
        if locked_params:
            model.set_params(**locked_params)

        backtest = WalkForwardBacktest(
            prices=full_prices,
            returns=full_returns,
            min_training_days=730,
            step_size_days=step_size_days,
            recency_weighting="exponential",
            recency_half_life_days=730,
            dates=dates,
        )

        t0 = time.time()
        wf_result = backtest.run(
            model=model,
            horizon_days=hz_days,
            n_simulations=n_simulations,
            scorer=scorer,
            seed=seed,
        )
        elapsed = time.time() - t0

        n_windows = wf_result["n_windows"]
        wf_composite = wf_result["weighted_composite"]

        # Collect per-window audit data
        window_details = []
        for i, card in enumerate(wf_result.get("window_scores", [])):
            window_details.append({
                "window_num": i,
                "train_end_date": card._train_end_date if hasattr(card, '_train_end_date') else None,
                "test_start_date": card._test_start_date if hasattr(card, '_test_start_date') else None,
                "test_end_date": card._test_end_date if hasattr(card, '_test_end_date') else None,
                "train_days": card._train_days if hasattr(card, '_train_days') else None,
                "weight": float(wf_result["weights"][i]),
                "composite_score": card.composite_score,
                "scoring_metrics": card.scoring_metrics,
                "diagnostic_metrics": card.diagnostic_metrics,
                "raw_metrics": card.raw_metrics,
                "fitted_params": card._fitted_params if hasattr(card, '_fitted_params') else card.params,
                "model_metadata": card._model_metadata if hasattr(card, '_model_metadata') else {},
                "sim_percentiles": card._sim_percentiles if hasattr(card, '_sim_percentiles') else None,
                "realized_prices": card._realized_prices if hasattr(card, '_realized_prices') else None,
                "locked_config": locked_params if locked_params else "MLE (no hyperparams)",
            })

        # Per-window score summary
        window_composites = [w["composite_score"] for w in window_details]
        console.print(f"    Windows: {n_windows}  |  WF composite: {wf_composite:.4f}  |  {elapsed:.1f}s")
        if window_composites:
            console.print(
                f"    Window range: [{min(window_composites):.4f}, {max(window_composites):.4f}]  "
                f"std={np.std(window_composites):.4f}"
            )

        # Final OOS scorecard with distributions (for dashboard percentile charts)
        final_distributions = {}
        final_scoring_metrics = {}
        if train_returns is not None and test_prices is not None:
            final_model = get_model(model_name)
            if locked_params:
                final_model.set_params(**locked_params)
            final_model.fit(train_returns)
            initial_price = float(train_prices[-1])
            n_oos = min(hz_days + 1, len(test_prices))
            oos_prices = test_prices[:n_oos]
            oos_returns = test_returns[:min(hz_days, len(test_returns))]
            sim_result = final_model.simulate(
                n_simulations=n_simulations,
                n_steps=len(oos_returns),
                initial_price=initial_price,
                seed=seed,
            )
            final_card = scorer.score(
                sim_result=sim_result,
                hist_prices=oos_prices,
                hist_returns=oos_returns,
                horizon=hz_str,
                train_prices=train_prices,
                include_distributions=True,
            )
            final_distributions = final_card.distributions
            final_scoring_metrics = final_card.scoring_metrics

        per_horizon[hz_str] = {
            "weighted_composite": wf_composite,
            "n_windows": n_windows,
            "elapsed_seconds": round(elapsed, 1),
            "window_composites": window_composites,
            "window_details": window_details,
            "final_distributions": final_distributions,
            "final_scoring_metrics": final_scoring_metrics,
        }

    # Cross-horizon aggregation
    vals = [per_horizon[hz]["weighted_composite"] for hz in HORIZONS]
    weighted = sum(per_horizon[hz]["weighted_composite"] * HZ_WEIGHTS[hz] for hz in HORIZONS)
    hz_std = float(np.std(vals))
    penalty = hz_std * 0.1
    final_score = weighted - penalty

    return {
        "model": model_name,
        "locked_params": locked_params,
        "per_horizon": per_horizon,
        "cross_horizon": {
            "weighted_composite": round(weighted, 4),
            "horizon_std": round(hz_std, 4),
            "stability_penalty": round(penalty, 4),
            "final_score": round(final_score, 4),
        },
    }


def print_report(rbb_result: dict, gbm_result: dict, mode: str,
                  config_num: int = 1, config_def: dict | None = None) -> None:
    """Print the full Phase 2 baseline report."""
    _cfg_name = config_def["name"] if config_def else f"Config {config_num}"
    console.print("\n" + "=" * 72)
    console.print(f"[bold]PHASE 2 {_cfg_name} — WALK-FORWARD REPORT[/bold]")
    console.print(f"Mode: {mode.upper()}")
    console.print("=" * 72)

    # ── Cross-horizon summary ──────────────────────────────────────
    rbb_ch = rbb_result["cross_horizon"]
    gbm_ch = gbm_result["cross_horizon"]

    table = Table(title="Cross-Horizon Summary")
    table.add_column("", style="bold")
    table.add_column("RBB (geo bl=30)", justify="right")
    table.add_column("GBM (MLE)", justify="right")
    table.add_column("Delta", justify="right")

    table.add_row(
        "Weighted Composite",
        f"{rbb_ch['weighted_composite']:.4f}",
        f"{gbm_ch['weighted_composite']:.4f}",
        f"{rbb_ch['weighted_composite'] - gbm_ch['weighted_composite']:+.4f}",
    )
    table.add_row(
        "Horizon Std",
        f"{rbb_ch['horizon_std']:.4f}",
        f"{gbm_ch['horizon_std']:.4f}",
        "",
    )
    table.add_row(
        "Stability Penalty",
        f"{rbb_ch['stability_penalty']:.4f}",
        f"{gbm_ch['stability_penalty']:.4f}",
        "",
    )
    table.add_row(
        "Final Score",
        f"[bold green]{rbb_ch['final_score']:.4f}[/bold green]",
        f"{gbm_ch['final_score']:.4f}",
        f"[bold]{rbb_ch['final_score'] - gbm_ch['final_score']:+.4f}[/bold]",
    )
    console.print(table)

    # ── Per-horizon breakdown ──────────────────────────────────────
    hz_table = Table(title="Per-Horizon Walk-Forward Composites")
    hz_table.add_column("Horizon", style="bold")
    hz_table.add_column("Weight", justify="right")
    hz_table.add_column("RBB", justify="right")
    hz_table.add_column("GBM", justify="right")
    hz_table.add_column("Delta", justify="right")
    hz_table.add_column("RBB Windows", justify="right")
    hz_table.add_column("GBM Windows", justify="right")

    for hz in HORIZONS:
        rbb_wf = rbb_result["per_horizon"][hz]["weighted_composite"]
        gbm_wf = gbm_result["per_horizon"][hz]["weighted_composite"]
        delta = rbb_wf - gbm_wf
        hz_table.add_row(
            hz,
            f"{HZ_WEIGHTS[hz]:.0%}",
            f"{rbb_wf:.4f}",
            f"{gbm_wf:.4f}",
            f"{delta:+.4f}",
            str(rbb_result["per_horizon"][hz]["n_windows"]),
            str(gbm_result["per_horizon"][hz]["n_windows"]),
        )
    console.print(hz_table)

    # ── Locked config verification ─────────────────────────────────
    locked_params = config_def["params"] if config_def else {}
    console.print("\n[bold]Locked Config (verified per-window):[/bold]")
    for k, v in locked_params.items():
        # Skip irrelevant regime params when regime is OFF
        if not locked_params.get("regime_enabled", True):
            if k in ("transition_matrix_method", "msm_variance_switching", "msm_frequency"):
                continue
        console.print(f"  {k}: {v}")

    # ── Window detail for RBB ──────────────────────────────────────
    for hz in HORIZONS:
        windows = rbb_result["per_horizon"][hz]["window_details"]
        if not windows:
            continue
        console.print(f"\n[bold]RBB Walk-Forward Windows — {hz}[/bold]")
        w_table = Table()
        w_table.add_column("#", justify="right")
        w_table.add_column("Train End")
        w_table.add_column("Test Start")
        w_table.add_column("Test End")
        w_table.add_column("Weight", justify="right")
        w_table.add_column("Composite", justify="right")
        w_table.add_column("Config OK", justify="center")

        for w in windows:
            # Verify locked params were used
            fp = w.get("fitted_params", {})
            config_ok = all(
                fp.get(k) == v for k, v in locked_params.items()
                if k in fp  # only check keys present in fitted_params
            ) if locked_params else True
            w_table.add_row(
                str(w["window_num"]),
                str(w.get("train_end_date", "?"))[:10],
                str(w.get("test_start_date", "?"))[:10],
                str(w.get("test_end_date", "?"))[:10],
                f"{w['weight']:.4f}",
                f"{w['composite_score']:.4f}",
                "[green]YES[/green]" if config_ok else "[red]NO[/red]",
            )
        console.print(w_table)


def save_results(rbb_result: dict, gbm_result: dict, mode: str, output_dir: Path,
                  config_num: int = 1, config_def: dict | None = None) -> Path:
    """Save full results to JSON for traceability."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
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

    locked_params = config_def["params"] if config_def else {}

    data = {
        "phase": f"phase2_config{config_num}",
        "config_name": config_def["name"] if config_def else "unknown",
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "locked_rbb_params": locked_params,
        "horizons": HORIZONS,
        "horizon_weights": HZ_WEIGHTS,
        "rbb": make_serializable(rbb_result),
        "gbm": make_serializable(gbm_result),
    }

    filepath = output_dir / f"phase2_config{config_num}_{mode}.json"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    console.print(f"\nResults saved to {filepath}")
    return filepath


@click.command()
@click.option("--config", "config_num", default=1, type=int,
              help="Phase 2 config number (1-5).")
@click.option("--mode", default="quick", type=click.Choice(["quick", "standard"]),
              help="Run mode: quick (validate) or standard (real scores).")
@click.option("--output-dir", default=None,
              help="Output directory (default: results/phase2/configN).")
@click.option("--seed", default=42, type=int, help="Random seed.")
def main(config_num: int, mode: str, output_dir: str | None, seed: int):
    """Phase 2: Walk-forward with locked RBB params."""
    if config_num not in PHASE2_CONFIGS:
        console.print(f"[red]Unknown config {config_num}. Available: {list(PHASE2_CONFIGS.keys())}[/red]")
        return

    config_def = PHASE2_CONFIGS[config_num]
    locked_params = config_def["params"]
    settings = MODE_SETTINGS[mode]
    n_sims = settings["n_simulations"]
    step_size = settings["step_size_days"]

    if output_dir is None:
        output_dir = f"results/phase2/config{config_num}"

    primary_model_name = config_def.get("model", "regime_block_bootstrap")
    console.print(f"\n[bold]{config_def['name']}[/bold]")
    console.print(f"Model: {primary_model_name}  |  Mode: {mode}  |  Sims: {n_sims}  |  Step: {step_size}d  |  Seed: {seed}")
    console.print(f"Params: {locked_params}")
    console.print(f"Horizons: {HORIZONS}")

    # ── Load data ──────────────────────────────────────────────────
    loader = BTCDataLoader(ticker="BTC-USD", start_date="2013-01-01")
    train_df, test_df = loader.get_train_test_split(train_pct=0.7)

    train_returns = loader.get_returns_array(train_df)
    train_prices = get_price_array(train_df)
    test_prices = get_price_array(test_df)
    test_returns = loader.get_returns_array(test_df)

    full_prices = np.concatenate([train_prices, test_prices[1:]])
    full_returns = np.concatenate([train_returns, test_returns])

    full_df = loader.load_processed_data()
    dates = np.array([str(d.date()) for d in full_df.index])

    console.print(f"Data: {dates[0]} to {dates[-1]} ({len(full_prices)} days)")
    console.print(f"Train: {len(train_prices)} days  |  Test: {len(test_prices)} days")

    # ── Run primary model (locked params) ─────────────────────────
    primary_model = config_def.get("model", "regime_block_bootstrap")
    console.print("\n" + "-" * 50)
    console.print(f"[bold]Running {primary_model} ({config_def['short']})[/bold]")
    t0 = time.time()
    rbb_result = run_walk_forward(
        model_name=primary_model,
        locked_params=locked_params,
        horizons=HORIZONS,
        full_prices=full_prices,
        full_returns=full_returns,
        dates=dates,
        n_simulations=n_sims,
        step_size_days=step_size,
        train_returns=train_returns,
        train_prices=train_prices,
        test_prices=test_prices,
        test_returns=test_returns,
        seed=seed,
    )
    rbb_elapsed = time.time() - t0
    console.print(f"\n{primary_model} total: {rbb_elapsed:.1f}s")

    # ── Run GBM (MLE baseline) ─────────────────────────────────────
    console.print("\n" + "-" * 50)
    console.print("[bold]Running GBM (MLE baseline)[/bold]")
    t0 = time.time()
    gbm_result = run_walk_forward(
        model_name="gbm",
        locked_params=None,
        horizons=HORIZONS,
        full_prices=full_prices,
        full_returns=full_returns,
        dates=dates,
        n_simulations=n_sims,
        step_size_days=step_size,
        train_returns=train_returns,
        train_prices=train_prices,
        test_prices=test_prices,
        test_returns=test_returns,
        seed=seed,
    )
    gbm_elapsed = time.time() - t0
    console.print(f"\nGBM total: {gbm_elapsed:.1f}s")

    # ── Report ─────────────────────────────────────────────────────
    print_report(rbb_result, gbm_result, mode,
                 config_num=config_num, config_def=config_def)

    # ── Save ───────────────────────────────────────────────────────
    out_path = Path(output_dir)
    save_results(rbb_result, gbm_result, mode, out_path,
                 config_num=config_num, config_def=config_def)

    console.print(f"\n[bold green]Phase 2 Config {config_num} ({mode}) complete.[/bold green]")
    console.print(f"Total time: {rbb_elapsed + gbm_elapsed:.1f}s")


if __name__ == "__main__":
    main()
