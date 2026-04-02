#!/usr/bin/env python
"""
Convert Phase 2 Config 1 results into experiment_data.json format
and re-export leaderboard (CSV/JSON/HTML).

The Phase 2 baseline script saves to a different schema than what the
dashboard expects. This script bridges the gap:
  1. Reads phase2_config1_standard.json
  2. Loads the date array to compute array indices from dates
  3. Reshapes into the experiment_data.json schema
  4. Writes to results/phase2/config1/experiment_data.json
  5. Re-exports leaderboard files

Usage:
    python scripts/export_phase2_results.py
"""
import json
import click
import numpy as np
from datetime import datetime
from pathlib import Path
from rich.console import Console

from data.loader import BTCDataLoader, get_price_array
from validation.scorer import ScoreCard
from reporting.leaderboard import Leaderboard
from reporting.export import ResultsExporter


console = Console()


def load_phase2_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_date_index(dates: np.ndarray) -> dict[str, int]:
    """Map date string -> array index for fast lookup."""
    return {d: i for i, d in enumerate(dates)}


def convert_window(
    w: dict,
    date_idx: dict,
    horizon_days: int,
) -> dict:
    """Convert a Phase 2 window record into experiment_data.json window format."""
    # Compute array indices from dates
    train_end_date = w.get("train_end_date")
    test_start_date = w.get("test_start_date")
    test_end_date = w.get("test_end_date")

    train_end_idx = date_idx.get(train_end_date, 730) if train_end_date else 730
    test_start_idx = date_idx.get(test_start_date, train_end_idx) if test_start_date else train_end_idx
    test_end_idx = date_idx.get(test_end_date, test_start_idx + horizon_days) if test_end_date else test_start_idx + horizon_days

    return {
        "window_num": w["window_num"],
        "train_start_idx": 0,
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
        "test_end_idx": test_end_idx,
        "train_start_date": None,  # Walk-forward always starts from index 0
        "train_end_date": train_end_date,
        "test_start_date": test_start_date,
        "test_end_date": test_end_date,
        "weight": w["weight"],
        "fitted_params": w.get("fitted_params", {}),
        "model_metadata": w.get("model_metadata", {}),
        "composite_score": w["composite_score"],
        "scoring_metrics": w.get("scoring_metrics", {}),
        "raw_metrics": w.get("raw_metrics", {}),
        "diagnostic_metrics": w.get("diagnostic_metrics", {}),
        "sim_percentiles": w.get("sim_percentiles"),
        "realized_prices": w.get("realized_prices"),
    }


def convert_model_runs(
    model_data: dict,
    model_name: str,
    date_idx: dict,
    locked_params: dict | None,
) -> list[dict]:
    """Convert Phase 2 per-model data into experiment_data.json run records."""
    runs = []
    per_horizon = model_data["per_horizon"]

    for hz_str, hz_data in per_horizon.items():
        hz_days = int(hz_str.replace("d", ""))

        windows = [
            convert_window(w, date_idx, hz_days)
            for w in hz_data.get("window_details", [])
        ]

        # Build the final_scorecard from the walk-forward aggregate
        # Use scoring_metrics from the most recent (highest-weight) window as representative
        last_window = hz_data["window_details"][-1] if hz_data["window_details"] else {}
        final_scoring_metrics = last_window.get("scoring_metrics", {})

        best_params = locked_params or {}

        run_record = {
            "model": model_name,
            "horizon": hz_str,
            "horizon_days": hz_days,
            "optimization": {
                "best_params": best_params,
                "best_score": hz_data["weighted_composite"],
                "n_trials": 0,  # locked params, no search
                "search_phase": "locked" if locked_params else "mle",
                "trials": [],
            },
            "walk_forward": {
                "weighted_composite": hz_data["weighted_composite"],
                "n_windows": hz_data["n_windows"],
                "recency_weighting": "exponential",
                "recency_half_life_days": 730,
                "step_size_days": 90,
                "min_training_days": 730,
                "windows": windows,
            },
            "final_scorecard": {
                "composite_score": hz_data["weighted_composite"],
                "scoring_metrics": hz_data.get("final_scoring_metrics", final_scoring_metrics),
                "raw_metrics": {},
                "diagnostic_metrics": {},
                "distributions": hz_data.get("final_distributions", {}),
                "params": best_params,
            },
        }
        runs.append(run_record)

    return runs


def build_experiment_data(phase2: dict, dates: np.ndarray, data_info: dict) -> dict:
    """Build full experiment_data.json from Phase 2 results."""
    date_idx = build_date_index(dates)
    locked_params = phase2["locked_rbb_params"]

    # Convert RBB runs
    rbb_runs = convert_model_runs(
        phase2["rbb"], "regime_block_bootstrap", date_idx, locked_params,
    )

    # Convert GBM runs
    gbm_runs = convert_model_runs(
        phase2["gbm"], "gbm", date_idx, None,
    )

    all_runs = gbm_runs + rbb_runs

    # Cross-horizon aggregation
    HZ_WEIGHTS = {"180d": 0.10, "365d": 0.20, "730d": 0.35, "1460d": 0.35}
    cross_hz = {}
    for model_name in ["regime_block_bootstrap", "gbm"]:
        per_hz = {}
        for run in all_runs:
            if run["model"] == model_name:
                per_hz[run["horizon"]] = run["walk_forward"]["weighted_composite"]
        if per_hz:
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

    return {
        "experiment_name": "phase2_config1_baseline",
        "run_mode": phase2["mode"],
        "timestamp": phase2["timestamp"],
        "config": {
            "experiment": {
                "name": "phase2_config1_baseline",
                "seed": 42,
                "run_mode": phase2["mode"],
            },
            "simulation": {
                "n_simulations_search": 0,
                "n_simulations_final": 5000,
                "search_phase": "locked",
            },
            "data": {
                "source": "yfinance",
                "ticker": "BTC-USD",
                "start_date": "2013-01-01",
                "train_pct": 0.7,
            },
            "models": ["gbm", "regime_block_bootstrap"],
            "horizons": phase2["horizons"],
            "scoring": {
                "weights": {
                    "drawdown_duration_dist": 0.13,
                    "recovery_time_dist": 0.12,
                    "time_in_drawdown_ratio": 0.10,
                    "tail_index_accuracy": 0.13,
                    "percentile_band_coverage": 0.12,
                    "ks_statistic": 0.07,
                    "qq_divergence": 0.08,
                    "vol_clustering_acf": 0.10,
                    "mape": 0.15,
                },
            },
            "walk_forward": {
                "step_size_days": 90,
                "min_training_days": 730,
                "recency_weighting": "exponential",
                "recency_half_life_days": 730,
            },
            "locked_rbb_params": locked_params,
        },
        "data_info": data_info,
        "runs": all_runs,
        "cross_horizon": cross_hz,
    }


def build_scorecards(experiment_data: dict) -> list[ScoreCard]:
    """Build ScoreCard objects from experiment_data for leaderboard export."""
    cards = []
    for run in experiment_data["runs"]:
        wf = run["walk_forward"]
        fs = run["final_scorecard"]
        card = ScoreCard(
            model_name=run["model"],
            horizon=run["horizon"],
            params=run["optimization"]["best_params"],
            scoring_metrics=fs.get("scoring_metrics", {}),
            diagnostic_metrics=fs.get("diagnostic_metrics", {}),
            composite_score=wf["weighted_composite"],
        )
        cards.append(card)

    cards.sort(key=lambda c: c.composite_score, reverse=True)
    for i, card in enumerate(cards):
        card.rank = i + 1
    return cards


@click.command()
@click.option("--source", default=None, help="Path to phase2 JSON results file.")
@click.option("--config", "config_num", default=1, type=int,
              help="Config number (used to find default source path).")
def main(source: str | None, config_num: int):
    if source:
        source = Path(source)
        output_dir = source.parent
    else:
        source = Path(f"results/phase2/config{config_num}/phase2_config{config_num}_standard.json")
        output_dir = Path(f"results/phase2/config{config_num}")

    if not source.exists():
        console.print(f"[red]Source not found: {source}[/red]")
        return

    console.print(f"[bold]Converting Phase 2 Config {config_num} results for dashboard/export[/bold]")
    console.print(f"Source: {source}")

    # Load Phase 2 data
    phase2 = load_phase2_data(str(source))

    # Load date array for index computation
    loader = BTCDataLoader(ticker="BTC-USD", start_date="2013-01-01")
    train_df, test_df = loader.get_train_test_split(train_pct=0.7)
    full_df = loader.load_processed_data()
    dates = np.array([str(d.date()) for d in full_df.index])

    data_info = {
        "train_days": len(train_df),
        "test_days": len(test_df),
        "train_start": str(train_df.index[0].date()),
        "train_end": str(train_df.index[-1].date()),
        "test_start": str(test_df.index[0].date()),
        "test_end": str(test_df.index[-1].date()),
    }

    # Convert
    experiment_data = build_experiment_data(phase2, dates, data_info)

    # Verify window counts
    total_windows = sum(r["walk_forward"]["n_windows"] for r in experiment_data["runs"])
    console.print(f"\nConverted: {len(experiment_data['runs'])} runs, {total_windows} total windows")

    for run in experiment_data["runs"]:
        wf = run["walk_forward"]
        n_idx_ok = sum(
            1 for w in wf["windows"]
            if w["train_end_idx"] > 0 and w["test_start_idx"] > 0
        )
        console.print(
            f"  {run['model']:30s} {run['horizon']:>6s}  "
            f"windows={wf['n_windows']}  wf={wf['weighted_composite']:.4f}  "
            f"indices_ok={n_idx_ok}/{wf['n_windows']}"
        )

    # Write experiment_data.json
    exp_path = output_dir / "experiment_data.json"
    with open(exp_path, "w") as f:
        json.dump(experiment_data, f, indent=2, default=str)
    console.print(f"\nWrote {exp_path}")

    # Also overwrite the stale root-level experiment_data.json
    root_exp_path = Path("results/experiment_data.json")
    with open(root_exp_path, "w") as f:
        json.dump(experiment_data, f, indent=2, default=str)
    console.print(f"Wrote {root_exp_path} (replaced stale data)")

    # Build ScoreCards and re-export leaderboard
    cards = build_scorecards(experiment_data)

    leaderboard = Leaderboard(cards)
    leaderboard.print_summary()

    # Export to all formats in both locations
    for out_dir in [output_dir, Path("results")]:
        exporter = ResultsExporter(cards, output_dir=str(out_dir))
        paths = exporter.export_all()
        console.print(f"\nExported to {out_dir}:")
        for fmt, path in paths.items():
            console.print(f"  {fmt}: {path}")

    # Summary
    console.print(f"\n[bold green]Export complete.[/bold green]")
    rbb_ch = experiment_data["cross_horizon"].get("regime_block_bootstrap", {})
    gbm_ch = experiment_data["cross_horizon"].get("gbm", {})
    console.print(f"RBB final: {rbb_ch.get('final_score', 0):.4f}  |  GBM final: {gbm_ch.get('final_score', 0):.4f}")


if __name__ == "__main__":
    main()
