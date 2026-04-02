#!/usr/bin/env python
"""
GARCH walk-forward comparison: Config A (p=2,q=2) vs Config B (p=1,q=1).

Runs both GARCH configs + one shared GBM baseline through the full
walk-forward pipeline. Standard mode: 5000 sims, 90d step.

Usage:
    python scripts/run_garch_walkforward.py
"""
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table

from data.loader import BTCDataLoader, get_price_array
from validation.scorer import Scorer
from validation.backtest import WalkForwardBacktest
from models.registry import get_model


console = Console()

HORIZONS = ["180d", "365d", "730d", "1460d"]
HZ_WEIGHTS = {"180d": 0.10, "365d": 0.20, "730d": 0.35, "1460d": 0.35}
N_SIMS = 5000
STEP_SIZE = 90
SEED = 42

CONFIGS = {
    "A": {
        "label": "GARCH(2,2) t Constant",
        "params": {"p": 2, "q": 2, "dist": "t", "mean_model": "Constant"},
    },
    "B": {
        "label": "GARCH(1,1) t Constant",
        "params": {"p": 1, "q": 1, "dist": "t", "mean_model": "Constant"},
    },
}


def run_model_walkforward(
    model_name: str,
    params: dict | None,
    label: str,
    full_prices: np.ndarray,
    full_returns: np.ndarray,
    dates: np.ndarray,
    train_returns: np.ndarray,
    train_prices: np.ndarray,
    test_prices: np.ndarray,
    test_returns: np.ndarray,
) -> dict:
    """Run walk-forward for one model config across all horizons."""
    scorer = Scorer()
    per_horizon = {}

    for hz_str in HORIZONS:
        hz_days = int(hz_str.replace("d", ""))
        console.print(f"\n  {label} @ {hz_str} ({hz_days}d, {N_SIMS} sims)")

        model = get_model(model_name)
        if params:
            model.set_params(**params)

        backtest = WalkForwardBacktest(
            prices=full_prices,
            returns=full_returns,
            min_training_days=730,
            step_size_days=STEP_SIZE,
            recency_weighting="exponential",
            recency_half_life_days=730,
            dates=dates,
        )

        t0 = time.time()
        wf_result = backtest.run(
            model=model,
            horizon_days=hz_days,
            n_simulations=N_SIMS,
            scorer=scorer,
            seed=SEED,
        )
        elapsed = time.time() - t0

        # Collect per-window data
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
                "fitted_params": card._fitted_params if hasattr(card, '_fitted_params') else card.params,
            })

        wf_comp = wf_result["weighted_composite"]
        w_scores = [w["composite_score"] for w in window_details]
        console.print(
            f"    Windows: {wf_result['n_windows']}  |  WF composite: {wf_comp:.4f}  |  {elapsed:.1f}s"
        )
        if w_scores:
            console.print(
                f"    Range: [{min(w_scores):.4f}, {max(w_scores):.4f}]  std={np.std(w_scores):.4f}"
            )

        # Final OOS scorecard with distributions
        final_model = get_model(model_name)
        if params:
            final_model.set_params(**params)
        final_model.fit(train_returns)
        n_oos = min(hz_days + 1, len(test_prices))
        oos_px = test_prices[:n_oos]
        oos_ret = test_returns[:min(hz_days, len(test_returns))]
        final_sim = final_model.simulate(
            n_simulations=N_SIMS, n_steps=len(oos_ret),
            initial_price=float(train_prices[-1]), seed=SEED,
        )
        final_card = scorer.score(
            final_sim, oos_px, oos_ret, hz_str, train_prices, include_distributions=True,
        )

        per_horizon[hz_str] = {
            "weighted_composite": wf_comp,
            "n_windows": wf_result["n_windows"],
            "elapsed_seconds": round(elapsed, 1),
            "window_composites": w_scores,
            "window_details": window_details,
            "final_distributions": final_card.distributions,
            "final_scoring_metrics": final_card.scoring_metrics,
        }

    # Cross-horizon
    vals = [per_horizon[hz]["weighted_composite"] for hz in HORIZONS]
    weighted = sum(per_horizon[hz]["weighted_composite"] * HZ_WEIGHTS[hz] for hz in HORIZONS)
    hz_std = float(np.std(vals))
    penalty = hz_std * 0.1
    final_score = weighted - penalty

    return {
        "model": model_name,
        "label": label,
        "params": params,
        "per_horizon": per_horizon,
        "cross_horizon": {
            "weighted_composite": round(weighted, 4),
            "horizon_std": round(hz_std, 4),
            "stability_penalty": round(penalty, 4),
            "final_score": round(final_score, 4),
        },
    }


def main():
    console.print("\n[bold]GARCH Walk-Forward Comparison: Config A vs Config B[/bold]")
    console.print(f"Config A: {CONFIGS['A']['label']} — {CONFIGS['A']['params']}")
    console.print(f"Config B: {CONFIGS['B']['label']} — {CONFIGS['B']['params']}")
    console.print(f"Standard mode: {N_SIMS} sims, {STEP_SIZE}d step")
    console.print()

    # Load data
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

    common_args = dict(
        full_prices=full_prices, full_returns=full_returns, dates=dates,
        train_returns=train_returns, train_prices=train_prices,
        test_prices=test_prices, test_returns=test_returns,
    )

    # Run Config A
    console.print("-" * 60)
    console.print(f"[bold]Config A: {CONFIGS['A']['label']}[/bold]")
    t0 = time.time()
    result_a = run_model_walkforward(
        "garch_1_1", CONFIGS["A"]["params"], CONFIGS["A"]["label"], **common_args,
    )
    elapsed_a = time.time() - t0
    console.print(f"\nConfig A total: {elapsed_a:.1f}s")

    # Run Config B
    console.print("\n" + "-" * 60)
    console.print(f"[bold]Config B: {CONFIGS['B']['label']}[/bold]")
    t0 = time.time()
    result_b = run_model_walkforward(
        "garch_1_1", CONFIGS["B"]["params"], CONFIGS["B"]["label"], **common_args,
    )
    elapsed_b = time.time() - t0
    console.print(f"\nConfig B total: {elapsed_b:.1f}s")

    # Run GBM baseline (shared)
    console.print("\n" + "-" * 60)
    console.print("[bold]GBM (MLE baseline)[/bold]")
    t0 = time.time()
    result_gbm = run_model_walkforward(
        "gbm", None, "GBM (MLE)", **common_args,
    )
    elapsed_gbm = time.time() - t0
    console.print(f"\nGBM total: {elapsed_gbm:.1f}s")

    # ── Report ─────────────────────────────────────────────────────
    a_ch = result_a["cross_horizon"]
    b_ch = result_b["cross_horizon"]
    gbm_ch = result_gbm["cross_horizon"]

    console.print("\n" + "=" * 70)
    console.print("[bold]GARCH WALK-FORWARD COMPARISON[/bold]")
    console.print("=" * 70)

    table = Table(title="Cross-Horizon Final Scores")
    table.add_column("Config")
    table.add_column("Final", justify="right")
    table.add_column("Composite", justify="right")
    table.add_column("Penalty", justify="right")
    table.add_column("Std", justify="right")
    table.add_row("A: GARCH(2,2)", f"{a_ch['final_score']:.4f}", f"{a_ch['weighted_composite']:.4f}",
                   f"{a_ch['stability_penalty']:.4f}", f"{a_ch['horizon_std']:.4f}")
    table.add_row("B: GARCH(1,1)", f"{b_ch['final_score']:.4f}", f"{b_ch['weighted_composite']:.4f}",
                   f"{b_ch['stability_penalty']:.4f}", f"{b_ch['horizon_std']:.4f}")
    table.add_row("GBM", f"{gbm_ch['final_score']:.4f}", f"{gbm_ch['weighted_composite']:.4f}",
                   f"{gbm_ch['stability_penalty']:.4f}", f"{gbm_ch['horizon_std']:.4f}")
    table.add_row("RBB (reference)", "0.8109", "0.8122", "0.0013", "0.0131")
    console.print(table)

    # Per-horizon
    hz_table = Table(title="Per-Horizon Walk-Forward Composites")
    hz_table.add_column("Horizon")
    hz_table.add_column("A: (2,2)", justify="right")
    hz_table.add_column("B: (1,1)", justify="right")
    hz_table.add_column("GBM", justify="right")
    hz_table.add_column("RBB ref", justify="right")
    hz_table.add_column("A vs RBB", justify="right")
    hz_table.add_column("B vs RBB", justify="right")

    rbb_ref = {"180d": 0.8098, "365d": 0.8221, "730d": 0.8265, "1460d": 0.7928}
    for hz in HORIZONS:
        a_v = result_a["per_horizon"][hz]["weighted_composite"]
        b_v = result_b["per_horizon"][hz]["weighted_composite"]
        g_v = result_gbm["per_horizon"][hz]["weighted_composite"]
        r_v = rbb_ref[hz]
        hz_table.add_row(hz, f"{a_v:.4f}", f"{b_v:.4f}", f"{g_v:.4f}", f"{r_v:.4f}",
                          f"{a_v - r_v:+.4f}", f"{b_v - r_v:+.4f}")
    console.print(hz_table)

    # Decision
    diff = a_ch["final_score"] - b_ch["final_score"]
    if diff > 0.01:
        winner = "A"
        reason = f"A beats B by {diff:.4f} (> 0.01 threshold)"
    elif diff < -0.01:
        winner = "B"
        reason = f"B beats A by {-diff:.4f} (> 0.01 threshold)"
    else:
        winner = "B"
        reason = f"Within 0.01 ({diff:+.4f}) — simpler model wins ties"

    console.print(f"\n[bold]Decision: Lock Config {winner}[/bold]")
    console.print(f"  {reason}")
    console.print(f"  Winner: {CONFIGS[winner]['label']} — {CONFIGS[winner]['params']}")

    # Save results
    output_dir = Path("results/phase3/garch_walkforward")
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

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "config_a": make_serializable(result_a),
        "config_b": make_serializable(result_b),
        "gbm": make_serializable(result_gbm),
        "decision": {"winner": winner, "reason": reason},
    }

    filepath = output_dir / "garch_walkforward_comparison.json"
    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    console.print(f"\nResults saved to {filepath}")
    console.print(f"Total time: {elapsed_a + elapsed_b + elapsed_gbm:.1f}s")


if __name__ == "__main__":
    main()
