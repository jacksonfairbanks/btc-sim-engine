#!/usr/bin/env python
"""
GARCH hyperparameter exhaustive grid search.

Tests all 36 combinations of (p, q, dist, mean_model) across all 4 horizons
using single-OOS evaluation. Same methodology as Phase 1 block length sweep.

Ranks by cross-horizon weighted composite:
  180d=10%, 365d=20%, 730d=35%, 1460d=35%, stability penalty = std×0.1

Usage:
    python scripts/run_garch_sweep.py
"""
import json
import time
import numpy as np
from datetime import datetime
from itertools import product
from pathlib import Path
from rich.console import Console
from rich.table import Table

from data.loader import BTCDataLoader, get_price_array
from models.registry import get_model
from validation.scorer import Scorer


console = Console()

HORIZONS = ["180d", "365d", "730d", "1460d"]
HZ_WEIGHTS = {"180d": 0.10, "365d": 0.20, "730d": 0.35, "1460d": 0.35}

# Search space
P_VALUES = [1, 2]
Q_VALUES = [1, 2]
DIST_VALUES = ["normal", "t", "skewt"]
MEAN_VALUES = ["Zero", "Constant", "ARX"]

N_SIMS = 2000  # Single-OOS evaluation, same as Phase 1 sweep


def run_single_oos(
    model_name: str,
    params: dict,
    train_returns: np.ndarray,
    train_prices: np.ndarray,
    test_prices: np.ndarray,
    test_returns: np.ndarray,
    horizon_days: int,
    scorer: Scorer,
    seed: int = 42,
) -> dict:
    """Fit model, simulate one OOS window, return composite + per-metric scores."""
    model = get_model(model_name)
    model.set_params(**params)

    try:
        model.fit(train_returns)
    except Exception as e:
        return {"composite": 0.0, "metrics": {}, "error": str(e)[:80]}

    initial_price = float(train_prices[-1])
    n_oos = min(horizon_days + 1, len(test_prices))
    oos_prices = test_prices[:n_oos]
    oos_returns = test_returns[:min(horizon_days, len(test_returns))]

    try:
        sim = model.simulate(
            n_simulations=N_SIMS,
            n_steps=len(oos_returns),
            initial_price=initial_price,
            seed=seed,
        )
    except Exception as e:
        return {"composite": 0.0, "metrics": {}, "error": str(e)[:80]}

    card = scorer.score(
        sim_result=sim,
        hist_prices=oos_prices,
        hist_returns=oos_returns,
        horizon=f"{horizon_days}d",
        train_prices=train_prices,
        include_distributions=False,
    )

    return {
        "composite": card.composite_score,
        "metrics": card.scoring_metrics,
    }


def main():
    console.print("\n[bold]GARCH Hyperparameter Exhaustive Grid Search[/bold]")
    console.print(f"Search space: p={P_VALUES}, q={Q_VALUES}, dist={DIST_VALUES}, mean={MEAN_VALUES}")
    total_configs = len(P_VALUES) * len(Q_VALUES) * len(DIST_VALUES) * len(MEAN_VALUES)
    console.print(f"Total configs: {total_configs}")
    console.print(f"Horizons: {HORIZONS}")
    console.print(f"Sims per evaluation: {N_SIMS}")
    console.print(f"Total evaluations: {total_configs * len(HORIZONS)}")
    console.print()

    # Load data
    loader = BTCDataLoader(ticker="BTC-USD", start_date="2013-01-01")
    train_df, test_df = loader.get_train_test_split(train_pct=0.7)
    train_returns = loader.get_returns_array(train_df)
    train_prices = get_price_array(train_df)
    test_prices = get_price_array(test_df)
    test_returns = loader.get_returns_array(test_df)

    scorer = Scorer()
    results = []
    t0_total = time.time()

    for i, (p, q, dist, mean_model) in enumerate(product(P_VALUES, Q_VALUES, DIST_VALUES, MEAN_VALUES)):
        params = {"p": p, "q": q, "dist": dist, "mean_model": mean_model}
        config_label = f"p={p} q={q} dist={dist:6s} mean={mean_model:8s}"

        per_hz = {}
        t0 = time.time()

        for hz_str in HORIZONS:
            hz_days = int(hz_str.replace("d", ""))
            result = run_single_oos(
                "garch_1_1", params,
                train_returns, train_prices, test_prices, test_returns,
                hz_days, scorer, seed=42,
            )
            per_hz[hz_str] = result

        elapsed = time.time() - t0

        # Cross-horizon composite
        hz_scores = [per_hz[hz]["composite"] for hz in HORIZONS]
        weighted = sum(per_hz[hz]["composite"] * HZ_WEIGHTS[hz] for hz in HORIZONS)
        hz_std = float(np.std(hz_scores))
        penalty = hz_std * 0.1
        final_score = weighted - penalty

        entry = {
            "params": params,
            "per_horizon": {hz: per_hz[hz]["composite"] for hz in HORIZONS},
            "per_horizon_metrics": {hz: per_hz[hz].get("metrics", {}) for hz in HORIZONS},
            "weighted_composite": round(weighted, 4),
            "horizon_std": round(hz_std, 4),
            "stability_penalty": round(penalty, 4),
            "final_score": round(final_score, 4),
            "elapsed": round(elapsed, 1),
            "errors": {hz: per_hz[hz].get("error") for hz in HORIZONS if per_hz[hz].get("error")},
        }
        results.append(entry)

        # Progress
        error_str = f"  ERRORS: {entry['errors']}" if entry["errors"] else ""
        console.print(
            f"  [{i+1:2d}/{total_configs}] {config_label}  "
            f"final={final_score:.4f}  "
            f"180d={per_hz['180d']['composite']:.4f}  "
            f"365d={per_hz['365d']['composite']:.4f}  "
            f"730d={per_hz['730d']['composite']:.4f}  "
            f"1460d={per_hz['1460d']['composite']:.4f}  "
            f"({elapsed:.1f}s){error_str}"
        )

    total_elapsed = time.time() - t0_total

    # Sort by final score
    results.sort(key=lambda x: x["final_score"], reverse=True)

    # Print ranked table
    console.print(f"\n[bold]Results (sorted by cross-horizon final score)[/bold]")
    table = Table()
    table.add_column("Rank", justify="right")
    table.add_column("p", justify="right")
    table.add_column("q", justify="right")
    table.add_column("dist")
    table.add_column("mean")
    table.add_column("Final", justify="right")
    table.add_column("180d", justify="right")
    table.add_column("365d", justify="right")
    table.add_column("730d", justify="right")
    table.add_column("1460d", justify="right")
    table.add_column("Std", justify="right")

    for rank, r in enumerate(results, 1):
        p = r["params"]
        table.add_row(
            str(rank),
            str(p["p"]), str(p["q"]), p["dist"], p["mean_model"],
            f"{r['final_score']:.4f}",
            f"{r['per_horizon']['180d']:.4f}",
            f"{r['per_horizon']['365d']:.4f}",
            f"{r['per_horizon']['730d']:.4f}",
            f"{r['per_horizon']['1460d']:.4f}",
            f"{r['horizon_std']:.4f}",
        )
    console.print(table)

    # Summary stats
    best = results[0]
    worst = results[-1]
    spread = best["final_score"] - worst["final_score"]

    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"Best:  {best['params']}  final={best['final_score']:.4f}")
    console.print(f"Worst: {worst['params']}  final={worst['final_score']:.4f}")
    console.print(f"Spread: {spread:.4f}")
    console.print(f"Default baseline (p=1,q=1,t,Constant) single-OOS final: "
                  f"{[r for r in results if r['params'] == {'p':1,'q':1,'dist':'t','mean_model':'Constant'}][0]['final_score']:.4f}")
    console.print(f"Total time: {total_elapsed:.0f}s")

    # Best per horizon
    console.print(f"\n[bold]Best config per horizon:[/bold]")
    for hz in HORIZONS:
        best_hz = max(results, key=lambda r: r["per_horizon"][hz])
        console.print(
            f"  {hz}: {best_hz['params']}  score={best_hz['per_horizon'][hz]:.4f}"
        )

    # Save results
    output_dir = Path("results/phase3/garch_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "garch_sweep_results.json"

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "search_space": {
            "p": P_VALUES, "q": Q_VALUES,
            "dist": DIST_VALUES, "mean_model": MEAN_VALUES,
        },
        "n_configs": total_configs,
        "n_sims": N_SIMS,
        "horizons": HORIZONS,
        "horizon_weights": HZ_WEIGHTS,
        "evaluation_method": "single_oos",
        "total_elapsed_seconds": round(total_elapsed, 1),
        "results": results,
        "summary": {
            "best": {"params": best["params"], "final_score": best["final_score"]},
            "worst": {"params": worst["params"], "final_score": worst["final_score"]},
            "spread": round(spread, 4),
        },
    }

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    console.print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
