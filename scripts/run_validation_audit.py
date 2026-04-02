#!/usr/bin/env python
"""
Pre-publication validation audit.

Re-runs all three models (RBB, GARCH, GBM) through the full walk-forward
pipeline from scratch and compares results against documented values in
CLAUDE.md. Flags any discrepancies.

Usage:
    python scripts/run_validation_audit.py
"""
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table

from data.loader import BTCDataLoader, get_price_array
from models.registry import get_model
from validation.scorer import Scorer
from validation.backtest import WalkForwardBacktest


console = Console()

HORIZONS = ["180d", "365d", "730d", "1460d"]
HZ_WEIGHTS = {"180d": 0.10, "365d": 0.20, "730d": 0.35, "1460d": 0.35}
N_SIMS = 5000
STEP_SIZE = 90
SEED = 42

MODELS = {
    "regime_block_bootstrap": {
        "label": "RBB",
        "params": {
            "block_length_sampling": "geometric",
            "mean_block_length": 30,
            "min_block_length": 5,
            "block_stride": 5,
            "min_pool_size": 8,
            "regime_enabled": False,
        },
    },
    "garch_1_1": {
        "label": "GARCH(1,1)",
        "params": {"p": 1, "q": 1, "dist": "t", "mean_model": "Constant"},
    },
    "gbm": {
        "label": "GBM",
        "params": {},
    },
}

# Documented values from CLAUDE.md to validate against
EXPECTED = {
    "RBB": {
        "final_score": 0.8109,
        "per_horizon": {"180d": 0.8098, "365d": 0.8221, "730d": 0.8265, "1460d": 0.7928},
        "n_windows": {"180d": 37, "365d": 35, "730d": 31, "1460d": 23},
    },
    "GARCH(1,1)": {
        "final_score": 0.7800,
        "per_horizon": {"180d": 0.7973, "365d": 0.8024, "730d": 0.7976, "1460d": 0.7506},
        "n_windows": {"180d": 37, "365d": 35, "730d": 31, "1460d": 23},
    },
    "GBM": {
        "final_score": 0.7630,
        "per_horizon": {"180d": 0.7589, "365d": 0.7705, "730d": 0.7618, "1460d": 0.7625},
        "n_windows": {"180d": 37, "365d": 35, "730d": 31, "1460d": 23},
    },
}

TOLERANCE = 0.0005  # Allow tiny floating point differences


def run_model(model_name, config, full_prices, full_returns, dates,
              train_returns, train_prices, test_prices, test_returns):
    """Run full walk-forward for one model."""
    scorer = Scorer()
    per_horizon = {}

    for hz_str in HORIZONS:
        hz_days = int(hz_str.replace("d", ""))
        model = get_model(model_name)
        if config["params"]:
            model.set_params(**config["params"])

        backtest = WalkForwardBacktest(
            prices=full_prices, returns=full_returns,
            min_training_days=730, step_size_days=STEP_SIZE,
            recency_weighting="exponential", recency_half_life_days=730,
            dates=dates,
        )

        t0 = time.time()
        wf = backtest.run(model=model, horizon_days=hz_days,
                          n_simulations=N_SIMS, scorer=scorer, seed=SEED)
        elapsed = time.time() - t0

        # Verify config on every window
        config_ok_count = 0
        for card in wf.get("window_scores", []):
            fp = card._fitted_params if hasattr(card, '_fitted_params') else card.params
            if config["params"]:
                ok = all(fp.get(k) == v for k, v in config["params"].items() if k in fp)
            else:
                ok = True
            if ok:
                config_ok_count += 1

        per_horizon[hz_str] = {
            "weighted_composite": wf["weighted_composite"],
            "n_windows": wf["n_windows"],
            "config_ok": config_ok_count,
            "elapsed": round(elapsed, 1),
        }

        console.print(
            f"  {config['label']:12s} @ {hz_str:>6s}  "
            f"wf={wf['weighted_composite']:.4f}  "
            f"windows={wf['n_windows']}  "
            f"config_ok={config_ok_count}/{wf['n_windows']}  "
            f"({elapsed:.1f}s)"
        )

    # Cross-horizon
    vals = [per_horizon[hz]["weighted_composite"] for hz in HORIZONS]
    weighted = sum(per_horizon[hz]["weighted_composite"] * HZ_WEIGHTS[hz] for hz in HORIZONS)
    hz_std = float(np.std(vals))
    penalty = hz_std * 0.1
    final = weighted - penalty

    return {
        "label": config["label"],
        "final_score": round(final, 4),
        "weighted_composite": round(weighted, 4),
        "stability_penalty": round(penalty, 4),
        "horizon_std": round(hz_std, 4),
        "per_horizon": per_horizon,
    }


def main():
    console.print("\n[bold]PRE-PUBLICATION VALIDATION AUDIT[/bold]")
    console.print(f"Models: {[m['label'] for m in MODELS.values()]}")
    console.print(f"Pipeline: {N_SIMS} sims, {STEP_SIZE}d step, seed={SEED}")
    console.print(f"Tolerance: {TOLERANCE}")
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

    console.print(f"Data: {dates[0]} to {dates[-1]} ({len(full_prices)} days)")
    console.print(f"Train: {len(train_prices)}  Test: {len(test_prices)}")
    console.print()

    # Run all models
    results = {}
    t0_total = time.time()

    for model_name, config in MODELS.items():
        console.print(f"[bold]{config['label']}[/bold]")
        results[config["label"]] = run_model(
            model_name, config, full_prices, full_returns, dates,
            train_returns, train_prices, test_prices, test_returns,
        )
        console.print()

    total_elapsed = time.time() - t0_total

    # ── Validation against documented values ───────────────────────
    console.print("=" * 70)
    console.print("[bold]VALIDATION RESULTS[/bold]")
    console.print("=" * 70)

    discrepancies = []

    for label, expected in EXPECTED.items():
        actual = results[label]

        # Final score
        diff = abs(actual["final_score"] - expected["final_score"])
        status = "PASS" if diff <= TOLERANCE else "FAIL"
        if status == "FAIL":
            discrepancies.append(f"{label} final_score: expected {expected['final_score']:.4f}, got {actual['final_score']:.4f} (diff {diff:.4f})")
        console.print(
            f"  [{status}] {label:12s} final_score: expected={expected['final_score']:.4f}  "
            f"actual={actual['final_score']:.4f}  diff={diff:.4f}"
        )

        # Per-horizon
        for hz in HORIZONS:
            exp_wf = expected["per_horizon"][hz]
            act_wf = actual["per_horizon"][hz]["weighted_composite"]
            diff = abs(act_wf - exp_wf)
            status = "PASS" if diff <= TOLERANCE else "FAIL"
            if status == "FAIL":
                discrepancies.append(f"{label} {hz}: expected {exp_wf:.4f}, got {act_wf:.4f} (diff {diff:.4f})")
            console.print(
                f"  [{status}] {label:12s} {hz:>6s}: expected={exp_wf:.4f}  "
                f"actual={act_wf:.4f}  diff={diff:.4f}"
            )

        # Window counts
        for hz in HORIZONS:
            exp_n = expected["n_windows"][hz]
            act_n = actual["per_horizon"][hz]["n_windows"]
            status = "PASS" if act_n == exp_n else "FAIL"
            if status == "FAIL":
                discrepancies.append(f"{label} {hz} window_count: expected {exp_n}, got {act_n}")

        # Config OK
        for hz in HORIZONS:
            n_win = actual["per_horizon"][hz]["n_windows"]
            n_ok = actual["per_horizon"][hz]["config_ok"]
            status = "PASS" if n_ok == n_win else "FAIL"
            if status == "FAIL":
                discrepancies.append(f"{label} {hz} config_ok: {n_ok}/{n_win}")

    console.print()

    # Summary
    if discrepancies:
        console.print(f"[bold red]AUDIT FAILED — {len(discrepancies)} discrepancies:[/bold red]")
        for d in discrepancies:
            console.print(f"  [red]- {d}[/red]")
    else:
        console.print("[bold green]AUDIT PASSED — all values match documented results[/bold green]")

    # Summary table
    console.print()
    table = Table(title="Validation Summary")
    table.add_column("Model")
    table.add_column("Expected Final", justify="right")
    table.add_column("Actual Final", justify="right")
    table.add_column("Diff", justify="right")
    table.add_column("Status")

    for label in ["RBB", "GARCH(1,1)", "GBM"]:
        exp = EXPECTED[label]["final_score"]
        act = results[label]["final_score"]
        diff = abs(act - exp)
        status = "[green]PASS[/green]" if diff <= TOLERANCE else "[red]FAIL[/red]"
        table.add_row(label, f"{exp:.4f}", f"{act:.4f}", f"{diff:.4f}", status)
    console.print(table)

    console.print(f"\nTotal time: {total_elapsed:.0f}s")

    # Save audit results
    output_dir = Path("results/validation_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_data = {
        "timestamp": datetime.now().isoformat(),
        "tolerance": TOLERANCE,
        "passed": len(discrepancies) == 0,
        "discrepancies": discrepancies,
        "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        "expected": EXPECTED,
        "total_elapsed": round(total_elapsed, 1),
    }
    filepath = output_dir / "validation_audit.json"
    with open(filepath, "w") as f:
        json.dump(audit_data, f, indent=2, default=str)
    console.print(f"Audit saved to {filepath}")


if __name__ == "__main__":
    main()
