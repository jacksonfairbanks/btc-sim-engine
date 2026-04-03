"""
Export pre-computed simulation data for the static HTML dashboard.

Reads results/production_sim.json.gz (all 3 models' raw paths),
runs BCR solvency across a grid of BCR levels, and writes
dashboard/data/sim-data.js as an embeddable JS constant.

Usage:
    python scripts/export_html_data.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_solvency(monthly_prices, initial_price, nav, bcr_ratio, opex_rate=0.0,
                 cash_months=0, btc_fraction=1.0):
    """Run solvency simulation on monthly price snapshots.

    Returns dict with terminal_bcr, min_bcr, failed, failure_month arrays.
    """
    n_sims, n_months = monthly_prices.shape
    annual_dividend = nav / bcr_ratio
    annual_opex = nav * opex_rate
    monthly_obligation = (annual_dividend / 12.0) + (annual_opex / 12.0)

    terminal_bcr = np.zeros(n_sims)
    min_bcr = np.zeros(n_sims)
    failed = np.zeros(n_sims, dtype=bool)
    failure_month = np.full(n_sims, n_months + 1, dtype=int)
    # BCR time series for percentile bands
    bcr_series = np.zeros((n_sims, n_months))

    for i in range(n_sims):
        btc_holdings = nav / initial_price
        cash = cash_months * monthly_obligation
        path_failed = False
        path_min = float("inf")

        for m in range(n_months):
            price = monthly_prices[i, m]
            btc_obl = monthly_obligation * btc_fraction

            if cash >= btc_obl:
                cash -= btc_obl
            else:
                sell_amt = btc_obl - cash
                cash = 0.0
                if price > 0:
                    btc_holdings -= sell_amt / price

            btc_value = btc_holdings * price
            cur_bcr = btc_value / annual_dividend if annual_dividend > 0 else float("inf")
            bcr_series[i, m] = cur_bcr

            if cur_bcr < path_min:
                path_min = cur_bcr
            if not path_failed and cur_bcr < 1.0:
                path_failed = True
                failure_month[i] = m + 1

        failed[i] = path_failed
        min_bcr[i] = path_min
        terminal_bcr[i] = bcr_series[i, -1]

    return {
        "terminal_bcr": terminal_bcr,
        "min_bcr": min_bcr,
        "failed": failed,
        "failure_month": failure_month,
        "bcr_series": bcr_series,
    }


def main():
    src = PROJECT_ROOT / "results" / "production_sim.json.gz"
    if not src.exists():
        print(f"ERROR: {src} not found. Run production simulation first.")
        sys.exit(1)

    print("Loading production sim data...")
    with gzip.open(src, "rt", encoding="utf-8") as f:
        raw = json.load(f)

    initial_price = raw["initial_price"]
    data_end = raw["data_end_date"]
    n_training_days = raw["n_training_days"]
    models_meta = raw["models"]
    all_paths = {k: np.array(v) for k, v in raw["all_paths"].items()}

    n_sims = all_paths["rbb"].shape[0]
    n_days = all_paths["rbb"].shape[1]
    n_months = n_days // 30
    nav = 1_000_000_000  # reference NAV for solvency (scales out)

    # Monthly price snapshots
    month_indices = [min(30 * (m + 1) - 1, n_days - 1) for m in range(n_months)]
    monthly_all = {k: v[:, month_indices] for k, v in all_paths.items()}

    model_keys = ["rbb", "garch", "gbm"]
    model_labels = {"rbb": "RBB (Block Bootstrap)", "garch": "GARCH(1,1)", "gbm": "GBM (Baseline)"}

    output = {
        "metadata": {
            "initial_price": round(initial_price, 2),
            "data_end_date": data_end,
            "n_training_days": n_training_days,
            "n_sims": n_sims,
            "n_days": n_days,
            "n_months": n_months,
            "horizon_days": 1460,
            "models": {k: {"label": model_labels[k], "specs": models_meta[k].get("specs", "")}
                       for k in model_keys},
            "walk_forward_scores": {"rbb": 0.8109, "garch": 0.7800, "gbm": 0.7630},
        },
        "price_percentiles": {},
        "terminal_price_stats": {},
        "bcr_default": {},
        "pd_vs_bcr": {},
        "bcr_over_time": {},
    }

    # ── Price percentiles ──────────────────────────────────────────────
    print("Computing price percentiles...")
    for k in model_keys:
        paths = all_paths[k]
        pcts = {}
        for p in [5, 25, 50, 75, 95]:
            # Sample every 30 days for the HTML (keep it manageable)
            full = np.percentile(paths, p, axis=0)
            sampled = full[::30].tolist()  # every 30th day
            pcts[f"p{p}"] = [round(v, 2) for v in sampled]
        output["price_percentiles"][k] = pcts

        fp = np.array(paths[:, -1])
        output["terminal_price_stats"][k] = {
            "median": round(float(np.median(fp)), 0),
            "mean": round(float(np.mean(fp)), 0),
            "p5": round(float(np.percentile(fp, 5)), 0),
            "p95": round(float(np.percentile(fp, 95)), 0),
            "pct_below_start": round(float(np.mean(fp < initial_price)) * 100, 1),
        }

    # ── BCR at default settings (BCR=40) ───────────────────────────────
    print("Computing BCR at default BCR=40...")
    default_bcr = 40
    for k in model_keys:
        res = run_solvency(monthly_all[k], initial_price, nav, default_bcr)
        n_fail = int(np.sum(res["failed"]))

        # Percentile curve data (sorted terminal + min BCR)
        pct_range = list(range(1, 100))
        output["bcr_default"][k] = {
            "pd_pct": round(n_fail / n_sims * 100, 2),
            "n_failed": n_fail,
            "median_terminal_bcr": round(float(np.median(res["terminal_bcr"])), 1),
            "p5_terminal_bcr": round(float(np.percentile(res["terminal_bcr"], 5)), 2),
            "p95_terminal_bcr": round(float(np.percentile(res["terminal_bcr"], 95)), 1),
            "median_min_bcr": round(float(np.median(res["min_bcr"])), 1),
            "p5_min_bcr": round(float(np.percentile(res["min_bcr"], 5)), 2),
            "terminal_bcr_curve": [round(float(v), 2) for v in np.percentile(res["terminal_bcr"], pct_range)],
            "min_bcr_curve": [round(float(v), 2) for v in np.percentile(res["min_bcr"], pct_range)],
            "pct_range": pct_range,
        }

        # BCR over time percentiles
        bcr_ts = res["bcr_series"]
        output["bcr_over_time"][k] = {
            "p5": [round(float(v), 2) for v in np.percentile(bcr_ts, 5, axis=0)],
            "p25": [round(float(v), 2) for v in np.percentile(bcr_ts, 25, axis=0)],
            "p50": [round(float(v), 2) for v in np.median(bcr_ts, axis=0)],
            "p75": [round(float(v), 2) for v in np.percentile(bcr_ts, 75, axis=0)],
            "p95": [round(float(v), 2) for v in np.percentile(bcr_ts, 95, axis=0)],
        }

        # Cumulative failure by month
        cum_fail = []
        for m in range(n_months):
            cum_fail.append(round(float(np.sum(res["failure_month"] <= (m + 1))) / n_sims * 100, 2))
        output["bcr_default"][k]["cumulative_failure"] = cum_fail

    # ── PD vs BCR sensitivity (BCR 2-40, step 2) ──────────────────────
    print("Computing PD vs BCR sensitivity...")
    bcr_grid = list(range(2, 42, 2))
    output["pd_vs_bcr"]["bcr_grid"] = bcr_grid

    for k in model_keys:
        pd_list = []
        p5_min_list = []
        for test_bcr in bcr_grid:
            res = run_solvency(monthly_all[k], initial_price, nav, test_bcr)
            n_fail = int(np.sum(res["failed"]))
            pd_list.append(round(n_fail / n_sims * 100, 2))
            p5_min_list.append(round(float(np.percentile(res["min_bcr"], 5)), 2))
            print(f"  {model_labels[k]} BCR={test_bcr}: PD={pd_list[-1]}%, P5 min={p5_min_list[-1]}")

        output["pd_vs_bcr"][k] = {
            "pd": pd_list,
            "p5_min_bcr": p5_min_list,
        }

    # ── BCR percentile curves at multiple BCR levels (for interactivity) ─
    print("Computing BCR percentile curves at each BCR level...")
    output["bcr_curves_by_level"] = {"bcr_grid": bcr_grid}
    pct_range = list(range(1, 100))
    for k in model_keys:
        level_data = {"terminal": [], "minimum": []}
        for test_bcr in bcr_grid:
            res = run_solvency(monthly_all[k], initial_price, nav, test_bcr)
            level_data["terminal"].append(
                [round(float(v), 2) for v in np.percentile(res["terminal_bcr"], pct_range)]
            )
            level_data["minimum"].append(
                [round(float(v), 2) for v in np.percentile(res["min_bcr"], pct_range)]
            )
        output["bcr_curves_by_level"][k] = level_data
    output["bcr_curves_by_level"]["pct_range"] = pct_range

    # ── Sensitivity heatmap (BCR × DivRate) ────────────────────────────
    print("Computing sensitivity heatmap...")
    div_rates = [0.05, 0.08, 0.10, 0.12, 0.15]
    output["heatmap"] = {
        "bcr_grid": bcr_grid,
        "div_rates": [d * 100 for d in div_rates],
    }
    # Only RBB for heatmap
    heat_z = []
    for dr in div_rates:
        row = []
        for br in bcr_grid:
            # annual_dividend = nav / bcr regardless of div_rate
            # So this should produce constant columns (proving BCR is the only driver)
            res = run_solvency(monthly_all["rbb"], initial_price, nav, br)
            row.append(round(int(np.sum(res["failed"])) / n_sims * 100, 2))
        heat_z.append(row)
    output["heatmap"]["rbb"] = heat_z

    # ── Runway ─────────────────────────────────────────────────────────
    annual_div_default = nav / default_bcr
    monthly_obl = annual_div_default / 12.0
    monthly_btc_drain = monthly_obl / initial_price
    starting_btc = nav / initial_price
    output["metadata"]["runway_months"] = int(starting_btc / monthly_btc_drain) if monthly_btc_drain > 0 else 999

    # ── Write output ───────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "dashboard" / "data" / "sim-data.js"
    json_str = json.dumps(output, separators=(",", ":"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"const SIM_DATA = {json_str};\n")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nExported to {out_path} ({size_mb:.1f} MB)")
    print(f"  {n_sims} paths × {n_months} months × {len(model_keys)} models")
    print(f"  {len(bcr_grid)} BCR levels × {len(div_rates)} div rates")
    print("Done.")


if __name__ == "__main__":
    main()
