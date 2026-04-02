"""
Pipeline Audit — full step-by-step audit of one walk-forward window.

Computes every diagnostic for a single model+window+horizon combination,
including sanity checks with pass/fail flags. Designed for dashboard
rendering and JSON/Markdown export.
"""
import time
import json
import numpy as np
from datetime import datetime
from typing import Any

from models.base import BaseModel
from models.registry import get_model
from validation.scorer import Scorer


def _safe_float(v: Any) -> float | None:
    """Convert to float, handle None/inf/nan."""
    if v is None:
        return None
    v = float(v)
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def run_audit(
    model_name: str,
    train_returns: np.ndarray,
    train_prices: np.ndarray,
    test_prices: np.ndarray,
    test_returns: np.ndarray,
    horizon_days: int,
    n_simulations: int = 2000,
    seed: int = 42,
    train_start_date: str | None = None,
    train_end_date: str | None = None,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    model_params: dict | None = None,
) -> dict:
    """
    Run full pipeline audit for one window.

    Returns a structured dict matching the audit spec with all sections
    and red flags.
    """
    flags: list[str] = []
    sections: dict[str, dict] = {}
    section_pass: dict[str, bool] = {}
    timings: dict[str, float] = {}

    # ════════════════════════════════════════════════════════════════
    # Section 1: Data Selection
    # ════════════════════════════════════════════════════════════════
    t0 = time.time()
    train_days = len(train_returns)
    train_years = train_days / 365.25
    total_train_return = float(np.exp(np.sum(train_returns)) - 1)

    s1 = {
        "train_start": train_start_date,
        "train_end": train_end_date,
        "train_days": train_days,
        "train_years": round(train_years, 1),
        "test_start": test_start_date,
        "test_end": test_end_date,
        "test_days": len(test_returns),
        "train_stats": {
            "mean_daily_return": _safe_float(np.mean(train_returns)),
            "std_daily_return": _safe_float(np.std(train_returns)),
            "min_daily_return": _safe_float(np.min(train_returns)),
            "max_daily_return": _safe_float(np.max(train_returns)),
            "total_return_pct": round(total_train_return * 100, 1),
        },
        "train_prices": train_prices[::max(1, len(train_prices)//200)].tolist(),
    }

    s1_flags = []
    if train_days < 730:
        s1_flags.append("Training period < 730 days")
    s1["flags"] = s1_flags
    flags.extend(s1_flags)
    sections["data_selection"] = s1
    section_pass["data_selection"] = len(s1_flags) == 0
    timings["data_selection"] = time.time() - t0

    # ════════════════════════════════════════════════════════════════
    # Fit model (handles both regime and non-regime paths)
    # ════════════════════════════════════════════════════════════════
    t0 = time.time()
    model = get_model(model_name)
    if model_params:
        model.set_params(**model_params)
    model.fit(train_returns)

    is_rbb = model_name == "regime_block_bootstrap"
    regime_enabled = getattr(model, "regime_enabled", False) if is_rbb else False

    # ════════════════════════════════════════════════════════════════
    # Section 2: Block Pool Construction
    # ════════════════════════════════════════════════════════════════
    t0 = time.time()
    s2_bp = {"model_name": model_name, "regime_enabled": regime_enabled}
    s2_bp_flags = []

    if is_rbb:
        pools = model._block_pools
        n_reg = getattr(model, "_n_regimes", 1)
        regime_labels_map = {0: "pool" if n_reg == 1 else "bull",
                             1: "bear", 2: "chop"}

        for regime in range(n_reg):
            label = regime_labels_map.get(regime, f"regime_{regime}")
            pool = pools.get(regime, [])
            lens = [len(b) for b in pool]
            s2_bp[f"{label}_pool_size"] = len(pool)
            if lens:
                s2_bp[f"{label}_min_len"] = min(lens)
                s2_bp[f"{label}_max_len"] = max(lens)
                s2_bp[f"{label}_median_len"] = int(np.median(lens))
            else:
                s2_bp[f"{label}_min_len"] = 0
                s2_bp[f"{label}_max_len"] = 0
                s2_bp[f"{label}_median_len"] = 0

            if len(pool) < 8:
                s2_bp_flags.append(f"{label.title()} pool has fewer than 8 blocks ({len(pool)})")

            if lens and max(lens) > train_days * 0.3:
                s2_bp_flags.append(
                    f"Largest {label} block is {max(lens)/train_days*100:.0f}% "
                    f"of training data — pool diversity is low"
                )

        s2_bp["block_length_sampling"] = model.block_length_sampling
        s2_bp["mean_block_length"] = model.mean_block_length
        s2_bp["block_stride"] = model.block_stride
        s2_bp["n_regimes"] = n_reg
    else:
        s2_bp["params"] = model.get_params()
        s2_bp["fitted_params"] = model._fitted_params

    s2_bp["flags"] = s2_bp_flags
    flags.extend(s2_bp_flags)
    sections["block_pools"] = s2_bp
    section_pass["block_pools"] = len(s2_bp_flags) == 0
    timings["block_pools"] = time.time() - t0

    # ════════════════════════════════════════════════════════════════
    # Section 3: Regime Classification (only if regime_enabled)
    # ════════════════════════════════════════════════════════════════
    s3_regime = {"regime_enabled": regime_enabled}
    s3_flags = []

    if is_rbb and regime_enabled:
        tm = model._transition_matrix
        labels = model._regime_labels
        regime_means = model._regime_means
        regime_vars = model._regime_variances

        bull_days = int(np.sum(labels == 0))
        bear_days = int(np.sum(labels == 1)) if model._n_regimes > 1 else 0
        bull_pct = bull_days / len(labels) * 100
        bear_pct = bear_days / len(labels) * 100 if model._n_regimes > 1 else 0

        bull_runs, bear_runs = [], []
        current_run = 1
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                current_run += 1
            else:
                (bull_runs if labels[i - 1] == 0 else bear_runs).append(current_run)
                current_run = 1
        (bull_runs if labels[-1] == 0 else bear_runs).append(current_run)

        s3_regime.update({
            "msm_converged": model._msm_converged,
            "fallback_used": model._fallback_used,
            "convergence_log": model._convergence_log,
            "transition_matrix": tm.tolist(),
            "bull_days": bull_days,
            "bear_days": bear_days,
            "bull_pct": round(bull_pct, 1),
            "bear_pct": round(bear_pct, 1),
            "avg_bull_run_days": round(float(np.mean(bull_runs)), 1) if bull_runs else 0,
            "avg_bear_run_days": round(float(np.mean(bear_runs)), 1) if bear_runs else 0,
            "regime_stats": {
                "bull_mean": _safe_float(regime_means[0]),
                "bull_std": _safe_float(np.sqrt(regime_vars[0])),
                "bear_mean": _safe_float(regime_means[1]) if model._n_regimes > 1 else None,
                "bear_std": _safe_float(np.sqrt(regime_vars[1])) if model._n_regimes > 1 else None,
            },
        })

        if not model._msm_converged:
            s3_flags.append("MSM failed to converge — using empirical fallback")
        if model._n_regimes > 1 and min(bull_pct, bear_pct) < 15:
            s3_flags.append(
                f"One regime contains only {min(bull_pct, bear_pct):.0f}% of training data"
            )
        avg_bear = float(np.mean(bear_runs)) if bear_runs else 0
        if avg_bear < 30 and model._n_regimes > 1:
            s3_flags.append(f"Average bear period {avg_bear:.0f}d < 30d")

    s3_regime["flags"] = s3_flags
    flags.extend(s3_flags)
    sections["regime_classification"] = s3_regime
    section_pass["regime_classification"] = len(s3_flags) == 0

    # ════════════════════════════════════════════════════════════════
    # Section 3b + 4 + 4c: Simulation with audit tracking
    # ════════════════════════════════════════════════════════════════
    t0 = time.time()
    initial_price = float(train_prices[-1])
    n_oos = min(horizon_days + 1, len(test_prices))
    oos_prices = test_prices[:n_oos]
    oos_returns = test_returns[:min(horizon_days, len(test_returns))]

    sim_kwargs = dict(
        n_simulations=n_simulations,
        n_steps=len(oos_returns),
        initial_price=initial_price,
        seed=seed,
    )
    if is_rbb:
        sim_kwargs["audit_mode"] = True
    sim_result = model.simulate(**sim_kwargs)
    timings["simulation"] = time.time() - t0

    # Section 3b: Resampling Verification
    s3b = {}
    s3b_flags = []
    audit_data = sim_result.metadata.get("audit", {})
    if audit_data:
        for regime, label in [(0, "bull"), (1, "bear")]:
            usage = np.array(audit_data.get(f"block_usage_{label}", []))
            total_draws = int(np.sum(usage))
            used = int(np.sum(usage > 0))
            pool_total = audit_data.get(f"{label}_pool_total", 0)

            s3b[f"{label}_blocks_used"] = used
            s3b[f"{label}_pool_total"] = pool_total
            s3b[f"{label}_utilization_pct"] = round(used / pool_total * 100, 1) if pool_total > 0 else 0
            s3b[f"{label}_usage_counts"] = usage.tolist()

            if total_draws > 0:
                max_usage_pct = float(np.max(usage)) / total_draws * 100
                s3b[f"{label}_max_single_block_pct"] = round(max_usage_pct, 1)
                if max_usage_pct > 20:
                    s3b_flags.append(
                        f"Single {label} block accounts for {max_usage_pct:.0f}% of draws"
                    )
                dead = int(np.sum(usage == 0))
                if dead > 0:
                    s3b_flags.append(
                        f"{dead} {label} blocks were never sampled (dead blocks)"
                    )

    s3b["flags"] = s3b_flags
    flags.extend(s3b_flags)
    sections["resampling_verification"] = s3b
    section_pass["resampling_verification"] = len(s3b_flags) == 0

    # Section 4: Simulation Output
    s4 = {}
    s4_flags = []
    paths = sim_result.paths

    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    inside = (oos_prices[:paths.shape[1]] >= p5[:len(oos_prices)]) & \
             (oos_prices[:paths.shape[1]] <= p95[:len(oos_prices)])
    ci_coverage = float(np.mean(inside)) * 100

    # Per-path stats
    final_prices = paths[:, -1]
    max_dds = []
    paths_50dd = 0
    paths_75dd = 0
    for i in range(paths.shape[0]):
        rm = np.maximum.accumulate(paths[i])
        dd = (paths[i] - rm) / rm
        max_dd = float(np.min(dd))
        max_dds.append(max_dd)
        if max_dd <= -0.50:
            paths_50dd += 1
        if max_dd <= -0.75:
            paths_75dd += 1

    actual_final = float(oos_prices[-1]) if len(oos_prices) > 0 else None
    median_final = float(np.median(final_prices))

    s4.update({
        "n_paths": n_simulations,
        "n_steps": paths.shape[1] - 1,
        "median_final_price": round(median_final, 2),
        "mean_final_price": round(float(np.mean(final_prices)), 2),
        "p5_final_price": round(float(np.percentile(final_prices, 5)), 2),
        "p95_final_price": round(float(np.percentile(final_prices, 95)), 2),
        "actual_final_price": round(actual_final, 2) if actual_final else None,
        "pct_below_start": round(float(np.mean(final_prices < initial_price)) * 100, 1),
        "pct_50_drawdown": round(paths_50dd / n_simulations * 100, 1),
        "pct_75_drawdown": round(paths_75dd / n_simulations * 100, 1),
        "max_drawdown_all_paths": round(float(np.min(max_dds)) * 100, 1),
        "ci_coverage_pct": round(ci_coverage, 1),
        # 10 sample paths for plotting (evenly spaced)
        "sample_paths": paths[::max(1, n_simulations // 10)][:10].tolist(),
        "realized_prices": oos_prices.tolist(),
        "p5": p5.tolist(),
        "p95": p95.tolist(),
    })

    # Compute tail events using distributions module
    from validation.distributions import compute_distributions
    tail_dist = compute_distributions(
        sim_result.paths, sim_result.log_returns, oos_prices, oos_returns,
    )
    tail_events = tail_dist.get("tail_events", {})
    s4["tail_events"] = tail_events

    if paths_50dd == 0:
        s4_flags.append("0% of paths produced a 50%+ drawdown — may not capture tail events")
    if ci_coverage < 50:
        s4_flags.append(
            f"Realized path fell outside 90% CI for {100 - ci_coverage:.0f}% of test period"
        )
    if actual_final and median_final > 3 * actual_final:
        s4_flags.append(
            f"Median final price ${median_final:,.0f} is >{3}x realized "
            f"${actual_final:,.0f} — strong directional bias"
        )
    # Flag named scenarios that produce 0 paths
    for scenario_key, scenario in tail_events.get("named_scenarios", {}).items():
        if scenario["count"] == 0:
            # Only flag if horizon is long enough for the scenario to be possible
            min_dur = {"2014_mt_gox": 390, "2018_crash": 365,
                       "2022_crash": 390, "2020_flash_crash": 30}
            if horizon_days >= min_dur.get(scenario_key, 0):
                s4_flags.append(
                    f"0 of {tail_events['n_paths']} paths produced "
                    f"{scenario_key.replace('_', ' ')} scenario — "
                    f"model cannot produce this known historical outcome"
                )

    s4["flags"] = s4_flags
    flags.extend(s4_flags)
    sections["simulation_output"] = s4
    section_pass["simulation_output"] = len(s4_flags) == 0

    # Section 4b: Simulation Convergence
    t0 = time.time()
    s4b = {}
    s4b_flags = []
    convergence_scores = {}
    scorer = Scorer()
    for n_paths in [500, 1000, 2000]:
        if n_paths > n_simulations:
            break
        sub_result = type(sim_result)(
            paths=sim_result.paths[:n_paths],
            log_returns=sim_result.log_returns[:n_paths],
            params_used=sim_result.params_used,
            model_name=sim_result.model_name,
            metadata=sim_result.metadata,
        )
        score = scorer.score_quick(sub_result, oos_prices, oos_returns)
        convergence_scores[n_paths] = round(score, 4)

    s4b["scores_by_path_count"] = convergence_scores
    counts = sorted(convergence_scores.keys())
    if len(counts) >= 2:
        delta = abs(convergence_scores[counts[-1]] - convergence_scores[counts[-2]])
        s4b["last_delta"] = round(delta, 4)
        s4b["last_delta_pct"] = round(delta * 100, 2)
        if delta > 0.02:
            s4b_flags.append(
                f"Score delta between {counts[-2]} and {counts[-1]} paths: "
                f"{delta*100:.1f}% — score has not converged"
            )
        elif delta < 0.005:
            s4b["converged_note"] = f"Converged at ~{counts[-2]} paths"

    s4b["flags"] = s4b_flags
    flags.extend(s4b_flags)
    sections["simulation_convergence"] = s4b
    section_pass["simulation_convergence"] = len(s4b_flags) == 0
    timings["convergence_check"] = time.time() - t0

    # Section 4c: Regime Transitions During Simulation
    # NOTE: The model evaluates regime transitions at BLOCK BOUNDARIES,
    # not every day. The transition matrix gives daily probabilities, but
    # the effective switch rate depends on mean block length. Compare
    # average run lengths rather than raw switch frequencies.
    s4c = {}
    s4c_flags = []
    if audit_data and is_rbb and regime_enabled and model._transition_matrix.shape[0] >= 2:
        tm = model._transition_matrix
        daily_bull_to_bear = float(tm[0, 1])
        daily_bear_to_bull = float(tm[1, 0])
        mean_bl = model.mean_block_length

        # Expected run length in simulation:
        # At each block boundary, there's a daily_transition_prob chance of switching.
        # Expected number of blocks before switch = 1 / daily_transition_prob.
        # Expected run length in days = mean_block_length / daily_transition_prob.
        # (Each block contributes ~mean_block_length days, and we need ~1/p blocks)
        expected_bull_run = mean_bl / daily_bull_to_bear if daily_bull_to_bear > 0 else float('inf')
        expected_bear_run = mean_bl / daily_bear_to_bull if daily_bear_to_bull > 0 else float('inf')

        sim_bull_runs = audit_data.get("sim_bull_run_lengths", [])
        sim_bear_runs = audit_data.get("sim_bear_run_lengths", [])
        avg_sim_bull = float(np.mean(sim_bull_runs)) if sim_bull_runs else 0
        avg_sim_bear = float(np.mean(sim_bear_runs)) if sim_bear_runs else 0

        hist_bull_run = sections["regime_classification"].get("avg_bull_run_days", 0)
        hist_bear_run = sections["regime_classification"].get("avg_bear_run_days", 0)

        s4c.update({
            "daily_bull_to_bear_pct": round(daily_bull_to_bear * 100, 2),
            "daily_bear_to_bull_pct": round(daily_bear_to_bull * 100, 2),
            "mean_block_length": mean_bl,
            "expected_bull_run_days": round(expected_bull_run, 1),
            "expected_bear_run_days": round(expected_bear_run, 1),
            "avg_sim_bull_run": round(avg_sim_bull, 1),
            "avg_sim_bear_run": round(avg_sim_bear, 1),
            "avg_hist_bull_run": hist_bull_run,
            "avg_hist_bear_run": hist_bear_run,
        })

        # Flag: compare simulated run lengths against block-adjusted expected
        if expected_bull_run > 0 and expected_bull_run != float('inf'):
            bull_ratio = avg_sim_bull / expected_bull_run
            if abs(bull_ratio - 1.0) > 0.3:
                s4c_flags.append(
                    f"Sim bull runs avg {avg_sim_bull:.0f}d vs expected "
                    f"{expected_bull_run:.0f}d — {abs(bull_ratio-1)*100:.0f}% deviation"
                )

        if expected_bear_run > 0 and expected_bear_run != float('inf'):
            bear_ratio = avg_sim_bear / expected_bear_run
            if abs(bear_ratio - 1.0) > 0.3:
                s4c_flags.append(
                    f"Sim bear runs avg {avg_sim_bear:.0f}d vs expected "
                    f"{expected_bear_run:.0f}d — {abs(bear_ratio-1)*100:.0f}% deviation"
                )

    s4c["flags"] = s4c_flags
    flags.extend(s4c_flags)
    sections["regime_transitions_sim"] = s4c
    section_pass["regime_transitions_sim"] = len(s4c_flags) == 0

    # ════════════════════════════════════════════════════════════════
    # Section 5: Scoring Breakdown
    # ════════════════════════════════════════════════════════════════
    t0 = time.time()
    horizon_str = f"{horizon_days}d"
    card = scorer.score(
        sim_result, oos_prices, oos_returns, horizon_str, train_prices,
        include_distributions=False,
    )

    s5 = {
        "composite_score": round(card.composite_score, 4),
        "metrics": {},
    }
    s5_flags = []

    weights = scorer.weights
    for name, score_val in card.scoring_metrics.items():
        w = weights.get(name, 0)
        raw = card.raw_metrics.get(name, {})
        s5["metrics"][name] = {
            "normalized": round(score_val, 4),
            "weight": w,
            "contribution": round(score_val * w, 4),
            "raw": raw,
        }
        if score_val < 0.2:
            s5_flags.append(f"Severe failure on {name}: score={score_val:.3f}")

    s5["diagnostics"] = card.diagnostic_metrics

    # Check MASE
    mase = card.diagnostic_metrics.get("mase")
    if mase and mase > 1.0:
        s5_flags.append(f"MASE={mase:.2f} — model is worse than naive baseline")

    # Check band coverage
    pbc = card.scoring_metrics.get("percentile_band_coverage", 1.0)
    if pbc < 0.5:
        pbc_raw = card.raw_metrics.get("percentile_band_coverage", {}).get("coverage_pct", 0)
        s5_flags.append(f"Percentile Band Coverage {pbc_raw:.0f}% — envelope is miscalibrated")

    s5["flags"] = s5_flags
    flags.extend(s5_flags)
    sections["scoring"] = s5
    section_pass["scoring"] = len(s5_flags) == 0
    timings["scoring"] = time.time() - t0

    # ════════════════════════════════════════════════════════════════
    # Section 5b: Overfitting Check
    # ════════════════════════════════════════════════════════════════
    t0 = time.time()
    s5b = {}
    s5b_flags = []

    # Score on training data (in-sample)
    n_train_oos = min(horizon_days + 1, len(train_prices))
    train_oos_prices = train_prices[-n_train_oos:]
    train_oos_returns = train_returns[-min(horizon_days, len(train_returns)):]
    in_sample_sim = model.simulate(
        n_simulations=min(500, n_simulations),
        n_steps=len(train_oos_returns),
        initial_price=float(train_prices[-n_train_oos]),
        seed=seed + 999,
    )
    in_sample_score = scorer.score_quick(in_sample_sim, train_oos_prices, train_oos_returns)

    s5b.update({
        "in_sample_score": round(in_sample_score, 4),
        "out_of_sample_score": round(card.composite_score, 4),
        "delta": round(in_sample_score - card.composite_score, 4),
    })

    if in_sample_score - card.composite_score > 0.15:
        s5b_flags.append(
            f"Possible overfitting: in-sample {in_sample_score:.3f} vs "
            f"out-of-sample {card.composite_score:.3f} (delta={in_sample_score - card.composite_score:.3f})"
        )

    s5b["flags"] = s5b_flags
    flags.extend(s5b_flags)
    sections["overfitting_check"] = s5b
    section_pass["overfitting_check"] = len(s5b_flags) == 0
    timings["overfitting_check"] = time.time() - t0

    # ════════════════════════════════════════════════════════════════
    # Section 5c: Baseline Comparison (GBM)
    # ════════════════════════════════════════════════════════════════
    t0 = time.time()
    s5c = {}
    s5c_flags = []

    if model_name != "gbm":
        gbm = get_model("gbm")
        gbm.fit(train_returns)
        gbm_sim = gbm.simulate(
            n_simulations=n_simulations,
            n_steps=len(oos_returns),
            initial_price=initial_price,
            seed=seed,
        )
        gbm_card = scorer.score(
            gbm_sim, oos_prices, oos_returns, horizon_str, train_prices,
            include_distributions=False,
        )

        per_metric_delta = {}
        for name in card.scoring_metrics:
            rbb_val = card.scoring_metrics.get(name, 0)
            gbm_val = gbm_card.scoring_metrics.get(name, 0)
            per_metric_delta[name] = {
                "rbb": round(rbb_val, 4),
                "gbm": round(gbm_val, 4),
                "delta": round(rbb_val - gbm_val, 4),
                "beats_gbm": rbb_val > gbm_val,
            }

        composite_delta = card.composite_score - gbm_card.composite_score
        s5c.update({
            "gbm_composite": round(gbm_card.composite_score, 4),
            "rbb_composite": round(card.composite_score, 4),
            "delta": round(composite_delta, 4),
            "delta_pct": round(composite_delta / max(gbm_card.composite_score, 0.001) * 100, 1),
            "per_metric": per_metric_delta,
        })

        if composite_delta < 0:
            s5c_flags.append(
                f"Block bootstrap underperforms GBM by {abs(composite_delta):.3f} on this window"
            )

    s5c["flags"] = s5c_flags
    flags.extend(s5c_flags)
    sections["baseline_comparison"] = s5c
    section_pass["baseline_comparison"] = len(s5c_flags) == 0
    timings["baseline_comparison"] = time.time() - t0

    # ════════════════════════════════════════════════════════════════
    # Section 6: Red Flag Summary
    # ════════════════════════════════════════════════════════════════
    total_checks = len(section_pass)
    passed = sum(1 for v in section_pass.values() if v)

    result = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "horizon": f"{horizon_days}d",
        "window": {
            "train_start": train_start_date,
            "train_end": train_end_date,
            "test_start": test_start_date,
            "test_end": test_end_date,
        },
        "sections": sections,
        "section_pass": section_pass,
        "red_flags": flags,
        "checks_passed": passed,
        "checks_total": total_checks,
        "timings": timings,
    }

    return result


def audit_to_markdown(audit: dict) -> str:
    """Convert audit dict to formatted markdown document."""
    lines = []
    audit_date = audit['timestamp'][:10]
    lines.append(f"# Pipeline Audit: {audit['model']} @ {audit['horizon']} ({audit_date})")
    lines.append(f"**Window:** {audit['window']['test_start']} to {audit['window']['test_end']}")
    lines.append(f"**Generated:** {audit['timestamp'][:19]}")
    lines.append(f"**Checks:** {audit['checks_passed']}/{audit['checks_total']} passed")
    lines.append("")

    if audit["red_flags"]:
        lines.append("## Red Flags")
        for flag in audit["red_flags"]:
            lines.append(f"- {flag}")
        lines.append("")

    s = audit["sections"]

    # Section 1
    d = s.get("data_selection", {})
    check = "PASS" if audit["section_pass"].get("data_selection") else "FAIL"
    lines.append(f"## 1. Data Selection [{check}]")
    lines.append(f"- Training: {d.get('train_start')} to {d.get('train_end')} ({d.get('train_days')} days, {d.get('train_years')}y)")
    lines.append(f"- Test: {d.get('test_start')} to {d.get('test_end')} ({d.get('test_days')} days)")
    ts = d.get("train_stats", {})
    lines.append(f"- Mean daily return: {ts.get('mean_daily_return',0):.6f}")
    lines.append(f"- Std daily return: {ts.get('std_daily_return',0):.6f}")
    lines.append(f"- Total return: {ts.get('total_return_pct',0):.1f}%")
    lines.append("")

    # Section 2
    d = s.get("regime_classification", {})
    check = "PASS" if audit["section_pass"].get("regime_classification") else "FAIL"
    lines.append(f"## 2. Regime Classification [{check}]")
    if d.get("transition_matrix"):
        tm = d["transition_matrix"]
        lines.append(f"- MSM Converged: {d.get('msm_converged')}")
        lines.append(f"- Bull days: {d.get('bull_days')} ({d.get('bull_pct')}%)")
        lines.append(f"- Bear days: {d.get('bear_days')} ({d.get('bear_pct')}%)")
        lines.append(f"- Avg bull run: {d.get('avg_bull_run_days')}d | Avg bear run: {d.get('avg_bear_run_days')}d")
        lines.append(f"- Transition matrix: P(Bull->Bear)={tm[0][1]:.4f}, P(Bear->Bull)={tm[1][0]:.4f}")
    lines.append("")

    # Section 3
    d = s.get("block_pools", {})
    check = "PASS" if audit["section_pass"].get("block_pools") else "FAIL"
    lines.append(f"## 3. Block Pools [{check}]")
    lines.append(f"- Sampling: {d.get('block_length_sampling')} | Length: {d.get('mean_block_length')}d | Stride: {d.get('block_stride')}d")
    lines.append(f"- Bull pool: {d.get('bull_pool_size')} blocks (median {d.get('bull_median_len')}d)")
    lines.append(f"- Bear pool: {d.get('bear_pool_size')} blocks (median {d.get('bear_median_len')}d)")
    lines.append("")

    # Section 4
    d = s.get("simulation_output", {})
    check = "PASS" if audit["section_pass"].get("simulation_output") else "FAIL"
    lines.append(f"## 4. Simulation Output [{check}]")
    lines.append(f"- Paths: {d.get('n_paths')} x {d.get('n_steps')} days")
    lines.append(f"- Median final: ${d.get('median_final_price',0):,.0f} | Actual: ${d.get('actual_final_price',0):,.0f}")
    lines.append(f"- 5th pct: ${d.get('p5_final_price',0):,.0f} | 95th pct: ${d.get('p95_final_price',0):,.0f}")
    lines.append(f"- Below start: {d.get('pct_below_start')}%")
    lines.append(f"- 50%+ DD: {d.get('pct_50_drawdown')}% | 75%+ DD: {d.get('pct_75_drawdown')}%")
    lines.append(f"- 90% CI coverage: {d.get('ci_coverage_pct')}%")
    # Tail events
    te = d.get("tail_events", {})
    if te:
        n_p = te.get("n_paths", 0)
        lines.append(f"\n### Tail Events ({n_p} paths)")
        gen = te.get("generic", {})
        for key, label in [("dd_50pct", "DD >= 50%"), ("dd_75pct", "DD >= 75%"),
                           ("dur_180d", "Duration >= 180d"), ("dur_365d", "Duration >= 365d")]:
            g = gen.get(key, {})
            lines.append(f"- {label}: {g.get('count', 0)}/{n_p} ({g.get('pct', 0)}%)")
        lines.append("")
        lines.append("**Named Stress Scenarios:**")
        for key, sc_data in te.get("named_scenarios", {}).items():
            flag = " *** MODEL CANNOT PRODUCE THIS ***" if sc_data["count"] == 0 else ""
            lines.append(
                f"- {key.replace('_', ' ').title()}: "
                f"{sc_data['count']}/{n_p} ({sc_data['pct']}%) — {sc_data['desc']}{flag}"
            )
    lines.append("")

    # Section 5
    d = s.get("scoring", {})
    check = "PASS" if audit["section_pass"].get("scoring") else "FAIL"
    lines.append(f"## 5. Scoring [{check}]")
    lines.append(f"**Composite: {d.get('composite_score', 0):.4f}**")
    lines.append("")
    lines.append("| Metric | Score | Weight | Contribution |")
    lines.append("|--------|-------|--------|-------------|")
    for name, m in d.get("metrics", {}).items():
        lines.append(f"| {name} | {m['normalized']:.4f} | {m['weight']:.2f} | {m['contribution']:.4f} |")
    lines.append("")

    # Section 5c
    d = s.get("baseline_comparison", {})
    if d.get("gbm_composite"):
        check = "PASS" if audit["section_pass"].get("baseline_comparison") else "FAIL"
        lines.append(f"## Baseline Comparison [{check}]")
        lines.append(f"- GBM: {d['gbm_composite']:.4f} | RBB: {d['rbb_composite']:.4f} | Delta: {d['delta']:+.4f} ({d['delta_pct']:+.1f}%)")
        lines.append("")

    return "\n".join(lines)


def audit_to_signal(audit: dict) -> str:
    """
    Condensed audit export — only decision-relevant signals.

    Designed to fit in an LLM context window. Strips all arrays, sample paths,
    usage counts, and raw metric internals. Keeps: scores, flags, key stats,
    and the verdict on each section.
    """
    s = audit["sections"]
    scoring = s.get("scoring", {})
    sim = s.get("simulation_output", {})
    bp = s.get("block_pools", {})
    regime = s.get("regime_classification", {})
    overfit = s.get("overfitting_check", {})
    baseline = s.get("baseline_comparison", {})
    convergence = s.get("simulation_convergence", {})
    transitions = s.get("regime_transitions_sim", {})

    lines = []
    lines.append(f"# Audit Signal: {audit['model']} @ {audit['horizon']} ({audit['timestamp'][:10]})")
    lines.append(f"Window: {audit['window']['test_start']} to {audit['window']['test_end']}")
    lines.append(f"Checks: {audit['checks_passed']}/{audit['checks_total']} passed")
    lines.append("")

    # Red flags — the most important part
    flags = audit.get("red_flags", [])
    if flags:
        lines.append(f"## RED FLAGS ({len(flags)})")
        for f in flags:
            lines.append(f"- {f}")
        lines.append("")
    else:
        lines.append("## No red flags")
        lines.append("")

    # Section verdicts — one line each
    lines.append("## Section Verdicts")
    for section, passed in audit.get("section_pass", {}).items():
        tag = "PASS" if passed else "FAIL"
        lines.append(f"- {section}: {tag}")
    lines.append("")

    # Composite + per-metric scores (no raw internals)
    lines.append(f"## Scoring (composite: {scoring.get('composite_score', 0):.4f})")
    lines.append("| Metric | Score | Weight | Contribution |")
    lines.append("|--------|-------|--------|-------------|")
    for name, m in scoring.get("metrics", {}).items():
        lines.append(f"| {name} | {m['normalized']:.4f} | {m['weight']:.2f} | {m['contribution']:.4f} |")
    lines.append("")

    # Baseline comparison
    if baseline.get("gbm_composite"):
        lines.append(
            f"## vs GBM: {baseline['rbb_composite']:.4f} vs {baseline['gbm_composite']:.4f} "
            f"(delta: {baseline['delta']:+.4f}, {baseline['delta_pct']:+.1f}%)"
        )
        lines.append("")

    # Key simulation stats (no arrays)
    lines.append("## Simulation")
    lines.append(f"- Paths: {sim.get('n_paths')} x {sim.get('n_steps')}d")
    lines.append(f"- Median final: ${sim.get('median_final_price', 0):,.0f} | Actual: ${sim.get('actual_final_price', 0):,.0f}")
    lines.append(f"- 90% CI coverage: {sim.get('ci_coverage_pct')}%")
    lines.append(f"- 50%+ DD: {sim.get('pct_50_drawdown')}% | 75%+ DD: {sim.get('pct_75_drawdown')}%")

    # Named scenario counts only
    te = sim.get("tail_events", {})
    named = te.get("named_scenarios", {})
    if named:
        lines.append("")
        lines.append("## Tail Scenarios")
        for key, sc in named.items():
            flag = " *** CANNOT PRODUCE ***" if sc["count"] == 0 else ""
            lines.append(f"- {key}: {sc['count']}/{te.get('n_paths', 0)} ({sc['pct']}%){flag}")
    lines.append("")

    # Block pools (counts only)
    lines.append("## Config")
    lines.append(f"- Sampling: {bp.get('block_length_sampling')} bl={bp.get('mean_block_length')}d stride={bp.get('block_stride')}d")
    lines.append(f"- Regime: {'ON' if regime.get('regime_enabled') else 'OFF'}")
    if bp.get("pool_pool_size") is not None:
        lines.append(f"- Pool: {bp.get('pool_pool_size')} blocks")
    elif bp.get("bull_pool_size") is not None:
        lines.append(f"- Bull pool: {bp.get('bull_pool_size')} | Bear pool: {bp.get('bear_pool_size', 0)}")
    lines.append("")

    # Overfitting check
    if overfit.get("in_sample_score") is not None:
        lines.append(
            f"## Overfitting: in={overfit['in_sample_score']:.4f} "
            f"out={overfit['out_of_sample_score']:.4f} "
            f"delta={overfit['delta']:+.4f}"
        )
        lines.append("")

    # Convergence
    if convergence.get("scores_by_path_count"):
        scores_str = " | ".join(
            f"{k}p={v:.4f}" for k, v in convergence["scores_by_path_count"].items()
        )
        lines.append(f"## Convergence: {scores_str}")
        lines.append("")

    # Regime transitions (key numbers only, no arrays)
    if transitions.get("avg_sim_bull_run"):
        lines.append(
            f"## Regime Runs: sim bull={transitions['avg_sim_bull_run']:.0f}d "
            f"bear={transitions['avg_sim_bear_run']:.0f}d | "
            f"expected bull={transitions['expected_bull_run_days']:.0f}d "
            f"bear={transitions['expected_bear_run_days']:.0f}d"
        )
        lines.append("")

    return "\n".join(lines)
