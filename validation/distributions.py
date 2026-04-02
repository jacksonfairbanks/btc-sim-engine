"""
Distribution data for visualization — captures KDE curves and percentile
summaries from simulated vs historical data.

These are NOT scoring metrics. They exist purely to feed the dashboard
with the raw distributional shapes needed for overlay charts.
"""
import numpy as np
from scipy import stats
from validation.metrics import ScoringMetrics


def _percentile_curve(values: np.ndarray, n_points: int = 99) -> dict:
    """
    Compute a percentile curve: x = percentile (1-99), y = value at that percentile.

    This is the inverse CDF / quantile function. Directly readable:
    "at the 95th percentile, drawdowns last X days."
    """
    if len(values) == 0:
        return {"x": [], "y": [], "n": 0}

    pcts = np.linspace(1, 99, n_points)
    vals = np.percentile(values, pcts)

    return {
        "x": pcts.tolist(),
        "y": vals.tolist(),
        "n": int(len(values)),
    }


def _percentile_summary(values: np.ndarray) -> dict:
    """Compute percentile summary of an array."""
    if len(values) == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "n": int(len(values)),
    }


def compute_distributions(
    sim_paths: np.ndarray,
    sim_returns: np.ndarray,
    hist_prices: np.ndarray,
    hist_returns: np.ndarray,
) -> dict:
    """
    Compute all distributional data for dashboard visualization.

    Parameters
    ----------
    sim_paths : np.ndarray
        Simulated price paths, shape (n_sims, n_steps+1).
    sim_returns : np.ndarray
        Simulated log returns, shape (n_sims, n_steps).
    hist_prices : np.ndarray
        Historical price series, 1D.
    hist_returns : np.ndarray
        Historical log returns, 1D.

    Returns
    -------
    dict
        Distribution data keyed by chart name.
    """
    result = {}

    # ── 1. Drawdown Duration Distribution ───────────────────────────
    hist_dds = ScoringMetrics._compute_drawdowns(hist_prices)
    hist_durations = np.array([d["duration"] for d in hist_dds])

    sim_durations = []
    sim_depths = []
    sim_recovery_times = []
    sim_tidd_ratios = []
    sim_max_dds = []

    for i in range(sim_paths.shape[0]):
        path = sim_paths[i]
        dds = ScoringMetrics._compute_drawdowns(path)
        sim_durations.extend([d["duration"] for d in dds])
        sim_depths.extend([d["depth"] for d in dds])
        sim_recovery_times.extend(
            [d["recovery_time"] for d in dds if d["recovered"]]
        )

        # Per-path time-in-drawdown ratio
        running_max = np.maximum.accumulate(path)
        ratio = np.mean(path < running_max * (1 - 1e-8))
        sim_tidd_ratios.append(ratio)

        # Per-path max drawdown
        dd_series = (path - running_max) / running_max
        sim_max_dds.append(float(np.min(dd_series)))

    sim_durations = np.array(sim_durations)
    sim_depths = np.array(sim_depths)
    sim_recovery_arr = np.array(sim_recovery_times)
    sim_tidd_ratios = np.array(sim_tidd_ratios)
    sim_max_dds = np.array(sim_max_dds)

    result["drawdown_duration"] = {
        "sim": _percentile_curve(sim_durations),
        "hist": _percentile_curve(hist_durations),
        "sim_summary": _percentile_summary(sim_durations),
        "hist_summary": _percentile_summary(hist_durations),
    }

    # ── 2. Recovery Time Distribution ───────────────────────────────
    hist_recovery = np.array(
        [d["recovery_time"] for d in hist_dds if d["recovered"]]
    )

    result["recovery_time"] = {
        "sim": _percentile_curve(sim_recovery_arr),
        "hist": _percentile_curve(hist_recovery),
        "sim_summary": _percentile_summary(sim_recovery_arr),
        "hist_summary": _percentile_summary(hist_recovery),
    }

    # ── 3. Time-in-Drawdown Ratio ───────────────────────────────────
    hist_running_max = np.maximum.accumulate(hist_prices)
    hist_tidd = float(np.mean(hist_prices < hist_running_max * (1 - 1e-8)))

    result["time_in_drawdown"] = {
        "sim": _percentile_curve(sim_tidd_ratios),
        "hist_value": hist_tidd,
        "sim_summary": _percentile_summary(sim_tidd_ratios),
    }

    # ── 4. Max Drawdown Depth Distribution ──────────────────────────
    # Historical: rolling window max drawdowns (90-day windows)
    window_size = min(90, len(hist_prices) // 3)
    hist_rolling_dds = []
    for start in range(0, len(hist_prices) - window_size + 1, window_size // 2):
        chunk = hist_prices[start:start + window_size]
        rm = np.maximum.accumulate(chunk)
        dd = (chunk - rm) / rm
        hist_rolling_dds.append(float(np.min(dd)))
    hist_rolling_dds = np.array(hist_rolling_dds)

    result["max_drawdown_depth"] = {
        "sim": _percentile_curve(sim_max_dds),
        "hist": _percentile_curve(hist_rolling_dds),
        "sim_summary": _percentile_summary(sim_max_dds),
        "hist_summary": _percentile_summary(hist_rolling_dds),
    }

    # ── 5. Log Return Density ───────────────────────────────────────
    sim_flat = sim_returns.flatten()

    result["log_returns"] = {
        "sim": _percentile_curve(sim_flat),
        "hist": _percentile_curve(hist_returns),
        "sim_summary": _percentile_summary(sim_flat),
        "hist_summary": _percentile_summary(hist_returns),
    }

    # ── 6. ACF Profile (lags 1-20) ─────────────────────────────────
    max_lag = 20
    hist_acf = ScoringMetrics._acf_squared(hist_returns, max_lag)

    sim_acfs = []
    n_paths = min(sim_returns.shape[0], 200)
    for i in range(n_paths):
        if sim_returns.shape[1] > max_lag + 1:
            sim_acfs.append(
                ScoringMetrics._acf_squared(sim_returns[i], max_lag)
            )

    if sim_acfs:
        sim_acf_mean = np.mean(sim_acfs, axis=0)
        sim_acf_p25 = np.percentile(sim_acfs, 25, axis=0)
        sim_acf_p75 = np.percentile(sim_acfs, 75, axis=0)
    else:
        sim_acf_mean = np.zeros(max_lag)
        sim_acf_p25 = np.zeros(max_lag)
        sim_acf_p75 = np.zeros(max_lag)

    result["acf_squared"] = {
        "lags": list(range(1, max_lag + 1)),
        "hist_acf": hist_acf.tolist(),
        "sim_acf_mean": sim_acf_mean.tolist(),
        "sim_acf_p25": sim_acf_p25.tolist(),
        "sim_acf_p75": sim_acf_p75.tolist(),
    }

    # ── 7. VaR Calibration ──────────────────────────────────────────
    var_data = {}
    for alpha in [0.01, 0.05]:
        label = f"{int(alpha*100)}pct"
        var_threshold = float(np.percentile(sim_flat, alpha * 100))
        n_breaches = int(np.sum(hist_returns < var_threshold))
        n_total = len(hist_returns)
        observed = n_breaches / n_total if n_total > 0 else 0
        var_data[label] = {
            "expected_rate": alpha,
            "observed_rate": float(observed),
            "var_threshold": var_threshold,
            "n_breaches": n_breaches,
            "n_total": n_total,
        }
    result["var_calibration"] = var_data

    # ── 8. Terminal Price Distribution ──────────────────────────────
    sim_terminal = sim_paths[:, -1]
    actual_terminal = float(hist_prices[-1]) if len(hist_prices) > 0 else None

    result["terminal_price"] = {
        "sim": _percentile_curve(sim_terminal),
        "actual_value": actual_terminal,
        "sim_summary": _percentile_summary(sim_terminal),
    }

    # ── 9. Tail Event Summary ───────────────────────────────────────
    n_sims = sim_paths.shape[0]
    n_steps_sim = sim_paths.shape[1]

    # Per-path max drawdown depth and duration
    dd_50 = 0
    dd_75 = 0
    dur_180 = 0
    dur_365 = 0
    crash_2014 = 0  # Mt. Gox: DD >= 85%, duration >= 390d
    crash_2018 = 0  # DD >= 84%, duration >= 365d
    crash_2022 = 0  # DD >= 77%, duration >= 390d (13 months)
    crash_2020 = 0  # DD >= 50% within first 30 days

    for i in range(n_sims):
        path = sim_paths[i]
        running_max = np.maximum.accumulate(path)
        dd_series = (path - running_max) / running_max
        max_dd = float(np.min(dd_series))

        if max_dd <= -0.50:
            dd_50 += 1
        if max_dd <= -0.75:
            dd_75 += 1

        # Drawdown durations from this path
        in_dd = False
        dd_start = 0
        max_dur = 0
        for j in range(len(dd_series)):
            if dd_series[j] < -0.01:
                if not in_dd:
                    in_dd = True
                    dd_start = j
            else:
                if in_dd:
                    dur = j - dd_start
                    if dur > max_dur:
                        max_dur = dur
                    in_dd = False
        if in_dd:
            dur = len(dd_series) - dd_start
            if dur > max_dur:
                max_dur = dur

        if max_dur >= 180:
            dur_180 += 1
        if max_dur >= 365:
            dur_365 += 1

        # Named scenarios
        # 2014 Mt. Gox: DD >= 85%, duration >= 390d (13 months)
        if max_dd <= -0.85 and max_dur >= 390:
            crash_2014 += 1
        # 2018-type: DD >= 84%, duration >= 365d
        if max_dd <= -0.84 and max_dur >= 365:
            crash_2018 += 1
        # 2022-type: DD >= 77%, duration >= 390d
        if max_dd <= -0.77 and max_dur >= 390:
            crash_2022 += 1
        # March 2020-type: DD >= 50% within first 30 days
        if n_steps_sim > 30:
            early_dd = float(np.min(dd_series[:31]))
            if early_dd <= -0.50:
                crash_2020 += 1

    result["tail_events"] = {
        "n_paths": n_sims,
        "generic": {
            "dd_50pct": {"count": dd_50, "pct": round(dd_50 / n_sims * 100, 1)},
            "dd_75pct": {"count": dd_75, "pct": round(dd_75 / n_sims * 100, 1)},
            "dur_180d": {"count": dur_180, "pct": round(dur_180 / n_sims * 100, 1)},
            "dur_365d": {"count": dur_365, "pct": round(dur_365 / n_sims * 100, 1)},
        },
        "named_scenarios": {
            "2014_mt_gox": {
                "desc": "Drawdown >= 85%, duration >= 13 months (Mt. Gox collapse)",
                "count": crash_2014,
                "pct": round(crash_2014 / n_sims * 100, 2),
            },
            "2018_crash": {
                "desc": "Drawdown >= 84%, duration >= 12 months",
                "count": crash_2018,
                "pct": round(crash_2018 / n_sims * 100, 2),
            },
            "2022_crash": {
                "desc": "Drawdown >= 77%, duration >= 13 months (Luna/FTX)",
                "count": crash_2022,
                "pct": round(crash_2022 / n_sims * 100, 2),
            },
            "2020_flash_crash": {
                "desc": "Drawdown >= 50% within 30 days (COVID)",
                "count": crash_2020,
                "pct": round(crash_2020 / n_sims * 100, 2),
            },
        },
    }

    return result
