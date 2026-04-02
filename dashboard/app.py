"""
BTC Sim Engine — Dashboard

Streamlit app with Bloomberg-terminal aesthetic.

Tabs:
1. Overview — Simplified front page: fan chart, params, radar, metric scorecards
2. Walk-Forward Inspector — per-window scores, dates, weights, sim vs actual
3. Parameter Optimization — per-trial search landscape
4. Simulation Sandbox — manual parameter input, run on demand
"""
import os
import sys
import json
import subprocess
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Sim Engine",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bloomberg-terminal theme ────────────────────────────────────────────
BG = "#0a0a0a"
PANEL = "#111111"
GRID = "#1a1a1a"
TEXT = "#e0e0e0"
TEXT_DIM = "#666666"
BTC_ORANGE = "#f7931a"
BLUE = "#3b82f6"
GREEN = "#10b981"
RED = "#ef4444"
PURPLE = "#8b5cf6"
YELLOW = "#eab308"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(family="JetBrains Mono, IBM Plex Mono, Consolas, monospace",
              color="#cccccc", size=12),
    title_font=dict(color="#ffffff", size=14),
    legend=dict(font=dict(color="#cccccc", size=11)),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID,
               tickfont=dict(color="#bbbbbb"), title=dict(font=dict(color="#cccccc"))),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID,
               tickfont=dict(color="#bbbbbb"), title=dict(font=dict(color="#cccccc"))),
    margin=dict(l=60, r=30, t=50, b=50),
    colorway=[BTC_ORANGE, BLUE, GREEN, RED, PURPLE, YELLOW],
)

CATEGORY_COLORS = {
    "Distributional": BLUE,
    "Tail Risk": RED,
    "Path Dynamics": BTC_ORANGE,
    "Temporal Dependence": PURPLE,
    "Forecast Accuracy": GREEN,
}

# Metric metadata — ordered by category as specified
METRIC_INFO = [
    # ── Distributional ──
    {"key": "ks_statistic", "name": "KS Statistic", "category": "Distributional",
     "weight": 0.07,
     "desc": "Kolmogorov-Smirnov test comparing simulated vs historical return distributions. Catches gross distributional mismatch."},
    {"key": "qq_divergence", "name": "QQ Divergence", "category": "Distributional",
     "weight": 0.08,
     "desc": "Mean squared error between quantiles on a QQ plot. Reveals where distributions diverge at the tails, center, and shoulders."},
    # ── Tail Risk ──
    {"key": "tail_index_accuracy", "name": "Tail Index Accuracy", "category": "Tail Risk",
     "weight": 0.13,
     "desc": "Compares Hill estimator tail indices on the crash tail. Controls how likely extreme crashes are in the simulation."},
    {"key": "percentile_band_coverage", "name": "Percentile Band Coverage", "category": "Tail Risk",
     "weight": 0.12,
     "desc": "Percentage of realized daily prices within the simulated 5th-95th percentile envelope. Directly measures whether the simulation brackets reality."},
    # ── Path Dynamics ──
    {"key": "drawdown_duration_dist", "name": "Drawdown Duration", "category": "Path Dynamics",
     "weight": 0.13,
     "desc": "Compares how long drawdowns last (peak-to-recovery). Critical for projecting multi-month credit exposure windows."},
    {"key": "recovery_time_dist", "name": "Recovery Time", "category": "Path Dynamics",
     "weight": 0.12,
     "desc": "Compares time from trough back to prior peak. Determines how long credit instruments stay under stress after a crash."},
    {"key": "time_in_drawdown_ratio", "name": "Time in Drawdown", "category": "Path Dynamics",
     "weight": 0.10,
     "desc": "Percentage of total time below running maximum. Captures overall path shape — unrealistically smooth vs realistically choppy."},
    # ── Temporal Dependence ──
    {"key": "vol_clustering_acf", "name": "Volatility Clustering", "category": "Temporal Dependence",
     "weight": 0.10,
     "desc": "ACF of squared returns. Big moves follow big moves, calm follows calm. Without this, volatility is too evenly spread."},
    # ── Forecast Accuracy ──
    {"key": "mape", "name": "MAPE", "category": "Forecast Accuracy",
     "weight": 0.15,
     "desc": "Mean Absolute Percentage Error of median simulated path vs realized. The most intuitive accuracy metric for stakeholders."},
]

# Raw metric display config: which raw fields to show and how to format them
RAW_METRIC_DISPLAY = {
    "ks_statistic": [
        ("ks_stat", "KS Statistic", ".4f", "0 = identical distributions"),
        ("p_value", "p-value", ".4f", ">0.05 = fail to reject H0"),
    ],
    "qq_divergence": [
        ("mse", "QQ MSE", ".6f", "Raw MSE between quantiles"),
        ("normalized_mse", "Normalized MSE", ".4f", "MSE / hist quantile variance"),
    ],
    "tail_index_accuracy": [
        ("hist_hill_index", "Hist Hill Index", ".3f", "Historical crash tail index"),
        ("sim_hill_index", "Sim Hill Index", ".3f", "Simulated crash tail index"),
        ("relative_error", "Relative Error", ".4f", "|sim - hist| / |hist|"),
    ],
    "percentile_band_coverage": [
        ("coverage_pct", "Coverage %", ".1f", "% of realized prices in 5th-95th envelope"),
        ("target_pct", "Target %", ".0f", "Expected from 90% CI"),
        ("n_inside", "Days Inside", "d", ""),
        ("n_total", "Total Days", "d", ""),
        ("n_below_p5", "Below P5", "d", "Days below 5th percentile"),
        ("n_above_p95", "Above P95", "d", "Days above 95th percentile"),
    ],
    "drawdown_duration_dist": [
        ("hist_median_duration_days", "Hist Median Duration", ".0f", "days"),
        ("sim_median_duration_days", "Sim Median Duration", ".0f", "days"),
        ("hist_max_duration_days", "Hist Max Duration", ".0f", "days"),
        ("sim_max_duration_days", "Sim Max Duration", ".0f", "days"),
        ("hist_n_drawdowns", "Hist # Drawdowns", "d", ""),
        ("sim_n_drawdowns", "Sim # Drawdowns", "d", ""),
        ("ks_stat", "KS Statistic", ".4f", "0 = identical distributions"),
    ],
    "recovery_time_dist": [
        ("hist_median_recovery_days", "Hist Median Recovery", ".0f", "days"),
        ("sim_median_recovery_days", "Sim Median Recovery", ".0f", "days"),
        ("hist_n_recoveries", "Hist # Recoveries", "d", ""),
        ("sim_n_recoveries", "Sim # Recoveries", "d", ""),
        ("ks_stat", "KS Statistic", ".4f", "0 = identical distributions"),
    ],
    "time_in_drawdown_ratio": [
        ("hist_ratio", "Hist Ratio", ".4f", "Fraction of time underwater"),
        ("sim_median_ratio", "Sim Median Ratio", ".4f", "Median across sim paths"),
        ("relative_error", "Relative Error", ".4f", "|sim - hist| / hist"),
    ],
    "vol_clustering_acf": [
        ("hist_acf_lag1", "Hist ACF(1) of r^2", ".4f", "Lag-1 autocorrelation"),
        ("sim_acf_lag1", "Sim ACF(1) of r^2", ".4f", "Lag-1 autocorrelation"),
        ("hist_acf_lag5", "Hist ACF(5) of r^2", ".4f", "Lag-5 autocorrelation"),
        ("sim_acf_lag5", "Sim ACF(5) of r^2", ".4f", "Lag-5 autocorrelation"),
        ("acf_rmse", "ACF RMSE", ".4f", "RMSE across lag profile"),
    ],
    "mape": [
        ("mape_pct", "MAPE %", ".2f", "Mean absolute percentage error"),
        ("median_final_price", "Median Final Price", ",.0f", "USD"),
        ("actual_final_price", "Actual Final Price", ",.0f", "USD"),
    ],
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
    .stApp { background-color: #0a0a0a; font-family: 'JetBrains Mono', monospace; }
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #1a1a1a; }
    h1, h2, h3, h4 { color: #f7931a !important; font-family: 'JetBrains Mono', monospace !important; }
    [data-testid="stMetric"] { background-color: #111111; border: 1px solid #1a1a1a; border-radius: 4px; padding: 12px; }
    [data-testid="stMetricLabel"] { color: #666666 !important; }
    [data-testid="stMetricValue"] { color: #f7931a !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #111111; border-bottom: 1px solid #1a1a1a; }
    .stTabs [data-baseweb="tab"] { color: #666666; background-color: transparent; border: 1px solid #1a1a1a; border-radius: 4px 4px 0 0; }
    .stTabs [aria-selected="true"] { color: #f7931a !important; border-bottom: 2px solid #f7931a !important; background-color: #111111 !important; }
    .stMarkdown, .stText, p, span, label { color: #e0e0e0; }
    .metric-card { background: #111111; border: 1px solid #1a1a1a; border-radius: 6px; padding: 16px; margin-bottom: 12px; }
    .metric-card h4 { margin: 0 0 4px 0; font-size: 0.95rem; }
    .metric-card .weight { color: #666666; font-size: 0.75rem; }
    .metric-card .desc { color: #888888; font-size: 0.75rem; margin-top: 6px; line-height: 1.4; }
    .metric-card .score-val { font-size: 1.4rem; font-weight: 700; }
    .score-bar-bg { background: #1a1a1a; border-radius: 3px; height: 8px; width: 100%; margin: 8px 0; }
    .score-bar-fill { border-radius: 3px; height: 8px; }
    .category-header { color: #666666; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; margin: 20px 0 10px 0; padding-bottom: 6px; border-bottom: 1px solid #1a1a1a; }
    .footer { color: #666666; font-size: 0.7rem; text-align: right; padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ────────────────────────────────────────────────────────
@st.cache_data
def load_experiment_data(path: str, _mtime: float = 0.0) -> dict | None:
    """Load experiment JSON. _mtime param busts cache when file changes on disk."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def find_experiment_files(results_dir: str = "results") -> list[str]:
    p = Path(results_dir)
    return [str(f) for f in p.glob("**/experiment_data.json")]


def make_fan_chart(sim_pct: dict, realized: list, title: str = "",
                   start_date: str | None = None) -> go.Figure:
    """Reusable fan chart builder. If start_date given, x-axis shows dates."""
    n = len(realized)
    if start_date:
        try:
            start = pd.Timestamp(start_date)
            x = pd.date_range(start=start, periods=n, freq="D").tolist()
        except Exception:
            x = list(range(n))
    else:
        x = list(range(n))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=sim_pct["p5"][:n] + sim_pct["p95"][:n][::-1],
        fill="toself", fillcolor="rgba(247,147,26,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="90% CI",
    ))
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=sim_pct["p25"][:n] + sim_pct["p75"][:n][::-1],
        fill="toself", fillcolor="rgba(247,147,26,0.20)",
        line=dict(color="rgba(0,0,0,0)"), name="50% CI",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=sim_pct["p50"][:n], mode="lines",
        line=dict(color=BTC_ORANGE, width=2), name="Median Sim",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=realized, mode="lines",
        line=dict(color=TEXT, width=2, dash="dash"), name="Realized",
    ))
    xaxis_title = "" if start_date else "Days"
    fig.update_layout(title=title, xaxis_title=xaxis_title,
                      yaxis_title="Price (USD)", yaxis_type="log",
                      **PLOTLY_LAYOUT)
    return fig


def score_color(val: float) -> str:
    """Return hex color for a 0-1 score."""
    if val >= 0.8:
        return GREEN
    elif val >= 0.6:
        return BTC_ORANGE
    elif val >= 0.4:
        return YELLOW
    else:
        return RED


def render_metric_card(info: dict, score: float) -> str:
    """Render a single metric scorecard as HTML."""
    color = score_color(score)
    pct = max(0, min(100, score * 100))
    return f"""
    <div class="metric-card">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <h4 style="color:{color};">{info['name']}</h4>
            <span class="score-val" style="color:{color};">{score:.3f}</span>
        </div>
        <span class="weight">Weight: {info['weight']*100:.0f}% | Weighted contribution: {score*info['weight']:.4f}</span>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
        <div class="desc">{info['desc']}</div>
    </div>
    """


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# BTC Sim Engine")
    st.markdown(f"<span style='color:{TEXT_DIM}'>Price Path Simulation</span>", unsafe_allow_html=True)
    st.divider()

    # Run Experiment
    st.markdown("### Run Experiment")
    run_mode = st.radio("Mode", ["quick", "standard", "full"], horizontal=True, index=0)

    if st.button("Run Experiment", type="primary", use_container_width=True):
        with st.spinner(f"Running {run_mode} experiment..."):
            result = subprocess.run(
                [sys.executable, "scripts/run_experiment.py",
                 "--config", "config/default.yaml",
                 "--mode", run_mode, "--no-charts"],
                capture_output=True, text=True, cwd=PROJECT_ROOT,
                env={**__import__('os').environ, "PYTHONPATH": PROJECT_ROOT},
            )
            if result.returncode == 0:
                st.success("Experiment complete!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Experiment failed")
                with st.expander("Error output"):
                    st.code(result.stderr[-2000:] if result.stderr else result.stdout[-2000:])

    st.divider()

    files = find_experiment_files()
    if not files:
        st.warning("No experiment data found. Click **Run Experiment** above.")
        st.stop()

    selected_file = st.selectbox("Experiment Data", files, index=0)
    _file_mtime = os.path.getmtime(selected_file) if os.path.exists(selected_file) else 0.0
    data = load_experiment_data(selected_file, _mtime=_file_mtime)

    if data is None:
        st.error("Failed to load data.")
        st.stop()

    all_runs = data.get("runs", [])

    st.markdown(f"**Run:** `{data.get('run_mode', '?')}` | {data.get('timestamp', '?')[:16]}")

    data_info = data.get("data_info", {})
    if data_info.get("train_start"):
        st.markdown(f"**Train:** {data_info['train_start']} to {data_info['train_end']}")
        st.markdown(f"**Test:** {data_info['test_start']} to {data_info['test_end']}")

    st.divider()

    runs = data.get("runs", [])
    model_names = sorted(set(r["model"] for r in runs))
    horizons = sorted(set(r["horizon"] for r in runs), key=lambda h: int(h.replace("d", "")))

    selected_model = st.selectbox("Model", model_names)
    selected_horizon = st.selectbox("Horizon", horizons)

    st.divider()
    st.markdown(f"<div class='footer'>@LongGamma</div>", unsafe_allow_html=True)


# ── Find selected run ──────────────────────────────────────────────────
def get_run(model: str, horizon: str) -> dict | None:
    for r in runs:
        if r["model"] == model and r["horizon"] == horizon:
            return r
    return None

selected_run = get_run(selected_model, selected_horizon)
if selected_run is None:
    st.error(f"No data for {selected_model} @ {selected_horizon}")
    st.stop()


# ── Tabs ────────────────────────────────────────────────────────────────
tab_exec, tab_lb, tab_overview, tab_wf, tab_opt, tab_audit, tab_export, tab_prod, tab_bcr, tab_docs = st.tabs([
    "Executive Summary",
    "Leaderboard",
    "Overview",
    "Walk-Forward Inspector",
    "Parameter Optimization",
    "Pipeline Audit",
    "Export Results",
    "Production Simulation",
    "BCR Stress Test",
    "Phase Documentation",
])


HORIZONS = ["180d", "365d", "730d", "1460d"]
HZ_WEIGHTS = {"180d": 0.10, "365d": 0.20, "730d": 0.35, "1460d": 0.35}

PROD_HORIZON = 1460
PROD_N_SIMS = 5000
PROD_RBB_PARAMS = {
    "block_length_sampling": "geometric",
    "mean_block_length": 30,
    "min_block_length": 5,
    "block_stride": 5,
    "min_pool_size": 8,
    "regime_enabled": False,
}


@st.cache_resource
def load_prod_deps():
    from models.registry import get_model
    from data.loader import BTCDataLoader, get_price_array
    return get_model, BTCDataLoader, get_price_array


def run_prod_simulation():
    """Run production RBB simulation and store in session state."""
    _get_model, _Loader, _get_px = load_prod_deps()
    loader = _Loader()
    full_df = loader.load_processed_data()
    all_returns = full_df["log_return"].values
    all_prices = full_df["Close"].values
    initial_price = float(all_prices[-1])
    data_end_date = str(full_df.index[-1].date())
    n_training_days = len(all_returns)

    model = _get_model("regime_block_bootstrap")
    model.set_params(**PROD_RBB_PARAMS)
    model.fit(all_returns)
    sim = model.simulate(
        n_simulations=PROD_N_SIMS, n_steps=PROD_HORIZON,
        initial_price=initial_price, seed=42,
    )

    # Tail events
    n_sims = sim.paths.shape[0]
    dd_50 = dd_75 = dur_180 = dur_365 = 0
    crash_2014 = crash_2018 = crash_2022 = crash_2020 = 0
    for i in range(n_sims):
        path = sim.paths[i]
        rm = np.maximum.accumulate(path)
        dd_s = (path - rm) / rm
        max_dd = float(np.min(dd_s))
        if max_dd <= -0.50: dd_50 += 1
        if max_dd <= -0.75: dd_75 += 1
        in_dd = False; dd_start = 0; max_dur = 0
        for j in range(len(dd_s)):
            if dd_s[j] < -0.01:
                if not in_dd: in_dd = True; dd_start = j
            else:
                if in_dd:
                    d = j - dd_start
                    if d > max_dur: max_dur = d
                    in_dd = False
        if in_dd:
            d = len(dd_s) - dd_start
            if d > max_dur: max_dur = d
        if max_dur >= 180: dur_180 += 1
        if max_dur >= 365: dur_365 += 1
        if max_dd <= -0.85 and max_dur >= 390: crash_2014 += 1
        if max_dd <= -0.84 and max_dur >= 365: crash_2018 += 1
        if max_dd <= -0.77 and max_dur >= 390: crash_2022 += 1
        if len(dd_s) > 30 and float(np.min(dd_s[:31])) <= -0.50: crash_2020 += 1

    st.session_state["prod_sim"] = {
        "initial_price": initial_price,
        "data_end_date": data_end_date,
        "n_training_days": n_training_days,
        "p5": np.percentile(sim.paths, 5, axis=0).tolist(),
        "p25": np.percentile(sim.paths, 25, axis=0).tolist(),
        "p50": np.median(sim.paths, axis=0).tolist(),
        "p75": np.percentile(sim.paths, 75, axis=0).tolist(),
        "p95": np.percentile(sim.paths, 95, axis=0).tolist(),
        "final_prices": sim.paths[:, -1].tolist(),
        "pool_size": len(model._block_pools[0]),
        "tail_events": {
            "n_paths": n_sims,
            "generic": {
                "dd_50pct": {"count": dd_50, "pct": round(dd_50/n_sims*100, 1)},
                "dd_75pct": {"count": dd_75, "pct": round(dd_75/n_sims*100, 1)},
                "dur_180d": {"count": dur_180, "pct": round(dur_180/n_sims*100, 1)},
                "dur_365d": {"count": dur_365, "pct": round(dur_365/n_sims*100, 1)},
            },
            "named_scenarios": {
                "2014_mt_gox": {"desc": "DD >= 85%, duration >= 13mo", "count": crash_2014, "pct": round(crash_2014/n_sims*100, 2)},
                "2018_crash": {"desc": "DD >= 84%, duration >= 12mo", "count": crash_2018, "pct": round(crash_2018/n_sims*100, 2)},
                "2022_crash": {"desc": "DD >= 77%, duration >= 13mo", "count": crash_2022, "pct": round(crash_2022/n_sims*100, 2)},
                "2020_flash_crash": {"desc": "DD >= 50% within 30d", "count": crash_2020, "pct": round(crash_2020/n_sims*100, 2)},
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# TAB: Executive Summary
# ═══════════════════════════════════════════════════════════════════════
with tab_exec:
    st.markdown(
        f"<h1 style='color:{BTC_ORANGE};margin-bottom:0;'>BTC Price Path Simulation Engine</h1>"
        f"<span style='color:{TEXT_DIM};font-size:0.9rem;'>Scoring-driven model comparison for "
        f"Bitcoin credit stress-testing &mdash; @LongGamma</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Three-Model Leaderboard ────────────────────────────────────
    st.subheader("Model Leaderboard")
    _exec_models = [
        {"Model": "RBB (Block Bootstrap)", "Final Score": 0.8109, "vs GBM": "+6.3%",
         "desc": "**Production model.** Resamples actual historical BTC return sequences as contiguous blocks. "
                 "Nonparametric — no assumptions about return distribution. Preserves fat tails, "
                 "volatility clustering, and multi-day drawdown patterns by construction. Scored highest "
                 "at every horizon. The block bootstrap's structural advantage: it draws from what actually "
                 "happened, not from a parametric approximation of what happened."},
        {"Model": "GARCH(1,1) t-dist", "Final Score": 0.7800, "vs GBM": "+2.2%",
         "desc": "**Volatility clustering benchmark.** Models conditional variance as a time-varying process "
                 "with Student's t innovations. Captures the well-documented 'big moves follow big moves' "
                 "pattern in BTC returns. Falls short on tail risk (-0.12 on tail index accuracy) because "
                 "the t-distribution produces fat tails but not the *right shape* of tails. Loses to GBM "
                 "at 1460d due to IGARCH variance drift over 4-year horizons."},
        {"Model": "GBM (Log-Normal)", "Final Score": 0.7630, "vs GBM": "baseline",
         "desc": "**Baseline control.** Geometric Brownian Motion — constant drift, constant volatility, "
                 "normal innovations. The simplest possible price model. No volatility clustering, no fat "
                 "tails, no regime dynamics. Exists to answer: 'is your model adding value over a random walk?' "
                 "Both RBB and GARCH beat it, confirming that BTC's non-normal return structure matters."},
    ]

    for m in _exec_models:
        _color = BTC_ORANGE if "RBB" in m["Model"] else BLUE if "GARCH" in m["Model"] else TEXT_DIM
        st.markdown(
            f"<div style='background:{PANEL};border-left:4px solid {_color};"
            f"border-radius:4px;padding:12px;margin-bottom:12px;'>"
            f"<span style='color:{_color};font-weight:700;font-size:1.1rem;'>"
            f"{m['Model']}</span>"
            f"<span style='color:{TEXT};float:right;font-size:1.3rem;font-weight:700;'>"
            f"{m['Final Score']:.4f}</span><br>"
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>{m['desc']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Per-Horizon Bar Chart ──────────────────────────────────────
    st.divider()
    st.subheader("Walk-Forward Composite by Horizon")
    _exec_hz_data = {
        "RBB":   [0.8098, 0.8221, 0.8265, 0.7928],
        "GARCH": [0.7973, 0.8024, 0.7976, 0.7506],
        "GBM":   [0.7589, 0.7705, 0.7618, 0.7625],
    }
    fig_hz = go.Figure()
    for model_name, vals, color in [
        ("RBB", _exec_hz_data["RBB"], BTC_ORANGE),
        ("GARCH(1,1)", _exec_hz_data["GARCH"], BLUE),
        ("GBM", _exec_hz_data["GBM"], TEXT_DIM),
    ]:
        fig_hz.add_trace(go.Bar(
            name=model_name, x=HORIZONS, y=vals,
            marker_color=color, opacity=0.85,
            text=[f"{v:.3f}" for v in vals], textposition="auto",
            textfont=dict(size=11),
        ))
    fig_hz.update_layout(
        barmode="group", yaxis_range=[0.7, 0.86],
        yaxis_title="Walk-Forward Composite",
        xaxis_title="Horizon",
        height=400, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_hz, use_container_width=True)
    st.caption(
        "Horizon weights: 180d=10%, 365d=20%, 730d=35%, 1460d=35%. "
        "RBB's advantage widens at longer horizons where multi-day path structure matters most. "
        "GARCH loses to GBM at 1460d due to IGARCH variance drift."
    )

    # ── Methodology ────────────────────────────────────────────────
    # ── Hero Number ──────────────────────────────────────────────
    st.divider()
    st.markdown(
        f"<div style='text-align:center;padding:30px 0 10px 0;'>"
        f"<span style='color:{BTC_ORANGE};font-size:3.5rem;font-weight:800;letter-spacing:-2px;'>"
        f"630,000</span><br>"
        f"<span style='color:{TEXT};font-size:1.1rem;'>simulated price paths, each scored against reality</span><br>"
        f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
        f"126 independent out-of-sample windows &times; 5,000 paths per window &times; 4 horizons &times; 3 models</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    _bn_style = f"text-align:center;background:{PANEL};border-radius:6px;padding:12px 6px;"
    bn1, bn2, bn3, bn4 = st.columns(4)
    for col, number, label in [
        (bn1, "126", "out-of-sample windows"),
        (bn2, "4,209", "days of BTC history"),
        (bn3, "9", "scoring dimensions"),
        (bn4, "0", "lookahead bias"),
    ]:
        with col:
            st.markdown(
                f"<div style='{_bn_style}'>"
                f"<span style='color:{BTC_ORANGE};font-size:1.5rem;font-weight:800;'>{number}</span><br>"
                f"<span style='color:{TEXT_DIM};font-size:0.7rem;'>{label}</span></div>",
                unsafe_allow_html=True,
            )

    # ── Step 1: "Here's what happens once" ─────────────────────────
    st.divider()
    st.subheader("How We Validated: A Single Window Walk-Through")
    st.markdown(
        f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
        f"Pick one window. Train the model on everything before it. "
        f"Simulate 5,000 paths forward. Score against what actually happened. "
        f"Here's Window 25 at the 730-day horizon.</span>",
        unsafe_allow_html=True,
    )

    # Find a good example window — 730d, a recent-ish one with good data
    _example_run = None
    for r in all_runs:
        if r["model"] == "regime_block_bootstrap" and r["horizon"] == "730d":
            _example_run = r
            break

    if _example_run:
        _ex_windows = _example_run["walk_forward"].get("windows", [])
        # Pick window ~25 (2022-2024 range — covers the recovery)
        _ex_idx = min(25, len(_ex_windows) - 1)
        _ex_w = _ex_windows[_ex_idx]

        # Fan chart for this window
        _ex_pct = _ex_w.get("sim_percentiles")
        _ex_real = _ex_w.get("realized_prices")
        _ex_scores = _ex_w.get("scoring_metrics", {})
        _ex_composite = _ex_w.get("composite_score", 0)

        st.markdown(
            f"<div style='background:{PANEL};border-radius:6px;padding:10px;margin-bottom:8px;'>"
            f"<span style='color:{BTC_ORANGE};font-weight:700;'>Window {_ex_w['window_num']}</span>"
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
            f" &mdash; Train: start to {_ex_w.get('train_end_date', '?')}"
            f" ({_ex_w.get('train_days', '?')} days)"
            f" &mdash; Test: {_ex_w.get('test_start_date', '?')} to {_ex_w.get('test_end_date', '?')}"
            f" (730 days)"
            f" &mdash; Composite: <b>{_ex_composite:.4f}</b></span></div>",
            unsafe_allow_html=True,
        )

        _ex_col1, _ex_col2 = st.columns([2, 1])

        with _ex_col1:
            if _ex_pct and _ex_real:
                fig_ex = make_fan_chart(
                    _ex_pct, _ex_real,
                    title=f"5,000 simulated paths vs what actually happened",
                    start_date=_ex_w.get("test_start_date"),
                )
                fig_ex.update_layout(height=400)
                st.plotly_chart(fig_ex, use_container_width=True)

        with _ex_col2:
            # Score breakdown for this window
            st.markdown(f"<span style='color:{TEXT};font-weight:700;'>9-Metric Scores</span>",
                        unsafe_allow_html=True)
            _metric_cat_colors = {
                "drawdown_duration_dist": BTC_ORANGE, "recovery_time_dist": BTC_ORANGE,
                "time_in_drawdown_ratio": BTC_ORANGE,
                "tail_index_accuracy": RED, "percentile_band_coverage": RED,
                "ks_statistic": BLUE, "qq_divergence": BLUE,
                "vol_clustering_acf": PURPLE, "mape": GREEN,
            }
            _metric_short = {
                "drawdown_duration_dist": "DD Duration", "recovery_time_dist": "Recovery",
                "time_in_drawdown_ratio": "Time in DD",
                "tail_index_accuracy": "Tail Index", "percentile_band_coverage": "Band Cov",
                "ks_statistic": "KS Stat", "qq_divergence": "QQ Div",
                "vol_clustering_acf": "Vol Cluster", "mape": "MAPE",
            }
            for mk in ["drawdown_duration_dist", "recovery_time_dist", "time_in_drawdown_ratio",
                        "tail_index_accuracy", "percentile_band_coverage",
                        "ks_statistic", "qq_divergence", "vol_clustering_acf", "mape"]:
                v = _ex_scores.get(mk, 0)
                c = _metric_cat_colors.get(mk, TEXT_DIM)
                bar_w = max(0, min(100, v * 100))
                st.markdown(
                    f"<div style='margin:2px 0;'>"
                    f"<span style='color:{c};font-size:0.75rem;'>{_metric_short.get(mk, mk)}</span>"
                    f"<span style='color:{TEXT};font-size:0.75rem;float:right;'>{v:.3f}</span>"
                    f"<div style='background:{GRID};border-radius:2px;height:4px;margin-top:2px;'>"
                    f"<div style='background:{c};width:{bar_w}%;height:4px;border-radius:2px;'></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    # ── Step 2: "Here's how many times we did it" ──────────────────
    st.divider()
    st.subheader("This Process Repeated 126 Times")
    st.markdown(
        f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
        f"Every 90 days, the training window expands and a new out-of-sample test runs. "
        f"No window ever sees data from its own test period. Across 4 horizons, that's 126 "
        f"independent validations.</span>",
        unsafe_allow_html=True,
    )

    # Build actual timeline from experiment data — all horizons
    _hz_colors = {"180d": BLUE, "365d": GREEN, "730d": BTC_ORANGE, "1460d": RED}
    fig_all_windows = go.Figure()

    for hz_str in HORIZONS:
        for r in all_runs:
            if r["model"] == "regime_block_bootstrap" and r["horizon"] == hz_str:
                windows = r["walk_forward"].get("windows", [])
                n_win = len(windows)
                train_days = [w.get("train_days", 730) for w in windows]
                hz_days = int(hz_str.replace("d", ""))

                # Show all windows as thin horizontal bars
                for j, w in enumerate(windows):
                    td = w.get("train_days", 730)
                    fig_all_windows.add_trace(go.Bar(
                        x=[td + hz_days], y=[hz_str], orientation="h",
                        marker_color=_hz_colors[hz_str],
                        opacity=0.15 + 0.6 * (j / max(n_win - 1, 1)),  # fade old to bright new
                        showlegend=False,
                        hovertemplate=(
                            f"W{w['window_num']}: {w.get('test_start_date','?')} "
                            f"score={w.get('composite_score',0):.4f}<extra></extra>"
                        ),
                    ))
                break

    fig_all_windows.update_layout(
        title=f"All 126 Walk-Forward Windows Across 4 Horizons",
        barmode="overlay",
        xaxis_title="Days from data start (train + test)",
        height=220,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"},
        yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT, size=11)),
    )
    st.plotly_chart(fig_all_windows, use_container_width=True)

    # ── Step 3: "Here's how we combined it" ────────────────────────
    st.divider()
    st.subheader("From 126 Scores to One Final Number")
    st.markdown(
        f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
        f"Each window produces a composite score from 9 metrics. "
        f"Recent windows are weighted higher (exponential decay, 2-year half-life). "
        f"Per-horizon scores are combined: 180d=10%, 365d=20%, 730d=35%, 1460d=35%, "
        f"minus a stability penalty (horizon std &times; 0.1). "
        f"The result: one number per model, validated from scratch with zero discrepancy."
        f"</span>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(f"**Each window is scored on 9 dimensions:**")

    _scoring_groups = [
        (BTC_ORANGE, "Path Dynamics", 35, "Do simulated paths move like real BTC?", [
            ("Drawdown Duration", "13%", "How long do drawdowns last? 2018 bear: ~12mo, 2022: ~14mo."),
            ("Recovery Time", "12%", "Time from trough to prior peak. Determines stress duration."),
            ("Time-in-Drawdown", "10%", "% of time below running max. BTC spends most time underwater."),
        ]),
        (RED, "Tail Risk", 25, "Right crash frequency? Critical for credit stress-testing.", [
            ("Tail Index", "13%", "Hill estimator on crash tail. RBB: 0.90, GARCH: 0.77, GBM: 0.48."),
            ("Band Coverage", "12%", "% of prices inside sim 5th-95th envelope. Target: 90%."),
        ]),
        (BLUE, "Distributional", 15, "Does the return shape match BTC's actual distribution?", [
            ("KS Statistic", "7%", "Catches gross mismatch at the center of the distribution."),
            ("QQ Divergence", "8%", "Pinpoints where tails or shoulders diverge."),
        ]),
        (PURPLE, "Vol Clustering", 10, "Big moves follow big moves, calm follows calm.", [
            ("ACF of r²", "10%", "Autocorrelation of squared returns — volatility persistence."),
        ]),
        (GREEN, "Forecast", 15, "How close is the median path to what actually happened?", [
            ("MAPE", "15%", "'The model was off by X% on average.'"),
        ]),
    ]

    # Donut chart + metric cards side by side
    _donut_col, _cards_col = st.columns([1, 2])

    with _donut_col:
        fig_donut = go.Figure(go.Pie(
            labels=[g[1] for g in _scoring_groups],
            values=[g[2] for g in _scoring_groups],
            hole=0.55,
            marker=dict(colors=[g[0] for g in _scoring_groups]),
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(size=10, color=TEXT),
            hovertemplate="%{label}: %{value}%<extra></extra>",
            sort=False,
            pull=[0.02] * len(_scoring_groups),
        ))
        fig_donut.update_layout(
            showlegend=False, height=400, margin=dict(l=40, r=40, t=30, b=30),
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family="JetBrains Mono, monospace"),
            annotations=[dict(
                text="<b>Composite</b>", x=0.5, y=0.5, font_size=13,
                font_color=TEXT_DIM, showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with _cards_col:
        for color, cat, cat_wt, cat_desc, metrics in _scoring_groups:
            metric_lines = "".join(
                f"<span style='color:{color};'>{name}</span>"
                f"<span style='color:{TEXT_DIM};font-size:0.72rem;'> ({wt}) — {desc}</span><br>"
                for name, wt, desc in metrics
            )
            st.markdown(
                f"<div style='background:{PANEL};border-left:3px solid {color};"
                f"border-radius:4px;padding:8px 10px;margin-bottom:6px;'>"
                f"<span style='color:{color};font-weight:700;font-size:0.85rem;'>"
                f"{cat}</span>"
                f"<span style='color:{TEXT_DIM};font-size:0.72rem;'> — {cat_wt}% — {cat_desc}</span><br>"
                f"<span style='font-size:0.78rem;'>{metric_lines}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Hero Final Score ──────────────────────────────────────────
    st.markdown(
        f"<div style='text-align:center;background:{PANEL};border:2px solid {BTC_ORANGE};"
        f"border-radius:8px;padding:24px;margin:16px 0;'>"
        f"<span style='color:{TEXT_DIM};font-size:0.9rem;'>RBB Walk-Forward Final Score</span><br>"
        f"<span style='color:{BTC_ORANGE};font-size:3rem;font-weight:800;'>0.8109</span><br>"
        f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
        f"Validated across 126 out-of-sample windows. Zero lookahead bias. "
        f"Reproduced from scratch with zero discrepancy.</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── What We Tested ─────────────────────────────────────────────
    st.divider()
    st.subheader("What We Tested")

    _wt1, _wt2, _wt3 = st.columns(3)
    with _wt1:
        st.markdown(
            f"<div style='background:{PANEL};border-top:3px solid {BTC_ORANGE};"
            f"border-radius:4px;padding:12px;height:100%;'>"
            f"<span style='color:{BTC_ORANGE};font-size:1.4rem;font-weight:800;'>30</span>"
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'> configs tested</span><br>"
            f"<span style='color:{TEXT};font-weight:700;font-size:0.9rem;'>Block Length Selection</span><br>"
            f"<span style='color:{TEXT_DIM};font-size:0.75rem;'>"
            f"15 block lengths &times; 2 sampling methods. "
            f"Scores flat across all (0.871-0.878). "
            f"Locked geometric bl=30d for maximum pool diversity.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with _wt2:
        st.markdown(
            f"<div style='background:{PANEL};border-top:3px solid {RED};"
            f"border-radius:4px;padding:12px;height:100%;'>"
            f"<span style='color:{RED};font-size:1.4rem;font-weight:800;'>3</span>"
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'> approaches tested</span><br>"
            f"<span style='color:{TEXT};font-weight:700;font-size:0.9rem;'>Regime Conditioning</span><br>"
            f"<span style='color:{TEXT_DIM};font-size:0.75rem;'>"
            f"2-state MSM, 3-state HMM, no regime. "
            f"Regime OFF won. Pool splitting suppresses tail events &mdash; "
            f"2022 crash reproduction dropped 23% &rarr; 13% with 3 pools.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with _wt3:
        st.markdown(
            f"<div style='background:{PANEL};border-top:3px solid {BLUE};"
            f"border-radius:4px;padding:12px;height:100%;'>"
            f"<span style='color:{BLUE};font-size:1.4rem;font-weight:800;'>3</span>"
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'> models compared</span><br>"
            f"<span style='color:{TEXT};font-weight:700;font-size:0.9rem;'>Model Comparison</span><br>"
            f"<span style='color:{TEXT_DIM};font-size:0.75rem;'>"
            f"RBB 0.8109 &gt; GARCH 0.7800 &gt; GBM 0.7630. "
            f"Block bootstrap beats both parametric approaches. "
            f"Gap widens at longer horizons.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Key Findings ──────────────────────────────────────────────
    st.divider()
    st.subheader("Key Findings")

    _findings_left = [
        (BTC_ORANGE, "Robust to block length",
         "30 configs tested (bl=10-180). Spread: 0.007. Not sensitive — locked bl=30 for max diversity."),
        (RED, "Regime conditioning rejected",
         "3 approaches tested. 2022 crash reproduction: 23.3% → 12.7% with 3 pools. More pools = worse tails."),
        (BLUE, "BTC vol is non-stationary (IGARCH)",
         "GARCH persistence = 1.0. Vol shocks never decay. Variance drifts at long horizons."),
        (BTC_ORANGE, "89% of paths hit 50%+ drawdown",
         "For Bitcoin-collateralized products, severe drawdowns are the norm, not the exception."),
    ]
    _findings_right = [
        (RED, "HMM found wrong regimes",
         "3-state Baum-Welch grouped crashes + rallies in same high-vol state. Crisis pool never materialized."),
        (BLUE, "GARCH loses on its own strength",
         "Vol clustering ACF: 0.75 GARCH vs 0.78 RBB. Bootstrap captures clustering better."),
        (PURPLE, "36 GARCH configs: default was optimal",
         "Sweep winner (p=2,q=2): 0.7714. Default (p=1,q=1): 0.7800. More params = more overfitting."),
    ]

    _kf1, _kf2 = st.columns(2)
    for col, findings in [(_kf1, _findings_left), (_kf2, _findings_right)]:
        with col:
            for color, title, desc in findings:
                st.markdown(
                    f"<div style='background:{PANEL};border-left:3px solid {color};"
                    f"border-radius:4px;padding:8px 10px;margin-bottom:6px;'>"
                    f"<span style='color:{color};font-weight:700;font-size:0.82rem;'>{title}</span><br>"
                    f"<span style='color:{TEXT_DIM};font-size:0.72rem;'>{desc}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Tail Event Comparison (moved before Production) ────────────
    st.divider()
    st.subheader("Tail Event Reproduction @ 1460d")
    st.markdown(
        f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
        f"Can each model produce known historical crash patterns? 5,000 simulated 4-year paths.</span>",
        unsafe_allow_html=True,
    )

    _exec_te = {}
    _p3_rbb_path = Path("results/phase2/config1/phase2_config1_standard.json")
    _p3_garch_path = Path("results/phase3/garch_walkforward/garch_walkforward_comparison.json")
    if _p3_rbb_path.exists() and _p3_garch_path.exists():
        with open(_p3_rbb_path) as f:
            _exec_rbb = json.load(f)
        with open(_p3_garch_path) as f:
            _exec_garch = json.load(f)

        _exec_te["RBB"] = _exec_rbb["rbb"]["per_horizon"]["1460d"].get("final_distributions", {}).get("tail_events", {})
        _exec_te["GARCH"] = _exec_garch["config_b"]["per_horizon"]["1460d"].get("final_distributions", {}).get("tail_events", {})
        _exec_te["GBM"] = _exec_garch["gbm"]["per_horizon"]["1460d"].get("final_distributions", {}).get("tail_events", {})

    if all(_exec_te.get(m) for m in ["RBB", "GARCH"]):
        _te_cmp = []
        for key, lbl in [("dd_50pct", "DD >= 50%"), ("dd_75pct", "DD >= 75%"),
                          ("dur_180d", "Duration >= 180d"), ("dur_365d", "Duration >= 365d")]:
            row = {"Event": lbl}
            for m, te in _exec_te.items():
                row[m] = f"{te['generic'][key]['pct']}%"
            _te_cmp.append(row)
        for key in ["2014_mt_gox", "2018_crash", "2022_crash", "2020_flash_crash"]:
            row = {"Event": key.replace("_", " ").title()}
            for m, te in _exec_te.items():
                sc = te["named_scenarios"][key]
                row[m] = f"{sc['count']:,}/{te['n_paths']:,} ({sc['pct']}%)"
            _te_cmp.append(row)
        st.dataframe(pd.DataFrame(_te_cmp), use_container_width=True, hide_index=True)

        st.markdown(
            f"<div style='background:{PANEL};border-left:3px solid {BTC_ORANGE};"
            f"border-radius:4px;padding:10px;margin-top:8px;'>"
            f"<span style='color:{BTC_ORANGE};font-weight:700;font-size:0.85rem;'>"
            f"Drawdown frequency ≠ drawdown fidelity</span><br>"
            f"<span style='color:{TEXT_DIM};font-size:0.75rem;'>"
            f"GBM produces more 75%+ drawdowns than RBB (38.6% vs 23.5%) but tail index accuracy "
            f"tells the real story — RBB: 0.90, GARCH: 0.77, GBM: 0.48. "
            f"GBM's crashes are random walk noise with wrong duration and recovery shape. "
            f"RBB's are built from actual historical crash blocks. Fewer, but structurally correct."
            f"</span></div>",
            unsafe_allow_html=True,
        )

    # ── Production Simulation ──────────────────────────────────────
    st.divider()
    st.subheader("Production Forward Projection — 1,460-Day (4-Year)")

    _exec_prod = st.session_state.get("prod_sim")
    if _exec_prod:
        _exec_ip = _exec_prod["initial_price"]
        _exec_end = _exec_prod["data_end_date"]
        _exec_final = np.array(_exec_prod["final_prices"])

        # Model specs
        st.markdown(
            f"<div style='background:{PANEL};border:1px solid {BTC_ORANGE};"
            f"border-radius:6px;padding:12px;margin-bottom:16px;'>"
            f"<span style='color:{BTC_ORANGE};font-weight:700;font-size:1rem;'>"
            f"Regime Block Bootstrap — Production Config</span><br>"
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
            f"Block sampling: geometric, mean 30d | Min block: 5d | Stride: 5d<br>"
            f"Regime switching: OFF (single pool) | Pool size: {_exec_prod.get('pool_size', '?')} blocks<br>"
            f"Training data: {_exec_prod['n_training_days']:,} days through {_exec_end}<br>"
            f"Simulation: 5,000 paths x 1,460d (4 years) | "
            f"Starting price: ${_exec_ip:,.0f}"
            f"</span></div>",
            unsafe_allow_html=True,
        )

        # Terminal price cards
        _exec_pct_below = float(np.mean(_exec_final < _exec_ip)) * 100
        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
        ec1.metric("Median Terminal", f"${np.median(_exec_final):,.0f}")
        ec2.metric("Mean Terminal", f"${np.mean(_exec_final):,.0f}")
        ec3.metric("5th Percentile", f"${np.percentile(_exec_final, 5):,.0f}")
        ec4.metric("95th Percentile", f"${np.percentile(_exec_final, 95):,.0f}")
        ec5.metric("% Below Start", f"{_exec_pct_below:.1f}%")

        # Fan chart
        _exec_start_ts = pd.Timestamp(_exec_end) + pd.Timedelta(days=1)
        _exec_n_pts = len(_exec_prod["p50"])
        _exec_x = pd.date_range(start=_exec_start_ts, periods=_exec_n_pts, freq="D").tolist()

        fig_exec_fan = go.Figure()
        fig_exec_fan.add_trace(go.Scatter(
            x=_exec_x + _exec_x[::-1],
            y=_exec_prod["p5"] + _exec_prod["p95"][::-1],
            fill="toself", fillcolor="rgba(247,147,26,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI",
        ))
        fig_exec_fan.add_trace(go.Scatter(
            x=_exec_x + _exec_x[::-1],
            y=_exec_prod["p25"] + _exec_prod["p75"][::-1],
            fill="toself", fillcolor="rgba(247,147,26,0.20)",
            line=dict(color="rgba(0,0,0,0)"), name="50% CI",
        ))
        fig_exec_fan.add_trace(go.Scatter(
            x=_exec_x, y=_exec_prod["p50"], mode="lines",
            line=dict(color=BTC_ORANGE, width=2.5), name="Median",
        ))
        fig_exec_fan.add_hline(
            y=_exec_ip, line_dash="dot", line_color=TEXT_DIM, line_width=1,
            annotation_text=f"Start: ${_exec_ip:,.0f}",
            annotation_font_color=TEXT_DIM,
        )
        fig_exec_fan.update_layout(
            title=f"4-Year Forward Projection from {_exec_end}",
            xaxis_title="", yaxis_title="Price (USD)", yaxis_type="log",
            height=550, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_exec_fan, use_container_width=True)

        # Tail events
        _exec_te_data = _exec_prod.get("tail_events", {})
        if _exec_te_data:
            _exec_n_paths = _exec_te_data["n_paths"]
            _exec_gen = _exec_te_data["generic"]

            etc1, etc2, etc3, etc4 = st.columns(4)
            for col, (key, label) in zip(
                [etc1, etc2, etc3, etc4],
                [("dd_50pct", "DD >= 50%"), ("dd_75pct", "DD >= 75%"),
                 ("dur_180d", "Duration >= 180d"), ("dur_365d", "Duration >= 365d")],
            ):
                g = _exec_gen[key]
                col.metric(label, f"{g['count']:,} / {_exec_n_paths:,} ({g['pct']}%)")

            _exec_named = _exec_te_data["named_scenarios"]
            for key, scenario in _exec_named.items():
                count = scenario["count"]
                pct = scenario["pct"]
                desc = scenario["desc"]
                name = key.replace("_", " ").title()
                color = RED if count == 0 else (GREEN if pct >= 1.0 else BTC_ORANGE)
                note = " — model cannot produce this scenario" if count == 0 else ""
                st.markdown(
                    f"<div style='padding:6px 0;border-bottom:1px solid {GRID};'>"
                    f"<span style='font-weight:600;'>{name}</span>"
                    f"<span style='color:{TEXT_DIM};font-size:0.8rem;'> — {desc}</span><br>"
                    f"<span style='color:{color};font-size:1.1rem;font-weight:700;'>"
                    f"{count:,} / {_exec_n_paths:,} ({pct}%)</span>"
                    f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>{note}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # What This Means block
            _ns = _exec_te_data["named_scenarios"]
            _n = _exec_te_data["n_paths"]
            _c22 = _ns["2022_crash"]
            _c18 = _ns["2018_crash"]
            _cmg = _ns["2014_mt_gox"]

            st.markdown("")
            st.markdown(
                f"<div style='background:{PANEL};border-left:3px solid {BTC_ORANGE};"
                f"border-radius:4px;padding:14px;margin-top:12px;'>"
                f"<span style='color:{BTC_ORANGE};font-weight:700;font-size:0.95rem;'>"
                f"What This Means</span><br><br>"
                f"<span style='color:{TEXT};font-size:0.82rem;'>"
                f"Of {_n:,} simulated 4-year paths starting from today's BTC price:</span>"
                f"<ul style='color:{TEXT};font-size:0.82rem;margin:8px 0;padding-left:20px;'>"
                f"<li><b>~{_c22['pct']}%</b> produce a drawdown matching the <b>2022 crash</b> profile: "
                f"77%+ decline lasting at least 13 months. "
                f"Nearly 1 in {max(1, round(_n / max(_c22['count'], 1)))} paths.</li>"
                f"<li><b>~{_c18['pct']}%</b> produce a drawdown matching the <b>2018 crash</b>: "
                f"84%+ decline lasting over 12 months. "
                f"Roughly 1 in {max(1, round(_n / max(_c18['count'], 1)))} paths.</li>"
                f"<li><b>~{_cmg['pct']}%</b> produce a drawdown matching the <b>2014 Mt. Gox collapse</b>: "
                f"85%+ decline lasting over 13 months. "
                f"The most severe historical crash pattern still appears in roughly "
                f"1 in {max(1, round(_n / max(_cmg['count'], 1)))} paths.</li>"
                f"</ul>"
                f"<span style='color:{TEXT_DIM};font-size:0.78rem;font-style:italic;'>"
                f"These are not hypothetical scenarios imposed on the model — they emerge naturally "
                f"from resampling actual historical BTC return sequences. The model was not told about "
                f"these crashes. It reproduces them because the return blocks that caused them are in the pool."
                f"</span></div>",
                unsafe_allow_html=True,
            )
    else:
        if st.button("Run Production Simulation", type="primary", key="exec_run_prod"):
            with st.spinner("Training on full history, simulating 5000 paths x 1460d..."):
                run_prod_simulation()
                st.rerun()

    # ── Export Executive Summary ──────────────────────────────────
    st.divider()
    st.subheader("Export Executive Summary")

    def _build_exec_html() -> str:
        """Build standalone HTML executive summary for CRO walk-through."""
        _prod = st.session_state.get("prod_sim")
        _date = datetime.now().strftime("%Y-%m-%d")

        css = """
  * { box-sizing: border-box; }
  body { background: #0a0a0a; color: #e0e0e0; font-family: 'JetBrains Mono', 'IBM Plex Mono', monospace; padding: 2rem; max-width: 900px; margin: 0 auto; line-height: 1.5; }
  @media print { body { background: #000; } }
  h1 { color: #f7931a; font-size: 1.5rem; margin-bottom: 0.3rem; }
  h2 { color: #f7931a; font-size: 1.15rem; border-bottom: 1px solid #1a1a1a; padding-bottom: 6px; margin-top: 2rem; }
  h3 { color: #e0e0e0; font-size: 0.95rem; margin-top: 1.2rem; }
  table { border-collapse: collapse; width: 100%; margin: 0.8rem 0; }
  th { background: #1a1a1a; color: #f7931a; padding: 7px 10px; text-align: left; font-size: 0.78rem; border-bottom: 2px solid #333; }
  td { padding: 5px 10px; border-bottom: 1px solid #1a1a1a; font-size: 0.78rem; }
  tr:hover { background: #111; }
  .dim { color: #888; font-size: 0.78rem; }
  .hero { text-align: center; padding: 30px 0 15px 0; }
  .hero .number { color: #f7931a; font-size: 3.2rem; font-weight: 800; letter-spacing: -2px; }
  .hero .sub { color: #e0e0e0; font-size: 1rem; }
  .hero .sub2 { color: #888; font-size: 0.78rem; }
  .stats { display: flex; gap: 10px; margin: 12px 0; }
  .stat { flex: 1; text-align: center; background: #111; border-radius: 6px; padding: 12px 4px; }
  .stat .num { color: #f7931a; font-size: 1.4rem; font-weight: 800; }
  .stat .lbl { color: #888; font-size: 0.68rem; }
  .cards { display: flex; gap: 10px; margin: 12px 0; }
  .card { flex: 1; background: #111; border-radius: 4px; padding: 10px; }
  .card-orange { border-top: 3px solid #f7931a; }
  .card-red { border-top: 3px solid #ef4444; }
  .card-blue { border-top: 3px solid #3b82f6; }
  .card .big { font-size: 1.3rem; font-weight: 800; }
  .card .title { color: #e0e0e0; font-weight: 700; font-size: 0.85rem; }
  .card .desc { color: #888; font-size: 0.72rem; }
  .model-row { background: #111; border-radius: 4px; padding: 10px; margin: 6px 0; display: flex; justify-content: space-between; align-items: center; }
  .model-row .name { font-weight: 700; font-size: 0.9rem; }
  .model-row .score { color: #f7931a; font-size: 1.2rem; font-weight: 800; }
  .model-row .desc { color: #888; font-size: 0.72rem; }
  .model-row-orange { border-left: 4px solid #f7931a; }
  .model-row-blue { border-left: 4px solid #3b82f6; }
  .model-row-dim { border-left: 4px solid #666; }
  .scoring-grid { display: flex; gap: 10px; margin: 10px 0; flex-wrap: wrap; }
  .scoring-card { flex: 1; min-width: 160px; background: #111; border-radius: 4px; padding: 8px; font-size: 0.75rem; }
  .scoring-card .cat { font-weight: 700; font-size: 0.82rem; }
  .scoring-card .met { color: #888; font-size: 0.7rem; }
  .final-score { text-align: center; background: #111; border: 2px solid #f7931a; border-radius: 8px; padding: 20px; margin: 20px 0; }
  .final-score .num { color: #f7931a; font-size: 2.8rem; font-weight: 800; }
  .final-score .lbl { color: #888; font-size: 0.8rem; }
  .footer { text-align: center; color: #666; font-size: 0.78rem; margin-top: 2rem; padding: 1rem 0; border-top: 1px solid #1a1a1a; }
  .footer .brand { color: #f7931a; font-weight: 600; }
"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>BTC Sim Engine — Executive Summary ({_date})</title>
<style>{css}</style></head><body>

<div class="hero">
<div class="number">630,000</div>
<div class="sub">simulated price paths, each scored against reality</div>
<div class="sub2">126 independent out-of-sample windows &times; 5,000 paths &times; 4 horizons &times; 3 models</div>
</div>

<div class="stats">
<div class="stat"><div class="num">126</div><div class="lbl">out-of-sample windows</div></div>
<div class="stat"><div class="num">4,209</div><div class="lbl">days of BTC history</div></div>
<div class="stat"><div class="num">9</div><div class="lbl">scoring dimensions</div></div>
<div class="stat"><div class="num">0</div><div class="lbl">lookahead bias</div></div>
</div>

<h2>Model Leaderboard</h2>
<div class="model-row model-row-orange">
<div><div class="name" style="color:#f7931a;">RBB (Block Bootstrap)</div>
<div class="desc">Resamples actual historical return sequences. Nonparametric. Preserves fat tails, vol clustering, multi-day drawdowns.</div></div>
<div class="score">0.8109</div></div>

<div class="model-row model-row-blue">
<div><div class="name" style="color:#3b82f6;">GARCH(1,1) t-dist</div>
<div class="desc">Parametric volatility clustering. Student's t innovations. Loses to GBM at 1460d due to IGARCH variance drift.</div></div>
<div class="score" style="color:#3b82f6;">0.7800</div></div>

<div class="model-row model-row-dim">
<div><div class="name" style="color:#888;">GBM (Log-Normal)</div>
<div class="desc">Baseline control. Constant drift + vol + normal innovations. The "does your model add value?" benchmark.</div></div>
<div class="score" style="color:#888;">0.7630</div></div>

<h2>What We Tested</h2>
<div class="cards">
<div class="card card-orange">
<div class="big" style="color:#f7931a;">30</div><span class="dim"> configs tested</span><br>
<div class="title">Block Length Selection</div>
<div class="desc">15 lengths &times; 2 sampling methods. Scores flat (0.871-0.878). Locked geometric bl=30d for maximum pool diversity.</div></div>

<div class="card card-red">
<div class="big" style="color:#ef4444;">3</div><span class="dim"> approaches tested</span><br>
<div class="title">Regime Conditioning</div>
<div class="desc">2-state MSM, 3-state HMM, no regime. Regime OFF won. Pool splitting suppresses tail events — 2022 crash 23% &rarr; 13%.</div></div>

<div class="card card-blue">
<div class="big" style="color:#3b82f6;">36</div><span class="dim"> GARCH configs tested</span><br>
<div class="title">Model Comparison</div>
<div class="desc">Exhaustive grid search. Default (p=1,q=1,t) was already optimal. RBB wins all horizons.</div></div>
</div>

<h2>Per-Horizon Walk-Forward Composites</h2>
<table>
<tr><th>Horizon</th><th>Weight</th><th>RBB</th><th>GARCH</th><th>GBM</th></tr>
<tr><td>180d</td><td>10%</td><td><b>0.8098</b></td><td>0.7973</td><td>0.7589</td></tr>
<tr><td>365d</td><td>20%</td><td><b>0.8221</b></td><td>0.8024</td><td>0.7705</td></tr>
<tr><td>730d</td><td>35%</td><td><b>0.8265</b></td><td>0.7976</td><td>0.7618</td></tr>
<tr><td>1460d</td><td>35%</td><td><b>0.7928</b></td><td>0.7506</td><td>0.7625</td></tr>
</table>
<p class="dim">RBB wins every horizon. Gap widens at longer horizons. GARCH loses to GBM at 1460d due to IGARCH variance drift.</p>

<h2>Composite Scoring — 9 Metrics</h2>
<p class="dim">Each metric: 0-1 (1 = perfect match). Weighted composite determines the leaderboard.</p>
<div class="scoring-grid">
<div class="scoring-card" style="border-left:3px solid #f7931a;">
<div class="cat" style="color:#f7931a;">Path Dynamics — 35%</div>
<div class="met">Drawdown Duration (13%) — how long do crashes last<br>
Recovery Time (12%) — time from trough to prior peak<br>
Time-in-Drawdown (10%) — % of time below running max</div></div>

<div class="scoring-card" style="border-left:3px solid #ef4444;">
<div class="cat" style="color:#ef4444;">Tail Risk — 25%</div>
<div class="met">Tail Index (13%) — crash tail thickness. RBB: 0.90, GARCH: 0.77, GBM: 0.48<br>
Band Coverage (12%) — % of prices inside 5th-95th envelope</div></div>

<div class="scoring-card" style="border-left:3px solid #3b82f6;">
<div class="cat" style="color:#3b82f6;">Distributional — 15%</div>
<div class="met">KS Statistic (7%) — gross distributional mismatch<br>
QQ Divergence (8%) — where tails/shoulders diverge</div></div>

<div class="scoring-card" style="border-left:3px solid #8b5cf6;">
<div class="cat" style="color:#8b5cf6;">Vol Clustering — 10%</div>
<div class="met">ACF of r&sup2; (10%) — volatility persistence</div></div>

<div class="scoring-card" style="border-left:3px solid #10b981;">
<div class="cat" style="color:#10b981;">Forecast — 15%</div>
<div class="met">MAPE (15%) — median path vs realized</div></div>
</div>

<div class="final-score">
<div class="lbl">RBB Walk-Forward Final Score</div>
<div class="num">0.8109</div>
<div class="lbl">Validated across 126 out-of-sample windows. Zero lookahead bias. Reproduced from scratch with zero discrepancy.</div>
</div>

<h2>Key Findings</h2>
<div style="border-left:3px solid #f7931a;padding:4px 0 4px 10px;margin:6px 0;">
<span style="color:#f7931a;font-weight:700;font-size:0.82rem;">Model is robust to block length</span><br>
<span class="dim">30 configurations tested (bl=10 to bl=180). Score spread: 0.007. Selected bl=30 for maximum resampling diversity.</span></div>

<div style="border-left:3px solid #ef4444;padding:4px 0 4px 10px;margin:6px 0;">
<span style="color:#ef4444;font-weight:700;font-size:0.82rem;">Regime conditioning tested and rejected on evidence</span><br>
<span class="dim">3 approaches tested. 2022 crash reproduction dropped from 23.3% to 12.7% with 3 pools. More pools = worse tails.</span></div>

<div style="border-left:3px solid #ef4444;padding:4px 0 4px 10px;margin:6px 0;">
<span style="color:#ef4444;font-weight:700;font-size:0.82rem;">Baum-Welch found volatility regimes, not directional regimes</span><br>
<span class="dim">3-state HMM grouped crashes and rallies in the same high-vol state. The intended crisis pool never materialized.</span></div>

<div style="border-left:3px solid #3b82f6;padding:4px 0 4px 10px;margin:6px 0;">
<span style="color:#3b82f6;font-weight:700;font-size:0.82rem;">BTC volatility is non-stationary (IGARCH)</span><br>
<span class="dim">GARCH fitted persistence = 1.0. Vol shocks never decay. Variance drifts over long horizons, making GARCH lose to GBM at 1460d.</span></div>

<div style="border-left:3px solid #3b82f6;padding:4px 0 4px 10px;margin:6px 0;">
<span style="color:#3b82f6;font-weight:700;font-size:0.82rem;">GARCH loses on its own strength</span><br>
<span class="dim">Vol clustering ACF: 0.75 GARCH vs 0.78 RBB. Bootstrap's within-block autocorrelation captures clustering better than the parametric model designed for it.</span></div>

<div style="border-left:3px solid #8b5cf6;padding:4px 0 4px 10px;margin:6px 0;">
<span style="color:#8b5cf6;font-weight:700;font-size:0.82rem;">Hyperparameter search confirmed default config is optimal</span><br>
<span class="dim">36 GARCH configs exhaustively tested. Sweep winner (p=2,q=2) scored 0.7714 vs default (p=1,q=1) at 0.7800. More parameters = more overfitting.</span></div>

<div style="border-left:3px solid #f7931a;padding:4px 0 4px 10px;margin:6px 0;">
<span style="color:#f7931a;font-weight:700;font-size:0.82rem;">89% of 4-year paths experience a 50%+ drawdown</span><br>
<span class="dim">For perpetual Bitcoin-collateralized products, severe drawdowns are the norm, not the exception.</span></div>

<h2>Tail Event Reproduction @ 1460d</h2>
<p class="dim">Can each model produce known historical crash patterns? 5,000 simulated 4-year paths.</p>
<table>
<tr><th>Event</th><th>RBB</th><th>GARCH</th><th>GBM</th></tr>
<tr><td>DD &ge; 50%</td><td><b>92.3%</b></td><td>59.2%</td><td>&mdash;</td></tr>
<tr><td>DD &ge; 75%</td><td><b>28.9%</b></td><td>17.8%</td><td>&mdash;</td></tr>
<tr><td>Duration &ge; 180d</td><td><b>95.4%</b></td><td>86.1%</td><td>&mdash;</td></tr>
<tr><td>Duration &ge; 365d</td><td><b>64.4%</b></td><td>44.7%</td><td>&mdash;</td></tr>
<tr><td>2014 Mt Gox &mdash; DD &ge; 85%, &ge; 13mo</td><td><b>9.0%</b></td><td>6.7%</td><td>&mdash;</td></tr>
<tr><td>2018 Crash &mdash; DD &ge; 84%, &ge; 12mo</td><td><b>10.3%</b></td><td>7.6%</td><td>&mdash;</td></tr>
<tr><td>2022 Crash &mdash; DD &ge; 77%, &ge; 13mo</td><td><b>23.3%</b></td><td>11.9%</td><td>&mdash;</td></tr>
<tr><td>2020 Flash Crash &mdash; DD &ge; 50% in 30d</td><td>0.4%</td><td>0.3%</td><td>&mdash;</td></tr>
</table>

<div style="border-left:3px solid #f7931a;padding:10px;margin:10px 0;background:#111;border-radius:4px;">
<b style="color:#f7931a;">Drawdown frequency &ne; drawdown fidelity</b><br>
<span class="dim">GBM produces more 75%+ drawdowns than RBB (38.6% vs 23.5%) but tail index accuracy tells the real story &mdash; RBB: 0.90, GARCH: 0.77, GBM: 0.48. GBM's crashes are random walk noise with wrong duration and recovery shape. RBB's are built from actual historical crash blocks. Fewer, but structurally correct.</span>
</div>
"""

        if _prod:
            _ip = _prod["initial_price"]
            _fp = np.array(_prod["final_prices"])
            _pct = float(np.mean(_fp < _ip)) * 100
            _te = _prod.get("tail_events", {})
            html += f"""
<h2>Production Forward Projection — 1,460-Day (4-Year)</h2>
<p class="dim">RBB block bootstrap | geometric bl=30 | {_prod.get('pool_size', '?')} blocks | {_prod['n_training_days']:,} training days through {_prod['data_end_date']} | Starting: ${_ip:,.0f}</p>

<div class="stats">
<div class="stat"><div class="num">${np.median(_fp):,.0f}</div><div class="lbl">Median Terminal</div></div>
<div class="stat"><div class="num">${np.mean(_fp):,.0f}</div><div class="lbl">Mean Terminal</div></div>
<div class="stat"><div class="num">${np.percentile(_fp, 5):,.0f}</div><div class="lbl">5th Percentile</div></div>
<div class="stat"><div class="num">${np.percentile(_fp, 95):,.0f}</div><div class="lbl">95th Percentile</div></div>
<div class="stat"><div class="num">{_pct:.1f}%</div><div class="lbl">Below Start</div></div>
</div>

<h3>Price Path Percentile Bands</h3>
<table>
<tr><th>Timepoint</th><th>P5</th><th>P25</th><th>Median</th><th>P75</th><th>P95</th></tr>
"""
            for day_idx, label in [(89, "90d"), (179, "180d"), (364, "1yr"), (729, "2yr"), (1094, "3yr"), (1459, "4yr")]:
                if day_idx < len(_prod["p50"]):
                    html += (f"<tr><td>{label}</td>"
                             f"<td>${_prod['p5'][day_idx]:,.0f}</td>"
                             f"<td>${_prod['p25'][day_idx]:,.0f}</td>"
                             f"<td><b>${_prod['p50'][day_idx]:,.0f}</b></td>"
                             f"<td>${_prod['p75'][day_idx]:,.0f}</td>"
                             f"<td>${_prod['p95'][day_idx]:,.0f}</td></tr>\n")
            html += "</table>\n"

            if _te:
                _n = _te["n_paths"]
                _g = _te["generic"]
                _ns = _te["named_scenarios"]
                html += f"""
<h3>Forward Projection Tail Events ({_n:,} paths)</h3>
<table>
<tr><th>Event</th><th>Count</th><th>Rate</th></tr>
<tr><td>DD &ge; 50%</td><td>{_g['dd_50pct']['count']:,}</td><td>{_g['dd_50pct']['pct']}%</td></tr>
<tr><td>DD &ge; 75%</td><td>{_g['dd_75pct']['count']:,}</td><td>{_g['dd_75pct']['pct']}%</td></tr>
<tr><td>Duration &ge; 180d</td><td>{_g['dur_180d']['count']:,}</td><td>{_g['dur_180d']['pct']}%</td></tr>
<tr><td>Duration &ge; 365d</td><td>{_g['dur_365d']['count']:,}</td><td>{_g['dur_365d']['pct']}%</td></tr>
"""
                for k in ["2014_mt_gox", "2018_crash", "2022_crash", "2020_flash_crash"]:
                    s = _ns[k]
                    html += f"<tr><td>{k.replace('_',' ').title()} — {s['desc']}</td><td>{s['count']:,}</td><td>{s['pct']}%</td></tr>\n"
                html += "</table>\n"

                # What This Means
                _h_c22 = _ns["2022_crash"]
                _h_c18 = _ns["2018_crash"]
                _h_cmg = _ns["2014_mt_gox"]
                html += f"""
<div style="border-left:3px solid #f7931a;padding:10px;margin:12px 0;background:#111;border-radius:4px;">
<b style="color:#f7931a;">What This Means</b><br><br>
Of {_n:,} simulated 4-year paths starting from today's BTC price:
<ul>
<li><b>~{_h_c22['pct']}%</b> produce a drawdown matching the <b>2022 crash</b>: 77%+ decline, &ge; 13 months. Nearly 1 in {max(1, round(_n / max(_h_c22['count'], 1)))} paths.</li>
<li><b>~{_h_c18['pct']}%</b> produce a drawdown matching the <b>2018 crash</b>: 84%+ decline, &ge; 12 months. Roughly 1 in {max(1, round(_n / max(_h_c18['count'], 1)))} paths.</li>
<li><b>~{_h_cmg['pct']}%</b> produce a drawdown matching <b>2014 Mt. Gox</b>: 85%+ decline, &ge; 13 months. Roughly 1 in {max(1, round(_n / max(_h_cmg['count'], 1)))} paths.</li>
</ul>
<span class="dim" style="font-style:italic;">These are not hypothetical scenarios imposed on the model &mdash; they emerge naturally from resampling actual historical BTC return sequences. The model was not told about these crashes. It reproduces them because the return blocks that caused them are in the pool.</span>
</div>
"""

        html += f"""
<div class="footer">
<span class="brand">@LongGamma</span> &mdash; BTC Sim Engine &mdash; Walk-forward validated price path simulation for digital credit stress-testing<br>
<span style="font-size:0.7rem;">Generated {_date}</span>
</div>
</body></html>"""
        return html

    _exec_date = datetime.now().strftime("%Y-%m-%d")
    _exec_html = _build_exec_html()

    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "Download as HTML",
            data=_exec_html,
            file_name=f"btc_sim_executive_summary_{_exec_date}.html",
            mime="text/html",
            key="download_exec_html",
        )
    with ec2:
        st.download_button(
            "Download as Markdown",
            data=_exec_html.replace("<", "&lt;").replace(">", "&gt;"),  # escaped for reference
            file_name=f"btc_sim_executive_summary_{_exec_date}.md",
            mime="text/markdown",
            key="download_exec_md",
        )

    # ── Attribution ────────────────────────────────────────────────
    st.divider()
    st.markdown(
        f"<div style='text-align:center;color:{TEXT_DIM};font-size:0.8rem;padding:20px 0;'>"
        f"<span style='color:{BTC_ORANGE};font-weight:600;'>@LongGamma</span> "
        f"&mdash; BTC Sim Engine &mdash; "
        f"Walk-forward validated price path simulation for digital credit stress-testing"
        f"</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# TAB 0: Leaderboard (Default Landing Page)
# ═══════════════════════════════════════════════════════════════════════
with tab_lb:

    # Build leaderboard rows from ALL runs (not just selected model)
    lb_rows = []
    for r in all_runs:
        wf = r["walk_forward"]
        sc = r.get("final_scorecard", {})
        sm = sc.get("scoring_metrics", {})
        opt_p = r.get("optimization", {}).get("best_params", {})

        path_dyn = np.mean([sm.get("drawdown_duration_dist", 0),
                            sm.get("recovery_time_dist", 0),
                            sm.get("time_in_drawdown_ratio", 0)])
        tail = np.mean([sm.get("tail_index_accuracy", 0),
                        sm.get("percentile_band_coverage", 0)])
        distrib = np.mean([sm.get("ks_statistic", 0), sm.get("qq_divergence", 0)])

        lb_rows.append({
            "model": r["model"],
            "horizon": r["horizon"],
            "composite": wf["weighted_composite"],
            "path_dynamics": path_dyn,
            "tail_risk": tail,
            "distributional": distrib,
            "vol_clustering": sm.get("vol_clustering_acf", 0),
            "mape": sm.get("mape", 0),
            "band_coverage": sm.get("percentile_band_coverage", 0),
            "params": opt_p,
            "scoring_metrics": sm,
            "n_windows": wf["n_windows"],
            "windows": wf.get("windows", []),
        })

    lb_rows.sort(key=lambda x: x["composite"], reverse=True)

    # GBM baselines per horizon for comparison
    gbm_by_horizon = {}
    for r in lb_rows:
        if r["model"] == "gbm":
            gbm_by_horizon[r["horizon"]] = r["composite"]

    # ── Cross-Horizon Composite Scoring ─────────────────────────────
    HZ_WEIGHTS = {"180d": 0.10, "365d": 0.20, "730d": 0.35, "1460d": 0.35}

    def _cross_horizon_score(per_hz: dict) -> dict:
        """Compute cross-horizon final score from per-horizon composites."""
        vals = [per_hz.get(k, 0) for k in HZ_WEIGHTS]
        composite = sum(per_hz.get(k, 0) * w for k, w in HZ_WEIGHTS.items())
        penalty = float(np.std(vals)) * 0.1
        return {
            "composite": round(composite, 4),
            "stability_penalty": round(penalty, 4),
            "final_score": round(composite - penalty, 4),
            "per_horizon": per_hz,
            "horizon_std": round(float(np.std(vals)), 4),
            "horizon_weights": HZ_WEIGHTS,
        }

    # Per-horizon tuned scores (from walk-forward)
    cross_hz_models = {}
    all_model_names = sorted(set(r["model"] for r in lb_rows))
    for m_name in all_model_names:
        per_hz = {}
        per_hz_params = {}
        for r in lb_rows:
            if r["model"] == m_name:
                per_hz[r["horizon"]] = r["composite"]
                per_hz_params[r["horizon"]] = r.get("params", {})
        if len(per_hz) >= 2:
            ch = _cross_horizon_score(per_hz)
            # Detect if all horizons share the same params (locked config)
            param_vals = list(per_hz_params.values())
            is_locked = (
                len(param_vals) >= 2
                and all(p == param_vals[0] for p in param_vals[1:])
                and param_vals[0]  # not empty
            )
            if is_locked:
                ch["label"] = f"{m_name.replace('_',' ').title()} (single config locked)"
                ch["config_type"] = "single_config_locked"
            else:
                ch["label"] = f"{m_name.replace('_',' ').title()} (per-horizon tuned)"
                ch["config_type"] = "per_horizon_tuned"
            ch["model"] = m_name
            cross_hz_models[f"{m_name}_tuned"] = ch

    # Sort cross-horizon entries by final_score
    cross_hz_ranked = sorted(cross_hz_models.values(),
                             key=lambda x: x["final_score"], reverse=True)

    # ── Section 0: Cross-Horizon Final Score ────────────────────────
    if cross_hz_ranked:
        st.subheader("Cross-Horizon Final Score")
        st.markdown(
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
            f"Single number ranking. Weights: 180d={HZ_WEIGHTS['180d']:.0%}, "
            f"365d={HZ_WEIGHTS['365d']:.0%}, 730d={HZ_WEIGHTS['730d']:.0%}, "
            f"1460d={HZ_WEIGHTS['1460d']:.0%}. "
            f"Stability penalty = std across horizons x 0.1."
            f"</span>", unsafe_allow_html=True,
        )

        ch_rows = []
        for ch in cross_hz_ranked:
            ch_rows.append({
                "Model": ch["label"],
                "Final Score": ch["final_score"],
                "Composite": ch["composite"],
                "Penalty": ch["stability_penalty"],
                "180d": ch["per_horizon"].get("180d", 0),
                "365d": ch["per_horizon"].get("365d", 0),
                "730d": ch["per_horizon"].get("730d", 0),
                "1460d": ch["per_horizon"].get("1460d", 0),
                "Std": ch["horizon_std"],
            })
        ch_df = pd.DataFrame(ch_rows)
        float_cols = ["Final Score", "Composite", "Penalty",
                      "180d", "365d", "730d", "1460d", "Std"]
        st.dataframe(
            ch_df.style.format({c: "{:.4f}" for c in float_cols})
                .background_gradient(subset=["Final Score"], cmap="YlOrRd", vmin=0.5, vmax=1),
            use_container_width=True, hide_index=True,
        )

        # Winner callout
        ch_winner = cross_hz_ranked[0]
        st.markdown(
            f"<div style='background:{PANEL};border:2px solid {GREEN};"
            f"border-radius:6px;padding:12px;margin:8px 0;text-align:center;'>"
            f"<span style='color:{GREEN};font-size:1.2rem;font-weight:700;'>"
            f"Overall Winner: {ch_winner['label']}</span><br>"
            f"<span style='color:{BTC_ORANGE};font-size:1.5rem;font-weight:700;'>"
            f"Final Score: {ch_winner['final_score']:.4f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

    # ── Section 1: Per-Horizon Leaderboard ──────────────────────────
    winner = lb_rows[0]
    gbm_base = gbm_by_horizon.get(winner["horizon"], 0)
    gbm_delta = ((winner["composite"] - gbm_base) / gbm_base * 100) if gbm_base > 0 else 0

    # Winner params summary
    wp = winner["params"]
    if winner["model"] == "regime_block_bootstrap" and wp:
        param_str = (
            f"{wp.get('block_length_sampling','?')} sampling, "
            f"block={wp.get('mean_block_length','?')}d, "
            f"tm={wp.get('transition_matrix_method','?')}"
        )
    elif wp:
        param_str = str(wp)
    else:
        param_str = "MLE from data"

    st.markdown(
        f"<div style='background:{PANEL};border:2px solid {BTC_ORANGE};"
        f"border-radius:8px;padding:24px;margin-bottom:16px;'>"
        f"<div style='color:{BTC_ORANGE};font-size:0.8rem;text-transform:uppercase;"
        f"letter-spacing:2px;margin-bottom:8px;'>Overall Winner</div>"
        f"<div style='font-size:2rem;font-weight:700;color:#ffffff;'>"
        f"{winner['model'].replace('_',' ').title()}</div>"
        f"<div style='font-size:1.2rem;color:{TEXT};margin:4px 0;'>"
        f"Horizon: {winner['horizon']} | Composite: "
        f"<span style='color:{BTC_ORANGE};font-weight:700;'>{winner['composite']:.4f}</span></div>"
        f"<div style='color:{TEXT_DIM};font-size:0.85rem;'>{param_str}</div>"
        f"<div style='color:{GREEN};font-size:1rem;margin-top:8px;font-weight:600;'>"
        f"Beats GBM baseline by {gbm_delta:+.1f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Section 2: Leaderboard Table ────────────────────────────────
    st.subheader("Leaderboard")

    table_rows = []
    for i, r in enumerate(lb_rows):
        gbm_base_h = gbm_by_horizon.get(r["horizon"], 0)
        if r["model"] == "gbm":
            vs_gbm = "baseline"
        elif gbm_base_h > 0:
            delta = (r["composite"] - gbm_base_h) / gbm_base_h * 100
            vs_gbm = f"{delta:+.1f}%"
        else:
            vs_gbm = "N/A"

        # Compact params for RBB
        p = r["params"]
        if r["model"] == "regime_block_bootstrap" and p:
            param_display = f"{p.get('block_length_sampling','?')[:4]} bl={p.get('mean_block_length','?')}d"
        else:
            param_display = ""

        table_rows.append({
            "Rank": i + 1,
            "Model": r["model"].replace("_", " ").title(),
            "Horizon": r["horizon"],
            "Composite": r["composite"],
            "Path Dyn": r["path_dynamics"],
            "Tail Risk": r["tail_risk"],
            "Distrib": r["distributional"],
            "MAPE": r["mape"],
            "Band Cov": r["band_coverage"],
            "vs GBM": vs_gbm,
            "Params": param_display,
        })

    lb_df = pd.DataFrame(table_rows)
    float_cols = ["Composite", "Path Dyn", "Tail Risk", "Distrib", "MAPE", "Band Cov"]
    st.dataframe(
        lb_df.style.format({c: "{:.4f}" for c in float_cols})
            .background_gradient(subset=["Composite"], cmap="YlOrRd", vmin=0.5, vmax=1.0),
        use_container_width=True,
        hide_index=True,
        height=min(len(table_rows) * 40 + 50, 500),
    )

    # ── Section 3: Horizon Breakdown ────────────────────────────────
    st.subheader("Horizon Breakdown")
    all_horizons = sorted(set(r["horizon"] for r in lb_rows),
                          key=lambda h: int(h.replace("d", "")))
    hz_cols = st.columns(len(all_horizons))

    for col, hz in zip(hz_cols, all_horizons):
        hz_runs = [r for r in lb_rows if r["horizon"] == hz]
        hz_runs.sort(key=lambda x: x["composite"], reverse=True)
        w = hz_runs[0]
        runner = hz_runs[1] if len(hz_runs) > 1 else None

        # Find biggest metric gap
        biggest_gap_metric = ""
        biggest_gap = 0
        if runner:
            for mk in ["path_dynamics", "tail_risk", "distributional",
                        "vol_clustering", "mape", "band_coverage"]:
                gap = w[mk] - runner[mk]
                if abs(gap) > abs(biggest_gap):
                    biggest_gap = gap
                    biggest_gap_metric = mk.replace("_", " ").title()

        with col:
            color = GREEN if w["model"] != "gbm" else BTC_ORANGE
            st.markdown(
                f"<div style='background:{PANEL};border:1px solid {GRID};"
                f"border-radius:6px;padding:12px;text-align:center;'>"
                f"<div style='color:{TEXT_DIM};font-size:0.75rem;'>{hz}</div>"
                f"<div style='color:{color};font-size:1.1rem;font-weight:700;'>"
                f"{w['model'].replace('_',' ').title()[:12]}</div>"
                f"<div style='color:{BTC_ORANGE};font-size:1.3rem;font-weight:700;'>"
                f"{w['composite']:.4f}</div>"
                + (f"<div style='color:{TEXT_DIM};font-size:0.7rem;margin-top:4px;'>"
                   f"Runner: {runner['model'].replace('_',' ').title()[:10]} "
                   f"{runner['composite']:.4f}</div>"
                   f"<div style='color:{TEXT_DIM};font-size:0.65rem;'>"
                   f"Gap: {biggest_gap_metric} {biggest_gap:+.3f}</div>"
                   if runner else "")
                + f"</div>",
                unsafe_allow_html=True,
            )

    # ── Section 4: Score Heatmap ────────────────────────────────────
    st.subheader("Score Heatmap")
    all_models = sorted(set(r["model"] for r in lb_rows))
    hm_z = []
    hm_text = []
    for m in all_models:
        row = []
        text_row = []
        for hz in all_horizons:
            match = [r for r in lb_rows if r["model"] == m and r["horizon"] == hz]
            if match:
                row.append(match[0]["composite"])
                text_row.append(f"{match[0]['composite']:.4f}")
            else:
                row.append(0)
                text_row.append("")
        hm_z.append(row)
        hm_text.append(text_row)

    fig_hm = go.Figure(data=go.Heatmap(
        z=hm_z,
        x=all_horizons,
        y=[m.replace("_", " ").title() for m in all_models],
        colorscale=[[0, RED], [0.5, YELLOW], [1, GREEN]],
        zmin=0.5, zmax=1.0,
        text=hm_text, texttemplate="%{text}",
        textfont=dict(size=13, color="#ffffff"),
    ))
    fig_hm.update_layout(
        title="Composite Score: Model x Horizon",
        height=max(200, len(all_models) * 80 + 100),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Section 5: Metric-Level Comparison ──────────────────────────
    st.subheader("Metric-Level Comparison")
    metric_keys = [m["key"] for m in METRIC_INFO]
    metric_names = [m["name"] for m in METRIC_INFO]

    for mk, mn in zip(metric_keys, metric_names):
        fig_m = go.Figure()
        for m_name in all_models:
            m_scores = []
            m_horizons = []
            for hz in all_horizons:
                match = [r for r in lb_rows if r["model"] == m_name and r["horizon"] == hz]
                if match:
                    m_scores.append(match[0]["scoring_metrics"].get(mk, 0))
                    m_horizons.append(hz)
            fig_m.add_trace(go.Bar(
                x=m_horizons, y=m_scores,
                name=m_name.replace("_", " ").title(),
                opacity=0.8,
            ))
        fig_m.update_layout(
            title=mn, xaxis_title="Horizon", yaxis_title="Score",
            yaxis_range=[0, 1], barmode="group", height=300,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_m, use_container_width=True)

    # ── Section 6: Cross-Window Stability ───────────────────────────
    st.subheader("Cross-Window Stability")
    st.markdown(
        f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
        f"Composite score per walk-forward window over time. "
        f"Consistent models are more trustworthy than volatile ones."
        f"</span>", unsafe_allow_html=True,
    )

    # Use the selected horizon's windows
    fig_stab = go.Figure()
    for m_name in all_models:
        match = [r for r in lb_rows if r["model"] == m_name and r["horizon"] == selected_horizon]
        if match:
            windows = match[0]["windows"]
            w_dates = [w.get("test_start_date", f"W{w['window_num']}") for w in windows]
            w_scores = [w.get("composite_score", 0) for w in windows]
            w_std = float(np.std(w_scores)) if w_scores else 0

            fig_stab.add_trace(go.Scatter(
                x=w_dates, y=w_scores,
                mode="lines+markers",
                name=f"{m_name.replace('_',' ').title()} (std={w_std:.3f})",
                marker=dict(size=7),
                line=dict(width=2),
            ))

    fig_stab.update_layout(
        title=f"Composite Score per Window ({selected_horizon})",
        xaxis_title="Window Start Date",
        yaxis_title="Composite Score",
        yaxis_range=[0.4, 1.0],
        height=400,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_stab, use_container_width=True)

    # ── Export ──────────────────────────────────────────────────────
    st.divider()

    def _lb_to_markdown() -> str:
        lines = [f"# BTC Sim Engine — Leaderboard ({data.get('timestamp', '?')[:10]})"]
        lines.append(f"**Generated:** {data.get('timestamp', '?')[:19]}")
        lines.append(f"**Run mode:** {data.get('run_mode', '?')}")
        lines.append("")

        # Cross-horizon winner
        if cross_hz_ranked:
            ch_w = cross_hz_ranked[0]
            lines.append(f"## Cross-Horizon Final Score")
            lines.append(f"**Winner: {ch_w['label']}** — Final Score: **{ch_w['final_score']:.4f}**")
            lines.append(f"- Composite: {ch_w['composite']:.4f} | Stability penalty: {ch_w['stability_penalty']:.4f}")
            lines.append(f"- Per-horizon: " + " | ".join(
                f"{k}={v:.4f}" for k, v in ch_w["per_horizon"].items()))
            lines.append(f"- Weights: " + " | ".join(
                f"{k}={v:.0%}" for k, v in HZ_WEIGHTS.items()))
            lines.append("")
            lines.append("| Model | Final Score | Composite | Penalty | 180d | 365d | 730d | 1460d | Std |")
            lines.append("|-------|------------|-----------|---------|------|------|------|-------|-----|")
            for ch in cross_hz_ranked:
                ph = ch["per_horizon"]
                lines.append(
                    f"| {ch['label']} | {ch['final_score']:.4f} | "
                    f"{ch['composite']:.4f} | {ch['stability_penalty']:.4f} | "
                    f"{ph.get('180d',0):.4f} | {ph.get('365d',0):.4f} | "
                    f"{ph.get('730d',0):.4f} | {ph.get('1460d',0):.4f} | "
                    f"{ch['horizon_std']:.4f} |")
            lines.append("")

        lines.append(f"## Per-Horizon Winner: {winner['model'].replace('_',' ').title()} @ {winner['horizon']}")
        lines.append(f"- Composite: **{winner['composite']:.4f}**")
        lines.append(f"- Parameters: {param_str}")
        lines.append(f"- Beats GBM by {gbm_delta:+.1f}%")
        lines.append("")
        lines.append("## Per-Horizon Leaderboard")
        lines.append("| Rank | Model | Horizon | Composite | Path Dyn | Tail Risk | Distrib | MAPE | Band Cov | vs GBM |")
        lines.append("|------|-------|---------|-----------|----------|-----------|---------|------|----------|--------|")
        for tr in table_rows:
            lines.append(
                f"| {tr['Rank']} | {tr['Model']} | {tr['Horizon']} | "
                f"{tr['Composite']:.4f} | {tr['Path Dyn']:.4f} | "
                f"{tr['Tail Risk']:.4f} | {tr['Distrib']:.4f} | "
                f"{tr['MAPE']:.4f} | {tr['Band Cov']:.4f} | {tr['vs GBM']} |"
            )
        lines.append("")
        lines.append("## Horizon Winners")
        for hz in all_horizons:
            hz_runs = [r for r in lb_rows if r["horizon"] == hz]
            hz_runs.sort(key=lambda x: x["composite"], reverse=True)
            w = hz_runs[0]
            lines.append(f"- **{hz}**: {w['model'].replace('_',' ').title()} ({w['composite']:.4f})")
        return "\n".join(lines)

    def _lb_to_json() -> str:
        return json.dumps({
            "timestamp": data.get("timestamp"),
            "cross_horizon": {
                ch["model"]: {
                    "composite": ch["composite"],
                    "stability_penalty": ch["stability_penalty"],
                    "final_score": ch["final_score"],
                    "per_horizon": ch["per_horizon"],
                    "horizon_std": ch["horizon_std"],
                    "horizon_weights": ch["horizon_weights"],
                    "config_type": ch["config_type"],
                }
                for ch in cross_hz_ranked
            },
            "overall_winner": {
                "model": cross_hz_ranked[0]["label"] if cross_hz_ranked else "?",
                "final_score": cross_hz_ranked[0]["final_score"] if cross_hz_ranked else 0,
            },
            "per_horizon_winner": {
                "model": winner["model"],
                "horizon": winner["horizon"],
                "composite": winner["composite"],
                "params": winner["params"],
                "vs_gbm_pct": round(gbm_delta, 1),
            },
            "leaderboard": [
                {k: v for k, v in tr.items()}
                for tr in table_rows
            ],
            "per_horizon_winners": {
                hz: next(
                    (r["model"] for r in sorted(
                        [r for r in lb_rows if r["horizon"] == hz],
                        key=lambda x: x["composite"], reverse=True
                    )), None
                )
                for hz in all_horizons
            },
        }, indent=2, default=str)

    _lb_date = data.get("timestamp", "")[:10]
    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        st.download_button(
            "Export as JSON", data=_lb_to_json(),
            file_name=f"leaderboard_{_lb_date}.json", mime="application/json",
        )
    with exp_c2:
        st.download_button(
            "Export as Markdown", data=_lb_to_markdown(),
            file_name=f"leaderboard_{_lb_date}.md", mime="text/markdown",
        )


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Overview (Simplified Front Page)
# ═══════════════════════════════════════════════════════════════════════
with tab_overview:
    wf = selected_run["walk_forward"]
    sc = selected_run.get("final_scorecard", {})
    opt = selected_run["optimization"]
    windows = wf.get("windows", [])

    # ── Top metrics row ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weighted Composite", f"{wf['weighted_composite']:.4f}")
    c2.metric("Model", selected_model)
    c3.metric("Horizon", selected_horizon)
    c4.metric("WF Windows", wf["n_windows"])

    bp = opt.get("best_params", {})
    if bp:
        st.markdown(f"**Parameters:** `{bp}`")

    # ── Chart helpers ────────────────────────────────────────────────
    def _percentile_chart(sim_data: dict, hist_data: dict | None, title: str,
                          yaxis_title: str, hist_label: str = "Historical",
                          sim_label: str = "Simulated",
                          horizontal_line: float | None = None,
                          hline_label: str = "",
                          annotation: str = "",
                          log_y: bool = False) -> go.Figure:
        """Build percentile distribution chart. X=percentile (1-99%), Y=value."""
        fig = go.Figure()
        if sim_data and sim_data.get("y"):
            fig.add_trace(go.Scatter(
                x=sim_data["x"], y=sim_data["y"], mode="lines",
                line=dict(color=BTC_ORANGE, width=2.5),
                name=f"{sim_label} (n={sim_data.get('n',0):,})",
                hovertemplate="P%{x:.0f}: %{y:.2f}<extra></extra>",
            ))
        if hist_data and hist_data.get("y"):
            fig.add_trace(go.Scatter(
                x=hist_data["x"], y=hist_data["y"], mode="lines",
                line=dict(color=TEXT, width=2, dash="dash"),
                name=f"{hist_label} (n={hist_data.get('n',0):,})",
                hovertemplate="P%{x:.0f}: %{y:.2f}<extra></extra>",
            ))
        if horizontal_line is not None:
            fig.add_hline(y=horizontal_line, line_dash="dash", line_color=GREEN, line_width=2,
                          annotation_text=hline_label, annotation_font_color=GREEN)
        if annotation:
            fig.add_annotation(text=annotation, xref="paper", yref="paper", x=0.98, y=0.95,
                               showarrow=False, font=dict(color=TEXT_DIM, size=10), xanchor="right")
        layout_kwargs = dict(title=title, xaxis_title="Percentile",
                             yaxis_title=yaxis_title, **PLOTLY_LAYOUT)
        if log_y:
            layout_kwargs["yaxis_type"] = "log"
        fig.update_layout(**layout_kwargs)
        return fig

    def _render_distribution_charts(dist_data: dict) -> None:
        """Render all percentile distribution charts from a distributions dict."""
        if not dist_data:
            return

        # Row 1: Drawdown Duration + Recovery Time
        dd = dist_data.get("drawdown_duration", {})
        rt = dist_data.get("recovery_time", {})
        col_dd, col_rt = st.columns(2)
        if dd.get("sim"):
            ss, hs = dd.get("sim_summary", {}), dd.get("hist_summary", {})
            if ss and hs:
                ann = f"Sim median: {ss.get('p50',0):.0f}d | Hist median: {hs.get('p50',0):.0f}d"
            else:
                ann = ""
            with col_dd:
                st.plotly_chart(_percentile_chart(
                    dd["sim"], dd.get("hist"), "Drawdown Duration",
                    "Duration (days)", annotation=ann), use_container_width=True)
        if rt.get("sim"):
            ss, hs = rt.get("sim_summary", {}), rt.get("hist_summary", {})
            if ss and hs:
                ann = f"Sim median: {ss.get('p50',0):.0f}d | Hist median: {hs.get('p50',0):.0f}d"
            else:
                ann = ""
            with col_rt:
                st.plotly_chart(_percentile_chart(
                    rt["sim"], rt.get("hist"), "Recovery Time",
                    "Days to recover", annotation=ann), use_container_width=True)

        # Row 2: Time-in-Drawdown + Max Drawdown Depth
        tidd = dist_data.get("time_in_drawdown", {})
        mdd = dist_data.get("max_drawdown_depth", {})
        col_tidd, col_mdd = st.columns(2)
        if tidd.get("sim"):
            with col_tidd:
                st.plotly_chart(_percentile_chart(
                    tidd["sim"], None, "Time-in-Drawdown Ratio",
                    "Ratio (0-1)",
                    horizontal_line=tidd.get("hist_value"),
                    hline_label=f"Historical: {tidd.get('hist_value',0):.2f}"),
                    use_container_width=True)
        if mdd.get("sim"):
            ss, hs = mdd.get("sim_summary", {}), mdd.get("hist_summary", {})
            if ss and hs:
                ann = f"Sim median: {ss.get('p50',0)*100:.1f}% | Hist median: {hs.get('p50',0)*100:.1f}%"
            else:
                ann = ""
            with col_mdd:
                st.plotly_chart(_percentile_chart(
                    mdd["sim"], mdd.get("hist"), "Max Drawdown Depth",
                    "Drawdown (%)", annotation=ann), use_container_width=True)

        # Row 3: Log Returns + Terminal Price
        lr = dist_data.get("log_returns", {})
        tp = dist_data.get("terminal_price", {})
        col_ret, col_tp = st.columns(2)
        if lr.get("sim"):
            ss, hs = lr.get("sim_summary", {}), lr.get("hist_summary", {})
            if ss and hs:
                ann = f"Sim std: {ss.get('std',0):.4f} | Hist std: {hs.get('std',0):.4f}"
            else:
                ann = ""
            with col_ret:
                st.plotly_chart(_percentile_chart(
                    lr["sim"], lr.get("hist"), "Daily Log Returns",
                    "Log Return", annotation=ann), use_container_width=True)
        if tp.get("sim"):
            with col_tp:
                st.plotly_chart(_percentile_chart(
                    tp["sim"], None, "Terminal Price Distribution",
                    "Price (USD)", log_y=True), use_container_width=True)

        # Row 4: ACF Profile + VaR Calibration
        acf = dist_data.get("acf_squared", {})
        var = dist_data.get("var_calibration", {})
        if acf.get("lags") or var:
            col_acf, col_var = st.columns(2)
            if acf.get("lags"):
                with col_acf:
                    lags = acf["lags"]
                    hist_acf = acf.get("hist_acf", [])
                    sim_acf = acf.get("sim_acf_mean", acf.get("sim_acf", []))
                    fig_acf = go.Figure()
                    if hist_acf:
                        for i, lag in enumerate(lags):
                            fig_acf.add_trace(go.Scatter(
                                x=[lag, lag], y=[0, hist_acf[i]], mode="lines",
                                line=dict(color=TEXT, width=2),
                                showlegend=(i == 0), name="Historical",
                            ))
                        fig_acf.add_trace(go.Scatter(
                            x=lags, y=hist_acf, mode="markers",
                            marker=dict(color=TEXT, size=6), showlegend=False,
                        ))
                    if sim_acf:
                        lags_offset = [l + 0.25 for l in lags]
                        for i, lag in enumerate(lags_offset):
                            fig_acf.add_trace(go.Scatter(
                                x=[lag, lag], y=[0, sim_acf[i]], mode="lines",
                                line=dict(color=BTC_ORANGE, width=2),
                                showlegend=(i == 0), name="Simulated",
                            ))
                        fig_acf.add_trace(go.Scatter(
                            x=lags_offset, y=sim_acf, mode="markers",
                            marker=dict(color=BTC_ORANGE, size=6), showlegend=False,
                        ))
                    n_obs = max(len(hist_acf) * 10, 365)
                    sig_band = 1.96 / np.sqrt(n_obs)
                    fig_acf.add_hline(y=sig_band, line_dash="dash", line_color=BLUE,
                                      line_width=1, opacity=0.6)
                    fig_acf.add_hline(y=-sig_band, line_dash="dash", line_color=BLUE,
                                      line_width=1, opacity=0.6)
                    fig_acf.add_hline(y=0, line_color=TEXT_DIM, line_width=1)
                    fig_acf.update_layout(
                        title="ACF of Squared Returns",
                        xaxis_title="Lag", yaxis_title="Autocorrelation",
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig_acf, use_container_width=True)
            if var:
                with col_var:
                    fig_var = go.Figure()
                    levels, expected, observed = [], [], []
                    for label in ["1pct", "5pct"]:
                        v = var.get(label, {})
                        if v:
                            levels.append(f"{label.replace('pct','%')} VaR")
                            expected.append(v["expected_rate"] * 100)
                            observed.append(v["observed_rate"] * 100)
                    if levels:
                        fig_var.add_trace(go.Bar(
                            x=levels, y=expected, name="Expected Rate",
                            marker_color=BLUE, opacity=0.7,
                            text=[f"{v:.1f}%" for v in expected], textposition="auto",
                        ))
                        fig_var.add_trace(go.Bar(
                            x=levels, y=observed, name="Observed Rate",
                            marker_color=BTC_ORANGE, opacity=0.7,
                            text=[f"{v:.1f}%" for v in observed], textposition="auto",
                        ))
                        fig_var.update_layout(
                            title="VaR Calibration: Expected vs Observed",
                            xaxis_title="VaR Level", yaxis_title="Breach Rate (%)",
                            barmode="group", **PLOTLY_LAYOUT,
                        )
                        st.plotly_chart(fig_var, use_container_width=True)

    # ── On-demand window re-simulation ───────────────────────────────
    @st.cache_data
    def _load_full_arrays():
        from data.loader import BTCDataLoader, get_price_array as _gpa
        _loader = BTCDataLoader()
        _df = _loader.load_processed_data()
        return _df["log_return"].values, _df["Close"].values

    def _run_window_detail(w_data: dict, model_name: str) -> dict | None:
        """Re-simulate a window and compute distributions."""
        from models.registry import get_model as _dg
        from validation.distributions import compute_distributions as _cd

        t_end = w_data.get("train_end_idx")
        ts_start = w_data.get("test_start_idx")
        ts_end = w_data.get("test_end_idx")
        if not t_end or not ts_start or not ts_end:
            return None

        full_returns, full_prices = _load_full_arrays()

        train_ret = full_returns[:t_end]
        train_px = full_prices[:t_end + 1]
        test_px = full_prices[ts_start:ts_end + 1]
        test_ret = full_returns[ts_start:ts_end]

        if len(test_ret) < 10:
            return None

        model = _dg(model_name)
        fp = w_data.get("fitted_params", {})
        if fp:
            model.set_params(**{k: v for k, v in fp.items()
                                if hasattr(model, k)})
        model.fit(train_ret)

        sim = model.simulate(
            n_simulations=2000,
            n_steps=len(test_ret),
            initial_price=float(train_px[-1]),
            seed=42,
        )

        return _cd(sim.paths, sim.log_returns, test_px, test_ret)

    # ── Helper: render a single window's content ────────────────────
    def _render_window_view(w_data: dict, key_prefix: str) -> None:
        """Render fan chart + radar + metric cards for one window."""
        w_scores = w_data.get("scoring_metrics", {})

        # Fan chart — large, dominant visual
        sim_pct = w_data.get("sim_percentiles")
        realized = w_data.get("realized_prices")
        if sim_pct and realized:
            fig_fan = make_fan_chart(
                sim_pct, realized,
                title=(f"Simulated vs Realized: "
                       f"{w_data.get('test_start_date','?')} to {w_data.get('test_end_date','?')}"),
                start_date=w_data.get("test_start_date"),
            )
            fig_fan.update_layout(height=550)
            st.plotly_chart(fig_fan, use_container_width=True)

        # Radar + top scores side by side
        col_radar, col_scores = st.columns([1, 1])

        with col_radar:
            categories = [m["key"] for m in METRIC_INFO]
            values = [w_scores.get(k, 0) for k in categories]
            labels = [m["name"] for m in METRIC_INFO]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor="rgba(247,147,26,0.15)",
                line=dict(color=BTC_ORANGE, width=2),
                marker=dict(size=6, color=BTC_ORANGE),
            ))
            radar_layout = {k: v for k, v in PLOTLY_LAYOUT.items()
                            if k not in ("xaxis", "yaxis")}
            fig_radar.update_layout(
                polar=dict(
                    bgcolor=PANEL,
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID,
                                    tickfont=dict(color=TEXT_DIM, size=9)),
                    angularaxis=dict(gridcolor=GRID,
                                     tickfont=dict(color=TEXT, size=10)),
                ),
                title=f"W{w_data.get('window_num','')}: {w_data.get('test_start_date','?')[:4]}",
                showlegend=False, height=380, **radar_layout,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_scores:
            # Compact score summary
            composite = w_data.get("composite_score", 0)
            weight = w_data.get("weight", 0)
            st.markdown(
                f"<div style='background:{PANEL};border:1px solid {GRID};"
                f"border-radius:6px;padding:12px;margin-bottom:12px;'>"
                f"<span style='color:{BTC_ORANGE};font-size:1.4rem;font-weight:700;'>"
                f"{composite:.4f}</span>"
                f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
                f"  composite  |  weight: {weight:.4f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Category averages as compact rows
            cat_groups = {}
            for info in METRIC_INFO:
                cat = info["category"]
                if cat not in cat_groups:
                    cat_groups[cat] = []
                cat_groups[cat].append(info)

            for cat, infos in cat_groups.items():
                cat_color = CATEGORY_COLORS.get(cat, TEXT_DIM)
                cat_scores = [w_scores.get(m["key"], 0) for m in infos]
                cat_avg = np.mean(cat_scores)
                avg_color = score_color(cat_avg)

                details = " | ".join(
                    f"{m['name']}: {w_scores.get(m['key'],0):.3f}"
                    for m in infos
                )
                st.markdown(
                    f"<div style='padding:4px 0;border-bottom:1px solid {GRID};'>"
                    f"<span style='color:{cat_color};font-weight:600;'>{cat}</span>"
                    f"<span style='color:{avg_color};float:right;font-weight:700;'>"
                    f"{cat_avg:.3f}</span><br>"
                    f"<span style='color:{TEXT_DIM};font-size:0.75rem;'>{details}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Detailed metric cards (collapsed by default)
        with st.expander("Detailed Metric Cards"):
            current_category = None
            for info in METRIC_INFO:
                if info["category"] != current_category:
                    current_category = info["category"]
                    cat_color = CATEGORY_COLORS.get(current_category, TEXT_DIM)
                    cat_weight = sum(m["weight"] for m in METRIC_INFO
                                     if m["category"] == current_category)
                    cat_avg = np.mean([w_scores.get(m["key"], 0) for m in METRIC_INFO
                                       if m["category"] == current_category])
                    st.markdown(
                        f'<div class="category-header">'
                        f'<span style="color:{cat_color};">{current_category}</span>'
                        f' &mdash; {cat_weight*100:.0f}% total weight'
                        f' &mdash; avg score: {cat_avg:.3f}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                s = w_scores.get(info["key"], 0)
                st.markdown(render_metric_card(info, s), unsafe_allow_html=True)

        # ── On-demand detailed analysis ────────────────────────────
        st.divider()
        _detail_key = f"detail_{key_prefix}"

        if st.button("Run Detailed Analysis", key=f"btn_{key_prefix}",
                      help="Re-simulates this window to compute percentile distributions and ACF"):
            with st.spinner(f"Re-simulating W{w_data.get('window_num', 0)} (2000 paths)..."):
                _win_dist = _run_window_detail(w_data, selected_model)
                if _win_dist:
                    st.session_state[_detail_key] = _win_dist

        _cached_detail = st.session_state.get(_detail_key)
        if _cached_detail:
            _render_distribution_charts(_cached_detail)

    # ── Summary + per-window views ────────────────────────────────────
    if not windows:
        st.warning("No walk-forward windows for this combination.")
    else:
        MAX_WINDOW_TABS = 12

        # Extract year range from window dates for the range slider
        _win_years = sorted(set(
            int(w.get("test_start_date", "2016")[:4])
            for w in windows if w.get("test_start_date")
        ))
        _year_min, _year_max = _win_years[0], _win_years[-1]

        if _year_max - _year_min >= 2 and len(windows) > MAX_WINDOW_TABS:
            _yr_range = st.slider(
                "Window date range",
                min_value=_year_min, max_value=_year_max,
                value=(_year_min, _year_max),
                key="overview_year_range",
            )
            # Filter windows to the selected range
            filtered = [
                (i, w) for i, w in enumerate(windows)
                if w.get("test_start_date") and
                _yr_range[0] <= int(w["test_start_date"][:4]) <= _yr_range[1]
            ]
        else:
            filtered = list(enumerate(windows))

        # Sample evenly if still too many
        if len(filtered) <= MAX_WINDOW_TABS:
            display_windows = filtered
        else:
            step = len(filtered) / MAX_WINDOW_TABS
            pick_indices = [int(round(i * step)) for i in range(MAX_WINDOW_TABS)]
            pick_indices[-1] = len(filtered) - 1
            pick_indices = sorted(set(pick_indices))
            display_windows = [filtered[i] for i in pick_indices]

        tab_labels = ["Summary"] + [
            f"W{w['window_num']}: {w.get('test_start_date','?')[:7]}"
            for _, w in display_windows
        ]
        overview_tabs = st.tabs(tab_labels)

        if len(windows) > MAX_WINDOW_TABS:
            st.caption(
                f"Showing {len(display_windows)} of {len(windows)} windows. "
                f"Adjust the date range to zoom in."
            )

        # ── Summary tab ─────────────────────────────────────────────
        with overview_tabs[0]:
            scoring_metrics = sc.get("scoring_metrics", {})

            # ── Fan charts: ~4-year intervals ───────────────────────
            # Pick windows at roughly 4-year (1460-day) intervals from most recent
            if windows:
                # Sort by test_start_date descending
                sorted_wins = sorted(windows, key=lambda w: w.get("test_start_date", ""), reverse=True)
                # Most recent
                picks = [sorted_wins[0]]
                most_recent_date = sorted_wins[0].get("test_start_date", "")

                # Find window ~4 years before most recent
                for w in sorted_wins:
                    d = w.get("test_start_date", "")
                    if d and most_recent_date:
                        try:
                            diff = (pd.Timestamp(most_recent_date) - pd.Timestamp(d)).days
                            if diff >= 1200:  # ~3.3 years minimum gap
                                picks.append(w)
                                # Find another ~4 years before that
                                for w2 in sorted_wins:
                                    d2 = w2.get("test_start_date", "")
                                    if d2:
                                        diff2 = (pd.Timestamp(d) - pd.Timestamp(d2)).days
                                        if diff2 >= 1200:
                                            picks.append(w2)
                                            break
                                break
                        except Exception:
                            pass

                picks.reverse()  # chronological order
                if picks:
                    fan_cols = st.columns(len(picks))
                    for col, pw in zip(fan_cols, picks):
                        sim_pct = pw.get("sim_percentiles")
                        realized = pw.get("realized_prices")
                        if sim_pct and realized:
                            fig_fan = make_fan_chart(
                                sim_pct, realized,
                                title=(f"{pw.get('test_start_date','?')[:4]}: "
                                       f"Score {pw.get('composite_score',0):.3f}"),
                                start_date=pw.get("test_start_date"),
                            )
                            fig_fan.update_layout(height=350)
                            col.plotly_chart(fig_fan, use_container_width=True)

            # ── Aggregated scores across all windows ────────────────
            # Compute weighted averages using recency weights
            wf_data = selected_run["walk_forward"]
            all_weights = [w.get("weight", 1.0) for w in windows]
            weight_sum = sum(all_weights) if all_weights else 1.0

            agg_scores = {}
            for mk in [m["key"] for m in METRIC_INFO]:
                per_window = [w.get("scoring_metrics", {}).get(mk, 0) for w in windows]
                weighted = sum(s * wt for s, wt in zip(per_window, all_weights)) / weight_sum
                agg_scores[mk] = {
                    "weighted_avg": weighted,
                    "mean": float(np.mean(per_window)) if per_window else 0,
                    "std": float(np.std(per_window)) if per_window else 0,
                    "min": float(np.min(per_window)) if per_window else 0,
                    "max": float(np.max(per_window)) if per_window else 0,
                }

            # ── Radar chart (weighted averages) ─────────────────────
            col_radar, col_summary = st.columns([1, 1])

            with col_radar:
                categories = [m["key"] for m in METRIC_INFO]
                values = [agg_scores[k]["weighted_avg"] for k in categories]
                labels = [m["name"] for m in METRIC_INFO]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=labels + [labels[0]],
                    fill="toself",
                    fillcolor="rgba(247,147,26,0.15)",
                    line=dict(color=BTC_ORANGE, width=2),
                    marker=dict(size=6, color=BTC_ORANGE),
                ))
                radar_layout = {k: v for k, v in PLOTLY_LAYOUT.items()
                                if k not in ("xaxis", "yaxis")}
                fig_radar.update_layout(
                    title="Weighted Average Across All Windows",
                    polar=dict(
                        bgcolor=PANEL,
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID,
                                        tickfont=dict(color=TEXT_DIM, size=9)),
                        angularaxis=dict(gridcolor=GRID,
                                         tickfont=dict(color=TEXT, size=10)),
                    ),
                    showlegend=False, height=420, **radar_layout,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with col_summary:
                # Composite headline
                composite_weighted = wf_data.get("weighted_composite", 0)
                st.markdown(
                    f"<div style='background:{PANEL};border:1px solid {GRID};"
                    f"border-radius:6px;padding:12px;margin-bottom:12px;'>"
                    f"<span style='color:{BTC_ORANGE};font-size:1.6rem;font-weight:700;'>"
                    f"{composite_weighted:.4f}</span>"
                    f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
                    f"  weighted composite  |  {len(windows)} windows</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Category scores with range
                cat_groups = {}
                for info in METRIC_INFO:
                    cat = info["category"]
                    if cat not in cat_groups:
                        cat_groups[cat] = []
                    cat_groups[cat].append(info)

                for cat, infos in cat_groups.items():
                    cat_color = CATEGORY_COLORS.get(cat, TEXT_DIM)
                    cat_vals = [agg_scores[m["key"]]["weighted_avg"] for m in infos]
                    cat_avg = float(np.mean(cat_vals))
                    avg_color = score_color(cat_avg)

                    details_parts = []
                    for m in infos:
                        a = agg_scores[m["key"]]
                        details_parts.append(
                            f"{m['name']}: {a['weighted_avg']:.3f} "
                            f"[{a['min']:.2f}-{a['max']:.2f}]"
                        )
                    details = " | ".join(details_parts)

                    st.markdown(
                        f"<div style='padding:6px 0;border-bottom:1px solid {GRID};'>"
                        f"<span style='color:{cat_color};font-weight:600;'>{cat}</span>"
                        f"<span style='color:{avg_color};float:right;font-weight:700;'>"
                        f"{cat_avg:.3f}</span><br>"
                        f"<span style='color:{TEXT_DIM};font-size:0.72rem;'>{details}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # ── Tail Event Summary ───────────────────────────────
            tail = sc.get("distributions", {}).get("tail_events")
            if tail:
                st.divider()
                st.subheader("Tail Event Summary")

                n_paths = tail["n_paths"]
                gen = tail["generic"]
                named = tail["named_scenarios"]

                # Generic thresholds
                col_g1, col_g2, col_g3, col_g4 = st.columns(4)
                for col, (key, label) in zip(
                    [col_g1, col_g2, col_g3, col_g4],
                    [("dd_50pct", "DD >= 50%"),
                     ("dd_75pct", "DD >= 75%"),
                     ("dur_180d", "Duration >= 180d"),
                     ("dur_365d", "Duration >= 365d")],
                ):
                    g = gen[key]
                    col.metric(label, f"{g['count']:,} of {n_paths:,} ({g['pct']}%)")

                # Named stress scenarios
                st.markdown(
                    f"<div style='margin-top:12px;color:{TEXT_DIM};font-size:0.8rem;'>"
                    f"Named stress scenarios — does the model produce known historical crashes?</div>",
                    unsafe_allow_html=True,
                )
                for key, scenario in named.items():
                    count = scenario["count"]
                    pct = scenario["pct"]
                    desc = scenario["desc"]
                    name = key.replace("_", " ").title()

                    if count == 0:
                        color = RED
                        icon = "X"
                        note = " — model cannot produce this known historical outcome"
                    else:
                        color = GREEN
                        icon = "+"
                        note = ""

                    st.markdown(
                        f"<div style='background:{PANEL};border:1px solid {color};"
                        f"border-radius:6px;padding:10px;margin:6px 0;'>"
                        f"<span style='color:{color};font-weight:700;'>[{icon}]</span> "
                        f"<span style='color:#ffffff;font-weight:600;'>{name}</span>"
                        f"<span style='color:{TEXT_DIM};'> — {desc}</span><br>"
                        f"<span style='color:{color};font-size:1.1rem;font-weight:700;'>"
                        f"{count:,} of {n_paths:,} paths ({pct}%)</span>"
                        f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>{note}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # ── Aggregated metric cards ─────────────────────────────
            st.divider()
            current_category = None
            for info in METRIC_INFO:
                if info["category"] != current_category:
                    current_category = info["category"]
                    cat_color = CATEGORY_COLORS.get(current_category, TEXT_DIM)
                    cat_weight = sum(m["weight"] for m in METRIC_INFO
                                     if m["category"] == current_category)
                    cat_metrics = [m for m in METRIC_INFO if m["category"] == current_category]
                    cat_avg = float(np.mean([
                        agg_scores[m["key"]]["weighted_avg"] for m in cat_metrics
                    ]))
                    st.markdown(
                        f'<div class="category-header">'
                        f'<span style="color:{cat_color};">{current_category}</span>'
                        f' &mdash; {cat_weight*100:.0f}% total weight'
                        f' &mdash; avg: {cat_avg:.3f}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                a = agg_scores[info["key"]]
                s_val = a["weighted_avg"]
                color = score_color(s_val)
                pct = max(0, min(100, s_val * 100))
                range_str = f"Range [{a['min']:.3f} - {a['max']:.3f}] | Std: {a['std']:.3f}"

                st.markdown(
                    f"""<div class="metric-card">
                        <div style="display:flex;justify-content:space-between;align-items:baseline;">
                            <h4 style="color:{color};">{info['name']}</h4>
                            <span class="score-val" style="color:{color};">{s_val:.3f}</span>
                        </div>
                        <span class="weight">Weight: {info['weight']*100:.0f}% | Weighted contribution: {s_val*info['weight']:.4f}</span>
                        <div class="score-bar-bg">
                            <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
                        </div>
                        <div style="color:{TEXT_DIM};font-size:0.75rem;">{range_str}</div>
                        <div class="desc">{info['desc']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # ── Distributional Charts ──────────────────────────────────────
            dist_data = sc.get("distributions", {})
            if dist_data:
                st.divider()
                st.subheader("Simulated vs Historical Distributions")
                _di = data.get("data_info", {})
                st.markdown(
                    f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
                    f"Single OOS scorecard — model trained on full training split "
                    f"({_di.get('train_start','?')} to {_di.get('train_end','?')}), "
                    f"simulated {selected_horizon} forward, "
                    f"compared against realized test data "
                    f"({_di.get('test_start','?')} to {_di.get('test_end','?')}). "
                    f"Not aggregated across walk-forward windows. "
                    f"Use <b>Run Detailed Analysis</b> on individual window tabs "
                    f"for per-window distributions."
                    f"</span>", unsafe_allow_html=True,
                )
                _render_distribution_charts(dist_data)

        # ── Per-window tabs ─────────────────────────────────────────
        for tab_i, (win_i, w) in enumerate(display_windows):
            with overview_tabs[tab_i + 1]:  # +1 because Summary is index 0
                st.markdown(
                    f"**Window {w['window_num']}** | "
                    f"Train: {w.get('train_start_date','?')} to {w.get('train_end_date','?')} | "
                    f"Test: {w.get('test_start_date','?')} to {w.get('test_end_date','?')}"
                )
                _render_window_view(w, key_prefix=f"ov_w{w['window_num']}")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Walk-Forward Inspector
# ═══════════════════════════════════════════════════════════════════════
with tab_wf:
    st.header(f"Walk-Forward Inspector -- {selected_model} @ {selected_horizon}")

    wf = selected_run["walk_forward"]
    windows = wf.get("windows", [])

    if not windows:
        st.warning("No walk-forward windows for this combination.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Weighted Composite", f"{wf['weighted_composite']:.4f}")
        col2.metric("Windows", wf["n_windows"])
        col3.metric("Recency Weighting", wf["recency_weighting"])
        col4.metric("Half-Life (days)", wf["recency_half_life_days"])

        st.divider()

        # ── Composite scores per window ─────────────────────────────
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            x=[f"W{w['window_num']}" for w in windows],
            y=[w["composite_score"] for w in windows],
            marker_color=BTC_ORANGE, opacity=0.7, name="Window Composite",
            hovertemplate=(
                "Window %{x}<br>Score: %{y:.4f}<br>"
                "Weight: %{customdata[0]:.4f}<br>"
                "Train: %{customdata[1]} to %{customdata[2]}<br>"
                "Test: %{customdata[3]} to %{customdata[4]}<extra></extra>"
            ),
            customdata=[[w.get("weight",0), w.get("train_start_date","?"),
                         w.get("train_end_date","?"), w.get("test_start_date","?"),
                         w.get("test_end_date","?")] for w in windows],
        ))
        fig_bars.add_hline(y=wf["weighted_composite"], line_dash="dash", line_color=YELLOW,
                           annotation_text=f"Weighted: {wf['weighted_composite']:.4f}",
                           annotation_font_color=YELLOW)
        fig_bars.update_layout(title="Composite Score per Window",
                               xaxis_title="Window", yaxis_title="Score", yaxis_range=[0,1],
                               **PLOTLY_LAYOUT)
        st.plotly_chart(fig_bars, use_container_width=True)

        # ── Weights ─────────────────────────────────────────────────
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(
            x=[f"W{w['window_num']}" for w in windows],
            y=[w.get("weight",0) for w in windows],
            marker_color=BLUE, opacity=0.7, name="Recency Weight",
        ))
        fig_w.update_layout(title="Recency Weights", xaxis_title="Window",
                            yaxis_title="Weight", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_w, use_container_width=True)

        # ── Heatmap (normalized scores) ─────────────────────────────
        st.subheader("Normalized Scores Across Windows")
        metric_names = list(windows[0].get("scoring_metrics", {}).keys())
        hm_data = [[w.get("scoring_metrics", {}).get(m, 0) for m in metric_names] for w in windows]
        hm_labels = [m.replace("_", " ").title() for m in metric_names]
        hm_y = [f"W{w['window_num']}" for w in windows]

        fig_hm = go.Figure(data=go.Heatmap(
            z=hm_data, x=hm_labels, y=hm_y,
            colorscale=[[0,"#1a0500"],[0.5,"#7a3d0d"],[1,BTC_ORANGE]],
            zmin=0, zmax=1, text=np.round(hm_data, 3), texttemplate="%{text}",
            textfont=dict(size=9),
        ))
        hm_ly = {**PLOTLY_LAYOUT}
        hm_ly["yaxis"] = {**hm_ly.get("yaxis",{}), "autorange": "reversed"}
        fig_hm.update_layout(title="Normalized Scores [0-1]", height=max(300, len(windows)*30+100), **hm_ly)
        st.plotly_chart(fig_hm, use_container_width=True)

        # ── Sim vs Actual: any window ───────────────────────────────
        st.subheader("Simulated vs Realized")
        max_w_idx = max(range(len(windows)), key=lambda i: windows[i].get("weight",0))
        win_choice = st.selectbox(
            "Select window",
            range(len(windows)),
            format_func=lambda i: (
                f"W{windows[i]['window_num']} -- "
                f"{windows[i].get('test_start_date','?')} to {windows[i].get('test_end_date','?')} "
                f"(Score: {windows[i].get('composite_score',0):.4f}, "
                f"Weight: {windows[i].get('weight',0):.4f})"
            ),
            index=max_w_idx,
        )
        cw = windows[win_choice]
        cw_pct = cw.get("sim_percentiles")
        cw_real = cw.get("realized_prices")
        if cw_pct and cw_real:
            st.plotly_chart(make_fan_chart(cw_pct, cw_real,
                title=f"Window {cw['window_num']} -- {cw.get('test_start_date','?')} to {cw.get('test_end_date','?')}",
                start_date=cw.get("test_start_date")),
                use_container_width=True)
        else:
            st.info("No percentile data for this window.")

        # ── Per-window raw metric detail (selected window) ──────────
        st.subheader(f"Raw Metric Detail -- Window {cw['window_num']}")
        st.markdown(
            f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
            f"Actual computed values before normalization to [0,1]"
            f"</span>", unsafe_allow_html=True,
        )

        cw_raw = cw.get("raw_metrics", {})
        cw_scores = cw.get("scoring_metrics", {})

        if cw_raw:
            for info in METRIC_INFO:
                key = info["key"]
                raw_data = cw_raw.get(key, {})
                norm_score = cw_scores.get(key, 0)
                display_fields = RAW_METRIC_DISPLAY.get(key, [])
                if not raw_data or not display_fields:
                    continue

                color = score_color(norm_score)
                cat_color = CATEGORY_COLORS.get(info["category"], TEXT_DIM)

                # Build raw values HTML rows
                raw_rows = ""
                for field_key, field_label, fmt, hint in display_fields:
                    val = raw_data.get(field_key)
                    if val is None:
                        val_str = "N/A"
                    elif isinstance(val, int):
                        val_str = f"{val:{fmt}}"
                    else:
                        try:
                            val_str = f"{val:{fmt}}"
                        except (ValueError, TypeError):
                            val_str = str(val)
                    hint_html = f"<span style='color:{TEXT_DIM};font-size:0.7rem;'> {hint}</span>" if hint else ""
                    raw_rows += (
                        f"<div style='display:flex;justify-content:space-between;padding:2px 0;'>"
                        f"<span style='color:{TEXT};font-size:0.8rem;'>{field_label}{hint_html}</span>"
                        f"<span style='color:{TEXT};font-size:0.8rem;font-weight:600;'>{val_str}</span>"
                        f"</div>"
                    )

                pct = max(0, min(100, norm_score * 100))
                st.markdown(
                    f"""<div class="metric-card">
                        <div style="display:flex;justify-content:space-between;align-items:baseline;">
                            <h4 style="color:{cat_color};margin:0;">{info['name']}</h4>
                            <span style="color:{color};font-size:1.1rem;font-weight:700;">
                                Score: {norm_score:.4f}
                            </span>
                        </div>
                        <div class="score-bar-bg" style="margin:6px 0;">
                            <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
                        </div>
                        <div style="border-top:1px solid #1a1a1a;padding-top:8px;margin-top:4px;">
                            {raw_rows}
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "Raw metrics not available. Re-run the experiment to generate raw metric data."
            )

        # ── Window Details table ────────────────────────────────────
        st.subheader("Window Summary Table")
        detail_rows = []
        for w in windows:
            row = {"Window": w["window_num"],
                   "Train": f"{w.get('train_start_date','?')} to {w.get('train_end_date','?')}",
                   "Test": f"{w.get('test_start_date','?')} to {w.get('test_end_date','?')}",
                   "Weight": w.get("weight",0), "Composite": w.get("composite_score",0)}
            for mn, mv in w.get("scoring_metrics", {}).items():
                row[mn.replace("_"," ").title()] = mv
            detail_rows.append(row)
        detail_df = pd.DataFrame(detail_rows)
        num_cols = [c for c in detail_df.columns if c not in ["Window","Train","Test"]]
        st.dataframe(
            detail_df.style.format({c: "{:.4f}" for c in num_cols})
                .background_gradient(subset=["Composite"], cmap="YlOrRd", vmin=0, vmax=1),
            use_container_width=True, height=min(len(detail_rows)*38+50, 600))

        # ── Raw metrics table across all windows ────────────────────
        has_raw = any(w.get("raw_metrics") for w in windows)
        if has_raw:
            st.subheader("Key Raw Values Across Windows")
            # Build a table with one key raw value per metric per window
            raw_summary_rows = []
            for w in windows:
                w_raw = w.get("raw_metrics", {})
                row = {"Window": w["window_num"]}
                # Pick the most informative raw value per metric
                _raw_keys = {
                    "ks_statistic": "ks_stat",
                    "qq_divergence": "normalized_mse",
                    "tail_index_accuracy": "relative_error",
                    "var_backtest_kupiec": "observed_rate_5pct",
                    "drawdown_duration_dist": "sim_median_duration_days",
                    "recovery_time_dist": "sim_median_recovery_days",
                    "time_in_drawdown_ratio": "sim_median_ratio",
                    "vol_clustering_acf": "acf_rmse",
                    "mape": "mape_pct",
                }
                _raw_labels = {
                    "ks_statistic": "KS Stat",
                    "qq_divergence": "QQ Norm MSE",
                    "tail_index_accuracy": "Tail Rel Err",
                    "var_backtest_kupiec": "5% Breach Rate",
                    "drawdown_duration_dist": "Sim Med DD (d)",
                    "recovery_time_dist": "Sim Med Rec (d)",
                    "time_in_drawdown_ratio": "Sim DD Ratio",
                    "vol_clustering_acf": "ACF RMSE",
                    "mape": "MAPE %",
                }
                for metric_key, raw_field in _raw_keys.items():
                    label = _raw_labels[metric_key]
                    val = w_raw.get(metric_key, {}).get(raw_field)
                    row[label] = val if val is not None else float('nan')
                raw_summary_rows.append(row)

            raw_df = pd.DataFrame(raw_summary_rows)
            raw_num_cols = [c for c in raw_df.columns if c != "Window"]
            st.dataframe(
                raw_df.style.format({c: "{:.4f}" for c in raw_num_cols}, na_rep="N/A"),
                use_container_width=True, height=min(len(raw_summary_rows)*38+50, 600))


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: Parameter Optimization
# ═══════════════════════════════════════════════════════════════════════
with tab_opt:
    st.header(f"Parameter Optimization -- {selected_model} @ {selected_horizon}")

    opt = selected_run["optimization"]
    wf_opt = selected_run["walk_forward"]
    windows_opt = wf_opt.get("windows", [])

    # ── GBM: MLE parameter estimates per walk-forward window ──────────
    if opt.get("search_phase") == "mle" or selected_model == "gbm":
        st.markdown(
            f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
            f"GBM parameters estimated from training data via MLE. "
            f"No hyperparameter optimization — mu and sigma are fully determined by the data."
            f"</span>", unsafe_allow_html=True,
        )

        if not windows_opt:
            st.warning("No walk-forward windows for this combination.")
        else:
            # Extract mu, sigma, MAPE from each window
            mus = []
            sigmas = []
            mapes = []
            train_days_list = []

            for w in windows_opt:
                meta = w.get("model_metadata", {})
                fp = w.get("fitted_params", {})
                # New pipeline: mu/sigma are the MLE values directly
                mu = fp.get("mu") or meta.get("mu_annual")
                sigma = fp.get("sigma") or meta.get("sigma_annual")
                mus.append(mu)
                sigmas.append(sigma)
                # MAPE from raw metrics
                mape_pct = w.get("raw_metrics", {}).get("mape", {}).get("mape_pct")
                mapes.append(mape_pct)
                train_days_list.append(
                    w.get("train_end_idx", 0) - w.get("train_start_idx", 0)
                )

            x_dates = [w.get("train_end_date", f"W{w['window_num']}") for w in windows_opt]

            # ── Top metrics ─────────────────────────────────────────
            valid_mus = [m for m in mus if m is not None]
            valid_sigmas = [s for s in sigmas if s is not None]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Weighted Composite", f"{wf_opt['weighted_composite']:.4f}")
            c2.metric("Windows", len(windows_opt))
            if valid_mus:
                c3.metric("Latest mu (ann.)", f"{valid_mus[-1]:.4f}")
            if valid_sigmas:
                c4.metric("Latest sigma (ann.)", f"{valid_sigmas[-1]:.4f}")
            st.divider()

            # ── mu evolution ────────────────────────────────────────
            fig_mu = go.Figure()
            fig_mu.add_trace(go.Scatter(
                x=x_dates, y=mus, mode="markers+lines",
                marker=dict(color=BTC_ORANGE, size=10),
                line=dict(color=BTC_ORANGE, width=2),
                name="mu (annualized)",
                hovertemplate="Train end: %{x}<br>mu: %{y:.4f}<extra></extra>",
            ))
            fig_mu.add_hline(y=0, line_dash="dot", line_color=TEXT_DIM, line_width=1)
            fig_mu.update_layout(
                title="Annualized Drift (mu) -- MLE from Training Data",
                xaxis_title="Training Window End Date",
                yaxis_title="mu (annualized)",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_mu, use_container_width=True)

            # ── sigma evolution ─────────────────────────────────────
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(
                x=x_dates, y=sigmas, mode="markers+lines",
                marker=dict(color=BLUE, size=10),
                line=dict(color=BLUE, width=2),
                name="sigma (annualized)",
                hovertemplate="Train end: %{x}<br>sigma: %{y:.4f}<extra></extra>",
            ))
            fig_sig.update_layout(
                title="Annualized Volatility (sigma) -- MLE from Training Data",
                xaxis_title="Training Window End Date",
                yaxis_title="sigma (annualized)",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_sig, use_container_width=True)

            # ── mu vs sigma scatter (colored by score) ──────────────
            fig_ms = go.Figure()
            composites = [w.get("composite_score", 0) for w in windows_opt]
            fig_ms.add_trace(go.Scatter(
                x=mus, y=sigmas, mode="markers+text",
                marker=dict(
                    color=composites,
                    colorscale=[[0, "#1a0500"], [0.5, "#7a3d0d"], [1, BTC_ORANGE]],
                    size=14, showscale=True,
                    colorbar=dict(title="Score"),
                    line=dict(width=1, color=GRID),
                ),
                text=[f"W{w['window_num']}" for w in windows_opt],
                textposition="top center",
                textfont=dict(color=TEXT_DIM, size=9),
                hovertemplate=(
                    "Window: %{text}<br>"
                    "mu: %{x:.4f}<br>"
                    "sigma: %{y:.4f}<br>"
                    "Score: %{marker.color:.4f}<extra></extra>"
                ),
            ))
            fig_ms.update_layout(
                title="mu vs sigma per Window (colored by composite score)",
                xaxis_title="mu (annualized)",
                yaxis_title="sigma (annualized)",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_ms, use_container_width=True)

            # ── Per-window table ────────────────────────────────────
            st.subheader("Per-Window Results")
            param_rows = []
            for i, w in enumerate(windows_opt):
                param_rows.append({
                    "Window": w["window_num"],
                    "Training Window": (
                        f"{w.get('train_start_date', '?')} to "
                        f"{w.get('train_end_date', '?')}"
                    ),
                    "Testing Window": (
                        f"{w.get('test_start_date', '?')} to "
                        f"{w.get('test_end_date', '?')}"
                    ),
                    "mu (ann)": mus[i],
                    "sigma (ann)": sigmas[i],
                    "MAPE %": mapes[i],
                    "Composite": w.get("composite_score", 0),
                    "Weight": w.get("weight", 0),
                })
            param_df = pd.DataFrame(param_rows)
            float_cols = ["mu (ann)", "sigma (ann)", "MAPE %", "Composite", "Weight"]
            st.dataframe(
                param_df.style.format({c: "{:.4f}" for c in float_cols}, na_rep="N/A")
                    .background_gradient(subset=["Composite"], cmap="YlOrRd", vmin=0, vmax=1),
                use_container_width=True,
                height=min(len(param_rows) * 38 + 50, 600),
            )

    # ── RBB: Sensitivity Analysis with regime diagnostics ─────────
    elif opt.get("search_phase") == "sensitivity":
        bp = opt.get("best_params", {})

        # ── Winner callout ──────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Score", f"{opt['best_score']:.4f}")
        c2.metric("Sampling", bp.get("block_length_sampling", "?"))
        c3.metric("Block Length", f"{bp.get('mean_block_length', '?')}d")
        c4.metric("Transition Matrix", bp.get("transition_matrix_method", "?"))

        st.markdown(
            f"<div style='background:{PANEL};border:1px solid {BTC_ORANGE};border-radius:6px;"
            f"padding:12px;margin:8px 0;'>"
            f"<span style='color:{BTC_ORANGE};font-weight:700;'>Winner:</span> "
            f"<span style='color:{TEXT};'>"
            f"{bp.get('block_length_sampling','?')} sampling, "
            f"mean block = {bp.get('mean_block_length','?')}d, "
            f"transition = {bp.get('transition_matrix_method','?')}, "
            f"variance switching = {bp.get('msm_variance_switching','?')}"
            f"</span>"
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'> | "
            f"{opt['n_trials']} trials evaluated</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        trials = opt.get("trials", [])
        if trials:
            trials_df = pd.DataFrame(trials)
            params_exp = pd.json_normalize(trials_df["params"])
            param_cols = list(params_exp.columns)
            trials_flat = pd.concat([
                trials_df[["number", "score", "phase", "state"]], params_exp
            ], axis=1)

            # ── Phase 1: Block Length Sweep ─────────────────────────
            st.subheader("Phase 1: Block Length Sweep")
            st.markdown(
                f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
                f"Regime switching held at defaults (fitted Markov, variance_switching=True). "
                f"Sweep block lengths 10d-100d for geometric and fixed sampling."
                f"</span>", unsafe_allow_html=True,
            )

            p1 = trials_flat[trials_flat["phase"] == "phase1_block_length"]
            if len(p1) > 0 and "mean_block_length" in p1.columns:
                fig_bl = go.Figure()
                for samp, color in [("geometric", BTC_ORANGE), ("fixed", BLUE)]:
                    mask = p1["block_length_sampling"] == samp
                    if mask.any():
                        sub = p1[mask].sort_values("mean_block_length")
                        fig_bl.add_trace(go.Scatter(
                            x=sub["mean_block_length"], y=sub["score"],
                            mode="markers+lines", name=samp.title(),
                            marker=dict(size=10, color=color),
                            line=dict(color=color, width=2),
                            hovertemplate=(
                                "Block length: %{x}d<br>"
                                "Score: %{y:.4f}<extra></extra>"
                            ),
                        ))
                # Mark Phase 1 winner
                p1_best = p1.loc[p1["score"].idxmax()]
                fig_bl.add_trace(go.Scatter(
                    x=[p1_best["mean_block_length"]], y=[p1_best["score"]],
                    mode="markers", marker=dict(color=GREEN, size=18, symbol="star",
                                                line=dict(width=2, color=TEXT)),
                    name=f"Winner: {p1_best['block_length_sampling']} {int(p1_best['mean_block_length'])}d",
                    showlegend=True,
                ))
                fig_bl.update_layout(
                    title="Score by Block Length",
                    xaxis_title="Mean Block Length (days)",
                    yaxis_title="Composite Score", **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_bl, use_container_width=True)

            # ── Phase 2: Regime Switching Sweep ─────────────────────
            col_p2, col_p3 = st.columns(2)

            with col_p2:
                st.subheader("Phase 2: Regime Sweep")
                st.markdown(
                    f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
                    f"Block length locked at Phase 1 winner. "
                    f"Vary transition matrix method and variance switching."
                    f"</span>", unsafe_allow_html=True,
                )
                p2 = trials_flat[trials_flat["phase"] == "phase2_regime"]
                if len(p2) > 0:
                    p2_sorted = p2.sort_values("score", ascending=False)
                    p2_labels = [
                        f"tm={r.get('transition_matrix_method','?')}\nvar_sw={r.get('msm_variance_switching','?')}"
                        for _, r in p2_sorted.iterrows()
                    ]
                    colors = [GREEN if i == 0 else PURPLE for i in range(len(p2_sorted))]
                    fig_p2 = go.Figure()
                    fig_p2.add_trace(go.Bar(
                        x=p2_labels, y=p2_sorted["score"],
                        marker_color=colors, opacity=0.8,
                        text=[f"{s:.4f}" for s in p2_sorted["score"]],
                        textposition="auto",
                    ))
                    fig_p2.update_layout(
                        title="Score by Regime Configuration",
                        xaxis_title="Configuration", yaxis_title="Score",
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig_p2, use_container_width=True)

            # ── Phase 3: Joint Confirmation ─────────────────────────
            with col_p3:
                st.subheader("Phase 3: Joint Confirmation")
                st.markdown(
                    f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
                    f"Top 3 block configs x Top 3 regime configs = 9 combinations."
                    f"</span>", unsafe_allow_html=True,
                )
                p3 = trials_flat[trials_flat["phase"] == "phase3_joint"]
                if len(p3) > 0:
                    p3_sorted = p3.sort_values("score", ascending=False)
                    p3_labels = [
                        f"{r.get('block_length_sampling','?')[:4]} {int(r.get('mean_block_length',0))}d\n"
                        f"tm={r.get('transition_matrix_method','?')[:4]}"
                        for _, r in p3_sorted.iterrows()
                    ]
                    colors = [GREEN if i == 0 else BTC_ORANGE for i in range(len(p3_sorted))]
                    fig_p3 = go.Figure()
                    fig_p3.add_trace(go.Bar(
                        x=p3_labels, y=p3_sorted["score"],
                        marker_color=colors, opacity=0.8,
                        text=[f"{s:.4f}" for s in p3_sorted["score"]],
                        textposition="auto",
                    ))
                    fig_p3.update_layout(
                        title="Score by Joint Configuration",
                        xaxis_title="Configuration", yaxis_title="Score",
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig_p3, use_container_width=True)

            # Regime diagnostics from walk-forward metadata
            if windows_opt:
                last_w = windows_opt[-1]
                meta = last_w.get("model_metadata", {})
                regime_labels = meta.get("regime_labels")
                tm = meta.get("transition_matrix")

                if tm:
                    st.subheader("Fitted Regime Diagnostics")
                    col_tm, col_pool = st.columns(2)

                    with col_tm:
                        tm_arr = np.array(tm)
                        fig_tm = go.Figure(data=go.Heatmap(
                            z=tm_arr, x=["Bull", "Bear"], y=["Bull", "Bear"],
                            colorscale=[[0, PANEL], [1, BTC_ORANGE]],
                            text=np.round(tm_arr, 4), texttemplate="%{text}",
                            textfont=dict(size=14),
                        ))
                        fig_tm.update_layout(
                            title="Transition Matrix P(row -> col)",
                            height=350, **PLOTLY_LAYOUT,
                        )
                        st.plotly_chart(fig_tm, use_container_width=True)

                    with col_pool:
                        bp = meta.get("bull_pool_size", 0)
                        brp = meta.get("bear_pool_size", 0)
                        fig_pool = go.Figure()
                        fig_pool.add_trace(go.Bar(
                            x=["Bull", "Bear"], y=[bp, brp],
                            marker_color=[GREEN, RED], opacity=0.7,
                            text=[str(bp), str(brp)], textposition="auto",
                        ))
                        fig_pool.update_layout(
                            title="Block Pool Sizes",
                            yaxis_title="Number of Blocks",
                            height=350, **PLOTLY_LAYOUT,
                        )
                        st.plotly_chart(fig_pool, use_container_width=True)

            # All trials table
            st.subheader("All Trials")
            display_cols = ["number", "phase", "score"] + [
                c for c in param_cols
                if c in trials_flat.columns
            ]
            num_fmt = {c: "{:.4f}" for c in param_cols
                       if c in trials_flat.columns
                       and trials_flat[c].dtype in ['float64', 'int64', 'float32', 'int32']}
            num_fmt["score"] = "{:.4f}"
            st.dataframe(
                trials_flat[display_cols].style.format(num_fmt)
                    .background_gradient(subset=["score"], cmap="YlOrRd", vmin=0, vmax=1),
                use_container_width=True,
            )

            # ── Block Length Analysis (per-metric breakdown) ────────
            # Check if trials have per-metric data
            has_metrics = any(t.get("metrics") for t in trials)
            p1_trials = [t for t in trials if t.get("phase") == "phase1_block_length"]

            if has_metrics and p1_trials:
                st.divider()
                st.subheader("Block Length Analysis — Per-Metric Breakdown")

                metric_keys = [m["key"] for m in METRIC_INFO]
                metric_labels = [m["name"] for m in METRIC_INFO]

                for samp_type in ["geometric", "fixed"]:
                    samp_trials = [t for t in p1_trials
                                   if t["params"].get("block_length_sampling") == samp_type
                                   and t.get("metrics")]
                    if not samp_trials:
                        continue

                    samp_trials.sort(key=lambda t: t["params"].get("mean_block_length", 0))
                    block_lengths = [t["params"]["mean_block_length"] for t in samp_trials]
                    bl_labels = [f"bl={bl}" for bl in block_lengths]

                    st.markdown(f"**{samp_type.title()} Sampling**")

                    # Build the transposed table: metrics as rows, block lengths as columns
                    table_data = []
                    for mk, ml in zip(metric_keys, metric_labels):
                        row = {"Metric": ml}
                        vals = []
                        for t in samp_trials:
                            v = t["metrics"].get(mk, 0)
                            vals.append(v)
                            bl = t["params"]["mean_block_length"]
                            row[f"bl={bl}"] = v
                        best_val = max(vals)
                        best_bl = block_lengths[vals.index(best_val)]
                        row["Winner"] = f"bl={best_bl}"
                        table_data.append(row)

                    # Add composite row
                    comp_row = {"Metric": "COMPOSITE"}
                    comp_vals = []
                    for t in samp_trials:
                        bl = t["params"]["mean_block_length"]
                        comp_row[f"bl={bl}"] = t["score"]
                        comp_vals.append(t["score"])
                    best_comp = max(comp_vals)
                    comp_row["Winner"] = f"bl={block_lengths[comp_vals.index(best_comp)]}"
                    table_data.append(comp_row)

                    bl_df = pd.DataFrame(table_data)
                    bl_num_cols = [c for c in bl_df.columns if c.startswith("bl=")]
                    st.dataframe(
                        bl_df.style.format({c: "{:.4f}" for c in bl_num_cols})
                            .highlight_max(subset=bl_num_cols, axis=1,
                                           props="font-weight: bold; color: #10b981;"),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Line chart: score vs block length per metric
                    fig_lines = go.Figure()
                    colors = [BTC_ORANGE, BLUE, GREEN, RED, PURPLE,
                              YELLOW, TEXT, "#ff6b6b", "#4ecdc4"]
                    for idx, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
                        vals = [t["metrics"].get(mk, 0) for t in samp_trials]
                        fig_lines.add_trace(go.Scatter(
                            x=block_lengths, y=vals,
                            mode="lines+markers",
                            name=ml,
                            line=dict(color=colors[idx % len(colors)], width=2),
                            marker=dict(size=6),
                        ))
                    # Composite as thick dashed
                    comp_line = [t["score"] for t in samp_trials]
                    fig_lines.add_trace(go.Scatter(
                        x=block_lengths, y=comp_line,
                        mode="lines+markers",
                        name="COMPOSITE",
                        line=dict(color="#ffffff", width=3, dash="dash"),
                        marker=dict(size=8, symbol="diamond"),
                    ))
                    fig_lines.update_layout(
                        title=f"Score vs Block Length ({samp_type.title()})",
                        xaxis_title="Mean Block Length (days)",
                        yaxis_title="Score",
                        yaxis_range=[0.4, 1.05],
                        height=450,
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig_lines, use_container_width=True)

            elif p1_trials and not has_metrics:
                st.divider()
                st.info(
                    "Per-metric breakdown not available for this run. "
                    "Re-run the experiment to generate detailed block length analysis."
                )

    # ── Other models: full Optuna optimization landscape ────────────
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Score", f"{opt['best_score']:.4f}")
        c2.metric("Trials", opt["n_trials"])
        c3.metric("Search Phase", opt["search_phase"])
        st.markdown(f"**Best Parameters:** `{opt['best_params']}`")
        st.divider()

        trials = opt.get("trials", [])
        if not trials:
            st.warning("No trial data.")
        else:
            trials_df = pd.DataFrame(trials)
            params_exp = pd.json_normalize(trials_df["params"])
            param_cols = list(params_exp.columns)
            trials_flat = pd.concat([trials_df[["number","score","state"]], params_exp], axis=1)
            numeric_params = [c for c in param_cols if trials_flat[c].dtype in ['float64','int64','float32','int32']]

            # Score by trial
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=trials_flat["number"], y=trials_flat["score"],
                mode="markers+lines",
                marker=dict(color=trials_flat["score"],
                            colorscale=[[0,"#1a0500"],[0.5,"#7a3d0d"],[1,BTC_ORANGE]],
                            size=10, showscale=True, colorbar=dict(title="Score")),
                line=dict(color=TEXT_DIM, width=1),
                hovertemplate="Trial %{x}<br>Score: %{y:.4f}<extra></extra>",
            ))
            fig_t.add_hline(y=opt["best_score"], line_dash="dash", line_color=GREEN,
                            annotation_text=f"Best: {opt['best_score']:.4f}",
                            annotation_font_color=GREEN)
            fig_t.update_layout(title="Score by Trial", xaxis_title="Trial", yaxis_title="Score",
                                yaxis_range=[0, max(1, max(trials_flat['score'])*1.1)], **PLOTLY_LAYOUT)
            st.plotly_chart(fig_t, use_container_width=True)

            # Parameter landscape
            if len(numeric_params) >= 2:
                st.subheader("Parameter Landscape")
                ca, cb = st.columns(2)
                with ca:
                    x_p = st.selectbox("X axis", numeric_params, index=0, key="opt_x")
                with cb:
                    y_p = st.selectbox("Y axis", numeric_params, index=min(1,len(numeric_params)-1), key="opt_y")

                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(
                    x=trials_flat[x_p], y=trials_flat[y_p], mode="markers",
                    marker=dict(color=trials_flat["score"],
                                colorscale=[[0,"#1a0500"],[0.5,"#7a3d0d"],[1,BTC_ORANGE]],
                                size=12, showscale=True, colorbar=dict(title="Score"),
                                line=dict(width=1, color=GRID)),
                    hovertemplate=f"{x_p}: %{{x:.4f}}<br>{y_p}: %{{y:.4f}}<br>Score: %{{marker.color:.4f}}<extra></extra>",
                ))
                best_i = trials_flat["score"].idxmax()
                fig_s.add_trace(go.Scatter(
                    x=[trials_flat.loc[best_i, x_p]], y=[trials_flat.loc[best_i, y_p]],
                    mode="markers", marker=dict(color=GREEN, size=18, symbol="star",
                                                line=dict(width=2, color=TEXT)),
                    name="Best", showlegend=True))
                fig_s.update_layout(title=f"{x_p} vs {y_p}", xaxis_title=x_p, yaxis_title=y_p, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_s, use_container_width=True)

            # Per-param
            st.subheader("Score by Parameter Value")
            for pc in param_cols:
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(
                    x=trials_flat[pc], y=trials_flat["score"], mode="markers",
                    marker=dict(color=BTC_ORANGE, size=8, opacity=0.7)))
                fig_p.update_layout(title=f"Score vs {pc}", xaxis_title=pc, yaxis_title="Score",
                                    yaxis_range=[0,1], height=300, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_p, use_container_width=True)

            # Trial table
            st.subheader("All Trials")
            st.dataframe(
                trials_flat.style.format({"score":"{:.4f}", **{c:"{:.4f}" for c in numeric_params}})
                    .background_gradient(subset=["score"], cmap="YlOrRd", vmin=0, vmax=1),
                use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: Pipeline Audit
# ═══════════════════════════════════════════════════════════════════════
with tab_audit:
    st.header(f"Pipeline Audit -- {selected_model} @ {selected_horizon}")
    st.markdown(
        f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
        f"Step-by-step pipeline audit for a single walk-forward window. "
        f"Green = sanity check passed. Red = investigate."
        f"</span>", unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_audit_deps():
        from validation.audit import run_audit, audit_to_markdown, audit_to_signal
        from data.loader import BTCDataLoader, get_price_array
        return run_audit, audit_to_markdown, audit_to_signal, BTCDataLoader, get_price_array

    try:
        _run_audit, _audit_to_md, _audit_to_signal, _AuditLoader, _audit_get_px = load_audit_deps()

        wf_audit = selected_run["walk_forward"]
        audit_windows = wf_audit.get("windows", [])

        if not audit_windows:
            st.warning("No walk-forward windows available.")
        else:
            audit_win_idx = st.selectbox(
                "Select Window",
                range(len(audit_windows)),
                format_func=lambda i: (
                    f"W{audit_windows[i]['window_num']} | "
                    f"Test: {audit_windows[i].get('test_start_date','?')} to "
                    f"{audit_windows[i].get('test_end_date','?')} | "
                    f"Score: {audit_windows[i].get('composite_score',0):.4f}"
                ),
                index=len(audit_windows) - 1,
                key="audit_window",
            )
            aw = audit_windows[audit_win_idx]
            horizon_days = int(selected_horizon.replace("d", ""))

            # Key audit result by model+horizon+window so it invalidates on change
            audit_key = f"audit_{selected_model}_{selected_horizon}_W{aw['window_num']}"

            # Clear stale audit if selection changed
            if st.session_state.get("_audit_key") != audit_key:
                st.session_state.pop("audit_result", None)
                st.session_state["_audit_key"] = audit_key

            if st.button("Run Audit", type="primary", key="run_audit"):
                with st.spinner(f"Running full pipeline audit for {selected_model} @ {selected_horizon} W{aw['window_num']}..."):
                    # Load data
                    loader = _AuditLoader()
                    train_df, test_df = loader.get_train_test_split(train_pct=0.7)
                    full_returns = np.concatenate([
                        loader.get_returns_array(train_df),
                        loader.get_returns_array(test_df),
                    ])
                    full_prices = np.concatenate([
                        _audit_get_px(train_df),
                        _audit_get_px(test_df)[1:],
                    ])

                    # Extract window data
                    t_end = aw.get("train_end_idx", 730)
                    t_start_test = aw.get("test_start_idx", t_end)
                    t_end_test = aw.get("test_end_idx", t_start_test + horizon_days)

                    train_ret = full_returns[:t_end]
                    train_px = full_prices[:t_end + 1]
                    test_px = full_prices[t_start_test:t_end_test + 1]
                    test_ret = full_returns[t_start_test:t_end_test]

                    # Get model params from the experiment
                    model_params = aw.get("fitted_params") or selected_run.get(
                        "optimization", {}
                    ).get("best_params", {})

                    audit = _run_audit(
                        model_name=selected_model,
                        train_returns=train_ret,
                        train_prices=train_px,
                        test_prices=test_px,
                        test_returns=test_ret,
                        horizon_days=horizon_days,
                        n_simulations=2000,
                        seed=42,
                        train_start_date=aw.get("train_start_date"),
                        train_end_date=aw.get("train_end_date"),
                        test_start_date=aw.get("test_start_date"),
                        test_end_date=aw.get("test_end_date"),
                        model_params=model_params if model_params else None,
                    )

                st.session_state["audit_result"] = audit
                st.session_state["_audit_key"] = audit_key

            # Render audit results if available
            audit = st.session_state.get("audit_result")
            if audit:
                # ── Header: checks summary ──────────────────────────
                passed = audit["checks_passed"]
                total = audit["checks_total"]
                all_pass = passed == total

                st.markdown(
                    f"<div style='background:{PANEL};border:2px solid "
                    f"{GREEN if all_pass else RED};border-radius:8px;padding:16px;"
                    f"margin:12px 0;text-align:center;'>"
                    f"<span style='font-size:1.6rem;font-weight:700;"
                    f"color:{GREEN if all_pass else RED};'>"
                    f"{passed}/{total} checks passed</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Section status bar
                _regime_on = audit.get("sections", {}).get(
                    "regime_classification", {}
                ).get("regime_enabled", False)

                section_names = {"data_selection": "1. Data", "block_pools": "2. Pools"}
                if _regime_on:
                    section_names["regime_classification"] = "3. Regime"
                section_names["resampling_verification"] = "4. Resample"
                section_names["simulation_output"] = "5. Sim"
                section_names["simulation_convergence"] = "5b. Converge"
                if _regime_on:
                    section_names["regime_transitions_sim"] = "5c. Transitions"
                section_names["scoring"] = "6. Scoring"
                section_names["overfitting_check"] = "6b. Overfit"
                section_names["baseline_comparison"] = "6c. Baseline"
                status_cols = st.columns(len(section_names))
                for col, (key, label) in zip(status_cols, section_names.items()):
                    passed_s = audit["section_pass"].get(key, True)
                    icon = "+" if passed_s else "X"
                    color = GREEN if passed_s else RED
                    col.markdown(
                        f"<div style='text-align:center;padding:4px;'>"
                        f"<span style='color:{color};font-size:1.2rem;font-weight:700;'>"
                        f"{icon}</span><br>"
                        f"<span style='color:{TEXT_DIM};font-size:0.65rem;'>{label}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                st.divider()
                sections = audit["sections"]

                # ── Section 1: Data Selection ───────────────────────
                ds = sections.get("data_selection", {})
                with st.expander(
                    f"{'[PASS]' if audit['section_pass'].get('data_selection') else '[FAIL]'} "
                    f"Section 1: Data Selection", expanded=False
                ):
                    ts = ds.get("train_stats", {})
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Training Days", f"{ds.get('train_days',0):,}")
                    c2.metric("Training Years", ds.get("train_years", 0))
                    c3.metric("Test Days", ds.get("test_days", 0))
                    c4.metric("Total Return", f"{ts.get('total_return_pct',0):.0f}%")

                    # Sparkline
                    spark_prices = ds.get("train_prices", [])
                    if spark_prices:
                        fig_spark = go.Figure()
                        fig_spark.add_trace(go.Scatter(
                            y=spark_prices, mode="lines",
                            line=dict(color=BTC_ORANGE, width=1.5),
                        ))
                        fig_spark.update_layout(
                            height=150, showlegend=False,
                            yaxis_type="log",
                            margin=dict(l=40, r=10, t=10, b=10),
                            **{k: v for k, v in PLOTLY_LAYOUT.items()
                               if k not in ("margin",)},
                        )
                        st.plotly_chart(fig_spark, use_container_width=True)

                    for flag in ds.get("flags", []):
                        st.error(flag)

                # ── Section 2: Regime Classification ────────────────
                rc = sections.get("regime_classification", {})
                with st.expander(
                    f"{'[PASS]' if audit['section_pass'].get('regime_classification') else '[FAIL]'} "
                    f"Section 2: Regime Classification",
                    expanded=bool(rc.get("flags")),
                ):
                    if rc.get("transition_matrix"):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("MSM Converged", "Yes" if rc.get("msm_converged") else "No")
                        c2.metric("Bull Days", f"{rc.get('bull_days',0)} ({rc.get('bull_pct',0)}%)")
                        c3.metric("Bear Days", f"{rc.get('bear_days',0)} ({rc.get('bear_pct',0)}%)")
                        c4.metric("Avg Bear Run", f"{rc.get('avg_bear_run_days',0)}d")

                        # Transition matrix
                        tm = np.array(rc["transition_matrix"])
                        fig_tm = go.Figure(data=go.Heatmap(
                            z=tm, x=["Bull", "Bear"], y=["Bull", "Bear"],
                            colorscale=[[0, PANEL], [1, BTC_ORANGE]],
                            text=np.round(tm, 4), texttemplate="%{text}",
                            textfont=dict(size=14),
                        ))
                        fig_tm.update_layout(
                            title="Transition Matrix", height=300,
                            **PLOTLY_LAYOUT,
                        )
                        st.plotly_chart(fig_tm, use_container_width=True)

                    for flag in rc.get("flags", []):
                        st.error(flag)

                # ── Section 3: Block Pools ──────────────────────────
                bp = sections.get("block_pools", {})
                with st.expander(
                    f"{'[PASS]' if audit['section_pass'].get('block_pools') else '[FAIL]'} "
                    f"Section 3: Block Pool Construction",
                    expanded=bool(bp.get("flags")),
                ):
                    if bp.get("bull_pool_size") is not None:
                        c1, c2 = st.columns(2)
                        c1.metric("Bull Pool", f"{bp.get('bull_pool_size',0)} blocks (median {bp.get('bull_median_len',0)}d)")
                        c2.metric("Bear Pool", f"{bp.get('bear_pool_size',0)} blocks (median {bp.get('bear_median_len',0)}d)")
                        st.markdown(
                            f"Sampling: **{bp.get('block_length_sampling')}** | "
                            f"Mean length: **{bp.get('mean_block_length')}d** | "
                            f"Stride: **{bp.get('block_stride')}d**"
                        )
                    for flag in bp.get("flags", []):
                        st.error(flag)

                # ── Section 3b: Resampling ──────────────────────────
                rv = sections.get("resampling_verification", {})
                with st.expander(
                    f"{'[PASS]' if audit['section_pass'].get('resampling_verification') else '[FAIL]'} "
                    f"Section 3b: Resampling Verification",
                    expanded=bool(rv.get("flags")),
                ):
                    if rv.get("bull_blocks_used") is not None:
                        c1, c2 = st.columns(2)
                        c1.metric("Bull Utilization",
                                  f"{rv.get('bull_blocks_used',0)}/{rv.get('bull_pool_total',0)} "
                                  f"({rv.get('bull_utilization_pct',0)}%)")
                        c2.metric("Bear Utilization",
                                  f"{rv.get('bear_blocks_used',0)}/{rv.get('bear_pool_total',0)} "
                                  f"({rv.get('bear_utilization_pct',0)}%)")

                        # Block usage heatmaps
                        for label in ["bull", "bear"]:
                            usage = rv.get(f"{label}_usage_counts", [])
                            if usage:
                                fig_u = go.Figure()
                                fig_u.add_trace(go.Bar(
                                    y=usage, marker_color=GREEN if label == "bull" else RED,
                                    opacity=0.7,
                                ))
                                fig_u.update_layout(
                                    title=f"{label.title()} Block Usage",
                                    xaxis_title="Block Index", yaxis_title="Times Sampled",
                                    height=200, **PLOTLY_LAYOUT,
                                )
                                st.plotly_chart(fig_u, use_container_width=True)

                    for flag in rv.get("flags", []):
                        st.error(flag)

                # ── Section 4: Simulation Output ────────────────────
                so = sections.get("simulation_output", {})
                with st.expander(
                    f"{'[PASS]' if audit['section_pass'].get('simulation_output') else '[FAIL]'} "
                    f"Section 4: Simulation Output",
                    expanded=True,
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Median Final", f"${so.get('median_final_price',0):,.0f}")
                    c2.metric("Actual Final", f"${so.get('actual_final_price',0):,.0f}")
                    c3.metric("90% CI Coverage", f"{so.get('ci_coverage_pct',0):.0f}%")
                    c4.metric("50%+ DD Paths", f"{so.get('pct_50_drawdown',0):.0f}%")

                    # Sample paths chart
                    sample = so.get("sample_paths", [])
                    realized = so.get("realized_prices", [])
                    p5_line = so.get("p5", [])
                    p95_line = so.get("p95", [])
                    if sample and realized:
                        start_d = aw.get("test_start_date")
                        n_pts = len(realized)
                        try:
                            x_ax = pd.date_range(start=start_d, periods=n_pts, freq="D").tolist() if start_d else list(range(n_pts))
                        except Exception:
                            x_ax = list(range(n_pts))

                        fig_paths = go.Figure()
                        # 90% CI band
                        if p5_line and p95_line:
                            fig_paths.add_trace(go.Scatter(
                                x=x_ax + x_ax[::-1],
                                y=p5_line[:n_pts] + p95_line[:n_pts][::-1],
                                fill="toself", fillcolor="rgba(247,147,26,0.08)",
                                line=dict(color="rgba(0,0,0,0)"), name="90% CI",
                            ))
                        # Sample paths
                        for i, path in enumerate(sample[:10]):
                            fig_paths.add_trace(go.Scatter(
                                x=x_ax, y=path[:n_pts], mode="lines",
                                line=dict(color=BTC_ORANGE, width=0.7),
                                opacity=0.4, showlegend=i == 0, name="Sim Paths",
                            ))
                        # Realized
                        fig_paths.add_trace(go.Scatter(
                            x=x_ax, y=realized, mode="lines",
                            line=dict(color=GREEN, width=3), name="Realized",
                        ))
                        fig_paths.update_layout(
                            title="Sample Paths vs Realized",
                            yaxis_type="log", height=450,
                            **PLOTLY_LAYOUT,
                        )
                        st.plotly_chart(fig_paths, use_container_width=True)

                    # Stats table
                    stats_data = {
                        "Metric": ["Median Final", "5th Pct Final", "95th Pct Final",
                                   "Below Start", "50%+ DD", "75%+ DD",
                                   "Max DD (all paths)", "90% CI Coverage"],
                        "Value": [
                            f"${so.get('median_final_price',0):,.0f}",
                            f"${so.get('p5_final_price',0):,.0f}",
                            f"${so.get('p95_final_price',0):,.0f}",
                            f"{so.get('pct_below_start',0):.1f}%",
                            f"{so.get('pct_50_drawdown',0):.1f}%",
                            f"{so.get('pct_75_drawdown',0):.1f}%",
                            f"{so.get('max_drawdown_all_paths',0):.1f}%",
                            f"{so.get('ci_coverage_pct',0):.1f}%",
                        ],
                    }
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

                    for flag in so.get("flags", []):
                        st.error(flag)

                # ── Section 4b: Convergence ─────────────────────────
                sc_conv = sections.get("simulation_convergence", {})
                with st.expander(
                    f"{'[PASS]' if audit['section_pass'].get('simulation_convergence') else '[FAIL]'} "
                    f"Section 4b: Simulation Convergence",
                    expanded=bool(sc_conv.get("flags")),
                ):
                    scores_by = sc_conv.get("scores_by_path_count", {})
                    if scores_by:
                        fig_conv = go.Figure()
                        fig_conv.add_trace(go.Scatter(
                            x=list(scores_by.keys()),
                            y=list(scores_by.values()),
                            mode="markers+lines",
                            marker=dict(color=BTC_ORANGE, size=10),
                            line=dict(color=BTC_ORANGE, width=2),
                        ))
                        fig_conv.update_layout(
                            title="Composite Score vs Path Count",
                            xaxis_title="Number of Paths",
                            yaxis_title="Composite Score",
                            height=300, **PLOTLY_LAYOUT,
                        )
                        st.plotly_chart(fig_conv, use_container_width=True)

                        if sc_conv.get("converged_note"):
                            st.success(sc_conv["converged_note"])
                    for flag in sc_conv.get("flags", []):
                        st.warning(flag)

                # ── Section 4c: Regime Transitions ──────────────────
                rt_sim = sections.get("regime_transitions_sim", {})
                if rt_sim and rt_sim.get("expected_bull_to_bear_pct"):
                    with st.expander(
                        f"{'[PASS]' if audit['section_pass'].get('regime_transitions_sim') else '[FAIL]'} "
                        f"Section 4c: Regime Transitions in Simulation",
                        expanded=bool(rt_sim.get("flags")),
                    ):
                        c1, c2 = st.columns(2)
                        c1.metric("Expected Bull->Bear",
                                  f"{rt_sim.get('expected_bull_to_bear_pct',0):.2f}%",
                                  f"Actual: {rt_sim.get('actual_bull_to_bear_pct',0):.2f}%")
                        c2.metric("Expected Bear->Bull",
                                  f"{rt_sim.get('expected_bear_to_bull_pct',0):.2f}%",
                                  f"Actual: {rt_sim.get('actual_bear_to_bull_pct',0):.2f}%")

                        c3, c4 = st.columns(2)
                        c3.metric("Sim Avg Bull Run",
                                  f"{rt_sim.get('avg_sim_bull_run',0):.0f}d",
                                  f"Hist: {rt_sim.get('avg_hist_bull_run',0):.0f}d")
                        c4.metric("Sim Avg Bear Run",
                                  f"{rt_sim.get('avg_sim_bear_run',0):.0f}d",
                                  f"Hist: {rt_sim.get('avg_hist_bear_run',0):.0f}d")

                        for flag in rt_sim.get("flags", []):
                            st.warning(flag)

                # ── Section 5: Scoring ──────────────────────────────
                sc_score = sections.get("scoring", {})
                with st.expander(
                    f"{'[PASS]' if audit['section_pass'].get('scoring') else '[FAIL]'} "
                    f"Section 5: Scoring Breakdown",
                    expanded=True,
                ):
                    st.metric("Composite Score", f"{sc_score.get('composite_score',0):.4f}")
                    metrics = sc_score.get("metrics", {})
                    if metrics:
                        rows = []
                        for name, m in metrics.items():
                            rows.append({
                                "Metric": name.replace("_", " ").title(),
                                "Score": m["normalized"],
                                "Weight": m["weight"],
                                "Contribution": m["contribution"],
                            })
                        mdf = pd.DataFrame(rows)
                        st.dataframe(
                            mdf.style.format({"Score": "{:.4f}", "Weight": "{:.2f}",
                                              "Contribution": "{:.4f}"})
                                .background_gradient(subset=["Score"], cmap="YlOrRd", vmin=0, vmax=1),
                            use_container_width=True, hide_index=True,
                        )
                    for flag in sc_score.get("flags", []):
                        st.error(flag)

                # ── Section 5b: Overfitting ─────────────────────────
                ov = sections.get("overfitting_check", {})
                with st.expander(
                    f"{'[PASS]' if audit['section_pass'].get('overfitting_check') else '[FAIL]'} "
                    f"Section 5b: Overfitting Check",
                    expanded=bool(ov.get("flags")),
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("In-Sample Score", f"{ov.get('in_sample_score',0):.4f}")
                    c2.metric("Out-of-Sample Score", f"{ov.get('out_of_sample_score',0):.4f}")
                    c3.metric("Delta", f"{ov.get('delta',0):+.4f}")
                    for flag in ov.get("flags", []):
                        st.error(flag)

                # ── Section 5c: Baseline Comparison ─────────────────
                bl = sections.get("baseline_comparison", {})
                if bl.get("gbm_composite"):
                    with st.expander(
                        f"{'[PASS]' if audit['section_pass'].get('baseline_comparison') else '[FAIL]'} "
                        f"Section 5c: Baseline Comparison (vs GBM)",
                        expanded=True,
                    ):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("GBM", f"{bl['gbm_composite']:.4f}")
                        c2.metric("RBB", f"{bl['rbb_composite']:.4f}")
                        delta = bl["delta"]
                        c3.metric("Delta", f"{delta:+.4f} ({bl['delta_pct']:+.1f}%)")

                        per_m = bl.get("per_metric", {})
                        if per_m:
                            rows = []
                            for name, d in per_m.items():
                                rows.append({
                                    "Metric": name.replace("_", " ").title(),
                                    "RBB": d["rbb"],
                                    "GBM": d["gbm"],
                                    "Delta": d["delta"],
                                    "Winner": "RBB" if d["beats_gbm"] else "GBM",
                                })
                            bdf = pd.DataFrame(rows)
                            st.dataframe(
                                bdf.style.format({"RBB": "{:.4f}", "GBM": "{:.4f}",
                                                  "Delta": "{:+.4f}"}),
                                use_container_width=True, hide_index=True,
                            )

                        for flag in bl.get("flags", []):
                            st.error(flag)

                # ── Red Flag Summary ────────────────────────────────
                all_flags = audit.get("red_flags", [])
                if all_flags:
                    st.divider()
                    st.subheader(f"Red Flags ({len(all_flags)})")
                    for flag in all_flags:
                        st.error(flag)

                # ── Export ──────────────────────────────────────────
                st.divider()
                _audit_date = audit.get("timestamp", "")[:10]
                exp_c1, exp_c2, exp_c3 = st.columns(3)
                with exp_c1:
                    signal_text = _audit_to_signal(audit)
                    st.download_button(
                        "Signal (condensed)",
                        data=signal_text,
                        file_name=f"signal_{selected_model}_W{aw['window_num']}_{selected_horizon}_{_audit_date}.md",
                        mime="text/markdown",
                    )
                with exp_c2:
                    md_text = _audit_to_md(audit)
                    st.download_button(
                        "Full Markdown",
                        data=md_text,
                        file_name=f"audit_{selected_model}_W{aw['window_num']}_{selected_horizon}_{_audit_date}.md",
                        mime="text/markdown",
                    )
                with exp_c3:
                    audit_json = json.dumps(audit, indent=2, default=str)
                    st.download_button(
                        "Full JSON",
                        data=audit_json,
                        file_name=f"audit_{selected_model}_W{aw['window_num']}_{selected_horizon}_{_audit_date}.json",
                        mime="application/json",
                    )

            # ── Signal Export (from stored results) ─────────────────
            st.divider()
            st.subheader("Signal Export")
            st.markdown(
                f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
                f"Condensed summary of all scored results from the experiment. "
                f"Reports actual experiment scores — no re-simulation. "
                f"Designed for LLM context windows."
                f"</span>", unsafe_allow_html=True,
            )

            if st.button("Build Signal", type="primary", key="build_signal"):
                _exp_config = data.get("config", {})
                _exp_sim = _exp_config.get("simulation", {})
                _exp_wf_cfg = _exp_config.get("walk_forward", {})
                _exp_locked = _exp_config.get("locked_rbb_params", {})
                _exp_n_sims = _exp_sim.get("n_simulations_final", "?")
                _exp_mode = data.get("run_mode", "?")
                _exp_name = data.get("experiment_name", "?")
                _exp_di = data.get("data_info", {})
                _cross_hz = data.get("cross_horizon", {})
                _scoring_weights = _exp_config.get("scoring", {}).get("weights", {})
                _signal_date = datetime.now().strftime("%Y-%m-%d")

                sig = []
                sig.append(f"# Experiment Signal: {_exp_name} ({_signal_date})")
                sig.append("")
                sig.append("## Experiment Context")
                sig.append(f"- **Mode:** {_exp_mode}")
                sig.append(f"- **Simulations:** {_exp_n_sims} paths per walk-forward window")
                sig.append(
                    f"- **Walk-forward:** step={_exp_wf_cfg.get('step_size_days', '?')}d, "
                    f"min_train={_exp_wf_cfg.get('min_training_days', '?')}d, "
                    f"recency={_exp_wf_cfg.get('recency_weighting', '?')} "
                    f"(half-life={_exp_wf_cfg.get('recency_half_life_days', '?')}d)"
                )
                sig.append(
                    f"- **Data:** train {_exp_di.get('train_start','?')} to "
                    f"{_exp_di.get('train_end','?')} ({_exp_di.get('train_days','?')}d) | "
                    f"test {_exp_di.get('test_start','?')} to "
                    f"{_exp_di.get('test_end','?')} ({_exp_di.get('test_days','?')}d)"
                )
                if _exp_locked:
                    sig.append(
                        f"- **Locked RBB params:** "
                        + ", ".join(
                            f"{k}={v}" for k, v in _exp_locked.items()
                            if k not in ("transition_matrix_method", "msm_variance_switching", "msm_frequency")
                            or _exp_locked.get("regime_enabled", True)
                        )
                    )
                sig.append(
                    f"- **Scoring weights:** "
                    + ", ".join(f"{k}={v}" for k, v in _scoring_weights.items())
                )
                sig.append("")

                # Cross-horizon final scores
                if _cross_hz:
                    sig.append("## Cross-Horizon Final Scores")
                    _hz_w = list(_cross_hz.values())[0].get("horizon_weights", {}) if _cross_hz else {}
                    if _hz_w:
                        sig.append(
                            "Horizon weights: "
                            + ", ".join(f"{h}={w:.0%}" for h, w in _hz_w.items())
                        )
                    sig.append("")
                    sig.append("| Model | Final | Composite | Penalty | " +
                               " | ".join(sorted(_hz_w.keys())) + " | Std |")
                    sig.append("|" + "---|" * (5 + len(_hz_w)))
                    for m_name, ch in sorted(_cross_hz.items(),
                                             key=lambda x: x[1].get("final_score", 0),
                                             reverse=True):
                        ph = ch.get("per_horizon", {})
                        ph_vals = " | ".join(
                            f"{ph.get(h, 0):.4f}" for h in sorted(_hz_w.keys())
                        )
                        sig.append(
                            f"| {m_name} | **{ch.get('final_score', 0):.4f}** | "
                            f"{ch.get('composite', 0):.4f} | "
                            f"{ch.get('stability_penalty', 0):.4f} | "
                            f"{ph_vals} | {ch.get('horizon_std', 0):.4f} |"
                        )
                    sig.append("")

                # GBM lookup for deltas
                _gbm_wf = {}
                for r in all_runs:
                    if r["model"] == "gbm":
                        _gbm_wf[r["horizon"]] = r["walk_forward"]["weighted_composite"]

                # Per-model, per-horizon detail
                for m_name in sorted(set(r["model"] for r in all_runs)):
                    model_runs = [r for r in all_runs if r["model"] == m_name]
                    model_runs.sort(key=lambda r: int(r["horizon"].replace("d", "")))

                    sig.append(f"## {m_name.replace('_', ' ').title()}")
                    sig.append("")

                    for run in model_runs:
                        hz = run["horizon"]
                        wf = run["walk_forward"]
                        windows = wf.get("windows", [])
                        wf_comp = wf["weighted_composite"]
                        gbm_comp = _gbm_wf.get(hz, 0)
                        delta = wf_comp - gbm_comp if m_name != "gbm" else 0

                        # Per-window composites for stats
                        w_scores = [w["composite_score"] for w in windows]

                        sig.append(f"### {hz} — WF composite: {wf_comp:.4f}" + (
                            f" (vs GBM: {delta:+.4f})" if m_name != "gbm" else ""
                        ))
                        sig.append(f"- Windows: {len(windows)}, "
                                   f"range: [{min(w_scores):.4f}, {max(w_scores):.4f}], "
                                   f"std: {float(np.std(w_scores)):.4f}")

                        # Aggregate scoring metrics across windows
                        metric_agg = {}
                        for w in windows:
                            for mk, mv in w.get("scoring_metrics", {}).items():
                                if mk not in metric_agg:
                                    metric_agg[mk] = []
                                metric_agg[mk].append(mv)

                        if metric_agg:
                            sig.append("")
                            sig.append("| Metric | Mean | Min | Max | Weight |")
                            sig.append("|--------|------|-----|-----|--------|")
                            for mk in sorted(metric_agg.keys()):
                                vals = metric_agg[mk]
                                w = _scoring_weights.get(mk, 0)
                                sig.append(
                                    f"| {mk} | {np.mean(vals):.4f} | "
                                    f"{min(vals):.4f} | {max(vals):.4f} | {w:.2f} |"
                                )

                        # Flag weak windows (bottom 3)
                        worst = sorted(enumerate(windows),
                                       key=lambda x: x[1]["composite_score"])[:3]
                        sig.append("")
                        sig.append("Weakest windows:")
                        for idx, w in worst:
                            sig.append(
                                f"- W{w['window_num']} "
                                f"{w.get('test_start_date', '?')} to "
                                f"{w.get('test_end_date', '?')}: "
                                f"{w['composite_score']:.4f} "
                                f"(weight={w['weight']:.4f})"
                            )
                        sig.append("")

                # Params used
                sig.append("## Configuration per Window (verification)")
                for run in all_runs:
                    if run["model"] == "gbm":
                        continue
                    windows = run["walk_forward"].get("windows", [])
                    if not windows:
                        continue
                    fp = windows[0].get("fitted_params", {})
                    all_match = all(
                        w.get("fitted_params", {}) == fp for w in windows
                    )
                    sig.append(
                        f"- {run['model']} @ {run['horizon']}: "
                        + (f"ALL {len(windows)} windows locked to: "
                           + ", ".join(f"{k}={v}" for k, v in fp.items())
                           if all_match
                           else f"WARNING: params vary across windows")
                    )
                sig.append("")

                signal_text = "\n".join(sig)
                st.session_state["signal_text"] = signal_text
                st.session_state["signal_date"] = _signal_date

            # Render signal if built
            _sig_text = st.session_state.get("signal_text")
            if _sig_text:
                _sig_date = st.session_state.get("signal_date", "")
                st.download_button(
                    "Download Signal",
                    data=_sig_text,
                    file_name=f"experiment_signal_{_sig_date}.md",
                    mime="text/markdown",
                    key="download_signal",
                )
                with st.expander("Preview", expanded=True):
                    st.markdown(
                        f"<pre style='background:#111;color:#e0e0e0;padding:1rem;"
                        f"font-size:0.75rem;max-height:500px;overflow-y:auto;'>"
                        f"{_sig_text}</pre>",
                        unsafe_allow_html=True,
                    )

            # ── Mass Audit: Re-run All Windows ─────────────────────────
            st.divider()
            st.subheader("Full Audit — Re-run All Windows")
            st.markdown(
                f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
                f"Re-simulates every window at 500 paths for diagnostic verification "
                f"(block usage, convergence, overfitting checks). Slow — use Signal Export above "
                f"for communicating results."
                f"</span>", unsafe_allow_html=True,
            )

            if st.button("Re-run All Audits", type="secondary", key="mass_audit"):
                # Load data once
                loader = _AuditLoader()
                train_df, test_df = loader.get_train_test_split(train_pct=0.7)
                full_returns = np.concatenate([
                    loader.get_returns_array(train_df),
                    loader.get_returns_array(test_df),
                ])
                full_prices = np.concatenate([
                    _audit_get_px(train_df),
                    _audit_get_px(test_df)[1:],
                ])

                all_audits = []
                total_runs = len(all_runs)
                total_windows = sum(
                    len(r["walk_forward"].get("windows", []))
                    for r in all_runs
                )

                progress = st.progress(0, text="Running audits...")
                done = 0

                for run in all_runs:
                    m_name = run["model"]
                    h_str = run["horizon"]
                    h_days = int(h_str.replace("d", ""))
                    run_params = run.get("optimization", {}).get("best_params", {})
                    run_windows = run["walk_forward"].get("windows", [])

                    for w in run_windows:
                        t_end = w.get("train_end_idx", 730)
                        t_start_test = w.get("test_start_idx", t_end)
                        t_end_test = w.get("test_end_idx", t_start_test + h_days)

                        train_ret = full_returns[:t_end]
                        train_px = full_prices[:t_end + 1]
                        test_px = full_prices[t_start_test:t_end_test + 1]
                        test_ret = full_returns[t_start_test:t_end_test]

                        w_params = w.get("fitted_params") or run_params

                        try:
                            a = _run_audit(
                                model_name=m_name,
                                train_returns=train_ret,
                                train_prices=train_px,
                                test_prices=test_px,
                                test_returns=test_ret,
                                horizon_days=h_days,
                                n_simulations=500,  # Reduced for speed
                                seed=42,
                                train_start_date=w.get("train_start_date"),
                                train_end_date=w.get("train_end_date"),
                                test_start_date=w.get("test_start_date"),
                                test_end_date=w.get("test_end_date"),
                                model_params=w_params if w_params else None,
                            )
                            all_audits.append(a)
                        except Exception as e:
                            all_audits.append({
                                "model": m_name,
                                "horizon": h_str,
                                "window": {
                                    "test_start": w.get("test_start_date"),
                                    "test_end": w.get("test_end_date"),
                                },
                                "error": str(e),
                                "red_flags": [f"Audit failed: {str(e)[:100]}"],
                                "checks_passed": 0,
                                "checks_total": 1,
                            })

                        done += 1
                        progress.progress(
                            done / total_windows,
                            text=f"Auditing {m_name} @ {h_str} W{w['window_num']} ({done}/{total_windows})"
                        )

                progress.empty()

                # Build cross-window summary
                by_model = {}
                all_flags = []
                flag_counts = {}
                for a in all_audits:
                    m = a.get("model", "?")
                    if m not in by_model:
                        by_model[m] = []
                    by_model[m].append(a)
                    for f in a.get("red_flags", []):
                        all_flags.append(f"{m} {a.get('horizon','?')} {a.get('window',{}).get('test_start','?')}: {f}")
                        # Count flag types (first 40 chars as key)
                        ftype = f[:40]
                        flag_counts[ftype] = flag_counts.get(ftype, 0) + 1

                summary = {
                    "total_audits": len(all_audits),
                    "total_red_flags": len(all_flags),
                    "per_model": {},
                }
                for m, audits in by_model.items():
                    composites = [
                        a.get("sections", {}).get("scoring", {}).get("composite_score", 0)
                        for a in audits if a.get("sections")
                    ]
                    flags_per = sum(len(a.get("red_flags", [])) for a in audits)
                    summary["per_model"][m] = {
                        "n_windows": len(audits),
                        "avg_composite": round(float(np.mean(composites)), 4) if composites else 0,
                        "min_composite": round(float(np.min(composites)), 4) if composites else 0,
                        "max_composite": round(float(np.max(composites)), 4) if composites else 0,
                        "total_red_flags": flags_per,
                    }

                if flag_counts:
                    summary["most_common_flags"] = sorted(
                        flag_counts.items(), key=lambda x: x[1], reverse=True
                    )[:5]

                mass_export = {
                    "export_type": "mass_audit",
                    "timestamp": str(np.datetime64("now")),
                    "summary": summary,
                    "audits": all_audits,
                    "all_red_flags": all_flags,
                }

                # JSON export
                mass_json = json.dumps(mass_export, indent=2, default=str)

                # Markdown export
                md_lines = [
                    "# Mass Pipeline Audit — All Windows",
                    f"**Generated:** {mass_export['timestamp']}",
                    f"**Total audits:** {summary['total_audits']}",
                    f"**Total red flags:** {summary['total_red_flags']}",
                    "",
                    "## Summary by Model",
                ]
                for m, s in summary["per_model"].items():
                    md_lines.append(
                        f"### {m.replace('_',' ').title()}")
                    md_lines.append(
                        f"- Windows: {s['n_windows']}")
                    md_lines.append(
                        f"- Composite: avg={s['avg_composite']:.4f}, "
                        f"min={s['min_composite']:.4f}, max={s['max_composite']:.4f}")
                    md_lines.append(
                        f"- Red flags: {s['total_red_flags']}")
                    md_lines.append("")

                if all_flags:
                    md_lines.append("## All Red Flags")
                    for f in all_flags:
                        md_lines.append(f"- {f}")
                    md_lines.append("")

                md_lines.append("## Per-Window Details")
                for a in all_audits:
                    if a.get("sections"):
                        sc = a["sections"].get("scoring", {})
                        md_lines.append(
                            f"### {a['model']} @ {a['horizon']} | "
                            f"{a['window'].get('test_start','?')} to {a['window'].get('test_end','?')}")
                        md_lines.append(
                            f"- Composite: {sc.get('composite_score', 0):.4f}")
                        md_lines.append(
                            f"- Checks: {a['checks_passed']}/{a['checks_total']}")
                        metrics = sc.get("metrics", {})
                        if metrics:
                            md_lines.append("- Metrics:")
                            for mk, mv in metrics.items():
                                md_lines.append(
                                    f"  - {mk}: {mv['normalized']:.4f} "
                                    f"(weight {mv['weight']:.2f})")
                        # Tail events
                        sim_out = a["sections"].get("simulation_output", {})
                        te = sim_out.get("tail_events", {})
                        if te:
                            n_p = te.get("n_paths", 0)
                            gen = te.get("generic", {})
                            md_lines.append(f"- Tail events ({n_p} paths):")
                            for gk, gl in [("dd_50pct", "DD>=50%"), ("dd_75pct", "DD>=75%")]:
                                g = gen.get(gk, {})
                                md_lines.append(f"  - {gl}: {g.get('count',0)}/{n_p} ({g.get('pct',0)}%)")
                            for sk, sd in te.get("named_scenarios", {}).items():
                                flag_mark = " ***ZERO***" if sd["count"] == 0 else ""
                                md_lines.append(
                                    f"  - {sk.replace('_',' ').title()}: "
                                    f"{sd['count']}/{n_p} ({sd['pct']}%){flag_mark}")
                        flags = a.get("red_flags", [])
                        if flags:
                            md_lines.append("- Flags:")
                            for f in flags:
                                md_lines.append(f"  - {f}")
                        md_lines.append("")
                    elif a.get("error"):
                        md_lines.append(
                            f"### {a.get('model','?')} @ {a.get('horizon','?')} | "
                            f"{a.get('window',{}).get('test_start','?')}")
                        md_lines.append(f"- ERROR: {a['error']}")
                        md_lines.append("")

                mass_md = "\n".join(md_lines)

                st.success(f"Audited {len(all_audits)} windows. {summary['total_red_flags']} red flags found.")

                _mass_date = datetime.now().strftime("%Y-%m-%d")
                exp_c1, exp_c2 = st.columns(2)
                with exp_c1:
                    st.download_button(
                        "Full Markdown",
                        data=mass_md,
                        file_name=f"mass_audit_{len(all_audits)}windows_{_mass_date}.md",
                        mime="text/markdown",
                        key="mass_md",
                    )
                with exp_c2:
                    st.download_button(
                        "Full JSON",
                        data=mass_json,
                        file_name=f"mass_audit_{len(all_audits)}windows_{_mass_date}.json",
                        mime="application/json",
                        key="mass_json",
                    )

    except Exception as e:
        st.error(f"Could not load audit dependencies: {e}")
        import traceback
        st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
# TAB: Export Results
# ═══════════════════════════════════════════════════════════════════════
with tab_export:
    st.header("Export Results")
    st.markdown(
        f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
        f"Structured summary for sharing with colleagues or pasting into an LLM."
        f"</span>", unsafe_allow_html=True,
    )

    # Gather all data needed for the export
    _all_runs = data.get("runs", [])
    _data_info = data.get("data_info", {})
    _timestamp = data.get("timestamp", "?")

    # Build leaderboard rows
    _lb = []
    _gbm_base = {}
    for r in _all_runs:
        wf_r = r["walk_forward"]
        sc_r = r.get("final_scorecard", {})
        sm_r = sc_r.get("scoring_metrics", {})
        opt_r = r.get("optimization", {})
        _lb.append({
            "model": r["model"],
            "horizon": r["horizon"],
            "composite": wf_r["weighted_composite"],
            "params": opt_r.get("best_params", {}),
            "scoring_metrics": sm_r,
            "n_windows": wf_r["n_windows"],
            "windows": wf_r.get("windows", []),
            "distributions": sc_r.get("distributions", {}),
            "search_phase": opt_r.get("search_phase", "?"),
            "trials": opt_r.get("trials", []),
        })
        if r["model"] == "gbm":
            _gbm_base[r["horizon"]] = wf_r["weighted_composite"]
    _lb.sort(key=lambda x: x["composite"], reverse=True)

    # Find overall winner
    _winner = _lb[0] if _lb else {}

    # ── Build structured export data ────────────────────────────────

    def _build_export() -> dict:
        export = {}

        # Section 1: Config
        export["config"] = {
            "run_mode": data.get("run_mode", "?"),
            "timestamp": _timestamp,
            "train_period": f"{_data_info.get('train_start','?')} to {_data_info.get('train_end','?')}",
            "test_period": f"{_data_info.get('test_start','?')} to {_data_info.get('test_end','?')}",
            "train_days": _data_info.get("train_days"),
            "test_days": _data_info.get("test_days"),
            "models": sorted(set(r["model"] for r in _lb)),
            "horizons": sorted(set(r["horizon"] for r in _lb), key=lambda h: int(h.replace("d",""))),
            "per_horizon_config": {},
        }
        for r in _lb:
            if r["model"] != "gbm":
                p = r["params"]
                export["config"]["per_horizon_config"][r["horizon"]] = {
                    "block_length_sampling": p.get("block_length_sampling"),
                    "mean_block_length": p.get("mean_block_length"),
                    "regime_enabled": p.get("regime_enabled"),
                    "msm_frequency": p.get("msm_frequency"),
                    "msm_variance_switching": p.get("msm_variance_switching"),
                    "transition_matrix_method": p.get("transition_matrix_method"),
                    "n_windows": r["n_windows"],
                }

        # Section 2: Leaderboard
        export["leaderboard"] = []
        for r in _lb:
            gb = _gbm_base.get(r["horizon"], 0)
            delta = ((r["composite"] - gb) / gb * 100) if gb > 0 and r["model"] != "gbm" else 0
            p = r["params"]
            if r["model"] == "regime_block_bootstrap" and p:
                params_str = f"{p.get('block_length_sampling','?')} bl={p.get('mean_block_length','?')}d"
            else:
                params_str = "MLE" if r["model"] == "gbm" else str(p)
            export["leaderboard"].append({
                "model": r["model"],
                "horizon": r["horizon"],
                "composite": round(r["composite"], 4),
                "vs_gbm_delta": round(delta, 1),
                "params": params_str,
            })

        # Section 3: Per-Metric Summary (winning model)
        win_model = _winner["model"]
        win_runs = [r for r in _lb if r["model"] == win_model]
        # Aggregate across all horizons for the winner
        all_window_scores = {}
        for r in win_runs:
            for w in r["windows"]:
                for mk, mv in w.get("scoring_metrics", {}).items():
                    if mk not in all_window_scores:
                        all_window_scores[mk] = []
                    all_window_scores[mk].append(mv)

        export["per_metric_summary"] = {"model": win_model, "metrics": {}}
        metric_info_map = {m["key"]: m for m in METRIC_INFO}
        for mk, vals in all_window_scores.items():
            mi = metric_info_map.get(mk, {})
            export["per_metric_summary"]["metrics"][mk] = {
                "score": round(float(np.mean(vals)), 4),
                "weight": mi.get("weight", 0),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "category": mi.get("category", "?"),
            }

        # Section 4: Tail Events per horizon
        export["tail_events"] = {}
        for r in _lb:
            te = r["distributions"].get("tail_events")
            if te:
                export["tail_events"][f"{r['model']}_{r['horizon']}"] = {
                    "model": r["model"],
                    "horizon": r["horizon"],
                    "n_paths": te["n_paths"],
                    "generic": te["generic"],
                    "named_scenarios": te["named_scenarios"],
                }

        # Section 5: Stress Test Windows (most recent, ~4yr ago, ~8yr ago)
        export["stress_windows"] = []
        for r in _lb:
            if r["model"] == win_model:
                wins = r["windows"]
                if wins:
                    latest = wins[-1]
                    export["stress_windows"].append({
                        "horizon": r["horizon"],
                        "window": f"{latest.get('test_start_date','?')} to {latest.get('test_end_date','?')}",
                        "composite": latest.get("composite_score", 0),
                    })

        # Section 6: Block Length Analysis
        export["block_length_analysis"] = {}
        for r in _lb:
            if r["model"] != "gbm" and r["trials"]:
                p1 = [t for t in r["trials"]
                      if t.get("phase") == "phase1_block_length"]
                if p1:
                    p1.sort(key=lambda t: t["score"], reverse=True)
                    export["block_length_analysis"][r["horizon"]] = {
                        "winner": {
                            "bl": p1[0]["params"].get("mean_block_length"),
                            "sampling": p1[0]["params"].get("block_length_sampling"),
                            "score": round(p1[0]["score"], 4),
                        },
                        "top3": [
                            {
                                "bl": t["params"].get("mean_block_length"),
                                "sampling": t["params"].get("block_length_sampling"),
                                "score": round(t["score"], 4),
                            }
                            for t in p1[:3]
                        ],
                    }

        # Section 7: Red Flags
        all_flags = []
        flag_by_window = {}
        for r in _lb:
            for w in r["windows"]:
                w_key = f"{r['model']} {r['horizon']} W{w['window_num']}"
                # Check for common issues
                w_flags = []
                sm_w = w.get("scoring_metrics", {})
                for mk, mv in sm_w.items():
                    if mv < 0.2:
                        w_flags.append(f"Severe failure: {mk}={mv:.3f}")
                diag = w.get("diagnostic_metrics", {})
                if diag.get("mase") and diag["mase"] > 1.0:
                    w_flags.append(f"MASE={diag['mase']:.2f} > 1.0")
                pbc = sm_w.get("percentile_band_coverage", 1)
                if pbc < 0.5:
                    w_flags.append(f"Band coverage={pbc:.3f} < 0.5")
                if w_flags:
                    flag_by_window[w_key] = w_flags
                    all_flags.extend([f"{w_key}: {f}" for f in w_flags])

        flag_type_counts = {}
        for f in all_flags:
            ftype = f.split(": ", 1)[-1][:40] if ": " in f else f[:40]
            flag_type_counts[ftype] = flag_type_counts.get(ftype, 0) + 1

        total_windows = sum(r["n_windows"] for r in _lb)
        export["red_flags"] = {
            "total": len(all_flags),
            "total_windows": total_windows,
            "most_common": sorted(flag_type_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "windows_with_flags": len(flag_by_window),
            "all_flags": all_flags,
        }

        return export

    def _export_to_markdown(export: dict) -> str:
        lines = ["# BTC Sim Engine — Results Summary"]
        lines.append(f"**Generated:** {export['config']['timestamp'][:19]}")
        lines.append(f"**Mode:** {export['config']['run_mode']}")
        lines.append("")

        # Config
        c = export["config"]
        lines.append("## 1. Experiment Configuration")
        lines.append(f"- Training: {c['train_period']} ({c.get('train_days','?')} days)")
        lines.append(f"- Testing: {c['test_period']} ({c.get('test_days','?')} days)")
        lines.append(f"- Models: {', '.join(c['models'])}")
        lines.append(f"- Horizons: {', '.join(c['horizons'])}")
        for hz, hc in c.get("per_horizon_config", {}).items():
            lines.append(
                f"- {hz}: {hc.get('block_length_sampling','?')} bl={hc.get('mean_block_length','?')}d, "
                f"regime={'on' if hc.get('regime_enabled') else 'off'}, "
                f"windows={hc.get('n_windows','?')}")
        lines.append("")

        # Leaderboard
        lines.append("## 2. Leaderboard")
        lines.append("| Rank | Model | Horizon | Composite | vs GBM | Params |")
        lines.append("|------|-------|---------|-----------|--------|--------|")
        for i, row in enumerate(export["leaderboard"]):
            vs = f"{row['vs_gbm_delta']:+.1f}%" if row["vs_gbm_delta"] else "baseline"
            lines.append(
                f"| {i+1} | {row['model']} | {row['horizon']} | "
                f"{row['composite']:.4f} | {vs} | {row['params']} |")
        lines.append("")

        # Per-Metric
        pms = export["per_metric_summary"]
        lines.append(f"## 3. Per-Metric Summary ({pms['model']})")
        lines.append("| Metric | Score | Weight | Min | Max | Std | Category |")
        lines.append("|--------|-------|--------|-----|-----|-----|----------|")
        for mk, mv in pms["metrics"].items():
            lines.append(
                f"| {mk} | {mv['score']:.4f} | {mv['weight']:.2f} | "
                f"{mv['min']:.4f} | {mv['max']:.4f} | {mv['std']:.4f} | {mv['category']} |")
        lines.append("")

        # Tail Events
        lines.append("## 4. Tail Event Summary")
        for key, te in export["tail_events"].items():
            lines.append(f"### {te['model']} @ {te['horizon']} ({te['n_paths']} paths)")
            gen = te["generic"]
            lines.append(
                f"- DD>=50%: {gen['dd_50pct']['count']}/{te['n_paths']} ({gen['dd_50pct']['pct']}%) | "
                f"DD>=75%: {gen['dd_75pct']['count']}/{te['n_paths']} ({gen['dd_75pct']['pct']}%)")
            for sk, sd in te["named_scenarios"].items():
                flag = " **ZERO — model cannot produce this**" if sd["count"] == 0 else ""
                lines.append(f"- {sk}: {sd['count']}/{te['n_paths']} ({sd['pct']}%) — {sd['desc']}{flag}")
            lines.append("")

        # Block Length Analysis
        bla = export.get("block_length_analysis", {})
        if bla:
            lines.append("## 6. Block Length Analysis")
            for hz, bl_data in bla.items():
                w = bl_data["winner"]
                lines.append(f"- **{hz}**: {w['sampling']} bl={w['bl']}d (score={w['score']:.4f})")
                for j, alt in enumerate(bl_data["top3"][1:], 2):
                    lines.append(f"  - #{j}: {alt['sampling']} bl={alt['bl']}d ({alt['score']:.4f})")
            lines.append("")

        # Red Flags
        rf = export["red_flags"]
        lines.append(f"## 7. Red Flags ({rf['total']} total across {rf['total_windows']} windows)")
        if rf["most_common"]:
            lines.append("Most common:")
            for ftype, count in rf["most_common"]:
                lines.append(f"- {ftype}: {count} occurrences")
        if rf["all_flags"]:
            lines.append("\nAll flags:")
            for f in rf["all_flags"]:
                lines.append(f"- {f}")
        elif rf["total"] == 0:
            lines.append("No red flags detected.")
        lines.append("")

        return "\n".join(lines)

    # Build and display
    export_data = _build_export()

    # Section 1: Config
    st.subheader("1. Experiment Configuration")
    cfg = export_data["config"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Mode", cfg["run_mode"])
    c2.metric("Models", ", ".join(cfg["models"]))
    c3.metric("Horizons", ", ".join(cfg["horizons"]))
    st.markdown(f"**Train:** {cfg['train_period']} ({cfg.get('train_days','?')} days)")
    st.markdown(f"**Test:** {cfg['test_period']} ({cfg.get('test_days','?')} days)")

    # Section 2: Leaderboard
    st.subheader("2. Leaderboard")
    lb_table = []
    for i, row in enumerate(export_data["leaderboard"]):
        lb_table.append({
            "Rank": i + 1,
            "Model": row["model"].replace("_", " ").title(),
            "Horizon": row["horizon"],
            "Composite": row["composite"],
            "vs GBM": f"{row['vs_gbm_delta']:+.1f}%" if row["vs_gbm_delta"] else "baseline",
            "Params": row["params"],
        })
    st.dataframe(
        pd.DataFrame(lb_table).style.format({"Composite": "{:.4f}"})
            .background_gradient(subset=["Composite"], cmap="YlOrRd", vmin=0.5, vmax=1),
        use_container_width=True, hide_index=True,
    )

    # Section 3: Per-Metric
    pms = export_data["per_metric_summary"]
    st.subheader(f"3. Per-Metric Summary ({pms['model'].replace('_',' ').title()})")
    met_table = []
    for mk, mv in pms["metrics"].items():
        met_table.append({
            "Metric": mk.replace("_", " ").title(),
            "Score": mv["score"],
            "Weight": mv["weight"],
            "Min": mv["min"],
            "Max": mv["max"],
            "Std": mv["std"],
            "Category": mv["category"],
        })
    st.dataframe(
        pd.DataFrame(met_table).style.format(
            {"Score": "{:.4f}", "Weight": "{:.2f}", "Min": "{:.4f}",
             "Max": "{:.4f}", "Std": "{:.4f}"}
        ).background_gradient(subset=["Score"], cmap="YlOrRd", vmin=0, vmax=1),
        use_container_width=True, hide_index=True,
    )

    # Section 4: Tail Events
    st.subheader("4. Tail Event Summary")
    for key, te in export_data["tail_events"].items():
        with st.expander(f"{te['model'].replace('_',' ').title()} @ {te['horizon']}"):
            gen = te["generic"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("DD>=50%", f"{gen['dd_50pct']['pct']}%")
            c2.metric("DD>=75%", f"{gen['dd_75pct']['pct']}%")
            c3.metric("Dur>=180d", f"{gen['dur_180d']['pct']}%")
            c4.metric("Dur>=365d", f"{gen['dur_365d']['pct']}%")
            for sk, sd in te["named_scenarios"].items():
                color = RED if sd["count"] == 0 else GREEN
                st.markdown(
                    f"<span style='color:{color};'>"
                    f"{'[X]' if sd['count']==0 else '[+]'}</span> "
                    f"{sk.replace('_',' ').title()}: "
                    f"{sd['count']}/{te['n_paths']} ({sd['pct']}%) — {sd['desc']}",
                    unsafe_allow_html=True,
                )

    # Section 6: Block Length Analysis
    bla = export_data.get("block_length_analysis", {})
    if bla:
        st.subheader("6. Block Length Analysis")
        for hz, bl_data in bla.items():
            w = bl_data["winner"]
            st.markdown(
                f"**{hz}:** {w['sampling']} bl={w['bl']}d "
                f"(score={w['score']:.4f})")
            alt_str = " | ".join(
                f"#{j}: {a['sampling']} bl={a['bl']}d ({a['score']:.4f})"
                for j, a in enumerate(bl_data["top3"], 1)
            )
            st.markdown(f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>{alt_str}</span>",
                        unsafe_allow_html=True)

    # Section 7: Red Flags
    rf = export_data["red_flags"]
    st.subheader(f"7. Red Flags ({rf['total']})")
    if rf["total"] == 0:
        st.success("No red flags detected across all windows.")
    else:
        st.warning(f"{rf['total']} flags across {rf['windows_with_flags']}/{rf['total_windows']} windows")
        if rf["most_common"]:
            for ftype, count in rf["most_common"]:
                st.markdown(f"- **{ftype}**: {count} occurrences")

    # Export buttons
    st.divider()
    export_json = json.dumps(export_data, indent=2, default=str)
    export_md = _export_to_markdown(export_data)

    exp_c1, exp_c2, exp_c3 = st.columns(3)
    with exp_c1:
        st.download_button(
            "Download JSON",
            data=export_json,
            file_name=f"results_summary_{_timestamp[:10]}.json",
            mime="application/json",
        )
    with exp_c2:
        st.download_button(
            "Download Markdown",
            data=export_md,
            file_name=f"results_summary_{_timestamp[:10]}.md",
            mime="text/markdown",
        )
    with exp_c3:
        st.code(export_md, language="markdown")


# ═══════════════════════════════════════════════════════════════════════
# TAB: Production Simulation
# ═══════════════════════════════════════════════════════════════════════
with tab_prod:
    st.header("Production Simulation — 1,460-Day (4-Year) Forward Projection")
    st.markdown(
        f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
        f"Single horizon: <b>1,460 days (4 years)</b> — the full BTC halving cycle. "
        f"Trained on ALL available history (no holdout). "
        f"This is not walk-forward validation — there is no realized path to score against."
        f"</span>", unsafe_allow_html=True,
    )

    _PROD_MODELS_ALL = {
        "rbb": {
            "model_name": "regime_block_bootstrap",
            "params": PROD_RBB_PARAMS,
            "label": "RBB (Block Bootstrap)",
            "color": BTC_ORANGE,
            "desc": "Stationary block bootstrap — resamples actual historical return sequences. Walk-forward: 0.8109.",
        },
        "garch": {
            "model_name": "garch_1_1",
            "params": {},
            "label": "GARCH(1,1)",
            "color": BLUE,
            "desc": "Parametric volatility clustering (t-distribution, Constant mean). Walk-forward: 0.7800.",
        },
        "gbm": {
            "model_name": "gbm",
            "params": {},
            "label": "GBM (Baseline)",
            "color": TEXT_DIM,
            "desc": "Log-normal random walk — MLE drift and volatility. Walk-forward: 0.7630.",
        },
    }

    # ── Auto-load persisted results on first visit ───────────────────
    if "prod_all" not in st.session_state:
        _saved_path = Path("results/production_sim.json.gz")
        if _saved_path.exists():
            import gzip as _gzip
            with _gzip.open(_saved_path, "rt", encoding="utf-8") as _f:
                _saved = json.load(_f)
            # Handle both old format (rbb_paths) and new format (all_paths)
            if "all_paths" in _saved:
                _loaded_paths = {k: np.array(v) for k, v in _saved["all_paths"].items()}
            else:
                _loaded_paths = {"rbb": np.array(_saved["rbb_paths"])}
            st.session_state["prod_all"] = {
                "initial_price": _saved["initial_price"],
                "data_end_date": _saved["data_end_date"],
                "n_training_days": _saved["n_training_days"],
                "models": _saved["models"],
                "all_paths": _loaded_paths,
            }
            # Also populate prod_sim for Executive Summary backward compat
            _rbb_m = _saved["models"]["rbb"]
            st.session_state["prod_sim"] = {
                "initial_price": _saved["initial_price"],
                "data_end_date": _saved["data_end_date"],
                "n_training_days": _saved["n_training_days"],
                "model_meta": {"model_name": "regime_block_bootstrap",
                               "label": "RBB (Block Bootstrap)",
                               "specs": "Loaded from saved results"},
                "p5": _rbb_m["p5"], "p25": _rbb_m["p25"], "p50": _rbb_m["p50"],
                "p75": _rbb_m["p75"], "p95": _rbb_m["p95"],
                "final_prices": _rbb_m["final_prices"],
                "tail_events": _rbb_m["tail_events"],
            }

    try:
        _prod_get_model, _ProdLoader, _prod_get_px = load_prod_deps()

        if st.button("Run All 3 Models (re-simulate)", type="primary", key="run_prod_all"):
            loader = _ProdLoader()
            full_df = loader.load_processed_data()
            all_returns = full_df["log_return"].values
            all_prices = full_df["Close"].values
            initial_price = float(all_prices[-1])
            data_end_date = str(full_df.index[-1].date())
            n_training_days = len(all_returns)

            all_model_results = {}
            for mkey, mcfg in _PROD_MODELS_ALL.items():
                with st.spinner(f"Training {mcfg['label']} on full history..."):
                    model = _prod_get_model(mcfg["model_name"])
                    if mcfg["params"]:
                        model.set_params(**mcfg["params"])
                    model.fit(all_returns)
                    sim = model.simulate(
                        n_simulations=PROD_N_SIMS, n_steps=PROD_HORIZON,
                        initial_price=initial_price, seed=42,
                    )

                    # Percentiles
                    pcts = {}
                    for p in [5, 10, 25, 50, 75, 90, 95]:
                        pcts[f"p{p}"] = np.percentile(sim.paths, p, axis=0).tolist()
                    pcts["p50"] = np.median(sim.paths, axis=0).tolist()

                    # Tail events
                    n_sims = sim.paths.shape[0]
                    dd_50 = dd_75 = dur_180 = dur_365 = 0
                    crash_2014 = crash_2018 = crash_2022 = crash_2020 = 0
                    for i in range(n_sims):
                        path = sim.paths[i]
                        rm = np.maximum.accumulate(path)
                        dd_s = (path - rm) / rm
                        max_dd = float(np.min(dd_s))
                        if max_dd <= -0.50: dd_50 += 1
                        if max_dd <= -0.75: dd_75 += 1
                        in_dd = False; dd_start = 0; max_dur = 0
                        for j in range(len(dd_s)):
                            if dd_s[j] < -0.01:
                                if not in_dd: in_dd = True; dd_start = j
                            else:
                                if in_dd:
                                    d = j - dd_start
                                    if d > max_dur: max_dur = d
                                    in_dd = False
                        if in_dd:
                            d = len(dd_s) - dd_start
                            if d > max_dur: max_dur = d
                        if max_dur >= 180: dur_180 += 1
                        if max_dur >= 365: dur_365 += 1
                        if max_dd <= -0.85 and max_dur >= 390: crash_2014 += 1
                        if max_dd <= -0.84 and max_dur >= 365: crash_2018 += 1
                        if max_dd <= -0.77 and max_dur >= 390: crash_2022 += 1
                        if len(dd_s) > 30 and float(np.min(dd_s[:31])) <= -0.50: crash_2020 += 1

                    # Model specs
                    specs = ""
                    if mcfg["model_name"] == "regime_block_bootstrap":
                        specs = f"Block: geometric 30d | Pool: {len(model._block_pools[0])} blocks | Regime: OFF"
                    elif mcfg["model_name"] == "garch_1_1":
                        fp = getattr(model, "_fitted_params", {})
                        specs = f"p=1, q=1, dist=t, Constant | persistence={fp.get('persistence', '?')} | nu={fp.get('nu', '?'):.2f}" if fp.get("nu") else f"p=1, q=1, dist=t, Constant | persistence={fp.get('persistence', '?')}"
                    elif mcfg["model_name"] == "gbm":
                        fp = getattr(model, "_fitted_params", {})
                        specs = f"mu={fp.get('mu', 0):.4f} | sigma={fp.get('sigma', 0):.4f} | Normal innovations"

                    all_model_results[mkey] = {
                        **pcts,
                        "paths": sim.paths,
                        "final_prices": sim.paths[:, -1].tolist(),
                        "specs": specs,
                        "tail_events": {
                            "n_paths": n_sims,
                            "generic": {
                                "dd_50pct": {"count": dd_50, "pct": round(dd_50 / n_sims * 100, 1)},
                                "dd_75pct": {"count": dd_75, "pct": round(dd_75 / n_sims * 100, 1)},
                                "dur_180d": {"count": dur_180, "pct": round(dur_180 / n_sims * 100, 1)},
                                "dur_365d": {"count": dur_365, "pct": round(dur_365 / n_sims * 100, 1)},
                            },
                            "named_scenarios": {
                                "2014_mt_gox": {"desc": "DD >= 85%, duration >= 13mo", "count": crash_2014, "pct": round(crash_2014 / n_sims * 100, 2)},
                                "2018_crash": {"desc": "DD >= 84%, duration >= 12mo", "count": crash_2018, "pct": round(crash_2018 / n_sims * 100, 2)},
                                "2022_crash": {"desc": "DD >= 77%, duration >= 13mo", "count": crash_2022, "pct": round(crash_2022 / n_sims * 100, 2)},
                                "2020_flash_crash": {"desc": "DD >= 50% within 30d", "count": crash_2020, "pct": round(crash_2020 / n_sims * 100, 2)},
                            },
                        },
                    }

            _prod_all_data = {
                "initial_price": initial_price,
                "data_end_date": data_end_date,
                "n_training_days": n_training_days,
                "models": {k: {kk: vv for kk, vv in v.items() if kk != "paths"} for k, v in all_model_results.items()},
                "all_paths": {k: all_model_results[k]["paths"] for k in all_model_results},
            }
            st.session_state["prod_all"] = _prod_all_data

            # Keep backward compat with single-model session state
            rbb = all_model_results["rbb"]
            st.session_state["prod_sim"] = {
                "initial_price": initial_price, "data_end_date": data_end_date,
                "n_training_days": n_training_days,
                "model_meta": {"model_name": "regime_block_bootstrap", "label": "RBB (Block Bootstrap)",
                               "specs": _PROD_MODELS_ALL["rbb"]["desc"]},
                "p5": rbb["p5"], "p25": rbb["p25"], "p50": rbb["p50"],
                "p75": rbb["p75"], "p95": rbb["p95"],
                "final_prices": rbb["final_prices"], "tail_events": rbb["tail_events"],
            }

            # ── Persist to disk ────────────────────────────────────────
            import gzip as _gzip
            _save_data = {
                "initial_price": initial_price,
                "data_end_date": data_end_date,
                "n_training_days": n_training_days,
                "models": _prod_all_data["models"],
                "all_paths": {k: v.tolist() for k, v in _prod_all_data["all_paths"].items()},
            }
            _save_path = Path("results/production_sim.json.gz")
            _save_path.parent.mkdir(parents=True, exist_ok=True)
            with _gzip.open(_save_path, "wt", encoding="utf-8") as _f:
                json.dump(_save_data, _f, separators=(",", ":"))
            st.toast("Results saved to results/production_sim.json.gz")

        # ── Render 3-Model Comparison ──────────────────────────────────
        _prod_all = st.session_state.get("prod_all")
        if _prod_all:
            initial_price = _prod_all["initial_price"]
            data_end = _prod_all["data_end_date"]
            models_data = _prod_all["models"]

            start_ts = pd.Timestamp(data_end) + pd.Timedelta(days=1)
            n_pts = len(models_data["rbb"]["p50"])
            x_dates = pd.date_range(start=start_ts, periods=n_pts, freq="D").tolist()

            # ── 3-Model Price Fan Chart ────────────────────────────────
            st.subheader("3-Model Price Path Comparison")

            fig_3m = go.Figure()
            # RBB 90% CI envelope (reference)
            rbb_d = models_data["rbb"]
            fig_3m.add_trace(go.Scatter(
                x=x_dates + x_dates[::-1],
                y=rbb_d["p5"] + rbb_d["p95"][::-1],
                fill="toself", fillcolor="rgba(247,147,26,0.06)",
                line=dict(color="rgba(0,0,0,0)"), name="RBB 90% CI", showlegend=True,
            ))
            fig_3m.add_trace(go.Scatter(
                x=x_dates + x_dates[::-1],
                y=rbb_d["p25"] + rbb_d["p75"][::-1],
                fill="toself", fillcolor="rgba(247,147,26,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="RBB 50% CI", showlegend=True,
            ))

            # Median lines for all 3 models
            for mkey, mcfg in _PROD_MODELS_ALL.items():
                md = models_data[mkey]
                fig_3m.add_trace(go.Scatter(
                    x=x_dates, y=md["p50"], mode="lines",
                    line=dict(color=mcfg["color"], width=2.5 if mkey == "rbb" else 2,
                              dash="solid" if mkey == "rbb" else "dash"),
                    name=f"{mcfg['label']} Median",
                ))

            fig_3m.add_hline(
                y=initial_price, line_dash="dot", line_color=TEXT_DIM, line_width=1,
                annotation_text=f"Start: ${initial_price:,.0f}",
                annotation_font_color=TEXT_DIM,
            )
            fig_3m.update_layout(
                title=f"4-Year Forward Projection — All Models (from {data_end})",
                xaxis_title="", yaxis_title="Price (USD)", yaxis_type="log",
                height=600, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_3m, use_container_width=True)

            # ── Summary Stats Table ────────────────────────────────────
            st.subheader("Terminal Price Distribution (4-Year)")
            summary_rows = []
            for mkey, mcfg in _PROD_MODELS_ALL.items():
                fp = np.array(models_data[mkey]["final_prices"])
                summary_rows.append({
                    "Model": mcfg["label"],
                    "Median": f"${np.median(fp):,.0f}",
                    "Mean": f"${np.mean(fp):,.0f}",
                    "P5": f"${np.percentile(fp, 5):,.0f}",
                    "P95": f"${np.percentile(fp, 95):,.0f}",
                    "% Below Start": f"{float(np.mean(fp < initial_price)) * 100:.1f}%",
                    "WF Score": {"rbb": "0.8109", "garch": "0.7800", "gbm": "0.7630"}[mkey],
                })
            st.table(pd.DataFrame(summary_rows).set_index("Model"))

            # ── Model Specs ────────────────────────────────────────────
            with st.expander("Model Specifications"):
                for mkey, mcfg in _PROD_MODELS_ALL.items():
                    md = models_data[mkey]
                    st.markdown(
                        f"<div style='background:{PANEL};border-left:3px solid {mcfg['color']};"
                        f"border-radius:4px;padding:10px;margin-bottom:8px;'>"
                        f"<span style='color:{mcfg['color']};font-weight:700;'>{mcfg['label']}</span><br>"
                        f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>{md['specs']}</span>"
                        f"</div>", unsafe_allow_html=True,
                    )

            # ── Tail Event Comparison ──────────────────────────────────
            st.subheader("Tail Event Comparison — Forward Projection")
            te_rows = []
            for key, label in [("dd_50pct", "DD >= 50%"), ("dd_75pct", "DD >= 75%"),
                                ("dur_180d", "Duration >= 180d"), ("dur_365d", "Duration >= 365d")]:
                row = {"Event": label}
                for mkey, mcfg in _PROD_MODELS_ALL.items():
                    te = models_data[mkey]["tail_events"]["generic"][key]
                    row[mcfg["label"]] = f"{te['pct']}%"
                te_rows.append(row)
            for key in ["2014_mt_gox", "2018_crash", "2022_crash", "2020_flash_crash"]:
                row = {"Event": key.replace("_", " ").title()}
                for mkey, mcfg in _PROD_MODELS_ALL.items():
                    sc = models_data[mkey]["tail_events"]["named_scenarios"][key]
                    row[mcfg["label"]] = f"{sc['pct']}%"
                te_rows.append(row)
            st.table(pd.DataFrame(te_rows).set_index("Event"))

        elif not st.session_state.get("prod_all"):
            st.info("Click **Run All 3 Models** to generate forward projections.")

    except Exception as e:
        st.error(f"Could not load production simulation dependencies: {e}")
        import traceback
        st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
# TAB: BCR Stress Test
# ═══════════════════════════════════════════════════════════════════════
with tab_bcr:
    st.header("BCR Stress Test — Perpetual Preferred Solvency")
    st.markdown(
        f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
        f"Bitcoin-collateralized perpetual preferred product solvency analysis. "
        f"Deterministic cash-flow simulation over RBB price paths. "
        f"Run <b>Production Simulation</b> first, then adjust parameters below and click <b>Run BCR Analysis</b>.</span>",
        unsafe_allow_html=True,
    )

    # ── Inline Parameters ──────────────────────────────────────────────
    st.subheader("Parameters")
    bcr_c1, bcr_c2, bcr_c3 = st.columns(3)
    with bcr_c1:
        bcr_nav = st.number_input("NAV — BTC Treasury ($)", value=1_000_000_000, step=100_000_000, format="%d", key="bcr_nav")
        bcr_ratio = st.number_input("BCR (Coverage Ratio)", value=40.0, step=5.0, format="%.0f", key="bcr_ratio")
    with bcr_c2:
        bcr_div_rate = st.number_input("Dividend Rate (annual %)", value=10.0, step=1.0, format="%.1f", key="bcr_div_rate") / 100.0
        bcr_cash_months = st.number_input("Cash Reserve (months)", value=0, step=1, min_value=0, max_value=24, key="bcr_cash_months")
    with bcr_c3:
        bcr_btc_fraction = st.number_input("BTC Fraction (%)", value=100.0, step=5.0, min_value=0.0, max_value=100.0, format="%.0f", key="bcr_btc_frac") / 100.0
        bcr_opex_rate = st.number_input("OpEx (annual % of NAV)", value=0.0, step=0.5, format="%.1f", key="bcr_opex") / 100.0
    bcr_fail_mode = st.selectbox("Failure Condition", ["BCR < 1", "NAV < PPE Notional"], key="bcr_fail_mode")

    # ── Derived values ─────────────────────────────────────────────────
    _bcr_ppe_leverage = 1.0 / (bcr_ratio * bcr_div_rate) if (bcr_ratio * bcr_div_rate) > 0 else 0
    _bcr_ppe_notional = bcr_nav * _bcr_ppe_leverage
    _bcr_annual_dividend = _bcr_ppe_notional * bcr_div_rate
    _bcr_annual_opex = bcr_nav * bcr_opex_rate
    _bcr_monthly_dividend = _bcr_annual_dividend / 12.0
    _bcr_monthly_opex = _bcr_annual_opex / 12.0
    _bcr_monthly_obligation = _bcr_monthly_dividend + _bcr_monthly_opex

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("PPE Notional", f"${_bcr_ppe_notional:,.0f}")
    dc2.metric("Annual Dividend", f"${_bcr_annual_dividend:,.0f}")
    dc3.metric("Monthly Obligation", f"${_bcr_monthly_obligation:,.0f}")
    dc4.metric("Starting BCR", f"{bcr_ratio:.0f}x")

    st.divider()

    # ── Run BCR Analysis ───────────────────────────────────────────────
    _prod_all = st.session_state.get("prod_all")
    if _prod_all is None:
        st.warning("Run **Production Simulation** (All 3 Models) first to generate price paths.")
    else:
        initial_price = _prod_all["initial_price"]
        data_end = _prod_all["data_end_date"]
        start_ts = pd.Timestamp(data_end) + pd.Timedelta(days=1)
        _all_paths = _prod_all.get("all_paths", {})

        if not _all_paths:
            st.warning("Price paths not available. Re-run Production Simulation.")
        elif st.button("Run BCR Analysis", type="primary", key="run_bcr"):
            with st.spinner("Running solvency simulation across all 3 models..."):
                _bcr_model_results = {}
                for mkey, m_paths in _all_paths.items():
                    n_sims_bcr = m_paths.shape[0]
                    n_days = m_paths.shape[1]
                    n_months = n_days // 30

                    month_indices = [min(30 * (m + 1) - 1, n_days - 1) for m in range(n_months)]
                    monthly_prices = m_paths[:, month_indices]

                    all_bcr_series = np.zeros((n_sims_bcr, n_months))
                    failed = np.zeros(n_sims_bcr, dtype=bool)
                    failure_month = np.full(n_sims_bcr, n_months + 1, dtype=int)
                    min_bcr_arr = np.zeros(n_sims_bcr)
                    terminal_bcr_arr = np.zeros(n_sims_bcr)

                    for i in range(n_sims_bcr):
                        btc_holdings = bcr_nav / initial_price
                        cash = bcr_cash_months * _bcr_monthly_obligation
                        path_failed = False

                        for m in range(n_months):
                            price = monthly_prices[i, m]
                            btc_obligation = _bcr_monthly_obligation * bcr_btc_fraction

                            if cash >= btc_obligation:
                                cash -= btc_obligation
                            else:
                                btc_sell_amount = btc_obligation - cash
                                cash = 0.0
                                if price > 0:
                                    btc_holdings -= btc_sell_amount / price

                            btc_value = btc_holdings * price
                            current_bcr = btc_value / _bcr_annual_dividend if _bcr_annual_dividend > 0 else float("inf")
                            all_bcr_series[i, m] = current_bcr

                            if not path_failed:
                                if bcr_fail_mode == "BCR < 1":
                                    if current_bcr < 1.0:
                                        path_failed = True
                                        failure_month[i] = m + 1
                                else:
                                    if btc_value < _bcr_ppe_notional:
                                        path_failed = True
                                        failure_month[i] = m + 1

                        failed[i] = path_failed
                        min_bcr_arr[i] = np.min(all_bcr_series[i, :])
                        terminal_bcr_arr[i] = all_bcr_series[i, -1]

                    _bcr_model_results[mkey] = {
                        "all_bcr_series": all_bcr_series,
                        "failed": failed,
                        "failure_month": failure_month,
                        "min_bcr": min_bcr_arr,
                        "terminal_bcr": terminal_bcr_arr,
                        "n_sims": n_sims_bcr,
                        "n_months": n_months,
                        "monthly_prices": monthly_prices,
                    }

                st.session_state["bcr_results"] = {
                    "models": _bcr_model_results,
                    "initial_price": initial_price,
                    "start_ts": start_ts,
                    "n_months": n_months,
                    "params": {
                        "nav": bcr_nav, "bcr": bcr_ratio, "div_rate": bcr_div_rate,
                        "cash_months": bcr_cash_months, "btc_fraction": bcr_btc_fraction,
                        "opex_rate": bcr_opex_rate, "fail_mode": bcr_fail_mode,
                        "ppe_notional": _bcr_ppe_notional, "annual_dividend": _bcr_annual_dividend,
                        "monthly_obligation": _bcr_monthly_obligation,
                    },
                }

        # ── Render BCR Results ─────────────────────────────────────────
        _bcr_res = st.session_state.get("bcr_results")
        if _bcr_res:
            _bcr_models = _bcr_res["models"]
            n_months = _bcr_res["n_months"]
            _bcr_start_ts = _bcr_res["start_ts"]
            _bcr_params = _bcr_res["params"]
            _init_px = _bcr_res["initial_price"]

            _BCR_MODEL_COLORS = {"rbb": BTC_ORANGE, "garch": BLUE, "gbm": TEXT_DIM}
            _BCR_MODEL_LABELS = {"rbb": "RBB (Block Bootstrap)", "garch": "GARCH(1,1)", "gbm": "GBM (Baseline)"}
            month_dates = [_bcr_start_ts + pd.Timedelta(days=30 * (m + 1)) for m in range(n_months)]

            # ── HERO CHART: BCR Percentile Curve (Inverse CDF) ──────────
            st.subheader("BCR Percentile Curve — Probability of Default")

            _bcr_curve_mode = st.selectbox(
                "BCR Metric", ["Terminal BCR (end of horizon)", "Minimum BCR (worst point along path)"],
                key="bcr_curve_mode",
            )
            _use_terminal = "Terminal" in _bcr_curve_mode

            _pct_range = list(range(1, 100))
            fig_bcr_curve = go.Figure()

            for mkey in ["rbb", "garch", "gbm"]:
                if mkey not in _bcr_models:
                    continue
                mr = _bcr_models[mkey]
                data_arr = mr["terminal_bcr"] if _use_terminal else mr["min_bcr"]
                y_vals = np.percentile(data_arr, _pct_range)

                fig_bcr_curve.add_trace(go.Scatter(
                    x=_pct_range, y=y_vals.tolist(), mode="lines",
                    line=dict(color=_BCR_MODEL_COLORS[mkey], width=2.5 if mkey == "rbb" else 2),
                    name=_BCR_MODEL_LABELS[mkey],
                ))

            # Failure threshold
            fig_bcr_curve.add_hline(
                y=1.0, line_dash="dash", line_color=RED, line_width=2,
                annotation_text="BCR = 1 (Failure Threshold)",
                annotation_font_color=RED, annotation_font_size=12,
                annotation_position="top left",
            )

            _bcr_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("yaxis", "xaxis")}
            _metric_label = "Terminal BCR" if _use_terminal else "Minimum BCR"
            fig_bcr_curve.update_layout(
                title=f"{_metric_label} by Percentile — {_bcr_models['rbb']['n_sims']:,} Paths Per Model",
                xaxis=dict(title="Percentile", gridcolor=GRID, zerolinecolor=GRID,
                           tickfont=dict(color="#bbbbbb"), title_font=dict(color="#cccccc"),
                           tickvals=[1, 5, 10, 25, 50, 75, 90, 95, 99],
                           range=[0, 100]),
                yaxis=dict(title=_metric_label, type="log", gridcolor=GRID, zerolinecolor=GRID,
                           tickfont=dict(color="#bbbbbb"), title_font=dict(color="#cccccc")),
                height=650, **_bcr_layout,
            )
            st.plotly_chart(fig_bcr_curve, use_container_width=True)

            # ── BCR Time Series Fan Chart (secondary) ──────────────────
            with st.expander("BCR Trajectory Over Time (Fan Chart)"):
                rbb_bcr = _bcr_models["rbb"]["all_bcr_series"]
                rbb_p5 = np.percentile(rbb_bcr, 5, axis=0)
                rbb_p25 = np.percentile(rbb_bcr, 25, axis=0)
                rbb_p50 = np.median(rbb_bcr, axis=0)
                rbb_p75 = np.percentile(rbb_bcr, 75, axis=0)
                rbb_p95 = np.percentile(rbb_bcr, 95, axis=0)

                fig_bcr_ts = go.Figure()
                fig_bcr_ts.add_trace(go.Scatter(
                    x=month_dates + month_dates[::-1],
                    y=rbb_p5.tolist() + rbb_p95.tolist()[::-1],
                    fill="toself", fillcolor="rgba(247,147,26,0.06)",
                    line=dict(color="rgba(0,0,0,0)"), name="RBB 90% CI",
                ))
                fig_bcr_ts.add_trace(go.Scatter(
                    x=month_dates + month_dates[::-1],
                    y=rbb_p25.tolist() + rbb_p75.tolist()[::-1],
                    fill="toself", fillcolor="rgba(247,147,26,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), name="RBB 50% CI",
                ))
                for mkey in ["rbb", "garch", "gbm"]:
                    if mkey not in _bcr_models:
                        continue
                    m_p50 = np.median(_bcr_models[mkey]["all_bcr_series"], axis=0)
                    is_rbb = mkey == "rbb"
                    fig_bcr_ts.add_trace(go.Scatter(
                        x=month_dates, y=m_p50.tolist(), mode="lines",
                        line=dict(color=_BCR_MODEL_COLORS[mkey], width=2.5 if is_rbb else 2,
                                  dash="solid" if is_rbb else "dash"),
                        name=f"{_BCR_MODEL_LABELS[mkey]} Median",
                    ))
                fig_bcr_ts.add_hline(y=1.0, line_dash="dash", line_color=RED, line_width=2)
                y_max = max(float(rbb_p95.max()) * 1.1, _bcr_params["bcr"] * 1.5)
                _ts_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"}
                fig_bcr_ts.update_layout(
                    title=f"BCR Over Time — {n_months} Months",
                    xaxis_title="", yaxis_title="BCR",
                    yaxis=dict(range=[0, y_max], gridcolor=GRID, zerolinecolor=GRID,
                               tickfont=dict(color="#bbbbbb"), title_font=dict(color="#cccccc")),
                    height=500, **_ts_layout,
                )
                st.plotly_chart(fig_bcr_ts, use_container_width=True)

            # ── Summary Table — All 3 Models ───────────────────────────
            st.subheader("Solvency Summary — All Models")
            _summary_rows = []
            for mkey in ["rbb", "garch", "gbm"]:
                if mkey not in _bcr_models:
                    continue
                mr = _bcr_models[mkey]
                n_f = int(np.sum(mr["failed"]))
                n_s = mr["n_sims"]
                _summary_rows.append({
                    "Model": _BCR_MODEL_LABELS[mkey],
                    "Failure %": f"{n_f / n_s * 100:.1f}%",
                    "Failed Paths": f"{n_f:,} / {n_s:,}",
                    "Median Terminal BCR": f"{np.median(mr['terminal_bcr']):.1f}x",
                    "P5 Terminal BCR": f"{np.percentile(mr['terminal_bcr'], 5):.1f}x",
                    "P5 Min BCR": f"{np.percentile(mr['min_bcr'], 5):.2f}x",
                    "Median Min BCR": f"{np.median(mr['min_bcr']):.1f}x",
                })
            st.table(pd.DataFrame(_summary_rows).set_index("Model"))

            # Runway metric
            monthly_btc_drain = (_bcr_params["monthly_obligation"] * _bcr_params["btc_fraction"]) / _init_px if _init_px > 0 else 0
            starting_btc = _bcr_params["nav"] / _init_px
            runway_months = int(starting_btc / monthly_btc_drain) if monthly_btc_drain > 0 else 999
            st.metric("Months of Runway (at constant price)", f"{runway_months}")

            # ── Min BCR Distribution — All 3 Models ────────────────────
            st.subheader("Minimum BCR Distribution")
            fig_min_hist = go.Figure()
            for mkey in ["rbb", "garch", "gbm"]:
                if mkey not in _bcr_models:
                    continue
                mr = _bcr_models[mkey]
                min_bcr_display = np.clip(mr["min_bcr"], 0, np.percentile(mr["min_bcr"], 99))
                fig_min_hist.add_trace(go.Histogram(
                    x=min_bcr_display, nbinsx=60, name=_BCR_MODEL_LABELS[mkey],
                    marker_color=_BCR_MODEL_COLORS[mkey], opacity=0.5,
                ))
            fig_min_hist.add_vline(x=1.0, line_dash="dash", line_color=RED, line_width=2,
                                   annotation_text="BCR=1", annotation_font_color=RED)
            fig_min_hist.update_layout(
                title="Min BCR Reached Per Path — All Models",
                xaxis_title="Minimum BCR", yaxis_title="Count",
                barmode="overlay", height=400, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_min_hist, use_container_width=True)

            # ── Cumulative Failure Curve — All 3 Models ────────────────
            st.subheader("Cumulative Failure Probability by Month")
            fig_cum = go.Figure()
            for mkey in ["rbb", "garch", "gbm"]:
                if mkey not in _bcr_models:
                    continue
                mr = _bcr_models[mkey]
                cum_fail = np.zeros(n_months)
                for m in range(n_months):
                    cum_fail[m] = float(np.sum(mr["failure_month"] <= (m + 1))) / mr["n_sims"] * 100
                fig_cum.add_trace(go.Scatter(
                    x=list(range(1, n_months + 1)), y=cum_fail.tolist(),
                    mode="lines", line=dict(color=_BCR_MODEL_COLORS[mkey], width=2),
                    name=_BCR_MODEL_LABELS[mkey],
                ))
            fig_cum.update_layout(
                title="Probability of Having Failed by Month X — All Models",
                xaxis_title="Month", yaxis_title="Cumulative Failure (%)",
                height=400, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_cum, use_container_width=True)

            # ── Sensitivity Heatmap (RBB only — production model) ──────
            st.subheader("Sensitivity Heatmap — Failure Probability (RBB)")
            st.markdown(
                f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
                f"Failure probability across BCR x Dividend Rate combinations. "
                f"Uses RBB price paths (production model).</span>",
                unsafe_allow_html=True,
            )

            rbb_monthly = _bcr_models["rbb"]["monthly_prices"]
            n_sims_heat = _bcr_models["rbb"]["n_sims"]
            _heat_bcrs = [10, 20, 30, 40, 50, 60, 80, 100]
            _heat_divs = [0.05, 0.08, 0.10, 0.12, 0.15]
            _heat_z = np.zeros((len(_heat_divs), len(_heat_bcrs)))

            for di, dr in enumerate(_heat_divs):
                for bi, br in enumerate(_heat_bcrs):
                    h_ppe_lev = 1.0 / (br * dr) if (br * dr) > 0 else 0
                    h_ppe_not = _bcr_params["nav"] * h_ppe_lev
                    h_ann_div = h_ppe_not * dr
                    h_ann_opex = _bcr_params["nav"] * _bcr_params["opex_rate"]
                    h_monthly = (h_ann_div / 12.0) + (h_ann_opex / 12.0)

                    n_fail = 0
                    for i in range(n_sims_heat):
                        btc_h = _bcr_params["nav"] / _init_px
                        cash_h = _bcr_params["cash_months"] * h_monthly
                        path_ok = True
                        for m in range(n_months):
                            price = rbb_monthly[i, m]
                            btc_obl = h_monthly * _bcr_params["btc_fraction"]
                            if cash_h >= btc_obl:
                                cash_h -= btc_obl
                            else:
                                sell_amt = btc_obl - cash_h
                                cash_h = 0.0
                                if price > 0:
                                    btc_h -= sell_amt / price
                            btc_val = btc_h * price
                            cur_bcr = btc_val / h_ann_div if h_ann_div > 0 else float("inf")
                            if _bcr_params["fail_mode"] == "BCR < 1":
                                if cur_bcr < 1.0:
                                    path_ok = False
                                    break
                            else:
                                if btc_val < h_ppe_not:
                                    path_ok = False
                                    break
                        if not path_ok:
                            n_fail += 1
                    _heat_z[di, bi] = n_fail / n_sims_heat * 100

            fig_heat = go.Figure(data=go.Heatmap(
                z=_heat_z,
                x=[f"{b}x" for b in _heat_bcrs],
                y=[f"{d*100:.0f}%" for d in _heat_divs],
                colorscale=[[0, "#0a0a0a"], [0.01, "#10b981"], [0.05, "#eab308"],
                            [0.2, "#f7931a"], [0.5, "#ef4444"], [1.0, "#7f1d1d"]],
                text=[[f"{v:.1f}%" for v in row] for row in _heat_z],
                texttemplate="%{text}",
                textfont=dict(size=12, color="#ffffff"),
                colorbar=dict(title="Fail %", tickfont=dict(color="#cccccc"),
                              titlefont=dict(color="#cccccc")),
                zmin=0, zmax=max(100, float(_heat_z.max())),
            ))
            fig_heat.update_layout(
                title="4-Year Failure Probability: BCR vs Dividend Rate",
                xaxis_title="Bitcoin Coverage Ratio",
                yaxis_title="Annual Dividend Rate",
                height=450, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # ── What This Means ────────────────────────────────────────
            _rbb_mr = _bcr_models["rbb"]
            _rbb_fail_pct = int(np.sum(_rbb_mr["failed"])) / _rbb_mr["n_sims"] * 100
            st.markdown(
                f"<div style='background:{PANEL};border-left:3px solid {BTC_ORANGE};"
                f"border-radius:4px;padding:14px;margin-top:12px;'>"
                f"<span style='color:{BTC_ORANGE};font-weight:700;font-size:0.95rem;'>"
                f"What This Means</span><br><br>"
                f"<span style='color:{TEXT};font-size:0.82rem;'>"
                f"With a <b>${_bcr_params['nav']/1e9:.1f}B</b> BTC treasury, "
                f"<b>{_bcr_params['bcr']:.0f}x</b> BCR, "
                f"and <b>{_bcr_params['div_rate']*100:.0f}%</b> dividend rate:</span>"
                f"<ul style='color:{TEXT};font-size:0.82rem;margin:8px 0;padding-left:20px;'>"
                f"<li>PPE issuance: <b>${_bcr_params['ppe_notional']:,.0f}</b> notional, "
                f"paying <b>${_bcr_params['annual_dividend']:,.0f}/yr</b> in dividends</li>"
                f"<li>RBB (production model): <b>{_rbb_fail_pct:.1f}%</b> failure rate | "
                f"Median terminal BCR: <b>{np.median(_rbb_mr['terminal_bcr']):.1f}x</b></li>"
                f"<li>At constant BTC price, the treasury has <b>{runway_months} months</b> of runway</li>"
                f"</ul>"
                f"<span style='color:{TEXT_DIM};font-size:0.78rem;font-style:italic;'>"
                f"BCR comparison across models shows how different price path assumptions "
                f"affect solvency outcomes. RBB paths are the most realistic (walk-forward 0.8109)."
                f"</span></div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════
# TAB: Phase Documentation (Phase 2 Regime + Phase 3 Models combined)
# ═══════════════════════════════════════════════════════════════════════
with tab_docs:
    st.header("Phase 2 — Regime Conditioning Comparison")
    st.markdown(
        f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
        f"Side-by-side comparison of all tested regime configurations. "
        f"Same locked block params (geometric bl=30), same walk-forward pipeline "
        f"(5000 sims, 90d step, exponential recency weighting)."
        f"</span>", unsafe_allow_html=True,
    )

    # ── Load all three config files ────────────────────────────────
    _regime_configs = {
        "Config 1: No regime (baseline)": "results/phase2/config1/phase2_config1_standard.json",
        "Config 2: 2-state MSM": "results/phase2/config2/phase2_config2_standard.json",
        "Config 6: 3-state HMM": "results/phase2/config6/phase2_config6_standard.json",
    }
    _regime_short = {
        "Config 1: No regime (baseline)": "C1 (none)",
        "Config 2: 2-state MSM": "C2 (2-MSM)",
        "Config 6: 3-state HMM": "C6 (3-HMM)",
    }
    _regime_data = {}
    _missing = []
    for label, path in _regime_configs.items():
        p = Path(path)
        if p.exists():
            with open(p) as f:
                _regime_data[label] = json.load(f)
        else:
            _missing.append(path)

    if _missing:
        st.warning(f"Missing config data: {_missing}")
    elif len(_regime_data) < 2:
        st.warning("Need at least 2 configs for comparison.")
    else:
        # ── Phase 1 Context ────────────────────────────────────────
        with st.expander("Phase 1 Context: Block Length Selection", expanded=False):
            st.markdown(
                "**What we tested:** 30 configurations (15 block lengths from 10-180d "
                "x 2 sampling methods: geometric and fixed) with `regime_enabled=False`. "
                "Each config applied across all four horizons (180d, 365d, 730d, 1460d) simultaneously.\n\n"
                "**What we found:** Scores were remarkably flat. Spread from #1 to #30 was only "
                "0.0070 (0.8779 to 0.8709). The model is robust to block length choice. "
                "Geometric sampling consistently beat fixed by tiny margins. Shorter blocks "
                "(10-30d) favored tail accuracy and MAPE; longer blocks (80-180d) favored "
                "drawdown duration and KS statistic. These tradeoffs cancel in the composite.\n\n"
                "**Decision:** Selected **geometric bl=30d** — scores within 0.0004 of #1, "
                "smallest blocks for maximum resampling diversity, least likely to overfit "
                "to specific historical sequences."
            )

        # ── Phase 2 Narrative ──────────────────────────────────────
        st.subheader("Phase 2 Narrative")
        st.markdown(
            "### What we tested\n\n"
            "Whether constraining the sequence in which historical return blocks are chained — "
            "using regime labels to control which pool blocks are drawn from — produces more "
            "realistic Bitcoin price paths than drawing from one undifferentiated pool.\n\n"
            "Three configurations were tested against a common baseline:\n"
            "- **Config 1 (baseline):** Single pool of all historical returns. No regime labels, "
            "no transition logic. Every block has equal probability of being drawn at any time.\n"
            "- **Config 2:** 2-state Markov Switching Model (MSM) classifies weeks as bull or bear. "
            "Daily returns from bull weeks go into the bull pool, bear weeks into the bear pool. "
            "A fitted 2x2 transition matrix governs switching during simulation.\n"
            "- **Config 6:** 3-state Gaussian HMM via Baum-Welch EM. Three pools with a 3x3 "
            "transition matrix. Intended to create a dedicated crisis pool for extreme events.\n\n"
            "### Why we tested it\n\n"
            "The hypothesis was that regime-conditioned block drawing would produce more realistic "
            "path dynamics — particularly drawdown duration and recovery patterns. Bear-market blocks "
            "would only appear during simulated bear regimes, preserving the temporal clustering of "
            "negative returns observed in real BTC cycles. A dedicated crisis pool (Config 6) was "
            "expected to improve reproduction of extreme historical scenarios (2018 crash, 2022 crash, "
            "Mt Gox collapse) that the single-pool model struggled to produce.\n\n"
            "### What we found\n\n"
            "**Composite scores are near-identical across all configs.** Config 1: 0.8109, "
            "Config 2: 0.8118 (+0.0009), Config 6: 0.8140 (+0.0031). The differences are within "
            "simulation noise.\n\n"
            "**But tail event reproduction — the metric that matters most for stress-testing "
            "Bitcoin-collateralized debt — degrades monotonically with more pools.** At the 1460d "
            "horizon:\n"
            "- 2022 crash reproduction: 23.3% (1 pool) → 14.1% (2 pools) → 12.7% (3 pools)\n"
            "- Drawdowns ≥ 75%: 28.9% → 19.5% → 17.5%\n"
            "- Duration ≥ 365d: 64.4% → 55.1% → 51.6%\n\n"
            "### Why regime OFF won\n\n"
            "**1. Pool splitting constrains draws, reducing path diversity and tail event frequency.** "
            "When regime is ON, the model must draw from the current regime's pool. The bull pool "
            "gets ~53% of stationary probability. A simulated path spends the majority of its time "
            "locked into bull-regime blocks — crash blocks can only appear after the transition "
            "matrix flips to bear. This reduces the frequency and severity of extreme paths "
            "relative to a single pool where any block can follow any other block.\n\n"
            "**2. Baum-Welch found volatility regimes, not directional regimes.** The 3-state HMM "
            "separated returns by volatility level (explosive/calm/normal), not by direction "
            "(bull/bear/crisis). Extreme crash days (-46%, -20%) and explosive rally days ended up "
            "in the same high-volatility state. The intended crisis pool never materialized — "
            "there is no clean statistical separation between crash dynamics and rally dynamics "
            "in BTC returns.\n\n"
            "**3. The block bootstrap already preserves volatility clustering within blocks.** "
            "A 30-day block drawn from a volatile period carries that period's autocorrelation "
            "structure, fat tails, and clustering naturally. Regime-based classification adds a "
            "second layer of volatility grouping on top of what the blocks already capture, "
            "providing no new information while constraining the draw sequence.\n\n"
            "**4. Composite scores are insensitive to regime conditioning, but tail reproduction "
            "is not.** The 9-metric composite averages across path dynamics, distributional fit, "
            "tail risk, and forecast accuracy. Regime ON slightly improves some metrics (e.g., "
            "MAPE at 1460d) while slightly degrading others (tail index, percentile band coverage). "
            "These offset in the composite — but for the downstream use case (stress-testing "
            "BCR thresholds), tail reproduction is the binding constraint, not the composite average."
        )

        st.divider()

        # ── Cross-Horizon Final Scores ─────────────────────────────
        st.subheader("Cross-Horizon Final Scores")

        _ch_rows = []
        _c1_final = 0
        for label, d in _regime_data.items():
            ch = d["rbb"]["cross_horizon"]
            if "Config 1" in label:
                _c1_final = ch["final_score"]
            _ch_rows.append({
                "Config": _regime_short[label],
                "Final Score": ch["final_score"],
                "Composite": ch["weighted_composite"],
                "Penalty": ch["stability_penalty"],
                "Std": ch["horizon_std"],
            })
        for row in _ch_rows:
            row["vs Baseline"] = row["Final Score"] - _c1_final

        ch_df = pd.DataFrame(_ch_rows)
        st.dataframe(
            ch_df.style.format({
                "Final Score": "{:.4f}", "Composite": "{:.4f}",
                "Penalty": "{:.4f}", "Std": "{:.4f}", "vs Baseline": "{:+.4f}",
            }).background_gradient(subset=["Final Score"], cmap="YlOrRd"),
            use_container_width=True, hide_index=True,
        )

        # ── Per-Horizon Composites ─────────────────────────────────
        st.subheader("Per-Horizon Walk-Forward Composites")

        _hz_rows = []
        _c1_per_hz = {}
        for label, d in _regime_data.items():
            for hz in ["180d", "365d", "730d", "1460d"]:
                wf_comp = d["rbb"]["per_horizon"][hz]["weighted_composite"]
                n_win = d["rbb"]["per_horizon"][hz]["n_windows"]
                if "Config 1" in label:
                    _c1_per_hz[hz] = wf_comp
                _hz_rows.append({
                    "Horizon": hz,
                    "Config": _regime_short[label],
                    "WF Composite": wf_comp,
                    "Windows": n_win,
                })
        for row in _hz_rows:
            row["vs C1"] = row["WF Composite"] - _c1_per_hz.get(row["Horizon"], 0)

        hz_df = pd.DataFrame(_hz_rows)
        hz_pivot = hz_df.pivot(index="Horizon", columns="Config", values="WF Composite")
        # Reorder columns
        col_order = [_regime_short[k] for k in _regime_data.keys()]
        hz_pivot = hz_pivot[[c for c in col_order if c in hz_pivot.columns]]
        st.dataframe(
            hz_pivot.style.format("{:.4f}").background_gradient(axis=1, cmap="YlOrRd"),
            use_container_width=True,
        )

        # ── Tail Event Comparison @ 1460d ──────────────────────────
        st.subheader("Tail Event Reproduction Rates @ 1460d")
        st.markdown(
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
            f"From the final OOS scorecard (single simulation, full train/test split, 5000 paths). "
            f"Higher = model produces more crash-like paths = better for stress testing."
            f"</span>", unsafe_allow_html=True,
        )

        _te_rows = []
        _c1_te = {}
        for label, d in _regime_data.items():
            te = d["rbb"]["per_horizon"]["1460d"].get("final_distributions", {}).get("tail_events", {})
            if not te:
                continue
            short = _regime_short[label]
            n = te["n_paths"]

            for key, lbl in [("dd_50pct", "DD >= 50%"), ("dd_75pct", "DD >= 75%"),
                              ("dur_180d", "Duration >= 180d"), ("dur_365d", "Duration >= 365d")]:
                pct = te["generic"][key]["pct"]
                if "Config 1" in label:
                    _c1_te[lbl] = pct
                _te_rows.append({"Event": lbl, "Config": short, "Rate": pct})

            for key in ["2014_mt_gox", "2018_crash", "2022_crash", "2020_flash_crash"]:
                sc = te["named_scenarios"][key]
                lbl = key.replace("_", " ").title()
                if "Config 1" in label:
                    _c1_te[lbl] = sc["pct"]
                _te_rows.append({"Event": lbl, "Config": short, "Rate": sc["pct"]})

        if _te_rows:
            te_df = pd.DataFrame(_te_rows)
            te_pivot = te_df.pivot(index="Event", columns="Config", values="Rate")
            te_pivot = te_pivot[[c for c in col_order if c in te_pivot.columns]]

            # Add delta column
            if "C1 (none)" in te_pivot.columns:
                for col in te_pivot.columns:
                    if col != "C1 (none)":
                        te_pivot[f"{col} vs C1"] = te_pivot[col] - te_pivot["C1 (none)"]

            # Reorder rows
            event_order = ["DD >= 50%", "DD >= 75%", "Duration >= 180d", "Duration >= 365d",
                           "2014 Mt Gox", "2018 Crash", "2022 Crash", "2020 Flash Crash"]
            te_pivot = te_pivot.reindex([e for e in event_order if e in te_pivot.index])

            st.dataframe(
                te_pivot.style.format("{:.1f}").background_gradient(
                    subset=[c for c in te_pivot.columns if "vs C1" not in c],
                    cmap="YlOrRd",
                ),
                use_container_width=True,
            )

        # ── Per-Metric Comparison @ 1460d ──────────────────────────
        st.subheader("Per-Metric Mean Scores by Horizon")
        st.markdown(
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
            f"Mean of each scoring metric across all walk-forward windows per horizon. "
            f"Shows where regime conditioning helps or hurts at each timescale."
            f"</span>", unsafe_allow_html=True,
        )

        _metric_rows = {}  # keyed by horizon
        for hz in ["180d", "365d", "730d", "1460d"]:
            _metric_rows[hz] = []
            for label, d in _regime_data.items():
                windows = d["rbb"]["per_horizon"][hz]["window_details"]
                short = _regime_short[label]
                metric_agg = {}
                for w in windows:
                    for mk, mv in w.get("scoring_metrics", {}).items():
                        metric_agg.setdefault(mk, []).append(mv)
                for mk, vals in metric_agg.items():
                    _metric_rows[hz].append({
                        "Metric": mk,
                        "Config": short,
                        "Mean": float(np.mean(vals)),
                    })

        _metric_hz_tabs = st.tabs(["180d", "365d", "730d", "1460d"])
        for hz_tab, hz in zip(_metric_hz_tabs, ["180d", "365d", "730d", "1460d"]):
            with hz_tab:
                rows = _metric_rows[hz]
                if rows:
                    met_df = pd.DataFrame(rows)
                    met_pivot = met_df.pivot(index="Metric", columns="Config", values="Mean")
                    met_pivot = met_pivot[[c for c in col_order if c in met_pivot.columns]]
                    if "C1 (none)" in met_pivot.columns:
                        for col in met_pivot.columns:
                            if col != "C1 (none)":
                                met_pivot[f"{col} vs C1"] = met_pivot[col] - met_pivot["C1 (none)"]
                    st.dataframe(
                        met_pivot.style.format("{:.4f}").background_gradient(
                            subset=[c for c in met_pivot.columns if "vs C1" not in c],
                            axis=1, cmap="YlOrRd",
                        ),
                        use_container_width=True,
                    )

        # ── Per-Window Comparison @ 1460d ──────────────────────────
        st.subheader("Per-Window Scores @ 1460d (sorted by Config 1 ascending)")
        st.markdown(
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
            f"All 23 walk-forward windows at 1460d. Sorted by Config 1 score ascending "
            f"to show whether regime conditioning helped on the weakest windows."
            f"</span>", unsafe_allow_html=True,
        )

        # Build per-window table
        _win_data = {}  # keyed by test_start_date
        for label, d in _regime_data.items():
            short = _regime_short[label]
            windows = d["rbb"]["per_horizon"]["1460d"]["window_details"]
            for w in windows:
                key = w.get("test_start_date", str(w["window_num"]))
                if key not in _win_data:
                    _win_data[key] = {"Window": f"W{w['window_num']}", "Test Start": key}
                _win_data[key][short] = w["composite_score"]

        if _win_data:
            win_df = pd.DataFrame(list(_win_data.values()))
            win_df = win_df.sort_values("C1 (none)", ascending=True).reset_index(drop=True)

            # Add delta columns
            for col in ["C2 (2-MSM)", "C6 (3-HMM)"]:
                if col in win_df.columns:
                    win_df[f"{col} vs C1"] = win_df[col] - win_df["C1 (none)"]

            st.dataframe(
                win_df.style.format({
                    c: "{:.4f}" for c in win_df.columns if c not in ("Window", "Test Start")
                }).background_gradient(
                    subset=[c for c in win_df.columns if c in col_order],
                    axis=1, cmap="YlOrRd",
                ),
                use_container_width=True, hide_index=True,
            )

            # ── Scatter: Config 2 & 6 vs Config 1 ─────────────────
            st.markdown("")
            fig_scatter = go.Figure()

            if "C2 (2-MSM)" in win_df.columns:
                fig_scatter.add_trace(go.Scatter(
                    x=win_df["C1 (none)"], y=win_df["C2 (2-MSM)"],
                    mode="markers+text",
                    marker=dict(color=BLUE, size=9, opacity=0.8),
                    text=win_df["Window"],
                    textposition="top center",
                    textfont=dict(size=8, color=TEXT_DIM),
                    name="C2 (2-state MSM)",
                    hovertemplate="C1: %{x:.4f}<br>C2: %{y:.4f}<br>%{text}<extra></extra>",
                ))

            if "C6 (3-HMM)" in win_df.columns:
                fig_scatter.add_trace(go.Scatter(
                    x=win_df["C1 (none)"], y=win_df["C6 (3-HMM)"],
                    mode="markers+text",
                    marker=dict(color=BTC_ORANGE, size=9, symbol="diamond", opacity=0.8),
                    text=win_df["Window"],
                    textposition="bottom center",
                    textfont=dict(size=8, color=TEXT_DIM),
                    name="C6 (3-state HMM)",
                    hovertemplate="C1: %{x:.4f}<br>C6: %{y:.4f}<br>%{text}<extra></extra>",
                ))

            # Diagonal line (y=x): points above = regime beats baseline
            _min = min(win_df["C1 (none)"].min(), 0.55)
            _max = max(win_df["C1 (none)"].max(), 0.92)
            fig_scatter.add_trace(go.Scatter(
                x=[_min, _max], y=[_min, _max],
                mode="lines", line=dict(color=TEXT_DIM, width=1, dash="dash"),
                showlegend=False,
            ))

            fig_scatter.update_layout(
                title="Per-Window: Regime Config vs Baseline (above diagonal = regime wins)",
                xaxis_title="Config 1 (no regime) composite",
                yaxis_title="Regime config composite",
                height=550,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Win/loss summary
            for col, label in [("C2 (2-MSM)", "Config 2"), ("C6 (3-HMM)", "Config 6")]:
                delta_col = f"{col} vs C1"
                if delta_col in win_df.columns:
                    wins = (win_df[delta_col] > 0).sum()
                    losses = (win_df[delta_col] < 0).sum()
                    ties = (win_df[delta_col] == 0).sum()
                    avg_delta = win_df[delta_col].mean()
                    # Wins on weakest 5
                    weak5 = win_df.head(5)
                    weak5_wins = (weak5[delta_col] > 0).sum()
                    st.markdown(
                        f"**{label}:** {wins}W / {losses}L / {ties}T "
                        f"(avg delta: {avg_delta:+.4f}) — "
                        f"wins on {weak5_wins}/5 weakest C1 windows"
                    )

        # ── Decision ───────────────────────────────────────────────
        st.divider()
        st.subheader("Decision")
        st.markdown(
            f"**Lock `regime_enabled=False` for Phase 3.** "
            f"Config 1 (single-pool pure block bootstrap, {_c1_final:.4f}) is the production configuration. "
            f"Regime conditioning in any form (2-state MSM or 3-state HMM) systematically suppresses "
            f"tail event reproduction while providing negligible composite score improvement. "
            f"The tail suppression is structural: splitting returns into regime pools constrains "
            f"which blocks can appear when, reducing the frequency of extreme paths."
        )

        # ── Export Signal ──────────────────────────────────────────
        st.divider()
        st.subheader("Export Phase 2 Signal")

        def _build_regime_signal() -> str:
            sig = []
            sig.append(f"# BTC Sim Engine — Phase 1 + Phase 2 Summary ({datetime.now().strftime('%Y-%m-%d')})")
            sig.append("")

            # Phase 1 Context
            sig.append("## Phase 1: Block Length Selection")
            sig.append("")
            sig.append("Tested 30 configurations (15 block lengths from 10-180d x 2 sampling methods: "
                       "geometric and fixed) with `regime_enabled=False`. Each config applied across "
                       "all four horizons (180d, 365d, 730d, 1460d) simultaneously.")
            sig.append("")
            sig.append("**Finding:** Scores remarkably flat. Spread from #1 to #30 was only 0.0070 "
                       "(0.8779 to 0.8709). The model is robust to block length choice. "
                       "Geometric sampling consistently beat fixed by tiny margins. Shorter blocks "
                       "(10-30d) favored tail accuracy and MAPE; longer blocks (80-180d) favored "
                       "drawdown duration and KS statistic. Tradeoffs cancel in the composite.")
            sig.append("")
            sig.append("**Decision:** Selected **geometric bl=30d** — scores within 0.0004 of #1, "
                       "smallest blocks for maximum resampling diversity, least likely to overfit.")
            sig.append("")

            # Phase 2 Narrative
            sig.append("## Phase 2: Regime Conditioning — Narrative")
            sig.append("")
            sig.append("### What we tested")
            sig.append("Whether constraining the sequence in which historical return blocks are chained — "
                       "using regime labels to control which pool blocks are drawn from — produces more "
                       "realistic Bitcoin price paths than drawing from one undifferentiated pool.")
            sig.append("")
            sig.append("### Why we tested it")
            sig.append("The hypothesis was that regime-conditioned block drawing would produce more "
                       "realistic path dynamics — particularly drawdown duration and recovery patterns. "
                       "A dedicated crisis pool (Config 6) was expected to improve reproduction of "
                       "extreme historical scenarios (2018 crash, 2022 crash, Mt Gox collapse).")
            sig.append("")
            sig.append("### What we found")
            sig.append("**Composite scores are near-identical across all configs.** Config 1: 0.8109, "
                       "Config 2: 0.8118 (+0.0009), Config 6: 0.8140 (+0.0031). Differences are "
                       "within simulation noise.")
            sig.append("")
            sig.append("**Tail event reproduction — the metric that matters most for stress-testing "
                       "Bitcoin-collateralized debt — degrades monotonically with more pools.** At 1460d:")
            sig.append("- 2022 crash reproduction: 23.3% (1 pool) -> 14.1% (2 pools) -> 12.7% (3 pools)")
            sig.append("- Drawdowns >= 75%: 28.9% -> 19.5% -> 17.5%")
            sig.append("- Duration >= 365d: 64.4% -> 55.1% -> 51.6%")
            sig.append("")
            sig.append("### Why regime OFF won")
            sig.append("1. **Pool splitting constrains draws, reducing path diversity and tail event "
                       "frequency.** The bull pool gets ~53% of stationary probability. Crash blocks "
                       "can only appear after the transition matrix flips to bear.")
            sig.append("2. **Baum-Welch found volatility regimes, not directional regimes.** Crashes "
                       "and rallies share the same high-vol state. The intended crisis pool never "
                       "materialized.")
            sig.append("3. **The block bootstrap already preserves volatility clustering within blocks.** "
                       "A 30-day block carries that period's autocorrelation structure naturally. "
                       "Regime-based classification adds redundant volatility grouping.")
            sig.append("4. **Composite scores are insensitive to regime conditioning, but tail "
                       "reproduction is not.** For BCR stress-testing, tail reproduction is the "
                       "binding constraint.")
            sig.append("")

            # Experiment setup
            sig.append("## Experiment Setup")
            sig.append("- **Shared config:** geometric bl=30, 5000 sims, 90d step, exponential recency weighting (2yr half-life)")
            sig.append("- **Horizons:** 180d, 365d, 730d, 1460d")
            sig.append("- **Horizon weights:** 180d=10%, 365d=20%, 730d=35%, 1460d=35%")
            sig.append("")
            sig.append("| Config | Description | Method | States |")
            sig.append("|--------|-------------|--------|--------|")
            sig.append("| Config 1 | No regime (baseline) | n/a | 1 pool |")
            sig.append("| Config 2 | Regime ON | 2-state MSM, fitted transition, variance switching ON | 2 pools |")
            sig.append("| Config 6 | Regime ON | 3-state Baum-Welch HMM, fitted transition, variance switching ON | 3 pools |")
            sig.append("")

            # Cross-horizon
            sig.append("## Cross-Horizon Final Scores")
            sig.append("| Config | Final Score | Composite | Penalty | Std | vs C1 |")
            sig.append("|--------|------------|-----------|---------|-----|-------|")
            for row in _ch_rows:
                sig.append(
                    f"| {row['Config']} | **{row['Final Score']:.4f}** | "
                    f"{row['Composite']:.4f} | {row['Penalty']:.4f} | "
                    f"{row['Std']:.4f} | {row['vs Baseline']:+.4f} |"
                )
            sig.append("")

            # Per-horizon
            sig.append("## Per-Horizon Composites")
            sig.append("| Horizon | " + " | ".join(col_order) + " |")
            sig.append("|---------|" + "|".join(["-------"] * len(col_order)) + "|")
            for hz in ["180d", "365d", "730d", "1460d"]:
                vals = []
                for label, d in _regime_data.items():
                    short = _regime_short[label]
                    v = d["rbb"]["per_horizon"][hz]["weighted_composite"]
                    vals.append(f"{v:.4f}")
                sig.append(f"| {hz} | " + " | ".join(vals) + " |")
            sig.append("")

            # Tail events
            sig.append("## Tail Event Reproduction @ 1460d")
            sig.append("Higher = model produces more crash-like paths = better for stress testing.")
            sig.append("")
            if _te_rows:
                sig.append("| Event | " + " | ".join(col_order) + " | C6 vs C1 |")
                sig.append("|-------|" + "|".join(["------"] * len(col_order)) + "|----------|")
                for event in ["DD >= 50%", "DD >= 75%", "Duration >= 180d", "Duration >= 365d",
                              "2014 Mt Gox", "2018 Crash", "2022 Crash", "2020 Flash Crash"]:
                    vals = []
                    c1_val = 0
                    c6_val = 0
                    for label, d in _regime_data.items():
                        short = _regime_short[label]
                        te = d["rbb"]["per_horizon"]["1460d"].get("final_distributions", {}).get("tail_events", {})
                        if not te:
                            vals.append("n/a")
                            continue
                        # Find the matching event
                        found = False
                        for key, lbl in [("dd_50pct", "DD >= 50%"), ("dd_75pct", "DD >= 75%"),
                                          ("dur_180d", "Duration >= 180d"), ("dur_365d", "Duration >= 365d")]:
                            if lbl == event:
                                pct = te["generic"][key]["pct"]
                                vals.append(f"{pct:.1f}%")
                                if "Config 1" in label:
                                    c1_val = pct
                                if "Config 6" in label:
                                    c6_val = pct
                                found = True
                                break
                        if not found:
                            for key in ["2014_mt_gox", "2018_crash", "2022_crash", "2020_flash_crash"]:
                                if key.replace("_", " ").title() == event:
                                    pct = te["named_scenarios"][key]["pct"]
                                    vals.append(f"{pct:.1f}%")
                                    if "Config 1" in label:
                                        c1_val = pct
                                    if "Config 6" in label:
                                        c6_val = pct
                                    break
                    delta = c6_val - c1_val
                    sig.append(f"| {event} | " + " | ".join(vals) + f" | {delta:+.1f}% |")
            sig.append("")

            # Per-metric
            for hz in ["180d", "365d", "730d", "1460d"]:
                n_win = _regime_data[list(_regime_data.keys())[0]]["rbb"]["per_horizon"][hz]["n_windows"]
                sig.append(f"## Per-Metric Mean Scores @ {hz} ({n_win} windows)")
                hz_met_rows = _metric_rows.get(hz, [])
                if hz_met_rows:
                    sig.append("| Metric | " + " | ".join(col_order) + " |")
                    sig.append("|--------|" + "|".join(["------"] * len(col_order)) + "|")
                    metrics_by_name = {}
                    for row in hz_met_rows:
                        metrics_by_name.setdefault(row["Metric"], {})[row["Config"]] = row["Mean"]
                    for mk in sorted(metrics_by_name.keys()):
                        vals = [f"{metrics_by_name[mk].get(c, 0):.4f}" for c in col_order]
                        sig.append(f"| {mk} | " + " | ".join(vals) + " |")
                sig.append("")

            # Per-window
            sig.append("## Per-Window Scores @ 1460d (sorted by C1 ascending)")
            if _win_data:
                sig.append("| Window | Test Start | C1 | C2 | C6 | C2 vs C1 | C6 vs C1 |")
                sig.append("|--------|-----------|------|------|------|----------|----------|")
                sorted_wins = sorted(_win_data.values(), key=lambda x: x.get("C1 (none)", 0))
                for w in sorted_wins:
                    c1v = w.get("C1 (none)", 0)
                    c2v = w.get("C2 (2-MSM)", 0)
                    c6v = w.get("C6 (3-HMM)", 0)
                    sig.append(
                        f"| {w['Window']} | {w['Test Start']} | "
                        f"{c1v:.4f} | {c2v:.4f} | {c6v:.4f} | "
                        f"{c2v - c1v:+.4f} | {c6v - c1v:+.4f} |"
                    )

                # Win/loss summary
                sig.append("")
                sig.append("### Win/Loss Record vs Config 1")
                for col_name, label in [("C2 (2-MSM)", "Config 2"), ("C6 (3-HMM)", "Config 6")]:
                    wins = sum(1 for w in sorted_wins if w.get(col_name, 0) > w.get("C1 (none)", 0))
                    losses = sum(1 for w in sorted_wins if w.get(col_name, 0) < w.get("C1 (none)", 0))
                    avg_d = np.mean([w.get(col_name, 0) - w.get("C1 (none)", 0) for w in sorted_wins])
                    weak5_wins = sum(1 for w in sorted_wins[:5]
                                     if w.get(col_name, 0) > w.get("C1 (none)", 0))
                    sig.append(
                        f"- **{label}:** {wins}W / {losses}L "
                        f"(avg delta: {avg_d:+.4f}) — "
                        f"wins on {weak5_wins}/5 weakest C1 windows"
                    )
            sig.append("")

            # Decision
            sig.append("## Decision")
            sig.append(
                f"**Lock `regime_enabled=False` for Phase 3.** "
                f"Config 1 (single-pool pure block bootstrap, {_c1_final:.4f}) is the production configuration. "
                f"Regime conditioning in any form (2-state MSM or 3-state HMM) systematically suppresses "
                f"tail event reproduction while providing negligible composite score improvement. "
                f"The tail suppression is structural: splitting returns into regime pools constrains "
                f"which blocks can appear when, reducing the frequency of extreme paths."
            )

            return "\n".join(sig)

        _regime_signal = _build_regime_signal()

        st.download_button(
            "Download Phase 2 Signal",
            data=_regime_signal,
            file_name=f"btc_sim_phase1_phase2_summary_{datetime.now().strftime('%Y-%m-%d')}.md",
            mime="text/markdown",
            key="download_regime_signal",
        )
        with st.expander("Preview Signal"):
            st.markdown(
                f"<pre style='background:#111;color:#e0e0e0;padding:1rem;"
                f"font-size:0.75rem;max-height:500px;overflow-y:auto;'>"
                f"{_regime_signal}</pre>",
                unsafe_allow_html=True,
            )


    # ═══════════════════════════════════════════════════════════════════
    # Phase 3 Model Comparison (continued in same tab)
    # ═══════════════════════════════════════════════════════════════════
    st.divider()
    st.header("Phase 3 — Three-Model Comparison")
    st.markdown(
        f"<span style='color:{TEXT_DIM};font-size:0.85rem;'>"
        f"RBB (block bootstrap) vs GARCH(1,1) vs GBM (log-normal). "
        f"All walk-forward validated: 5000 sims, 90d step, exponential recency weighting."
        f"</span>", unsafe_allow_html=True,
    )

    _p3_files = {
        "rbb": "results/phase2/config1/phase2_config1_standard.json",
        "garch_wf": "results/phase3/garch_walkforward/garch_walkforward_comparison.json",
    }
    _p3_data = {}
    _p3_missing = []
    for key, path in _p3_files.items():
        p = Path(path)
        if p.exists():
            with open(p) as f:
                _p3_data[key] = json.load(f)
        else:
            _p3_missing.append(path)

    if _p3_missing:
        st.warning(f"Missing data: {_p3_missing}")
    else:
        _rbb_d = _p3_data["rbb"]
        _garch_wf = _p3_data["garch_wf"]
        _garch_d = _garch_wf["config_b"]  # locked winner: GARCH(1,1)
        _gbm_d = _garch_wf["gbm"]

        _p3_models = {
            "RBB": {"data": _rbb_d, "key": "rbb", "color": BTC_ORANGE},
            "GARCH(1,1)": {"data": _garch_d, "key": "direct", "color": BLUE},
            "GBM": {"data": _gbm_d, "key": "direct", "color": TEXT_DIM},
        }
        _p3_labels = ["RBB", "GARCH(1,1)", "GBM"]

        def _get_hz(model_data, model_key, hz):
            if model_key == "rbb":
                return model_data["rbb"]["per_horizon"][hz]
            else:
                return model_data["per_horizon"][hz]

        def _get_ch(model_data, model_key):
            if model_key == "rbb":
                return model_data["rbb"]["cross_horizon"]
            else:
                return model_data["cross_horizon"]

        # ── Cross-Horizon Final Scores ─────────────────────────────
        st.subheader("Cross-Horizon Final Scores")
        _ch_rows = []
        for label in _p3_labels:
            md = _p3_models[label]
            ch = _get_ch(md["data"], md["key"])
            _ch_rows.append({
                "Model": label,
                "Final Score": ch["final_score"],
                "Composite": ch["weighted_composite"],
                "Penalty": ch["stability_penalty"],
                "Std": ch["horizon_std"],
            })
        _rbb_final = _ch_rows[0]["Final Score"]
        for row in _ch_rows:
            row["vs RBB"] = row["Final Score"] - _rbb_final

        ch_df = pd.DataFrame(_ch_rows)
        st.dataframe(
            ch_df.style.format({
                "Final Score": "{:.4f}", "Composite": "{:.4f}",
                "Penalty": "{:.4f}", "Std": "{:.4f}", "vs RBB": "{:+.4f}",
            }).background_gradient(subset=["Final Score"], cmap="YlOrRd"),
            use_container_width=True, hide_index=True,
        )

        # ── Per-Horizon Composites ─────────────────────────────────
        st.subheader("Per-Horizon Walk-Forward Composites")
        _hz_rows = []
        for hz in HORIZONS:
            row = {"Horizon": hz}
            for label in _p3_labels:
                md = _p3_models[label]
                hz_data = _get_hz(md["data"], md["key"], hz)
                row[label] = hz_data["weighted_composite"]
            row["GARCH vs RBB"] = row["GARCH(1,1)"] - row["RBB"]
            _hz_rows.append(row)

        hz_df = pd.DataFrame(_hz_rows)
        st.dataframe(
            hz_df.style.format({c: "{:.4f}" for c in hz_df.columns if c != "Horizon"}),
            use_container_width=True, hide_index=True,
        )

        # ── Tail Events @ 1460d ────────────────────────────────────
        st.subheader("Tail Event Reproduction @ 1460d")
        st.markdown(
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
            f"From final OOS scorecard (5000 paths). Higher = model produces more crash-like paths."
            f"</span>", unsafe_allow_html=True,
        )

        _te_rows = []
        _te_data = {}
        for label in _p3_labels:
            md = _p3_models[label]
            hz_data = _get_hz(md["data"], md["key"], "1460d")
            te = hz_data.get("final_distributions", {}).get("tail_events", {})
            _te_data[label] = te
            if not te:
                continue
            for key, lbl in [("dd_50pct", "DD >= 50%"), ("dd_75pct", "DD >= 75%"),
                              ("dur_180d", "Duration >= 180d"), ("dur_365d", "Duration >= 365d")]:
                _te_rows.append({"Event": lbl, "Model": label, "Rate": te["generic"][key]["pct"]})
            for key in ["2014_mt_gox", "2018_crash", "2022_crash", "2020_flash_crash"]:
                sc = te["named_scenarios"][key]
                _te_rows.append({"Event": key.replace("_", " ").title(), "Model": label, "Rate": sc["pct"]})

        if _te_rows:
            te_df = pd.DataFrame(_te_rows)
            te_pivot = te_df.pivot(index="Event", columns="Model", values="Rate")
            te_pivot = te_pivot[[l for l in _p3_labels if l in te_pivot.columns]]
            event_order = ["DD >= 50%", "DD >= 75%", "Duration >= 180d", "Duration >= 365d",
                           "2014 Mt Gox", "2018 Crash", "2022 Crash", "2020 Flash Crash"]
            te_pivot = te_pivot.reindex([e for e in event_order if e in te_pivot.index])
            if "RBB" in te_pivot.columns:
                for col in te_pivot.columns:
                    if col != "RBB":
                        te_pivot[f"{col} vs RBB"] = te_pivot[col] - te_pivot["RBB"]
            st.dataframe(
                te_pivot.style.format("{:.1f}").background_gradient(
                    subset=[c for c in te_pivot.columns if "vs" not in c], cmap="YlOrRd",
                ),
                use_container_width=True,
            )

        # ── Per-Metric Comparison (all horizons) ───────────────────
        st.subheader("Per-Metric Mean Scores (all horizons)")
        _met_hz_tabs = st.tabs(["180d", "365d", "730d", "1460d"])
        for met_tab, hz in zip(_met_hz_tabs, HORIZONS):
            with met_tab:
                _met_rows = []
                for label in _p3_labels:
                    md = _p3_models[label]
                    hz_data = _get_hz(md["data"], md["key"], hz)
                    windows = hz_data.get("window_details", [])
                    metric_agg = {}
                    for w in windows:
                        for mk, mv in w.get("scoring_metrics", {}).items():
                            metric_agg.setdefault(mk, []).append(mv)
                    for mk, vals in metric_agg.items():
                        _met_rows.append({"Metric": mk, "Model": label, "Mean": float(np.mean(vals))})

                if _met_rows:
                    met_df = pd.DataFrame(_met_rows)
                    met_pivot = met_df.pivot(index="Metric", columns="Model", values="Mean")
                    met_pivot = met_pivot[[l for l in _p3_labels if l in met_pivot.columns]]
                    if "RBB" in met_pivot.columns:
                        for col in met_pivot.columns:
                            if col != "RBB":
                                met_pivot[f"{col} vs RBB"] = met_pivot[col] - met_pivot["RBB"]
                    st.dataframe(
                        met_pivot.style.format("{:.4f}").background_gradient(
                            subset=[c for c in met_pivot.columns if "vs" not in c],
                            axis=1, cmap="YlOrRd",
                        ),
                        use_container_width=True,
                    )

        # ── Per-Window Scatter @ 1460d ─────────────────────────────
        st.subheader("Per-Window Scores @ 1460d")
        st.markdown(
            f"<span style='color:{TEXT_DIM};font-size:0.8rem;'>"
            f"Points above the diagonal = model beats RBB on that window."
            f"</span>", unsafe_allow_html=True,
        )

        _win_data = {}
        for label in _p3_labels:
            md = _p3_models[label]
            hz_data = _get_hz(md["data"], md["key"], "1460d")
            for w in hz_data.get("window_details", []):
                key = w.get("test_start_date", str(w["window_num"]))
                if key not in _win_data:
                    _win_data[key] = {"Window": f"W{w['window_num']}", "Test Start": key}
                _win_data[key][label] = w["composite_score"]

        if _win_data:
            win_df = pd.DataFrame(list(_win_data.values()))
            win_df = win_df.sort_values("RBB", ascending=True).reset_index(drop=True)
            for col in ["GARCH(1,1)", "GBM"]:
                if col in win_df.columns:
                    win_df[f"{col} vs RBB"] = win_df[col] - win_df["RBB"]

            st.dataframe(
                win_df.style.format({c: "{:.4f}" for c in win_df.columns if c not in ("Window", "Test Start")}),
                use_container_width=True, hide_index=True,
            )

            # Scatter
            fig_scatter = go.Figure()
            if "GARCH(1,1)" in win_df.columns:
                fig_scatter.add_trace(go.Scatter(
                    x=win_df["RBB"], y=win_df["GARCH(1,1)"],
                    mode="markers+text",
                    marker=dict(color=BLUE, size=9, opacity=0.8),
                    text=win_df["Window"], textposition="top center",
                    textfont=dict(size=8, color=TEXT_DIM),
                    name="GARCH(1,1)",
                    hovertemplate="RBB: %{x:.4f}<br>GARCH: %{y:.4f}<br>%{text}<extra></extra>",
                ))
            if "GBM" in win_df.columns:
                fig_scatter.add_trace(go.Scatter(
                    x=win_df["RBB"], y=win_df["GBM"],
                    mode="markers+text",
                    marker=dict(color=GREEN, size=9, symbol="diamond", opacity=0.6),
                    text=win_df["Window"], textposition="bottom center",
                    textfont=dict(size=8, color=TEXT_DIM),
                    name="GBM",
                    hovertemplate="RBB: %{x:.4f}<br>GBM: %{y:.4f}<br>%{text}<extra></extra>",
                ))
            _mn = min(win_df["RBB"].min(), 0.55)
            _mx = max(win_df["RBB"].max(), 0.92)
            fig_scatter.add_trace(go.Scatter(
                x=[_mn, _mx], y=[_mn, _mx], mode="lines",
                line=dict(color=TEXT_DIM, width=1, dash="dash"), showlegend=False,
            ))
            fig_scatter.update_layout(
                title="Per-Window: GARCH/GBM vs RBB (above diagonal = beats RBB)",
                xaxis_title="RBB composite", yaxis_title="Model composite",
                height=550, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Win/loss
            for col, label in [("GARCH(1,1)", "GARCH"), ("GBM", "GBM")]:
                delta_col = f"{col} vs RBB"
                if delta_col in win_df.columns:
                    wins = (win_df[delta_col] > 0).sum()
                    losses = (win_df[delta_col] < 0).sum()
                    avg_d = win_df[delta_col].mean()
                    weak5 = win_df.head(5)
                    w5_wins = (weak5[delta_col] > 0).sum()
                    st.markdown(
                        f"**{label}:** {wins}W / {losses}L "
                        f"(avg delta: {avg_d:+.4f}) — "
                        f"wins on {w5_wins}/5 weakest RBB windows"
                    )

        # ── Conclusion ─────────────────────────────────────────────
        st.divider()
        st.subheader("Conclusion")
        st.markdown(
            f"**RBB (block bootstrap) wins Phase 3.** Final score 0.8109 vs "
            f"GARCH 0.7800 (-0.031) vs GBM 0.7630 (-0.048). "
            f"RBB's advantage is structural: resampling actual historical return blocks "
            f"preserves BTC's exact distributional shape, volatility clustering, and "
            f"multi-day drawdown patterns. GARCH captures volatility clustering explicitly "
            f"but is limited by its parametric innovation distribution and day-by-day "
            f"generation. The production configuration is locked: RBB with geometric bl=30, "
            f"regime OFF."
        )

        # ── Export Signal ──────────────────────────────────────────
        st.divider()
        st.subheader("Export Phase 3 Signal")

        def _build_p3_signal() -> str:
            sig = []
            sig.append(f"# Phase 3 Model Comparison: RBB vs GARCH vs GBM ({datetime.now().strftime('%Y-%m-%d')})")
            sig.append("")
            sig.append("## Models Tested")
            sig.append("| Model | Type | Key Feature | Walk-Forward Final |")
            sig.append("|-------|------|-------------|-------------------|")
            sig.append("| RBB | Nonparametric block bootstrap | Resamples actual historical return sequences | **0.8109** |")
            sig.append("| GARCH(1,1) | Parametric (t-dist, Constant mean) | Explicit volatility clustering model | 0.7800 |")
            sig.append("| GBM | Log-normal random walk | Constant drift + constant vol | 0.7630 |")
            sig.append("")

            sig.append("## Cross-Horizon Final Scores")
            sig.append("| Model | Final | Composite | Penalty | Std | vs RBB |")
            sig.append("|-------|-------|-----------|---------|-----|--------|")
            for row in _ch_rows:
                sig.append(f"| {row['Model']} | {row['Final Score']:.4f} | {row['Composite']:.4f} | "
                           f"{row['Penalty']:.4f} | {row['Std']:.4f} | {row['vs RBB']:+.4f} |")
            sig.append("")

            sig.append("## Per-Horizon Composites")
            sig.append("| Horizon | RBB | GARCH | GBM | GARCH vs RBB |")
            sig.append("|---------|-----|-------|-----|-------------|")
            for row in _hz_rows:
                sig.append(f"| {row['Horizon']} | {row['RBB']:.4f} | {row['GARCH(1,1)']:.4f} | "
                           f"{row['GBM']:.4f} | {row['GARCH vs RBB']:+.4f} |")
            sig.append("")

            # Tail events
            if _te_data.get("RBB") and _te_data.get("GARCH(1,1)"):
                sig.append("## Tail Event Reproduction @ 1460d")
                sig.append("| Event | RBB | GARCH | GBM |")
                sig.append("|-------|-----|-------|-----|")
                rte = _te_data["RBB"]
                gte = _te_data["GARCH(1,1)"]
                bte = _te_data.get("GBM", {})
                for key, lbl in [("dd_50pct", "DD >= 50%"), ("dd_75pct", "DD >= 75%"),
                                  ("dur_180d", "Dur >= 180d"), ("dur_365d", "Dur >= 365d")]:
                    rv = rte["generic"][key]["pct"]
                    gv = gte["generic"][key]["pct"] if gte else 0
                    bv = bte.get("generic", {}).get(key, {}).get("pct", 0) if bte else 0
                    sig.append(f"| {lbl} | {rv:.1f}% | {gv:.1f}% | {bv:.1f}% |")
                for key in ["2014_mt_gox", "2018_crash", "2022_crash", "2020_flash_crash"]:
                    rv = rte["named_scenarios"][key]["pct"]
                    gv = gte["named_scenarios"][key]["pct"] if gte else 0
                    bv = bte.get("named_scenarios", {}).get(key, {}).get("pct", 0) if bte else 0
                    sig.append(f"| {key.replace('_',' ').title()} | {rv:.1f}% | {gv:.1f}% | {bv:.1f}% |")
                sig.append("")

            # Per-metric at each horizon
            for hz in ["180d", "365d", "730d", "1460d"]:
                sig.append(f"## Per-Metric Means @ {hz}")
                sig.append("| Metric | RBB | GARCH | GBM | GARCH vs RBB |")
                sig.append("|--------|-----|-------|-----|-------------|")
                for label in _p3_labels:
                    md = _p3_models[label]
                    hz_data = _get_hz(md["data"], md["key"], hz)
                    windows = hz_data.get("window_details", [])
                    _agg = {}
                    for w in windows:
                        for mk, mv in w.get("scoring_metrics", {}).items():
                            _agg.setdefault(mk, {}).setdefault(label, []).append(mv)
                # Build rows
                all_metrics = sorted(set(
                    mk for label in _p3_labels
                    for md in [_p3_models[label]]
                    for w in _get_hz(md["data"], md["key"], hz).get("window_details", [])
                    for mk in w.get("scoring_metrics", {}).keys()
                ))
                for mk in all_metrics:
                    vals = {}
                    for label in _p3_labels:
                        md = _p3_models[label]
                        windows = _get_hz(md["data"], md["key"], hz).get("window_details", [])
                        v = [w["scoring_metrics"].get(mk, 0) for w in windows]
                        vals[label] = float(np.mean(v)) if v else 0
                    delta = vals.get("GARCH(1,1)", 0) - vals.get("RBB", 0)
                    sig.append(f"| {mk} | {vals.get('RBB',0):.4f} | {vals.get('GARCH(1,1)',0):.4f} | "
                               f"{vals.get('GBM',0):.4f} | {delta:+.4f} |")
                sig.append("")

            # Per-window at 1460d
            if _win_data:
                sig.append("## Per-Window @ 1460d (sorted by RBB ascending)")
                sig.append("| Window | Test Start | RBB | GARCH | GBM | GARCH vs RBB |")
                sig.append("|--------|-----------|------|-------|------|-------------|")
                sorted_wins = sorted(_win_data.values(), key=lambda x: x.get("RBB", 0))
                for w in sorted_wins:
                    rv = w.get("RBB", 0)
                    gv = w.get("GARCH(1,1)", 0)
                    bv = w.get("GBM", 0)
                    sig.append(f"| {w['Window']} | {w['Test Start']} | {rv:.4f} | {gv:.4f} | "
                               f"{bv:.4f} | {gv - rv:+.4f} |")

                sig.append("")
                for col, label in [("GARCH(1,1)", "GARCH"), ("GBM", "GBM")]:
                    wins = sum(1 for w in sorted_wins if w.get(col, 0) > w.get("RBB", 0))
                    losses = sum(1 for w in sorted_wins if w.get(col, 0) < w.get("RBB", 0))
                    avg_d = np.mean([w.get(col, 0) - w.get("RBB", 0) for w in sorted_wins])
                    w5_wins = sum(1 for w in sorted_wins[:5] if w.get(col, 0) > w.get("RBB", 0))
                    sig.append(f"- **{label}:** {wins}W / {losses}L (avg: {avg_d:+.4f}) — {w5_wins}/5 weakest RBB windows")
                sig.append("")

            sig.append("## Conclusion")
            sig.append("RBB (block bootstrap) wins Phase 3. Final 0.8109 vs GARCH 0.7800 (-0.031) vs GBM 0.7630 (-0.048). "
                       "RBB's advantage is structural: resampling actual historical return blocks preserves BTC's exact "
                       "distributional shape, volatility clustering, and multi-day drawdown patterns. "
                       "Production config locked: RBB geometric bl=30, regime OFF.")

            return "\n".join(sig)

        _p3_signal = _build_p3_signal()
        _p3_date = datetime.now().strftime("%Y-%m-%d")

        st.download_button(
            "Download Phase 3 Signal",
            data=_p3_signal,
            file_name=f"phase3_model_comparison_{_p3_date}.md",
            mime="text/markdown",
            key="download_p3_signal",
        )
        with st.expander("Preview Signal"):
            st.markdown(
                f"<pre style='background:#111;color:#e0e0e0;padding:1rem;"
                f"font-size:0.75rem;max-height:500px;overflow-y:auto;'>"
                f"{_p3_signal}</pre>",
                unsafe_allow_html=True,
            )

