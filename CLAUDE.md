# BTC Price Simulation Engine

Modular, scoring-driven engine for generating realistic **Bitcoin price paths** for digital credit stress-testing (BCR thresholds). Full methodology, results, and analysis in [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Architecture

```
btc-sim-engine/
├── CLAUDE.md                  # This file — operational essentials
├── docs/RESULTS.md            # Full methodology, phase results, analysis
├── config/default.yaml        # Default experiment config
├── data/loader.py             # Data fetching + preprocessing
├── models/
│   ├── base.py                # Abstract base class — THE contract
│   ├── gbm.py                 # Geometric Brownian Motion (baseline)
│   ├── garch.py               # GARCH(1,1) with configurable innovation dist
│   ├── regime_block_bootstrap.py  # Stationary block bootstrap (flagship)
│   ├── hmm.py                 # Pure numpy Baum-Welch Gaussian HMM
│   └── registry.py            # Model registry
├── validation/
│   ├── metrics.py             # 9 scoring metrics → composite score
│   ├── diagnostics.py         # Diagnostic metrics (reported, not scored)
│   ├── scorer.py              # Composite scoring engine
│   ├── backtest.py            # Walk-forward out-of-sample framework
│   ├── distributions.py       # Percentile distribution data
│   └── audit.py               # Pipeline audit for walk-forward windows
├── optimization/
│   ├── search.py              # Sobol + Bayesian (Optuna)
│   ├── sensitivity.py         # Univariate sensitivity analysis
│   └── experiment.py          # Full experiment runner
├── dashboard/app.py           # Streamlit dashboard (9 tabs)
├── reporting/                 # Leaderboard, comparison, export
├── viz/                       # Charts: paths, distributions, diagnostics
├── scripts/                   # CLI entry points
└── tests/
```

---

## Model Interface (`models/base.py`)

Every model MUST implement `BaseModel`:
- `fit(historical_returns: np.ndarray) -> None`
- `simulate(n_simulations, n_steps, initial_price, seed) -> SimulationResult`
- `get_default_params() -> dict`
- `get_param_space() -> dict`
- `name -> str` (property)

`SimulationResult`: paths (n_sims, n_steps), log_returns (n_sims, n_steps-1), params_used, model_name, metadata.

**Key rule:** Validation layer NEVER knows which model generated the paths. Pure black-box scoring.

---

## Implementation Status

| Model | Score | Config | Notes |
|-------|-------|--------|-------|
| **RBB** | **0.8109** | geometric bl=30, regime_enabled=False | Production model. Pure block bootstrap. |
| **GARCH** | 0.7800 | p=1, q=1, t, Constant | Volatility clustering benchmark. IGARCH (persistence=1.0). |
| **GBM** | 0.7630 | MLE mu/sigma | Baseline control. No hyperparameters. |
| **HMM** | n/a | Integrated into RBB | Pure numpy Baum-Welch. 3-state didn't improve tails. |

Scores are walk-forward composites (5000 sims, 90d step, exponential recency weighting).

### Model Addition Policy
Future candidates (MS-GARCH, Merton, EGARCH) deprioritized. MS-GARCH required GARCH within 0.02 of RBB — actual gap is 0.031.

### Validation Audit
All three model scores reproduced from scratch with zero discrepancy (2026-04-01). `results/validation_audit/validation_audit.json`.

---

## Scoring System

### 9 Scoring Metrics (→ composite score, all normalized [0,1])

| Category | Metric | Weight |
|----------|--------|--------|
| **Path Dynamics (35%)** | drawdown_duration_dist | 0.13 |
| | recovery_time_dist | 0.12 |
| | time_in_drawdown_ratio | 0.10 |
| **Tail Risk (25%)** | tail_index_accuracy | 0.13 |
| | percentile_band_coverage | 0.12 |
| **Distributional (15%)** | ks_statistic | 0.07 |
| | qq_divergence | 0.08 |
| **Temporal (10%)** | vol_clustering_acf | 0.10 |
| **Forecast (15%)** | mape | 0.15 |

### Diagnostic Metrics (reported, NOT scored)
MAE, MASE, Moment Matching, Expected Shortfall Match, Max Drawdown Depth Dist, VaR Backtest (Kupiec).

### Cross-Horizon Weighting
180d: 10%, 365d: 20%, 730d: 35%, 1460d: 35%. Stability penalty: std x 0.1.

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/fetch_data.py` | Download historical BTC data |
| `scripts/run_experiment.py` | Full experiment (--mode quick/standard/full) |
| `scripts/run_single_model.py` | Quick single-model test |
| `scripts/run_phase2_baseline.py` | Walk-forward runner for any config (1-12) |
| `scripts/run_garch_sweep.py` | 36-config exhaustive GARCH grid search |
| `scripts/run_garch_walkforward.py` | GARCH A vs B walk-forward comparison |
| `scripts/run_validation_audit.py` | Pre-publication validation audit |
| `scripts/export_phase2_results.py` | Export to dashboard format |

---

## Dashboard

Streamlit (`dashboard/app.py`), 9 tabs: Executive Summary, Leaderboard, Overview, Walk-Forward Inspector, Parameter Optimization, Pipeline Audit, Export Results, Production Simulation, Phase 2 Regime, Phase 3 Models. All tabs support signal/HTML export.

---

## Walk-Forward Config

- Step size: 90 days
- Min training: 730 days
- Recency weighting: exponential, 2-year half-life
- Standard mode: 5000 sims

---

## Data

- Source: Yahoo Finance via `yfinance` (BTC-USD, daily OHLCV, 2013-present)
- Split: 70% train / 30% OOS
- Preprocessing: log returns, parquet storage

---

## Visualization Style

Dark Bloomberg-terminal aesthetic:
- Background: `#0a0a0a`, panels: `#111111`, grid: `#1a1a1a`
- Accent: `#f7931a` (BTC orange), blue `#3b82f6`, green `#10b981`, red `#ef4444`
- Font: JetBrains Mono (fallback: IBM Plex Mono)
- Watermark: `@LongGamma`

---

## Tech Stack

Core: numpy, pandas, scipy, statsmodels, arch>=6.2, optuna
Viz: matplotlib, seaborn
Data: yfinance, pyarrow
CLI: pyyaml, click, rich
Test: pytest, pytest-cov

---

## Coding Standards

- Type hints on all function signatures
- Docstrings on public methods (NumPy style)
- Each model in its own file
- No model-specific logic in validation layer
- Config-driven: no magic numbers
- All randomness seeded via config
- **If the scoring system says GBM wins, GBM wins.** The scoring system is the arbiter.
