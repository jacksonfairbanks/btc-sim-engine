# BTC Price Simulation Engine — Full Documentation

## Overview

A modular, scoring-driven engine for generating and validating **Bitcoin price paths**. The sole focus is producing realistic simulated BTC price trajectories — not return distributions in isolation, not volatility forecasts, not portfolio construction. Price paths are the deliverable.

The system tests multiple stochastic models (GARCH, regime switching, block bootstrap, jump diffusion, etc.) across multiple time horizons, scores the realism of their **simulated price paths** against historical BTC price behavior, optimizes parameters, and produces a ranked leaderboard of model+params+horizon combinations.

Built for a risk analyst and actuarial scientist working in digital credit stress-testing. The downstream use case is overlaying these price paths onto digital credit instruments to stress-test BCR (Bitcoin Credit Ratio) thresholds.

---

## Architecture

```
btc-sim-engine/
├── CLAUDE.md                  # Project context for Claude Code (operational essentials)
├── docs/
│   └── RESULTS.md             # This file — full methodology, results, and analysis
├── pyproject.toml             # Project config (use Poetry or pip + pyproject.toml)
├── README.md
├── config/
│   ├── default.yaml           # Default experiment config
│   └── experiments/           # Saved experiment configs
├── data/
│   ├── raw/                   # Raw BTC price CSVs
│   ├── processed/             # Cleaned, preprocessed data
│   └── loader.py              # Data fetching + preprocessing
├── models/
│   ├── __init__.py
│   ├── base.py                # Abstract base class — THE contract all models implement
│   ├── gbm.py                 # Geometric Brownian Motion (baseline control)
│   ├── garch.py               # GARCH(1,1) with configurable innovation distribution
│   ├── regime_block_bootstrap.py  # Markov regime-conditioned stationary block bootstrap
│   ├── hmm.py                 # Pure numpy Baum-Welch Gaussian HMM
│   └── registry.py            # Model registry — discover + instantiate by name
├── validation/
│   ├── __init__.py
│   ├── metrics.py             # Scoring metrics (9 metrics -> composite score)
│   ├── diagnostics.py         # Diagnostic metrics (reported, not scored)
│   ├── scorer.py              # Composite scoring engine (scoring metrics only)
│   ├── backtest.py            # Walk-forward out-of-sample framework
│   ├── distributions.py       # Percentile distribution data for dashboard charts
│   └── audit.py               # Full pipeline audit for a single walk-forward window
├── optimization/
│   ├── __init__.py
│   ├── search.py              # Parameter search (Sobol + Bayesian via Optuna)
│   ├── sensitivity.py         # Univariate sensitivity analysis (for RBB)
│   └── experiment.py          # Full experiment runner (model x params x horizon)
├── dashboard/
│   └── app.py                 # Streamlit dashboard (9 tabs)
├── reporting/
│   ├── __init__.py
│   ├── leaderboard.py         # Ranked results table
│   ├── comparison.py          # Side-by-side model comparison charts
│   └── export.py              # Export results to CSV/JSON/HTML
├── viz/
│   ├── __init__.py
│   ├── paths.py               # Price path fan charts
│   ├── distributions.py       # Return distribution plots
│   ├── diagnostics.py         # QQ plots, ACF plots, tail diagnostics
│   └── style.py               # Dark theme + orange accents, JetBrains Mono
├── tests/
│   ├── test_models.py
│   ├── test_validation.py
│   └── test_optimization.py
├── scripts/
│   ├── run_experiment.py      # CLI entry point for full experiment
│   ├── run_single_model.py    # Quick single-model test
│   ├── fetch_data.py          # Download historical BTC data
│   ├── run_phase2_baseline.py # Walk-forward runner for any config (1-12)
│   ├── run_garch_sweep.py     # 36-config exhaustive GARCH grid search
│   ├── run_garch_walkforward.py   # GARCH A vs B walk-forward comparison
│   ├── run_validation_audit.py    # Pre-publication validation audit (all 3 models)
│   └── export_phase2_results.py   # Converts Phase 2 results to dashboard format
└── notebooks/
    └── exploration.ipynb
```

---

## Model Interface Contract (`models/base.py`)

Every model MUST implement this interface. No exceptions.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationResult:
    """Standard output from any model."""
    paths: np.ndarray            # Shape: (n_simulations, n_steps) -- price levels
    log_returns: np.ndarray      # Shape: (n_simulations, n_steps-1) -- log returns
    params_used: dict            # Exact parameters used for reproducibility
    model_name: str              # e.g. "garch_1_1", "regime_switching_2state"
    metadata: dict               # Model-specific diagnostics (convergence, etc.)

class BaseModel(ABC):
    """Contract for all price path generators."""

    @abstractmethod
    def fit(self, historical_returns: np.ndarray) -> None:
        """Fit model to historical log returns."""
        ...

    @abstractmethod
    def simulate(
        self,
        n_simulations: int,
        n_steps: int,
        initial_price: float,
        seed: int | None = None
    ) -> SimulationResult:
        """Generate simulated price paths."""
        ...

    @abstractmethod
    def get_default_params(self) -> dict:
        """Return default parameter dict for this model."""
        ...

    @abstractmethod
    def get_param_space(self) -> dict:
        """Return parameter search space for optimization.
        Format: {"param_name": {"type": "float", "low": 0.01, "high": 0.99}}
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier string."""
        ...
```

### Key Rule: Models are black boxes

The validation layer NEVER knows which model generated the paths. It receives `SimulationResult` objects and scores them purely on statistical properties. This is what keeps the system clean.

---

## Model Descriptions

Three models. One novel, two benchmarks. Built and validated before considering additions.

### Model 1: Regime-Conditioned Block Bootstrap (`models/regime_block_bootstrap.py`)

**The flagship model.** Combines Markov regime switching with stationary block bootstrap to produce price paths that preserve both regime-dependent dynamics and empirical distributional properties.

**Academic foundations:**
- Hamilton (1989) -- Markov switching model for regime identification
- Politis & Romano (1994) -- Stationary block bootstrap
- Regime-dependent resampling follows from the conditional bootstrap literature (Kreiss & Paparoditis, 2011)

**How it works:**

*Step 1 -- Regime Classification (fit phase):*
- Fit a 2-state Markov Switching Model (MSM) to historical BTC log returns using `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`
- States are labeled Bull (high-mean, lower-vol) and Bear (low/negative-mean, higher-vol) based on fitted parameters
- Each historical day is assigned a regime probability; classify using the smoothed probabilities (P > 0.5 threshold)
- Extract the fitted transition matrix P:
  ```
  P = [[p_bull_bull,  p_bull_bear ],
       [p_bear_bull,  p_bear_bear ]]
  ```
- Partition historical return blocks into two pools: Bull blocks and Bear blocks

*Step 2 -- Block Construction:*
- Within each regime's historical periods, extract contiguous blocks of returns
- Use optimal block length selection (Politis & White, 2004 automatic method) independently for each regime's pool
- Store as two separate block libraries: `bull_blocks[]` and `bear_blocks[]`

*Step 3 -- Simulation (simulate phase):*
- Initialize: set starting price, draw initial regime from stationary distribution of P
- At each simulation step:
  1. If current block is exhausted, use transition matrix P to probabilistically determine next regime
  2. Draw a random block from the corresponding regime's block library
  3. Apply the block's returns sequentially to evolve the price path
- Repeat for n_simulations paths of n_steps length

**Why this should work for path dynamics:**
Bear blocks are drawn from historical bear markets, so they carry the actual duration, depth, and autocorrelation structure of real BTC downturns. The Markov transition matrix governs how long the model stays in bear (or bull) before switching -- fitted from the historical frequency of regime persistence. This directly addresses drawdown duration realism without imposing parametric assumptions on path shape.

**Key parameters (search space):**
- `n_regimes`: 2 (fixed for Phase 1; 3-state tested in Phase 2b)
- `block_length_method`: "optimal" (Politis-White) or fixed integer
- `min_block_length`: minimum block size to prevent degenerate short blocks
- `transition_matrix_method`: "fitted" (from MSM) or "empirical" (count-based from classified regimes)
- `msm_variance_switching`: bool -- whether variance also switches across regimes (default: True)

**Implementation notes:**
- Use `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression` with `k_regimes=2, switching_variance=True`
- Block resampling done with replacement within each regime pool
- All randomness seeded for reproducibility
- Fitted MSM object and transition matrix stored in `SimulationResult.metadata`
- MSM/HMM fits cached to `results/cache/`
- Simulation instrumented for audit tracking (block usage, regime transitions)
- Also supports 3-state Baum-Welch HMM via `regime_method="hmm_baum_welch"` (pure numpy, no hmmlearn dependency)

**Production config:** geometric bl=30, regime_enabled=False. Walk-forward final: **0.8109**.

---

### Model 2: GBM (`models/gbm.py`)

**Baseline control.** Geometric Brownian Motion -- the standard log-normal random walk. If the other models can't beat GBM on the scoring system, they are not worth their complexity.

- Drift (mu) and volatility (sigma) estimated from historical log returns
- Paths: `S(t+1) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt) * Z)` where Z ~ N(0,1)
- No volatility clustering, no regime dynamics, no fat tails
- Serves as the "is your model adding value?" benchmark

**Key parameters:**
- `mu`: drift (estimated or fixed)
- `sigma`: volatility (estimated or fixed)
- `dist`: "normal" (default)

**Implementation:** MLE-only (no Optuna). Mu and sigma estimated from training data on every fit() call. No hyperparameters. Walk-forward final: **0.7630**.

---

### Model 3: GARCH(1,1) (`models/garch.py`)

**Volatility clustering benchmark.** Captures the well-documented autocorrelation in BTC's volatility -- big moves follow big moves.

#### What GARCH is and how it differs from the block bootstrap

- **Block bootstrap (RBB)** resamples real historical return sequences -- nonparametric, no assumptions about the return distribution. It preserves whatever structure exists in the data (fat tails, volatility clustering within blocks, asymmetry) automatically. Its limitation: it can only produce return patterns that actually occurred historically.

- **GARCH** is parametric -- it models volatility as a time-varying process and generates synthetic returns from that process. It can produce return sequences that never occurred historically, because each day's return is drawn fresh from a parametric distribution scaled by the current conditional volatility. Its limitation: the innovation distribution is assumed (normal, Student's t, or skewed t) -- it can't reproduce BTC's exact return shape the way bootstrap can.

- **GARCH's strength** is volatility clustering: big moves follow big moves, calm follows calm. This is the most well-documented stylized fact of BTC returns, and GARCH models it explicitly through the persistence parameters.

- **GARCH's weakness** for path simulation: it generates one day at a time. Multi-day structures (sustained drawdowns, recovery patterns, regime-like behavior) emerge only if the single-day dynamics happen to chain together correctly. The block bootstrap preserves these multi-day structures by construction.

#### The GARCH(1,1) equation

```
sigma^2(t) = omega + alpha * r^2(t-1) + beta * sigma^2(t-1)
```

- **omega:** Base volatility floor -- volatility never drops below this even in calm periods. Fitted by MLE.
- **alpha:** Shock reactivity -- how much yesterday's return surprise feeds into today's volatility. High alpha means the model reacts strongly to new information. Fitted by MLE.
- **beta:** Persistence -- how much yesterday's volatility carries forward. High beta means vol shocks decay slowly. Fitted by MLE.
- **alpha + beta = persistence.** For BTC, expect 0.95-0.99. Close to 1.0 means volatility is highly persistent (IGARCH territory).
- **Half-life of a vol shock** = log(0.5) / log(alpha + beta). If persistence=0.95, half-life is ~14 days. If 0.99, ~69 days.

#### How simulation works

1. Start with the last fitted conditional variance from training data
2. Each day: draw an innovation (random shock) from the chosen distribution (normal, t, or skewt)
3. Scale the innovation by the current conditional standard deviation: `r(t) = mu + sigma(t) * z(t)`
4. Update conditional variance: `sigma^2(t+1) = omega + alpha * r^2(t) + beta * sigma^2(t)`
5. Convert return to price: `P(t+1) = P(t) * exp(r(t))`
6. Repeat for n_steps days, across n_simulations paths

#### Hyperparameters (Optuna searches these)

| Parameter | Type | Search Space | Purpose |
|-----------|------|-------------|---------|
| `p` | int | 1-2 | GARCH lag order -- how many past volatility values to use |
| `q` | int | 1-2 | ARCH lag order -- how many past squared returns to use |
| `dist` | categorical | "normal", "t", "skewt" | Innovation distribution -- controls tail thickness |
| `mean_model` | categorical | "Zero", "Constant", "ARX" | How average return is modeled |

#### Fitted parameters (MLE estimates these given the hyperparameters)

- **omega, alpha, beta** -- the GARCH equation coefficients
- **nu** -- degrees of freedom for t/skewt distribution (lower = fatter tails, typically 4-8 for BTC)

**Implementation notes:**
- Uses `arch` library v8.0.0 for MLE fitting
- Simulation rolls forward from last fitted conditional variance (not unconditional), so forward projections reflect the current volatility state
- Variance capped at 10x unconditional to prevent IGARCH blow-ups at long horizons
- Paths with NaN/inf or single-day moves > 100% are redrawn
- Supports configurable p, q, dist (normal/t/skewt), mean_model (Zero/Constant/ARX)
- Fits cached to `results/cache/`

**Production config:** p=1, q=1, dist=t, mean_model=Constant. Walk-forward final: **0.7800**.

---

### Model 4: Baum-Welch HMM (`models/hmm.py`)

Implemented as a regime estimation method for RBB, not a standalone simulation model.

- Pure numpy Gaussian HMM with forward-backward EM algorithm
- Supports N states, tied/full covariance, multiple restarts, BIC model selection
- No hmmlearn dependency (requires C extensions unavailable on Python 3.14)
- Integrated into RBB model via `regime_method="hmm_baum_welch"`
- Phase 2b testing showed 3-state HMM does not improve tail events

---

### Naming Convention
- `"regime_block_bootstrap"` or `"rbb"`
- `"gbm"`
- `"garch_1_1"`

### Model Addition Policy
Future candidates (Merton jump-diffusion, EGARCH, MS-GARCH) are documented but deprioritized. MS-GARCH required GARCH within 0.02 of RBB -- actual gap is 0.031.

---

## Validation & Scoring System

The validation system has two distinct buckets:
1. **Scoring Metrics** -- used to rank models. These produce the composite score that determines the leaderboard.
2. **Diagnostic Metrics** -- reported alongside results for context and sanity-checking, but do NOT feed into the composite score.

All metrics operate on **simulated price paths vs. historical price paths**. Not returns in isolation -- paths.

---

### Scoring Metrics (9 metrics -- feed into composite score)

Each metric: `(simulated_paths, historical_price_data, horizon) -> float` normalized to [0, 1] where 1 = perfect match.

#### Distributional (2)

- **KS Statistic** -- Kolmogorov-Smirnov test comparing simulated vs. historical return distributions. A single-number hypothesis test that catches gross distributional mismatch. Sensitive at the center of the distribution. Important because if the basic return shape is wrong, nothing downstream will be right.

- **QQ Divergence** -- Mean squared error between quantiles on a QQ plot (simulated vs. historical). More granular than KS -- reveals where specifically the distributions diverge (tails, center, shoulders). Important because it catches subtle mismatches that KS misses, especially at the extremes.

#### Tail / Extreme Risk (2)

- **Tail Index Accuracy** -- Compare Hill estimator tail indices on the left (crash) tail of simulated vs. historical returns. The tail index controls how likely extreme crashes are. Important because if the model's tail is too thin, it systematically underestimates crash severity -- fatal for credit stress-testing.

- **Percentile Band Coverage** -- Percentage of realized daily prices that fell within the simulated 5th-95th percentile envelope. Target: 90% (by construction). Score: 90%+ = 1.0, linear decay below. Directly measures whether the simulation brackets reality -- the core question for credit stress-testing. Replaced VaR Backtest (Kupiec) which penalized too harshly for marginal tail frequency mismatches on no-margin-call products. VaR Kupiec is now a diagnostic metric.

#### Path Dynamics (3) -- CRITICAL CATEGORY

- **Drawdown Duration Distribution** -- Compare the distribution of how long drawdowns last (peak-to-recovery) between simulated and historical paths. BTC's 2018 bear market drawdown lasted ~12 months; the 2022 drawdown lasted ~14 months. Important because a model that never produces drawdowns longer than 60 days is useless for projecting multi-month credit exposure, regardless of how well it matches return distributions.

- **Recovery Time Distribution** -- Compare the distribution of time from trough back to prior peak. Distinct from drawdown duration -- a drawdown can reach its bottom quickly but take far longer to recover. Important because recovery time directly determines how long a digital credit instrument stays under stress after a crash.

- **Time-in-Drawdown Ratio** -- What percentage of total time is the simulated path below its running maximum? Compare this ratio between simulated and historical. Real BTC spends a large percentage of time underwater from all-time highs. Important because it captures the overall "shape" of the path -- a model can match return distributions perfectly but still produce paths that are unrealistically smooth or unrealistically choppy.

#### Temporal Dependence (1)

- **Volatility Clustering (ACF of r^2)** -- Compare the autocorrelation function of squared returns between simulated and historical paths. BTC exhibits strong volatility clustering -- large moves follow large moves, calm follows calm. Important because models that don't reproduce clustering will generate paths where volatility is too evenly spread, producing unrealistic drawdown and recovery dynamics.

#### Forecast Accuracy (1)

- **MAPE** -- Mean Absolute Percentage Error of the median simulated path vs. the realized historical path (evaluated on out-of-sample windows). Measures how close the model's central tendency is to what actually happened, as a percentage. Important because it's the most intuitive accuracy metric for communicating to stakeholders -- "the model was off by X% on average."

---

### Diagnostic Metrics (reported, NOT scored)

These are computed and displayed alongside scoring metrics for context, but do not affect the composite score or leaderboard ranking.

- **MAE** -- Mean Absolute Error of median simulated path vs. realized, in dollar terms. Complements MAPE when percentage distortion is an issue (MAPE inflates at low price levels).

- **MASE** -- Mean Absolute Scaled Error. Normalizes MAE by the in-sample naive forecast error (naive = "tomorrow's price = today's price"). A MASE > 1 means the model is literally worse than predicting no change. Acts as a "does this model add any value at all?" sanity check.

- **Moment Matching** -- Compare first 4 moments (mean, variance, skewness, kurtosis) of log returns. Quick sanity check that basic statistical properties are preserved.

- **Expected Shortfall Match** -- Compare average loss given a tail event between simulated and historical. VaR tells you the threshold; ES tells you how bad it gets past that threshold.

- **Max Drawdown Depth Distribution** -- Compare distribution of max drawdown magnitudes across rolling windows. Provides context on whether the model produces crash depths in the right ballpark.

- **VaR Backtest (Kupiec)** -- Tests whether 1% and 5% VaR thresholds are breached at expected frequency. Moved from scoring to diagnostics because the binary hypothesis test penalizes too harshly for marginal tail frequency mismatches on no-margin-call credit products.

---

### Composite Scorer (`validation/scorer.py`)

```python
@dataclass
class ScoreCard:
    model_name: str
    horizon: str               # "30d", "90d", "180d", "365d", "1460d"
    params: dict
    scoring_metrics: dict      # metric_name -> score (feeds into composite)
    diagnostic_metrics: dict   # metric_name -> value (reported only)
    composite_score: float     # Weighted average of scoring_metrics only
    rank: int | None = None

class Scorer:
    def __init__(self, weights: dict[str, float] | None = None):
        """
        weights: metric_name -> weight for SCORING metrics only.
        Default weights emphasize path dynamics and tail behavior
        (most important for credit risk / price path realism).
        """
        self.weights = weights or {
            # Path Dynamics -- 35% total (highest priority)
            "drawdown_duration_dist": 0.13,
            "recovery_time_dist": 0.12,
            "time_in_drawdown_ratio": 0.10,
            # Tail / Extreme Risk -- 25% total
            "tail_index_accuracy": 0.13,
            "percentile_band_coverage": 0.12,
            # Distributional -- 15% total
            "ks_statistic": 0.07,
            "qq_divergence": 0.08,
            # Temporal Dependence -- 10%
            "vol_clustering_acf": 0.10,
            # Forecast Accuracy -- 15%
            "mape": 0.15,
        }

    def score(
        self,
        sim_result: SimulationResult,
        historical_data: np.ndarray,
        horizon: str
    ) -> ScoreCard:
        ...
```

### Default Weight Rationale
Path dynamics gets 35% -- highest priority because the primary use case is projecting realistic BTC price trajectories for credit stress-testing, where drawdown duration and recovery time directly determine exposure windows. Tail risk gets 25% because crash severity calibration is critical for BCR thresholds. MAPE gets 15% because central tendency accuracy matters for forward projections. Distributional gets 15% as foundational correctness. Vol clustering gets 10% because it shapes path texture over multi-month horizons.

Weights are configurable per experiment via YAML config.

---

## Cross-Horizon Scoring Methodology

Used consistently across Phase 1 block length selection, Phase 2 regime testing, and Phase 3 GARCH comparison.

**Cross-horizon scoring weights:**
- 180d: 10%, 365d: 20%, 730d: 35%, 1460d: 35%
- Stability penalty: std(per-horizon scores) x 0.1
- Final score = weighted composite - stability penalty

**Rationale:** 730d and 1460d tied at 35% each. 1460d is most relevant for 4-8yr projections but has fewer walk-forward windows (~6 vs ~8-10 for shorter horizons). 730d captures full bear cycles (2022 was 14 months). Shorter horizons carry 30% collectively as validation checks.

---

## Time Horizons

Score every model at each horizon independently:

| Horizon ID | Days | Use Case |
|-----------|------|----------|
| `30d`     | 30   | Short-term credit monitoring, margin calls |
| `90d`     | 90   | Quarterly risk reviews |
| `180d`    | 180  | Semi-annual stress tests |
| `365d`    | 365  | Annual risk assessment |
| `1460d`   | 1460 | Full BTC cycle (~4 years, halving to halving) |

Models may perform differently at different horizons. A GARCH model might dominate at 30-90d but fail at 1460d where regime dynamics matter more. The leaderboard captures this.

---

## Walk-Forward Validation Methodology (`validation/backtest.py`)

To avoid overfitting, use expanding-window walk-forward with **recency weighting**:
1. Train on data up to time T
2. Simulate paths from T to T+horizon
3. Score against actual realized path
4. Slide window forward by step_size
5. Compute **recency-weighted** average across all windows

**Recency weighting:** More recent windows carry more weight than older ones. BTC's market structure evolves (ETF era, institutional adoption, derivatives depth), so a model's accuracy on recent regimes is more practically relevant than its accuracy on 2014 data.

Default: exponential decay with a 2-year half-life. A window from 2 years ago gets 50% the weight of today's window. A window from 4 years ago gets 25%.

```python
# Weight for window at time t, where t_latest is the most recent window
weight(t) = exp(-lambda * (t_latest - t))
# where lambda = ln(2) / half_life_days
```

Configurable: set `recency_weighting: "equal"` to disable and use flat averaging.

**Standard mode:** 5000 sims, 90d step, exponential recency weighting.

---

## Parameter Optimization Methodology (`optimization/`)

### Search Strategy

Two-phase search for maximum parameter coverage with efficient refinement.

**Phase 1 -- Broad Coverage (Sobol quasi-random sampling):**
First N trials use Sobol sequences to fill the parameter space as evenly as possible. Sobol is a low-discrepancy quasi-random method -- it guarantees no gaps or clusters in coverage, unlike pure random sampling. This ensures every region of the parameter space gets tested at least once.

**Phase 2 -- Bayesian Refinement (TPE):**
Remaining trials use Optuna's Tree-structured Parzen Estimator, seeded with Phase 1 results, to drill into the most promising parameter regions identified during the sweep.

Optuna supports this natively via `n_startup_trials` (Phase 1 count before TPE activates).

**Budget Allocation by Model Complexity:**
Models with larger parameter spaces get more trials. Spending 100 trials on GBM (2 params) is wasteful; spending only 100 on the regime block bootstrap (5-6 params) may miss important regions.

```python
# Default trial budgets per model x horizon combination
TRIAL_BUDGETS = {
    "gbm": 30,                    # 2 params -- small space, converges fast
    "garch_1_1": 80,              # 4 params -- moderate space
    "regime_block_bootstrap": 150, # 5-6 params -- largest space, needs coverage
}

# Phase 1 (Sobol) gets ~40% of budget, Phase 2 (TPE) gets ~60%
SOBOL_RATIO = 0.4
```

**Total budget:** Depends on run mode. Quick: ~150 evaluations (minutes). Standard: ~1,300 evaluations (moderate). Full: ~12,500 evaluations across 5 seeds with confidence intervals (hours, production-grade).

During parameter search, simulations run at reduced path count (`n_simulations_search`). Once the winning model+params are identified per horizon, the final scored run uses `n_simulations_final` for statistical confidence.

### Experiment Runner (`optimization/experiment.py`)

```
for each model in [gbm, garch_1_1, regime_block_bootstrap]:
    for each horizon in [30d, 90d, 180d, 365d, 1460d]:
        optimize params via Optuna (N trials):
            1. Set params on model
            2. Fit to training data
            3. Simulate N paths
            4. Score via Scorer
            5. Return composite score (Optuna maximizes this)
        Record best params + score for this model+horizon
Produce leaderboard
```

---

## Data Requirements

### Historical BTC Price Data
- **Source**: CoinGecko API (free) or Yahoo Finance via `yfinance`, or CryptoCompare
- **Frequency**: Daily OHLCV
- **History needed**: At minimum 2013-present (captures 2013, 2017, 2021 cycles + crashes)
- **Preprocessing**:
  - Compute log returns: `log(close_t / close_t-1)`
  - Handle missing days (weekends don't apply to crypto, but exchange outages do)
  - Store processed data as parquet for fast loading

### Data Split Strategy
- **In-sample**: First 70% of history (for fitting)
- **Out-of-sample**: Last 30% (for walk-forward scoring)
- Configurable via YAML

---

## Tech Stack

### Core Dependencies
```
numpy>=1.26
pandas>=2.1
scipy>=1.12
statsmodels>=0.14        # Regime switching, ARIMA
arch>=6.2                # GARCH family
optuna>=3.5              # Bayesian optimization
```

### Visualization
```
matplotlib>=3.8
seaborn>=0.13
```

### Data
```
yfinance>=0.2            # Historical price data
pyarrow>=14.0            # Parquet storage
```

### Config & CLI
```
pyyaml>=6.0
click>=8.1               # CLI interface
rich>=13.0               # Terminal output formatting (tables, progress bars)
```

### Testing
```
pytest>=7.4
pytest-cov>=4.1
```

---

## Visualization Style

All charts follow a consistent dark Bloomberg-terminal aesthetic:

- **Background**: `#0a0a0a` (near black)
- **Card/panel background**: `#111111`
- **Grid lines**: `#1a1a1a` (subtle)
- **Primary accent**: `#f7931a` (Bitcoin orange)
- **Secondary colors**: `#3b82f6` (blue), `#10b981` (green), `#ef4444` (red)
- **Font**: JetBrains Mono (monospace, data-dense)
- **Fallback font**: IBM Plex Mono
- **Attribution**: `@LongGamma` watermark on exported charts

Use `matplotlib` with a custom style sheet defined in `viz/style.py`. All chart functions should accept an `ax` parameter for composability.

---

## CLI Interface

```bash
# Fetch latest BTC data
python scripts/fetch_data.py

# Quick test -- sanity check, minutes not hours
python scripts/run_experiment.py --config config/default.yaml --mode quick

# Standard -- meaningful model comparison
python scripts/run_experiment.py --config config/default.yaml --mode standard

# Full -- production-grade, multi-seed, confidence intervals
python scripts/run_experiment.py --config config/default.yaml --mode full

# Single model quick test (always uses quick settings)
python scripts/run_single_model.py --model regime_block_bootstrap --horizon 180d
```

---

## Config Format (`config/default.yaml`)

```yaml
experiment:
  name: "baseline_run"
  seed: 42
  run_mode: "quick"            # "quick", "standard", or "full"

# Run Mode Defaults (override any field per-experiment)
#
# QUICK -- fast iteration, sanity-checking, debugging
#   n_simulations_search: 500
#   n_simulations_final: 2000
#   search_phase: "sobol_only"       (no TPE refinement)
#   trial_budgets: { gbm: 5, garch_1_1: 15, regime_block_bootstrap: 30 }
#   horizons: [30d, 180d, 1460d]     (3 representative horizons)
#   walk_forward_step_size: 180      (fewer windows)
#   seeds: [42]                      (single seed)
#   ~150 total evaluations, minutes not hours
#
# STANDARD -- development runs, comparing models meaningfully
#   n_simulations_search: 1000
#   n_simulations_final: 5000
#   search_phase: "sobol_then_tpe"   (full two-phase)
#   trial_budgets: { gbm: 30, garch_1_1: 80, regime_block_bootstrap: 150 }
#   horizons: [30d, 90d, 180d, 365d, 1460d]
#   walk_forward_step_size: 90
#   seeds: [42]
#   ~1,300 total evaluations
#
# FULL -- production/publication-grade results
#   n_simulations_search: 2000
#   n_simulations_final: 10000
#   search_phase: "sobol_then_tpe"
#   trial_budgets: { gbm: 50, garch_1_1: 150, regime_block_bootstrap: 300 }
#   horizons: [30d, 90d, 180d, 365d, 1460d]
#   walk_forward_step_size: 60       (more windows, finer granularity)
#   seeds: [42, 123, 456, 789, 1024] (5 seeds, report mean +/- std)
#   ~2,500 evaluations x 5 seeds = ~12,500 total
#   Confidence intervals on all scores

data:
  source: "yfinance"
  ticker: "BTC-USD"
  start_date: "2013-01-01"
  train_pct: 0.7

models:
  - regime_block_bootstrap
  - gbm
  - garch_1_1

horizons:
  - 30d
  - 90d
  - 180d
  - 365d
  - 1460d

scoring:
  weights:
    # Path Dynamics -- 35%
    drawdown_duration_dist: 0.13
    recovery_time_dist: 0.12
    time_in_drawdown_ratio: 0.10
    # Tail / Extreme Risk -- 25%
    tail_index_accuracy: 0.13
    percentile_band_coverage: 0.12
    # Distributional -- 15%
    ks_statistic: 0.07
    qq_divergence: 0.08
    # Temporal Dependence -- 10%
    vol_clustering_acf: 0.10
    # Forecast Accuracy -- 15%
    mape: 0.15
  diagnostics:
    - mae
    - mase
    - moment_matching
    - es_match
    - max_drawdown_depth_dist

walk_forward:
  step_size_days: 90
  min_training_days: 730
  recency_weighting: "exponential"
  recency_half_life_days: 730

output:
  dir: "results/"
  export_formats: ["csv", "json", "html"]
  charts: true
```

---

## Dashboard

Streamlit dashboard (`dashboard/app.py`) with 9 tabs:
1. **Executive Summary** -- methodology visualizations (630K paths hero, pipeline flow, walk-forward timeline, scoring donut), key findings, tail events, and production forward projection with "What This Means" interpretation
2. **Leaderboard** -- ranked model comparison
3. **Overview** -- high-level metrics
4. **Walk-Forward Inspector** -- per-window drill-down
5. **Parameter Optimization** -- search results and sensitivity
6. **Pipeline Audit** -- full audit trail
7. **Export Results** -- CSV/JSON/HTML export
8. **Production Simulation** -- supports RBB and GBM model selection
9. **Phase 2 Regime** -- regime conditioning comparison
10. **Phase 3 Models** -- GARCH comparison

All tabs have signal/HTML export capabilities.

---

## Validation Audit

All three model scores reproduced from scratch with zero discrepancy (2026-04-01). Audit saved to `results/validation_audit/validation_audit.json`. Script: `scripts/run_validation_audit.py`.

---

# Phase Results

---

## Phase 1: Block Length Selection

### Methodology
Univariate sensitivity analysis. Single-config test: each block length applied across ALL four horizons (180d, 365d, 730d, 1460d) simultaneously with `regime_enabled=False`. Cross-horizon final score used to rank configurations.

### Configurations Tested
- Block lengths: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180]
- Sampling methods: geometric, fixed
- Total: 30 configurations x 4 horizons = 120 evaluations
- Regime switching: OFF for all (single pool of all historical returns)
- Simulations: 500 paths per evaluation

### Results (sorted by cross-horizon final score)

| Rank | Config | 180d | 365d | 730d | 1460d | Final |
|------|--------|------|------|------|-------|-------|
| 1 | geometric bl=160d | 0.823 | 0.854 | 0.897 | 0.898 | 0.8779 |
| 2 | geometric bl=30d | 0.822 | 0.855 | 0.894 | 0.899 | 0.8775 |
| 3 | geometric bl=90d | 0.822 | 0.851 | 0.896 | 0.898 | 0.8772 |
| 4 | fixed bl=180d | 0.825 | 0.853 | 0.898 | 0.894 | 0.8772 |
| 5 | geometric bl=180d | 0.825 | 0.851 | 0.898 | 0.894 | 0.8769 |
| ... | ... | ... | ... | ... | ... | ... |
| 30 | fixed bl=140d | 0.820 | 0.851 | 0.889 | 0.887 | 0.8709 |

### Key Findings

1. **Scores are remarkably flat.** Spread from #1 to #30 is only 0.0070 (0.8779 to 0.8709). The model is robust to block length choice -- no single block length is dramatically better.

2. **Geometric sampling consistently beats fixed** at the same block length, but by tiny margins. Geometric draws block lengths from a geometric distribution centered on the mean, avoiding artificial periodicity.

3. **Per-metric tradeoffs exist but cancel out in composite:**
   - Shorter blocks (10-30d): better tail index accuracy (0.84 vs 0.70), better MAPE, better time-in-drawdown
   - Longer blocks (80-180d): better drawdown duration distribution (0.88 vs 0.87), better QQ divergence (0.65 vs 0.59), better KS statistic
   - These tradeoffs offset each other in the weighted composite

4. **Daily MSM regime classification was abandoned.** Daily MSM produced micro-regimes (6-day avg runs) which constrained block pools to tiny blocks regardless of the block length parameter. Weekly MSM produced macro-regimes (44-day avg runs) with proper bull/bear structure. However, even with weekly MSM, regime switching hurt composite scores by ~1% at every horizon compared to a single pool.

5. **Regime switching (tested separately) hurts at all horizons:**
   - 180d: no-regime 0.826 vs best-regime 0.811 (-0.014)
   - 365d: no-regime 0.855 vs best-regime 0.846 (-0.010)
   - 730d: no-regime 0.896 vs best-regime 0.884 (-0.012)
   - 1460d: no-regime 0.905 vs best-regime 0.893 (-0.012)
   - Reason: splitting into two pools reduces diversity without adding information the blocks don't already carry.
   - **Caveat:** These results used the old per-horizon-tuned block lengths, not the locked bl=30 geometric config. Phase 2 re-tested regime conditioning with the proper locked baseline.

### Production Recommendation
**Geometric bl=30d, regime_enabled=False.** Scores within 0.0004 of the #1 config (statistical noise with 500 paths), uses smallest blocks for maximum resampling diversity, least likely to overfit to specific historical sequences. The model is a pure stationary block bootstrap of all historical BTC returns.

### vs GBM Baseline
| Model | Cross-Hz Final | 180d | 365d | 730d | 1460d |
|-------|---------------|------|------|------|-------|
| RBB (geo bl=30d) | **0.8775** | 0.822 | 0.855 | 0.894 | 0.899 |
| GBM (MLE) | 0.7689 | 0.729 | 0.785 | 0.772 | 0.775 |
| Delta | +0.1086 | +0.093 | +0.070 | +0.122 | +0.124 |

RBB outperforms GBM by 14.1% on cross-horizon final score. The margin widens at longer horizons where BTC's non-normal return structure matters most.

---

## Phase 2: Regime Conditioning Test

### Goal
Determine whether constraining the sequence in which historical blocks are chained -- using regime labels to control which pool blocks are drawn from -- produces more realistic price paths than drawing from one undifferentiated pool.

### How regime conditioning works in this engine
MSM's only role is **classification**. It fits mean and volatility parameters per regime to determine which historical weeks were bull and which were bear. Those fitted parameters never enter the simulation -- they exist solely to produce labels.

Once every week is labeled, actual daily returns from bull-labeled weeks go into the bull block pool. Bear-labeled weeks go into the bear block pool. The blocks are real historical return sequences, not parametric.

**Simulation loop with regime ON:**
1. Start in a regime drawn from the stationary distribution (~53% bull, ~47% bear)
2. Draw a real historical block from that regime's pool
3. Lay down actual daily returns from that block
4. Block ends -> roll the transition matrix to decide next regime
5. Draw next block from the (possibly new) regime's pool
6. Repeat until full path length is reached

**Simulation loop with regime OFF:**
1. One pool containing all historical blocks
2. Draw a block, lay down returns, draw another block
3. No transition logic, no regime labels, every block has equal probability

### What the MSM parameters control (classification only, not simulation)
- **Variance switching ON**: MSM uses both mean and volatility to classify regimes. A period with moderate returns but extreme vol can be labeled bear.
- **Variance switching OFF**: MSM uses only mean returns to separate states. Volatility is assumed equal across regimes. This changes which weeks get which label, which changes which blocks land in which pool.

### What fitted vs empirical transition means
- **Fitted transition**: The transition matrix comes from MSM's internal MLE optimization. It mathematically solved for the switching probability that best explains the full return series. Smooth, theoretically optimal.
- **Empirical transition**: Ignores MSM's modeled transition probabilities. Takes the regime labels and counts raw transitions. Walk through the label sequence week by week, tally how often bull->bear and bear->bull actually occurred. Pure counting, no smoothing.

### Phase 2a: MSM 2-state

All configs use bl=30 geometric locked across all horizons.

| Config | regime_enabled | transition_method | variance_switching |
|--------|---------------|-------------------|-------------------|
| 1 | False | n/a | n/a |
| 2 | True | fitted | ON |
| 3 | True | fitted | OFF |
| 4 | True | empirical | ON |
| 5 | True | empirical | OFF |

**What the comparisons tell us:**
- Configs 2-5 vs 1: Does regime conditioning help at all with a proper baseline?
- Config 2 vs 3: Does letting MSM use volatility for classification improve pool separation?
- Config 2 vs 4: Does the model's smoothed transition estimate beat raw counting?
- Config 3 vs 5: Same question but without variance switching

### Phase 2a Results

Config 2 (2-state MSM, fitted transition, variance switching ON) walk-forward composite: **0.8118** vs Config 1 baseline **0.8109**. Delta +0.0009 -- within noise. Composite is a dead heat.

**But the tail event data is the real finding.** At 1460d:

| Metric | Config 1 (no regime) | Config 2 (2-state) | Delta |
|--------|---------------------|---------------------|-------|
| 2022 crash reproduction | 23.3% | 14.1% | -9.2% |
| 2018 crash reproduction | 10.3% | 5.0% | -5.3% |
| Mt Gox crash reproduction | 9.0% | 4.1% | -4.9% |
| Drawdowns >= 75% | 28.9% | 19.5% | -9.4% |
| Duration >= 365d | 64.4% | 55.1% | -9.3% |

Regime conditioning systematically reduces tail severity. Splitting returns into bull/bear pools and constraining draws to the current regime's pool means the model spends ~53% of time locked into bull blocks. Crash blocks can only appear after the transition matrix flips to bear -- reducing the frequency and severity of extreme paths.

**Decision: Skip Configs 3-5.** The tail suppression problem is structural to 2-state pool splitting, not specific to transition method or variance switching parameters. Different classification or switching rates don't fix the fundamental constraint: two pools always dilute tail event frequency relative to one pool.

### Phase 2b: Baum-Welch HMM with 3 states

**What changes:**
- Swap MSM estimation for Baum-Welch (EM algorithm). Standard HMM fitting -- jointly estimates transition probabilities, emission distributions, and initial state probabilities through forward-backward + EM.
- Add a third regime state: Bull / Bear / Crisis. Crisis captures extreme dislocation events (flash crashes, capitulation cascades). This creates a small high-severity block pool that gets sampled at low but nonzero probability via the transition matrix.
- Motivation: The Phase 1 audit showed 0 of 500 paths could produce 2020 flash crash in 32 of 33 windows, and 0 of 500 could produce 2018 crash in 16 of 33 windows. A dedicated crisis pool could fix this.
- Baum-Welch handles 3+ states natively. MSM gets awkward past 2 states.

### Phase 2b Implementation

Implemented pure-numpy Baum-Welch Gaussian HMM (`models/hmm.py`) -- no hmmlearn dependency (requires C extensions unavailable on Python 3.14). Integrated into RBB model via `regime_method="hmm_baum_welch"` and `n_regimes=3` parameters.

Config 6 tested: 3-state HMM, fitted transition, variance switching ON, weekly frequency.

**What Baum-Welch actually found (3 states on full training data):**

| State | Days | % | Daily Mean | Daily Std | Character |
|-------|------|---|-----------|-----------|-----------|
| State 0 ("Bull") | 602 | 20% | +0.0067 | 0.0574 | Explosive volatile -- big moves both directions |
| State 1 ("Bear") | 308 | 10% | +0.0012 | 0.0175 | Calm/sideways -- very low vol |
| State 2 ("Crisis") | 2037 | 69% | -0.0003 | 0.0340 | Normal BTC vol -- the dominant state |

Baum-Welch optimizes for statistical fit (maximum likelihood), not for our downstream use case. It found "explosive volatile" / "calm" / "normal" -- not the clean bull/bear/crisis separation hypothesized. Extreme crash days (-46%, -20%) ended up in the "Bull" state because they co-occur with high-volatility periods.

### Phase 2b Results

Config 6 (3-state Baum-Welch HMM) walk-forward composite: **0.8140** vs Config 1 baseline **0.8109** (+0.0031) vs Config 2 **0.8118** (+0.0022).

| Horizon | Config 1 (none) | Config 2 (2-MSM) | Config 6 (3-HMM) | C6 vs C1 |
|---------|----------------|------------------|------------------|----------|
| 180d | 0.8098 | 0.8067 | 0.8081 | -0.0017 |
| 365d | 0.8221 | 0.8211 | 0.8186 | -0.0036 |
| 730d | 0.8265 | 0.8276 | 0.8222 | -0.0044 |
| 1460d | 0.7928 | 0.7956 | **0.8068** | **+0.0141** |

Config 6 is the best at 1460d (+0.0141 vs baseline, meaningful). Worse at 180-730d. The cross-horizon final score improvement (+0.0031) is largely driven by 1460d's 35% weight.

**Tail events at 1460d -- the 3-state HMM made things worse, not better:**

| Metric | C1 (none) | C2 (2-MSM) | C6 (3-HMM) | C6 vs C1 |
|--------|----------|-----------|-----------|----------|
| 2022 crash | 23.3% | 14.1% | 12.7% | -10.6% |
| 2018 crash | 10.3% | 5.0% | 4.8% | -5.5% |
| Mt Gox | 9.0% | 4.1% | 3.8% | -5.2% |
| DD >= 75% | 28.9% | 19.5% | 17.5% | -11.4% |
| Duration >= 365d | 64.4% | 55.1% | 51.6% | -12.8% |
| 2020 flash crash | 0.4% | 0.5% | 0.5% | +0.2% |

3-state HMM suppresses tail events even more than 2-state MSM. The dedicated "crisis" pool did not concentrate crash dynamics -- instead, Baum-Welch split the return series by volatility level, putting extreme days into the high-vol state alongside explosive rallies. Pool splitting in any form (2-state or 3-state) systematically reduces the frequency of extreme paths relative to a single undifferentiated pool.

### Phase 2 Decision Tree (final)
- ~~Phase 2a regime OFF wins convincingly -> lock regime_enabled=False, skip 2b, move to Phase 3~~
- ~~Phase 2a regime ON beats or matches baseline -> run Phase 2b to test whether better estimation + 3 states improves further~~
- **Phase 2a result: composite dead heat, but 2-state regime conditioning structurally suppresses tail events. Configs 3-5 skipped.**
- **Phase 2b result: 3-state HMM marginally improves composite (+0.0031) but suppresses tail events even further than 2-state. The crisis pool hypothesis is invalidated -- Baum-Welch doesn't separate crashes from rallies, it separates volatility levels.**
- **Final decision: lock `regime_enabled=False` for Phase 3. The single-pool pure block bootstrap (Config 1, 0.8109) remains the production configuration.**

---

## Phase 3: GARCH(1,1) Implementation & Comparison

### What we expected

- GARCH should beat GBM (0.7630) because it captures volatility clustering that GBM misses entirely
- GARCH may struggle against RBB (0.8109) on path dynamics and tail risk because parametric distributions can't match BTC's empirical return shape as precisely as resampling actual returns
- GARCH's best chance to beat RBB is at shorter horizons (180d, 365d) where volatility dynamics dominate over long-horizon path structure
- At 1460d, RBB likely wins because the block bootstrap preserves full multi-month drawdown sequences while GARCH generates one day at a time

### Testing plan

The procedure mirrors what was done for the block bootstrap in Phase 1-2, ensuring apples-to-apples methodology:

1. **Hyperparameter search** -- Exhaustive grid across p, q, dist, mean_model for each horizon
2. **Observe the spread** -- How much scores vary across hyperparameter choices
3. **Lock a single config** -- Highest cross-horizon weighted composite
4. **Full walk-forward** -- Same pipeline as RBB Config 1
5. **Compare** -- Cross-horizon, per-horizon, per-metric, tail events
6. **Dashboard integration**

### Phase 3 Baseline Results: GARCH(1,1) with default params

**Config tested:** p=1, q=1, dist=t, mean_model=Constant. Locked across all four horizons. Standard mode: 5000 sims, 90d step, exponential recency weighting.

**Fitted parameters (on full 70% training split, 2947 days):**

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| omega | 0.1418 | Base volatility floor (scaled units) |
| alpha | 0.1048 | Shock reactivity -- moderate |
| beta | 0.8952 | Persistence -- high |
| **persistence (alpha+beta)** | **1.0000** | **IGARCH -- integrated. Vol shocks never decay.** |
| nu (DoF) | 3.28 | Very fat tails (normal=inf, typical equities=5-8) |
| mu (daily) | 0.00154 | ~56% annualized drift |
| Last cond vol (daily) | 2.25% | Volatility at end of training period |
| LL / AIC / BIC | -7549 / 15108 / 15138 | Model fit quality |

**Key observation:** Persistence = 1.0 (IGARCH). This is well-documented for BTC -- volatility is non-stationary. The unconditional variance is technically infinite, meaning the GARCH process doesn't have a long-run average volatility level. For simulation, this means variance can drift arbitrarily high over long horizons. We cap variance at 10x the last fitted value to prevent path explosions; 31 of 500 test paths required redrawing at 1460d due to variance blow-up.

#### Cross-Horizon Final Scores

| Model | Final Score | vs RBB | vs GBM |
|-------|-----------|--------|--------|
| **RBB (Config 1)** | **0.8109** | -- | +0.0479 |
| **GARCH (Config 10)** | **0.7800** | **-0.0309** | +0.0170 |
| **GBM** | **0.7630** | -0.0479 | -- |

GARCH sits between RBB and GBM. Beats GBM by +0.0170, loses to RBB by -0.0309. The gap to RBB is 4x the gap between any Phase 2 regime configs -- this is a real difference, not noise.

#### Per-Horizon Walk-Forward Composites

| Horizon | Weight | RBB | GARCH | GBM | GARCH vs RBB | GARCH vs GBM |
|---------|--------|-------|-------|-------|-------------|-------------|
| 180d | 10% | 0.8098 | 0.7973 | 0.7589 | -0.0126 | +0.0384 |
| 365d | 20% | 0.8221 | 0.8024 | 0.7705 | -0.0197 | +0.0319 |
| 730d | 35% | 0.8265 | 0.7976 | 0.7618 | -0.0290 | +0.0358 |
| 1460d | 35% | 0.7928 | **0.7506** | **0.7625** | **-0.0421** | **-0.0119** |

**Critical finding: GARCH loses to GBM at 1460d.** At the longest horizon (which carries 35% weight), GARCH's parametric day-by-day generation produces less realistic paths than GBM's simple log-normal random walk. This is because:
1. IGARCH persistence means conditional variance random-walks, producing extreme variance drift over 4 years
2. Variance capping (needed to prevent blow-ups) artificially constrains the high-vol paths, distorting the tail structure
3. GBM's constant volatility, while simplistic, doesn't suffer from these compounding artifacts

The gap widens monotonically with horizon: -0.013 at 180d -> -0.042 at 1460d.

#### Tail Event Reproduction @ 1460d

| Event | RBB | GARCH | GBM | GARCH vs RBB |
|-------|-----|-------|-----|-------------|
| DD >= 50% | 92.3% | 59.2% | -- | **-33.1%** |
| DD >= 75% | 28.9% | 17.8% | -- | -11.1% |
| Duration >= 180d | 95.4% | 86.1% | -- | -9.3% |
| Duration >= 365d | 64.4% | 44.7% | -- | -19.7% |
| 2014 Mt Gox | 9.0% | 6.7% | -- | -2.3% |
| 2018 crash | 10.3% | 7.6% | -- | -2.6% |
| 2022 crash | 23.3% | 11.9% | -- | -11.5% |
| 2020 flash crash | 0.4% | 0.3% | -- | -0.1% |

GARCH produces fewer extreme paths than RBB across every tail metric. The DD >= 50% gap is striking: 92.3% vs 59.2%. The block bootstrap, by resampling actual historical crash sequences as contiguous blocks, naturally preserves the depth and duration of real drawdowns. GARCH must chain together individual daily shocks to produce a sustained drawdown -- possible but less probable.

#### Per-Metric Mean Scores @ 1460d (23 windows)

| Metric | RBB | GARCH | Delta | Winner |
|--------|-------|-------|-------|--------|
| drawdown_duration_dist | 0.8493 | 0.8577 | +0.0085 | GARCH |
| recovery_time_dist | 0.8594 | 0.8676 | +0.0082 | GARCH |
| ks_statistic | 0.9393 | 0.9143 | -0.0250 | RBB |
| qq_divergence | 0.9498 | 0.9057 | -0.0441 | RBB |
| tail_index_accuracy | **0.8972** | **0.7742** | **-0.1231** | **RBB** |
| percentile_band_coverage | 0.8350 | 0.7928 | -0.0423 | RBB |
| time_in_drawdown_ratio | 0.9390 | 0.8800 | -0.0589 | RBB |
| vol_clustering_acf | 0.7761 | 0.7488 | -0.0272 | RBB |
| mape | 0.3282 | 0.3326 | +0.0044 | GARCH |

GARCH wins on 3 of 9 metrics (drawdown duration, recovery time, MAPE) -- all by small margins (<0.01). RBB wins on 6 of 9, with the largest gap on tail_index_accuracy (-0.12). The t-distribution with nu=3.28 produces fat tails, but not the *right shape* of tails -- the Hill estimator on the left tail of GARCH-t returns doesn't match BTC's empirical tail index as precisely as resampling actual historical returns.

Notably, GARCH loses on vol_clustering_acf (-0.027) despite volatility clustering being its explicit strength. This suggests that the block bootstrap's within-block autocorrelation structure already captures volatility clustering adequately, and GARCH's parametric approach doesn't add enough to overcome its other disadvantages.

#### Assessment Before Optuna Search

The GARCH-RBB gap (-0.0309) is large enough that Optuna tuning across dist/p/q/mean_model is unlikely to close it. The structural limitation is at 1460d where GARCH loses to GBM -- no hyperparameter choice fixes the fundamental problem of IGARCH variance drift over 4-year horizons.

---

### Phase 3 Hyperparameter Sweep: 36-Config Exhaustive Grid Search

**Methodology:** Same as Phase 1 block length sweep -- single-OOS evaluation (fit on 70% training split, simulate against test data), cross-horizon weighted composite. 2000 sims per evaluation. Total: 36 configs x 4 horizons = 144 evaluations in 171 seconds.

**Search space:** p in {1,2}, q in {1,2}, dist in {normal, t, skewt}, mean_model in {Zero, Constant, ARX}

#### Results (top 10 of 36, sorted by cross-horizon final score)

| Rank | p | q | dist | mean | Final | 180d | 365d | 730d | 1460d | Std |
|------|---|---|------|------|-------|------|------|------|-------|-----|
| 1 | 2 | 2 | t | Constant | **0.8953** | 0.8761 | 0.8945 | 0.9039 | 0.8955 | 0.0101 |
| 2 | 1 | 1 | t | Constant | 0.8834 | 0.8851 | 0.8788 | 0.8903 | 0.8800 | 0.0040 |
| 3 | 2 | 1 | t | Constant | 0.8834 | 0.8851 | 0.8788 | 0.8903 | 0.8800 | 0.0040 |
| 4 | 1 | 2 | t | Constant | 0.8817 | 0.8846 | 0.8781 | 0.8888 | 0.8771 | 0.0044 |
| 5 | 1 | 1 | t | ARX | 0.8310 | 0.8554 | 0.8611 | 0.8352 | 0.8085 | 0.0203 |
| 6 | 2 | 1 | t | ARX | 0.8307 | 0.8554 | 0.8611 | 0.8352 | 0.8078 | 0.0206 |
| 7 | 1 | 1 | t | Zero | 0.8302 | 0.8422 | 0.8724 | 0.8376 | 0.8024 | 0.0244 |
| 8 | 2 | 1 | t | Zero | 0.8283 | 0.8551 | 0.8590 | 0.8312 | 0.8064 | 0.0202 |
| 9 | 2 | 2 | normal | Constant | 0.8249 | 0.7801 | 0.7982 | 0.8382 | 0.8478 | 0.0269 |
| 10 | 1 | 2 | t | Zero | 0.8226 | 0.8573 | 0.8583 | 0.8265 | 0.7958 | 0.0243 |

**Bottom 5:**

| Rank | p | q | dist | mean | Final |
|------|---|---|------|------|-------|
| 32 | 1 | 2 | skewt | ARX | 0.7653 |
| 33 | 1 | 1 | skewt | ARX | 0.7628 |
| 34 | 2 | 1 | skewt | ARX | 0.7628 |
| 35 | 1 | 2 | skewt | Zero | 0.7622 |
| 36 | 1 | 1 | skewt | Zero | 0.7604 |

#### Key Findings

**1. The spread is wide -- hyperparameters matter significantly.**
- Best: 0.8953 (p=2, q=2, t, Constant)
- Worst: 0.7604 (p=1, q=1, skewt, Zero)
- Spread: **0.1349**
- Compare to Phase 1 block length spread: 0.0070. GARCH hyperparameters have 19x more impact than block length choice. This is expected -- GARCH is parametric, so the choice of distribution and mean model fundamentally changes the generated paths.

**2. dist=t with mean_model=Constant dominates.**
- The top 4 configs are ALL t-distribution with Constant mean. They differ only in p and q.
- t-distribution captures BTC's fat tails; Constant mean provides a positive drift term that matches BTC's historical upward bias.
- skewt performs worst -- despite having more parameters, the skewness adds noise without improving fit on this data.
- normal-distribution configs cluster in the middle -- they miss the fat tails but benefit from simplicity (fewer params to estimate, more stable MLE).

**3. p and q have minimal impact within the t+Constant family.**
- p=2,q=2 (0.8953) vs p=1,q=1 (0.8834): +0.0119 difference
- The extra GARCH/ARCH lags provide marginal improvement -- the second lag captures slightly more of the variance dynamics.
- However, p=2,q=2 has higher horizon std (0.0101 vs 0.0040) -- the extra parameters reduce stability across horizons.

**4. mean_model=Constant is critical.**
- Every dist family shows the same pattern: Constant >> Zero ~ ARX
- Zero mean (no drift) produces paths that center on the starting price -- unrealistic for BTC's long-run upward trajectory over 4 years.
- ARX (autoregressive mean) adds complexity without benefit -- the AR component doesn't help at daily frequency.

**5. Best per horizon:**
- 180d: p=2,q=1,t,Constant (0.8851) -- but tied with p=1,q=1
- 365d: p=2,q=2,t,Constant (0.8945)
- 730d: p=2,q=2,t,Constant (0.9039)
- 1460d: p=2,q=2,t,Constant (0.8955)
- p=2,q=2,t,Constant wins at 3 of 4 horizons and is a close second at 180d.

**6. Single-OOS vs walk-forward comparison:**
- Default baseline (p=1,q=1,t,Constant) single-OOS: 0.8834 vs walk-forward: 0.7800
- The 0.1034 gap is larger than RBB's gap (0.8775 single-OOS vs 0.8109 walk-forward = 0.0666 gap)
- This suggests GARCH overfits more to the specific OOS window than RBB does -- the parametric model's performance is more variable across different evaluation periods.

---

### Phase 3 Walk-Forward Results: GARCH(2,2) vs GARCH(1,1)

Both configs run through the same walk-forward pipeline: 5000 sims, 90d step, exponential recency weighting. Decision rule: if A beats B by >0.01, lock A; within 0.01, lock B (simpler model wins ties); if B beats A, lock B.

#### Cross-Horizon Final Scores

| Config | Final Score | Composite | Penalty | Std |
|--------|-----------|-----------|---------|-----|
| **A: GARCH(2,2) t Constant** | **0.7714** | 0.7740 | 0.0026 | 0.0257 |
| **B: GARCH(1,1) t Constant** | **0.7800** | 0.7821 | 0.0021 | 0.0211 |
| GBM | 0.7630 | 0.7635 | 0.0004 | 0.0043 |
| RBB (reference) | 0.8109 | 0.8122 | 0.0013 | 0.0131 |

**Config B (1,1) beats Config A (2,2) by +0.0086.** The extra lags hurt rather than helped -- more parameters increased overfitting, especially at 1460d.

#### Per-Horizon Walk-Forward Composites

| Horizon | A: GARCH(2,2) | B: GARCH(1,1) | GBM | RBB | A vs RBB | B vs RBB |
|---------|-------------|-------------|-------|-------|----------|----------|
| 180d | 0.7945 | 0.7973 | 0.7589 | 0.8098 | -0.0153 | -0.0125 |
| 365d | 0.7995 | 0.8024 | 0.7705 | 0.8221 | -0.0226 | -0.0197 |
| 730d | 0.7913 | 0.7976 | 0.7618 | 0.8265 | -0.0352 | -0.0289 |
| 1460d | **0.7361** | **0.7506** | **0.7625** | 0.7928 | -0.0567 | -0.0422 |

Config B beats Config A at every horizon. The gap widens at 1460d: 0.7361 vs 0.7506 (+0.0145). Both GARCH configs lose to GBM at 1460d -- Config A more severely (0.7361 vs 0.7625).

**The single-OOS to walk-forward gap confirms the overfitting concern:**
- Config A: single-OOS 0.8953 -> walk-forward 0.7714 (gap: **0.1239**)
- Config B: single-OOS 0.8834 -> walk-forward 0.7800 (gap: **0.1034**)
- RBB: single-OOS 0.8775 -> walk-forward 0.8109 (gap: 0.0666)

GARCH(2,2) overfits 87% more than RBB on the single-OOS to walk-forward transition. GARCH(1,1) overfits 55% more. The parametric model's performance is fundamentally more variable across evaluation periods.

#### Decision: Lock GARCH(1,1) t Constant (Config B)

Delta of -0.0086 is within the 0.01 threshold. Simpler model wins ties. Config B is the GARCH representative.

#### Tail Events @ 1460d -- Config A vs Config B

| Event | A: GARCH(2,2) | B: GARCH(1,1) | RBB |
|-------|-------------|-------------|-----|
| DD >= 50% | 35.0% | 59.2% | 92.3% |
| DD >= 75% | 2.4% | 17.8% | 28.9% |
| Duration >= 180d | 81.7% | 86.1% | 95.4% |
| Duration >= 365d | 35.6% | 44.7% | 64.4% |
| 2014 Mt Gox | 0.3% | 6.7% | 9.0% |
| 2018 crash | 0.3% | 7.6% | 10.3% |
| 2022 crash | 1.7% | 11.9% | 23.3% |
| 2020 flash crash | 0.2% | 0.3% | 0.4% |

Config A's tail events are dramatically worse than Config B. GARCH(2,2) produces almost no named crash scenarios (0.3% Mt Gox, 0.3% 2018) -- the extra variance parameters smooth out the extreme paths. Simpler is better for GARCH tail generation.

#### Per-Metric Means @ 1460d (23 windows) -- Config A vs B

| Metric | A: (2,2) | B: (1,1) | Delta |
|--------|---------|---------|-------|
| drawdown_duration_dist | 0.8551 | 0.8577 | -0.0026 |
| ks_statistic | 0.9006 | 0.9143 | -0.0137 |
| mape | 0.3340 | 0.3326 | +0.0014 |
| percentile_band_coverage | 0.7526 | 0.7928 | -0.0401 |
| qq_divergence | 0.8808 | 0.9057 | -0.0248 |
| recovery_time_dist | 0.8652 | 0.8676 | -0.0024 |
| tail_index_accuracy | 0.7585 | 0.7742 | -0.0157 |
| time_in_drawdown_ratio | 0.8682 | 0.8800 | -0.0118 |
| vol_clustering_acf | 0.7228 | 0.7488 | -0.0260 |

Config B wins 8 of 9 metrics. Config A wins only on MAPE by +0.0014 -- negligible.

---

### Per-Metric Mean Scores: All Horizons (Locked GARCH vs RBB vs GBM)

Mean of each scoring metric across all walk-forward windows at each horizon. GARCH = locked Config B (p=1, q=1, t, Constant).

#### 180d (37 windows)

| Metric | RBB | GARCH | GBM | GARCH vs RBB |
|--------|-------|-------|-------|-------------|
| drawdown_duration_dist | 0.7103 | 0.7120 | 0.7156 | +0.0017 |
| recovery_time_dist | 0.7167 | 0.7197 | 0.7224 | +0.0029 |
| time_in_drawdown_ratio | 0.8495 | 0.8469 | 0.8446 | -0.0027 |
| tail_index_accuracy | 0.8047 | 0.7401 | 0.5105 | **-0.0646** |
| percentile_band_coverage | 0.8972 | 0.8109 | 0.8882 | **-0.0863** |
| ks_statistic | 0.8859 | 0.8785 | 0.8470 | -0.0074 |
| qq_divergence | 0.8343 | 0.8795 | 0.7643 | +0.0452 |
| vol_clustering_acf | 0.6722 | 0.6673 | 0.6635 | -0.0050 |
| mape | 0.7412 | 0.7445 | 0.7516 | +0.0033 |

At 180d: GARCH wins on QQ divergence (+0.045) -- the t-distribution's shape matches well over 6 months. GARCH loses on percentile band coverage (-0.086) and tail index (-0.065).

#### 365d (35 windows)

| Metric | RBB | GARCH | GBM | GARCH vs RBB |
|--------|-------|-------|-------|-------------|
| drawdown_duration_dist | 0.7491 | 0.7533 | 0.7586 | +0.0042 |
| recovery_time_dist | 0.7450 | 0.7459 | 0.7537 | +0.0009 |
| time_in_drawdown_ratio | 0.8820 | 0.8864 | 0.8783 | +0.0045 |
| tail_index_accuracy | 0.8398 | 0.7585 | 0.5259 | **-0.0814** |
| percentile_band_coverage | 0.8383 | 0.7595 | 0.8445 | **-0.0788** |
| ks_statistic | 0.9119 | 0.8950 | 0.8653 | -0.0169 |
| qq_divergence | 0.8856 | 0.8893 | 0.8184 | +0.0038 |
| vol_clustering_acf | 0.7458 | 0.7330 | 0.7118 | -0.0128 |
| mape | 0.6256 | 0.6325 | 0.6431 | +0.0069 |

At 365d: Same pattern -- GARCH wins on distributional shape metrics (QQ, time-in-drawdown) but loses on tail index (-0.081) and band coverage (-0.079).

#### 730d (31 windows)

| Metric | RBB | GARCH | GBM | GARCH vs RBB |
|--------|-------|-------|-------|-------------|
| drawdown_duration_dist | 0.7866 | 0.8010 | 0.8002 | +0.0143 |
| recovery_time_dist | 0.7893 | 0.7987 | 0.8034 | +0.0094 |
| time_in_drawdown_ratio | 0.9179 | 0.9066 | 0.9119 | -0.0113 |
| tail_index_accuracy | 0.8825 | 0.7637 | 0.5053 | **-0.1188** |
| percentile_band_coverage | 0.8199 | 0.7500 | 0.8436 | **-0.0699** |
| ks_statistic | 0.9318 | 0.9097 | 0.8807 | -0.0221 |
| qq_divergence | 0.9180 | 0.9047 | 0.8507 | -0.0132 |
| vol_clustering_acf | 0.7780 | 0.7400 | 0.7489 | **-0.0380** |
| mape | 0.5021 | 0.5118 | 0.5277 | +0.0097 |

At 730d: tail_index gap widens to -0.119. GARCH also starts losing on vol_clustering (-0.038) -- the block bootstrap's within-block autocorrelation now outperforms GARCH's explicit volatility model over 2-year horizons. GARCH still wins on drawdown duration (+0.014) and recovery time (+0.009).

#### 1460d (23 windows)

| Metric | RBB | GARCH | GBM | GARCH vs RBB |
|--------|-------|-------|-------|-------------|
| drawdown_duration_dist | 0.8493 | 0.8577 | 0.8607 | +0.0085 |
| recovery_time_dist | 0.8594 | 0.8676 | 0.8769 | +0.0082 |
| time_in_drawdown_ratio | 0.9390 | 0.8800 | 0.9622 | -0.0589 |
| tail_index_accuracy | 0.8972 | 0.7742 | 0.4825 | **-0.1231** |
| percentile_band_coverage | 0.8350 | 0.7928 | 0.8563 | -0.0423 |
| ks_statistic | 0.9393 | 0.9143 | 0.8922 | -0.0250 |
| qq_divergence | 0.9498 | 0.9057 | 0.8888 | -0.0441 |
| vol_clustering_acf | 0.7761 | 0.7488 | 0.8001 | -0.0272 |
| mape | 0.3282 | 0.3326 | 0.4817 | +0.0044 |

At 1460d: the most revealing horizon. GARCH loses to RBB on 6 of 9 metrics. The tail_index gap is the largest at -0.123. Critically, **GBM beats GARCH on vol_clustering_acf** (0.800 vs 0.749) -- at 4-year horizons, IGARCH variance drift produces unrealistic volatility patterns that score worse than GBM's constant vol. GBM also beats GARCH on time_in_drawdown (0.962 vs 0.880) and band coverage (0.856 vs 0.793).

#### Cross-Horizon Pattern Summary

| Metric | GARCH wins at | RBB wins at | Trend |
|--------|-------------|-------------|-------|
| tail_index_accuracy | never | all horizons | Gap widens: -0.06 -> -0.12 |
| percentile_band_coverage | never | all horizons | Consistent -0.07 to -0.09 |
| qq_divergence | 180d, 365d | 730d, 1460d | GARCH's t-dist advantage fades at long horizons |
| drawdown_duration_dist | all horizons | never | Small GARCH edge (+0.002 to +0.014) |
| recovery_time_dist | all horizons | never | Small GARCH edge (+0.003 to +0.009) |
| vol_clustering_acf | never | all horizons | Gap widens: -0.005 -> -0.038 |
| mape | all horizons | never | Negligible GARCH edge (+0.003 to +0.010) |

GARCH's strengths (drawdown duration, recovery time, MAPE) are all small-margin wins (<0.015). RBB's strengths (tail index, band coverage) are large-margin wins (0.06-0.12). The scoring weights amplify this: tail risk gets 25% weight, path dynamics 35%.

---

### Phase 3 Final Three-Model Comparison

| Model | Final Score | vs RBB | Character |
|-------|-----------|--------|-----------|
| **RBB** | **0.8109** | -- | Nonparametric block bootstrap. Preserves empirical return structure, fat tails, multi-day drawdown sequences. |
| **GARCH(1,1)** | **0.7800** | **-0.0309** | Parametric volatility clustering. Captures vol dynamics explicitly but limited by assumed t-distribution and day-by-day generation. |
| **GBM** | **0.7630** | **-0.0479** | Log-normal baseline. No vol clustering, no fat tails. The "does your model add value?" benchmark. |

**RBB wins.** The gap is meaningful (-0.031 GARCH, -0.048 GBM) and consistent across horizons. RBB's advantage is structural: resampling actual historical return blocks preserves BTC's exact distributional shape, volatility clustering (within blocks), and multi-day drawdown patterns. GARCH must reconstruct all of this from a parametric model of the variance process -- possible in theory but less accurate in practice, especially over long horizons where IGARCH variance drift compounds errors.

**GARCH beats GBM** (+0.017), confirming that volatility clustering adds real value over a constant-volatility model. But the improvement is modest -- the block bootstrap captures volatility clustering implicitly (through the within-block autocorrelation structure) while also preserving everything else.

**The 1460d anomaly remains:** GARCH loses to GBM at the longest horizon (0.7506 vs 0.7625). This is the IGARCH persistence problem -- conditional variance random-walks over 4 years, requiring variance capping that distorts the tail structure. At shorter horizons (180-730d), GARCH beats GBM by +0.03-0.04.

---

## Future Extensions (deprioritized)

- **Regime-switching GARCH (MS-GARCH):** Two sets of GARCH parameters with Markov transition between them. Unlike bootstrap regime switching, this doesn't reduce pool diversity -- it switches the generative process itself. Would use Hamilton filter for regime estimation.
  - Only pursue if vanilla GARCH scores within ~0.02 of RBB. **Final gap is -0.0309 -- does not meet threshold.** MS-GARCH is deprioritized.

- **Merton jump-diffusion, EGARCH** -- documented but deprioritized pending RBB dominance.

---

## Development Approach

### Build Order
1. `data/loader.py` -- Get historical BTC data flowing
2. `models/base.py` + `models/gbm.py` -- Establish the interface with simplest model
3. `validation/metrics.py` + `validation/scorer.py` + `validation/diagnostics.py` -- Build scoring before more models
4. `viz/style.py` + `viz/paths.py` -- Visualize GBM paths to verify pipeline end-to-end
5. `models/garch.py` -- GARCH(1,1), score it, compare to GBM
6. `optimization/search.py` -- Wire up Optuna
7. `models/regime_block_bootstrap.py` -- Flagship model: fit MSM, build regime block pools, simulate
8. `validation/backtest.py` -- Walk-forward framework
9. `optimization/experiment.py` -- Full experiment orchestrator (3 models x 5 horizons)
10. `reporting/leaderboard.py` -- Final ranked output

### Coding Standards
- Type hints on all function signatures
- Docstrings on all public methods (NumPy style)
- Each model in its own file
- No model-specific logic in validation layer
- Config-driven: no magic numbers in code
- Reproducible: all randomness seeded via config
- Tests for each model's `fit()` + `simulate()` contract compliance

### Key Principle
**If the scoring system says GBM wins, GBM wins.** Don't add complexity for complexity's sake. The scoring system is the arbiter. Every model must prove it beats the baseline.
