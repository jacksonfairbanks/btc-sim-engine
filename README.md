# BTC Price Simulation Engine

A modular, scoring-driven engine for generating and validating **Bitcoin price paths**.

## Overview

Tests multiple stochastic models (GARCH, regime switching, block bootstrap, jump diffusion) across multiple time horizons, scores the realism of their simulated price paths against historical BTC behavior, optimizes parameters, and produces a ranked leaderboard of model+params+horizon combinations.

Built for digital credit stress-testing and risk analysis.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Install dev dependencies (optional)
pip install -e ".[dev]"

# Fetch latest BTC data
python scripts/fetch_data.py

# Run quick test (sanity check)
python scripts/run_experiment.py --config config/default.yaml --mode quick

# Run standard experiment (meaningful comparison)
python scripts/run_experiment.py --config config/default.yaml --mode standard

# Run full experiment (production-grade with confidence intervals)
python scripts/run_experiment.py --config config/default.yaml --mode full

# Test single model
python scripts/run_single_model.py --model gbm --horizon 180d
```

## Architecture

```
btc-sim-engine/
├── config/              # Experiment configurations
├── data/                # Data loading and preprocessing
├── models/              # Stochastic price path models
├── validation/          # Scoring metrics and backtesting
├── optimization/        # Parameter search and experiment runner
├── reporting/           # Leaderboards and comparisons
├── viz/                 # Visualization and plotting
├── scripts/             # CLI entry points
└── tests/               # Unit and integration tests
```

## Models

- **Regime-Conditioned Block Bootstrap** — Flagship model combining Markov regime switching with stationary block bootstrap
- **GBM** — Geometric Brownian Motion baseline
- **GARCH(1,1)** — Volatility clustering benchmark

## Time Horizons

- 30d — Short-term credit monitoring
- 90d — Quarterly risk reviews
- 180d — Semi-annual stress tests
- 365d — Annual risk assessment
- 1460d — Full BTC cycle (~4 years)

## Validation System

**9 Scoring Metrics** (feed into composite score):
- Path Dynamics (35%): Drawdown duration, recovery time, time-in-drawdown
- Tail Risk (25%): Tail index accuracy, VaR backtest
- Distributional (15%): KS statistic, QQ divergence
- Temporal (10%): Volatility clustering
- Forecast (15%): MAPE

**Diagnostic Metrics** (reported, not scored):
- MAE, MASE, moment matching, ES match, max drawdown depth

## License

Proprietary — Internal use only

## Attribution

Built with Claude Code by @LongGamma
