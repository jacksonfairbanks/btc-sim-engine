#!/usr/bin/env python
"""
Quick single-model test — fit, simulate, score, visualize.

Always uses quick settings. For validating a model before running
full experiments.
"""
import click
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from data.loader import BTCDataLoader, get_price_array
from models.registry import get_model, list_models
from validation.scorer import Scorer
from viz.style import apply_style
from viz.paths import plot_fan_chart
from viz.distributions import plot_return_distributions
from viz.diagnostics import plot_qq, plot_acf_squared, plot_drawdown_analysis
from reporting.leaderboard import Leaderboard


@click.command()
@click.option("--model", "model_name", default="gbm",
              help=f"Model to test. Available: {', '.join(list_models())}")
@click.option("--horizon", default="180d", help="Horizon (e.g. 30d, 90d, 180d, 365d, 1460d).")
@click.option("--n-sims", default=2000, help="Number of simulated paths.")
@click.option("--seed", default=42, help="Random seed.")
@click.option("--output-dir", default="results", help="Output directory for charts.")
@click.option("--no-charts", is_flag=True, help="Skip chart generation.")
def main(model_name: str, horizon: str, n_sims: int, seed: int,
         output_dir: str, no_charts: bool):
    """Run a single model test with scoring and visualization."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    horizon_days = int(horizon.replace("d", ""))

    # Load data
    print("Loading BTC data...")
    loader = BTCDataLoader()
    train_df, test_df = loader.get_train_test_split(train_pct=0.7)

    train_returns = loader.get_returns_array(train_df)
    train_prices = get_price_array(train_df)
    test_prices = get_price_array(test_df)
    test_returns = loader.get_returns_array(test_df)

    # Fit model
    print(f"\nFitting {model_name}...")
    model = get_model(model_name)
    model.fit(train_returns)
    print(f"  Fitted params: {model.get_params()}")

    # Simulate
    initial_price = train_prices[-1]
    print(f"\nSimulating {n_sims} paths x {horizon_days} days from ${initial_price:,.2f}...")
    result = model.simulate(
        n_simulations=n_sims,
        n_steps=horizon_days,
        initial_price=initial_price,
        seed=seed,
    )

    # Score
    n_oos = min(horizon_days + 1, len(test_prices))
    oos_prices = test_prices[:n_oos]
    oos_returns = test_returns[:min(horizon_days, len(test_returns))]

    print(f"\nScoring against {len(oos_prices)} days of realized data...")
    scorer = Scorer()
    card = scorer.score(
        sim_result=result,
        hist_prices=oos_prices,
        hist_returns=oos_returns,
        horizon=horizon,
        train_prices=train_prices,
    )

    # Display results
    leaderboard = Leaderboard([card])
    leaderboard.print_summary()
    leaderboard.print_detailed()

    # Charts
    if not no_charts:
        apply_style()
        prefix = f"{model_name}_{horizon}"

        print(f"\nGenerating charts to {output_path}/...")

        # Fan chart
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_fan_chart(result.paths, initial_price, horizon, oos_prices, ax=ax)
        fig.savefig(output_path / f"{prefix}_fan_chart.png")
        plt.close(fig)

        # Return distribution
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_return_distributions(result.log_returns, oos_returns, model_name, ax=ax)
        fig.savefig(output_path / f"{prefix}_returns_dist.png")
        plt.close(fig)

        # QQ plot
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_qq(result.log_returns, oos_returns, ax=ax)
        fig.savefig(output_path / f"{prefix}_qq_plot.png")
        plt.close(fig)

        # ACF of squared returns
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_acf_squared(result.log_returns, oos_returns, ax=ax)
        fig.savefig(output_path / f"{prefix}_acf_squared.png")
        plt.close(fig)

        # Drawdown analysis
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_drawdown_analysis(result.paths, oos_prices, ax=ax)
        fig.savefig(output_path / f"{prefix}_drawdowns.png")
        plt.close(fig)

        print(f"  Saved 5 charts to {output_path}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
