#!/usr/bin/env python
"""
CLI entry point for full experiment runs.

Runs model x horizon grid with parameter optimization + walk-forward,
produces leaderboard, saves granular data for dashboard, and exports results.
"""
import click
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from data.loader import BTCDataLoader, get_price_array
from optimization.experiment import ExperimentRunner
from reporting.leaderboard import Leaderboard
from reporting.comparison import plot_score_comparison, plot_composite_by_horizon
from reporting.export import ResultsExporter
from viz.style import apply_style
from viz.paths import plot_fan_chart


@click.command()
@click.option("--config", "config_path", default="config/default.yaml",
              help="Path to experiment YAML config.")
@click.option("--mode", "run_mode", default=None,
              help="Override run mode: quick, standard, full.")
@click.option("--output-dir", default=None, help="Override output directory.")
@click.option("--no-charts", is_flag=True, help="Skip chart generation.")
def main(config_path: str, run_mode: str | None, output_dir: str | None,
         no_charts: bool):
    """Run full experiment with parameter optimization and scoring."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if run_mode:
        config["experiment"]["run_mode"] = run_mode

    out_dir = Path(output_dir or config.get("output", {}).get("dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_config = config.get("data", {})
    loader = BTCDataLoader(
        ticker=data_config.get("ticker", "BTC-USD"),
        start_date=data_config.get("start_date", "2013-01-01"),
    )
    train_df, test_df = loader.get_train_test_split(
        train_pct=data_config.get("train_pct", 0.7)
    )

    train_returns = loader.get_returns_array(train_df)
    train_prices = get_price_array(train_df)
    test_prices = get_price_array(test_df)
    test_returns = loader.get_returns_array(test_df)

    # Build full date array for walk-forward labeling
    full_df = loader.load_processed_data()
    dates = np.array([str(d.date()) for d in full_df.index])

    # Run experiment
    runner = ExperimentRunner(
        config=config,
        train_returns=train_returns,
        train_prices=train_prices,
        test_prices=test_prices,
        test_returns=test_returns,
        dates=dates,
        train_end_idx=len(train_df),
    )
    results = runner.run()

    # Save granular data for dashboard
    runner.save_granular_results(output_dir=str(out_dir))

    # Leaderboard
    leaderboard = Leaderboard(results)
    leaderboard.print_summary()
    leaderboard.print_detailed(top_n=5)

    # Export
    exporter = ResultsExporter(results, output_dir=str(out_dir))
    paths = exporter.export_all()
    print(f"\nExported results:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

    # Charts
    if not no_charts:
        apply_style()

        # Composite by horizon
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_composite_by_horizon(results, ax=ax)
        fig.savefig(out_dir / "composite_by_horizon.png")
        plt.close(fig)

        # Score comparison per horizon
        best_per_horizon = leaderboard.get_best_per_horizon()
        if len(best_per_horizon) > 0:
            fig, ax = plt.subplots(figsize=(14, 7))
            plot_score_comparison(list(best_per_horizon.values()), ax=ax)
            fig.savefig(out_dir / "score_comparison.png")
            plt.close(fig)

        # Fan charts for best model at each horizon
        for horizon_str, card in best_per_horizon.items():
            horizon_days = int(horizon_str.replace("d", ""))
            model_name = card.model_name

            from models.registry import get_model
            model = get_model(model_name)
            model.set_params(**card.params)
            model.fit(train_returns)

            initial_price = train_prices[-1]
            sim = model.simulate(
                n_simulations=2000,
                n_steps=horizon_days,
                initial_price=initial_price,
                seed=42,
            )

            n_oos = min(horizon_days + 1, len(test_prices))
            fig, ax = plt.subplots(figsize=(14, 7))
            plot_fan_chart(sim.paths, initial_price, horizon_str,
                          test_prices[:n_oos], ax=ax,
                          title=f"{model_name} -- {horizon_str} (Composite: {card.composite_score:.4f})")
            fig.savefig(out_dir / f"fan_{model_name}_{horizon_str}.png")
            plt.close(fig)

        print(f"  Charts saved to {out_dir}/")

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
