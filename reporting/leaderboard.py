"""
Ranked results leaderboard.

Displays model x horizon results sorted by composite score,
with per-metric breakdowns.
"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from validation.scorer import ScoreCard


console = Console()


class Leaderboard:
    """
    Ranked leaderboard of model+horizon+params results.

    Parameters
    ----------
    results : list[ScoreCard]
        Scored results, typically from ExperimentRunner.
    """

    def __init__(self, results: list[ScoreCard]):
        self.results = sorted(results, key=lambda c: c.composite_score, reverse=True)
        for i, card in enumerate(self.results):
            card.rank = i + 1

    def print_summary(self) -> None:
        """Print compact leaderboard to terminal."""
        table = Table(
            title="BTC Price Path Simulation — Leaderboard",
            title_style="bold bright_yellow",
            border_style="dim",
            show_lines=True,
        )

        table.add_column("Rank", style="bold", width=5)
        table.add_column("Model", style="cyan", width=28)
        table.add_column("Horizon", width=8)
        table.add_column("Composite", style="bold yellow", width=10)
        table.add_column("Path Dyn", width=9)
        table.add_column("Tail Risk", width=9)
        table.add_column("Distrib", width=9)
        table.add_column("Vol Clust", width=9)
        table.add_column("MAPE", width=9)

        for card in self.results:
            m = card.scoring_metrics

            # Category averages
            path_dyn = (
                m.get("drawdown_duration_dist", 0)
                + m.get("recovery_time_dist", 0)
                + m.get("time_in_drawdown_ratio", 0)
            ) / 3

            tail_risk = (
                m.get("tail_index_accuracy", 0)
                + m.get("var_backtest_kupiec", 0)
            ) / 2

            distrib = (
                m.get("ks_statistic", 0)
                + m.get("qq_divergence", 0)
            ) / 2

            vol_clust = m.get("vol_clustering_acf", 0)
            mape_score = m.get("mape", 0)

            table.add_row(
                str(card.rank),
                card.model_name,
                card.horizon,
                f"{card.composite_score:.4f}",
                f"{path_dyn:.4f}",
                f"{tail_risk:.4f}",
                f"{distrib:.4f}",
                f"{vol_clust:.4f}",
                f"{mape_score:.4f}",
            )

        console.print(table)

    def print_detailed(self, top_n: int | None = None) -> None:
        """Print detailed per-metric breakdown for top N results."""
        cards = self.results[:top_n] if top_n else self.results

        for card in cards:
            console.print(
                Panel(
                    self._format_card(card),
                    title=f"#{card.rank} {card.model_name} @ {card.horizon}",
                    border_style="yellow" if card.rank == 1 else "dim",
                )
            )

    def _format_card(self, card: ScoreCard) -> str:
        """Format a single ScoreCard as a string."""
        lines = [f"[bold]Composite Score: {card.composite_score:.4f}[/bold]\n"]

        lines.append("[underline]Scoring Metrics:[/underline]")
        for name, val in sorted(card.scoring_metrics.items()):
            filled = int(val * 20)
            bar = "#" * filled + "." * (20 - filled)
            lines.append(f"  {name:30s} {val:.4f}  {bar}")

        lines.append("\n[underline]Diagnostics:[/underline]")
        for name, val in sorted(card.diagnostic_metrics.items()):
            if isinstance(val, dict):
                lines.append(f"  {name}:")
                for k, v in val.items():
                    if isinstance(v, float):
                        lines.append(f"    {k:30s} {v:.6f}")
                    else:
                        lines.append(f"    {k:30s} {v}")
            elif isinstance(val, float):
                lines.append(f"  {name:30s} {val:.6f}")

        lines.append(f"\n[dim]Params: {card.params}[/dim]")
        return "\n".join(lines)

    def get_best_per_horizon(self) -> dict[str, ScoreCard]:
        """Return best model for each horizon."""
        best = {}
        for card in self.results:
            if card.horizon not in best:
                best[card.horizon] = card
        return best

    def to_records(self) -> list[dict]:
        """Convert leaderboard to list of flat dicts for export."""
        records = []
        for card in self.results:
            row = {
                "rank": card.rank,
                "model": card.model_name,
                "horizon": card.horizon,
                "composite_score": card.composite_score,
                **{f"score_{k}": v for k, v in card.scoring_metrics.items()},
                **{f"param_{k}": v for k, v in card.params.items()},
            }
            records.append(row)
        return records
