"""Visualization module — dark Bloomberg-terminal aesthetic."""
from .style import apply_style, get_colors
from .paths import plot_fan_chart, plot_paths_sample
from .distributions import plot_return_distributions, plot_tail_comparison
from .diagnostics import plot_qq, plot_acf_squared, plot_drawdown_analysis

__all__ = [
    "apply_style",
    "get_colors",
    "plot_fan_chart",
    "plot_paths_sample",
    "plot_return_distributions",
    "plot_tail_comparison",
    "plot_qq",
    "plot_acf_squared",
    "plot_drawdown_analysis",
]
