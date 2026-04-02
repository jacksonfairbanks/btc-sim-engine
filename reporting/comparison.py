"""
Side-by-side model comparison charts.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from validation.scorer import ScoreCard
from viz.style import COLORS, MODEL_COLORS, add_watermark


def plot_score_comparison(
    results: list[ScoreCard],
    ax: Optional[plt.Axes] = None,
    title: str = "Model Comparison — Scoring Metrics",
) -> plt.Axes:
    """
    Grouped bar chart comparing scoring metrics across models.

    Parameters
    ----------
    results : list[ScoreCard]
        ScoreCards to compare (typically best per model for a given horizon).
    ax : matplotlib.axes.Axes, optional
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    metric_names = sorted(results[0].scoring_metrics.keys())
    n_metrics = len(metric_names)
    n_models = len(results)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, card in enumerate(results):
        values = [card.scoring_metrics.get(m, 0) for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * width
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.bar(
            x + offset, values, width,
            label=f"{card.model_name} ({card.horizon})",
            color=color,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in metric_names],
        fontsize=8,
        rotation=0,
    )
    ax.set_ylabel("Score (0-1)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=14, color=COLORS["text"])
    ax.legend(loc="lower right", fontsize=9)
    add_watermark(ax)

    return ax


def plot_composite_by_horizon(
    results: list[ScoreCard],
    ax: Optional[plt.Axes] = None,
    title: str = "Composite Score by Horizon",
) -> plt.Axes:
    """
    Line chart of composite score across horizons per model.

    Parameters
    ----------
    results : list[ScoreCard]
        All results (will be grouped by model).
    ax : matplotlib.axes.Axes, optional
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    # Group by model
    models = {}
    for card in results:
        if card.model_name not in models:
            models[card.model_name] = []
        models[card.model_name].append(card)

    for i, (model_name, cards) in enumerate(sorted(models.items())):
        # Sort by horizon days
        cards.sort(key=lambda c: int(c.horizon.replace("d", "")))
        horizons = [c.horizon for c in cards]
        scores = [c.composite_score for c in cards]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        ax.plot(horizons, scores, "o-", color=color, label=model_name,
                linewidth=2, markersize=8)

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Composite Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=14, color=COLORS["text"])
    ax.legend()
    add_watermark(ax)

    return ax
