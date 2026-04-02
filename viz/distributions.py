"""
Return distribution plots — histograms, KDE overlays.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .style import COLORS, add_watermark


def plot_return_distributions(
    sim_returns: np.ndarray,
    hist_returns: np.ndarray,
    model_name: str = "",
    ax: Optional[plt.Axes] = None,
    title: str | None = None,
    n_bins: int = 100,
) -> plt.Axes:
    """
    Overlay histogram of simulated vs historical return distributions.

    Parameters
    ----------
    sim_returns : np.ndarray
        Simulated log returns (flattened internally).
    hist_returns : np.ndarray
        Historical log returns, 1D.
    model_name : str
        For labeling.
    ax : matplotlib.axes.Axes, optional
    title : str, optional
    n_bins : int

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    sim_flat = sim_returns.flatten()

    # Shared bin edges
    all_data = np.concatenate([sim_flat, hist_returns])
    lo, hi = np.percentile(all_data, [0.5, 99.5])
    bins = np.linspace(lo, hi, n_bins)

    ax.hist(hist_returns, bins=bins, alpha=0.5, density=True,
            color=COLORS["text"], label="Historical", edgecolor="none")
    ax.hist(sim_flat, bins=bins, alpha=0.5, density=True,
            color=COLORS["btc_orange"], label=f"Simulated ({model_name})", edgecolor="none")

    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.set_title(title or f"Return Distribution — {model_name}", fontsize=14, color=COLORS["text"])
    ax.legend()
    add_watermark(ax)

    return ax


def plot_tail_comparison(
    sim_returns: np.ndarray,
    hist_returns: np.ndarray,
    tail: str = "left",
    percentile: float = 5.0,
    ax: Optional[plt.Axes] = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Zoom into the tail of the return distribution.

    Parameters
    ----------
    sim_returns : np.ndarray
        Simulated log returns (flattened internally).
    hist_returns : np.ndarray
        Historical log returns, 1D.
    tail : str
        "left" (crash tail) or "right".
    percentile : float
        Percentile cutoff for tail.
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    sim_flat = sim_returns.flatten()

    if tail == "left":
        threshold_h = np.percentile(hist_returns, percentile)
        threshold_s = np.percentile(sim_flat, percentile)
        hist_tail = hist_returns[hist_returns <= threshold_h]
        sim_tail = sim_flat[sim_flat <= threshold_s]
        label = f"Left Tail (<{percentile}th pctile)"
    else:
        threshold_h = np.percentile(hist_returns, 100 - percentile)
        threshold_s = np.percentile(sim_flat, 100 - percentile)
        hist_tail = hist_returns[hist_returns >= threshold_h]
        sim_tail = sim_flat[sim_flat >= threshold_s]
        label = f"Right Tail (>{100-percentile}th pctile)"

    n_bins = 50
    ax.hist(hist_tail, bins=n_bins, alpha=0.5, density=True,
            color=COLORS["text"], label="Historical", edgecolor="none")
    ax.hist(sim_tail, bins=n_bins, alpha=0.5, density=True,
            color=COLORS["btc_orange"], label="Simulated", edgecolor="none")

    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.set_title(title or label, fontsize=14, color=COLORS["text"])
    ax.legend()
    add_watermark(ax)

    return ax
