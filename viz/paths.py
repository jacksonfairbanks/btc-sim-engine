"""
Price path visualizations — fan charts and sample paths.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .style import COLORS, add_watermark, format_price_axis


def plot_fan_chart(
    sim_paths: np.ndarray,
    initial_price: float,
    horizon_label: str = "",
    hist_prices: np.ndarray | None = None,
    ax: Optional[plt.Axes] = None,
    title: str | None = None,
    show_median: bool = True,
) -> plt.Axes:
    """
    Fan chart showing confidence bands of simulated price paths.

    Parameters
    ----------
    sim_paths : np.ndarray
        Simulated price paths, shape (n_sims, n_steps).
    initial_price : float
        Starting price (for annotation).
    horizon_label : str
        Label like "180d" for chart title.
    hist_prices : np.ndarray, optional
        Realized historical prices to overlay.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    title : str, optional
        Custom title. Defaults to auto-generated.
    show_median : bool
        Whether to plot the median path.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    n_steps = sim_paths.shape[1]
    x = np.arange(n_steps)

    # Compute percentile bands
    p5 = np.percentile(sim_paths, 5, axis=0)
    p12_5 = np.percentile(sim_paths, 12.5, axis=0)
    p25 = np.percentile(sim_paths, 25, axis=0)
    p37_5 = np.percentile(sim_paths, 37.5, axis=0)
    p50 = np.median(sim_paths, axis=0)
    p62_5 = np.percentile(sim_paths, 62.5, axis=0)
    p75 = np.percentile(sim_paths, 75, axis=0)
    p87_5 = np.percentile(sim_paths, 87.5, axis=0)
    p95 = np.percentile(sim_paths, 95, axis=0)

    # 90% band (5th-95th)
    ax.fill_between(x, p5, p95, alpha=0.08, color=COLORS["btc_orange"], label="90% CI")
    # 75% band
    ax.fill_between(x, p12_5, p87_5, alpha=0.12, color=COLORS["btc_orange"], label="75% CI")
    # 50% band
    ax.fill_between(x, p25, p75, alpha=0.20, color=COLORS["btc_orange"], label="50% CI")
    # 25% band
    ax.fill_between(x, p37_5, p62_5, alpha=0.35, color=COLORS["btc_orange"], label="25% CI")

    # Median path
    if show_median:
        ax.plot(x, p50, color=COLORS["btc_orange"], linewidth=2, label="Median", zorder=5)

    # Historical overlay
    if hist_prices is not None:
        n_hist = min(len(hist_prices), n_steps)
        ax.plot(
            x[:n_hist], hist_prices[:n_hist],
            color=COLORS["text"],
            linewidth=1.5,
            linestyle="--",
            label="Realized",
            zorder=6,
        )

    # Formatting
    if title:
        ax.set_title(title, fontsize=14, color=COLORS["text"])
    elif horizon_label:
        ax.set_title(
            f"Simulated BTC Price Paths — {horizon_label}",
            fontsize=14,
            color=COLORS["text"],
        )

    ax.set_xlabel("Days", color=COLORS["text_dim"])
    ax.set_ylabel("Price (USD)", color=COLORS["text_dim"])
    format_price_axis(ax)
    ax.legend(loc="upper left", framealpha=0.8)
    add_watermark(ax)

    return ax


def plot_paths_sample(
    sim_paths: np.ndarray,
    n_show: int = 20,
    hist_prices: np.ndarray | None = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Sample Simulated Paths",
    alpha: float = 0.3,
) -> plt.Axes:
    """
    Plot a sample of individual simulated paths.

    Parameters
    ----------
    sim_paths : np.ndarray
        Simulated price paths, shape (n_sims, n_steps).
    n_show : int
        Number of paths to display.
    hist_prices : np.ndarray, optional
        Realized prices to overlay.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str
        Chart title.
    alpha : float
        Transparency of individual paths.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    n_steps = sim_paths.shape[1]
    x = np.arange(n_steps)

    # Randomly sample paths
    indices = np.random.choice(sim_paths.shape[0], size=min(n_show, sim_paths.shape[0]), replace=False)

    for idx in indices:
        ax.plot(x, sim_paths[idx], color=COLORS["btc_orange"], alpha=alpha, linewidth=0.8)

    # Historical overlay
    if hist_prices is not None:
        n_hist = min(len(hist_prices), n_steps)
        ax.plot(
            x[:n_hist], hist_prices[:n_hist],
            color=COLORS["text"],
            linewidth=2,
            linestyle="--",
            label="Realized",
            zorder=10,
        )
        ax.legend(loc="upper left")

    ax.set_title(title, fontsize=14, color=COLORS["text"])
    ax.set_xlabel("Days", color=COLORS["text_dim"])
    ax.set_ylabel("Price (USD)", color=COLORS["text_dim"])
    format_price_axis(ax)
    add_watermark(ax)

    return ax
