"""
Diagnostic visualizations — QQ plots, ACF plots, tail diagnostics.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional

from .style import COLORS, add_watermark


def plot_qq(
    sim_returns: np.ndarray,
    hist_returns: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "QQ Plot — Simulated vs Historical",
    n_quantiles: int = 200,
) -> plt.Axes:
    """
    QQ plot comparing simulated and historical return quantiles.

    Parameters
    ----------
    sim_returns : np.ndarray
        Simulated log returns (flattened internally).
    hist_returns : np.ndarray
        Historical log returns, 1D.
    ax : matplotlib.axes.Axes, optional
    title : str
    n_quantiles : int

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    sim_flat = sim_returns.flatten()
    quantiles = np.linspace(0.5, 99.5, n_quantiles)

    hist_q = np.percentile(hist_returns, quantiles)
    sim_q = np.percentile(sim_flat, quantiles)

    ax.scatter(hist_q, sim_q, s=8, color=COLORS["btc_orange"], alpha=0.7, zorder=5)

    # Perfect match line
    lo = min(hist_q.min(), sim_q.min())
    hi = max(hist_q.max(), sim_q.max())
    ax.plot([lo, hi], [lo, hi], color=COLORS["text_dim"], linestyle="--", linewidth=1, zorder=3)

    ax.set_xlabel("Historical Quantiles")
    ax.set_ylabel("Simulated Quantiles")
    ax.set_title(title, fontsize=14, color=COLORS["text"])
    ax.set_aspect("equal", adjustable="box")
    add_watermark(ax)

    return ax


def plot_acf_squared(
    sim_returns: np.ndarray,
    hist_returns: np.ndarray,
    max_lag: int = 30,
    ax: Optional[plt.Axes] = None,
    title: str = "ACF of Squared Returns (Volatility Clustering)",
) -> plt.Axes:
    """
    Compare autocorrelation of squared returns.

    Parameters
    ----------
    sim_returns : np.ndarray
        Simulated log returns, shape (n_sims, n_steps).
    hist_returns : np.ndarray
        Historical log returns, 1D.
    max_lag : int
    ax : matplotlib.axes.Axes, optional
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    def _acf_squared(returns: np.ndarray, max_lag: int) -> np.ndarray:
        r2 = returns ** 2
        r2c = r2 - np.mean(r2)
        var = np.var(r2)
        if var == 0:
            return np.zeros(max_lag)
        return np.array([
            np.mean(r2c[lag:] * r2c[:-lag]) / var
            for lag in range(1, max_lag + 1)
        ])

    lags = np.arange(1, max_lag + 1)

    # Historical
    hist_acf = _acf_squared(hist_returns, max_lag)
    ax.bar(lags - 0.2, hist_acf, width=0.4, color=COLORS["text"], alpha=0.6, label="Historical")

    # Average across simulated paths
    n_paths = min(sim_returns.shape[0], 200)
    sim_acfs = np.array([_acf_squared(sim_returns[i], max_lag) for i in range(n_paths)])
    sim_acf_mean = np.mean(sim_acfs, axis=0)
    sim_acf_std = np.std(sim_acfs, axis=0)

    ax.bar(lags + 0.2, sim_acf_mean, width=0.4, color=COLORS["btc_orange"], alpha=0.6, label="Simulated (mean)")
    ax.errorbar(lags + 0.2, sim_acf_mean, yerr=sim_acf_std, fmt="none",
                ecolor=COLORS["btc_orange"], alpha=0.4, capsize=2)

    ax.axhline(0, color=COLORS["text_dim"], linewidth=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title, fontsize=14, color=COLORS["text"])
    ax.legend()
    add_watermark(ax)

    return ax


def plot_drawdown_analysis(
    sim_paths: np.ndarray,
    hist_prices: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Drawdown Duration Distribution",
) -> plt.Axes:
    """
    Compare drawdown duration distributions between simulated and historical.

    Parameters
    ----------
    sim_paths : np.ndarray
        Simulated price paths, shape (n_sims, n_steps).
    hist_prices : np.ndarray
        Historical price series, 1D.
    ax : matplotlib.axes.Axes, optional
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    from validation.metrics import ScoringMetrics

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    # Historical drawdown durations
    hist_dds = ScoringMetrics._compute_drawdowns(hist_prices)
    hist_durations = [d["duration"] for d in hist_dds]

    # Simulated drawdown durations (pooled)
    sim_durations = []
    for i in range(min(sim_paths.shape[0], 200)):
        dds = ScoringMetrics._compute_drawdowns(sim_paths[i])
        sim_durations.extend([d["duration"] for d in dds])

    max_dur = max(
        max(hist_durations) if hist_durations else 1,
        max(sim_durations) if sim_durations else 1,
    )
    bins = np.linspace(0, min(max_dur, 500), 50)

    ax.hist(hist_durations, bins=bins, alpha=0.5, density=True,
            color=COLORS["text"], label="Historical", edgecolor="none")
    ax.hist(sim_durations, bins=bins, alpha=0.5, density=True,
            color=COLORS["btc_orange"], label="Simulated", edgecolor="none")

    ax.set_xlabel("Drawdown Duration (days)")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=14, color=COLORS["text"])
    ax.legend()
    add_watermark(ax)

    return ax
