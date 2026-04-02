"""
Dark Bloomberg-terminal chart theme.

Background: #0a0a0a (near black)
Card/panel: #111111
Grid: #1a1a1a
Primary accent: #f7931a (Bitcoin orange)
Secondary: #3b82f6 (blue), #10b981 (green), #ef4444 (red)
Font: JetBrains Mono / IBM Plex Mono fallback
Attribution: @LongGamma watermark
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ── Color palette ───────────────────────────────────────────────────────
COLORS = {
    "background": "#0a0a0a",
    "panel": "#111111",
    "grid": "#1a1a1a",
    "text": "#e0e0e0",
    "text_dim": "#666666",
    "btc_orange": "#f7931a",
    "blue": "#3b82f6",
    "green": "#10b981",
    "red": "#ef4444",
    "yellow": "#eab308",
    "purple": "#8b5cf6",
    # Fan chart confidence bands (orange gradient, low to high alpha)
    "fan_90": "#f7931a15",
    "fan_75": "#f7931a25",
    "fan_50": "#f7931a40",
    "fan_25": "#f7931a60",
}

# Ordered palette for multi-model comparison
MODEL_COLORS = [
    COLORS["btc_orange"],
    COLORS["blue"],
    COLORS["green"],
    COLORS["red"],
    COLORS["purple"],
    COLORS["yellow"],
]

FONT_FAMILY = "JetBrains Mono, IBM Plex Mono, Consolas, monospace"


def get_colors() -> dict[str, str]:
    """Return the color palette dict."""
    return COLORS.copy()


def apply_style() -> None:
    """Apply the dark Bloomberg-terminal style globally."""
    plt.style.use("dark_background")

    mpl.rcParams.update({
        # Figure
        "figure.facecolor": COLORS["background"],
        "figure.edgecolor": COLORS["background"],
        "figure.figsize": (14, 7),
        "figure.dpi": 100,

        # Axes
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "axes.grid": True,
        "axes.titlesize": 14,
        "axes.labelsize": 11,

        # Grid
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,

        # Text
        "text.color": COLORS["text"],
        "font.family": "monospace",
        "font.size": 10,

        # Ticks
        "xtick.color": COLORS["text_dim"],
        "ytick.color": COLORS["text_dim"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # Legend
        "legend.facecolor": COLORS["panel"],
        "legend.edgecolor": COLORS["grid"],
        "legend.fontsize": 9,

        # Lines
        "lines.linewidth": 1.5,
        "lines.antialiased": True,

        # Savefig
        "savefig.facecolor": COLORS["background"],
        "savefig.edgecolor": COLORS["background"],
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def add_watermark(ax: plt.Axes, text: str = "@LongGamma") -> None:
    """Add subtle watermark to bottom-right of axes."""
    ax.text(
        0.99, 0.01, text,
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["text_dim"],
        alpha=0.4,
        ha="right",
        va="bottom",
        family="monospace",
    )


def format_price_axis(ax: plt.Axes) -> None:
    """Format y-axis for price display (dollar sign, commas)."""
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
