"""Report layer: efficient frontier and weights charts written to reports/.

Palette is the validated three-slot categorical set (blue/orange/aqua); it
clears the all-pairs CVD and normal-vision floors, which is what a scatter
needs. Aqua sits under 3:1 on the light surface, so every point is direct
labelled and a table view is written alongside the figures.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import REPORTS_DIR, ensure_dirs  # noqa: E402
from .optimize import Portfolio  # noqa: E402

log = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e7e6e2"
# Three validated categorical slots. A fourth hue would break the all-pairs
# colour-vision floor, so the benchmark is encoded by shape in neutral ink
# instead -- which also says the right thing: it is a reference, not a strategy.
SERIES = {"equal_weight": "#2a78d6", "min_variance": "#eb6834", "max_sharpe": "#1baf7a"}
LABELS = {
    "equal_weight": "Equal weight",
    "min_variance": "Min variance",
    "max_sharpe": "Max Sharpe",
}


def _label(portfolio: Portfolio) -> str:
    """Display name. The prefix strip is cosmetic -- branching is on the flag."""
    if portfolio.is_benchmark:
        return f"{portfolio.name.removeprefix('benchmark_')} (buy & hold)"
    return LABELS.get(portfolio.name, portfolio.name)


def _style(ax: plt.Axes) -> None:
    """Recessive axes: no top/right spines, hairline grid, muted tick ink."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=1, alpha=0.9)
    ax.set_axisbelow(True)


def plot_frontier(
    frontier: pd.DataFrame, portfolios: list[Portfolio], path=None
) -> plt.Figure:
    """Frontier curve with each strategy marked, direct-labelled, and legended."""
    ensure_dirs()
    path = path or REPORTS_DIR / "efficient_frontier.png"

    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
    _style(ax)

    ax.plot(
        frontier["volatility"],
        frontier["expected_return"],
        color=INK_MUTED,
        linewidth=2,
        zorder=2,
        label="Efficient frontier",
    )

    for p in portfolios:
        ax.scatter(
            p.volatility,
            p.expected_return,
            s=130 if p.is_benchmark else 110,
            color=INK if p.is_benchmark else SERIES[p.name],
            marker="D" if p.is_benchmark else "o",
            edgecolor=SURFACE,  # 2px surface ring keeps overlapping marks legible
            linewidth=2,
            zorder=3,
            label=_label(p),
        )
        ax.annotate(
            f"{_label(p)}\nSharpe {p.sharpe:.2f}",
            (p.volatility, p.expected_return),
            textcoords="offset points",
            xytext=(10, 6),
            fontsize=9,
            color=INK_SECONDARY,
            zorder=4,
        )

    ax.set_title(
        "Efficient frontier — S&P sector ETFs",
        fontsize=13,
        color=INK,
        loc="left",
        pad=14,
    )
    ax.set_xlabel("Annualised volatility", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Annualised expected return", fontsize=10, color=INK_SECONDARY)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s", path)
    return fig


def plot_weights(portfolio: Portfolio, path=None) -> plt.Figure:
    """Single-series horizontal bars — one hue, no legend, values labelled."""
    ensure_dirs()
    path = path or REPORTS_DIR / f"weights_{portfolio.name}.png"

    weights = portfolio.to_frame()["weight"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(weights) + 2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)

    ax.barh(
        weights.index,
        weights.to_numpy(),
        color="#2a78d6",
        height=0.68,  # leaves a visible surface gap between adjacent bars
        zorder=2,
    )
    for name, value in weights.items():
        ax.annotate(
            f"{value:.1%}",
            (value, name),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    ax.set_title(
        f"{_label(portfolio)} — allocation",
        fontsize=13,
        color=INK,
        loc="left",
        pad=14,
    )
    ax.set_xlabel("Portfolio weight", fontsize=10, color=INK_SECONDARY)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, min(1.0, weights.max() * 1.18))
    ax.tick_params(axis="y", labelcolor=INK)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s", path)
    return fig


def write_table(portfolios: list[Portfolio], path=None) -> pd.DataFrame:
    """Table view of every strategy — the accessible counterpart to the charts."""
    ensure_dirs()
    path = path or REPORTS_DIR / "summary.csv"
    frame = pd.DataFrame(
        [
            {
                "strategy": p.name,
                "expected_return": p.expected_return,
                "volatility": p.volatility,
                "sharpe": p.sharpe,
                **p.weights.round(4).to_dict(),
            }
            for p in portfolios
        ]
    )
    frame.to_csv(path, index=False)
    log.info("wrote %s", path)
    return frame
