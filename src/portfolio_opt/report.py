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
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import REPORTS_DIR, ensure_dirs  # noqa: E402
from .optimize import Portfolio  # noqa: E402

log = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e7e6e2"
# Five categorical slots in fixed order. Validated for *adjacent* pair
# separation, which is the pairlist that applies to line charts -- so these are
# used on the equity and drawdown curves. Scatter plots compare every pair at
# once and only clear the floor at three slots, so the frontier chart uses
# neutral marks with direct labels instead of colour.
#
# Three of these sit below 3:1 contrast on the light surface, which obliges the
# relief rule: the table views (summary.csv, backtest_summary.csv) are it.
SERIES = {
    "equal_weight": "#2a78d6",
    "min_variance": "#eb6834",
    "max_sharpe": "#1baf7a",
    "risk_parity": "#eda100",
    "max_diversification": "#e87ba4",
}
LABELS = {
    "equal_weight": "Equal weight",
    "min_variance": "Min variance",
    "max_sharpe": "Max Sharpe",
    "risk_parity": "Risk parity",
    "max_diversification": "Max diversification",
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
            # Neutral on purpose: six marks exceed the three-slot all-pairs
            # colour floor, and every point is directly labelled anyway.
            color=INK if p.is_benchmark else INK_SECONDARY,
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


# --------------------------------------------------------------------------
# backtest charts
# --------------------------------------------------------------------------


def _colour(result) -> str:
    return INK if result.is_benchmark else SERIES[result.name]


def _name_label(name: str) -> str:
    """Display label from a bare strategy name (no Portfolio object to hand)."""
    if name.startswith("benchmark_"):
        return f"{name.removeprefix('benchmark_')} (buy & hold)"
    return LABELS.get(name, name)


def _result_label(result) -> str:
    return _name_label(result.name)


def _tick_label(name: str) -> str:
    """Two-line axis label. "SPY (buy & hold)" would wrap to four lines."""
    if name.startswith("benchmark_"):
        return f"{name.removeprefix('benchmark_')}\n(benchmark)"
    return LABELS.get(name, name).replace(" ", "\n")


def plot_equity_curve(results: list, path=None) -> plt.Figure:
    """Net cumulative growth of 1 unit, out of sample.

    Log y-axis: on a linear axis a 7-year curve makes early differences
    invisible and late ones look larger than they are. Equal vertical distance
    should mean equal percentage change.
    """
    ensure_dirs()
    path = path or REPORTS_DIR / "equity_curve.png"

    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=SURFACE)
    _style(ax)

    for result in results:
        equity = result.equity
        ax.plot(
            equity.index,
            equity.to_numpy(),
            color=_colour(result),
            linewidth=2.2 if result.is_benchmark else 2,
            linestyle="--" if result.is_benchmark else "-",
            zorder=3 if result.is_benchmark else 2,
            label=_result_label(result),
        )

    ax.set_yscale("log")
    # A log axis defaults to decade ticks, which on a 0.6x-3x range leaves one
    # label and a scattering of scientific-notation minors. Pin sensible ones.
    span = [r.equity for r in results]
    low = min(float(e.min()) for e in span)
    high = max(float(e.max()) for e in span)
    ticks = [
        t
        for t in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
        if low * 0.95 <= t <= high * 1.05
    ]
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}x")
    ax.set_title(
        "Out-of-sample growth of 1 unit, net of costs",
        fontsize=13,
        color=INK,
        loc="left",
        pad=14,
    )
    ax.set_ylabel("Cumulative growth (log scale)", fontsize=10, color=INK_SECONDARY)

    legend = ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=2)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s", path)
    return fig


def plot_drawdown(results: list, path=None) -> plt.Figure:
    """Underwater curves. The benchmark's depth is the number to beat."""
    ensure_dirs()
    path = path or REPORTS_DIR / "drawdown.png"

    fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=SURFACE)
    _style(ax)

    for result in results:
        drawdown = result.drawdown
        ax.plot(
            drawdown.index,
            drawdown.to_numpy(),
            color=_colour(result),
            linewidth=2.2 if result.is_benchmark else 2,
            linestyle="--" if result.is_benchmark else "-",
            zorder=3 if result.is_benchmark else 2,
            label=_result_label(result),
        )

    ax.axhline(0, color=GRID, linewidth=1, zorder=1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Drawdown from peak", fontsize=13, color=INK, loc="left", pad=14)
    ax.set_ylabel("Loss from high-water mark", fontsize=10, color=INK_SECONDARY)

    legend = ax.legend(frameon=False, fontsize=9, loc="lower left", ncols=3)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s", path)
    return fig


def plot_sharpe_decay(
    in_sample: dict[str, float], out_of_sample: dict[str, float], path=None
) -> plt.Figure:
    """In-sample vs out-of-sample Sharpe, side by side.

    The single most useful chart in the project: it shows how much of each
    strategy's apparent edge survives contact with data it has not seen. Two
    series, so two colours -- and grouped bars carry a 2px surface gap.
    """
    ensure_dirs()
    path = path or REPORTS_DIR / "sharpe_decay.png"

    names = [n for n in in_sample if n in out_of_sample]
    names.sort(key=lambda n: out_of_sample[n], reverse=True)
    positions = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)

    for offset, (values, colour, label) in enumerate(
        [
            (in_sample, INK_MUTED, "In-sample (ex-ante)"),
            (out_of_sample, "#2a78d6", "Out-of-sample (walk-forward)"),
        ]
    ):
        bars = [values[n] for n in names]
        ax.bar(
            positions + (offset - 0.5) * (width + 0.02),
            bars,
            width=width,
            color=colour,
            zorder=2,
            label=label,
        )
        for x, value in zip(positions, bars, strict=True):
            ax.annotate(
                f"{value:.2f}",
                (x + (offset - 0.5) * (width + 0.02), value),
                textcoords="offset points",
                xytext=(0, 4 if value >= 0 else -12),
                ha="center",
                fontsize=8.5,
                color=INK_SECONDARY,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([_tick_label(n) for n in names], fontsize=9, color=INK)
    ax.axhline(0, color=GRID, linewidth=1, zorder=1)
    # Headroom for the value labels, so the legend cannot sit on top of them.
    ceiling = max([*in_sample.values(), *out_of_sample.values()])
    ax.set_ylim(top=ceiling * 1.22)
    ax.set_title(
        "How much of the edge survives out of sample?",
        fontsize=13,
        color=INK,
        loc="left",
        pad=14,
    )
    ax.set_ylabel("Sharpe ratio", fontsize=10, color=INK_SECONDARY)

    legend = ax.legend(
        frameon=False, fontsize=9, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.0)
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s", path)
    return fig


def write_backtest_table(results: list, path=None) -> pd.DataFrame:
    """Out-of-sample metrics, one row per strategy. The accessible table view."""
    ensure_dirs()
    path = path or REPORTS_DIR / "backtest_summary.csv"
    frame = pd.DataFrame({r.name: r.metrics for r in results}).T.sort_values(
        "sharpe", ascending=False
    )
    frame.to_csv(path, index_label="strategy")
    log.info("wrote %s", path)
    return frame


def plot_sweep_sensitivity(
    long_frame: pd.DataFrame,
    rebalance: str = "ME",
    cost_bps: float = 10.0,
    path=None,
) -> plt.Figure:
    """Out-of-sample Sharpe against estimation window, one line per strategy.

    Filtered to a single rebalance frequency and cost level so this stays one
    panel of lines: small multiples would compare every colour pair at once and
    only three slots clear that floor, whereas five clear the adjacent-pair gates
    that apply to lines.

    The chart exists to answer one question -- would a different, equally
    arbitrary parameter choice have produced a different headline?
    """
    ensure_dirs()
    path = path or REPORTS_DIR / "sweep_sensitivity.png"

    frame = long_frame[
        (long_frame["rebalance"] == rebalance)
        & (long_frame["transaction_cost_bps"] == cost_bps)
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.6), facecolor=SURFACE)
    _style(ax)

    for name, group in frame.groupby("strategy"):
        group = group.sort_values("estimation_window")
        is_benchmark = str(name).startswith("benchmark_")
        ax.plot(
            group["estimation_window"],
            group["sharpe"],
            color=INK if is_benchmark else SERIES[name],
            linewidth=2.2 if is_benchmark else 2,
            linestyle="--" if is_benchmark else "-",
            marker="D" if is_benchmark else "o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3 if is_benchmark else 2,
            label=_name_label(str(name)),
        )

    ax.set_xticks(sorted(frame["estimation_window"].unique()))
    ax.set_title(
        f"Does the ranking survive a different estimation window? "
        f"({rebalance} rebalance, {cost_bps:.0f}bps)",
        fontsize=12,
        color=INK,
        loc="left",
        pad=14,
    )
    ax.set_xlabel("Estimation window (trading days)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Out-of-sample Sharpe", fontsize=10, color=INK_SECONDARY)

    # Below the axes: the max-Sharpe line sweeps through every in-plot corner.
    legend = ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncols=3,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s", path)
    return fig
