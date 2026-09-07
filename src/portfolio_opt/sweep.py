"""Parameter sweeps over the backtest's unexamined knobs.

M3 produced a clean ranking from a single configuration: a 756-day estimation
window, month-end rebalancing, 10bps costs. Three arbitrary choices. If the
ranking flips when any of them moves, the M3 conclusion is an artefact of the
settings rather than a property of the strategies -- and there is no way to know
without running it.

Every cell of the sweep is one MLflow run, which is why M4 pairs with this
rather than being ceremony over a single backtest.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import replace

import pandas as pd

from .backtest import buy_and_hold_backtest, summary, walk_forward
from .config import Settings, settings
from .optimize import STRATEGIES
from .tracking import configure, log_backtest, log_comparison

log = logging.getLogger(__name__)

# The three knobs, and why each could plausibly change the answer:
#   estimation_window   -- shorter reacts faster but estimates worse
#   rebalance           -- less often means less turnover, staler weights
#   transaction_cost_bps-- the high-turnover strategies are the ones exposed
GRID: dict[str, tuple] = {
    "estimation_window": (252, 504, 756, 1260),
    "rebalance": ("ME", "QE"),
    "transaction_cost_bps": (0.0, 10.0, 50.0),
}


def one_configuration(
    daily: pd.DataFrame,
    benchmark: pd.Series,
    cfg: Settings,
    track: bool = True,
) -> pd.DataFrame:
    """Backtest every strategy under one configuration; return the table."""
    results = [walk_forward(daily, fn, name, cfg) for name, fn in STRATEGIES.items()]
    results.append(
        buy_and_hold_backtest(cfg.benchmark, benchmark, results[0].net_returns.index, cfg)
    )
    table = summary(results)

    if track:
        tags = {
            "window": str(cfg.estimation_window),
            "rebalance": cfg.rebalance,
            "cost_bps": str(cfg.transaction_cost_bps),
            "estimator": cfg.covariance_estimator,
        }
        for result in results:
            log_backtest(result, cfg, tags=tags)
        log_comparison(table, cfg, name="sweep_cell")
    return table


def grid(
    daily: pd.DataFrame,
    benchmark: pd.Series,
    base: Settings = settings,
    axes: dict[str, tuple] | None = None,
    track: bool = True,
) -> pd.DataFrame:
    """Cartesian sweep. Returns a long frame: one row per (config, strategy)."""
    axes = axes or GRID
    if track:
        configure()

    names = list(axes)
    combinations = list(itertools.product(*(axes[n] for n in names)))
    log.info(
        "sweep: %s configurations x %s strategies = %s backtests",
        len(combinations),
        len(STRATEGIES) + 1,
        len(combinations) * (len(STRATEGIES) + 1),
    )

    rows = []
    for values in combinations:
        overrides = dict(zip(names, values, strict=True))
        cfg = replace(base, **overrides)
        log.info("  %s", overrides)
        table = one_configuration(daily, benchmark, cfg, track=track)
        for strategy, metrics in table.iterrows():
            rows.append({**overrides, "strategy": strategy, **metrics.to_dict()})

    return pd.DataFrame(rows)


def stability(long_frame: pd.DataFrame) -> pd.DataFrame:
    """How often does each strategy win, and how much does its Sharpe move?

    A strategy whose rank swings wildly across configurations has not been shown
    to be good or bad -- only that the M3 headline was sensitive to a setting
    nobody justified.
    """
    per_config = long_frame.groupby(
        ["estimation_window", "rebalance", "transaction_cost_bps"], dropna=False
    )
    winners = per_config["sharpe"].idxmax()
    win_counts = long_frame.loc[winners, "strategy"].value_counts()

    stats = (
        long_frame.groupby("strategy")["sharpe"]
        .agg(["mean", "std", "min", "max"])
        .rename(
            columns={
                "mean": "sharpe_mean",
                "std": "sharpe_std",
                "min": "sharpe_min",
                "max": "sharpe_max",
            }
        )
    )
    stats["sharpe_range"] = stats["sharpe_max"] - stats["sharpe_min"]
    stats["wins"] = win_counts.reindex(stats.index).fillna(0).astype(int)
    stats["rank_mean"] = (
        long_frame.assign(rank=per_config["sharpe"].rank(ascending=False))
        .groupby("strategy")["rank"]
        .mean()
    )
    return stats.sort_values("sharpe_mean", ascending=False)
