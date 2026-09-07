"""Walk-forward backtest -- the only part of this project that can say anything
honest about whether a strategy works.

Everything before this milestone was in-sample: the optimiser saw the returns it
was optimising over, so max-Sharpe won by construction. Here, at each month-end
the weights are computed from a *trailing* window only, then held forward into
data the optimiser has never seen.

Three details do the real work:

1. **No look-ahead.** Weights decided using data through date `t` are applied
   from `t+1`. `_execution_schedule()` enforces that, and it is the single most
   important detail in the file; collapse it to same-day execution and every
   Sharpe below roughly doubles.
2. **Turnover is charged on a drifting portfolio.** Holdings move with returns
   between rebalances and are only pulled back on execution days, so keeping a
   target costs money even when the target itself never changes. Pinning
   holdings to the target would report 1/N as free, which it is not.
3. **The comparison is against 1/N and buy-and-hold**, both of which require no
   estimation at all and are therefore immune to estimation error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TRADING_DAYS, Settings, settings
from .covariance import estimate
from .optimize import Portfolio

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    """Out-of-sample record for one strategy."""

    name: str
    net_returns: pd.Series  # daily, after transaction costs
    gross_returns: pd.Series  # daily, before costs
    weights: pd.DataFrame  # rebalance date x asset
    turnover: pd.Series  # one-way, per rebalance
    metrics: dict[str, float] = field(default_factory=dict)
    is_benchmark: bool = False

    @property
    def equity(self) -> pd.Series:
        """Net cumulative growth of 1 unit."""
        return (1 + self.net_returns).cumprod()

    @property
    def drawdown(self) -> pd.Series:
        equity = self.equity
        return equity / equity.cummax() - 1


def rebalance_dates(index: pd.DatetimeIndex, cfg: Settings) -> pd.DatetimeIndex:
    """Last trading day of each period, excluding a partial final period."""
    marks = pd.Series(index, index=index).resample(cfg.rebalance).last().dropna()
    return pd.DatetimeIndex(marks.to_numpy())


def _execution_schedule(
    targets: pd.DataFrame, index: pd.DatetimeIndex
) -> dict[pd.Timestamp, pd.Series]:
    """Map each target to the first trading day it can actually be traded on.

    A target computed from the close of date t is executed at t+1. This is the
    no-look-ahead guarantee; collapse it to same-day and every Sharpe below
    roughly doubles.
    """
    schedule: dict[pd.Timestamp, pd.Series] = {}
    for mark, target in targets.iterrows():
        later = index[index > mark]
        if len(later):
            schedule[later[0]] = target
    return schedule


def _simulate(
    targets: pd.DataFrame, daily: pd.DataFrame, cfg: Settings
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Day-by-day simulation of a drifting portfolio.

    Holdings are *not* pinned to the target between rebalances -- they drift
    with returns and are only pulled back on execution days. That is what makes
    the turnover figure real: an unchanged target still costs money to maintain,
    which is precisely why 1/N is not free. Returning targets-as-holdings would
    understate every strategy's cost, equal weight most of all.
    """
    schedule = _execution_schedule(targets, daily.index)
    holdings: pd.Series | None = None
    dates, gross, traded = [], [], []

    for date in daily.index:
        period = daily.loc[date]
        trade = 0.0

        if date in schedule:
            target = schedule[date]
            # From cash the whole position is a trade; afterwards only the gap
            # between target and where drift has left us.
            trade = (
                float(target.abs().sum())
                if holdings is None
                else float((target - holdings).abs().sum() / 2)
            )
            holdings = target

        if holdings is None:
            continue  # not invested yet

        dates.append(date)
        gross.append(float(holdings @ period))
        traded.append(trade)

        grown = holdings * (1 + period)
        holdings = grown / grown.sum()  # fully invested, no cash drag

    index = pd.DatetimeIndex(dates)
    gross_returns = pd.Series(gross, index=index)
    turnover = pd.Series(traded, index=index)
    cost = turnover * cfg.transaction_cost_bps / 10_000
    return gross_returns, gross_returns - cost, turnover


def walk_forward(
    daily: pd.DataFrame,
    strategy,
    name: str,
    cfg: Settings = settings,
) -> BacktestResult:
    """Roll the estimation window forward, rebalancing on schedule."""
    marks = rebalance_dates(daily.index, cfg)
    rows: dict[pd.Timestamp, pd.Series] = {}
    failures = 0

    for mark in marks:
        window = daily.loc[:mark].tail(cfg.estimation_window)
        if len(window) < cfg.min_estimation_days:
            continue
        mu, cov = estimate(window, cfg.covariance_estimator)
        try:
            portfolio: Portfolio = strategy(mu, cov, cfg)
        except (RuntimeError, ValueError) as exc:
            # A window with no positive excess return leaves the tangency
            # portfolio undefined. Hold the previous weights rather than
            # silently jumping to cash.
            failures += 1
            log.debug("%s: %s at %s", name, exc, mark.date())
            continue
        rows[mark] = portfolio.weights

    if not rows:
        raise RuntimeError(
            f"{name}: no rebalance produced weights; need at least "
            f"{cfg.min_estimation_days} days of history before the first mark"
        )
    if failures:
        log.warning(
            "%s: %s of %s rebalances failed to solve; previous weights held",
            name,
            failures,
            len(marks),
        )

    weights = pd.DataFrame(rows).T.sort_index()
    gross, net, turnover = _simulate(weights, daily, cfg)

    log.info(
        "%-20s %s rebalances, %s out-of-sample days from %s",
        name,
        len(weights),
        f"{len(net):,}",
        net.index.min().date(),
    )
    return BacktestResult(
        name=name,
        net_returns=net,
        gross_returns=gross,
        weights=weights,
        turnover=turnover[turnover > 0],
        metrics=metrics(net, gross, turnover, cfg),
    )


def buy_and_hold_backtest(
    ticker: str, daily: pd.Series, index: pd.DatetimeIndex, cfg: Settings = settings
) -> BacktestResult:
    """The benchmark over the same out-of-sample window.

    One entry trade and nothing after -- a single asset cannot drift away from
    a 100% target, so buy-and-hold genuinely has zero maintenance turnover.
    That structural advantage is the point of including it.
    """
    aligned = daily.reindex(index).dropna()
    weights = pd.DataFrame({ticker: [1.0]}, index=[aligned.index[0]])
    turnover = pd.Series([1.0], index=[aligned.index[0]])  # one entry trade
    cost = pd.Series(0.0, index=aligned.index)
    cost.iloc[0] = cfg.transaction_cost_bps / 10_000
    net = aligned - cost
    return BacktestResult(
        name=f"benchmark_{ticker}",
        net_returns=net,
        gross_returns=aligned,
        weights=weights,
        turnover=turnover,
        metrics=metrics(net, aligned, turnover, cfg),
        is_benchmark=True,
    )


def metrics(
    net: pd.Series, gross: pd.Series, turnover: pd.Series, cfg: Settings = settings
) -> dict[str, float]:
    """Standard out-of-sample performance table.

    Sharpe uses the annualised arithmetic mean (the convention), while CAGR is
    geometric -- they differ by roughly half the variance, so both are reported
    rather than one being passed off as the other.
    """
    years = len(net) / TRADING_DAYS
    equity = (1 + net).cumprod()
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    max_dd = float((equity / equity.cummax() - 1).min())
    downside = net[net < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)

    return {
        "cagr": cagr,
        "ann_return": float(net.mean() * TRADING_DAYS),
        "ann_volatility": vol,
        "sharpe": float((net.mean() * TRADING_DAYS - cfg.risk_free_rate) / vol)
        if vol > 0
        else np.nan,
        "sortino": float((net.mean() * TRADING_DAYS - cfg.risk_free_rate) / downside)
        if downside > 0
        else np.nan,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        "annual_turnover": float(turnover.sum() / years) if years > 0 else np.nan,
        "cost_drag_annual": float((gross.mean() - net.mean()) * TRADING_DAYS),
        "years": float(years),
    }


def summary(results: list[BacktestResult]) -> pd.DataFrame:
    """One row per strategy, sorted by Sharpe."""
    frame = pd.DataFrame({r.name: r.metrics for r in results}).T
    return frame.sort_values("sharpe", ascending=False)
