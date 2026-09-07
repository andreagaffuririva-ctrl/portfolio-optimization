"""Walk-forward engine tests.

The critical property is that nothing here can see the future. That is tested
two ways: mechanically (execution lands on t+1) and behaviourally (mutating
future returns must not change past results).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_opt.backtest import (
    _execution_schedule,
    buy_and_hold_backtest,
    metrics,
    rebalance_dates,
    summary,
    walk_forward,
)
from portfolio_opt.config import TRADING_DAYS, Settings
from portfolio_opt.optimize import Portfolio, min_variance

CFG = Settings(
    max_weight=1.0,
    risk_free_rate=0.0,
    estimation_window=60,
    min_estimation_days=40,
    transaction_cost_bps=10.0,
)


@pytest.fixture
def daily() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2020-01-01", periods=400)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(len(index), 3)),
        index=index,
        columns=["AAA", "BBB", "CCC"],
    )


def fixed_weights(mu, cov, cfg):
    """A strategy that always holds the first asset -- isolates the accounting."""
    weights = pd.Series(0.0, index=mu.index)
    weights.iloc[0] = 1.0
    return Portfolio("fixed", weights, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# no look-ahead
# --------------------------------------------------------------------------


def test_execution_lands_on_the_day_after_the_decision(daily):
    targets = pd.DataFrame(
        [[1.0, 0.0, 0.0]], index=[daily.index[10]], columns=daily.columns
    )
    schedule = _execution_schedule(targets, daily.index)
    assert list(schedule) == [daily.index[11]]


def test_first_return_uses_the_day_after_the_first_rebalance(daily):
    result = walk_forward(daily, fixed_weights, "fixed", CFG)
    first_mark = result.weights.index[0]
    expected_day = daily.index[daily.index > first_mark][0]

    assert result.gross_returns.index[0] == expected_day
    # Holding 100% of AAA, so the portfolio return *is* AAA's return.
    assert result.gross_returns.iloc[0] == pytest.approx(daily.loc[expected_day, "AAA"])


def test_mutating_the_future_cannot_change_the_past(daily):
    """The decisive look-ahead test.

    Replace the tail of the return series with garbage. Every net return before
    the mutation must be bit-for-bit identical, using a strategy (min variance)
    whose weights genuinely depend on the estimation window.
    """
    cut = 300
    tampered = daily.copy()
    tampered.iloc[cut:] = tampered.iloc[cut:] * -5.0 + 0.02

    original = walk_forward(daily, min_variance, "min_variance", CFG)
    altered = walk_forward(tampered, min_variance, "min_variance", CFG)

    boundary = daily.index[cut]
    left = original.net_returns[original.net_returns.index < boundary]
    right = altered.net_returns[altered.net_returns.index < boundary]

    assert len(left) > 100  # the comparison must be non-trivial
    pd.testing.assert_series_equal(left, right)


# --------------------------------------------------------------------------
# costs and turnover
# --------------------------------------------------------------------------


def test_unchanged_target_still_costs_money(daily):
    """Drift means maintaining a fixed target requires real trades.

    Regression guard: an earlier version pinned holdings to the target between
    rebalances, which reported equal weight as free to run.
    """

    def equal_target(mu, cov, cfg):
        n = len(mu)
        return Portfolio("eq", pd.Series(1.0 / n, index=mu.index), 0.0, 0.0, 0.0)

    result = walk_forward(daily, equal_target, "eq", CFG)
    maintenance = result.turnover.iloc[1:]  # exclude the initial build from cash
    assert (maintenance > 0).any()
    assert result.metrics["annual_turnover"] > 0


def test_net_returns_are_below_gross_whenever_trading_happens(daily):
    result = walk_forward(daily, min_variance, "min_variance", CFG)
    assert result.net_returns.sum() < result.gross_returns.sum()
    assert result.metrics["cost_drag_annual"] > 0


def test_zero_cost_config_makes_net_equal_gross(daily):
    free = Settings(
        max_weight=1.0,
        risk_free_rate=0.0,
        estimation_window=60,
        min_estimation_days=40,
        transaction_cost_bps=0.0,
    )
    result = walk_forward(daily, min_variance, "min_variance", free)
    pd.testing.assert_series_equal(result.net_returns, result.gross_returns)


def test_buy_and_hold_has_no_maintenance_turnover(daily):
    result = buy_and_hold_backtest("SPY", daily["AAA"], daily.index, CFG)
    assert result.turnover.sum() == pytest.approx(1.0)  # the entry trade only
    assert result.is_benchmark is True


# --------------------------------------------------------------------------
# scheduling
# --------------------------------------------------------------------------


def test_rebalance_dates_are_period_ends_within_the_index(daily):
    marks = rebalance_dates(daily.index, CFG)
    assert set(marks) <= set(daily.index)
    assert marks.is_monotonic_increasing
    # ~400 business days is ~19 months
    assert 17 <= len(marks) <= 21


def test_short_history_refuses_to_trade(daily):
    strict = Settings(min_estimation_days=10_000, estimation_window=60)
    with pytest.raises(RuntimeError, match="no rebalance produced weights"):
        walk_forward(daily, min_variance, "min_variance", strict)


def test_weights_are_recorded_per_rebalance(daily):
    result = walk_forward(daily, min_variance, "min_variance", CFG)
    assert list(result.weights.columns) == list(daily.columns)
    assert result.weights.sum(axis=1).round(6).eq(1.0).all()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_metrics_on_a_constant_return_series():
    """A fixed daily return has an exactly known CAGR and zero drawdown."""
    rate = 0.0004
    net = pd.Series(
        [rate] * TRADING_DAYS, index=pd.bdate_range("2021-01-01", periods=TRADING_DAYS)
    )
    result = metrics(net, net, pd.Series([0.0]), Settings(risk_free_rate=0.0))

    assert result["cagr"] == pytest.approx((1 + rate) ** TRADING_DAYS - 1, rel=1e-6)
    assert result["ann_volatility"] == pytest.approx(0.0)
    assert result["max_drawdown"] == pytest.approx(0.0)
    assert result["years"] == pytest.approx(1.0)


def test_max_drawdown_matches_a_hand_calculation():
    # 1.0 -> 1.2 -> 0.6 -> 0.9 : peak 1.2, trough 0.6, so -50%
    net = pd.Series([0.2, -0.5, 0.5], index=pd.bdate_range("2021-01-01", periods=3))
    result = metrics(net, net, pd.Series([0.0]), Settings())
    assert result["max_drawdown"] == pytest.approx(-0.5)


def test_cagr_and_ann_return_differ_by_volatility_drag():
    """Geometric < arithmetic whenever returns vary. Both are reported."""
    net = pd.Series(
        np.tile([0.02, -0.018], 126),
        index=pd.bdate_range("2021-01-01", periods=252),
    )
    result = metrics(net, net, pd.Series([0.0]), Settings())
    assert result["cagr"] < result["ann_return"]


def test_summary_is_sorted_by_sharpe(daily):
    results = [
        walk_forward(daily, min_variance, "min_variance", CFG),
        buy_and_hold_backtest("SPY", daily["AAA"], daily.index, CFG),
    ]
    table = summary(results)
    assert table["sharpe"].is_monotonic_decreasing
    assert set(table.index) == {"min_variance", "benchmark_SPY"}
