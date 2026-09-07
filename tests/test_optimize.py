"""Optimiser tests use synthetic covariance so they run offline and fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_opt.config import TRADING_DAYS, Settings
from portfolio_opt.optimize import (
    buy_and_hold,
    efficient_frontier,
    equal_weight,
    max_sharpe,
    min_variance,
)
from portfolio_opt.transform import annualise

CFG = Settings(max_weight=1.0, risk_free_rate=0.0)


@pytest.fixture
def mu_cov() -> tuple[pd.Series, pd.DataFrame]:
    names = ["A", "B", "C"]
    mu = pd.Series([0.05, 0.10, 0.15], index=names)
    # C is the riskiest, A the safest; all uncorrelated.
    cov = pd.DataFrame(np.diag([0.01, 0.04, 0.09]), index=names, columns=names)
    return mu, cov


def test_weights_sum_to_one(mu_cov):
    mu, cov = mu_cov
    for fn in (equal_weight, min_variance, max_sharpe):
        assert fn(mu, cov, CFG).weights.sum() == pytest.approx(1.0)


def test_long_only_respects_bounds(mu_cov):
    mu, cov = mu_cov
    w = max_sharpe(mu, cov, Settings(max_weight=0.5, allow_short=False)).weights
    assert (w >= -1e-8).all() and (w <= 0.5 + 1e-8).all()


def test_min_variance_tilts_to_lowest_variance_asset(mu_cov):
    mu, cov = mu_cov
    w = min_variance(mu, cov, CFG).weights
    assert w.idxmax() == "A"


def test_min_variance_is_the_lowest_vol_portfolio(mu_cov):
    mu, cov = mu_cov
    floor = min_variance(mu, cov, CFG).volatility
    assert equal_weight(mu, cov, CFG).volatility >= floor - 1e-9
    assert max_sharpe(mu, cov, CFG).volatility >= floor - 1e-9


def test_max_sharpe_beats_equal_weight_ex_ante(mu_cov):
    mu, cov = mu_cov
    assert max_sharpe(mu, cov, CFG).sharpe >= equal_weight(mu, cov, CFG).sharpe


def test_frontier_is_monotone_in_return(mu_cov):
    mu, cov = mu_cov
    frontier = efficient_frontier(mu, cov, CFG, points=12)
    assert len(frontier) > 5
    assert frontier["expected_return"].is_monotonic_increasing


# --------------------------------------------------------------------------
# the benchmark
# --------------------------------------------------------------------------


def test_buy_and_hold_stats_match_hand_calculation():
    daily = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    p = buy_and_hold("SPY", daily, CFG)

    assert p.expected_return == pytest.approx(daily.mean() * TRADING_DAYS)
    assert p.volatility == pytest.approx(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))
    assert p.sharpe == pytest.approx(p.expected_return / p.volatility)


def test_buy_and_hold_is_fully_invested_in_one_asset():
    p = buy_and_hold("SPY", pd.Series([0.01, 0.02, -0.01]), CFG)
    assert p.weights.to_dict() == {"SPY": 1.0}
    assert p.weights.sum() == pytest.approx(1.0)


def test_buy_and_hold_is_flagged_as_a_benchmark():
    """The flag drives chart rendering, so pin it rather than the name."""
    assert buy_and_hold("SPY", pd.Series([0.01, 0.02]), CFG).is_benchmark is True
    mu = pd.Series([0.1], index=["SPY"])
    cov = pd.DataFrame([[0.04]], index=["SPY"], columns=["SPY"])
    assert equal_weight(mu, cov, CFG).is_benchmark is False


def test_buy_and_hold_agrees_with_single_asset_equal_weight():
    """Locks the two code paths together: sqrt(w @ cov @ w) vs std * sqrt(252).

    buy_and_hold derives its stats from the raw series while the strategies go
    through the covariance matrix. They must not be allowed to drift apart.
    """
    daily = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015], name="SPY")
    mu, cov = annualise(daily.to_frame())

    bench = buy_and_hold("SPY", daily, CFG)
    solo = equal_weight(mu, cov, CFG)

    assert bench.expected_return == pytest.approx(solo.expected_return)
    assert bench.volatility == pytest.approx(solo.volatility)
    assert bench.sharpe == pytest.approx(solo.sharpe)


def test_buy_and_hold_handles_a_flat_series():
    p = buy_and_hold("CASH", pd.Series([0.0] * 5), CFG)
    assert p.volatility == pytest.approx(0.0)
    assert p.sharpe == 0.0  # guarded, not a ZeroDivisionError
