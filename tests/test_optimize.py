"""Optimiser tests use synthetic covariance so they run offline and fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_opt.config import TRADING_DAYS, Settings
from portfolio_opt.optimize import (
    STRATEGIES,
    buy_and_hold,
    efficient_frontier,
    equal_weight,
    max_attainable_return,
    max_diversification,
    max_sharpe,
    min_variance,
    risk_contributions,
    risk_parity,
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


# --------------------------------------------------------------------------
# risk parity and max diversification (M3)
# --------------------------------------------------------------------------


def test_risk_parity_equalises_risk_contributions(mu_cov):
    """The defining property. Weights are unequal; risk shares are not."""
    mu, cov = mu_cov
    p = risk_parity(mu, cov, CFG)
    contributions = risk_contributions(p.weights, cov)

    assert contributions.sum() == pytest.approx(1.0)
    assert contributions.max() - contributions.min() < 1e-4
    assert contributions.iloc[0] == pytest.approx(1 / len(mu), abs=1e-4)
    # C has 9x the variance of A, so it must be held far more lightly.
    assert p.weights["A"] > p.weights["C"]


def test_risk_parity_is_long_only_and_fully_invested(mu_cov):
    mu, cov = mu_cov
    p = risk_parity(mu, cov, CFG)
    assert p.weights.sum() == pytest.approx(1.0)
    assert (p.weights > 0).all()


def test_risk_parity_warns_when_it_breaches_the_cap(mu_cov, caplog):
    """It cannot honour max_weight without breaking equal-risk, so it says so."""
    mu, cov = mu_cov
    with caplog.at_level("WARNING"):
        risk_parity(mu, cov, Settings(max_weight=0.20, risk_free_rate=0.0))
    assert "risk_parity ignores max_weight" in caplog.text


def test_max_diversification_needs_no_return_forecast(mu_cov):
    """Its weights must be invariant to mu -- it only reads the covariance."""
    mu, cov = mu_cov
    other = pd.Series([0.99, -0.5, 0.2], index=mu.index)

    first = max_diversification(mu, cov, CFG).weights
    second = max_diversification(other, cov, CFG).weights
    pd.testing.assert_series_equal(first, second, atol=1e-6)


def test_new_strategies_respect_bounds(mu_cov):
    mu, cov = mu_cov
    cfg = Settings(max_weight=0.5, risk_free_rate=0.0)
    w = max_diversification(mu, cov, cfg).weights
    assert (w >= -1e-8).all() and (w <= 0.5 + 1e-8).all()


def test_all_strategies_are_registered():
    assert set(STRATEGIES) == {
        "equal_weight",
        "min_variance",
        "max_sharpe",
        "risk_parity",
        "max_diversification",
    }


# --------------------------------------------------------------------------
# convexity and feasibility fixes (P1-3, P2-5, frontier bound)
# --------------------------------------------------------------------------


def test_max_sharpe_is_start_point_independent(mu_cov):
    """The convex reformulation returns the global optimum.

    The old SLSQP version started from 1/N and could only promise a local
    solution. Here, no long-only portfolio may beat the reported Sharpe.
    """
    mu, cov = mu_cov
    best = max_sharpe(mu, cov, CFG)

    rng = np.random.default_rng(11)
    for _ in range(400):
        candidate = rng.dirichlet(np.ones(len(mu)))
        ret = candidate @ mu.to_numpy()
        vol = np.sqrt(candidate @ cov.to_numpy() @ candidate)
        assert (ret - CFG.risk_free_rate) / vol <= best.sharpe + 1e-6


def test_max_sharpe_reports_an_undefined_tangency_portfolio(mu_cov):
    _, cov = mu_cov
    all_negative = pd.Series([-0.05, -0.10, -0.02], index=cov.index)
    with pytest.raises(RuntimeError, match="positive excess return"):
        max_sharpe(all_negative, cov, CFG)


def test_equal_weight_refuses_an_impossible_cap(mu_cov):
    """P2-5: 1/N used to silently breach max_weight instead of complaining."""
    mu, cov = mu_cov
    with pytest.raises(ValueError, match="cannot be equally weighted"):
        equal_weight(mu, cov, Settings(max_weight=0.20))


def test_max_attainable_return_respects_the_position_cap(mu_cov):
    """The cap makes the best single asset unreachable -- that was the old bug."""
    mu, cov = mu_cov
    capped = Settings(max_weight=0.5, risk_free_rate=0.0)
    attainable = max_attainable_return(mu, cov, capped)

    assert attainable < mu.max()  # 0.15 is out of reach at a 50% cap
    # Best case: 50% in C (0.15), 50% in B (0.10) = 0.125
    assert attainable == pytest.approx(0.125, abs=1e-6)


def test_frontier_spans_min_variance_to_max_attainable(mu_cov):
    mu, cov = mu_cov
    frontier = efficient_frontier(mu, cov, CFG, points=20)
    floor = min_variance(mu, cov, CFG)
    ceiling = max_attainable_return(mu, cov, CFG)

    assert frontier["expected_return"].min() == pytest.approx(
        floor.expected_return, abs=1e-6
    )
    assert frontier["expected_return"].max() == pytest.approx(ceiling, abs=1e-6)
    assert frontier["volatility"].min() == pytest.approx(floor.volatility, abs=1e-6)


def test_frontier_solves_every_target(mu_cov):
    """No silent skipping: all targets are feasible by construction now."""
    mu, cov = mu_cov
    assert len(efficient_frontier(mu, cov, CFG, points=25)) == 25
