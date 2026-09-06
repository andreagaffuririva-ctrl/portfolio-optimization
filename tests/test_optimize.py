"""Optimiser tests use synthetic covariance so they run offline and fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_opt.config import Settings
from portfolio_opt.optimize import (
    efficient_frontier,
    equal_weight,
    max_sharpe,
    min_variance,
)

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
