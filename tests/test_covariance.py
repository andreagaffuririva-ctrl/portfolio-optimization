"""Covariance estimator tests. Synthetic data, no network, no parquet."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_opt.config import TRADING_DAYS
from portfolio_opt.covariance import (
    estimate,
    ledoit_wolf_covariance,
    sample_covariance,
    shrinkage_intensity,
)


@pytest.fixture
def daily() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.multivariate_normal(
            mean=[0.0004, 0.0006, 0.0003],
            cov=np.array([[4e-4, 1e-4, 5e-5], [1e-4, 6e-4, 8e-5], [5e-5, 8e-5, 3e-4]]),
            size=500,
        ),
        columns=["AAA", "BBB", "CCC"],
    )


def test_estimate_annualises_by_trading_days(daily):
    mu, cov = estimate(daily, "sample")
    pd.testing.assert_series_equal(mu, daily.mean() * TRADING_DAYS)
    pd.testing.assert_frame_equal(cov, daily.cov() * TRADING_DAYS)


def test_ledoit_wolf_is_positive_semidefinite(daily):
    cov = ledoit_wolf_covariance(daily)
    assert (np.linalg.eigvalsh(cov.to_numpy()) >= -1e-12).all()
    assert list(cov.columns) == list(daily.columns)
    assert list(cov.index) == list(daily.columns)


def test_shrinkage_is_a_valid_weight(daily):
    assert 0.0 <= shrinkage_intensity(daily) <= 1.0


def test_shrinkage_grows_as_observations_get_scarce(daily):
    """The whole point: less data, more shrinkage toward the target."""
    plenty = shrinkage_intensity(daily)
    scarce = shrinkage_intensity(daily.head(20))
    assert scarce > plenty


def test_shrinkage_pulls_variances_toward_the_average(daily):
    """Shrinkage compresses the spread of variances -- that is the mechanism."""
    sample_spread = np.diag(sample_covariance(daily).to_numpy()).std()
    shrunk_spread = np.diag(ledoit_wolf_covariance(daily).to_numpy()).std()
    assert shrunk_spread < sample_spread


def test_unknown_estimator_is_rejected(daily):
    with pytest.raises(ValueError, match="unknown estimator"):
        estimate(daily, "magic")
