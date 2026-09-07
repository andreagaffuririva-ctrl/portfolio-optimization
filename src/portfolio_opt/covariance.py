"""Covariance estimators.

The sample covariance is unbiased but noisy: with N assets you estimate
N(N+1)/2 parameters, and mean-variance optimisation is pathologically sensitive
to the error. Min-variance in particular loads onto whichever asset's variance
happens to be *under*-estimated, which is exactly the wrong bet.

Ledoit-Wolf shrinks the sample matrix toward a structured target, trading a
little bias for a large variance reduction. The shrinkage intensity is chosen
analytically, not tuned -- so there is no hyperparameter to leak future
information through.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import TRADING_DAYS

log = logging.getLogger(__name__)

Estimator = Literal["sample", "ledoit_wolf"]


def sample_covariance(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.cov()


def ledoit_wolf_covariance(daily: pd.DataFrame) -> pd.DataFrame:
    """Analytically-shrunk covariance. Returns the daily (not annualised) matrix."""
    estimator = LedoitWolf(assume_centered=False).fit(daily.to_numpy())
    log.debug("Ledoit-Wolf shrinkage=%.3f", estimator.shrinkage_)
    return pd.DataFrame(estimator.covariance_, index=daily.columns, columns=daily.columns)


ESTIMATORS = {
    "sample": sample_covariance,
    "ledoit_wolf": ledoit_wolf_covariance,
}


def estimate(
    daily: pd.DataFrame, estimator: Estimator = "ledoit_wolf"
) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised (mu, cov) from daily *simple* returns.

    Only the covariance is shrunk. The mean is left alone here -- shrinking it
    (Black-Litterman, James-Stein) is a separate decision, deliberately not
    bundled into this one.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; try {list(ESTIMATORS)}")
    cov = ESTIMATORS[estimator](daily) * TRADING_DAYS
    return daily.mean() * TRADING_DAYS, cov


def shrinkage_intensity(daily: pd.DataFrame) -> float:
    """The chosen shrinkage weight, for diagnostics and logging."""
    return float(LedoitWolf(assume_centered=False).fit(daily.to_numpy()).shrinkage_)
