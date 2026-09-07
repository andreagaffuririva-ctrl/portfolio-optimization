"""Portfolio construction: min-variance, max-Sharpe, equal weight.

Uses scipy SLSQP so the constraint set stays explicit and readable. Swap in
cvxpy at M3 when we add turnover/cardinality constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import TRADING_DAYS, Settings, settings


@dataclass(frozen=True)
class Portfolio:
    """A set of weights plus the ex-ante stats implied by (mu, sigma)."""

    name: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    # Dispatch flag, not decoration: report and pipeline branch on this rather
    # than on a "benchmark_" name prefix, so renaming can't break rendering.
    is_benchmark: bool = False

    def to_frame(self) -> pd.DataFrame:
        return (
            self.weights[self.weights.abs() > 1e-4]
            .sort_values(ascending=False)
            .rename("weight")
            .to_frame()
        )


def _stats(
    w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float
) -> tuple[float, float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _bounds(n: int, cfg: Settings) -> list[tuple[float, float]]:
    lower = -cfg.max_weight if cfg.allow_short else 0.0
    return [(lower, cfg.max_weight)] * n


def _solve(
    objective, mu: pd.Series, cov: pd.DataFrame, cfg: Settings, name: str
) -> Portfolio:
    n = len(mu)
    mu_v, cov_m = mu.to_numpy(), cov.to_numpy()
    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        args=(mu_v, cov_m, cfg.risk_free_rate),
        method="SLSQP",
        bounds=_bounds(n, cfg),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"{name} optimisation failed: {result.message}")

    weights = pd.Series(result.x, index=mu.index)
    ret, vol, sharpe = _stats(result.x, mu_v, cov_m, cfg.risk_free_rate)
    return Portfolio(name, weights, ret, vol, sharpe)


def min_variance(mu: pd.Series, cov: pd.DataFrame, cfg: Settings = settings) -> Portfolio:
    return _solve(lambda w, m, c, rf: w @ c @ w, mu, cov, cfg, "min_variance")


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, cfg: Settings = settings) -> Portfolio:
    return _solve(lambda w, m, c, rf: -_stats(w, m, c, rf)[2], mu, cov, cfg, "max_sharpe")


def equal_weight(mu: pd.Series, cov: pd.DataFrame, cfg: Settings = settings) -> Portfolio:
    n = len(mu)
    w = np.full(n, 1.0 / n)
    ret, vol, sharpe = _stats(w, mu.to_numpy(), cov.to_numpy(), cfg.risk_free_rate)
    return Portfolio("equal_weight", pd.Series(w, index=mu.index), ret, vol, sharpe)


def buy_and_hold(ticker: str, daily: pd.Series, cfg: Settings = settings) -> Portfolio:
    """The benchmark: 100% in one asset, held. Stats come from its own series.

    Kept out of STRATEGIES -- it is the thing the strategies are measured
    against, not a candidate itself.
    """
    ret = float(daily.mean() * TRADING_DAYS)
    vol = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (ret - cfg.risk_free_rate) / vol if vol > 0 else 0.0
    return Portfolio(
        name=f"benchmark_{ticker}",
        weights=pd.Series({ticker: 1.0}),
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        is_benchmark=True,
    )


STRATEGIES = {
    "equal_weight": equal_weight,
    "min_variance": min_variance,
    "max_sharpe": max_sharpe,
}


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, cfg: Settings = settings, points: int = 30
) -> pd.DataFrame:
    """Sweep target returns to trace the frontier, for the report chart."""
    n = len(mu)
    mu_v, cov_m = mu.to_numpy(), cov.to_numpy()
    lo, hi = min_variance(mu, cov, cfg).expected_return, mu.max() * 0.99
    rows = []
    for target in np.linspace(lo, hi, points):
        result = minimize(
            lambda w, c=cov_m: w @ c @ w,
            x0=np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=_bounds(n, cfg),
            constraints=[
                {"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "eq", "fun": lambda w, t=target: w @ mu_v - t},
            ],
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if result.success:
            ret, vol, sharpe = _stats(result.x, mu_v, cov_m, cfg.risk_free_rate)
            rows.append({"expected_return": ret, "volatility": vol, "sharpe": sharpe})
    return pd.DataFrame(rows)
