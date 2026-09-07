"""Portfolio construction as convex programs.

Every objective here is expressed so that a solver returns *the* optimum rather
than a local one. That mattered most for max-Sharpe: maximising a ratio of a
linear to a quadratic form is not convex, so the previous SLSQP version only
found a local solution from its start point. The standard change of variables
(Cornuejols & Tutuncu) turns it into a QP:

    maximise  (mu - rf)'w / sqrt(w' S w)      subject to  1'w = 1

    substitute  y = w / ((mu - rf)'w)   =>    minimise  y' S y
                                              subject to  (mu - rf)'y = 1
    then recover  w = y / 1'y

The same trick handles maximum diversification with the volatility vector in
place of excess returns. Risk parity uses the Spinu (2013) convex form.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import TRADING_DAYS, Settings, settings

log = logging.getLogger(__name__)

_GOOD = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


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
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _psd(cov: pd.DataFrame) -> np.ndarray:
    """Symmetrise and clip tiny negative eigenvalues.

    Shrunk estimators are PSD by construction, but floating-point asymmetry is
    enough to make cvxpy reject the quadratic form outright.
    """
    matrix = cov.to_numpy()
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if (eigenvalues < 0).any():
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        matrix = (matrix + matrix.T) / 2
    return matrix


def _solve(problem: cp.Problem, name: str) -> None:
    problem.solve()
    if problem.status not in _GOOD:
        raise RuntimeError(f"{name} optimisation failed: status={problem.status}")


def _weight_constraints(w: cp.Variable, n: int, cfg: Settings) -> list[cp.Constraint]:
    """Fully invested, plus the long-only / concentration bounds."""
    constraints = [cp.sum(w) == 1, w <= cfg.max_weight]
    constraints.append(w >= (-cfg.max_weight if cfg.allow_short else 0.0))
    return constraints


def _finish(
    name: str, weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, cfg: Settings
) -> Portfolio:
    weights = np.asarray(weights, dtype=float)
    weights[np.abs(weights) < 1e-9] = 0.0  # scrub solver dust
    ret, vol, sharpe = _stats(weights, mu.to_numpy(), _psd(cov), cfg.risk_free_rate)
    return Portfolio(name, pd.Series(weights, index=mu.index), ret, vol, sharpe)


# --------------------------------------------------------------------------
# strategies
# --------------------------------------------------------------------------


def equal_weight(mu: pd.Series, cov: pd.DataFrame, cfg: Settings = settings) -> Portfolio:
    """1/N. Deceptively hard to beat out of sample -- the honest baseline."""
    n = len(mu)
    if 1.0 / n > cfg.max_weight + 1e-12:
        raise ValueError(
            f"equal weight needs {1 / n:.1%} per asset but max_weight is "
            f"{cfg.max_weight:.1%}; {n} assets cannot be equally weighted"
        )
    return _finish("equal_weight", np.full(n, 1.0 / n), mu, cov, cfg)


def min_variance(mu: pd.Series, cov: pd.DataFrame, cfg: Settings = settings) -> Portfolio:
    """Lowest achievable variance. Ignores mu entirely -- that is the point."""
    n = len(mu)
    w = cp.Variable(n)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(w, _psd(cov))), _weight_constraints(w, n, cfg)
    )
    _solve(problem, "min_variance")
    return _finish("min_variance", w.value, mu, cov, cfg)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, cfg: Settings = settings) -> Portfolio:
    """Tangency portfolio via the convex reformulation (see module docstring)."""
    excess = mu.to_numpy() - cfg.risk_free_rate
    if excess.max() <= 0:
        raise RuntimeError(
            "no asset has a positive excess return; the tangency portfolio is "
            "undefined for this window"
        )
    return _finish(
        "max_sharpe", _ratio_program(excess, cov, cfg, "max_sharpe"), mu, cov, cfg
    )


def max_diversification(
    mu: pd.Series, cov: pd.DataFrame, cfg: Settings = settings
) -> Portfolio:
    """Maximise the diversification ratio w'sigma / sqrt(w'Sw).

    Same program as max-Sharpe with volatilities standing in for excess
    returns, so it needs no return forecast at all.
    """
    vols = np.sqrt(np.diag(_psd(cov)))
    return _finish(
        "max_diversification",
        _ratio_program(vols, cov, cfg, "max_diversification"),
        mu,
        cov,
        cfg,
    )


def _ratio_program(
    numerator: np.ndarray, cov: pd.DataFrame, cfg: Settings, name: str
) -> np.ndarray:
    """Minimise y'Sy s.t. numerator'y = 1, then renormalise to w."""
    n = len(numerator)
    y = cp.Variable(n)
    constraints = [numerator @ y == 1]
    # The position cap is homogeneous in y, so it stays linear: w_i <= cap
    # becomes y_i <= cap * 1'y.
    constraints.append(y <= cfg.max_weight * cp.sum(y))
    constraints.append(y >= (-cfg.max_weight * cp.sum(y) if cfg.allow_short else 0.0))
    problem = cp.Problem(cp.Minimize(cp.quad_form(y, _psd(cov))), constraints)
    _solve(problem, name)
    total = float(np.sum(y.value))
    if abs(total) < 1e-12:
        raise RuntimeError(f"{name} produced a degenerate solution")
    return y.value / total


def risk_parity(mu: pd.Series, cov: pd.DataFrame, cfg: Settings = settings) -> Portfolio:
    """Equal risk contribution, via Spinu's convex formulation.

    Minimising 0.5 w'Sw - (1/n) sum(log w) has its optimum where every asset
    contributes the same share of total risk. Long-only by construction, since
    log requires w > 0.

    Note: the concentration cap is *not* enforced. Capping would break the
    equal-risk property outright, so a breach is logged instead of silently
    producing something that is neither risk parity nor within the cap.
    """
    n = len(mu)
    w = cp.Variable(n, pos=True)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(w, _psd(cov)) - cp.sum(cp.log(w)) / n)
    )
    _solve(problem, "risk_parity")
    weights = np.asarray(w.value, dtype=float)
    weights = weights / weights.sum()

    if weights.max() > cfg.max_weight + 1e-6:
        log.warning(
            "risk_parity ignores max_weight: %s at %.1f%% exceeds the %.1f%% cap",
            mu.index[int(weights.argmax())],
            100 * weights.max(),
            100 * cfg.max_weight,
        )
    return _finish("risk_parity", weights, mu, cov, cfg)


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
    "risk_parity": risk_parity,
    "max_diversification": max_diversification,
}


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Each asset's share of total portfolio variance. Sums to 1.

    Used to demonstrate that risk_parity does what its name claims.
    """
    w = weights.to_numpy()
    matrix = _psd(cov)
    total = float(w @ matrix @ w)
    if total <= 0:
        return pd.Series(0.0, index=weights.index)
    return pd.Series(w * (matrix @ w) / total, index=weights.index)


# --------------------------------------------------------------------------
# frontier
# --------------------------------------------------------------------------


def max_attainable_return(
    mu: pd.Series, cov: pd.DataFrame, cfg: Settings = settings
) -> float:
    """Highest return reachable *under the constraints*.

    The previous version swept up to mu.max() * 0.99, which the concentration
    cap makes infeasible -- those solves failed and were silently skipped, so
    the frontier's top end was decided by solver failure rather than by design.
    """
    n = len(mu)
    w = cp.Variable(n)
    problem = cp.Problem(cp.Maximize(mu.to_numpy() @ w), _weight_constraints(w, n, cfg))
    _solve(problem, "max_attainable_return")
    return float(mu.to_numpy() @ w.value)


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, cfg: Settings = settings, points: int = 40
) -> pd.DataFrame:
    """Trace the frontier between the min-variance and max-return endpoints."""
    n = len(mu)
    matrix = _psd(cov)
    lo = min_variance(mu, cov, cfg).expected_return
    hi = max_attainable_return(mu, cov, cfg)

    w = cp.Variable(n)
    target = cp.Parameter()
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(w, matrix)),
        [*_weight_constraints(w, n, cfg), mu.to_numpy() @ w == target],
    )

    rows = []
    for value in np.linspace(lo, hi, points):
        target.value = value
        problem.solve()
        if problem.status in _GOOD:
            ret, vol, sharpe = _stats(w.value, mu.to_numpy(), matrix, cfg.risk_free_rate)
            rows.append({"expected_return": ret, "volatility": vol, "sharpe": sharpe})
    if len(rows) < points // 2:
        log.warning("frontier solved only %s of %s targets", len(rows), points)
    return pd.DataFrame(rows)
