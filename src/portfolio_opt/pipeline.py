"""The end-to-end pipeline: ingest -> transform -> optimise -> backtest -> report.

Two passes over the same data, on purpose:

  * The **in-sample** pass fits every strategy on the whole history and draws the
    efficient frontier. It is an illustration of the geometry, nothing more.
  * The **walk-forward** pass is the one that counts. Weights come from a
    trailing window and are held into unseen data, net of trading costs.

Reporting both, side by side, is the point of the project: the gap between them
is the size of the lie that in-sample optimisation tells.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from .backtest import buy_and_hold_backtest, summary, walk_forward
from .config import Settings, settings
from .covariance import estimate, shrinkage_intensity
from .ingest import ingest
from .optimize import (
    STRATEGIES,
    Portfolio,
    buy_and_hold,
    efficient_frontier,
    risk_contributions,
)
from .report import (
    plot_drawdown,
    plot_equity_curve,
    plot_frontier,
    plot_sharpe_decay,
    plot_weights,
    write_backtest_table,
    write_table,
)
from .transform import build_gold, build_silver, returns_matrix

log = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "cagr",
    "ann_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "annual_turnover",
    "cost_drag_annual",
]


def in_sample_pass(
    daily: pd.DataFrame, benchmark: pd.Series, cfg: Settings
) -> list[Portfolio]:
    """Fit on everything and draw the frontier. Ex-ante figures only."""
    mu, cov = estimate(daily, cfg.covariance_estimator)
    log.info(
        "covariance: %s (shrinkage=%.3f on %s days x %s assets)",
        cfg.covariance_estimator,
        shrinkage_intensity(daily),
        f"{len(daily):,}",
        daily.shape[1],
    )

    portfolios = [fn(mu, cov, cfg) for fn in STRATEGIES.values()]
    portfolios.append(buy_and_hold(cfg.benchmark, benchmark, cfg))
    for p in portfolios:
        log.info(
            "  %-20s return %6.2f%%  vol %6.2f%%  sharpe %.2f",
            p.name,
            p.expected_return * 100,
            p.volatility * 100,
            p.sharpe,
        )

    contributions = risk_contributions(portfolios[3].weights, cov)
    log.info(
        "  risk_parity risk contributions: %.4f to %.4f (target %.4f)",
        contributions.min(),
        contributions.max(),
        1 / len(contributions),
    )

    plot_frontier(efficient_frontier(mu, cov, cfg), portfolios)
    for p in portfolios:
        if not p.is_benchmark:
            plot_weights(p)  # a single 100% bar tells you nothing
    return portfolios


def walk_forward_pass(daily: pd.DataFrame, benchmark: pd.Series, cfg: Settings):
    """The honest pass. Returns the list of BacktestResults."""
    log.info(
        "walk-forward: %s-day window, %s rebalance, %.0fbps cost",
        cfg.estimation_window,
        cfg.rebalance,
        cfg.transaction_cost_bps,
    )
    results = [walk_forward(daily, fn, name, cfg) for name, fn in STRATEGIES.items()]
    results.append(
        buy_and_hold_backtest(cfg.benchmark, benchmark, results[0].net_returns.index, cfg)
    )

    plot_equity_curve(results)
    plot_drawdown(results)
    return results


def run(
    cfg: Settings = settings,
    skip_ingest: bool = False,
    skip_backtest: bool = False,
) -> pd.DataFrame:
    if skip_ingest:
        log.info("skipping ingest, reusing data/bronze")
    else:
        ingest(cfg)

    build_silver()
    build_gold()

    # Pull the benchmark in the same call so both share one aligned calendar.
    universe = tuple(dict.fromkeys([*cfg.tickers, cfg.benchmark]))
    aligned = returns_matrix(universe)
    daily = aligned[list(cfg.tickers)]
    benchmark = aligned[cfg.benchmark]

    log.info("=== in-sample (illustrative only) ===")
    portfolios = in_sample_pass(daily, benchmark, cfg)
    write_table(portfolios)

    if skip_backtest:
        log.info("skipping walk-forward backtest")
        return pd.DataFrame({p.name: {"sharpe": p.sharpe} for p in portfolios}).T

    log.info("=== walk-forward (out-of-sample) ===")
    results = walk_forward_pass(daily, benchmark, cfg)
    table = write_backtest_table(results)

    plot_sharpe_decay(
        in_sample={p.name: p.sharpe for p in portfolios},
        out_of_sample={r.name: r.metrics["sharpe"] for r in results},
    )

    log.info("out-of-sample results (%.1f years):", table["years"].iloc[0])
    for name, row in table.iterrows():
        log.info(
            "  %-20s cagr %6.2f%%  vol %6.2f%%  sharpe %5.2f  maxdd %6.1f%%  "
            "turnover %4.0f%%/yr",
            name,
            row["cagr"] * 100,
            row["ann_volatility"] * 100,
            row["sharpe"],
            row["max_drawdown"] * 100,
            row["annual_turnover"] * 100,
        )

    best = table.index[0]
    log.info(
        "best out-of-sample Sharpe: %s (%.2f); in-sample it ranked %s",
        best,
        table["sharpe"].iloc[0],
        1
        + sorted((p.sharpe for p in portfolios), reverse=True).index(
            next(p.sharpe for p in portfolios if p.name == best)
        ),
    )
    return summary(results)[METRIC_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the portfolio pipeline")
    parser.add_argument(
        "--skip-ingest", action="store_true", help="reuse existing bronze data"
    )
    parser.add_argument(
        "--skip-backtest", action="store_true", help="in-sample pass only"
    )
    parser.add_argument("--start", default=None, help="override history start date")
    parser.add_argument(
        "--estimator",
        default=None,
        choices=["sample", "ledoit_wolf"],
        help="covariance estimator",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    overrides = {}
    if args.start:
        overrides["start"] = args.start
    if args.estimator:
        overrides["covariance_estimator"] = args.estimator
    cfg = Settings(**overrides) if overrides else settings

    run(cfg, skip_ingest=args.skip_ingest, skip_backtest=args.skip_backtest)


if __name__ == "__main__":
    main()
