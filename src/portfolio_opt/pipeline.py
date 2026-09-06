"""The end-to-end thin slice: ingest -> transform -> optimise -> report."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from .config import Settings, settings
from .ingest import ingest
from .optimize import STRATEGIES, Portfolio, efficient_frontier
from .report import plot_frontier, plot_weights, write_table
from .transform import annualise, build_gold, build_silver, returns_matrix

log = logging.getLogger(__name__)


def run(cfg: Settings = settings, skip_ingest: bool = False) -> pd.DataFrame:
    if skip_ingest:
        log.info("skipping ingest, reusing data/bronze")
    else:
        ingest(cfg)

    build_silver()
    build_gold()

    daily = returns_matrix(cfg.tickers)
    log.info("returns matrix: %s days x %s assets", *daily.shape)
    mu, cov = annualise(daily)

    portfolios: list[Portfolio] = [fn(mu, cov, cfg) for fn in STRATEGIES.values()]
    for p in portfolios:
        log.info(
            "%-13s return %6.2f%%  vol %6.2f%%  sharpe %.2f",
            p.name,
            p.expected_return * 100,
            p.volatility * 100,
            p.sharpe,
        )

    plot_frontier(efficient_frontier(mu, cov, cfg), portfolios)
    for p in portfolios:
        plot_weights(p)
    return write_table(portfolios)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the portfolio pipeline")
    parser.add_argument(
        "--skip-ingest", action="store_true", help="reuse existing bronze data"
    )
    parser.add_argument("--start", default=None, help="override history start date")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = settings if args.start is None else Settings(start=args.start)
    run(cfg, skip_ingest=args.skip_ingest)


if __name__ == "__main__":
    main()
