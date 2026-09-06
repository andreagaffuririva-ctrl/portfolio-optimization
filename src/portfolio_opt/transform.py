"""Silver/gold layers: validate bronze, then derive returns and covariance inputs.

DuckDB does the heavy lifting so the SQL skills transfer to a real warehouse.
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from .config import BRONZE_DIR, GOLD_DIR, SILVER_DIR, TRADING_DAYS, ensure_dirs

log = logging.getLogger(__name__)


class PriceSchema(pa.DataFrameModel):
    """Contract for the silver price table. Failures here stop the pipeline."""

    date: Series[pd.Timestamp]
    ticker: Series[str] = pa.Field(str_matches=r"^[A-Z]{1,5}$")
    close: Series[float] = pa.Field(gt=0)
    volume: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = False
        unique = ["date", "ticker"]


def build_silver() -> pd.DataFrame:
    """Clean + validate bronze into data/silver/prices.parquet."""
    ensure_dirs()
    con = duckdb.connect()
    frame = con.execute(
        f"""
        SELECT date, ticker, close, CAST(volume AS DOUBLE) AS volume
        FROM read_parquet('{BRONZE_DIR / "prices.parquet"}')
        WHERE close IS NOT NULL AND close > 0
        QUALIFY row_number() OVER (PARTITION BY date, ticker ORDER BY date) = 1
        ORDER BY ticker, date
        """
    ).df()
    con.close()

    frame = PriceSchema.validate(frame)
    out = SILVER_DIR / "prices.parquet"
    frame.to_parquet(out, index=False)
    log.info("silver: %s rows -> %s", f"{len(frame):,}", out)
    return frame


def build_gold() -> pd.DataFrame:
    """Daily log returns per ticker -> data/gold/returns.parquet."""
    ensure_dirs()
    con = duckdb.connect()
    frame = con.execute(
        f"""
        WITH lagged AS (
            SELECT date, ticker, close,
                   lag(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close
            FROM read_parquet('{SILVER_DIR / "prices.parquet"}')
        )
        SELECT date, ticker, close, ln(close / prev_close) AS log_return
        FROM lagged
        WHERE prev_close IS NOT NULL
        ORDER BY ticker, date
        """
    ).df()
    con.close()

    out = GOLD_DIR / "returns.parquet"
    frame.to_parquet(out, index=False)
    log.info("gold: %s return rows -> %s", f"{len(frame):,}", out)
    return frame


def returns_matrix(tickers: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Wide date x ticker matrix of simple returns, aligned on common dates."""
    frame = pd.read_parquet(GOLD_DIR / "returns.parquet")
    if tickers:
        frame = frame[frame["ticker"].isin(tickers)]
    wide = frame.pivot(index="date", columns="ticker", values="log_return")
    return wide.dropna(how="any")


def annualise(daily: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised mean return vector and covariance matrix from daily returns."""
    return daily.mean() * TRADING_DAYS, daily.cov() * TRADING_DAYS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_silver()
    build_gold()
