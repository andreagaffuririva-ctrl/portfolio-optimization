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
    """Daily returns per ticker -> data/gold/returns.parquet.

    Both conventions are emitted on purpose. Log returns are additive across
    *time* but not across *assets* -- ln(sum w_i exp(r_i)) != sum w_i r_i -- so
    mean-variance optimisation must use simple returns, while cumulative equity
    curves are cleaner in logs.
    """
    ensure_dirs()
    con = duckdb.connect()
    frame = con.execute(
        f"""
        WITH lagged AS (
            SELECT date, ticker, close,
                   lag(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close
            FROM read_parquet('{SILVER_DIR / "prices.parquet"}')
        )
        SELECT date, ticker, close,
               close / prev_close - 1        AS simple_return,
               ln(close / prev_close)        AS log_return
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


def returns_matrix(
    tickers: tuple[str, ...] | None = None, value: str = "simple_return"
) -> pd.DataFrame:
    """Wide date x ticker return matrix, aligned on dates every asset shares.

    The inner join is what mean-variance needs, but it is also a trap: one
    late-listing asset silently truncates every other column. Any date loss is
    logged with the ticker responsible -- see PROJECT_LOG.md P1-1.
    """
    frame = pd.read_parquet(GOLD_DIR / "returns.parquet")
    if tickers:
        frame = frame[frame["ticker"].isin(tickers)]
    wide = frame.pivot(index="date", columns="ticker", values=value)

    aligned = wide.dropna(how="any")
    if aligned.empty:
        raise RuntimeError("no dates are common to every requested ticker")

    _log_alignment(wide, aligned)
    return aligned


def _log_alignment(wide: pd.DataFrame, aligned: pd.DataFrame) -> None:
    """Report the effective window and attribute any date loss to its cause.

    Attribution is counted, not inferred. A ticker can cost dates three ways --
    listing late, delisting early, or a hole in the middle -- so guessing from
    first-valid-index alone names the wrong ticker whenever the loss came from a
    gap. `missing` counts the dropped dates each ticker is actually absent for,
    and `sole_cause` isolates the dates only that one ticker is missing, which is
    what you would recover by removing it.
    """
    dropped = len(wide) - len(aligned)
    log.info(
        "returns window %s -> %s (%s rows x %s assets)",
        aligned.index.min().date(),
        aligned.index.max().date(),
        f"{len(aligned):,}",
        aligned.shape[1],
    )
    if not dropped:
        return

    holes = wide.isna()
    lost = holes.loc[holes.any(axis=1)]
    missing = lost.sum().sort_values(ascending=False)
    sole_cause = lost.loc[lost.sum(axis=1) == 1].sum()

    log.warning(
        "alignment dropped %s of %s dates (%.1f%%). Dates missing per ticker: %s. "
        "Recoverable by dropping that ticker alone: %s. First/last data per ticker: %s",
        f"{dropped:,}",
        f"{len(wide):,}",
        100 * dropped / len(wide),
        {t: int(n) for t, n in missing.items() if n},
        {t: int(n) for t, n in sole_cause.sort_values(ascending=False).items() if n},
        {
            t: (
                wide[t].first_valid_index().date(),
                wide[t].last_valid_index().date(),
            )
            for t in wide.columns
        },
    )


def annualise(daily: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised mean return vector and covariance matrix from daily returns.

    Expects *simple* returns -- see build_gold on why log returns are wrong here.
    """
    return daily.mean() * TRADING_DAYS, daily.cov() * TRADING_DAYS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_silver()
    build_gold()
