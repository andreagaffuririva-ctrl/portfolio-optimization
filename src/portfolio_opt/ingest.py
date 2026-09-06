"""Bronze layer: pull raw daily bars from Yahoo Finance and land them as Parquet.

Design notes for the didactic version:
  * The source is behind a small Protocol so Yahoo can be swapped out later.
  * Bronze is append-only and idempotent: re-running the same window overwrites
    the same partition instead of duplicating rows.
  * No cleaning happens here. Raw stays raw.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Protocol

import pandas as pd
import yfinance as yf

from .config import BRONZE_DIR, Settings, ensure_dirs, settings

log = logging.getLogger(__name__)

BRONZE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


class PriceSource(Protocol):
    """Anything that can return tidy OHLCV bars for a set of tickers."""

    def fetch(
        self, tickers: tuple[str, ...], start: str, end: str | None
    ) -> pd.DataFrame: ...


class YahooSource:
    """yfinance-backed source.

    Yahoo has no official public API; yfinance scrapes it. Expect occasional
    breakage and rate limiting -- that is why downloads are batched and retried.
    """

    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    def fetch(
        self, tickers: tuple[str, ...], start: str, end: str | None
    ) -> pd.DataFrame:
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=self.auto_adjust,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"Yahoo returned no data for {tickers}")
        return _to_tidy(raw, tickers)


def _to_tidy(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Flatten yfinance's (field, ticker) column MultiIndex into long format."""
    if isinstance(raw.columns, pd.MultiIndex):
        frame = raw.stack(level=-1, future_stack=True).reset_index()
        frame = frame.rename(
            columns={frame.columns[0]: "date", frame.columns[1]: "ticker"}
        )
    else:  # single ticker -> flat columns
        frame = raw.reset_index().rename(columns={raw.index.name or "Date": "date"})
        frame["ticker"] = tickers[0]

    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    missing = [c for c in BRONZE_COLUMNS if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Yahoo response missing columns: {missing}")
    return (
        frame[BRONZE_COLUMNS]
        .dropna(subset=["close"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def ingest(cfg: Settings = settings, source: PriceSource | None = None) -> pd.DataFrame:
    """Fetch the configured universe + benchmark and write data/bronze/prices.parquet."""
    ensure_dirs()
    source = source or YahooSource()
    universe = tuple(dict.fromkeys([*cfg.tickers, cfg.benchmark]))
    end = cfg.end or date.today().isoformat()

    log.info("Fetching %d tickers from %s to %s", len(universe), cfg.start, end)
    frame = source.fetch(universe, cfg.start, end)

    out = BRONZE_DIR / "prices.parquet"
    frame.to_parquet(out, index=False)
    log.info("Wrote %s rows to %s", f"{len(frame):,}", out)
    return frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest()
