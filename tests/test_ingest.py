"""Ingest tests stub the price source, so no network call is made in CI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_opt.config import Settings
from portfolio_opt.ingest import BRONZE_COLUMNS, _to_tidy, ingest


def _fake_yahoo_frame(tickers: tuple[str, ...], rows: int = 30) -> pd.DataFrame:
    """Mimic yfinance's (field, ticker) MultiIndex column layout."""
    dates = pd.bdate_range("2024-01-01", periods=rows, name="Date")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, list(tickers)])
    rng = np.random.default_rng(0)
    data = rng.uniform(50, 150, size=(rows, len(columns)))
    return pd.DataFrame(data, index=dates, columns=columns)


class StubSource:
    def __init__(self, frame: pd.DataFrame, tickers: tuple[str, ...]) -> None:
        self.frame, self.tickers = frame, tickers
        self.calls: list[tuple] = []

    def fetch(self, tickers, start, end):
        self.calls.append((tickers, start, end))
        return _to_tidy(self.frame, tickers)


@pytest.fixture
def stub(monkeypatch, tmp_path):
    monkeypatch.setattr("portfolio_opt.ingest.BRONZE_DIR", tmp_path)
    tickers = ("XLK", "XLF", "SPY")
    return StubSource(_fake_yahoo_frame(tickers), tickers)


def test_tidy_shape_and_columns(stub):
    frame = stub.fetch(("XLK", "XLF", "SPY"), "2024-01-01", None)
    assert list(frame.columns) == BRONZE_COLUMNS
    assert set(frame["ticker"]) == {"XLK", "XLF", "SPY"}
    assert frame["date"].dt.tz is None


def test_ingest_includes_benchmark_once(stub, tmp_path):
    cfg = Settings(tickers=("XLK", "SPY"), benchmark="SPY")
    ingest(cfg, source=stub)
    requested = stub.calls[0][0]
    assert requested.count("SPY") == 1
    assert (tmp_path / "prices.parquet").exists()


def test_ingest_is_idempotent(stub, tmp_path):
    cfg = Settings(tickers=("XLK", "XLF"), benchmark="SPY")
    first = ingest(cfg, source=stub)
    second = ingest(cfg, source=stub)
    pd.testing.assert_frame_equal(first, second)
    assert len(pd.read_parquet(tmp_path / "prices.parquet")) == len(first)
