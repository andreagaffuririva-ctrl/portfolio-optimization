"""Ingest tests stub the price source, so no network call is made in CI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_opt.config import Settings
from portfolio_opt.ingest import (
    BRONZE_COLUMNS,
    TransientSourceError,
    YahooSource,
    _to_tidy,
    ingest,
)


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


# --------------------------------------------------------------------------
# retry behaviour (P1-4) -- the backoff is patched to zero to keep tests fast
# --------------------------------------------------------------------------


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr("portfolio_opt.ingest.wait_exponential", lambda **_: 0)


class FlakyYahoo:
    """Stands in for yf.download: fails `failures` times, then succeeds."""

    def __init__(self, frame: pd.DataFrame, failures: int, exc: Exception) -> None:
        self.frame, self.failures, self.exc = frame, failures, exc
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return self.frame


def test_transient_failures_are_retried_then_succeed(monkeypatch, no_backoff):
    tickers = ("XLK", "XLF")
    flaky = FlakyYahoo(_fake_yahoo_frame(tickers), failures=2, exc=OSError("rate limit"))
    monkeypatch.setattr("portfolio_opt.ingest.yf.download", flaky)

    frame = YahooSource(attempts=4).fetch(tickers, "2024-01-01", None)
    assert flaky.calls == 3
    assert set(frame["ticker"]) == set(tickers)


def test_retries_are_bounded_and_reraise(monkeypatch, no_backoff):
    flaky = FlakyYahoo(pd.DataFrame(), failures=99, exc=OSError("down"))
    monkeypatch.setattr("portfolio_opt.ingest.yf.download", flaky)

    with pytest.raises(TransientSourceError, match="Yahoo download failed"):
        YahooSource(attempts=3).fetch(("XLK",), "2024-01-01", None)
    assert flaky.calls == 3


def test_empty_response_is_treated_as_transient(monkeypatch, no_backoff):
    monkeypatch.setattr(
        "portfolio_opt.ingest.yf.download", lambda *a, **k: pd.DataFrame()
    )
    with pytest.raises(TransientSourceError, match="returned no data"):
        YahooSource(attempts=2).fetch(("XLK",), "2024-01-01", None)


def test_schema_break_is_not_retried(monkeypatch, no_backoff):
    """A missing column is a real bug -- fail once, do not hammer the source."""
    tickers = ("XLK",)
    frame = _fake_yahoo_frame(tickers).drop(columns=[("Close", "XLK")])
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return frame

    monkeypatch.setattr("portfolio_opt.ingest.yf.download", counting)
    with pytest.raises(RuntimeError, match="missing columns"):
        YahooSource(attempts=4).fetch(tickers, "2024-01-01", None)
    assert calls["n"] == 1
