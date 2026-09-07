"""Transform tests run on a synthetic Parquet lake in tmp_path -- no network.

These cover the two defects logged as P1-1 (silent history truncation) and
P1-2 (log vs simple returns), so neither can regress unnoticed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from portfolio_opt.config import TRADING_DAYS
from portfolio_opt.transform import (
    PriceSchema,
    annualise,
    build_gold,
    build_silver,
    returns_matrix,
)


def _prices(**series: list[float]) -> pd.DataFrame:
    """Long-format price frame; each kwarg is a ticker and its close series."""
    rows = []
    for ticker, closes in series.items():
        dates = pd.bdate_range("2024-01-01", periods=len(closes))
        for day, close in zip(dates, closes, strict=True):
            if close is not None:
                rows.append(
                    {"date": day, "ticker": ticker, "close": close, "volume": 1_000.0}
                )
    return pd.DataFrame(rows)


@pytest.fixture
def lake(monkeypatch, tmp_path):
    """Point every layer at tmp_path and return a writer for bronze."""
    for layer in ("BRONZE_DIR", "SILVER_DIR", "GOLD_DIR"):
        monkeypatch.setattr(f"portfolio_opt.transform.{layer}", tmp_path)

    def write(frame: pd.DataFrame) -> None:
        frame.to_parquet(tmp_path / "prices.parquet", index=False)

    return write


# --------------------------------------------------------------------------
# return maths (P1-2)
# --------------------------------------------------------------------------


def test_simple_and_log_returns_both_emitted_and_correct(lake):
    lake(_prices(AAA=[100.0, 110.0, 99.0]))
    build_silver()
    gold = build_gold()

    assert {"simple_return", "log_return"} <= set(gold.columns)
    assert gold["simple_return"].tolist() == pytest.approx([0.10, -0.10])
    assert gold["log_return"].tolist() == pytest.approx([np.log(1.10), np.log(0.90)])


def test_simple_and_log_returns_differ(lake):
    """If these ever coincide the fixture is too tame to catch a mix-up."""
    lake(_prices(AAA=[100.0, 130.0, 80.0]))
    build_silver()
    gold = build_gold()
    assert not np.allclose(gold["simple_return"], gold["log_return"])


def test_gold_loses_exactly_one_row_per_ticker(lake):
    lake(_prices(AAA=[10.0, 11.0, 12.0], BBB=[20.0, 21.0, 22.0]))
    silver = build_silver()
    gold = build_gold()
    assert len(gold) == len(silver) - 2  # the first day has no prior close


def test_annualise_scales_by_trading_days(lake):
    daily = pd.DataFrame({"AAA": [0.01] * 10, "BBB": [0.02] * 10})
    mu, cov = annualise(daily)
    assert mu["AAA"] == pytest.approx(0.01 * TRADING_DAYS)
    assert cov.loc["AAA", "AAA"] == pytest.approx(0.0)  # constant series
    assert cov.shape == (2, 2)


# --------------------------------------------------------------------------
# alignment (P1-1)
# --------------------------------------------------------------------------


def test_late_listing_asset_truncates_the_whole_matrix(lake, caplog):
    """The P1-1 regression guard: BBB starts late, so AAA loses those dates."""
    lake(_prices(AAA=[10.0, 11.0, 12.0, 13.0, 14.0], BBB=[None, None, 20.0, 21.0, 22.0]))
    build_silver()
    build_gold()

    with caplog.at_level("WARNING"):
        wide = returns_matrix(("AAA", "BBB"))

    assert list(wide.columns) == ["AAA", "BBB"]
    assert len(wide) == 2  # only the dates BBB also has a return for
    assert "alignment dropped" in caplog.text
    assert "BBB" in caplog.text  # the binding ticker is named


def test_no_warning_when_history_is_complete(lake, caplog):
    lake(_prices(AAA=[10.0, 11.0, 12.0], BBB=[20.0, 21.0, 22.0]))
    build_silver()
    build_gold()

    with caplog.at_level("WARNING"):
        wide = returns_matrix(("AAA", "BBB"))

    assert len(wide) == 2
    assert "alignment dropped" not in caplog.text


def test_ticker_filter_excludes_others(lake):
    lake(_prices(AAA=[10.0, 11.0], BBB=[20.0, 21.0]))
    build_silver()
    build_gold()
    assert list(returns_matrix(("AAA",)).columns) == ["AAA"]


def test_disjoint_history_raises_rather_than_returning_empty(lake):
    lake(_prices(AAA=[10.0, 11.0, None, None], BBB=[None, None, 20.0, 21.0]))
    build_silver()
    build_gold()
    with pytest.raises(RuntimeError, match="no dates are common"):
        returns_matrix(("AAA", "BBB"))


# --------------------------------------------------------------------------
# the schema gate
# --------------------------------------------------------------------------


def test_schema_rejects_non_positive_close():
    bad = _prices(AAA=[10.0, 11.0])
    bad.loc[1, "close"] = -1.0
    with pytest.raises(pandera.errors.SchemaError):
        PriceSchema.validate(bad)


def test_schema_rejects_duplicate_date_ticker():
    dupe = pd.concat([_prices(AAA=[10.0])] * 2, ignore_index=True)
    with pytest.raises(pandera.errors.SchemaError):
        PriceSchema.validate(dupe)


def test_silver_dedupes_before_validating(lake):
    """Bronze may contain repeats; silver's QUALIFY must collapse them."""
    doubled = pd.concat([_prices(AAA=[10.0, 11.0])] * 2, ignore_index=True)
    lake(doubled)
    silver = build_silver()
    assert len(silver) == 2
    assert not silver.duplicated(["date", "ticker"]).any()


def test_alignment_attributes_loss_to_a_mid_series_gap(lake, caplog):
    """Regression guard: a hole in the middle must not be blamed on a late start.

    The first version of this warning inferred the culprit from
    first_valid_index, so CCC's four-date gap was attributed to BBB's one-date
    late listing.
    """
    lake(
        _prices(
            AAA=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            BBB=[None, 20.0, 21.0, 22.0, 23.0, 24.0],
            CCC=[30.0, 31.0, None, None, 34.0, 35.0],
        )
    )
    build_silver()
    build_gold()

    with caplog.at_level("WARNING"):
        returns_matrix(("AAA", "BBB", "CCC"))

    assert "alignment dropped" in caplog.text
    # CCC costs strictly more dates than BBB and must be reported as such.
    # Two, not three: the missing rows are absent from silver rather than NaN,
    # so lag() spans the hole and CCC has a (multi-day) return on re-entry.
    assert "'CCC': 2" in caplog.text
    assert "'BBB': 1" in caplog.text
