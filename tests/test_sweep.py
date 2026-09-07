"""Sweep tests. Tracking is off so these stay fast and touch no database."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_opt.config import Settings
from portfolio_opt.sweep import GRID, grid, one_configuration, stability

CFG = Settings(
    tickers=("AAA", "BBB", "CCC"),
    benchmark="BENCH",
    max_weight=1.0,
    risk_free_rate=0.0,
    estimation_window=60,
    min_estimation_days=40,
)

AXES = {
    "estimation_window": (60, 90),
    "rebalance": ("ME", "QE"),
    "transaction_cost_bps": (0.0, 25.0),
}


@pytest.fixture
def data():
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2020-01-01", periods=400)
    daily = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(len(index), 3)),
        index=index,
        columns=["AAA", "BBB", "CCC"],
    )
    benchmark = pd.Series(rng.normal(0.0006, 0.011, len(index)), index=index)
    return daily, benchmark


def test_one_configuration_covers_every_strategy_plus_benchmark(data):
    daily, benchmark = data
    table = one_configuration(daily, benchmark, CFG, track=False)
    assert len(table) == 6  # five strategies plus the benchmark
    assert "benchmark_BENCH" in table.index
    assert table["sharpe"].is_monotonic_decreasing


def test_grid_produces_a_row_per_configuration_and_strategy(data):
    daily, benchmark = data
    long_frame = grid(daily, benchmark, base=CFG, axes=AXES, track=False)

    configurations = 2 * 2 * 2
    assert len(long_frame) == configurations * 6
    assert set(long_frame.columns) >= {
        "estimation_window",
        "rebalance",
        "transaction_cost_bps",
        "strategy",
        "sharpe",
    }
    assert long_frame["estimation_window"].nunique() == 2


def test_higher_costs_never_improve_a_strategy(data):
    """A sanity check on the sweep itself: cost is monotone in the right direction."""
    daily, benchmark = data
    axes = {
        "estimation_window": (60,),
        "rebalance": ("ME",),
        "transaction_cost_bps": (0.0, 100.0),
    }
    frame = grid(daily, benchmark, base=CFG, axes=axes, track=False)
    pivot = frame.pivot(index="strategy", columns="transaction_cost_bps", values="sharpe")
    assert (pivot[100.0] <= pivot[0.0] + 1e-9).all()


def test_stability_summarises_wins_and_spread(data):
    daily, benchmark = data
    frame = grid(daily, benchmark, base=CFG, axes=AXES, track=False)
    table = stability(frame)

    assert len(table) == 6
    assert table["wins"].sum() == 8  # one winner per configuration
    assert (table["sharpe_range"] >= 0).all()
    assert (table["rank_mean"] >= 1).all()
    assert (table["rank_mean"] <= 6).all()
    assert table["sharpe_mean"].is_monotonic_decreasing


def test_default_grid_is_declared_over_the_three_knobs():
    assert set(GRID) == {
        "estimation_window",
        "rebalance",
        "transaction_cost_bps",
    }
