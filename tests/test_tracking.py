"""MLflow tracking tests.

Each test gets its own SQLite store in tmp_path, so nothing touches the real
mlflow.db and the tests stay order-independent.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest

from portfolio_opt.backtest import BacktestResult
from portfolio_opt.config import Settings
from portfolio_opt.tracking import (
    PARAM_FIELDS,
    configure,
    log_backtest,
    log_comparison,
    params_from,
)

CFG = Settings(tickers=("XLK", "XLF"), estimation_window=60, transaction_cost_bps=5.0)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An isolated tracking database and artifact root."""
    monkeypatch.setattr("portfolio_opt.tracking.TRACKING_DB", tmp_path / "mlflow.db")
    monkeypatch.setattr("portfolio_opt.tracking.ARTIFACT_DIR", tmp_path / "artifacts")
    configure("test-experiment")
    yield tmp_path
    mlflow.end_run()


@pytest.fixture
def result() -> BacktestResult:
    index = pd.bdate_range("2021-01-01", periods=120)
    rng = np.random.default_rng(5)
    net = pd.Series(rng.normal(0.0005, 0.01, len(index)), index=index)
    weights = pd.DataFrame(
        {"XLK": [0.6, 0.4], "XLF": [0.4, 0.6]},
        index=[index[0], index[60]],
    )
    return BacktestResult(
        name="min_variance",
        net_returns=net,
        gross_returns=net + 0.0001,
        weights=weights,
        turnover=pd.Series([1.0, 0.2], index=weights.index),
        metrics={"sharpe": 0.8, "cagr": 0.11, "max_drawdown": -0.2, "calmar": 0.55},
    )


def _runs() -> pd.DataFrame:
    return mlflow.search_runs(experiment_names=["test-experiment"])


# --------------------------------------------------------------------------
# params
# --------------------------------------------------------------------------


def test_every_declared_param_field_is_captured():
    params = params_from(CFG)
    assert set(PARAM_FIELDS) <= set(params)
    assert params["estimation_window"] == 60
    assert params["transaction_cost_bps"] == 5.0


def test_universe_is_logged_order_independently():
    """Two configs over the same tickers must produce the same param string."""
    forward = params_from(Settings(tickers=("XLK", "XLF")))
    reversed_ = params_from(Settings(tickers=("XLF", "XLK")))
    assert forward["universe"] == reversed_["universe"] == "XLF,XLK"
    assert forward["n_assets"] == 2


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------


def test_log_backtest_records_metrics_and_params(store, result):
    run_id = log_backtest(result, CFG)
    run = mlflow.get_run(run_id)

    assert run.data.metrics["sharpe"] == pytest.approx(0.8)
    assert run.data.metrics["calmar"] == pytest.approx(0.55)
    assert run.data.metrics["n_rebalances"] == 2
    assert run.data.params["estimation_window"] == "60"
    assert run.data.tags["strategy"] == "min_variance"


def test_nan_metrics_are_skipped_not_logged_as_nan(store, result):
    """MLflow accepts NaN and it poisons comparisons -- drop them instead."""
    broken = BacktestResult(
        **{**result.__dict__, "metrics": {"sharpe": 0.5, "calmar": np.nan}}
    )
    run = mlflow.get_run(log_backtest(broken, CFG))
    assert "sharpe" in run.data.metrics
    assert "calmar" not in run.data.metrics


def test_weights_are_attached_as_an_artifact(store, result):
    run_id = log_backtest(result, CFG)
    artifacts = [f.path for f in mlflow.MlflowClient().list_artifacts(run_id, "weights")]
    assert "weights/min_variance.csv" in artifacts


def test_tags_flow_through(store, result):
    run = mlflow.get_run(log_backtest(result, CFG, tags={"window": "60"}))
    assert run.data.tags["window"] == "60"


# --------------------------------------------------------------------------
# comparison run
# --------------------------------------------------------------------------


def test_comparison_records_the_winner_and_the_benchmark_question(store):
    table = pd.DataFrame(
        {"sharpe": [0.75, 0.66, 0.38]},
        index=["benchmark_SPY", "equal_weight", "max_sharpe"],
    )
    run = mlflow.get_run(log_comparison(table, CFG))

    assert run.data.tags["best_strategy"] == "benchmark_SPY"
    assert run.data.metrics["best_sharpe"] == pytest.approx(0.75)
    assert run.data.metrics["benchmark_sharpe"] == pytest.approx(0.75)
    # The question the project exists to answer.
    assert run.data.metrics["strategies_beating_benchmark"] == 0


def test_comparison_counts_strategies_that_do_beat_the_benchmark(store):
    table = pd.DataFrame(
        {"sharpe": [0.9, 0.8, 0.5]},
        index=["equal_weight", "risk_parity", "benchmark_SPY"],
    )
    run = mlflow.get_run(log_comparison(table, CFG))
    assert run.data.metrics["strategies_beating_benchmark"] == 2
    assert run.data.metrics["spread_sharpe"] == pytest.approx(0.4)


def test_configure_is_idempotent(store):
    first = configure("test-experiment")
    second = configure("test-experiment")
    assert first == second
