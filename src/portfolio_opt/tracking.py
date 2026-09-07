"""MLflow experiment tracking.

Every backtest is a run: the Settings dataclass becomes the params, the metrics
table becomes the metrics, and the charts and CSVs become artifacts. That is the
whole point of having kept Settings frozen since M0 -- a run is reproducible from
its own logged parameters.

The tracking store is a local SQLite database (`mlflow.db`) with artifacts in
`mlartifacts/`, so this needs no server. MLflow put its filesystem backend into
maintenance mode, and SQLite is the documented local replacement -- it is also
what `mlflow ui --backend-store-uri sqlite:///mlflow.db` expects.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import mlflow
import pandas as pd

from .backtest import BacktestResult
from .config import PROJECT_ROOT, Settings

log = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "portfolio-optimization"
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"

# Logged as params rather than metrics: these define the run, they are not
# outcomes of it.
PARAM_FIELDS = (
    "start",
    "end",
    "risk_free_rate",
    "max_weight",
    "allow_short",
    "covariance_estimator",
    "estimation_window",
    "min_estimation_days",
    "rebalance",
    "transaction_cost_bps",
)


def configure(experiment: str = DEFAULT_EXPERIMENT, uri: str | None = None) -> str:
    """Point MLflow at the local store and select the experiment.

    Returns the experiment id. Idempotent: safe to call on every run.
    """
    mlflow.set_tracking_uri(uri or f"sqlite:///{TRACKING_DB}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    existing = mlflow.get_experiment_by_name(experiment)
    if existing is None:
        experiment_id = mlflow.create_experiment(
            experiment, artifact_location=ARTIFACT_DIR.as_uri()
        )
    else:
        experiment_id = existing.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def params_from(cfg: Settings) -> dict[str, object]:
    """Flatten Settings into MLflow params.

    The ticker tuple is logged as a count plus a sorted string: MLflow params are
    strings anyway, and a stable ordering keeps two runs over the same universe
    comparable even if the config order changes.
    """
    raw = asdict(cfg)
    params: dict[str, object] = {k: raw[k] for k in PARAM_FIELDS}
    params["benchmark"] = cfg.benchmark
    params["n_assets"] = len(cfg.tickers)
    params["universe"] = ",".join(sorted(cfg.tickers))
    return params


@contextmanager
def run(name: str, cfg: Settings, tags: dict[str, str] | None = None):
    """Open an MLflow run pre-populated with this configuration."""
    with mlflow.start_run(run_name=name) as active:
        mlflow.log_params(params_from(cfg))
        mlflow.set_tags({"strategy": name, **(tags or {})})
        yield active


def log_backtest(
    result: BacktestResult,
    cfg: Settings,
    tags: dict[str, str] | None = None,
    artifacts: list[Path] | None = None,
) -> str:
    """Record one strategy's walk-forward result. Returns the MLflow run id."""
    with run(result.name, cfg, tags) as active:
        mlflow.log_metrics({k: v for k, v in result.metrics.items() if pd.notna(v)})
        mlflow.log_metric("n_rebalances", len(result.weights))
        mlflow.log_metric("n_days", len(result.net_returns))

        # The equity curve as a step-indexed series makes runs comparable in the
        # MLflow UI without downloading artifacts.
        equity = result.equity
        for step, value in enumerate(equity.to_numpy()[:: max(1, len(equity) // 200)]):
            mlflow.log_metric("equity", float(value), step=step)

        mlflow.log_text(result.weights.to_csv(), f"weights/{result.name}.csv")
        for path in artifacts or []:
            if path.exists():
                mlflow.log_artifact(str(path))
        return active.info.run_id


def log_comparison(table: pd.DataFrame, cfg: Settings, name: str = "comparison") -> str:
    """A parent run holding the cross-strategy table and the winner."""
    with run(name, cfg, tags={"kind": "comparison"}) as active:
        mlflow.log_text(table.to_csv(), "backtest_summary.csv")
        best = table["sharpe"].idxmax()
        mlflow.set_tag("best_strategy", best)
        mlflow.log_metric("best_sharpe", float(table["sharpe"].max()))
        mlflow.log_metric(
            "spread_sharpe", float(table["sharpe"].max() - table["sharpe"].min())
        )
        # Did any strategy beat buy-and-hold? The question the project exists for.
        benchmarks = [i for i in table.index if i.startswith("benchmark_")]
        if benchmarks:
            bench = float(table.loc[benchmarks[0], "sharpe"])
            mlflow.log_metric("benchmark_sharpe", bench)
            mlflow.log_metric(
                "strategies_beating_benchmark",
                int((table.drop(index=benchmarks)["sharpe"] > bench).sum()),
            )
        return active.info.run_id
