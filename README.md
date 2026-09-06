# Portfolio Optimization — end-to-end DE / DS / MLOps project

A didactic, production-shaped project: pull daily prices for the 11 S&P sector
SPDR ETFs, build a validated data lake, construct optimal portfolios, and grow
the whole thing into an orchestrated, monitored ML system.

**Status: M1 complete** — a thin end-to-end slice runs today.

```bash
uv sync
uv run portfolio              # ingest -> transform -> optimise -> reports/
uv run portfolio --skip-ingest  # reuse cached bronze data
uv run pytest                 # fully offline; the price source is stubbed
```

## Architecture

```
Yahoo Finance ──▶ bronze/  raw OHLCV parquet, immutable, idempotent
                     │
                     ▼     DuckDB + pandera contract
                  silver/  cleaned, deduped, validated prices
                     │
                     ▼     DuckDB window functions
                   gold/   daily log returns
                     │
                     ▼     scipy SLSQP, long-only, 35% position cap
                 optimizer  min-variance · max-Sharpe · equal-weight
                     │
                     ▼
                  reports/  efficient frontier, weights, summary table
```

| Module | Role |
|---|---|
| `config.py` | every tunable in one frozen dataclass, so runs are reproducible |
| `ingest.py` | bronze layer; Yahoo behind a `PriceSource` Protocol so it's swappable |
| `transform.py` | silver + gold via DuckDB SQL; pandera schema gates the pipeline |
| `optimize.py` | portfolio construction and the efficient frontier |
| `report.py` | charts on a CVD-validated palette, plus a CSV table view |
| `pipeline.py` | wires the slice together; exposed as the `portfolio` CLI |

## A note on the data source

Yahoo Finance has **no official public API**. `yfinance` scrapes their
endpoints — free and fine here, but it breaks and rate-limits occasionally.
That is deliberate: it forces real ingestion hygiene, and `PriceSource` means
swapping in Stooq or Tiingo later touches one class.

## Roadmap

| | Milestone | Focus |
|---|---|---|
| ✅ | **M0** Scaffold — uv, ruff, pytest, pre-commit, CI | engineering hygiene |
| ✅ | **M1** Thin slice — ingest → transform → optimise → report | data engineering |
| ☐ | **M2** dbt models over DuckDB; incremental loads; freshness tests | analytics engineering |
| ☐ | **M3** Ledoit-Wolf shrinkage, risk parity, cvxpy, walk-forward backtest vs SPY | data science |
| ☐ | **M4** MLflow — every backtest is a tracked run | MLOps |
| ☐ | **M5** Expected-return forecasting model (and how to tell if it has any alpha) | ML |
| ☐ | **M6** Dagster assets, daily schedule, freshness checks | orchestration |
| ☐ | **M7** FastAPI `POST /optimize` + Streamlit dashboard | serving |
| ☐ | **M8** Evidently drift monitoring, model registry promotion gate | production MLOps |

## Current results

Sector ETFs, 2016→today, long-only, 35% cap:

| Strategy | Return | Volatility | Sharpe |
|---|---|---|---|
| Equal weight | 11.0% | 18.1% | 0.50 |
| Min variance | 9.7% | 15.3% | 0.50 |
| Max Sharpe | 13.6% | 17.0% | 0.68 |

These are **in-sample, ex-ante** figures — the optimiser saw the same returns it
is optimising over, so max-Sharpe winning here proves nothing. M3's walk-forward
backtest is what makes the comparison honest.
