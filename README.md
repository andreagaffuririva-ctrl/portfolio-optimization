# Portfolio Optimization — end-to-end DE / DS / MLOps project

A didactic, production-shaped project: pull daily prices for the 11 S&P sector
SPDR ETFs, build a validated data lake, construct optimal portfolios, and grow
the whole thing into an orchestrated, monitored ML system.

**Status: M1 complete + data-correctness pass** — a thin end-to-end slice runs today.

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
| `transform.py` | silver + gold via DuckDB SQL; pandera schema gates the pipeline; logs the effective date window |
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

Sector ETFs, long-only, 35% cap. **Effective window 2018-06-20 → today** (2,064
days): XLC was only created in June 2018, and mean-variance needs a common
calendar, so the inner join truncates the earlier history. The pipeline now logs
this explicitly rather than doing it silently.

| Strategy | Return | Volatility | Sharpe |
|---|---|---|---|
| Equal weight | 13.60% | 18.01% | 0.64 |
| Min variance | 11.42% | 15.27% | 0.62 |
| Max Sharpe | 16.22% | 17.32% | 0.82 |
| **SPY (buy & hold)** | **15.88%** | **19.17%** | **0.72** |

Two things to read off this table:

**Equal weight and min variance both lose to simply buying SPY.** All that
machinery, and two of three strategies are beaten by the benchmark on a
risk-adjusted basis. That is the normal result, and it is why the benchmark row
exists.

**Max Sharpe "winning" is meaningless.** These are **in-sample, ex-ante**
figures — the optimiser saw the exact returns it is optimising over, so it wins
by construction. Only M3's walk-forward backtest can say anything honest, and
the usual finding is that the advantage evaporates out of sample.
