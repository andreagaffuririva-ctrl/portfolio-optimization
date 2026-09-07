# Portfolio Optimization — end-to-end DE / DS / MLOps project

A didactic, production-shaped project: pull daily prices for the 11 S&P sector
SPDR ETFs, build a validated data lake, construct optimal portfolios, and grow
the whole thing into an orchestrated, monitored ML system.

**Status: M4 complete** — walk-forward backtesting, MLflow tracking, and a
parameter sweep that shows how much the M3 headline depended on its settings.

```bash
uv sync
uv run portfolio                     # ingest -> transform -> optimise -> backtest -> reports/
uv run portfolio --skip-ingest       # reuse cached bronze data
uv run portfolio --skip-backtest     # in-sample pass only (fast)
uv run portfolio --estimator sample  # compare against unshrunk covariance
uv run portfolio --track             # log runs to MLflow
uv run portfolio --sweep             # 24 configurations x 6 backtests, all tracked
uv run pytest                        # 78 tests, fully offline

mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs
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
| `covariance.py` | sample and Ledoit-Wolf shrinkage estimators |
| `optimize.py` | five strategies as convex programs, plus the efficient frontier |
| `backtest.py` | walk-forward engine: rolling window, monthly rebalance, drift, costs |
| `tracking.py` | MLflow: Settings become params, metrics become metrics, charts become artifacts |
| `sweep.py` | grid over estimation window, rebalance frequency and cost; stability analysis |
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
| ✅ | **M3** Ledoit-Wolf shrinkage, cvxpy, risk parity, walk-forward backtest vs SPY | data science |
| ✅ | **M4** MLflow — every backtest is a tracked run; parameter sweeps | MLOps |
| ☐ | **M5** Expected-return forecasting model (and how to tell if it has any alpha) | ML |
| ☐ | **M6** Dagster assets, daily schedule, freshness checks | orchestration |
| ☐ | **M7** FastAPI `POST /optimize` + Streamlit dashboard | serving |
| ☐ | **M8** Evidently drift monitoring, model registry promotion gate | production MLOps |

## Results

### Out-of-sample — the walk-forward backtest

756-day rolling estimation window, month-end rebalancing, 10bps one-way costs on
a portfolio that drifts between rebalances. 88 rebalances, 7.2 years, 2019-07 →
2026-09. Weights computed at each month-end are executed the **next** trading day.

| Strategy | CAGR | Vol | Sharpe | Max DD | Calmar | Turnover/yr |
|---|---|---|---|---|---|---|
| **SPY (buy & hold)** | **16.06%** | 19.67% | **0.75** | −33.7% | **0.48** | 14% |
| Max diversification | 13.24% | 18.47% | 0.66 | −39.6% | 0.33 | 55% |
| Equal weight | 13.15% | 18.49% | 0.65 | −36.2% | 0.36 | 30% |
| Risk parity | 12.39% | 17.96% | 0.63 | −35.7% | 0.35 | 31% |
| Min variance | 8.67% | 16.22% | 0.47 | −33.1% | 0.26 | 47% |
| Max Sharpe | 7.59% | 19.02% | 0.38 | −33.4% | 0.23 | **203%** |

### In-sample vs out-of-sample

| Strategy | In-sample Sharpe | Out-of-sample | Retained |
|---|---|---|---|
| Max Sharpe | 0.82 | 0.38 | **46%** |
| Min variance | 0.62 | 0.47 | 76% |
| Max diversification | 0.72 | 0.66 | 91% |
| Risk parity | 0.65 | 0.63 | 97% |
| Equal weight | 0.65 | 0.65 | **100%** |
| SPY (buy & hold) | 0.72 | 0.75 | 104% |

## What this actually shows

**Every optimiser loses to buying the index.** Not one of the five beats SPY's
0.75 Sharpe out of sample. The most sophisticated method available here is worse
than the least sophisticated thing possible.

**Max Sharpe is the worst strategy, having been the best in-sample.** It keeps
46% of its apparent edge, and it trades **203% of the portfolio per year** to
achieve that. It is a machine for converting estimation error into transaction
costs: sampling noise in the mean-return vector moves the optimum a long way,
so it chases sector rankings that do not persist.

**The less a strategy estimates, the better it travels.** The retention column
sorts almost perfectly by how much the strategy needs to know. Equal weight
estimates nothing and keeps 100%. Max diversification and risk parity use only
the covariance — the *stable* moment — and keep 91–97%. Max Sharpe needs the
mean, the hardest thing in finance to estimate, and keeps 46%. This reproduces
the well-known DeMiguel, Garlappi & Uppal (2009) result that 1/N is hard to beat.

**Shrinkage barely mattered here** — an honest null result. Ledoit-Wolf picks a
shrinkage intensity of only 0.014 on the full sample, and swapping it for the raw
sample covariance moves out-of-sample Sharpe by ≤0.01 (`--estimator sample`).
With 756 observations for 11 assets there is simply enough data. Shrinkage earns
its keep when T/N is small; try it with 30+ stocks and a 1-year window.

### Does that ranking survive a different setting?

M3 reported one configuration: 756-day window, month-end rebalance, 10bps. Three
arbitrary choices. The M4 sweep runs all 24 combinations of
`{252, 504, 756, 1260} x {monthly, quarterly} x {0, 10, 50}bps` — 144 backtests,
each an MLflow run.

| Strategy | Sharpe (mean ± sd) | Range | Mean rank | Configs won |
|---|---|---|---|---|
| **SPY (buy & hold)** | 0.75 ± 0.00 | 0.00 | **1.1** | **21 / 24** |
| Equal weight | 0.66 ± 0.01 | 0.02 | 2.6 | 0 |
| Max diversification | 0.65 ± 0.02 | 0.07 | 3.0 | 0 |
| Risk parity | 0.64 ± 0.01 | 0.05 | 3.9 | 0 |
| Max Sharpe | 0.54 ± 0.15 | **0.50** | 4.7 | **3** |
| Min variance | 0.49 ± 0.04 | 0.13 | 5.6 | 0 |

**The M3 headline was partly an artefact of one parameter.** Max Sharpe is not
reliably the worst strategy — it is the *least reliable* strategy. Its Sharpe
runs from 0.38 to 0.83 depending only on the estimation window, and the
relationship is not even monotonic:

| Window | 252 | 504 | 756 | 1260 |
|---|---|---|---|---|
| Max Sharpe (ME, 10bps) | **0.82** | 0.48 | **0.38** | 0.58 |
| Equal weight | 0.65 | 0.65 | 0.65 | 0.65 |

At a 252-day window it beats SPY outright and wins its three configurations.
M3 happened to pick 756 — max Sharpe's worst.

**This is not evidence that max Sharpe is good with a 252-day window.** A
strategy whose performance swings by 0.5 Sharpe on a parameter with no
theoretical justification has demonstrated parameter sensitivity, not skill.
Choosing 252 *after* seeing these results is precisely the selection bias that
walk-forward testing exists to prevent — the honest reading is that this family
of methods cannot be relied on here, which the ± column says more clearly than
any single run could.

Everything that estimates less stays flat: equal weight varies by 0.02 across all
24 configurations, and SPY by 0.00 because it depends on none of the knobs.

## Caveats on the above

- **One period, one universe.** 7.2 years, mostly a bull market with two sharp
  drawdowns. No significance test on the Sharpe differences, so "SPY wins" is
  not established at any confidence level — it is what happened. The sweep
  widens the parameter axis but not the data axis: all 24 configurations share
  the same 7.2 years, so they are 24 correlated views of one history, not 24
  independent trials.
- **Costs are a flat 10bps** on one-way turnover, with no spread, slippage or
  market impact. Max Sharpe's 203% turnover would suffer more than this in reality.
- **The risk-free rate is a hardcoded 2%** across a period spanning ZIRP and 5%+
  policy rates, so every Sharpe is only loosely calibrated. Tracked as P3-3.
- **Risk parity ignores the position cap** — capping would break the equal-risk
  property, so a breach is logged rather than silently mangled.
- **SPY is not a fair fight in one respect**: it is a market-cap-weighted index
  that had a historically concentrated run in mega-cap tech, which no
  equal-ish-weighted sector portfolio could match. That is a real explanation,
  not an excuse — the strategies still lost.

See `PROJECT_LOG.md` for the full issue list and next steps.
