# Project Log — Portfolio Optimization

Running engineering log: what exists, what is broken, what comes next.
Update this file at the end of every working session.

---

## Session 1 — 2026-09-06

**Outcome:** M0 (scaffold) and M1 (thin end-to-end slice) complete, verified against
live data, pushed to GitHub with green CI.

- **Repo:** https://github.com/andreagaffuririva-ctrl/portfolio-optimization (public)
- **Commit:** `2b5741c` — "M0+M1: thin end-to-end slice — ingest, transform, optimise, report"
- **CI run:** #34049402597 — success, 15s
- **Working tree:** clean except this file (untracked)

### Decisions taken, and why

| Decision | Chosen | Rationale | Reversibility |
|---|---|---|---|
| Build sequence | Thin end-to-end slice first | Results on day one; never a half-built system | n/a |
| Universe | 11 S&P sector SPDR ETFs + SPY benchmark | Small, liquid, long history, economically interpretable; diversification actually shows up | Easy — `config.SECTOR_ETFS` |
| Data source | Yahoo via `yfinance` | Free, no key. **No official API exists** — yfinance scrapes it | Easy — `PriceSource` Protocol |
| Storage | Parquet lake (bronze/silver/gold) + DuckDB engine | Zero infra, real warehouse semantics, modern local-analytics stack | Medium |
| Optimiser | `scipy` SLSQP | Constraint set stays explicit and readable | Planned swap to cvxpy at M3 |
| Orchestrator (target) | Dagster at M6 | Asset-based, pairs natively with dbt+DuckDB, strong on a CV | Not yet built |
| Package manager | `uv`, Python 3.12 | Fast, lockfile-based, reproducible in CI | Locked via `.python-version` |
| Env note | System Python is 3.13.1; project pins **3.12** | Avoids scientific-wheel gaps | Deliberate |

### What was built

```
portfolio_optimization/
├── src/portfolio_opt/
│   ├── config.py       50 loc   frozen Settings dataclass, paths, universe, TRADING_DAYS=252
│   ├── ingest.py      103 loc   bronze; PriceSource Protocol + YahooSource + _to_tidy
│   ├── transform.py    97 loc   silver (DuckDB + pandera gate) and gold (log returns)
│   ├── optimize.py    118 loc   Portfolio dataclass, min-var / max-Sharpe / equal-weight, frontier
│   ├── report.py      179 loc   frontier + weights charts, CSV table view
│   └── pipeline.py     62 loc   orchestration of the slice; `portfolio` CLI entrypoint
├── tests/
│   ├── test_ingest.py  60 loc   4 tests, network stubbed
│   └── test_optimize.py 63 loc  5 tests, synthetic covariance
├── .github/workflows/ci.yml     ruff check + ruff format --check + pytest on push/PR
├── .pre-commit-config.yaml      ruff, ruff-format, trailing-whitespace, large-file guard
├── pyproject.toml               deps, ruff (line-length 90, E/F/I/UP/B/SIM), pytest config
└── README.md                    architecture, roadmap, results, data-source caveat
```

734 lines of Python total.

**Design choices inside the code worth remembering:**

- `Settings` is a **frozen dataclass** and every function signature takes
  `cfg: Settings = settings`. Tests override one field without touching globals,
  and at M4 the whole object serialises straight into an MLflow run as params.
- `PriceSource` is a **Protocol**, not a base class. `YahooSource` is one
  implementation; tests inject `StubSource`. This is why CI never touches the network.
- `_to_tidy()` is **separate from fetching**. yfinance's `(field, ticker)` MultiIndex
  is the fiddliest part of ingestion and is now independently testable.
- Silver uses DuckDB's `QUALIFY` to dedupe on a window function without a subquery —
  the same dialect works in Snowflake and BigQuery.
- `PriceSchema.validate()` is a **hard gate**: bad data raises rather than silently
  producing a portfolio.
- Every strategy in `optimize.py` is *just an objective function*; `_solve()` owns
  the constraints. Adding risk parity at M3 is one lambda.
- Chart palette (`report.py`) is the validated blue/orange/aqua categorical set —
  passes all-pairs colour-vision-deficiency and normal-vision separation floors.
  Aqua sits under 3:1 contrast on the light surface, which is why every point is
  direct-labelled and `summary.csv` exists as the table view.

### Verification evidence

- **9/9 tests pass**, fully offline, in 21s locally and 15s in CI.
- **Live run succeeded:** 31,589 raw rows → 31,589 silver rows → 31,577 gold return
  rows → returns matrix of 2,064 days × 11 assets → 3 portfolios + 4 PNGs + CSV.
- `ruff check` and `ruff format --check` both clean.

### Results produced (2016→today config, long-only, 35% cap)

| Strategy | Ann. return | Ann. volatility | Sharpe |
|---|---|---|---|
| Equal weight | 11.00% | 18.08% | 0.50 |
| Min variance | 9.72% | 15.30% | 0.50 |
| Max Sharpe | 13.58% | 17.04% | 0.68 |

**These numbers are in-sample and ex-ante and prove nothing.** The optimiser saw the
exact returns it optimised over. Max-Sharpe "winning" is guaranteed by construction,
not evidence of skill. M3's walk-forward backtest is what makes the comparison honest.
This is the single most important caveat in the project.

---

## Session 2 — 2026-09-07

**Outcome:** `fix/data-correctness` — the six immediate items from session 1, all
verified. Test count 9 → 24.

### Done

| ID | Issue | Resolution |
|---|---|---|
| P3-4 | Pre-commit hooks never installed | `pre-commit install` — hook now at `.git/hooks/pre-commit` |
| P1-4 | Docstring claimed retries that did not exist | Implemented with `tenacity`: 4 attempts, exponential backoff 2→30s, **only** `TransientSourceError` retried |
| P1-1 | Silent 23% history truncation | `_log_alignment()` now WARNs with the % dropped, the binding ticker, and every per-ticker first date |
| P1-2 | Log returns used where simple returns belong | `build_gold()` emits both `simple_return` and `log_return`; `returns_matrix(value=...)` defaults to simple |
| P2-1 | SPY fetched but never used | `buy_and_hold()` in `optimize.py`; benchmark now on the chart and in `summary.csv` |
| P2-4 | No tests on `transform.py` | `tests/test_transform.py`, 11 tests, incl. explicit regression guards for P1-1 and P1-2 |

### What the fixes actually revealed

**P1-2 was biasing every return down by ~2.6 percentage points.** Switching from
log to simple returns moved equal weight from 11.00% → 13.60%, min variance
9.72% → 11.42%, max Sharpe 13.58% → 16.22%. Volatility barely moved, as expected
(the convention matters for the mean vector, not much for daily covariance). This
was not a rounding issue — it was large enough to change which strategy looks best.

**P1-1 is worse than estimated.** The live run reports `alignment dropped 619 of
2,683 dates (23.1%)`, naming XLC (first data 2018-06-20) as the binding ticker
while all eleven others start 2016-01-05.

**P2-1 produced the project's first genuinely interesting result.** With SPY on
the chart: SPY Sharpe **0.72** vs equal weight **0.64** and min variance **0.62**.
Two of three strategies are beaten by simply buying the index. Max Sharpe shows
0.82, but in-sample, so it means nothing yet. This is exactly the comparison the
project existed to make, and it was previously unanswerable.

### Design decisions taken this session

- **Benchmark encoded by shape, not a fourth colour.** The validated categorical
  palette only clears the all-pairs colour-vision floor at three slots; a fourth
  hue would put yellow beside orange and fail. SPY is a black diamond in neutral
  ink — which also reads correctly: a reference, not a strategy.
- **Retry is selective.** A missing column raises immediately (`test_schema_break_
  is_not_retried`) instead of hammering Yahoo four times over a real bug.
- **Both return conventions kept in gold**, rather than replacing one. Log returns
  are still the right choice for cumulative equity curves at M3.
- **Backoff patched to zero in tests** via a `no_backoff` fixture, so the retry
  path is genuinely exercised without a slow suite (24 tests in 1.65s).
- **XLC kept** (per the session-2 decision): a complete 11-sector map beats 2.5
  extra years, provided the truncation is loud. It now is.
- `equal_weight` still ignores `max_weight` (P2-5) and the covariance is still
  unshrunk (P2-3) — both deliberately deferred to M3.

### Verification

- 24/24 tests pass, offline, 1.65s.
- `ruff check` and `ruff format --check` clean; pre-commit ran on the commit.
- Live `--skip-ingest` run produced all 4 PNGs + `summary.csv`, warning fired as designed.
- Frontier chart inspected: no label collisions, benchmark marker legible.

### Still open — unchanged from session 1

P1-3 (non-convex max-Sharpe → cvxpy), P2-2 (ingest not incremental; **blocks the
M6 daily schedule**), P2-3 (no shrinkage), P2-5 (`equal_weight` ignores the cap),
P3-1 (TzCache warning), P3-2 (f-string SQL), P3-3 (hardcoded 2% risk-free rate
across a ZIRP-to-5% period), P3-5 (misc).

**Next session opens at M2 (dbt) or M3 (shrinkage + cvxpy + walk-forward
backtest).** Recommendation unchanged: M3. The table above now shows two
strategies losing to SPY, and only an out-of-sample backtest can say whether
that is the real verdict.

---

## Session 3 — 2026-09-07 (later)

**Outcome:** `fix/review-followups` — items 1–4 from the code review of PR #1.
No behaviour change: identical portfolio numbers, identical chart. These were
correctness-of-the-supporting-code fixes. Tests 24 → 32.

### Done

| # | Issue | Resolution |
|---|---|---|
| 1 | `except Exception` retried genuine bugs | `RETRYABLE_ERRORS = (OSError, YFRateLimitError)`. Every `requests` exception subclasses `OSError`, so the network surface is covered without importing requests; `TypeError`/`AttributeError`/`KeyError` now propagate on the first attempt |
| 2 | Alignment warning named the wrong ticker | Attribution is now **counted, not inferred**: per-ticker missing-date counts, sole-cause counts, and first/last data per ticker |
| 3 | `buy_and_hold` untested | 5 tests, incl. one locking it to the single-asset `equal_weight` path |
| 4 | `"benchmark_"` prefix used for control flow in 4 places | `Portfolio.is_benchmark: bool = False`; prefix is now display-only |

### How each was verified rather than assumed

Before fixing, each defect was reproduced:

- **#1** — a stubbed `TypeError: download() got an unexpected keyword argument`
  produced `yf.download called 4x`. Now `called 1x`, pinned by
  `test_api_signature_change_is_not_retried`, with
  `test_network_errors_are_still_retried` guarding the narrowing.
- **#2** — synthetic frame (AAA complete, BBB one day late, CCC a 4-day hole)
  reported `dropped 5 of 10 ... but BBB only has data from 2024-01-02` — BBB
  caused 1 of 5, CCC caused 4 and went unmentioned. Now reports
  `{'CCC': 4, 'BBB': 1}`. Pinned by
  `test_alignment_attributes_loss_to_a_mid_series_gap`.
- **#3** — `grep buy_and_hold tests/` returned nothing.
- **#4** — `grep -rn "benchmark_" src/` showed the prefix in 4 places, 3 of them
  branching on it.

Live output after the fix, now accurate:
`alignment dropped 619 of 2,683 dates (23.1%). Dates missing per ticker:
{'XLC': 619}. Recoverable by dropping that ticker alone: {'XLC': 619}.`

### Also changed (small, same expressions)

- `SERIES.get(p.name, "#2a78d6")` → `SERIES[p.name]` in `report.py`. The fallback
  silently painted an unknown strategy the same blue as equal weight. It now
  raises. **Consequence: adding risk parity at M3 requires assigning it a colour
  or the report crashes** — deliberate, since a silent colour collision is worse.
- Dropped a needless function-level `from .config import TRADING_DAYS` inside
  `buy_and_hold` (item 5 in the review; it was in a function being rewritten).

### New finding, logged not fixed — P2-6

`build_gold()`'s `lag(close)` spans data gaps. Missing prices are **absent rows**
in silver, not NaNs, so a ticker returning after a 4-day hole gets a single
"daily" return covering 5 calendar days. Surfaced while writing the test for #2:
the expected count was 3, the real count 2, and the discrepancy was this.

Harmless for the sector ETFs (no gaps), but it will silently inflate one return
per gap on any less liquid universe, and it distorts volatility. Fix at M2 with a
proper date spine: reindex each ticker onto the full trading calendar and require
`date - prev_date <= 4 business days`, or emit `days_elapsed` and filter.

### Still open

P1-3 (non-convex max-Sharpe → cvxpy), P2-2 (ingest not incremental; **blocks M6**),
P2-3 (no shrinkage), P2-5 (`equal_weight` ignores the cap), **P2-6 (new, above)**,
P3-1/2/3, `report.py` still untested, and the pre-existing `efficient_frontier`
upper bound being infeasible under the position cap (`hi = mu.max() * 0.99`) so the
frontier's top end is truncated by solver failure rather than by design.

---

## Session 4 — 2026-09-07 (M3)

**Outcome:** `feat/m3-backtest` — M3 complete. Shrinkage, cvxpy, two new
strategies, and the walk-forward backtest. Tests 32 → 64. This is the milestone
that turns the project from "three portfolios" into evidence.

### Built

| Component | File | Notes |
|---|---|---|
| Covariance estimators | `covariance.py` (new) | sample + Ledoit-Wolf; only the covariance is shrunk, not the mean |
| Convex optimiser | `optimize.py` (rewritten) | cvxpy throughout; five strategies |
| Risk parity | `optimize.py` | Spinu (2013) convex form: min 0.5 w'Sw - (1/n)sum(log w) |
| Max diversification | `optimize.py` | max-Sharpe program with volatilities in place of excess returns |
| Walk-forward engine | `backtest.py` (new) | rolling window, month-end rebalance, t+1 execution, drift, costs |
| Backtest charts | `report.py` | equity curve (log), drawdown, in-vs-out-of-sample Sharpe |
| Two-pass pipeline | `pipeline.py` | in-sample illustration and walk-forward, reported side by side |

### Issues closed

- **P1-3** — max-Sharpe was non-convex under SLSQP. Now the standard change of
  variables (`y = w / ((mu-rf)'w)`) makes it a QP with a guaranteed global
  optimum. Pinned by `test_max_sharpe_is_start_point_independent`, which checks
  400 random long-only portfolios cannot beat the reported Sharpe.
- **P2-3** — Ledoit-Wolf shrinkage available and now the default.
- **P2-5** — `equal_weight` silently breached `max_weight`; it now raises.
- **Pre-existing frontier bug** — `hi = mu.max() * 0.99` was infeasible under the
  position cap, so the top of the frontier was decided by solver failure.
  Replaced with `max_attainable_return()`, which solves for it. Measured: max
  attainable under the 35% cap is **18.59%** vs the best single asset's 24.74%,
  so roughly a quarter of the old frontier's range never existed.

### The result

Out-of-sample, 88 rebalances, 7.2 years, net of 10bps one-way costs:

| Strategy | CAGR | Sharpe | Turnover/yr | In-sample Sharpe | Retained |
|---|---|---|---|---|---|
| **SPY (buy & hold)** | 16.06% | **0.75** | 14% | 0.72 | 104% |
| Max diversification | 13.24% | 0.66 | 55% | 0.72 | 91% |
| Equal weight | 13.15% | 0.65 | 30% | 0.65 | **100%** |
| Risk parity | 12.39% | 0.63 | 31% | 0.65 | 97% |
| Min variance | 8.67% | 0.47 | 47% | 0.62 | 76% |
| Max Sharpe | 7.59% | 0.38 | **203%** | **0.82** | **46%** |

**Every optimiser loses to the index.** And the ranking inverts: max-Sharpe was
best in-sample (0.82) and is worst out-of-sample (0.38), while burning 203% of
the portfolio in annual turnover.

The retention column sorts by how much each strategy must estimate. Equal weight
estimates nothing → keeps 100%. Max diversification and risk parity use only the
covariance, the stable moment → 91-97%. Max Sharpe needs the mean vector, the
hardest quantity in finance to estimate → 46%. This is the DeMiguel, Garlappi &
Uppal (2009) 1/N result, reproduced on our own data.

### Honest null result worth keeping

**Shrinkage made almost no difference.** Ledoit-Wolf picks intensity 0.014 on the
full sample, and `--estimator sample` moves out-of-sample Sharpe by <= 0.01. With
756 observations for 11 assets, there is enough data that the sample estimator is
fine. Shrinkage pays when T/N is small — the natural demo is 30+ single stocks on
a 1-year window, which would be a good M3.5 exercise. Recording this rather than
implying the shrinkage "worked".

### Design decisions

- **Drift is simulated day by day.** An earlier draft expanded target weights to
  a daily held frame, which reported equal weight as having *zero* maintenance
  turnover — wrong, because a fixed target still needs trades as prices move.
  `_simulate()` now evolves holdings with returns and only pulls them back on
  execution days. Guarded by `test_unchanged_target_still_costs_money`.
- **Costs land on the execution day**, not the decision day, matching when the
  new weights start earning.
- **Look-ahead is tested behaviourally, not just structurally.**
  `test_mutating_the_future_cannot_change_the_past` replaces the tail of the
  return series with garbage and asserts every prior net return is unchanged.
  A structural test of `shift(1)` would pass even if the estimator leaked.
- **Frontier scatter dropped its colours.** Six marks exceed the three-slot
  all-pairs colour-vision floor, so strategy points are neutral with direct
  labels; the five categorical hues moved to the equity/drawdown lines, where
  the adjacent pairlist applies and five slots validate.
- **Risk parity does not honour `max_weight`.** Capping would destroy the
  equal-risk property, so a breach is logged instead of silently producing
  something that is neither. Verified equal risk on real data: all 11
  contributions 0.0909, spread 8.4e-6.
- **Both CAGR and annualised arithmetic mean are reported.** They differ by the
  volatility drag, and quoting one as the other is a standard way to flatter a
  backtest.

### Verification

- 64/64 tests pass, offline, 2.34s
- Live run produces 11 artefacts; both passes logged side by side
- All four charts inspected; two layout defects found and fixed (legend colliding
  with a value label, log-axis falling back to scientific notation)
- `--estimator sample` cross-check run to confirm the shrinkage null result

### Still open

P2-2 (ingest not incremental; **blocks M6**), P2-6 (lag spans gaps), P3-1/2/3,
`report.py` still untested. New for M3:

- **P2-7** — no significance test on the Sharpe differences. "SPY wins" is a
  description of one 7.2-year path, not an inference. Deflated Sharpe or a
  stationary bootstrap would let the project say how confident it is.
- **P2-8** — costs are flat 10bps with no spread, slippage or market impact.
  Max Sharpe's 203% turnover is exactly the case where that simplification
  flatters the result most.
- **~~P2-9~~ — RESOLVED session 5.** No sensitivity analysis over the estimation
  window, rebalance frequency, or cost assumption. All three swept in M4, and
  the ranking *did* change: max Sharpe wins 3 of 24 configurations. The "each
  capable of changing the ranking" worry was correct.

**Next: M2 (dbt + incremental) or M4 (MLflow).** M4 is now more attractive than
before — with three knobs worth sweeping (P2-9), experiment tracking has real
work to do rather than being ceremony over a single run.

---

## Session 5 — 2026-09-07 (M4)

**Outcome:** `feat/m4-mlflow` — MLflow tracking plus the parameter sweep it
exists to serve. Tests 64 → 78. The sweep materially revises the M3 conclusion.

### Built

| Component | File | Notes |
|---|---|---|
| MLflow tracking | `tracking.py` (new) | Settings -> params, metrics -> metrics, charts -> artifacts |
| Parameter sweep | `sweep.py` (new) | grid over the three knobs; `stability()` summarises wins and spread |
| Sensitivity chart | `report.py` | Sharpe vs estimation window, one line per strategy |
| CLI | `pipeline.py` | `--track` and `--sweep` |

### The finding — M3's headline was partly a parameter artefact

24 configurations (`{252,504,756,1260}d` x `{ME,QE}` x `{0,10,50}bps`), 144
backtests:

| Strategy | Sharpe mean +/- sd | Range | Mean rank | Configs won |
|---|---|---|---|---|
| benchmark_SPY | 0.75 +/- 0.00 | 0.00 | 1.1 | **21/24** |
| equal_weight | 0.66 +/- 0.01 | 0.02 | 2.6 | 0 |
| max_diversification | 0.65 +/- 0.02 | 0.07 | 3.0 | 0 |
| risk_parity | 0.64 +/- 0.01 | 0.05 | 3.9 | 0 |
| max_sharpe | 0.54 +/- 0.15 | **0.50** | 4.7 | **3** |
| min_variance | 0.49 +/- 0.04 | 0.13 | 5.6 | 0 |

**Max Sharpe wins 3 of 24 configurations**, all at a 252-day window, where it
scores 0.82 and beats SPY. M3 used 756 days, which is its *worst* setting
(0.38). The relationship is non-monotonic — 0.82 / 0.48 / 0.38 / 0.58 across
252 / 504 / 756 / 1260 — so it is not "shorter is better" either.

**Correct reading, and the one to keep:** this is not evidence that max Sharpe
works at 252 days. A method whose Sharpe swings 0.5 on a parameter nobody can
justify a priori has demonstrated *sensitivity*, not skill. Picking 252 after
seeing the sweep is exactly the selection bias walk-forward testing exists to
prevent. The `+/- 0.15` is the finding; the 0.82 is a draw from it.

The stability ordering is the same lesson as M3's retention column, restated:
what estimates less, varies less. Equal weight moves 0.02 across all 24
configurations, SPY 0.00.

### Design decisions

- **SQLite tracking backend, not the file store.** MLflow now raises on
  `file://` — the filesystem backend is in maintenance mode. `mlflow.db` plus
  `mlartifacts/` is the documented local replacement and what `mlflow ui
  --backend-store-uri sqlite:///mlflow.db` expects. Discovered by hitting the
  exception, not by reading ahead.
- **`params_from()` sorts the universe** before stringifying, so two runs over
  the same tickers compare equal regardless of config ordering.
- **NaN metrics are dropped, not logged.** MLflow accepts NaN and it poisons
  sorting and comparison in the UI. `test_nan_metrics_are_skipped` pins this.
- **The comparison run logs `strategies_beating_benchmark`** — one integer that
  answers the project's actual question, visible in the run list without
  opening anything.
- **Sensitivity chart is one panel, filtered to ME/10bps.** Small multiples
  would compare every colour pair at once and only three slots clear that
  floor; a single line panel uses the adjacent pairlist, where five validate.
  Legend sits below the axes because the max-Sharpe line sweeps through every
  in-plot corner.

### New issue — P3-6

The sweep re-solves every optimisation for each cost level, but transaction cost
does not affect the weights at all — only the net return series. Three cost
levels therefore do three times the solver work for identical portfolios. The
24-configuration sweep takes ~15 minutes and could take ~5. Restructure so
weights are computed once per (window, rebalance) and cost variants are applied
afterwards.

### Verification

- 78/78 tests pass, offline, 15.6s
- Tracked run inspected via `mlflow.search_runs`: 7 runs, params and metrics present
- Full 24-configuration sweep run to completion; `--sweep` wiring separately
  smoke-tested on a 2-configuration grid after the chart was added
- Sensitivity chart inspected; legend collision found and fixed

### Still open

P2-2 (incremental ingest; **blocks M6**), P2-6 (lag spans gaps), P2-7
(no significance test — the sweep widens the *parameter* axis but all 24 runs
share the same 7.2 years, so they are correlated views of one history, not 24
trials), P2-8 (flat costs), P3-1/2/3, P3-6 (new, above), `report.py` untested.

**Next: M2 (dbt + incremental ingest) or M5 (forecasting).** M2 is the one with
a downstream dependency — M6's daily schedule cannot exist until ingest is
incremental.

---

## Known issues

Ranked by how much they distort results or block later milestones.
Severity: **P1** = wrong output or blocks a milestone · **P2** = real defect, contained ·
**P3** = hygiene.

### ~~P1-1~~ — RESOLVED session 2 · Silent history truncation

- **Where:** `transform.py:returns_matrix()`, the `wide.dropna(how="any")` line.
- **What:** Config requests `start="2016-01-01"`, but the returns matrix actually
  begins **2018-06-20** (verified: 2,064 rows). XLC (Communication Services) was only
  created in June 2018, and `dropna(how="any")` drops every date where *any* asset is
  missing — so one late-listing ETF truncates all eleven.
- **Why it matters:** ~30% of the requested history is silently gone, including the
  2016 energy selloff and the Feb-2018 volatility spike. Nothing logs or warns. Every
  return, covariance and Sharpe figure above is computed on a shorter window than
  intended.
- **Fix options:** (a) log the effective window and per-ticker first-date so the
  truncation is visible; (b) drop XLC from the universe to recover 2016–2018;
  (c) estimate the covariance pairwise on overlapping data; (d) accept it and set
  `start="2018-06-20"` honestly. **At minimum do (a) — silent is the real bug.**

### ~~P1-2~~ — RESOLVED session 2 · Log returns used where simple returns are required

- **Where:** `transform.py:build_gold()` produces `ln(close/prev_close)`;
  `optimize.py` then treats `w @ mu` as the portfolio return.
- **What:** Log returns are additive *across time*, not *across assets*. The log
  return of a portfolio is **not** the weighted sum of constituent log returns —
  `ln(Σ wᵢ·exp(rᵢ)) ≠ Σ wᵢ·rᵢ`. Mean-variance optimisation assumes simple returns.
- **Why it matters:** A small but systematic bias in expected return (Jensen's
  inequality; the gap grows with volatility, so it penalises the volatile sectors
  most — exactly the ones the optimiser is deciding between). Covariance is barely
  affected at daily frequency; the mean vector is.
- **Fix:** emit **both** `simple_return` and `log_return` in gold. Use simple returns
  for optimisation inputs, log returns for time-aggregation and cumulative charts.

### ~~P1-3~~ — RESOLVED session 4 · Max-Sharpe objective non-convex under SLSQP

- **Where:** `optimize.py:max_sharpe()` minimises `-sharpe` directly.
- **What:** The Sharpe ratio is a ratio of a linear to a quadratic form — not convex.
  SLSQP converges to a local optimum from the equal-weight start point.
- **Why it matters:** Long-only with a 35% cap is a benign region so results are
  probably fine, but there is **no guarantee**, and it is not reproducible under a
  different start point. Not defensible in an interview.
- **Fix:** the standard convex reformulation (scale variables by `y = w/κ`, solve a
  QP, renormalise) — or move to cvxpy at M3, which is already planned.

### ~~P1-4~~ — RESOLVED session 2 · Docstring claimed retry logic that did not exist

- **Where:** `ingest.py:YahooSource` docstring — *"that is why downloads are batched
  and retried"*.
- **What:** Verified by grep — **there is no retry, backoff or sleep anywhere in
  `src/`.** Batching is real; retrying is fiction.
- **Why it matters:** A comment that lies is worse than no comment. And the underlying
  need is genuine: yfinance rate-limits and fails intermittently, so a transient
  failure currently kills the whole pipeline run.
- **Fix:** implement it (`tenacity`, exponential backoff, 3–5 attempts, retry only on
  transient network/HTTP errors) — or delete the clause. Prefer implementing.

### ~~P2-1~~ — RESOLVED session 2 · SPY fetched and never used

- **Where:** `config.BENCHMARK`, ingested by `ingest()`, excluded by
  `run()` passing `cfg.tickers` to `returns_matrix()`.
- **What:** The benchmark is downloaded, stored in bronze and silver, and then dropped.
  No strategy is ever compared to it.
- **Why it matters:** "Did we beat just buying SPY?" is the *only* question that
  matters for this project, and it is currently unanswerable.
- **Fix:** add SPY as a fourth series on the frontier chart and a fourth row in
  `summary.csv`. It is a one-asset "portfolio" — nearly free to add.

### P2-2 — Ingest is idempotent but not incremental

- **Where:** `ingest.py:ingest()` — full-history download, single-file overwrite.
- **What:** Every run re-downloads 2016→today and rewrites the whole Parquet file.
  Bronze is described as "append-only and immutable" but is in fact replace-in-place.
- **Why it matters:** ~30s and 12 full-history API hits per run, gratuitous
  rate-limit exposure, and it **blocks the daily Dagster schedule at M6** — a daily
  job must fetch one day, not ten years. Also means bronze has no history: a bad
  Yahoo response silently overwrites good data with no way back.
- **Fix:** partition bronze by year (`bronze/year=2024/prices.parquet`), read the
  existing max date per ticker, fetch only from there forward, and merge.

### ~~P2-3~~ — RESOLVED session 4 · Sample covariance, unshrunk

- **Where:** `transform.py:annualise()` — plain `daily.cov()`.
- **What:** 11 assets → 66 covariance parameters from 2,064 observations. Workable,
  but the sample estimator is still noisy and min-variance is notoriously sensitive
  to it (it loads onto whichever asset's variance happens to be *underestimated*).
- **Fix:** Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`) at M3. Already on
  the roadmap; noting it here as a known accuracy limitation, not a surprise.

### ~~P2-4~~ — PARTIALLY RESOLVED session 2 · `transform.py` now covered; `report.py` still not

- **What:** 9 tests cover `ingest.py` and `optimize.py` only. The two modules that
  own data correctness (`transform`) and output (`report`) are untested. Note that
  **P1-1 and P1-2 both live in the untested module** — that is not a coincidence.
- **Fix:** tests over a small synthetic Parquet fixture: known prices → known returns;
  assert the pandera gate rejects a negative close and a duplicate `(date, ticker)`;
  assert `returns_matrix` alignment behaviour explicitly (which would have caught P1-1).

### ~~P2-5~~ — RESOLVED session 4 · `equal_weight()` ignored `cfg.max_weight`

- **Where:** `optimize.py:equal_weight()` hardcodes `1/n`.
- **What:** With 11 assets, `1/11 = 9.1% < 35%`, so it is *currently* harmless — but
  it is a silent constraint violation waiting for a smaller universe.
- **Fix:** assert `1/n <= cfg.max_weight`, or clip and renormalise.

### P3-1 — yfinance TzCache warning on every run

- **What:** `Failed to create TzCache ... [Errno 17] File exists`, logged twice per run.
- **Why it matters:** Cosmetic, but it trains you to ignore warnings — and it means
  timezone lookups are uncached, so ingestion is slower than it needs to be.
- **Fix:** `yf.set_tz_cache_location()` pointed at a project-local path.

### P3-2 — SQL built by f-string path interpolation

- **Where:** both DuckDB queries in `transform.py`.
- **What:** Not an injection risk (paths are internal constants), but it breaks on any
  path containing a quote and is a habit worth not forming.
- **Fix:** pass paths as bound parameters, or `con.register()` the DataFrame.

### P3-3 — Hardcoded risk-free rate

- **Where:** `Settings.risk_free_rate = 0.02`.
- **What:** A flat 2% across 2018–2026 — a period spanning ZIRP and 5%+ policy rates.
  Every Sharpe figure above is therefore only loosely meaningful.
- **Fix:** pull `^IRX` (13-week T-bill) from the same source and use the realised
  time series.

### ~~P3-4~~ — RESOLVED session 2 · Pre-commit hooks configured but never installed

- **What:** `.pre-commit-config.yaml` exists; `.git/hooks/pre-commit` does **not**.
  Verified. The hooks have never actually run — CI is the only gate.
- **Fix:** `uv run pre-commit install` (one command, do it first thing next session).

### P3-5 — Smaller items

- `_to_tidy()`'s single-ticker branch is untested (only the MultiIndex path is exercised).
- No test asserts that gold return counts equal silver rows minus one per ticker.
- CI has no coverage reporting and no dependency caching beyond `setup-uv`.
- No Dependabot / renovate config.
- No `CONTRIBUTING.md` or docstring style convention declared.
- Dependency floors are `>=` only; `uv.lock` is the real pin (this is fine, but know it).

---

## Next steps

### ~~Immediate — session 2 opener~~ — DONE, see Session 2 above

All six items shipped on `fix/data-correctness`. Remaining quick wins, none blocking:
P2-5 (`equal_weight` cap assertion), P3-1 (TzCache location), P3-2 (bound SQL
params), P3-3 (real risk-free series), plus `report.py` test coverage.

### M2 — Analytics engineering (dbt over DuckDB)

- `dbt-duckdb`; port the two SQL blocks into `models/silver/` and `models/gold/`.
- Incremental materialisation for the price model — pairs with fixing P2-2.
- dbt tests: `unique` + `not_null` on `(date, ticker)`, `accepted_range` on close,
  a custom test for calendar gaps > 4 business days.
- Source freshness so a stale Yahoo feed fails loudly.
- **Decision to make:** does pandera stay for in-Python contracts while dbt owns
  in-warehouse tests, or does dbt replace it? Recommendation: keep both, different jobs.

### M3 — Data science (the milestone that makes results defensible)

- Ledoit-Wolf shrinkage covariance — P2-3.
- Move to **cvxpy**, resolving P1-3 properly, and unlock: turnover constraints,
  transaction costs, sector caps, cardinality.
- Add **risk parity** (equal risk contribution) and **max diversification**.
- **Walk-forward backtest** — the core deliverable: rolling 3-year estimation window,
  monthly rebalance, out-of-sample. Report CAGR, vol, Sharpe, max drawdown, Calmar,
  turnover, and cost-adjusted return **vs equal-weight and vs SPY**.
- Expect a humbling result: out-of-sample, max-Sharpe frequently *loses* to equal
  weight. That finding is the project's most valuable teaching moment — document it,
  do not hide it.

### M4 — Experiment tracking (MLflow)

- Every backtest = one run. Params from the `Settings` dataclass (this is why it is
  frozen). Metrics = the backtest table. Artifacts = charts + weights CSV.
- Tag by strategy and estimation window; compare runs in the MLflow UI.

### M5 — ML forecasting

- Features: momentum (1/3/6/12m), realised vol, cross-sectional rank, macro if desired.
- Model: Ridge, then gradient boosting. Target: next-month sector return.
- **Purged, embargoed walk-forward CV** — standard k-fold leaks in time series.
- The honest expected outcome is near-zero alpha. The deliverable is *the method
  proving it*: information coefficient, hit rate, and a comparison against shuffled
  labels.

### M6 — Orchestration (Dagster)

- Wrap M1–M4 as software-defined assets; `dagster-dbt` for the M2 models.
- Daily schedule (requires incremental ingest — P2-2 is the blocker).
- Asset checks for freshness and row-count deltas; local Dagster UI in the README.

### M7 — Serving

- FastAPI `POST /optimize` taking tickers, window, constraints; returning weights + stats.
- Streamlit dashboard: universe picker, frontier, weights, backtest equity curve.
- Dockerfile (**Docker is not installed on this machine** — install or use a cloud runner).

### M8 — Production MLOps

- Evidently drift reports on the feature distribution.
- Alert on schema drift, staleness, and outlier returns.
- Model registry with a promotion gate: a challenger ships only on out-of-sample
  Sharpe improvement.

---

## Open questions

1. **Truncation (P1-1):** drop XLC to recover 2016–2018, or keep 11 sectors and accept
   a 2018 start? *(Recommendation: keep XLC, start 2018, log it loudly. A complete
   sector map is worth more than 2.5 years.)*
2. **Rebalance frequency at M3:** monthly or quarterly? *(Monthly — more observations,
   and it makes transaction costs bite, which is realistic.)*
3. **Docker at M7:** install locally or defer to a cloud runner?
4. **Scope of M5:** is a genuine forecasting model wanted, or is the walk-forward
   backtest infrastructure the real learning target? This changes how much time M5 deserves.

---

## Environment reference

```bash
cd ~/Documents/projects/portfolio_optimization
source .venv/bin/activate     # or prefix everything with `uv run`

uv sync                       # install from lockfile
uv run portfolio              # full pipeline: ingest -> transform -> optimise -> report
uv run portfolio --skip-ingest  # reuse cached bronze
uv run portfolio --start 2020-01-01
uv run pytest                 # 9 tests, fully offline
uv run ruff check . && uv run ruff format .
```

- System Python 3.13.1; **project pinned to 3.12** via `.python-version`.
- `git`, `gh`, `uv` installed. **`docker` and `poetry` are not.**
- `data/**/*.parquet` and `reports/*` are gitignored — the lake is regenerable.
- Note: `source .venv/bin/activate` does not persist across separate agent shell
  invocations; `uv run` is used instead in automation.
