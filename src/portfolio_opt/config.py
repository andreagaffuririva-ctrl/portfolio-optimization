"""Central configuration. Keep every tunable here so runs are reproducible."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"  # raw, immutable, as-pulled
SILVER_DIR = DATA_DIR / "silver"  # cleaned, validated, tidy
GOLD_DIR = DATA_DIR / "gold"  # analysis-ready aggregates
REPORTS_DIR = PROJECT_ROOT / "reports"

# The 11 S&P 500 sector SPDR ETFs, plus SPY as the benchmark.
SECTOR_ETFS: tuple[str, ...] = (
    "XLB",  # Materials
    "XLC",  # Communication Services
    "XLE",  # Energy
    "XLF",  # Financials
    "XLI",  # Industrials
    "XLK",  # Technology
    "XLP",  # Consumer Staples
    "XLRE",  # Real Estate
    "XLU",  # Utilities
    "XLV",  # Health Care
    "XLY",  # Consumer Discretionary
)
BENCHMARK = "SPY"

TRADING_DAYS = 252


@dataclass(frozen=True)
class Settings:
    tickers: tuple[str, ...] = SECTOR_ETFS
    benchmark: str = BENCHMARK
    start: str = "2016-01-01"
    end: str | None = None  # None -> today
    risk_free_rate: float = 0.02  # annualised, used for Sharpe
    max_weight: float = 0.35  # concentration cap per asset
    allow_short: bool = False


settings = Settings()


def ensure_dirs() -> None:
    for d in (BRONZE_DIR, SILVER_DIR, GOLD_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
