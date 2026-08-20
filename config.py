"""Configuration for the IMA drawdown-risk screener.

Holds the current IMA portfolio, the feature taxonomy for the risk model,
risk-direction metadata used to rank clusters, PCA/clustering hyperparameters,
and data-IO paths.

DROPPED FEATURES (2026-05-03 cleanup):

- ``piotroski_f`` — yfinance provides only ~4 quarters of fundamental data,
  not the 8 needed for YoY comparisons. The score was always imputed to the
  cross-sectional median; leave-one-out ARI = 1.000 (zero impact on
  clustering). To restore: a richer fundamental data source is required.

- ``earnings_drag`` — ``yfinance.earnings_dates`` returned no historical
  data for the universe (0% coverage). Identical situation: median-imputed
  for everyone, leave-one-out ARI = 1.000.

- ``insider_net_sell_ratio`` — the v1 implementation just counted Form-4
  filings without parsing the underlying XML for transaction codes
  (P/S/A/M). The proxy was too coarse to load on any PC (max corr 0.084)
  and leave-one-out ARI = 0.989. To restore: build a proper Form-4 XML
  parser that distinguishes opens-market purchases from grants/exercises.

To restore any of these, fix the data source first, then add back to
``FEATURES`` and ``RISK_DIRECTION``.
"""

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Current IMA portfolio (long-only S&P 600 sleeve)
# -----------------------------------------------------------------------------
# Updated 2026-03-24 from "IMA Portfolio.csv" — Allocation of Total column,
# 20 positions, ex-cash (3.06% FEDXX cash buffer not modeled here).
PORTFOLIO: dict[str, float] = {
    "TDS":  0.0368, "PRDO": 0.0397, "MCRI": 0.0491, "CVCO": 0.0397,
    "UNFI": 0.0212, "CRGY": 0.0541, "AX":   0.0677, "PFBC": 0.0523,
    "ENVA": 0.0653, "NEOG": 0.0439, "KRYS": 0.0526, "FSS":  0.0578,
    "GTES": 0.0525, "MYRG": 0.0711, "DOCN": 0.0566, "KLIC": 0.0385,
    "VIAV": 0.0424, "MTRN": 0.0586, "CTRE": 0.0550, "AVA":  0.0142,
}

BENCHMARK_TICKER: str = "IJR"  # iShares S&P Small-Cap 600 ETF

# -----------------------------------------------------------------------------
# Feature taxonomy
# Each feature is z-scored before PCA, so raw scale doesn't matter.
# -----------------------------------------------------------------------------
FEATURES: list[str] = [
    # Financial health
    "altman_z",
    "current_ratio",
    "net_debt_to_ebitda",
    "fcf_yield",
    "interest_coverage",
    # Quality + capital allocation
    "accruals_ratio",       # OCF / NI TTM (Sloan accruals)
    "asset_growth_yoy",     # YoY total assets (CGS asset-growth)
    "net_issuance_yoy",     # YoY shares outstanding (Daniel-Titman dilution)
    # Market positioning
    "short_pct_float",
    "momentum_30d",
    "momentum_90d",
    "volatility_60d",
    "relative_volume",
    # Valuation stress
    "pe_ratio",
    "ev_to_ebitda",
    # Event risk proxy
    "filing_count_90d",
]

# Risk direction: 1 = higher value means MORE risk, -1 = higher value means LESS risk.
RISK_DIRECTION: dict[str, int] = {
    "altman_z": -1,
    "current_ratio": -1,
    "net_debt_to_ebitda": 1,
    "fcf_yield": -1,
    "interest_coverage": -1,
    "accruals_ratio": -1,
    "asset_growth_yoy": 1,
    "net_issuance_yoy": 1,
    "short_pct_float": 1,
    "momentum_30d": -1,
    "momentum_90d": -1,
    "volatility_60d": 1,
    "relative_volume": 1,
    "pe_ratio": 1,
    "ev_to_ebitda": 1,
    "filing_count_90d": 1,
}

# Subset of features used on the portfolio-dashboard radar/bar charts.
# Replaced piotroski_f with asset_growth_yoy (the most informative remaining
# fundamental quality signal — leave-one-out ARI was the most impactful drop).
DASHBOARD_FEATURES: list[str] = [
    "altman_z",
    "asset_growth_yoy",
    "short_pct_float",
    "momentum_90d",
    "net_debt_to_ebitda",
    "volatility_60d",
]

# -----------------------------------------------------------------------------
# PCA + clustering hyperparameters
# -----------------------------------------------------------------------------
N_PCA_COMPONENTS: int = 4
N_CLUSTERS: int = 3        # was 4 — k=3 silhouette 0.334 vs k=4 at 0.361 (within tolerance);
                            # k=3 eliminates the negative-silhouette borderline cluster
K_CANDIDATES: tuple[int, ...] = (3, 4, 5, 6, 7)
KMEANS_N_INIT: int = 50
SILHOUETTE_TOLERANCE: float = 0.05  # widened so k=3 is preferred over noisy k=4 wins

# Risk tiers are cut on the composite score's PERCENTILE within the scored
# cross-section, not on the raw composite. The raw composite (a mean of 16
# percentile ranks) concentrates near 50 by construction, so fixed raw-score
# buckets left ~97% of the universe in one bucket (2026-08 review, finding A2).
# Percentile buckets guarantee a calibrated 20 / 60 / 20 split by design.
# Labels are relative to the *current universe* — not absolute risk ratings.
TIER_LABELS: list[str] = ["Low Risk", "In Line", "Elevated"]

TIER_PERCENTILE_BUCKETS: list[tuple[float, float, str]] = [
    (0, 20, "Low Risk"),
    (20, 80, "In Line"),
    (80, 100.01, "Elevated"),
]

# k-means clusters are DESCRIPTIVE style groupings, not a risk ranking; they
# get style names ("Core", "High-Multiple Growth", ...) generated from each
# cluster's feature signature (see pca_cluster._style_clusters). Chart colors
# are assigned from this palette in cluster-risk-rank order (safest first).
CLUSTER_STYLE_PALETTE: list[str] = [
    "#4a7fb5",  # steel blue
    "#8a63bf",  # violet
    "#c05621",  # burnt orange
    "#8c1d18",  # oxblood
    "#5f6b76",  # slate (overflow)
    "#2c7a4b",  # green (overflow)
    "#946800",  # ochre (overflow)
]

# -----------------------------------------------------------------------------
# Trajectory settings
# -----------------------------------------------------------------------------
TRAJECTORY_QUARTERS: int = 4
TRAJECTORY_LOOKBACK_YEARS: int = 3        # extended price history for backfilled momentum/vol
DRIFT_SIGMA_THRESHOLD: float = 1.5        # 2-quarter PC drift above this = flagged
CLUSTER_BOUNDARY_RADIUS: float = 0.5      # borderline flag in PC space

# -----------------------------------------------------------------------------
# Data-IO settings
# -----------------------------------------------------------------------------
BATCH_SIZE: int = 20
BATCH_DELAY_SECONDS: int = 3
PRICE_LOOKBACK_DAYS: int = 252            # ~1 trading year for current features
FUNDAMENTALS_CACHE_MAX_AGE_DAYS: int = 7

# Sectors treated as "financial" (Altman Z not meaningful)
FINANCIAL_SECTORS: set[str] = {"Financials", "Financial Services", "Real Estate"}

# -----------------------------------------------------------------------------
# SEC EDGAR settings
# -----------------------------------------------------------------------------
SEC_USER_AGENT: str = "IMA-Risk-Screener/1.0 (Student research; dani@navaslabs.com)"
SEC_RATE_LIMIT_PER_SECOND: int = 8        # stay under 10/sec cap with margin
SEC_TICKER_MAP_URL: str = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL: str = "https://data.sec.gov/submissions/CIK{cik}.json"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache files
SP600_CONSTITUENTS_CSV: Path = DATA_DIR / "sp600_constituents.csv"
SP600_FALLBACK_CSV: Path = DATA_DIR / "sp600_fallback.csv"
FUNDAMENTALS_CACHE: Path = DATA_DIR / "fundamentals_cache.parquet"
PRICE_CACHE: Path = DATA_DIR / "price_cache.parquet"
FEATURES_CACHE: Path = DATA_DIR / "features_cache.parquet"
SEC_CIK_CACHE: Path = DATA_DIR / "sec_cik_map.json"
SEC_FILINGS_CACHE: Path = DATA_DIR / "sec_filings_cache.parquet"

CACHE_MAX_AGE_SECONDS: int = 24 * 60 * 60
SP600_WIKI_URL: str = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"

USER_AGENT: str = (
    "IMA-Risk-Screener/1.0 (Student-run equity fund; "
    "contact: dani@navaslabs.com)"
)

# -----------------------------------------------------------------------------
# Backtest / severe-drawdown label settings
# -----------------------------------------------------------------------------
# THE SEVERE-DRAWDOWN LABEL.  A name is a "severe-drawdown event" if, over the
# forward horizon, it suffers a peak-to-trough drawdown (measured from the
# snapshot date forward) at or beyond DD_THRESHOLD.  Both constants are the
# knobs the whole evaluation hangs on, so they live here and nowhere else.
DD_HORIZON_MONTHS: int = 6               # forward window for the primary label
DD_THRESHOLD: float = 0.25               # >= 25% forward max drawdown == event

# Continuous forward-return horizons stored alongside the binary label (months).
FORWARD_RETURN_HORIZONS_MONTHS: tuple[int, ...] = (1, 3, 6, 12)

# Rebalance cadence for the backtest panel: "M" (month-end) or "Q" (quarter-end).
BACKTEST_REBALANCE: str = "Q"

# Strategy backtest assumptions (be explicit — these are reported in the webapp).
BACKTEST_COST_BPS: float = 10.0          # one-way transaction cost, basis points
BACKTEST_TOP_TIER: str = "Elevated"      # tier the long-only sleeve AVOIDS / shorts
BACKTEST_BOTTOM_TIER: str = "Low Risk"   # tier the long/short sleeve goes long

# Cells (decile / tier) with fewer than this many realized events are flagged
# as thin-sample and their hit-rates are reported but de-emphasized.
MIN_EVENTS_PER_CELL: int = 5

# Trading days per calendar month — used to convert horizon months to a price
# window and to annualize the strategy backtest.
TRADING_DAYS_PER_MONTH: int = 21
TRADING_DAYS_PER_YEAR: int = 252

# -----------------------------------------------------------------------------
# Point-in-time historical store (see historical_loader.py for the schema).
# Populate these parquet files to get deep-history backtests; absent files
# trigger loud survivorship / coverage warnings rather than silent fallbacks.
# -----------------------------------------------------------------------------
HISTORICAL_DIR: Path = DATA_DIR / "historical"
HIST_FEATURES_PARQUET: Path = HISTORICAL_DIR / "features.parquet"
HIST_UNIVERSE_PARQUET: Path = HISTORICAL_DIR / "universe.parquet"
HIST_IMA_HOLDINGS_PARQUET: Path = HISTORICAL_DIR / "ima_holdings.parquet"
HIST_PRICES_PARQUET: Path = HISTORICAL_DIR / "prices.parquet"
HIST_DELIST_PARQUET: Path = HISTORICAL_DIR / "delistings.parquet"  # optional

# Backtest artifacts
BACKTEST_DIR: Path = OUTPUT_DIR / "backtest"
BACKTEST_PANEL_PARQUET: Path = BACKTEST_DIR / "panel.parquet"
