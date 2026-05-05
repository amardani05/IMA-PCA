"""Macro factor data loader (FRED + yfinance).

Loads a curated set of macro factors, applies stationarity transforms (level
changes for yields/spreads/VIX, log returns for prices), aligns them on
business-day frequency, and caches each series to ``data/macro/`` so the API is
only hit once per day.

Required: a ``FRED_API_KEY`` in the environment or in a ``.env`` file at the
repo root. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html.

Public entry point: :func:`load_all_macro_factors`.
"""

from __future__ import annotations

import logging
import os
from dotenv import load_dotenv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import config

# Load .env from project root if present.
try:
    from dotenv import load_dotenv
    load_dotenv(config.PROJECT_ROOT / ".env")
except ImportError:
    # python-dotenv is in requirements.txt; if it's missing we just rely on
    # the calling shell having FRED_API_KEY exported.
    pass

logger = logging.getLogger(__name__)

MACRO_CACHE_DIR: Path = config.DATA_DIR / "macro"
MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Factor definitions
# =============================================================================
MACRO_FACTORS: dict[str, dict[str, dict]] = {
    "rates": {
        "DGS10":  {"name": "10Y Treasury Yield",   "source": "fred",     "transform": "level_change"},
        "DGS2":   {"name": "2Y Treasury Yield",    "source": "fred",     "transform": "level_change"},
        "T10Y2Y": {"name": "10Y-2Y Spread",        "source": "fred",     "transform": "level_change"},
        "DGS30":  {"name": "30Y Treasury Yield",   "source": "fred",     "transform": "level_change"},
    },
    "credit": {
        "BAMLH0A0HYM2": {"name": "HY Credit Spread (OAS)", "source": "fred", "transform": "level_change"},
        "BAMLC0A0CM":   {"name": "IG Credit Spread (OAS)", "source": "fred", "transform": "level_change"},
        "TEDRATE":      {"name": "TED Spread",             "source": "fred", "transform": "level_change"},
    },
    "inflation": {
        "T5YIE":  {"name": "5Y Breakeven Inflation",  "source": "fred", "transform": "level_change"},
        "T10YIE": {"name": "10Y Breakeven Inflation", "source": "fred", "transform": "level_change"},
        "T5YIFR": {"name": "5Y/5Y Forward Inflation", "source": "fred", "transform": "level_change"},
    },
    "commodities": {
        "DCOILWTICO":   {"name": "WTI Crude Oil",      "source": "fred",     "transform": "log_return"},
        "DCOILBRENTEU": {"name": "Brent Crude",        "source": "fred",     "transform": "log_return"},
        "DHHNGSP":      {"name": "Henry Hub Nat Gas",  "source": "fred",     "transform": "log_return"},
        "GC=F":         {"name": "Gold",               "source": "yfinance", "transform": "log_return"},
        "HG=F":         {"name": "Copper",             "source": "yfinance", "transform": "log_return"},
        "SI=F":         {"name": "Silver",             "source": "yfinance", "transform": "log_return"},
    },
    "currency": {
        # DTWEXBGS (Broad index) contaminated the daily regression with EM
        # weekly-update noise. AFEGS covers only liquid majors (EUR, JPY,
        # GBP, CHF, AUD, CAD, SEK) and updates daily.
        "DTWEXAFEGS": {"name": "Trade-Weighted USD (Major)", "source": "fred", "transform": "log_return"},
        # ICE Dollar Index — committee-recognized, EUR/JPY-heavy. Pulled from
        # yfinance under DX-Y.NYB but exposed as "DXY" in MACRO_FACTORS so
        # the column name is the obvious one.
        "DXY":        {"name": "DXY (ICE Dollar Index)", "source": "yfinance",
                       "yf_ticker": "DX-Y.NYB", "transform": "log_return"},
        "DEXCHUS":    {"name": "USD/CNY", "source": "fred", "transform": "log_return"},
    },
    "volatility_liquidity": {
        "VIXCLS": {"name": "VIX",                   "source": "fred",     "transform": "level_change"},
        "^MOVE":  {"name": "MOVE Index (rate vol)", "source": "yfinance", "transform": "level_change"},
    },
    "financial_conditions": {
        "NFCI":     {"name": "Chicago Fed Financial Conditions",     "source": "fred", "transform": "level_change"},
        "ANFCI":    {"name": "Adj Chicago Fed Financial Conditions", "source": "fred", "transform": "level_change"},
        "STLFSI4":  {"name": "St. Louis Fed Financial Stress",       "source": "fred", "transform": "level_change"},
    },
    "thematic": {
        "SMH":  {"name": "Semiconductor ETF",  "source": "yfinance", "transform": "log_return"},
        "ITA":  {"name": "Defense ETF",        "source": "yfinance", "transform": "log_return"},
        "URA":  {"name": "Uranium ETF",        "source": "yfinance", "transform": "log_return"},
        "KWEB": {"name": "China Internet ETF", "source": "yfinance", "transform": "log_return"},
        "TAN":  {"name": "Solar ETF",          "source": "yfinance", "transform": "log_return"},
        "DRIV": {"name": "EV/Autonomous ETF",  "source": "yfinance", "transform": "log_return"},
    },
    "data_center_proxies": {
        "VST":  {"name": "Vistra (power gen)",        "source": "yfinance", "transform": "log_return"},
        "CEG":  {"name": "Constellation Energy",      "source": "yfinance", "transform": "log_return"},
        "NRG":  {"name": "NRG Energy",                "source": "yfinance", "transform": "log_return"},
        "EQIX": {"name": "Equinix (data center REIT)","source": "yfinance", "transform": "log_return"},
        "DLR":  {"name": "Digital Realty",            "source": "yfinance", "transform": "log_return"},
    },
}

# Curated set: one representative per category to control multicollinearity.
# FRED series we know are published weekly (Wednesday) — ffill them across
# the week so they don't blow out daily regressions via row-drop. T5YIE etc
# are daily but occasionally have gaps; for those the default 1-day ffill
# is correct.
WEEKLY_PUBLISHED_SERIES: set[str] = {"NFCI", "ANFCI", "STLFSI4"}


CURATED_FACTORS: list[str] = [
    # Curated set is what we estimate exposure TO. The v2 control set
    # (IJR-excess, VIX, HY OAS) is what we control FOR — including a series
    # in both lists creates degenerate residuals (residualizing VIX against
    # VIX leaves zero variance), so VIX and HY OAS now live exclusively in
    # the control set. Their portfolio exposures still surface via
    # ``control_betas`` on the v2 regression output.
    "rates_T10Y2Y",
    "inflation_T5YIE",
    "commodities_DCOILWTICO",
    "commodities_GC=F",
    "commodities_HG=F",
    # DXY (ICE Dollar Index) preferred for committee recognition; the
    # broader DTWEXAFEGS is still loaded and available in the "All" toggle.
    "currency_DXY",
    "financial_conditions_NFCI",
]

# Default scenario shocks (raw units; what the column-level macro series moves by).
# For level_change series this is in their native units (% for yields/spreads, points for VIX).
# For log_return series this is the proportional move (e.g., -0.10 = -10%).
SCENARIO_SHOCKS: dict[str, tuple[str, float]] = {
    "rates_DGS10":              ("+50bp", 0.50),
    "rates_T10Y2Y":             ("+25bp steepening", 0.25),
    "credit_BAMLH0A0HYM2":      ("+100bp HY widening", 1.00),
    "credit_BAMLC0A0CM":        ("+50bp IG widening", 0.50),
    "inflation_T5YIE":          ("+25bp", 0.25),
    "commodities_DCOILWTICO":   ("-10% oil", -0.10),
    "commodities_DCOILBRENTEU": ("-10% Brent", -0.10),
    "commodities_GC=F":         ("+5% gold", 0.05),
    "commodities_HG=F":         ("-10% copper", -0.10),
    "currency_DTWEXAFEGS":      ("+5% USD (Major)", 0.05),
    "currency_DXY":             ("+5% DXY", 0.05),
    "volatility_liquidity_VIXCLS":  ("+10pt VIX", 10.0),
    "financial_conditions_NFCI":    ("+0.5σ tightening", 0.5),
}


# =============================================================================
# FRED client
# =============================================================================
def _fred_api_key(required: bool = True) -> str | None:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    if not required:
        return None
    raise EnvironmentError(
        "FRED_API_KEY is not set. Add it to a `.env` file at the repo root or "
        "export it in your shell:\n"
        "  export FRED_API_KEY=your_key_here\n"
        "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
    )


@dataclass
class FredClient:
    api_key: str | None = None

    def __post_init__(self):
        from fredapi import Fred
        self.api_key = self.api_key or _fred_api_key(required=True)
        self._fred = Fred(api_key=self.api_key)

    def get_series(self, series_id: str, start: str, end: str | None = None) -> pd.Series:
        """Fetch a FRED series. Returns a pd.Series indexed by date."""
        s = self._fred.get_series(series_id, observation_start=start, observation_end=end)
        s.name = series_id
        s.index = pd.to_datetime(s.index)
        return s.dropna()


# =============================================================================
# Caching
# =============================================================================
def _cache_path(series_id: str) -> Path:
    safe = series_id.replace("/", "_").replace("=", "_eq_").replace("^", "carat_")
    return MACRO_CACHE_DIR / f"{safe}.parquet"


def _cache_is_fresh(path: Path, max_age_hours: int = 24) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def _load_from_cache(series_id: str) -> pd.Series | None:
    p = _cache_path(series_id)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    s = df.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s


def _save_to_cache(series_id: str, s: pd.Series) -> None:
    p = _cache_path(series_id)
    s.to_frame().to_parquet(p)


# =============================================================================
# Per-source fetchers
# =============================================================================
def _fetch_yfinance(series_id: str, start: str, end: str | None = None) -> pd.Series:
    """Pull a single yfinance series. Returns adjusted close."""
    end_ts = end or datetime.utcnow().date().isoformat()
    df = yf.download(
        series_id,
        start=start,
        end=end_ts,
        auto_adjust=True,
        progress=False,
        ignore_tz=True,
        multi_level_index=False,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float, name=series_id)
    if "Close" in df.columns:
        s = df["Close"].copy()
    else:
        # Single-ticker may collapse columns
        s = df.iloc[:, 0].copy()
    s.name = series_id
    s.index = pd.to_datetime(s.index)
    return s.dropna()


def _fetch_one(
    series_id: str,
    source: str,
    start: str,
    end: str | None,
    fred: FredClient | None,
    yf_ticker: str | None = None,
) -> pd.Series:
    """Fetch one series. ``yf_ticker`` overrides ``series_id`` for yfinance
    when the public-friendly key differs from the yfinance ticker (e.g. our
    DXY entry maps to ``DX-Y.NYB``)."""
    if source == "fred":
        if fred is None:
            raise RuntimeError("FredClient not initialized")
        return fred.get_series(series_id, start, end)
    if source == "yfinance":
        s = _fetch_yfinance(yf_ticker or series_id, start, end)
        # Always tag the series with the canonical key so downstream code
        # doesn't see DX-Y.NYB anywhere.
        if not s.empty:
            s.name = series_id
        return s
    raise ValueError(f"unknown source: {source}")


# =============================================================================
# Transforms
# =============================================================================
def apply_transform(s: pd.Series, transform: str) -> pd.Series:
    """Make the series stationary."""
    if transform == "level_change":
        return s.diff()
    if transform == "log_return":
        # Filter out non-positive values before log
        s_pos = s.where(s > 0)
        return np.log(s_pos / s_pos.shift(1))
    if transform == "pct_change":
        return s.pct_change()
    raise ValueError(f"unknown transform: {transform}")


# =============================================================================
# Public API
# =============================================================================
def load_all_macro_factors(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    force_refresh: bool = False,
    require_fred_key: bool = True,
) -> pd.DataFrame:
    """Load every defined macro factor, transform, and align on business days.

    Returns
    -------
    DataFrame indexed by date with columns named ``"<category>_<series_id>"``,
    each containing the stationarity-transformed factor.
    """
    if isinstance(start, pd.Timestamp):
        start = start.date().isoformat()
    if end is not None and isinstance(end, pd.Timestamp):
        end = end.date().isoformat()

    fred: FredClient | None = None
    needs_fred = any(
        defn["source"] == "fred"
        for cat in MACRO_FACTORS.values()
        for defn in cat.values()
    )
    if needs_fred and (require_fred_key or _fred_api_key(required=False)):
        try:
            fred = FredClient()
        except EnvironmentError:
            if require_fred_key:
                raise
            logger.warning("FRED_API_KEY unavailable; FRED-sourced factors will be skipped")

    raw_series: dict[str, pd.Series] = {}
    transforms: dict[str, str] = {}
    n_ok = n_skip = 0

    for category, defs in MACRO_FACTORS.items():
        for series_id, defn in defs.items():
            colname = f"{category}_{series_id}"
            cache_path = _cache_path(series_id)

            if not force_refresh and _cache_is_fresh(cache_path):
                s = _load_from_cache(series_id)
                if s is not None and not s.empty:
                    s = s.loc[start:end] if end else s.loc[start:]
                    raw_series[colname] = s
                    transforms[colname] = defn["transform"]
                    n_ok += 1
                    continue

            try:
                s = _fetch_one(series_id, defn["source"], start, end, fred,
                               yf_ticker=defn.get("yf_ticker"))
                if s.empty:
                    raise ValueError("empty response")
                _save_to_cache(series_id, s)
                raw_series[colname] = s
                transforms[colname] = defn["transform"]
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s (%s): %s", colname, defn["source"], exc)
                n_skip += 1

    if not raw_series:
        raise RuntimeError("No macro factors loaded; check FRED_API_KEY and network")

    # Build a single business-day-indexed DataFrame.
    # Default ffill limit is 1 so we only bridge true single-day holiday gaps;
    # filling levels for several days then computing returns manufactures one
    # big jump followed by zeros, which biased the regression. The exception
    # is series we KNOW are published weekly — NFCI/ANFCI/STLFSI4 from the
    # Chicago / St. Louis Fed are released on a single weekday — for those
    # we hold the value across the week (ffill limit=7).
    bdays = pd.bdate_range(start=start, end=end or pd.Timestamp.utcnow().date())
    panel = pd.DataFrame(index=bdays)
    for colname, s in raw_series.items():
        # colname is "{category}_{series_id}" but category itself may contain
        # underscores (e.g. "financial_conditions"). Use rsplit to peel off
        # only the trailing series_id.
        series_id = colname.rsplit("_", 1)[1] if "_" in colname else colname
        ffill_limit = 7 if series_id in WEEKLY_PUBLISHED_SERIES else 1
        panel[colname] = s.reindex(bdays).ffill(limit=ffill_limit)

    # Apply transforms (level_change for I(1) series, log_return for prices)
    transformed = pd.DataFrame(index=panel.index)
    for col, t in transforms.items():
        transformed[col] = apply_transform(panel[col], t)

    # Drop any row missing ANY curated factor — better to discard the day
    # than to feed an incomplete row into the regression.
    n_pre = len(transformed)
    curated_present = [c for c in CURATED_FACTORS if c in transformed.columns]
    if curated_present:
        transformed = transformed.dropna(subset=curated_present, how="any")
    else:
        transformed = transformed.dropna(how="all")
    n_dropped = n_pre - len(transformed)
    drop_pct = 100.0 * n_dropped / max(n_pre, 1)

    if drop_pct > 5.0:
        logger.warning(
            "macro factors: dropped %d/%d days (%.1f%%) with missing curated data — "
            "consider widening lookback or replacing the noisy series",
            n_dropped, n_pre, drop_pct,
        )
    else:
        logger.info(
            "macro factors: dropped %d/%d days (%.1f%%) with missing curated data",
            n_dropped, n_pre, drop_pct,
        )

    logger.info(
        "macro factors: %d loaded (%d skipped), %d trading days kept",
        n_ok, n_skip, len(transformed),
    )
    return transformed


# =============================================================================
# Validation
# =============================================================================
def validate_macro_data(factors: pd.DataFrame) -> pd.DataFrame:
    """Per-factor coverage / sanity stats. Logs warnings; returns the report."""
    rows = []
    for col in factors.columns:
        s = factors[col]
        valid = int(s.notna().sum())
        pct_missing = 100.0 * (1.0 - valid / max(len(factors), 1))
        std = float(s.std(ddof=1)) if valid > 1 else 0.0
        mean = float(s.mean()) if valid else 0.0
        z = (s - mean) / std if std > 0 else pd.Series(0.0, index=s.index)
        n_outliers = int((z.abs() > 10).sum())

        warnings = []
        if pct_missing > 10.0:
            warnings.append(f"{pct_missing:.1f}% missing")
        if std == 0:
            warnings.append("zero variance — broken")
        if n_outliers > 0:
            warnings.append(f"{n_outliers} extreme outliers (|z|>10)")

        rows.append({
            "factor": col,
            "n_valid": valid,
            "pct_missing": round(pct_missing, 2),
            "mean": round(mean, 6),
            "std": round(std, 6),
            "n_outliers": n_outliers,
            "warnings": "; ".join(warnings),
        })

        if warnings:
            logger.warning("macro factor %s: %s", col, "; ".join(warnings))

    return pd.DataFrame(rows)
