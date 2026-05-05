"""Compute all torpedo-risk features for every S&P 600 stock.

Produces, for each ticker, the 14 features listed in ``config.FEATURES`` plus
company/sector metadata. Features are built from yfinance fundamentals and
prices, with SEC EDGAR as a secondary source for the 8-K filing count.

Design notes
------------
* yfinance fundamental field names are inconsistent across tickers, so every
  lookup goes through :func:`get_field`, which tries a list of candidate names
  and returns ``None`` if none are found.
* Per-ticker errors are swallowed and logged; a single bad ticker never breaks
  the batch.
* Raw fundamentals are cached as parquet; the raw cache plus prices drive both
  the current snapshot and the historical trajectory snapshots in
  ``trajectory.py``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import config

logger = logging.getLogger(__name__)


# =============================================================================
# Robust field lookup
# =============================================================================
def get_field(df: pd.DataFrame | None, names: list[str], col_idx: int = 0):
    """Return the first non-NaN value found among ``names`` in the given column.

    yfinance returns fundamentals with column = quarter-end date (most recent
    first) and index = line-item name. Names vary ("Revenue" vs "Total Revenue",
    "EBIT" vs "Operating Income", etc.) so we try a list of candidates.
    """
    if df is None or df.empty:
        return None
    if col_idx >= df.shape[1]:
        return None
    for name in names:
        if name in df.index:
            try:
                val = df.loc[name].iloc[col_idx]
            except Exception:
                continue
            if isinstance(val, pd.Series):
                val = val.iloc[0] if len(val) else None
            if val is not None and pd.notna(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
    return None


# Canonical line-item names, with common yfinance aliases.
FIELDS = {
    # Balance sheet
    "total_assets": ["Total Assets"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "total_liabilities": [
        "Total Liabilities Net Minority Interest",
        "Total Liab",
        "Total Liabilities",
    ],
    "retained_earnings": ["Retained Earnings"],
    "long_term_debt": ["Long Term Debt"],
    "short_term_debt": ["Current Debt", "Short Long Term Debt"],
    "total_debt": ["Total Debt"],
    "cash": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash",
    ],
    "common_stock_shares": [
        "Ordinary Shares Number",
        "Share Issued",
        "Common Stock",
    ],
    # Income statement
    "revenue": ["Total Revenue", "Revenue", "Operating Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Operating Revenue"],
    "ebit": ["EBIT", "Operating Income"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "depreciation_amortization": [
        "Reconciled Depreciation",
        "Depreciation And Amortization",
        "Depreciation",
    ],
    # Cash flow
    "operating_cash_flow": [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
    ],
    "capex": [
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of PPE",
    ],
}


# =============================================================================
# Raw fundamentals fetch + cache
# =============================================================================
@dataclass
class RawBundle:
    """Everything we downloaded for one ticker from yfinance."""
    ticker: str
    info: dict = field(default_factory=dict)
    quarterly_financials: pd.DataFrame | None = None
    quarterly_balance_sheet: pd.DataFrame | None = None
    quarterly_cashflow: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None


def _cache_is_fresh(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < max_age_days * 86400


def _fetch_one(ticker: str) -> RawBundle:
    """Pull fundamentals + prices for a single ticker. Returns empty bundle on error."""
    bundle = RawBundle(ticker=ticker)
    try:
        tk = yf.Ticker(ticker)
        try:
            bundle.info = tk.info or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s: .info failed: %s", ticker, exc)
            bundle.info = {}
        bundle.quarterly_financials = tk.quarterly_financials
        bundle.quarterly_balance_sheet = tk.quarterly_balance_sheet
        bundle.quarterly_cashflow = tk.quarterly_cashflow
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s: fundamental fetch failed: %s", ticker, exc)
    return bundle


def _bundle_to_row(bundle: RawBundle) -> dict:
    """Flatten a RawBundle into a single row for parquet storage."""
    return {
        "ticker": bundle.ticker,
        "info_json": json.dumps(
            {k: v for k, v in (bundle.info or {}).items()
             if isinstance(v, (int, float, str, bool, type(None)))}
        ),
        "financials_json": _df_to_json(bundle.quarterly_financials),
        "balance_json": _df_to_json(bundle.quarterly_balance_sheet),
        "cashflow_json": _df_to_json(bundle.quarterly_cashflow),
    }


def _df_to_json(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return ""
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    return out.to_json(orient="split", date_format="iso")


def _json_to_df(s: str) -> pd.DataFrame | None:
    if not s:
        return None
    try:
        from io import StringIO
        df = pd.read_json(StringIO(s), orient="split")
        try:
            df.columns = pd.to_datetime(df.columns)
        except (ValueError, TypeError):
            pass
        return df
    except Exception as exc:
        logger.debug("_json_to_df failed: %s", exc)
        return None


def _row_to_bundle(row: pd.Series) -> RawBundle:
    info = {}
    try:
        info = json.loads(row["info_json"]) if row["info_json"] else {}
    except Exception:
        info = {}
    return RawBundle(
        ticker=row["ticker"],
        info=info,
        quarterly_financials=_json_to_df(row.get("financials_json", "")),
        quarterly_balance_sheet=_json_to_df(row.get("balance_json", "")),
        quarterly_cashflow=_json_to_df(row.get("cashflow_json", "")),
    )


def fetch_fundamentals(
    tickers: list[str],
    force_refresh: bool = False,
) -> dict[str, RawBundle]:
    """Fetch fundamentals for all tickers, caching to parquet.

    Returns ``{ticker: RawBundle}``. Tickers that fail are skipped silently
    (logged at WARNING).
    """
    cache_path = config.FUNDAMENTALS_CACHE
    cached: dict[str, RawBundle] = {}

    if not force_refresh and _cache_is_fresh(
        cache_path, config.FUNDAMENTALS_CACHE_MAX_AGE_DAYS
    ):
        logger.info("Loading fundamentals cache from %s", cache_path)
        cached_df = pd.read_parquet(cache_path)
        for _, row in cached_df.iterrows():
            cached[row["ticker"]] = _row_to_bundle(row)
        missing = [t for t in tickers if t not in cached]
        if not missing:
            return {t: cached[t] for t in tickers if t in cached}
        logger.info("Cache missing %d tickers, fetching incrementally", len(missing))
        to_fetch = missing
    else:
        to_fetch = tickers

    fetched: dict[str, RawBundle] = {}
    n_total = len(to_fetch)
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    for i in range(0, n_total, config.BATCH_SIZE):
        batch = to_fetch[i : i + config.BATCH_SIZE]
        for tk in batch:
            bundle = _fetch_one(tk)
            if (
                bundle.info
                or (bundle.quarterly_financials is not None
                    and not bundle.quarterly_financials.empty)
            ):
                fetched[tk] = bundle
                n_ok += 1
            else:
                n_fail += 1
        done = min(i + config.BATCH_SIZE, n_total)
        if done % 50 == 0 or done == n_total:
            elapsed = time.time() - t0
            logger.info(
                "Fundamentals: %d/%d (ok=%d fail=%d, %.1fs elapsed)",
                done, n_total, n_ok, n_fail, elapsed,
            )
        if done < n_total:
            time.sleep(config.BATCH_DELAY_SECONDS)

    # Merge with previously cached bundles
    all_bundles = {**cached, **fetched}

    # Persist union to parquet (only rows with data)
    rows = [_bundle_to_row(b) for b in all_bundles.values()]
    if rows:
        pd.DataFrame(rows).to_parquet(cache_path, index=False)
        logger.info("Cached %d ticker bundles to %s", len(rows), cache_path)

    logger.info(
        "Fundamentals: %d/%d succeeded (%d cached, %d fetched, %d failed)",
        len(all_bundles), len(tickers), len(cached), len(fetched), n_fail,
    )
    return {t: all_bundles[t] for t in tickers if t in all_bundles}


# =============================================================================
# Price data
# =============================================================================
def fetch_prices(
    tickers: list[str],
    lookback_days: int = config.PRICE_LOOKBACK_DAYS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download adjusted daily close + volume for all tickers.

    Returns a DataFrame with a MultiIndex of (ticker, field) columns where
    field in ``{"Close", "Volume"}``. Cached to parquet.
    """
    cache_path = config.PRICE_CACHE

    if not force_refresh and _cache_is_fresh(cache_path, 1):
        logger.info("Loading price cache from %s", cache_path)
        cached = pd.read_parquet(cache_path)
        cached.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split("|", 1)) for c in cached.columns]
        )
        cached_tickers = {t for t, _ in cached.columns}
        missing = [t for t in tickers if t not in cached_tickers]
        if not missing:
            return cached
        logger.info("Price cache missing %d tickers", len(missing))
        to_fetch = missing
        prior = cached
    else:
        to_fetch = tickers
        prior = None

    end = datetime.utcnow()
    start = end - timedelta(days=int(lookback_days * 1.6) + 10)

    frames: list[pd.DataFrame] = []
    n_total = len(to_fetch)
    t0 = time.time()
    for i in range(0, n_total, config.BATCH_SIZE):
        batch = to_fetch[i : i + config.BATCH_SIZE]
        try:
            raw = yf.download(
                batch,
                start=start.date().isoformat(),
                end=end.date().isoformat(),
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
                ignore_tz=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Price batch failed %s: %s", batch, exc)
            continue

        for tk in batch:
            df_tk = _extract_ticker_prices(raw, tk)
            if df_tk is None or df_tk.empty:
                continue
            df_tk.columns = pd.MultiIndex.from_product([[tk], df_tk.columns])
            frames.append(df_tk)

        done = min(i + config.BATCH_SIZE, n_total)
        if done % 50 == 0 or done == n_total:
            logger.info("Prices: %d/%d (%.1fs)", done, n_total, time.time() - t0)
        if done < n_total:
            time.sleep(1)

    if not frames and prior is None:
        return pd.DataFrame()

    new_df = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    merged = pd.concat([prior, new_df], axis=1) if prior is not None else new_df
    merged = merged.sort_index()

    # Persist
    persist = merged.copy()
    persist.columns = [f"{t}|{f}" for t, f in persist.columns]
    persist.to_parquet(cache_path)

    return merged


def _extract_ticker_prices(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Pull (Close, Volume) for ``ticker`` from a batched yfinance download."""
    if raw is None or raw.empty:
        return None
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            levels = raw.columns.get_level_values(0)
            if ticker in levels:
                sub = raw[ticker]
            elif ticker in raw.columns.get_level_values(-1):
                sub = raw.xs(ticker, axis=1, level=-1)
            else:
                return None
        else:
            sub = raw
        out = pd.DataFrame({
            "Close": sub["Close"] if "Close" in sub else pd.Series(dtype=float),
            "Volume": sub["Volume"] if "Volume" in sub else pd.Series(dtype=float),
        })
        out = out.dropna(how="all")
        return out if not out.empty else None
    except Exception:
        return None


# =============================================================================
# Individual feature computations
# =============================================================================
def compute_altman_z(b: RawBundle, is_financial: bool, as_of_col: int = 0) -> float | None:
    """Traditional Altman Z for non-financials. Returns NaN for financials."""
    if is_financial:
        return None
    bs = b.quarterly_balance_sheet
    fi = b.quarterly_financials
    if bs is None or bs.empty or fi is None or fi.empty:
        return None

    ta = get_field(bs, FIELDS["total_assets"], as_of_col)
    ca = get_field(bs, FIELDS["current_assets"], as_of_col)
    cl = get_field(bs, FIELDS["current_liabilities"], as_of_col)
    tl = get_field(bs, FIELDS["total_liabilities"], as_of_col)
    re = get_field(bs, FIELDS["retained_earnings"], as_of_col)
    ebit = get_field(fi, FIELDS["ebit"], as_of_col)
    rev = get_field(fi, FIELDS["revenue"], as_of_col)

    if not ta or ta <= 0 or not tl or tl <= 0:
        return None

    # Market cap: prefer info, else compute
    mcap = b.info.get("marketCap")
    if not mcap:
        price = b.info.get("currentPrice") or b.info.get("regularMarketPrice")
        shares = b.info.get("sharesOutstanding")
        if price and shares:
            mcap = price * shares
    if not mcap:
        return None

    wc = (ca - cl) if (ca is not None and cl is not None) else None
    parts = []
    if wc is not None:
        parts.append(1.2 * (wc / ta))
    if re is not None:
        parts.append(1.4 * (re / ta))
    if ebit is not None:
        parts.append(3.3 * (ebit / ta))
    parts.append(0.6 * (mcap / tl))
    if rev is not None:
        parts.append(1.0 * (rev / ta))

    if len(parts) < 3:
        return None
    return float(sum(parts))


def compute_current_ratio(b: RawBundle, as_of_col: int = 0) -> float | None:
    bs = b.quarterly_balance_sheet
    ca = get_field(bs, FIELDS["current_assets"], as_of_col)
    cl = get_field(bs, FIELDS["current_liabilities"], as_of_col)
    if ca is None or cl is None or cl <= 0:
        return None
    return ca / cl


def compute_net_debt_to_ebitda(b: RawBundle, as_of_col: int = 0) -> float | None:
    bs = b.quarterly_balance_sheet
    fi = b.quarterly_financials
    total_debt = get_field(bs, FIELDS["total_debt"], as_of_col)
    if total_debt is None:
        ltd = get_field(bs, FIELDS["long_term_debt"], as_of_col) or 0
        std = get_field(bs, FIELDS["short_term_debt"], as_of_col) or 0
        total_debt = ltd + std if (ltd or std) else None
    cash = get_field(bs, FIELDS["cash"], as_of_col) or 0

    # TTM EBITDA
    ebitda_ttm = 0.0
    n = 0
    for k in range(as_of_col, as_of_col + 4):
        v = get_field(fi, FIELDS["ebitda"], k)
        if v is None:
            oi = get_field(fi, FIELDS["operating_income"], k)
            da = get_field(fi, FIELDS["depreciation_amortization"], k)
            if oi is not None:
                v = oi + (da or 0)
        if v is not None:
            ebitda_ttm += v
            n += 1
    if n == 0:
        ebitda_ttm = None
    elif n < 4:
        # Scale partial TTM
        ebitda_ttm = ebitda_ttm * (4.0 / n)

    if total_debt is None:
        return None
    if ebitda_ttm is None or ebitda_ttm <= 0:
        return 20.0
    nd = total_debt - cash
    ratio = nd / ebitda_ttm
    return float(min(max(ratio, -20.0), 20.0))


def compute_fcf_yield(b: RawBundle, as_of_col: int = 0) -> float | None:
    cf = b.quarterly_cashflow
    # TTM OCF and CapEx
    ocf_ttm = 0.0
    capex_ttm = 0.0
    n_ocf = n_cx = 0
    for k in range(as_of_col, as_of_col + 4):
        v = get_field(cf, FIELDS["operating_cash_flow"], k)
        if v is not None:
            ocf_ttm += v
            n_ocf += 1
        c = get_field(cf, FIELDS["capex"], k)
        if c is not None:
            capex_ttm += abs(c)
            n_cx += 1
    if n_ocf == 0:
        return None
    if n_ocf < 4:
        ocf_ttm *= 4.0 / n_ocf
    if n_cx > 0 and n_cx < 4:
        capex_ttm *= 4.0 / n_cx

    fcf = ocf_ttm - capex_ttm

    ev = b.info.get("enterpriseValue")
    if not ev:
        mcap = b.info.get("marketCap")
        bs = b.quarterly_balance_sheet
        td = get_field(bs, FIELDS["total_debt"], as_of_col)
        cash = get_field(bs, FIELDS["cash"], as_of_col) or 0
        if mcap and td is not None:
            ev = mcap + td - cash
    if not ev or ev <= 0:
        return None
    return fcf / ev


def compute_interest_coverage(b: RawBundle, as_of_col: int = 0) -> float | None:
    fi = b.quarterly_financials
    ebit_ttm = 0.0
    int_ttm = 0.0
    n_e = n_i = 0
    for k in range(as_of_col, as_of_col + 4):
        v = get_field(fi, FIELDS["ebit"], k)
        if v is not None:
            ebit_ttm += v
            n_e += 1
        i = get_field(fi, FIELDS["interest_expense"], k)
        if i is not None:
            int_ttm += abs(i)
            n_i += 1
    if n_e == 0:
        return None
    if int_ttm <= 0 or n_i == 0:
        return 50.0
    coverage = ebit_ttm / int_ttm
    return float(min(max(coverage, -50.0), 50.0))


def compute_pe(b: RawBundle) -> tuple[float | None, bool]:
    """Trailing P/E. Returns (pe_value, is_negative_earnings)."""
    pe = b.info.get("trailingPE")
    if pe is None:
        price = b.info.get("currentPrice") or b.info.get("regularMarketPrice")
        eps = b.info.get("trailingEps")
        if price and eps:
            if eps == 0:
                return None, False
            pe = price / eps
    if pe is None:
        return None, False
    neg = pe < 0
    pe_abs = min(abs(pe), 200.0)
    return float(pe_abs), neg


def compute_ev_ebitda(b: RawBundle, as_of_col: int = 0) -> float | None:
    v = b.info.get("enterpriseToEbitda")
    if v is not None and pd.notna(v):
        if v <= 0:
            return 50.0
        return float(min(v, 50.0))

    # Compute from pieces
    ev = b.info.get("enterpriseValue")
    fi = b.quarterly_financials
    ebitda_ttm = 0.0
    n = 0
    for k in range(as_of_col, as_of_col + 4):
        val = get_field(fi, FIELDS["ebitda"], k)
        if val is None:
            oi = get_field(fi, FIELDS["operating_income"], k)
            da = get_field(fi, FIELDS["depreciation_amortization"], k)
            if oi is not None:
                val = oi + (da or 0)
        if val is not None:
            ebitda_ttm += val
            n += 1
    if not ev or n == 0:
        return None
    if n < 4:
        ebitda_ttm *= 4.0 / n
    if ebitda_ttm <= 0:
        return 50.0
    return float(min(ev / ebitda_ttm, 50.0))


def compute_short_pct_float(b: RawBundle) -> float | None:
    v = b.info.get("shortPercentOfFloat")
    if v is None:
        return None
    # yfinance sometimes returns 0.15 (ratio) and sometimes 15 (percent).
    # Normalize to percent.
    if v <= 1:
        v *= 100
    return float(v)


# --- Price-derived features ---------------------------------------------------
def compute_momentum(prices: pd.Series, days: int, as_of: pd.Timestamp | None = None) -> float | None:
    if prices is None or prices.empty:
        return None
    p = prices.dropna()
    if as_of is not None:
        p = p.loc[:as_of]
    if len(p) < 10:
        return None
    end_date = p.index[-1]
    target = end_date - pd.Timedelta(days=days)
    # Nearest date <= target
    past = p.loc[:target]
    if past.empty:
        return None
    p_then = past.iloc[-1]
    p_now = p.iloc[-1]
    if p_then <= 0:
        return None
    return float(p_now / p_then - 1)


def compute_volatility_60d(prices: pd.Series, as_of: pd.Timestamp | None = None) -> float | None:
    if prices is None or prices.empty:
        return None
    p = prices.dropna()
    if as_of is not None:
        p = p.loc[:as_of]
    if len(p) < 30:
        return None
    rets = p.pct_change().dropna().tail(60)
    if len(rets) < 20:
        return None
    return float(rets.std() * np.sqrt(252))


def compute_relative_volume(volume: pd.Series, as_of: pd.Timestamp | None = None) -> float | None:
    if volume is None or volume.empty:
        return None
    v = volume.dropna()
    if as_of is not None:
        v = v.loc[:as_of]
    if len(v) < 60:
        return None
    v20 = v.tail(20).mean()
    v60 = v.tail(60).mean()
    if v60 <= 0:
        return None
    return float(v20 / v60)


# =============================================================================
# Quality / capital-allocation / earnings-behavior features (Part 3 expansion)
# =============================================================================
def compute_accruals_ratio(b: RawBundle, as_of_col: int = 0) -> float | None:
    """OCF / Net Income TTM. Low values signal earnings not backed by cash
    (Sloan 1996 accruals anomaly). Capped at [-1.0, 3.0]."""
    cf = b.quarterly_cashflow
    fi = b.quarterly_financials
    if cf is None or fi is None or cf.empty or fi.empty:
        return None
    if cf.shape[1] < as_of_col + 4 or fi.shape[1] < as_of_col + 4:
        return None
    ocf = 0.0
    n_o = 0
    ni = 0.0
    n_n = 0
    for k in range(as_of_col, as_of_col + 4):
        o = get_field(cf, FIELDS["operating_cash_flow"], k)
        n = get_field(fi, FIELDS["net_income"], k)
        if o is not None:
            ocf += o
            n_o += 1
        if n is not None:
            ni += n
            n_n += 1
    if n_o < 2 or n_n < 2:
        return None
    if abs(ni) < 1e6:
        return None
    ratio = ocf / ni
    return max(-1.0, min(3.0, ratio))


def compute_asset_growth(b: RawBundle, as_of_col: int = 0) -> float | None:
    """YoY change in total assets. High asset growth predicts negative future
    returns (Cooper, Gulen, Schill 2008). Capped at [-0.50, 1.00]."""
    bs = b.quarterly_balance_sheet
    if bs is None or bs.empty or bs.shape[1] < as_of_col + 5:
        return None
    current = get_field(bs, FIELDS["total_assets"], as_of_col)
    prior = get_field(bs, FIELDS["total_assets"], as_of_col + 4)
    if not current or not prior or prior <= 0:
        return None
    growth = (current - prior) / prior
    return max(-0.50, min(1.00, growth))


def compute_net_issuance(b: RawBundle, as_of_col: int = 0) -> float | None:
    """YoY change in shares outstanding. Positive = dilution (Daniel & Titman
    2006). Capped at [-0.30, 0.50]."""
    bs = b.quarterly_balance_sheet
    if bs is None or bs.empty or bs.shape[1] < as_of_col + 5:
        return None
    current = get_field(bs, FIELDS["common_stock_shares"], as_of_col)
    prior = get_field(bs, FIELDS["common_stock_shares"], as_of_col + 4)
    if not current or not prior or prior <= 0:
        return None
    issuance = (current - prior) / prior
    return max(-0.30, min(0.50, issuance))


# =============================================================================
# SEC EDGAR: 8-K filing counts
# =============================================================================
class RateLimiter:
    def __init__(self, per_second: int):
        self.min_interval = 1.0 / per_second
        self.last = 0.0

    def wait(self):
        now = time.time()
        dt = now - self.last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self.last = time.time()


def _load_cik_map() -> dict[str, str]:
    """Download (and cache) the SEC ticker -> CIK mapping."""
    path = config.SEC_CIK_CACHE
    if path.exists() and (time.time() - path.stat().st_mtime) < 30 * 86400:
        with open(path) as f:
            return json.load(f)
    logger.info("Downloading SEC ticker->CIK map")
    try:
        resp = requests.get(
            config.SEC_TICKER_MAP_URL,
            headers={"User-Agent": config.SEC_USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("SEC CIK map download failed: %s", exc)
        return {}

    mapping: dict[str, str] = {}
    for _, entry in raw.items():
        tk = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if tk and cik:
            mapping[tk] = cik

    with open(path, "w") as f:
        json.dump(mapping, f)
    return mapping


def fetch_filing_counts(
    tickers: list[str],
    as_of: pd.Timestamp | None = None,
    days: int = 90,
) -> dict[str, int | None]:
    """Count 8-K filings in the trailing ``days`` per ticker via SEC EDGAR."""
    cik_map = _load_cik_map()
    if not cik_map:
        return {t: None for t in tickers}

    # Normalize ticker form: EDGAR uses no separators
    def _edgar_ticker(t: str) -> str:
        return t.replace("-", "").replace(".", "").upper()

    limiter = RateLimiter(config.SEC_RATE_LIMIT_PER_SECOND)
    cutoff = (as_of or pd.Timestamp.utcnow().tz_localize(None)) - pd.Timedelta(days=days)
    upper = as_of or pd.Timestamp.utcnow().tz_localize(None)
    results: dict[str, int | None] = {}

    for i, tk in enumerate(tickers, 1):
        cik = cik_map.get(_edgar_ticker(tk))
        if not cik:
            results[tk] = None
            continue
        url = config.SEC_SUBMISSIONS_URL.format(cik=cik)
        try:
            limiter.wait()
            resp = requests.get(
                url,
                headers={"User-Agent": config.SEC_USER_AGENT},
                timeout=15,
            )
            if resp.status_code != 200:
                results[tk] = None
                continue
            payload = resp.json()
            recent = payload.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            count = 0
            for f, d in zip(forms, dates):
                if f != "8-K":
                    continue
                try:
                    fd = pd.Timestamp(d)
                except Exception:
                    continue
                if cutoff <= fd <= upper:
                    count += 1
            results[tk] = count
        except Exception as exc:  # noqa: BLE001
            logger.debug("SEC fetch failed for %s: %s", tk, exc)
            results[tk] = None

        if i % 100 == 0:
            logger.info("EDGAR: %d/%d", i, len(tickers))

    return results


# =============================================================================
# Assembly
# =============================================================================
def _sector_from_bundle(b: RawBundle, fallback: str | None) -> str:
    return (b.info.get("sector") or fallback or "Unknown") or "Unknown"


def _company_from_bundle(b: RawBundle, fallback: str | None) -> str:
    return b.info.get("longName") or b.info.get("shortName") or fallback or b.ticker


def build_features(
    universe_df: pd.DataFrame,
    bundles: dict[str, RawBundle],
    prices: pd.DataFrame,
    filing_counts: dict[str, int | None] | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute the full feature matrix at a point in time (default: now).

    ``prices`` must be a MultiIndex DataFrame (ticker, field) with at least
    "Close" and "Volume". ``as_of`` scopes price-based features to data <= that
    timestamp; fundamentals are taken from the most recent quarter at or before
    ``as_of`` when available.
    """
    filing_counts = filing_counts or {}
    universe_meta = universe_df.set_index("Ticker")

    rows = []

    for tk in universe_meta.index:
        b = bundles.get(tk)
        if b is None:
            continue

        sector = _sector_from_bundle(b, universe_meta.loc[tk].get("Sector"))
        is_financial = sector in config.FINANCIAL_SECTORS

        # Pick the right quarter column given as_of
        col_idx = _quarter_col_for_date(b.quarterly_balance_sheet, as_of)

        row = {
            "Ticker": tk,
            "Company": _company_from_bundle(b, universe_meta.loc[tk].get("Company")),
            "Sector": sector,
            "is_financial": is_financial,
        }

        row["altman_z"] = compute_altman_z(b, is_financial, col_idx)
        row["current_ratio"] = compute_current_ratio(b, col_idx)
        row["net_debt_to_ebitda"] = compute_net_debt_to_ebitda(b, col_idx)
        row["fcf_yield"] = compute_fcf_yield(b, col_idx)
        row["interest_coverage"] = compute_interest_coverage(b, col_idx)
        row["short_pct_float"] = compute_short_pct_float(b)

        # Quality / capital allocation
        row["accruals_ratio"] = compute_accruals_ratio(b, col_idx)
        row["asset_growth_yoy"] = compute_asset_growth(b, col_idx)
        row["net_issuance_yoy"] = compute_net_issuance(b, col_idx)

        close = _ticker_series(prices, tk, "Close")
        volume = _ticker_series(prices, tk, "Volume")
        row["momentum_30d"] = compute_momentum(close, 30, as_of)
        row["momentum_90d"] = compute_momentum(close, 90, as_of)
        row["volatility_60d"] = compute_volatility_60d(close, as_of)
        row["relative_volume"] = compute_relative_volume(volume, as_of)

        pe, neg = compute_pe(b)
        row["pe_ratio"] = pe
        row["earnings_negative"] = neg
        row["ev_to_ebitda"] = compute_ev_ebitda(b, col_idx)

        row["filing_count_90d"] = filing_counts.get(tk)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("Ticker")
    logger.info("Feature build: %d tickers", len(df))
    return df


def _ticker_series(prices: pd.DataFrame, ticker: str, field_name: str) -> pd.Series | None:
    if prices is None or prices.empty:
        return None
    if (ticker, field_name) in prices.columns:
        return prices[(ticker, field_name)].dropna()
    return None


def _quarter_col_for_date(df: pd.DataFrame | None, as_of: pd.Timestamp | None) -> int:
    """Return the column index of the most recent quarter <= as_of."""
    if df is None or df.empty or as_of is None:
        return 0
    try:
        cols = pd.to_datetime(df.columns)
    except Exception:
        return 0
    # Columns are most-recent first. Find first col whose date <= as_of.
    for i, c in enumerate(cols):
        if c <= as_of:
            return i
    return len(cols) - 1


# =============================================================================
# Missing-data handling
# =============================================================================
def clean_features(
    features: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing values with cross-sectional medians.

    Drops stocks with >50% of features missing. Returns (clean_df, imputed_mask).

    Raises RuntimeError if fundamental features have <10% coverage — this
    indicates a yfinance data failure, not a real result.
    """
    feature_cols = feature_cols or config.FEATURES
    available = [c for c in feature_cols if c in features.columns]
    feats = features[available].copy()

    # Coverage report
    logger.info("Feature coverage (of %d stocks):", len(feats))
    fundamental_cols = [
        "altman_z", "current_ratio", "net_debt_to_ebitda",
        "fcf_yield", "interest_coverage",
    ]
    dead_fundamentals = []
    for c in available:
        n_valid = int(feats[c].notna().sum())
        pct = 100.0 * n_valid / max(len(feats), 1)
        logger.info("  %-22s %4d valid  (%.1f%%)", c, n_valid, pct)
        if c in fundamental_cols and pct < 10.0:
            dead_fundamentals.append((c, pct))

    # CIRCUIT BREAKER: if most fundamental features have <10% coverage,
    # the yfinance data pull failed. Do NOT silently proceed with zeros.
    if len(dead_fundamentals) >= 3:
        msg_lines = [f"    {c}: {p:.1f}% coverage" for c, p in dead_fundamentals]
        msg = "\n".join(msg_lines)
        logger.error(
            "FUNDAMENTAL DATA FAILURE: %d of %d fundamental features have <10%% "
            "valid data. This means yfinance returned empty financials for most "
            "stocks. PCA will be driven entirely by price-based features, "
            "producing misleading results.\n%s\n"
            "Actions:\n"
            "  1. Run with --refresh to force re-download\n"
            "  2. Check yfinance version: pip show yfinance\n"
            "  3. Test a single ticker: python -c \"import yfinance; "
            "print(yfinance.Ticker('AAPL').quarterly_financials)\"\n"
            "  4. If yfinance is broken, downgrade: pip install yfinance==0.2.36\n"
            "  5. Delete data/fundamentals_cache.parquet and retry",
            len(dead_fundamentals), len(fundamental_cols), msg,
        )
        raise RuntimeError(
            f"Fundamental data failure: {len(dead_fundamentals)} features have "
            f"<10% coverage. See log for details. Run with --refresh or check "
            f"yfinance installation."
        )

    # Drop rows with >50% missing
    pct_missing = feats.isna().mean(axis=1)
    drop_mask = pct_missing > 0.50
    if drop_mask.any():
        logger.info("Dropping %d stocks with >50%% features missing", int(drop_mask.sum()))
    feats = feats.loc[~drop_mask]
    meta = features.loc[feats.index, [c for c in features.columns if c not in feature_cols]]

    # Imputation mask BEFORE filling
    imputed = feats.isna()

    # Cross-sectional medians
    medians = feats.median(numeric_only=True)
    n_imp = int(imputed.sum().sum())
    feats = feats.fillna(medians)

    # Any residual NaNs (feature with all-NaN): fill with 0
    # Log a WARNING for any column that hits this fallback
    still_nan = feats.columns[feats.isna().any()].tolist()
    if still_nan:
        logger.warning(
            "Features with ALL-NaN values (filled with 0.0): %s — "
            "these features have zero variance and will not contribute to PCA",
            still_nan,
        )
    feats = feats.fillna(0.0)

    clean = pd.concat([meta, feats], axis=1)
    logger.info("Imputed %d missing values via cross-sectional median", n_imp)
    return clean, imputed


def save_features(df: pd.DataFrame) -> None:
    path = config.FEATURES_CACHE
    df.to_parquet(path)
    logger.info("Saved features to %s", path)


# =============================================================================
# Standalone diagnostic — run with: python feature_engine.py
# =============================================================================
def diagnose():
    """Test yfinance fundamental data retrieval on a few tickers.
    
    Run this to determine whether your yfinance installation can actually
    pull quarterly financials. If this prints empty DataFrames or errors,
    that's why your PCA has zero loadings on fundamental features.
    """
    import yfinance as yf
    
    test_tickers = ["AAPL", "CRGY", "DOCN", "AX", "INVA"]
    print("=" * 60)
    print("FEATURE ENGINE DIAGNOSTIC")
    print("=" * 60)
    print(f"yfinance version: {yf.__version__}")
    print()
    
    for tk_str in test_tickers:
        print(f"--- {tk_str} ---")
        try:
            tk = yf.Ticker(tk_str)
            
            # Test .info
            info = tk.info or {}
            mcap = info.get("marketCap", "MISSING")
            print(f"  info.marketCap: {mcap}")
            
            # Test .quarterly_financials
            qf = tk.quarterly_financials
            if qf is None or qf.empty:
                print(f"  quarterly_financials: EMPTY / NONE")
            else:
                print(f"  quarterly_financials: {qf.shape}")
                print(f"    columns (quarters): {list(qf.columns[:4])}")
                print(f"    index sample: {list(qf.index[:5])}")
                # Try to get revenue
                for name in ["Total Revenue", "Revenue", "Operating Revenue"]:
                    if name in qf.index:
                        val = qf.loc[name].iloc[0]
                        print(f"    {name}: {val}")
                        break
                else:
                    print(f"    Revenue: NOT FOUND in index")
            
            # Test .quarterly_balance_sheet
            qbs = tk.quarterly_balance_sheet
            if qbs is None or qbs.empty:
                print(f"  quarterly_balance_sheet: EMPTY / NONE")
            else:
                print(f"  quarterly_balance_sheet: {qbs.shape}")
                for name in ["Total Assets"]:
                    if name in qbs.index:
                        print(f"    {name}: {qbs.loc[name].iloc[0]}")
                        break
                else:
                    print(f"    Total Assets: NOT FOUND in index")
            
            # Test Altman Z computation
            bundle = _fetch_one(tk_str)
            z = compute_altman_z(bundle, is_financial=False)
            print(f"  Altman Z: {z}")
            
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
        print()
    
    # Check cache integrity
    cache_path = config.FUNDAMENTALS_CACHE
    if cache_path.exists():
        print("--- CACHE CHECK ---")
        cached_df = pd.read_parquet(cache_path)
        print(f"  Cache file: {cache_path}")
        print(f"  Rows: {len(cached_df)}")
        # Check if financials_json is populated
        empty_financials = cached_df["financials_json"].apply(
            lambda x: x == "" or x is None or pd.isna(x) if isinstance(x, str) else True
        ).sum()
        print(f"  Tickers with EMPTY financials_json: {empty_financials}/{len(cached_df)}")
        print(f"  Tickers with data: {len(cached_df) - empty_financials}/{len(cached_df)}")
        if empty_financials > len(cached_df) * 0.5:
            print(f"\n  *** >50% of cached tickers have empty financials! ***")
            print(f"  *** Delete {cache_path} and re-run with --refresh ***")
    else:
        print(f"  No cache file at {cache_path}")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    diagnose()