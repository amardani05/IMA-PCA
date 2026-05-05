"""Download and cache price, factor, and sector-ETF data.

- Prices: yfinance, batched, cached as parquet.
- Fama-French 5 + momentum: Ken French data library (zipped CSVs).
- Sector ETFs: yfinance, same caching treatment.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    BATCH_DELAY_SECONDS,
    BATCH_SIZE,
    BENCHMARK_RETURNS_PARQUET,
    BENCHMARK_TICKER,
    CACHE_MAX_AGE_SECONDS,
    FF5_DAILY_CSV,
    FF5_URL,
    LOOKBACK_DAYS,
    MIN_TRADING_DAYS,
    MOM_DAILY_CSV,
    MOM_URL,
    SECTOR_ETFS,
    SECTOR_RETURNS_PARQUET,
    SP600_RETURNS_PARQUET,
    USER_AGENT,
)

logger = logging.getLogger(__name__)


def _cache_is_fresh(path: Path, max_age: int = CACHE_MAX_AGE_SECONDS) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < max_age


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance output into a flat close-price DataFrame.

    yfinance 1.2.x sometimes ignores ``multi_level_index=False`` for multi-ticker
    downloads, so we handle both MultiIndex and flat-column cases.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # Typical shape: level 0 = field ('Close', 'Open', ...), level 1 = ticker
        if "Close" in raw.columns.get_level_values(0):
            close = raw.xs("Close", axis=1, level=0)
        elif "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", axis=1, level=-1)
        else:
            # Single-ticker requests may return only OHLC at level 0 with ticker at level 1
            close = raw["Close"] if "Close" in raw.columns else pd.DataFrame()
    else:
        if "Close" in raw.columns:
            # Single-ticker download
            close = raw[["Close"]].copy()
            if len(tickers) == 1:
                close.columns = [tickers[0]]
        else:
            close = raw.copy()

    # Drop tickers that are entirely NaN (silent yfinance failures)
    close = close.dropna(axis=1, how="all")
    return close


def _download_batch(
    tickers: list[str], start: str, end: str, max_retries: int = 2
) -> pd.DataFrame:
    """Download a batch of tickers. Returns a wide close-price DataFrame."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
                ignore_tz=True,
                threads=True,
            )
            close = _extract_close(raw, tickers)
            return close
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "yfinance batch attempt %d failed: %s", attempt + 1, exc
            )
            time.sleep(2 + attempt * 2)
    logger.error("yfinance batch failed after retries: %s", last_exc)
    return pd.DataFrame()


def download_prices(
    tickers: list[str],
    lookback_days: int = LOOKBACK_DAYS,
    batch_size: int = BATCH_SIZE,
    batch_delay: float = BATCH_DELAY_SECONDS,
) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers, batched."""
    # Pad the window a little so LOOKBACK_DAYS of *trading* days fits in calendar days.
    end_dt = datetime.utcnow().date()
    start_dt = end_dt - timedelta(days=int(lookback_days * 1.6) + 30)
    start, end = start_dt.isoformat(), end_dt.isoformat()

    tickers = sorted({t.upper() for t in tickers})
    logger.info(
        "Downloading prices for %d tickers from %s to %s", len(tickers), start, end
    )

    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logger.info(
            "Batch %d/%d (%d tickers)",
            i // batch_size + 1,
            (len(tickers) + batch_size - 1) // batch_size,
            len(batch),
        )
        close = _download_batch(batch, start, end)
        if not close.empty:
            frames.append(close)
        if i + batch_size < len(tickers):
            time.sleep(batch_delay)

    if not frames:
        raise RuntimeError("yfinance returned no data for any batch")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()
    return prices


def _prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a close-price matrix to daily simple returns."""
    returns = prices.pct_change().iloc[1:]
    # Trim rows that are entirely NaN (holidays, initial row)
    returns = returns.dropna(how="all")
    return returns


def load_universe_returns(
    tickers: list[str],
    force_refresh: bool = False,
    min_trading_days: int = MIN_TRADING_DAYS,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Return a T x N daily-returns matrix for the requested tickers.

    Caches to ``data/sp600_returns.parquet``. Drops any ticker with fewer than
    ``min_trading_days`` non-NaN return observations.
    """
    if not force_refresh and _cache_is_fresh(SP600_RETURNS_PARQUET):
        logger.info("Loading cached returns from %s", SP600_RETURNS_PARQUET)
        returns = pd.read_parquet(SP600_RETURNS_PARQUET)
        # Trim to requested window (in case cache is longer)
        returns = returns.tail(lookback_days)
        return returns

    prices = download_prices(tickers, lookback_days=lookback_days)
    returns = _prices_to_returns(prices)

    n_before = returns.shape[1]
    valid_counts = returns.notna().sum(axis=0)
    keep = valid_counts[valid_counts >= min_trading_days].index
    returns = returns[keep]
    logger.info(
        "Dropped %d tickers for insufficient data; %d remaining",
        n_before - returns.shape[1],
        returns.shape[1],
    )

    # Keep the most recent `lookback_days` trading days
    returns = returns.tail(lookback_days)

    # Final guard: drop residual all-NaN columns after trimming
    returns = returns.dropna(axis=1, how="all")

    returns.to_parquet(SP600_RETURNS_PARQUET)
    logger.info(
        "Cached returns: %d dates x %d tickers -> %s",
        returns.shape[0],
        returns.shape[1],
        SP600_RETURNS_PARQUET,
    )
    return returns


def load_benchmark_returns(
    ticker: str = BENCHMARK_TICKER,
    force_refresh: bool = False,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.Series:
    """Load daily returns for the benchmark ETF."""
    if not force_refresh and _cache_is_fresh(BENCHMARK_RETURNS_PARQUET):
        df = pd.read_parquet(BENCHMARK_RETURNS_PARQUET)
        return df[ticker].tail(lookback_days)

    prices = download_prices([ticker], lookback_days=lookback_days, batch_size=1)
    returns = _prices_to_returns(prices)
    returns.to_parquet(BENCHMARK_RETURNS_PARQUET)
    return returns[ticker].tail(lookback_days)


def load_sector_etf_returns(
    force_refresh: bool = False, lookback_days: int = LOOKBACK_DAYS
) -> pd.DataFrame:
    """Load daily returns for the 11 SPDR sector ETFs."""
    if not force_refresh and _cache_is_fresh(SECTOR_RETURNS_PARQUET):
        returns = pd.read_parquet(SECTOR_RETURNS_PARQUET)
        return returns.tail(lookback_days)

    tickers = list(SECTOR_ETFS.keys())
    prices = download_prices(tickers, lookback_days=lookback_days)
    returns = _prices_to_returns(prices)
    returns.to_parquet(SECTOR_RETURNS_PARQUET)
    return returns.tail(lookback_days)


# -----------------------------------------------------------------------------
# Fama-French data loaders
# -----------------------------------------------------------------------------

def _fetch_french_csv(url: str) -> str:
    """Download a Ken French zipped CSV and return the CSV body as text."""
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(name) as fh:
            return fh.read().decode("latin-1")


def _parse_french_daily(text: str) -> pd.DataFrame:
    """Parse a daily-frequency French CSV.

    Strategy: find the first line whose first comma-separated token is an
    8-digit date (YYYYMMDD). Read contiguous rows of the same format, stopping
    at the first row that no longer parses (blank line, annual section, etc.).
    """
    lines = text.splitlines()
    header: list[str] | None = None
    rows: list[list[str]] = []
    in_block = False

    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if not parts or not parts[0]:
            if in_block:
                break
            continue
        # Identify date rows
        if parts[0].isdigit() and len(parts[0]) == 8:
            if not in_block:
                in_block = True
            rows.append(parts)
        elif in_block:
            # First non-date row after the daily block -> stop
            break
        else:
            # Header candidate: last non-empty, non-date line before first date
            if all(p for p in parts[1:] if p):
                header = parts

    if not rows:
        raise RuntimeError("No daily data rows found in French CSV")

    n_cols = len(rows[0])
    if header is None or len(header) < n_cols:
        header = ["Date"] + [f"F{i}" for i in range(n_cols - 1)]
    else:
        # Header may lack a leading "Date" label; align with row length
        if len(header) == n_cols - 1:
            header = ["Date"] + header
        elif len(header) > n_cols:
            header = header[:n_cols]
            header[0] = "Date"
        else:
            header[0] = "Date"

    df = pd.DataFrame(rows, columns=header)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    df = df.dropna(how="all")
    return df


def load_fama_french(force_refresh: bool = False) -> pd.DataFrame:
    """Return daily FF5 + momentum as decimals, indexed by date.

    Columns: ``Mkt-RF``, ``SMB``, ``HML``, ``RMW``, ``CMA``, ``RF``, ``Mom``.
    """
    if (
        not force_refresh
        and _cache_is_fresh(FF5_DAILY_CSV)
        and _cache_is_fresh(MOM_DAILY_CSV)
    ):
        ff5 = pd.read_csv(FF5_DAILY_CSV, index_col=0, parse_dates=True)
        mom = pd.read_csv(MOM_DAILY_CSV, index_col=0, parse_dates=True)
    else:
        logger.info("Downloading Fama-French 5 factors")
        ff5 = _parse_french_daily(_fetch_french_csv(FF5_URL))
        logger.info("Downloading Momentum factor")
        mom = _parse_french_daily(_fetch_french_csv(MOM_URL))
        # Momentum file column is often named "Mom   " or "MOM"
        mom.columns = ["Mom" for _ in mom.columns][:1] + list(mom.columns[1:])
        if len(mom.columns) == 1:
            mom.columns = ["Mom"]
        ff5.to_csv(FF5_DAILY_CSV)
        mom.to_csv(MOM_DAILY_CSV)

    # Standardize momentum column name
    if "Mom" not in mom.columns:
        mom = mom.rename(columns={mom.columns[0]: "Mom"})

    merged = ff5.join(mom[["Mom"]], how="inner")
    expected = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]
    # French's column headers sometimes have extra whitespace; normalize
    merged.columns = [str(c).strip() for c in merged.columns]
    present = [c for c in expected if c in merged.columns]
    merged = merged[present]
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ff = load_fama_french(force_refresh=True)
    print(ff.tail())
