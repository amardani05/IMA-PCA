"""Fetch the S&P 600 constituent list.

Scrapes Wikipedia with a proper User-Agent. Falls back to a cached list on
failure. Normalizes tickers for yfinance (BRK.B -> BRK-B).
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import (
    SP600_CONSTITUENTS_CSV,
    SP600_FALLBACK_CSV,
    SP600_WIKI_URL,
    USER_AGENT,
    CACHE_MAX_AGE_SECONDS,
)

logger = logging.getLogger(__name__)


def _normalize_ticker(ticker: str) -> str:
    """Yahoo uses '-' instead of '.' for share classes."""
    return ticker.strip().upper().replace(".", "-")


def _cache_is_fresh(path: Path, max_age: int = CACHE_MAX_AGE_SECONDS) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < max_age


def _scrape_wikipedia() -> pd.DataFrame:
    """Scrape the S&P 600 constituent table from Wikipedia."""
    logger.info("Scraping S&P 600 constituents from Wikipedia")
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(SP600_WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})
    if not tables:
        raise RuntimeError("No wikitable found on S&P 600 page")

    # The constituents table is the first wikitable with a Symbol/Ticker column.
    df: pd.DataFrame | None = None
    for t in tables:
        try:
            candidate = pd.read_html(StringIO(str(t)))[0]
        except ValueError:
            continue
        cols = [c.lower() for c in candidate.columns.astype(str)]
        if any("symbol" in c or "ticker" in c for c in cols):
            df = candidate
            break
    if df is None:
        raise RuntimeError("Could not locate constituent table")

    # Normalize column names
    rename_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if "symbol" in cl or "ticker" in cl:
            rename_map[c] = "Ticker"
        elif "security" in cl or "company" in cl:
            rename_map[c] = "Company"
        elif "sector" in cl and "sub" not in cl:
            rename_map[c] = "Sector"
        elif "industry" in cl or "sub-industry" in cl or "sub industry" in cl:
            rename_map[c] = "Industry"
    df = df.rename(columns=rename_map)

    keep = [c for c in ["Ticker", "Company", "Sector", "Industry"] if c in df.columns]
    df = df[keep].copy()

    # Add missing columns with NaN so downstream code doesn't break
    for col in ["Company", "Sector", "Industry"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["Ticker"] = df["Ticker"].astype(str).map(_normalize_ticker)
    df = df.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)

    if len(df) < 400:
        raise RuntimeError(
            f"Suspiciously few tickers scraped ({len(df)}); Wikipedia layout may "
            f"have changed"
        )

    logger.info("Scraped %d S&P 600 constituents", len(df))
    return df[["Ticker", "Company", "Sector", "Industry"]]


def _load_fallback() -> pd.DataFrame:
    """Load the hardcoded fallback list saved from a previous successful scrape."""
    if not SP600_FALLBACK_CSV.exists():
        raise RuntimeError(
            "No fallback S&P 600 list available. First run must succeed against "
            "Wikipedia to seed the fallback."
        )
    logger.warning("Using cached fallback S&P 600 list at %s", SP600_FALLBACK_CSV)
    return pd.read_csv(SP600_FALLBACK_CSV)


def get_sp600_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Return the S&P 600 universe as a DataFrame.

    Columns: Ticker, Company, Sector, Industry.
    Caches to ``data/sp600_constituents.csv``. Falls back to a bundled list on
    any scrape failure.
    """
    if not force_refresh and _cache_is_fresh(SP600_CONSTITUENTS_CSV):
        logger.info("Loading cached S&P 600 list from %s", SP600_CONSTITUENTS_CSV)
        return pd.read_csv(SP600_CONSTITUENTS_CSV)

    try:
        df = _scrape_wikipedia()
        df.to_csv(SP600_CONSTITUENTS_CSV, index=False)
        # Seed / refresh the fallback as well
        df.to_csv(SP600_FALLBACK_CSV, index=False)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wikipedia scrape failed: %s", exc)
        return _load_fallback()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = get_sp600_universe(force_refresh=True)
    print(df.head())
    print(f"Total: {len(df)} tickers")
