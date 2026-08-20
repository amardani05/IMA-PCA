"""Point-in-time historical store: ingestion, schema validation, coverage.

The live screener pulls *current* features from yfinance, which only exposes
~4 quarters of fundamentals — so a deep (multi-year) backtest of the fundamental
features cannot be reconstructed from yfinance alone (see the README "Historical
data" section).  To get depth you feed a USER-SUPPLIED point-in-time store from
FactSet / S&P Global / manual collection, written as long-format parquet under
``data/historical/``:

================  ===========================================================
File              Columns
================  ===========================================================
features.parquet  date, ticker, Sector, <all config.FEATURES columns>
                  -> one row per (date, ticker): the feature *snapshot* that
                     was knowable AS OF ``date``.  This is the file you
                     populate to get fundamental depth.
universe.parquet  date, ticker, in_index(bool)
                  -> point-in-time index membership (the SURVIVORSHIP fix).
ima_holdings.parquet  date, ticker, weight
                  -> IMA portfolio composition over time.
prices.parquet    date, ticker, adj_close
                  -> daily adjusted closes INCLUDING delisted names.
delistings.parquet (optional)  ticker, delist_date, final_value
                  -> terminal value for delisted names so a name that goes to
                     zero is captured as a severe-drawdown event, not silently dropped.
================  ===========================================================

Honesty guards enforced here:
  * schema validation — wrong/missing columns fail loudly.
  * per-date coverage report — so you can see where the store is thin.
  * look-ahead assertion (``assert_no_lookahead``) — any feature row whose
    ``date`` is after the as-of snapshot date raises.
  * survivorship warning — if ``universe.parquet`` is absent the engine falls
    back to current membership and we WARN that results are survivorship-biased.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# =============================================================================
# Schema definitions
# =============================================================================
FEATURES_REQUIRED = ["date", "ticker", "Sector", *config.FEATURES]
UNIVERSE_REQUIRED = ["date", "ticker", "in_index"]
IMA_REQUIRED = ["date", "ticker", "weight"]
PRICES_REQUIRED = ["date", "ticker", "adj_close"]
DELIST_REQUIRED = ["ticker", "delist_date", "final_value"]


class SchemaError(ValueError):
    """Raised when a historical parquet file violates its documented schema."""


class LookAheadError(AssertionError):
    """Raised when a feature row's date is after the as-of snapshot date."""


@dataclass
class HistoricalStore:
    """In-memory view of the point-in-time store.

    ``universe`` / ``ima_holdings`` / ``delistings`` may be ``None`` when the
    corresponding file is absent; callers must handle (and warn about) that.
    """

    features: pd.DataFrame                 # date, ticker, Sector, <FEATURES>
    prices: pd.DataFrame                   # date, ticker, adj_close
    universe: pd.DataFrame | None          # date, ticker, in_index
    ima_holdings: pd.DataFrame | None      # date, ticker, weight
    delistings: pd.DataFrame | None        # ticker, delist_date, final_value

    @property
    def has_universe(self) -> bool:
        return self.universe is not None and not self.universe.empty

    @property
    def feature_dates(self) -> list[pd.Timestamp]:
        return sorted(self.features["date"].unique())


# =============================================================================
# Validation helpers
# =============================================================================
def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{name}: missing required column(s) {missing}. "
            f"Expected schema: {required}. Got: {list(df.columns)}"
        )


def _coerce_dates(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    if df[col].isna().any():
        bad = int(df[col].isna().sum())
        raise SchemaError(f"{name}: {bad} unparseable value(s) in date column '{col}'.")
    df[col] = df[col].dt.normalize()
    return df


def _validate_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, FEATURES_REQUIRED, "features.parquet")
    df = _coerce_dates(df, "date", "features.parquet")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    if df.duplicated(["date", "ticker"]).any():
        n = int(df.duplicated(["date", "ticker"]).sum())
        raise SchemaError(f"features.parquet: {n} duplicate (date, ticker) row(s).")
    return df


def _validate_universe(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, UNIVERSE_REQUIRED, "universe.parquet")
    df = _coerce_dates(df, "date", "universe.parquet")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["in_index"] = df["in_index"].astype(bool)
    return df


def _validate_ima(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, IMA_REQUIRED, "ima_holdings.parquet")
    df = _coerce_dates(df, "date", "ima_holdings.parquet")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    return df


def _validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, PRICES_REQUIRED, "prices.parquet")
    df = _coerce_dates(df, "date", "prices.parquet")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["adj_close"])
    return df.sort_values(["ticker", "date"])


def _validate_delistings(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, DELIST_REQUIRED, "delistings.parquet")
    df = _coerce_dates(df, "delist_date", "delistings.parquet")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["final_value"] = pd.to_numeric(df["final_value"], errors="coerce").fillna(0.0)
    return df


# =============================================================================
# Loader
# =============================================================================
def load_historical_store() -> HistoricalStore:
    """Load + validate every present parquet under ``config.HISTORICAL_DIR``.

    ``features.parquet`` and ``prices.parquet`` are mandatory; the rest are
    optional but their absence is WARNED about (survivorship / IMA coverage).
    """
    fpath = config.HIST_FEATURES_PARQUET
    ppath = config.HIST_PRICES_PARQUET
    if not fpath.exists():
        raise FileNotFoundError(
            f"Required historical file missing: {fpath}. "
            "Populate the point-in-time store (see historical_loader docstring) "
            "or generate a synthetic one with generate_synthetic_store()."
        )
    if not ppath.exists():
        raise FileNotFoundError(f"Required historical file missing: {ppath}.")

    features = _validate_features(pd.read_parquet(fpath))
    prices = _validate_prices(pd.read_parquet(ppath))

    universe = None
    if config.HIST_UNIVERSE_PARQUET.exists():
        universe = _validate_universe(pd.read_parquet(config.HIST_UNIVERSE_PARQUET))
    else:
        logger.warning(
            "SURVIVORSHIP WARNING: universe.parquet absent at %s — the backtest "
            "will fall back to whatever tickers appear in features.parquet, which "
            "is survivorship-biased (delisted names that left the index are "
            "invisible). Provide point-in-time membership to fix this.",
            config.HIST_UNIVERSE_PARQUET,
        )

    ima_holdings = None
    if config.HIST_IMA_HOLDINGS_PARQUET.exists():
        ima_holdings = _validate_ima(pd.read_parquet(config.HIST_IMA_HOLDINGS_PARQUET))
    else:
        logger.warning(
            "ima_holdings.parquet absent at %s — IMA-portfolio-specific "
            "evaluation will be skipped.", config.HIST_IMA_HOLDINGS_PARQUET,
        )

    delistings = None
    if config.HIST_DELIST_PARQUET.exists():
        delistings = _validate_delistings(pd.read_parquet(config.HIST_DELIST_PARQUET))
    else:
        logger.info(
            "delistings.parquet absent — terminal-value handling relies solely on "
            "the last observed price in prices.parquet.")

    store = HistoricalStore(
        features=features, prices=prices, universe=universe,
        ima_holdings=ima_holdings, delistings=delistings,
    )
    _report_coverage(store)
    return store


def _report_coverage(store: HistoricalStore) -> None:
    feat = store.features
    dates = store.feature_dates
    logger.info(
        "Historical store loaded: %d feature snapshots, %d (date,ticker) rows, "
        "%s .. %s; %d price rows over %d tickers.",
        len(dates), len(feat),
        dates[0].date() if dates else "—", dates[-1].date() if dates else "—",
        len(store.prices), store.prices["ticker"].nunique(),
    )
    per_date = feat.groupby("date")["ticker"].nunique()
    thin = per_date[per_date < 10]
    for d, n in per_date.items():
        logger.debug("  coverage %s: %d tickers", pd.Timestamp(d).date(), n)
    if not thin.empty:
        logger.warning(
            "%d snapshot date(s) have < 10 tickers — cross-sectional scores will "
            "be noisy there: %s",
            len(thin), [str(pd.Timestamp(d).date()) for d in thin.index[:5]],
        )


# =============================================================================
# Honesty guard: look-ahead assertion
# =============================================================================
def assert_no_lookahead(features_asof: pd.DataFrame, asof: pd.Timestamp) -> None:
    """Raise if any feature row used for snapshot ``asof`` is dated after it.

    This is the core anti-look-ahead guard: a feature snapshot that postdates
    the rebalance date would leak the future into the score.
    """
    if "date" not in features_asof.columns:
        return
    future = features_asof[features_asof["date"] > pd.Timestamp(asof)]
    if not future.empty:
        sample = future[["date", "ticker"]].head(5).to_dict("records")
        raise LookAheadError(
            f"LOOK-AHEAD DETECTED at as-of {pd.Timestamp(asof).date()}: "
            f"{len(future)} feature row(s) are dated AFTER the snapshot. "
            f"Sample: {sample}"
        )


def features_asof(store: HistoricalStore, asof: pd.Timestamp) -> pd.DataFrame:
    """Most-recent feature snapshot for each ticker with date <= ``asof``.

    Returns a frame indexed by ticker (the shape ``scoring`` / ``pca_cluster``
    expect), carrying ``Sector`` and the config.FEATURES columns, plus the
    point-in-time ``date`` each row was sourced from.
    """
    asof = pd.Timestamp(asof).normalize()
    elig = store.features[store.features["date"] <= asof]
    assert_no_lookahead(elig, asof)   # belt-and-suspenders
    if elig.empty:
        return pd.DataFrame()
    # take the latest snapshot per ticker that is still <= asof
    elig = elig.sort_values("date")
    latest = elig.groupby("ticker", as_index=False).tail(1)
    latest = latest.set_index("ticker")
    return latest


def universe_asof(store: HistoricalStore, asof: pd.Timestamp) -> list[str] | None:
    """Tickers that were index members as of ``asof`` (latest membership <= asof).

    Returns ``None`` when no universe file was provided (caller must fall back
    to feature-set membership and warn about survivorship).
    """
    if not store.has_universe:
        return None
    asof = pd.Timestamp(asof).normalize()
    u = store.universe[store.universe["date"] <= asof]
    if u.empty:
        return []
    latest = u.sort_values("date").groupby("ticker", as_index=False).tail(1)
    return sorted(latest[latest["in_index"]]["ticker"].tolist())


def ima_asof(store: HistoricalStore, asof: pd.Timestamp) -> dict[str, float]:
    """IMA holdings (ticker -> weight) effective as of ``asof``.

    Uses the most-recent holdings date <= asof.  Empty dict when no IMA file.
    """
    if store.ima_holdings is None or store.ima_holdings.empty:
        return {}
    asof = pd.Timestamp(asof).normalize()
    h = store.ima_holdings[store.ima_holdings["date"] <= asof]
    if h.empty:
        return {}
    eff_date = h["date"].max()
    snap = h[h["date"] == eff_date]
    return dict(zip(snap["ticker"], snap["weight"]))


# =============================================================================
# Synthetic store generator (for tests / the acceptance run)
# =============================================================================
def generate_synthetic_store(
    *,
    n_tickers: int = 120,
    start: str = "2017-01-31",
    n_quarters: int = 28,
    seed: int = 42,
    signal_strength: float = 1.6,
) -> HistoricalStore:
    """Generate a self-consistent synthetic point-in-time store and write it
    to ``config.HISTORICAL_DIR``.

    The data-generating process is deliberately HONEST: a latent per-(date,
    ticker) risk level drives BOTH the feature snapshot (so a high composite
    score is realizable) AND the forward price path's drawdown probability (so
    the screener genuinely predicts severe drawdowns — and the placebo test, which
    shuffles scores, must collapse the IC to ~0).  Some names delist to a low
    terminal value, exercising the survivorship + delisting paths.
    """
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start).normalize()
    # Quarter-end rebalance dates.
    dates = pd.date_range(start=start_ts, periods=n_quarters, freq="QE")
    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]
    sectors = np.array(["Tech", "Industrials", "Healthcare", "Energy",
                        "Consumer", "Materials"])
    ticker_sector = {t: sectors[rng.integers(len(sectors))] for t in tickers}

    # Daily price calendar spanning the snapshots plus a forward buffer for the
    # longest forward horizon, so every snapshot has full forward prices.
    buffer_days = (max(config.FORWARD_RETURN_HORIZONS_MONTHS) + 2) * 31
    price_index = pd.date_range(
        start=dates[0] - pd.Timedelta(days=5),
        end=dates[-1] + pd.Timedelta(days=buffer_days),
        freq="B",
    )

    feat_rows: list[dict] = []
    # Per-ticker latent risk that persists with quarterly shocks.
    latent = {t: rng.normal(0, 1) for t in tickers}

    # Build daily price paths per ticker, with drawdown events whose intensity
    # tracks the latent risk active at the time.
    price_panel = pd.DataFrame(index=price_index, columns=tickers, dtype=float)
    delist_records: list[dict] = []

    # We need the latent risk over time to drive drawdowns; store a step series.
    latent_by_date: dict[str, pd.Series] = {}
    for t in tickers:
        steps = []
        lv = latent[t]
        for _ in dates:
            lv = 0.8 * lv + 0.6 * rng.normal()
            steps.append(lv)
        latent_by_date[t] = pd.Series(steps, index=dates)

    for t in tickers:
        # Daily log returns: small positive drift + risk-scaled vol + jumps.
        risk_daily = latent_by_date[t].reindex(price_index, method="ffill")
        risk_daily = risk_daily.fillna(latent_by_date[t].iloc[0])
        base_vol = 0.012 + 0.004 * _sigmoid(risk_daily.to_numpy())
        drift = 0.0003 - 0.0002 * _sigmoid(risk_daily.to_numpy())
        shocks = rng.normal(0, 1, len(price_index)) * base_vol + drift
        # Jump (severe-drawdown) hazard increases with risk.
        jump_p = 0.0008 + 0.004 * _sigmoid(risk_daily.to_numpy() * signal_strength)
        jumps = rng.random(len(price_index)) < jump_p
        shocks = shocks + jumps * rng.normal(-0.12, 0.04, len(price_index))
        path = 100.0 * np.exp(np.cumsum(shocks))
        price_panel[t] = path

    # Delist ~8% of names: cut their price series at a random date to a low
    # terminal value (a terminal blow-up).
    n_delist = max(1, int(0.08 * n_tickers))
    delist_tickers = rng.choice(tickers, size=n_delist, replace=False)
    for t in delist_tickers:
        # Delist somewhere in the back half so most snapshots still see it.
        di = rng.integers(len(price_index) // 2, len(price_index) - 5)
        d_date = price_index[di]
        terminal = float(price_panel[t].iloc[di]) * float(rng.uniform(0.1, 0.4))
        price_panel.loc[price_panel.index > d_date, t] = np.nan
        price_panel.loc[d_date, t] = terminal
        delist_records.append(
            {"ticker": t, "delist_date": d_date, "final_value": terminal}
        )

    # Feature snapshots: features are noisy reads of the latent risk, oriented
    # by config.RISK_DIRECTION so that scoring.compute_composite_scores ranks
    # high-latent names as high risk.
    for d in dates:
        for t in tickers:
            lv = float(latent_by_date[t].loc[d])
            row = {"date": d, "ticker": t, "Sector": ticker_sector[t]}
            for f in config.FEATURES:
                direction = config.RISK_DIRECTION.get(f, 1)
                # higher latent risk -> higher feature if direction==1 else lower
                base = direction * lv * signal_strength + rng.normal(0, 1.0)
                row[f] = _scale_feature(f, base, rng)
            feat_rows.append(row)

    features = pd.DataFrame(feat_rows)

    # Universe membership: in_index True for snapshots before a name delists.
    uni_rows: list[dict] = []
    delist_map = {r["ticker"]: r["delist_date"] for r in delist_records}
    for d in dates:
        for t in tickers:
            dd = delist_map.get(t)
            in_idx = dd is None or d <= dd
            uni_rows.append({"date": d, "ticker": t, "in_index": bool(in_idx)})
    universe = pd.DataFrame(uni_rows)

    # IMA holdings: ~18 names per date, weights summing to 1, drifting over time.
    ima_rows: list[dict] = []
    held = list(rng.choice(tickers, size=18, replace=False))
    for d in dates:
        # occasionally rotate one name
        if rng.random() < 0.3:
            held[int(rng.integers(len(held)))] = tickers[int(rng.integers(n_tickers))]
        live = [t for t in held if delist_map.get(t) is None or d <= delist_map[t]]
        if not live:
            live = held
        w = rng.uniform(0.5, 1.5, len(live))
        w = w / w.sum()
        for t, wi in zip(live, w):
            ima_rows.append({"date": d, "ticker": t, "weight": float(wi)})
    ima_holdings = pd.DataFrame(ima_rows)

    # Long-format prices.
    prices = (
        price_panel.reset_index()
        .melt(id_vars="index", var_name="ticker", value_name="adj_close")
        .rename(columns={"index": "date"})
        .dropna(subset=["adj_close"])
    )
    delistings = pd.DataFrame(delist_records)

    # Persist.
    config.HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.HIST_FEATURES_PARQUET, index=False)
    universe.to_parquet(config.HIST_UNIVERSE_PARQUET, index=False)
    ima_holdings.to_parquet(config.HIST_IMA_HOLDINGS_PARQUET, index=False)
    prices.to_parquet(config.HIST_PRICES_PARQUET, index=False)
    delistings.to_parquet(config.HIST_DELIST_PARQUET, index=False)
    logger.info(
        "Synthetic store written to %s (%d snapshots, %d tickers, %d delistings).",
        config.HISTORICAL_DIR, len(dates), n_tickers, len(delist_records),
    )
    return load_historical_store()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _scale_feature(name: str, base: float, rng) -> float:
    """Map a standardized signal into a plausible raw range for ``name`` so the
    synthetic features look like the real ones (scale is irrelevant to scoring,
    which percentile-ranks, but realistic ranges aid debugging)."""
    s = _sigmoid(base)
    if name == "altman_z":
        return float(0.5 + 6.0 * (1 - s) + rng.normal(0, 0.2))
    if name in ("short_pct_float",):
        return float(1.0 + 25.0 * s)
    if name in ("momentum_30d", "momentum_90d"):
        return float(0.4 * (base / 3.0))
    if name in ("volatility_60d",):
        return float(0.15 + 0.5 * s)
    if name in ("pe_ratio", "ev_to_ebitda"):
        return float(5.0 + 40.0 * s)
    if name in ("net_debt_to_ebitda",):
        return float(-1.0 + 8.0 * s)
    if name in ("filing_count_90d", "relative_volume"):
        return float(max(0.0, base + 3.0))
    return float(base)
