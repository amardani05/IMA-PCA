"""Backtest engine + evaluation for the drawdown-risk screener.

This module answers the question the live screener cannot: *does a high composite
score actually precede a severe drawdown?*  It REUSES the production scoring / PCA /
clustering code unchanged (``scoring.compute_composite_scores``,
``pca_cluster.run_pca`` / ``run_clustering``) — fitting the scaler/PCA PER
SNAPSHOT so nothing from the future leaks into a past cross-section.

Pipeline per rebalance date:
  1. point-in-time universe membership + point-in-time features (look-ahead
     guarded in ``historical_loader``).
  2. score that snapshot with the EXISTING functions; optionally cluster it.
  3. compute forward returns (1/3/6/12m) and forward max-drawdown from prices,
     DELISTING-AWARE (a name that goes to a low terminal value counts, it is
     never silently dropped).
  4. derive the binary severe-drawdown label (forward max-DD >= config.DD_THRESHOLD).

The result is a long "panel" (one row per date x ticker) that every evaluation
function below consumes.  See ``diagnostics/backtest_sanity.py`` for the placebo /
look-ahead / survivorship guards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

import config
import historical_loader as hl
from scoring import compute_composite_scores

logger = logging.getLogger(__name__)


# =============================================================================
# Price accessor (delisting-aware)
# =============================================================================
class PriceAccessor:
    """Fast forward-return / max-drawdown lookups over the long price store.

    A delisted name's series terminates at ``final_value`` (from delistings.parquet
    when present, else the last observed price).  Forward returns and the
    forward drawdown window therefore run to the terminal value, capturing
    terminal blow-ups instead of dropping them.
    """

    def __init__(self, prices: pd.DataFrame, delistings: pd.DataFrame | None):
        self._by_ticker: dict[str, pd.Series] = {}
        for tk, grp in prices.groupby("ticker"):
            s = grp.set_index("date")["adj_close"].sort_index()
            s = s[~s.index.duplicated(keep="last")]
            self._by_ticker[tk] = s
        self._delist: dict[str, tuple[pd.Timestamp, float]] = {}
        if delistings is not None:
            for _, r in delistings.iterrows():
                self._delist[r["ticker"]] = (
                    pd.Timestamp(r["delist_date"]), float(r["final_value"]),
                )

    def has(self, ticker: str) -> bool:
        return ticker in self._by_ticker

    def _price_on_or_before(self, s: pd.Series, when: pd.Timestamp) -> float | None:
        sub = s[s.index <= when]
        return float(sub.iloc[-1]) if len(sub) else None

    def _price_on_or_after(self, s: pd.Series, when: pd.Timestamp) -> float | None:
        sub = s[s.index >= when]
        return float(sub.iloc[0]) if len(sub) else None

    def forward_window(
        self, ticker: str, asof: pd.Timestamp, months: int,
    ) -> pd.Series | None:
        """Price series over [asof, asof+months], delisting-aware.

        If the name delists inside the window, the series ends at the terminal
        value on the delist date.  Returns ``None`` if there is no usable start
        price.
        """
        s = self._by_ticker.get(ticker)
        if s is None or s.empty:
            return None
        end = asof + pd.DateOffset(months=months)
        win = s[(s.index >= asof) & (s.index <= end)]
        if win.empty:
            return None
        dl = self._delist.get(ticker)
        if dl is not None:
            d_date, term = dl
            if d_date <= end:
                # truncate at delist and append terminal value
                win = win[win.index <= d_date]
                win = pd.concat([win, pd.Series([term], index=[d_date])])
                win = win[~win.index.duplicated(keep="last")].sort_index()
        return win

    def forward_return(
        self, ticker: str, asof: pd.Timestamp, months: int,
    ) -> float | None:
        win = self.forward_window(ticker, asof, months)
        if win is None or len(win) < 2:
            return None
        return float(win.iloc[-1] / win.iloc[0] - 1.0)

    def forward_max_drawdown(
        self, ticker: str, asof: pd.Timestamp, months: int,
    ) -> float | None:
        """Worst peak-to-trough decline over the forward window (>= 0).

        Peak runs from the snapshot date forward; a terminal-value delisting is
        included so a name that craters to near-zero registers a ~100% DD.
        """
        win = self.forward_window(ticker, asof, months)
        if win is None or len(win) < 2:
            return None
        vals = win.to_numpy(dtype=float)
        running_peak = np.maximum.accumulate(vals)
        drawdowns = (running_peak - vals) / running_peak
        return float(np.nanmax(drawdowns))


# =============================================================================
# Engine
# =============================================================================
@dataclass
class BacktestConfig:
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    rebalance: str = config.BACKTEST_REBALANCE          # "M" or "Q"
    horizon_months: int = config.DD_HORIZON_MONTHS
    dd_threshold: float = config.DD_THRESHOLD
    portfolio_only: bool = False
    run_clusters: bool = True


@dataclass
class BacktestResult:
    panel: pd.DataFrame
    cfg: BacktestConfig
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)
    survivorship_safe: bool = True


def _select_rebalance_dates(store: hl.HistoricalStore, cfg: BacktestConfig) -> list:
    """Feature-snapshot dates filtered to the requested window + cadence.

    We rebalance only on dates that actually have a feature snapshot (you can't
    score a cross-section you don't have).  Cadence "Q" keeps quarter-ends; "M"
    keeps everything monthly.
    """
    dates = pd.DatetimeIndex(sorted(store.features["date"].unique()))
    if cfg.start is not None:
        dates = dates[dates >= pd.Timestamp(cfg.start)]
    if cfg.end is not None:
        dates = dates[dates <= pd.Timestamp(cfg.end)]
    if cfg.rebalance.upper().startswith("Q"):
        # keep the last snapshot in each calendar quarter
        ser = pd.Series(dates, index=dates)
        dates = pd.DatetimeIndex(
            ser.groupby([dates.year, dates.quarter]).last().values
        ).sort_values()
    return list(dates)


def run_backtest(
    store: hl.HistoricalStore,
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    """Build the per-(date, ticker) backtest panel from a historical store."""
    cfg = cfg or BacktestConfig()
    px = PriceAccessor(store.prices, store.delistings)
    rebalance_dates = _select_rebalance_dates(store, cfg)
    survivorship_safe = store.has_universe
    if not survivorship_safe:
        logger.warning(
            "SURVIVORSHIP BIAS: no point-in-time universe; using feature-set "
            "membership. Delisted names that left the index before a snapshot "
            "are invisible — event rates are biased LOW.")

    rows: list[dict] = []
    horizons = config.FORWARD_RETURN_HORIZONS_MONTHS
    for asof in rebalance_dates:
        feats = hl.features_asof(store, asof)        # look-ahead guarded inside
        if feats.empty:
            continue
        members = hl.universe_asof(store, asof)
        if members is not None:
            feats = feats[feats.index.isin(members)]
        ima_w = hl.ima_asof(store, asof)
        if cfg.portfolio_only:
            feats = feats[feats.index.isin(ima_w.keys())]
        # require the model's features; median-impute the odd gap (mirrors the
        # live clean_features step without re-implementing it).
        fcols = [c for c in config.FEATURES if c in feats.columns]
        fmat = feats[fcols].apply(pd.to_numeric, errors="coerce")
        fmat = fmat.fillna(fmat.median())
        scoreable = fmat.dropna(how="all")
        if len(scoreable) < 5:
            logger.debug("Skipping %s: only %d scoreable names", asof, len(scoreable))
            continue
        feats = feats.loc[scoreable.index]
        feats[fcols] = fmat.loc[scoreable.index]

        # --- EXISTING scoring, fit on THIS snapshot only ---
        ranks = compute_composite_scores(feats, fcols)
        clusters = _maybe_cluster(feats, cfg)

        for tk in feats.index:
            score = float(ranks.loc[tk, "composite_score"])
            pctile = float(ranks.loc[tk, "score_percentile"])
            tier = ranks.loc[tk, "risk_tier"]
            rec = {
                "date": asof,
                "ticker": tk,
                "Sector": feats.loc[tk].get("Sector", "Unknown"),
                "composite_score": score,
                "score_percentile": pctile,
                "risk_tier": tier,
                "cluster": clusters.get(tk, {}).get("cluster") if clusters else None,
                "cluster_style": clusters.get(tk, {}).get("style") if clusters else None,
                "is_ima": tk in ima_w,
                "ima_weight": float(ima_w.get(tk, 0.0)),
            }
            for h in horizons:
                rec[f"fwd_ret_{h}m"] = px.forward_return(tk, asof, h)
            mdd = px.forward_max_drawdown(tk, asof, cfg.horizon_months)
            rec["fwd_maxdd"] = mdd
            rec["label"] = (
                int(mdd >= cfg.dd_threshold) if mdd is not None else np.nan
            )
            rec["delisted"] = tk in px._delist
            rec["period_ret"] = None  # filled below
            rows.append(rec)

    panel = pd.DataFrame(rows)
    panel = _add_period_returns(panel, px, rebalance_dates)
    logger.info(
        "Backtest panel: %d rows over %d rebalance dates; %d severe-drawdown events "
        "(base rate %.1f%%).",
        len(panel), len(rebalance_dates),
        int(panel["label"].sum(skipna=True)) if "label" in panel else 0,
        100 * panel["label"].mean(skipna=True) if len(panel) else 0.0,
    )
    return BacktestResult(
        panel=panel, cfg=cfg, rebalance_dates=list(rebalance_dates),
        survivorship_safe=survivorship_safe,
    )


def _maybe_cluster(feats: pd.DataFrame, cfg: BacktestConfig) -> dict | None:
    """Per-snapshot PCA + k-means (fit on this cross-section only). Best-effort:
    clustering can fail on tiny snapshots, which must not abort the backtest."""
    if not cfg.run_clusters or len(feats) < 20:
        return None
    try:
        from pca_cluster import run_pca, run_clustering
        pca_res = run_pca(feats)
        clus = run_clustering(feats, pca_res)
        out: dict[str, dict] = {}
        for tk in feats.index:
            cid = int(clus.assignments.get(tk, -1))
            out[tk] = {"cluster": cid, "style": clus.style_labels.get(cid)}
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("Per-snapshot clustering skipped: %s", exc)
        return None


def _add_period_returns(
    panel: pd.DataFrame, px: PriceAccessor, rebalance_dates: list,
) -> pd.DataFrame:
    """Realized return from each rebalance date to the NEXT one (for the
    quarterly-rebalanced strategy backtest), delisting-aware."""
    if panel.empty:
        return panel
    dates = sorted(rebalance_dates)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    months_between = {}
    for d in dates[:-1]:
        delta = (nxt[d].year - d.year) * 12 + (nxt[d].month - d.month)
        months_between[d] = max(1, delta)
    out = []
    for d, mo in months_between.items():
        idx = panel.index[panel["date"] == d]
        for i in idx:
            tk = panel.at[i, "ticker"]
            out.append((i, px.forward_return(tk, d, mo)))
    for i, r in out:
        panel.at[i, "period_ret"] = r
    return panel


# =============================================================================
# Statistics helpers
# =============================================================================
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial rate. Returns (phat, lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (phat, max(0.0, center - half), min(1.0, center + half))


def _newey_west_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """Mean of a time series with a Newey-West (HAC) corrected t-stat.

    Used for Fama-MacBeth: ``x`` is the series of per-cross-section ICs; we test
    whether its mean differs from zero accounting for autocorrelation.
    """
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], dtype=float)
    n = len(x)
    if n < 2:
        return (float(np.mean(x)) if n else float("nan"), float("nan"))
    mean = float(np.mean(x))
    dem = x - mean
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
        lags = max(1, min(lags, n - 1))
    gamma0 = np.dot(dem, dem) / n
    var = gamma0
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        cov = np.dot(dem[L:], dem[:-L]) / n
        var += 2 * w * cov
    se = np.sqrt(var / n)
    t = mean / se if se > 0 else float("nan")
    return (mean, t)


# =============================================================================
# Evaluation: decile & tier hit-rates (THE test of whether it works)
# =============================================================================
def _hit_rate_block(sub: pd.DataFrame) -> dict:
    valid = sub.dropna(subset=["label"])
    n = int(len(valid))
    k = int(valid["label"].sum())
    phat, lo, hi = wilson_interval(k, n)
    return {
        "n": n, "events": k, "rate": phat, "ci_low": lo, "ci_high": hi,
        "thin": n < config.MIN_EVENTS_PER_CELL or k < config.MIN_EVENTS_PER_CELL,
    }


def decile_hit_rates(panel: pd.DataFrame) -> dict:
    """Event rate per composite-score decile, pooled and per-year, with Wilson
    CIs and thin-cell flags."""
    valid = panel.dropna(subset=["label", "composite_score"]).copy()
    if valid.empty:
        return {"pooled": [], "per_year": {}}
    valid["decile"] = pd.qcut(
        valid["composite_score"].rank(method="first"), 10, labels=False
    ) + 1
    pooled = []
    for dcl, sub in valid.groupby("decile"):
        blk = _hit_rate_block(sub)
        blk["decile"] = int(dcl)
        blk["mean_score"] = float(sub["composite_score"].mean())
        pooled.append(blk)
    per_year: dict[str, list] = {}
    valid["year"] = valid["date"].dt.year
    for yr, ysub in valid.groupby("year"):
        blocks = []
        for dcl, sub in ysub.groupby("decile"):
            blk = _hit_rate_block(sub)
            blk["decile"] = int(dcl)
            blocks.append(blk)
        per_year[str(int(yr))] = blocks
    return {"pooled": pooled, "per_year": per_year}


def tier_hit_rates(panel: pd.DataFrame) -> dict:
    """Event rate per risk tier (Low Risk / In Line / Elevated) + monotonicity test."""
    valid = panel.dropna(subset=["label"]).copy()
    order = list(config.TIER_LABELS)
    pooled = []
    rates_in_order = []
    for tier in order:
        sub = valid[valid["risk_tier"] == tier]
        blk = _hit_rate_block(sub)
        blk["tier"] = tier
        pooled.append(blk)
        rates_in_order.append(blk["rate"])
    # Monotonic increasing Low Risk -> In Line -> Elevated?
    clean_rates = [r for r in rates_in_order if not np.isnan(r)]
    monotonic = all(
        clean_rates[i] <= clean_rates[i + 1] for i in range(len(clean_rates) - 1)
    ) and len(clean_rates) == len(order)
    elevated = next((b for b in pooled if b["tier"] == config.BACKTEST_TOP_TIER), None)
    stable = next((b for b in pooled if b["tier"] == config.BACKTEST_BOTTOM_TIER), None)
    spread = (
        (elevated["rate"] - stable["rate"])
        if elevated and stable and not np.isnan(elevated["rate"])
        and not np.isnan(stable["rate"]) else float("nan")
    )
    per_year: dict[str, list] = {}
    valid["year"] = valid["date"].dt.year
    for yr, ysub in valid.groupby("year"):
        blocks = []
        for tier in order:
            blk = _hit_rate_block(ysub[ysub["risk_tier"] == tier])
            blk["tier"] = tier
            blocks.append(blk)
        per_year[str(int(yr))] = blocks
    return {
        "pooled": pooled, "per_year": per_year,
        "monotonic": bool(monotonic), "elevated_minus_low": spread,
        "tier_order": order,
    }


# =============================================================================
# Information Coefficient (Fama-MacBeth, Newey-West)
# =============================================================================
def information_coefficient(panel: pd.DataFrame) -> dict:
    """Per-cross-section Spearman IC of score vs forward return and vs forward
    max-DD, averaged Fama-MacBeth with a Newey-West t-stat.

    A *negative* IC vs forward return is the "good" direction (high risk score ->
    low return); IC vs forward max-DD should be POSITIVE (high score -> deeper DD).
    """
    ret_col = f"fwd_ret_{config.DD_HORIZON_MONTHS}m"
    if ret_col not in panel.columns:
        ret_col = "fwd_ret_6m"
    series_ret, series_dd, dates = [], [], []
    for d, sub in panel.groupby("date"):
        s = sub.dropna(subset=["composite_score"])
        rr = s.dropna(subset=[ret_col])
        dd = s.dropna(subset=["fwd_maxdd"])
        ic_r = (
            stats.spearmanr(rr["composite_score"], rr[ret_col]).correlation
            if len(rr) >= 5 else np.nan
        )
        ic_d = (
            stats.spearmanr(dd["composite_score"], dd["fwd_maxdd"]).correlation
            if len(dd) >= 5 else np.nan
        )
        series_ret.append(ic_r)
        series_dd.append(ic_d)
        dates.append(pd.Timestamp(d))
    mean_r, t_r = _newey_west_tstat(series_ret)
    mean_d, t_d = _newey_west_tstat(series_dd)
    ts = []
    for d, r, dd in zip(dates, series_ret, series_dd):
        ts.append({
            "date": d.isoformat(),
            "ic_return": None if np.isnan(r) else float(r),
            "ic_maxdd": None if np.isnan(dd) else float(dd),
        })
    return {
        "time_series": ts,
        "ic_return_mean": _nan_to_none(mean_r),
        "ic_return_tstat": _nan_to_none(t_r),
        "ic_maxdd_mean": _nan_to_none(mean_d),
        "ic_maxdd_tstat": _nan_to_none(t_d),
        "horizon_months": config.DD_HORIZON_MONTHS,
        "n_cross_sections": int(sum(1 for r in series_ret if not np.isnan(r))),
    }


# =============================================================================
# Classification: score as a severe-drawdown classifier
# =============================================================================
def classification_metrics(panel: pd.DataFrame) -> dict:
    valid = panel.dropna(subset=["label", "composite_score"])
    y = valid["label"].astype(int).to_numpy()
    s = valid["composite_score"].to_numpy()
    base = float(y.mean()) if len(y) else float("nan")
    out = {"base_rate": base, "n": int(len(y)),
           "roc": [], "auc": None, "tier_thresholds": []}
    if len(y) == 0 or y.min() == y.max():
        return out  # AUC undefined with a single class
    try:
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thr = roc_curve(y, s)
        out["auc"] = float(roc_auc_score(y, s))
        # subsample ROC points so the JSON stays small
        idx = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
        out["roc"] = [
            {"fpr": float(fpr[i]), "tpr": float(tpr[i]),
             "threshold": float(thr[i]) if np.isfinite(thr[i]) else None}
            for i in idx
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("ROC/AUC unavailable: %s", exc)
    # precision/recall/lift at each tier's lower PERCENTILE bound
    pctile = valid["score_percentile"].to_numpy() if "score_percentile" in valid else s
    for lo, hi, label in config.TIER_PERCENTILE_BUCKETS:
        pred = pctile >= lo
        tp = int(((pred) & (y == 1)).sum())
        fp = int(((pred) & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        lift = (precision / base) if base and not np.isnan(precision) else float("nan")
        out["tier_thresholds"].append({
            "tier": label, "score_floor": lo,
            "precision": _nan_to_none(precision), "recall": _nan_to_none(recall),
            "lift": _nan_to_none(lift), "n_flagged": int(pred.sum()),
        })
    return out


# =============================================================================
# Calibration / reliability
# =============================================================================
def calibration(panel: pd.DataFrame) -> dict:
    """Reliability curve: predicted bin (score decile midpoint as a probability
    proxy) vs realized event rate, plus the per-tier predicted-vs-realized."""
    valid = panel.dropna(subset=["label", "composite_score"]).copy()
    bins = []
    if not valid.empty:
        valid["bin"] = pd.qcut(
            valid["composite_score"].rank(method="first"), 10,
            labels=False, duplicates="drop",
        )
        for b, sub in valid.groupby("bin"):
            blk = _hit_rate_block(sub)
            bins.append({
                "bin": int(b),
                "mean_score": float(sub["composite_score"].mean()),
                "predicted": float(sub["composite_score"].mean() / 100.0),
                "realized": blk["rate"],
                "ci_low": blk["ci_low"], "ci_high": blk["ci_high"], "n": blk["n"],
            })
    tiers = []
    for tier in config.TIER_LABELS:
        sub = valid[valid["risk_tier"] == tier]
        blk = _hit_rate_block(sub)
        blk["tier"] = tier
        tiers.append(blk)
    return {"reliability": bins, "tier_calibration": tiers}


# =============================================================================
# Base-rate honesty
# =============================================================================
def base_rate_summary(panel: pd.DataFrame, result: "BacktestResult") -> dict:
    valid = panel.dropna(subset=["label"])
    by_year = valid.groupby(valid["date"].dt.year)["label"].agg(["sum", "count"])
    per_year = [
        {"year": int(y), "events": int(r["sum"]), "n": int(r["count"]),
         "rate": float(r["sum"] / r["count"]) if r["count"] else None}
        for y, r in by_year.iterrows()
    ]
    n_years = max(1, valid["date"].dt.year.nunique())
    return {
        "n_observations": int(len(valid)),
        "n_events": int(valid["label"].sum()),
        "base_rate": float(valid["label"].mean()) if len(valid) else None,
        "events_per_year": float(valid["label"].sum() / n_years),
        "per_year": per_year,
        "n_snapshots": len(result.rebalance_dates),
        "survivorship_safe": result.survivorship_safe,
        "dd_threshold": config.DD_THRESHOLD,
        "horizon_months": config.DD_HORIZON_MONTHS,
    }


# =============================================================================
# IMA-portfolio-specific evaluation
# =============================================================================
def ima_evaluation(panel: pd.DataFrame) -> dict:
    """Everything the IMA explicitly asked for: per-holding hit/miss, model score
    on holdings vs universe over time, named caught/missed events, and the
    avoid-top-tier counterfactual on the sleeve."""
    ima = panel[panel["is_ima"]].dropna(subset=["label"]).copy()
    if ima.empty:
        return {"available": False}

    # 1. per-holding hit/miss table
    hit_miss = []
    for _, r in ima.sort_values(["date", "ticker"]).iterrows():
        flagged = r["risk_tier"] == config.BACKTEST_TOP_TIER
        event = bool(r["label"] == 1)
        hit_miss.append({
            "date": pd.Timestamp(r["date"]).date().isoformat(),
            "ticker": r["ticker"],
            "score": round(float(r["composite_score"]), 1),
            "tier": r["risk_tier"],
            "flagged_elevated": bool(flagged),
            "fwd_maxdd": _nan_to_none(r["fwd_maxdd"]),
            "severe_dd": event,
            "outcome": _confusion_label(flagged, event),
        })

    # 2. model average score on IMA holdings vs universe, over time
    score_ts = []
    for d, sub in panel.groupby("date"):
        ima_sub = sub[sub["is_ima"]]
        if ima_sub.empty:
            continue
        score_ts.append({
            "date": pd.Timestamp(d).date().isoformat(),
            "ima_mean_score": float(ima_sub["composite_score"].mean()),
            "universe_mean_score": float(sub["composite_score"].mean()),
            "n_holdings": int(len(ima_sub)),
        })

    # 3. caught vs missed severe drawdowns the IMA actually held, named
    held_events = ima[ima["label"] == 1]
    caught = held_events[held_events["risk_tier"] == config.BACKTEST_TOP_TIER]
    missed = held_events[held_events["risk_tier"] != config.BACKTEST_TOP_TIER]
    caught_named = _name_events(caught)
    missed_named = _name_events(missed)

    # confusion totals
    flagged = ima["risk_tier"] == config.BACKTEST_TOP_TIER
    ev = ima["label"] == 1
    confusion = {
        "true_positive": int((flagged & ev).sum()),
        "false_positive": int((flagged & ~ev).sum()),
        "false_negative": int((~flagged & ev).sum()),
        "true_negative": int((~flagged & ~ev).sum()),
    }

    # 4. counterfactual: drop top-tier holdings each period, weight-renormalized.
    counterfactual = _ima_counterfactual(panel)

    return {
        "available": True,
        "hit_miss": hit_miss,
        "score_time_series": score_ts,
        "caught_events": caught_named,
        "missed_events": missed_named,
        "confusion": confusion,
        "counterfactual": counterfactual,
    }


def _name_events(df: pd.DataFrame) -> list[dict]:
    return [
        {"date": pd.Timestamp(r["date"]).date().isoformat(),
         "ticker": r["ticker"], "score": round(float(r["composite_score"]), 1),
         "tier": r["risk_tier"], "fwd_maxdd": _nan_to_none(r["fwd_maxdd"])}
        for _, r in df.sort_values("fwd_maxdd", ascending=False).iterrows()
    ]


def _confusion_label(flagged: bool, event: bool) -> str:
    if flagged and event:
        return "CAUGHT (true positive)"
    if flagged and not event:
        return "false alarm"
    if not flagged and event:
        return "MISSED (false negative)"
    return "correctly cleared"


def _ima_counterfactual(panel: pd.DataFrame) -> dict:
    """Compare the actual IMA-weighted sleeve return/max-DD against a variant
    that drops the model's top-tier holdings (renormalizing weights) each
    rebalance.  Uses ``period_ret`` (return to the next rebalance)."""
    df = panel[panel["is_ima"]].dropna(subset=["period_ret"]).copy()
    if df.empty:
        return {"available": False}
    actual_rets, avoid_rets, dates = [], [], []
    for d, sub in df.groupby("date"):
        w = sub["ima_weight"].to_numpy(dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(sub))
        w = w / w.sum()
        actual = float(np.dot(w, sub["period_ret"].to_numpy(dtype=float)))
        keep = sub["risk_tier"] != config.BACKTEST_TOP_TIER
        if keep.any():
            wk = sub.loc[keep, "ima_weight"].to_numpy(dtype=float)
            if wk.sum() <= 0:
                wk = np.ones(int(keep.sum()))
            wk = wk / wk.sum()
            avoid = float(np.dot(wk, sub.loc[keep, "period_ret"].to_numpy(dtype=float)))
        else:
            avoid = actual
        actual_rets.append(actual)
        avoid_rets.append(avoid)
        dates.append(pd.Timestamp(d))
    a_stats = _equity_stats(actual_rets, dates)
    v_stats = _equity_stats(avoid_rets, dates)
    return {
        "available": True,
        "actual": a_stats,
        "avoid_top_tier": v_stats,
        "delta_cagr": _safe_sub(v_stats["cagr"], a_stats["cagr"]),
        "delta_maxdd": _safe_sub(v_stats["max_drawdown"], a_stats["max_drawdown"]),
        "equity_curve": [
            {"date": dt.date().isoformat(), "actual": ae, "avoid_top_tier": ve}
            for dt, ae, ve in zip(dates, a_stats["equity"], v_stats["equity"])
        ],
    }


# =============================================================================
# Strategy backtest (S&P 600 universe)
# =============================================================================
def strategy_backtest(panel: pd.DataFrame) -> dict:
    """Two strategies on the universe panel, quarterly rebalanced:

    * ``avoid_top_tier`` — long-only equal-weight of every non-Elevated name vs
      an equal-weight universe benchmark (IJR proxy when no IJR price is fed).
    * ``long_short`` — sector-neutral: short Elevated, long Low Risk, dollar-neutral.

    Transaction cost = ``config.BACKTEST_COST_BPS`` one-way on turnover.  Reports
    CAGR, vol, Sharpe, max-DD, hit-rate vs benchmark, and average turnover.
    """
    df = panel.dropna(subset=["period_ret"]).copy()
    if df.empty:
        return {"available": False}
    cost = config.BACKTEST_COST_BPS / 1e4
    dates = sorted(df["date"].unique())

    bench_r, avoid_r, ls_r = [], [], []
    avoid_turn, ls_turn = [], []
    prev_avoid: dict[str, float] = {}
    prev_ls: dict[str, float] = {}

    for d in dates:
        sub = df[df["date"] == d]
        r = sub.set_index("ticker")["period_ret"].astype(float)

        # benchmark: equal-weight universe
        bench_r.append(float(r.mean()))

        # avoid-top-tier: EW of non-Elevated
        keep = sub[sub["risk_tier"] != config.BACKTEST_TOP_TIER]["ticker"].tolist()
        wa = _equal_weights(keep)
        gross_a = sum(wa[t] * r[t] for t in keep) if keep else float(r.mean())
        avoid_turn.append(_turnover(prev_avoid, wa))
        net_a = gross_a - cost * avoid_turn[-1]
        avoid_r.append(net_a)
        prev_avoid = wa

        # sector-neutral long/short: within each sector long Low Risk, short Elevated
        w_ls = _sector_neutral_long_short(sub)
        gross_ls = sum(w_ls[t] * r[t] for t in w_ls if t in r.index)
        ls_turn.append(_turnover(prev_ls, w_ls))
        net_ls = gross_ls - cost * ls_turn[-1]
        ls_r.append(net_ls)
        prev_ls = w_ls

    ts = [pd.Timestamp(d) for d in dates]
    bench = _equity_stats(bench_r, ts)
    avoid = _equity_stats(avoid_r, ts)
    ls = _equity_stats(ls_r, ts)
    avoid["hit_rate_vs_bench"] = _hit_rate_vs(avoid_r, bench_r)
    ls["hit_rate_vs_bench"] = _hit_rate_vs(ls_r, bench_r)
    avoid["avg_turnover"] = float(np.mean(avoid_turn)) if avoid_turn else 0.0
    ls["avg_turnover"] = float(np.mean(ls_turn)) if ls_turn else 0.0

    curve = [
        {"date": t.date().isoformat(),
         "benchmark": be, "avoid_top_tier": ae, "long_short": le}
        for t, be, ae, le in zip(ts, bench["equity"], avoid["equity"], ls["equity"])
    ]
    return {
        "available": True,
        "benchmark_label": "Universe EW (IJR proxy)",
        "cost_bps": config.BACKTEST_COST_BPS,
        "rebalance": config.BACKTEST_REBALANCE,
        "benchmark": bench,
        "avoid_top_tier": avoid,
        "long_short": ls,
        "equity_curve": curve,
    }


def _equal_weights(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def _sector_neutral_long_short(sub: pd.DataFrame) -> dict[str, float]:
    """Dollar-neutral weights: +1/N_long on Low Risk, -1/N_short on Elevated,
    balanced within each sector then scaled to gross 1 on each side."""
    longs, shorts = [], []
    for _, grp in sub.groupby("Sector"):
        longs += grp[grp["risk_tier"] == config.BACKTEST_BOTTOM_TIER]["ticker"].tolist()
        shorts += grp[grp["risk_tier"] == config.BACKTEST_TOP_TIER]["ticker"].tolist()
    w: dict[str, float] = {}
    if longs:
        wl = 0.5 / len(longs)
        for t in longs:
            w[t] = w.get(t, 0.0) + wl
    if shorts:
        ws = 0.5 / len(shorts)
        for t in shorts:
            w[t] = w.get(t, 0.0) - ws
    return w


def _turnover(prev: dict[str, float], cur: dict[str, float]) -> float:
    keys = set(prev) | set(cur)
    return float(sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys))


def _hit_rate_vs(strat: list[float], bench: list[float]) -> float:
    pairs = [(s, b) for s, b in zip(strat, bench)]
    if not pairs:
        return float("nan")
    return float(np.mean([1.0 if s > b else 0.0 for s, b in pairs]))


def _equity_stats(rets: list[float], dates: list[pd.Timestamp]) -> dict:
    r = np.array([x for x in rets if x is not None], dtype=float)
    if len(r) == 0:
        return {"cagr": None, "vol": None, "sharpe": None,
                "max_drawdown": None, "equity": [], "total_return": None}
    equity = np.cumprod(1 + r)
    # annualization factor from cadence
    per_year = 4 if config.BACKTEST_REBALANCE.upper().startswith("Q") else 12
    years = len(r) / per_year
    total = float(equity[-1] - 1)
    cagr = float(equity[-1] ** (1 / years) - 1) if years > 0 else float("nan")
    vol = float(np.std(r, ddof=1) * np.sqrt(per_year)) if len(r) > 1 else float("nan")
    sharpe = float((np.mean(r) * per_year) / vol) if vol and vol > 0 else float("nan")
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / peak))
    return {
        "cagr": _nan_to_none(cagr), "vol": _nan_to_none(vol),
        "sharpe": _nan_to_none(sharpe), "max_drawdown": max_dd,
        "total_return": total, "equity": [float(x) for x in equity],
    }


# =============================================================================
# Top-level evaluation bundle
# =============================================================================
def evaluate(result: BacktestResult) -> dict:
    """Run every evaluation and return a single JSON-ready dict."""
    panel = result.panel
    return {
        "metadata": {
            "date_range": [
                result.rebalance_dates[0].date().isoformat(),
                result.rebalance_dates[-1].date().isoformat(),
            ] if result.rebalance_dates else [None, None],
            "n_snapshots": len(result.rebalance_dates),
            "rebalance": result.cfg.rebalance,
            "horizon_months": result.cfg.horizon_months,
            "dd_threshold": result.cfg.dd_threshold,
            "label_definition": (
                f"Severe drawdown = forward {result.cfg.horizon_months}-month "
                f"peak-to-trough max drawdown >= {result.cfg.dd_threshold:.0%} "
                f"from the snapshot date."
            ),
            "survivorship_safe": result.survivorship_safe,
            "cost_bps": config.BACKTEST_COST_BPS,
        },
        "base_rate": base_rate_summary(panel, result),
        "deciles": decile_hit_rates(panel),
        "tiers": tier_hit_rates(panel),
        "information_coefficient": information_coefficient(panel),
        "classification": classification_metrics(panel),
        "calibration": calibration(panel),
        "ima": ima_evaluation(panel),
        "strategy": strategy_backtest(panel),
    }


# =============================================================================
# small utilities
# =============================================================================
def _nan_to_none(x):
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
    except (TypeError, ValueError):
        return None


def _safe_sub(a, b):
    if a is None or b is None:
        return None
    return float(a - b)
