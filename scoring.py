"""Composite risk scoring, tier assignment, portfolio report, opportunity screen."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# =============================================================================
# Composite risk score (0-100 percentile-mean)
# =============================================================================
def compute_composite_scores(
    features: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return per-feature percentile ranks (already flipped for risk direction)
    plus a ``composite_score`` column (mean of percentile ranks)."""
    feature_cols = feature_cols or config.FEATURES
    feature_cols = [c for c in feature_cols if c in features.columns]

    pct_ranks = pd.DataFrame(index=features.index)
    for c in feature_cols:
        r = features[c].rank(pct=True) * 100.0
        if config.RISK_DIRECTION.get(c, 1) == -1:
            r = 100.0 - r
        pct_ranks[c] = r

    composite = pct_ranks.mean(axis=1)
    out = pct_ranks.copy()
    out["composite_score"] = composite
    out["risk_tier"] = out["composite_score"].apply(score_to_tier)
    return out


def score_to_tier(score: float) -> str:
    for lo, hi, label in config.SCORE_BUCKETS:
        if lo <= score < hi:
            return label
    return config.SCORE_BUCKETS[-1][2]


def format_combined_label(cluster_label: str, composite_score: float) -> str:
    """Pair the cluster tier with the composite-score percentile.

    The cluster label (Stable / Mainstream / Elevated) is a coarse summary;
    the composite score is the granular measure. Showing both at once is the
    honest committee-presentable format — e.g. ``"Mainstream (47th pct)"``.
    """
    return f"{cluster_label} ({composite_score:.0f}th pct)"


# =============================================================================
# Portfolio report
# =============================================================================
def top_risk_drivers(
    percentile_ranks: pd.DataFrame,
    ticker: str,
    k: int = 3,
) -> list[tuple[str, float]]:
    """Return the ``k`` features with the highest risk-percentile for a ticker."""
    if ticker not in percentile_ranks.index:
        return []
    row = percentile_ranks.loc[ticker].drop(
        labels=[c for c in ("composite_score", "risk_tier") if c in percentile_ranks.columns]
    )
    top = row.sort_values(ascending=False).head(k)
    return [(name, float(val)) for name, val in top.items()]


def sector_comparison(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    ticker: str,
) -> dict:
    """Compare ticker's composite score to its sector's median."""
    if ticker not in features.index or "Sector" not in features.columns:
        return {}
    sector = features.loc[ticker, "Sector"]
    peers = features[features["Sector"] == sector].index
    peers = peers.intersection(scores.index).difference([ticker])
    if len(peers) == 0:
        return {"sector": sector, "peers": 0}
    median = float(scores.loc[peers, "composite_score"].median())
    own = float(scores.loc[ticker, "composite_score"])
    return {
        "sector": sector,
        "peers": int(len(peers)),
        "sector_median": median,
        "own": own,
        "delta": own - median,
        "label": "riskier than sector" if own > median else "safer than sector",
    }


def build_portfolio_report(
    features: pd.DataFrame,
    percentile_ranks: pd.DataFrame,
    cluster_assignments: pd.Series,
    tier_labels: dict[int, str],
    trajectories,
) -> pd.DataFrame:
    """Build the IMA portfolio risk detail table."""
    from trajectory import classify_trajectory

    rows = []
    for tk, weight in config.PORTFOLIO.items():
        if tk not in features.index:
            rows.append({
                "Ticker": tk,
                "Weight": weight,
                "Status": "NOT_IN_UNIVERSE",
            })
            continue
        score = float(percentile_ranks.loc[tk, "composite_score"])
        tier = percentile_ranks.loc[tk, "risk_tier"]
        cluster = int(cluster_assignments.get(tk, -1))
        cluster_label = tier_labels.get(cluster, "?") if cluster >= 0 else "?"

        drivers = top_risk_drivers(percentile_ranks, tk, k=3)
        driver_str = "; ".join(f"{n}({v:.0f})" for n, v in drivers)

        sector_info = sector_comparison(features, percentile_ranks, tk)

        trajectory_dir = "N/A"
        drift_2q = np.nan
        if trajectories is not None:
            trajectory_dir = classify_trajectory(
                trajectories, _ClusterResultFacade(tier_labels), tk
            ) if hasattr(trajectories, "cluster_paths") else "N/A"
            drift_2q = trajectories.two_quarter_drift.get(tk, np.nan)

        rows.append({
            "Ticker": tk,
            "Weight": weight,
            "Sector": features.loc[tk, "Sector"],
            "Composite_Score": round(score, 1),
            "Risk_Tier": tier,
            "Cluster": cluster,
            "Cluster_Label": cluster_label,
            "Combined_Display": format_combined_label(cluster_label, score),
            "Altman_Z": _safe_round(features.loc[tk].get("altman_z"), 2),
            "Short_Pct_Float": _safe_round(features.loc[tk].get("short_pct_float"), 1),
            "Momentum_90d": _safe_round(features.loc[tk].get("momentum_90d"), 3),
            "Net_Debt_EBITDA": _safe_round(features.loc[tk].get("net_debt_to_ebitda"), 2),
            "Volatility_60d": _safe_round(features.loc[tk].get("volatility_60d"), 3),
            "Top_Risk_Drivers": driver_str,
            "Sector_Comparison": sector_info.get("label", ""),
            "Sector_Delta": _safe_round(sector_info.get("delta"), 1),
            "Trajectory": trajectory_dir,
            "Two_Q_Drift": _safe_round(drift_2q, 2),
        })
    return pd.DataFrame(rows)


class _ClusterResultFacade:
    """Minimal adapter exposing tier_labels to classify_trajectory."""
    def __init__(self, tier_labels):
        self.tier_labels = tier_labels


def _safe_round(x, n):
    try:
        if x is None or pd.isna(x):
            return np.nan
        return round(float(x), n)
    except (TypeError, ValueError):
        return np.nan


# =============================================================================
# Opportunity screen (Cluster B contrarian candidates)
# =============================================================================
def opportunity_screen(
    features: pd.DataFrame,
    percentile_ranks: pd.DataFrame,
    limit: int = 25,
) -> pd.DataFrame:
    """Stocks with intact fundamentals but bearish market positioning.

    The original criterion paired Altman Z with Piotroski F ≥ 5; Piotroski
    was dropped from the model (yfinance only returns 4 quarters), so the
    fundamentals-intact gate now leans on altman_z + a positive accruals
    quality screen as the closest replacement.
    """
    df = features.copy()
    mask = (
        (df["altman_z"] > 2.0)
        & (df["accruals_ratio"] > 0.7)         # OCF/NI healthy
        & (df["short_pct_float"] > 8.0)
        & (df["momentum_90d"] < 0)
    )
    cand = df.loc[mask].copy()
    if cand.empty:
        return pd.DataFrame()

    cand["composite_score"] = percentile_ranks["composite_score"].reindex(cand.index)
    cand["risk_tier"] = percentile_ranks["risk_tier"].reindex(cand.index)

    # Rank by how bearish the sentiment is (high short + very negative mom)
    cand["contrarian_score"] = (
        cand["short_pct_float"].rank(pct=True) * 100
        - cand["momentum_90d"].rank(pct=True) * 100
    )
    cand = cand.sort_values("contrarian_score", ascending=False).head(limit)
    return cand[[
        "Company", "Sector", "altman_z", "accruals_ratio", "short_pct_float",
        "momentum_90d", "composite_score", "risk_tier", "contrarian_score",
    ]]
