"""Individual stock audit — sanity-check specific tickers.

Produces a per-ticker breakdown showing exactly which features are pushing
the stock to its position on each PC. Critical for diagnosing cases like
MYRG +70% but appearing near the centroid: the audit will show whether
momentum_30d genuinely loaded high or whether something washed out the signal.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def audit_ticker(
    ticker: str,
    pca_result,
    clean_features: pd.DataFrame,
    cluster_result,
    raw_features: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict:
    if ticker not in pca_result.scores.index:
        return {"error": f"{ticker} not in PCA universe"}

    findings: dict = {"ticker": ticker}
    feature_cols = pca_result.feature_cols

    # 1. Raw values + percentile rank, with imputation flag
    feature_summary = []
    for feat in feature_cols:
        if feat not in clean_features.columns:
            continue
        raw_val = clean_features.loc[ticker, feat]
        try:
            pct_rank = float(clean_features[feat].rank(pct=True).loc[ticker] * 100)
        except KeyError:
            pct_rank = float("nan")
        was_imputed = (
            bool(pd.isna(raw_features.loc[ticker, feat]))
            if (ticker in raw_features.index and feat in raw_features.columns)
            else False
        )
        feature_summary.append({
            "feature": feat,
            "value": float(raw_val) if pd.notna(raw_val) else float("nan"),
            "percentile": pct_rank,
            "imputed": was_imputed,
        })
    findings["features"] = feature_summary

    # 2. PC scores + percentile within universe
    pc_summary = []
    for pc in pca_result.scores.columns:
        score = float(pca_result.scores.loc[ticker, pc])
        pct_in_universe = float(
            (pca_result.scores[pc] < score).sum() / len(pca_result.scores) * 100
        )
        pc_summary.append({
            "pc": pc,
            "score": score,
            "universe_percentile": pct_in_universe,
        })
    findings["pc_scores"] = pc_summary

    # 3. Per-PC contribution = standardized_feature × loading
    scaler = pca_result.scaler
    raw_arr = clean_features.loc[ticker, feature_cols].to_numpy(dtype=float).reshape(1, -1)
    standardized = scaler.transform(raw_arr)[0]

    pc_contributions: dict[str, list[dict]] = {}
    for i, pc in enumerate(pca_result.scores.columns):
        loadings = pca_result.loadings[pc].to_numpy()
        contributions = standardized * loadings
        pairs = list(zip(feature_cols, contributions))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        pc_contributions[pc] = [
            {"feature": f, "contribution": float(c)}
            for f, c in pairs[:5]
        ]
    findings["pc_contributions"] = pc_contributions

    # 4. Cluster + centroid distances
    cluster = int(cluster_result.assignments.loc[ticker])
    cluster_tier = cluster_result.tier_labels.get(cluster, "?")
    findings["cluster"] = cluster
    findings["cluster_tier"] = cluster_tier

    score_vec = pca_result.scores.loc[ticker].to_numpy()
    centroid_distances = []
    for c_id, centroid in enumerate(cluster_result.centroids):
        centroid_distances.append({
            "cluster": int(c_id),
            "tier": cluster_result.tier_labels.get(int(c_id), "?"),
            "distance": float(np.linalg.norm(score_vec - centroid)),
            "is_assigned": bool(c_id == cluster),
        })
    centroid_distances.sort(key=lambda x: x["distance"])
    findings["centroid_distances"] = centroid_distances

    # 5. Nearest neighbors in PC space
    distances = ((pca_result.scores - pca_result.scores.loc[ticker]) ** 2).sum(axis=1).pow(0.5)
    nearest = distances.drop(ticker).nsmallest(10)
    findings["nearest_neighbors"] = [
        {"ticker": tk, "distance": float(d)} for tk, d in nearest.items()
    ]

    if output_dir is not None:
        lines = [f"INDIVIDUAL STOCK AUDIT: {ticker}", "=" * 60, ""]
        lines.append("PC SCORES (universe percentile in parens):")
        for pc in pc_summary:
            lines.append(
                f"  {pc['pc']}: {pc['score']:+.2f} "
                f"({pc['universe_percentile']:.0f}th pct)"
            )
        lines.append("")
        lines.append(f"CLUSTER: {cluster} ({cluster_tier})")
        if centroid_distances:
            assigned = next(c for c in centroid_distances if c["is_assigned"])
            others = [c for c in centroid_distances if not c["is_assigned"]]
            lines.append(f"Distance to assigned centroid: {assigned['distance']:.2f}")
            if others:
                no = others[0]
                lines.append(
                    f"Distance to nearest other centroid: {no['distance']:.2f} "
                    f"({no['tier']})"
                )
        lines.append("")
        lines.append("TOP CONTRIBUTORS PER PC:")
        for pc, contribs in pc_contributions.items():
            lines.append(f"\n  {pc}:")
            for c in contribs:
                lines.append(
                    f"    {c['feature']:<24} contribution = {c['contribution']:+.3f}"
                )
        lines.append("")
        lines.append("KEY FEATURE VALUES (sorted by distance from median):")
        sorted_feats = sorted(
            feature_summary,
            key=lambda x: abs((x.get("percentile") or 50) - 50),
            reverse=True,
        )
        for f in sorted_feats[:10]:
            imp = " [IMPUTED]" if f["imputed"] else ""
            lines.append(
                f"  {f['feature']:<24} value={f['value']:+.3f}  "
                f"percentile={f['percentile']:.0f}{imp}"
            )
        lines.append("")
        lines.append("NEAREST NEIGHBORS (in PC space):")
        for n in findings["nearest_neighbors"][:5]:
            lines.append(f"  {n['ticker']:<8} d={n['distance']:.2f}")
        (output_dir / f"audit_{ticker}.txt").write_text("\n".join(lines))
        logger.info("Wrote individual audit for %s", ticker)

    return findings


def audit_portfolio(
    portfolio_tickers: list[str],
    pca_result,
    clean_features: pd.DataFrame,
    cluster_result,
    raw_features: pd.DataFrame,
    output_dir: Path,
    extra_tickers: list[str] | None = None,
) -> dict:
    audit_dir = output_dir / "individual_audits"
    audit_dir.mkdir(exist_ok=True)
    all_tickers = list(portfolio_tickers)
    if extra_tickers:
        all_tickers.extend(t for t in extra_tickers if t not in all_tickers)
    return {
        tk: audit_ticker(
            tk, pca_result, clean_features, cluster_result, raw_features, audit_dir
        )
        for tk in all_tickers
    }
