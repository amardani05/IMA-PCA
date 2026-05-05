"""Ablation study — which features actually drive the cluster structure?

Compares baseline (all features) vs:
- group-only (fundamental / market_positioning / valuation / events)
- leave-one-out (drop each feature and re-run PCA + clustering)

Each variant is scored by adjusted Rand index against baseline labels and by
silhouette. Features whose removal barely moves the needle (ARI ≈ 1.0) are
candidates for removal; group-only configs that still have high ARI tell us
which axis genuinely structures the universe.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


FEATURE_GROUPS: dict[str, list[str]] = {
    "fundamental": [
        "altman_z", "current_ratio", "net_debt_to_ebitda",
        "fcf_yield", "interest_coverage", "accruals_ratio",
        "asset_growth_yoy", "net_issuance_yoy",
    ],
    "market_positioning": [
        "short_pct_float", "momentum_30d", "momentum_90d",
        "volatility_60d", "relative_volume",
    ],
    "valuation": ["pe_ratio", "ev_to_ebitda"],
    "events": ["filing_count_90d"],
}


def _fit_and_cluster(
    clean_features: pd.DataFrame,
    feature_subset: list[str],
    n_components: int,
    n_clusters: int,
):
    if not feature_subset:
        return None, None
    X = clean_features[feature_subset].to_numpy(dtype=float)
    if X.shape[1] < 2:
        return None, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    nc = min(n_components, Xs.shape[1])
    pca = PCA(n_components=nc, random_state=42)
    scores = pca.fit_transform(Xs)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km.fit_predict(scores)
    sil = silhouette_score(scores, labels) if len(set(labels)) > 1 else 0.0
    return labels, float(sil)


def run_ablation(
    clean_features: pd.DataFrame,
    feature_cols: list[str],
    n_components: int,
    n_clusters: int,
    output_dir: Path,
) -> pd.DataFrame:
    baseline_labels, baseline_sil = _fit_and_cluster(
        clean_features, feature_cols, n_components, n_clusters,
    )
    if baseline_labels is None:
        raise RuntimeError("ablation baseline failed — empty feature_cols?")

    rows = [{
        "config": "baseline (all features)",
        "n_features": len(feature_cols),
        "silhouette": baseline_sil,
        "ari_vs_baseline": 1.0,
    }]

    # Group-only
    for group_name, group_feats in FEATURE_GROUPS.items():
        active = [f for f in group_feats if f in feature_cols]
        if len(active) < 2:
            continue
        labels, sil = _fit_and_cluster(clean_features, active, n_components, n_clusters)
        if labels is not None:
            rows.append({
                "config": f"{group_name}_only",
                "n_features": len(active),
                "silhouette": sil,
                "ari_vs_baseline": float(adjusted_rand_score(baseline_labels, labels)),
            })

    # Leave-one-out
    for feat in feature_cols:
        subset = [f for f in feature_cols if f != feat]
        labels, sil = _fit_and_cluster(clean_features, subset, n_components, n_clusters)
        if labels is not None:
            rows.append({
                "config": f"drop_{feat}",
                "n_features": len(subset),
                "silhouette": sil,
                "ari_vs_baseline": float(adjusted_rand_score(baseline_labels, labels)),
            })

    df = pd.DataFrame(rows).sort_values("ari_vs_baseline")
    df.to_csv(output_dir / "ablation_study.csv", index=False)
    logger.info("Wrote ablation_study.csv")

    leave_one_out = df[df["config"].str.startswith("drop_")].sort_values("ari_vs_baseline")
    critical = leave_one_out.head(3)["config"].str.replace("drop_", "", regex=False).tolist()

    lines = ["ABLATION STUDY", "=" * 60, ""]
    lines.append(f"Baseline silhouette: {baseline_sil:.3f}")
    lines.append("")
    lines.append("Group-only configurations (does each group alone reproduce baseline?):")
    for _, row in df[df["config"].str.endswith("_only")].iterrows():
        lines.append(
            f"  {row['config']:<32} sil={row['silhouette']:.3f}  "
            f"ARI={row['ari_vs_baseline']:.3f}"
        )
    lines.append("")
    lines.append("Most critical features (largest cluster impact when dropped):")
    for f in critical:
        lines.append(f"  {f}")
    lines.append("")
    lines.append("Least critical features (drop barely changes clustering):")
    for _, row in leave_one_out.tail(3).iterrows():
        feat = row["config"].replace("drop_", "")
        lines.append(f"  {feat}  (ARI={row['ari_vs_baseline']:.3f})")
    lines.append("")
    lines.append(
        "Interpretation: features with LOW ARI when dropped are doing real work. "
        "Features whose removal barely moves the clustering (ARI ≈ 1.0) are "
        "candidates for removal."
    )
    (output_dir / "ablation_study.txt").write_text("\n".join(lines))
    return df
