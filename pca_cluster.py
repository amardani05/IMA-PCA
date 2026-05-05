"""PCA dimensionality reduction + k-means clustering for the torpedo screener.

Standardizes the 14-feature matrix, runs PCA, picks a k-means k via silhouette
score, characterizes each cluster, and auto-labels PCs + clusters by their
feature signatures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


# =============================================================================
# Heuristic PC labels
# =============================================================================
PC_LABEL_RULES: list[tuple[str, set[str]]] = [
    ("Financial Health", {"altman_z", "piotroski_f", "interest_coverage", "current_ratio"}),
    ("Market Sentiment", {"short_pct_float", "momentum_30d", "momentum_90d"}),
    ("Volatility/Attention", {"volatility_60d", "relative_volume"}),
    ("Valuation Stress", {"pe_ratio", "ev_to_ebitda", "fcf_yield"}),
    ("Event Activity", {"filing_count_90d"}),
    ("Leverage", {"net_debt_to_ebitda"}),
]


def label_pc(loadings: pd.Series, top_k: int = 4) -> str:
    """Assign a human-readable label to a PC by its dominant feature loadings."""
    top = loadings.abs().sort_values(ascending=False).head(top_k).index.tolist()
    best_label, best_hits = "Mixed", 0
    for label, feature_set in PC_LABEL_RULES:
        hits = len(set(top) & feature_set)
        if hits > best_hits:
            best_hits = hits
            best_label = label
    if best_hits == 0:
        return f"Mixed ({', '.join(top[:2])})"
    return best_label


# =============================================================================
# PCA
# =============================================================================
@dataclass
class PCAResult:
    scaler: StandardScaler
    pca: PCA
    feature_cols: list[str]
    scores: pd.DataFrame          # rows = tickers, cols = PC1..PCn
    loadings: pd.DataFrame        # rows = features, cols = PC1..PCn
    variance_explained: np.ndarray
    cumulative_variance: np.ndarray
    pc_labels: dict[str, str]     # "PC1" -> "Financial Health"


def run_pca(
    features: pd.DataFrame,
    n_components: int = config.N_PCA_COMPONENTS,
    feature_cols: list[str] | None = None,
) -> PCAResult:
    """Z-score standardize features and extract ``n_components`` PCs."""
    feature_cols = feature_cols or config.FEATURES
    feature_cols = [c for c in feature_cols if c in features.columns]
    X = features[feature_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(Xs)

    cols = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=features.index, columns=cols)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=cols,
    )

    pc_labels = {pc: label_pc(loadings[pc]) for pc in cols}
    logger.info("PCA variance explained:")
    cum = 0.0
    for i, pc in enumerate(cols):
        v = pca.explained_variance_ratio_[i]
        cum += v
        logger.info("  %s  %.2f%%  (cum %.2f%%)  label=%s",
                    pc, v * 100, cum * 100, pc_labels[pc])

    return PCAResult(
        scaler=scaler,
        pca=pca,
        feature_cols=feature_cols,
        scores=scores_df,
        loadings=loadings,
        variance_explained=pca.explained_variance_ratio_,
        cumulative_variance=np.cumsum(pca.explained_variance_ratio_),
        pc_labels=pc_labels,
    )


def transform(pca_result: PCAResult, features: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted scaler + PCA to a new features frame."""
    X = features[pca_result.feature_cols].to_numpy(dtype=float)
    Xs = pca_result.scaler.transform(X)
    scores = pca_result.pca.transform(Xs)
    cols = [f"PC{i+1}" for i in range(scores.shape[1])]
    return pd.DataFrame(scores, index=features.index, columns=cols)


# =============================================================================
# K-means clustering
# =============================================================================
@dataclass
class ClusterResult:
    k: int
    kmeans: KMeans
    assignments: pd.Series         # ticker -> cluster id
    centroids: np.ndarray          # (k, n_components)
    tier_labels: dict[int, str]    # cluster id -> "Low Risk" / ... / "Critical"
    silhouette: float
    diagnostics: pd.DataFrame      # k_candidates evaluation
    characterization: pd.DataFrame # per-cluster feature means


def _evaluate_k(scores: np.ndarray, k_candidates: Iterable[int]) -> pd.DataFrame:
    rows = []
    for k in k_candidates:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(scores)
        sil = silhouette_score(scores, labels) if k > 1 else np.nan
        ch = calinski_harabasz_score(scores, labels) if k > 1 else np.nan
        rows.append({
            "k": k,
            "silhouette": sil,
            "inertia": km.inertia_,
            "calinski_harabasz": ch,
        })
    return pd.DataFrame(rows).set_index("k")


def _select_k(diag: pd.DataFrame, default_k: int, tol: float) -> int:
    """Pick k with max silhouette; fall back to default if within tol."""
    best_k = int(diag["silhouette"].idxmax())
    best_s = diag.loc[best_k, "silhouette"]
    if default_k in diag.index and (best_s - diag.loc[default_k, "silhouette"]) <= tol:
        return default_k
    return best_k


def _label_clusters_by_risk(
    features: pd.DataFrame,
    assignments: pd.Series,
    feature_cols: list[str],
) -> dict[int, str]:
    """Rank clusters by composite risk score and assign tier labels."""
    # Standardize features once, then flip direction so high = risky for all.
    X = features[feature_cols].copy()
    z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    for c in feature_cols:
        if config.RISK_DIRECTION.get(c, 1) == -1:
            z[c] = -z[c]
    composite = z.mean(axis=1)
    composite = composite.reindex(assignments.index)

    cluster_risk = composite.groupby(assignments).mean().sort_values()
    # Lowest composite = safest = "Low Risk", highest = "Critical"
    labels = config.TIER_LABELS
    out: dict[int, str] = {}
    n_clusters = len(cluster_risk)
    # Map cluster rank (0=safest) to a tier label index (scale into TIER_LABELS)
    for rank, cid in enumerate(cluster_risk.index):
        idx = int(round(rank * (len(labels) - 1) / max(n_clusters - 1, 1)))
        out[int(cid)] = labels[idx]
    return out


def _characterize(
    features: pd.DataFrame,
    assignments: pd.Series,
    feature_cols: list[str],
    tier_labels: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for cid in sorted(assignments.unique()):
        members = assignments[assignments == cid].index
        sub = features.loc[members]
        row = {
            "cluster": cid,
            "tier": tier_labels[int(cid)],
            "n_stocks": len(members),
        }
        for c in feature_cols:
            row[f"{c}_mean"] = sub[c].mean()
            row[f"{c}_median"] = sub[c].median()
        rows.append(row)
    return pd.DataFrame(rows).set_index("cluster")


def run_clustering(
    features: pd.DataFrame,
    pca_result: PCAResult,
    override_k: int | None = None,
) -> ClusterResult:
    """Run k-means on PCA-reduced coords with silhouette-based k selection."""
    X = pca_result.scores.to_numpy()
    diag = _evaluate_k(X, config.K_CANDIDATES)
    logger.info("Cluster diagnostics:\n%s", diag.round(3).to_string())

    k = override_k or _select_k(diag, config.N_CLUSTERS, config.SILHOUETTE_TOLERANCE)
    logger.info("Selected k=%d (silhouette=%.3f)", k, diag.loc[k, "silhouette"])

    km = KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=42)
    labels = km.fit_predict(X)

    assignments = pd.Series(labels, index=pca_result.scores.index, name="cluster")
    tier_labels = _label_clusters_by_risk(features, assignments, pca_result.feature_cols)
    char = _characterize(features, assignments, pca_result.feature_cols, tier_labels)

    return ClusterResult(
        k=k,
        kmeans=km,
        assignments=assignments,
        centroids=km.cluster_centers_,
        tier_labels=tier_labels,
        silhouette=float(diag.loc[k, "silhouette"]),
        diagnostics=diag,
        characterization=char,
    )


def nearest_cluster_distance(scores: pd.DataFrame, cluster_result: ClusterResult) -> pd.DataFrame:
    """Per-stock distance to assigned and nearest-other centroid.

    Returns DataFrame indexed by ticker with columns:
        assigned, assigned_dist, nearest_other, nearest_other_dist, boundary_gap
    """
    X = scores.to_numpy()
    C = cluster_result.centroids  # (k, n_components)
    # (N, k)
    dists = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
    assigned = dists.argmin(axis=1)
    assigned_d = dists[np.arange(len(X)), assigned]

    other_d = dists.copy()
    other_d[np.arange(len(X)), assigned] = np.inf
    nearest_other = other_d.argmin(axis=1)
    nearest_other_d = other_d[np.arange(len(X)), nearest_other]

    return pd.DataFrame({
        "assigned": assigned,
        "assigned_dist": assigned_d,
        "nearest_other": nearest_other,
        "nearest_other_dist": nearest_other_d,
        "boundary_gap": nearest_other_d - assigned_d,
    }, index=scores.index)
