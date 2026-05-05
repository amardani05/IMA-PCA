"""Cluster quality audit — is k-means actually finding meaningful structure?

Tests:
1. Cluster size balance (warns on dominant or tiny clusters).
2. Per-cluster silhouette (in addition to the overall mean).
3. Within vs between centroid distances (separation ratio).
4. k=3..7 sweep with full metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

logger = logging.getLogger(__name__)


def audit_clusters(pca_result, cluster_result, output_dir: Path) -> dict:
    X = pca_result.scores.to_numpy()
    labels = cluster_result.assignments.to_numpy()
    n_clusters = len(np.unique(labels))
    n_total = len(labels)

    findings: dict = {}
    warnings: list[str] = []

    # 1. Sizes
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    cluster_size_pcts = (cluster_sizes / n_total) * 100
    findings["cluster_sizes"] = {int(k): int(v) for k, v in cluster_sizes.items()}
    findings["cluster_size_pcts"] = {int(k): float(v) for k, v in cluster_size_pcts.items()}

    max_size_pct = float(cluster_size_pcts.max())
    min_size_pct = float(cluster_size_pcts.min())

    if max_size_pct > 50:
        warnings.append(
            f"DOMINANT_CLUSTER: One cluster contains {max_size_pct:.0f}% of the "
            f"universe. K-means is finding 'one big blob' rather than separated "
            f"strata. Either the feature space lacks structure or different "
            f"features would yield better separation."
        )
    if min_size_pct < 3:
        warnings.append(
            f"TINY_CLUSTER: Smallest cluster is {min_size_pct:.1f}% of universe. "
            f"May be an outlier group rather than a meaningful tier — investigate "
            f"merging with a neighbor."
        )

    # 2. Silhouette
    if n_clusters > 1:
        sil_samples = silhouette_samples(X, labels)
        per_cluster_sil = {
            int(c): float(np.mean(sil_samples[labels == c]))
            for c in np.unique(labels)
        }
        findings["overall_silhouette"] = float(np.mean(sil_samples))
        findings["per_cluster_silhouette"] = per_cluster_sil

        for c, s in per_cluster_sil.items():
            if s < 0:
                warnings.append(
                    f"NEGATIVE_SILHOUETTE: Cluster {c} mean silhouette {s:.3f}. "
                    f"Members are on average closer to a different cluster's "
                    f"centroid — assignments unreliable."
                )
            elif s < 0.20:
                warnings.append(
                    f"WEAK_SILHOUETTE: Cluster {c} silhouette {s:.3f}. "
                    f"Boundary is poorly defined."
                )

    # 3. Within vs between
    centroids = cluster_result.centroids
    within_distances: list[float] = []
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts):
            within_distances.append(
                float(np.mean(np.linalg.norm(pts - centroids[c], axis=1)))
            )
    avg_within = float(np.mean(within_distances)) if within_distances else 0.0

    between_distances: list[float] = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between_distances.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    avg_between = float(np.mean(between_distances)) if between_distances else 0.0
    separation_ratio = (avg_between / avg_within) if avg_within > 0 else 0.0

    findings["avg_within_cluster_distance"] = avg_within
    findings["avg_between_cluster_distance"] = avg_between
    findings["separation_ratio"] = separation_ratio

    if separation_ratio < 1.5:
        warnings.append(
            f"POOR_SEPARATION: Between/within ratio {separation_ratio:.2f}. "
            f"Clusters are barely more separated than they are internally spread. "
            f"A ratio of 2.0+ indicates well-separated clusters."
        )

    # 4. k sweep
    k_comparison = []
    for k in range(3, 8):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        kl = km.fit_predict(X)
        sil = silhouette_score(X, kl) if k > 1 else 0.0
        sizes = pd.Series(kl).value_counts()
        size_min = float(sizes.min() / n_total * 100)
        size_max = float(sizes.max() / n_total * 100)
        k_comparison.append({
            "k": k,
            "silhouette": float(sil),
            "smallest_cluster_pct": size_min,
            "largest_cluster_pct": size_max,
            "size_imbalance_ratio": float(size_max / size_min) if size_min > 0 else float("inf"),
        })
    findings["k_comparison"] = k_comparison
    findings["warnings"] = warnings

    pd.DataFrame(k_comparison).to_csv(output_dir / "cluster_k_comparison.csv", index=False)

    lines = ["CLUSTER QUALITY AUDIT", "=" * 60, ""]
    lines.append(f"Selected k: {n_clusters}")
    lines.append(f"Overall silhouette: {findings.get('overall_silhouette', 0):.3f}")
    lines.append(f"Separation ratio: {separation_ratio:.2f} (>2.0 = well-separated)")
    lines.append("")
    lines.append("Cluster sizes:")
    for c, n in sorted(cluster_sizes.items()):
        lines.append(f"  Cluster {c}: {n} stocks ({cluster_size_pcts[c]:.1f}%)")
    lines.append("")
    if "per_cluster_silhouette" in findings:
        lines.append("Per-cluster silhouette:")
        for c, s in sorted(findings["per_cluster_silhouette"].items()):
            lines.append(f"  Cluster {c}: {s:.3f}")
        lines.append("")
    lines.append("k comparison:")
    for row in k_comparison:
        lines.append(
            f"  k={row['k']}: sil={row['silhouette']:.3f}  "
            f"sizes {row['smallest_cluster_pct']:.0f}%-{row['largest_cluster_pct']:.0f}%  "
            f"imbalance={row['size_imbalance_ratio']:.1f}x"
        )
    lines.append("")
    if warnings:
        lines.append("WARNINGS:")
        for w in warnings:
            lines.append(f"  ⚠ {w}")
            lines.append("")

    (output_dir / "cluster_quality.txt").write_text("\n".join(lines))
    logger.info("Wrote cluster_quality.txt")
    return findings
