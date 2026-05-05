"""Stability test — does the PCA + clustering survive perturbation?

We run PCA + KMeans on N random subsamples of the standardized feature matrix
and measure:

* Eigenvector cosine similarity to the baseline run (sign-invariant). >0.7
  is the conventional "stable" threshold.
* Adjusted Rand Index of cluster assignments vs baseline on the subsample.
  >0.5 = moderate, >0.7 = high stability.

Unstable PCs change meaning when the universe shifts; unstable clusters
shouldn't be presented as authoritative risk strata.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def test_subsample_stability(
    clean_features: pd.DataFrame,
    feature_cols: list[str],
    n_components: int,
    n_clusters: int,
    n_iterations: int = 10,
    subsample_frac: float = 0.80,
    output_dir: Path | None = None,
) -> dict:
    rng = np.random.default_rng(42)

    X = clean_features[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca_baseline = PCA(n_components=n_components, random_state=42)
    scores_baseline = pca_baseline.fit_transform(Xs)
    km_baseline = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels_baseline = km_baseline.fit_predict(scores_baseline)

    n_total = len(X)
    sample_size = int(n_total * subsample_frac)

    eigenvector_sims: list[dict] = []
    cluster_aris: list[float] = []

    for it in range(n_iterations):
        idx = rng.choice(n_total, size=sample_size, replace=False)
        Xs_sample = Xs[idx]

        pca_iter = PCA(n_components=n_components, random_state=it)
        scores_iter = pca_iter.fit_transform(Xs_sample)

        pc_sims: dict[str, float] = {}
        for i, pc_name in enumerate([f"PC{j+1}" for j in range(n_components)]):
            v_base = pca_baseline.components_[i]
            v_iter = pca_iter.components_[i]
            denom = np.linalg.norm(v_base) * np.linalg.norm(v_iter)
            cos = abs(float(np.dot(v_base, v_iter) / denom)) if denom > 0 else 0.0
            pc_sims[pc_name] = cos
        eigenvector_sims.append(pc_sims)

        km_iter = KMeans(n_clusters=n_clusters, n_init=20, random_state=it)
        labels_iter = km_iter.fit_predict(scores_iter)
        ari = adjusted_rand_score(labels_baseline[idx], labels_iter)
        cluster_aris.append(float(ari))

    pc_sim_means = {
        pc: float(np.mean([s[pc] for s in eigenvector_sims]))
        for pc in eigenvector_sims[0].keys()
    }

    findings = {
        "n_iterations": n_iterations,
        "subsample_frac": subsample_frac,
        "eigenvector_similarity_means": pc_sim_means,
        "cluster_ari_mean": float(np.mean(cluster_aris)),
        "cluster_ari_std": float(np.std(cluster_aris)),
        "cluster_ari_min": float(np.min(cluster_aris)),
        "cluster_ari_max": float(np.max(cluster_aris)),
    }

    warnings: list[str] = []
    for pc, sim in pc_sim_means.items():
        if sim < 0.7:
            warnings.append(
                f"UNSTABLE_PC: {pc} mean cosine similarity {sim:.2f} across "
                f"subsamples (need >0.7 for reliable interpretation). This "
                f"component is sensitive to which stocks are included."
            )
    if findings["cluster_ari_mean"] < 0.5:
        warnings.append(
            f"UNSTABLE_CLUSTERING: mean ARI {findings['cluster_ari_mean']:.2f} "
            f"across subsamples. Stocks shift between clusters depending on "
            f"which 20% is excluded. The clustering is not robust."
        )
    findings["warnings"] = warnings

    if output_dir is not None:
        lines = ["STABILITY TEST", "=" * 60, ""]
        lines.append(f"Subsample fraction: {subsample_frac:.0%}")
        lines.append(f"Iterations: {n_iterations}")
        lines.append("")
        lines.append("Eigenvector stability (cosine similarity to baseline):")
        for pc, sim in pc_sim_means.items():
            verdict = "STABLE" if sim >= 0.7 else "UNSTABLE"
            lines.append(f"  {pc}: {sim:.3f}  [{verdict}]")
        lines.append("")
        lines.append(
            f"Cluster stability (Adjusted Rand Index): "
            f"mean={findings['cluster_ari_mean']:.3f} "
            f"min={findings['cluster_ari_min']:.3f} "
            f"max={findings['cluster_ari_max']:.3f}"
        )
        lines.append("  ARI > 0.7 = highly stable, 0.5-0.7 = moderate, <0.5 = unstable")
        lines.append("")
        if warnings:
            lines.append("WARNINGS:")
            for w in warnings:
                lines.append(f"  ⚠ {w}")
                lines.append("")
        (output_dir / "stability_test.txt").write_text("\n".join(lines))
        logger.info("Wrote stability_test.txt")

    return findings
