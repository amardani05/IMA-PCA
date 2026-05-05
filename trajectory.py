"""Historical trajectory mapping through PCA risk space.

For each of ``TRAJECTORY_QUARTERS`` past snapshots, re-compute features as-of
that date, project them using the SAME fitted scaler + PCA, and re-assign
clusters using the SAME fitted k-means. Produces per-stock paths through PC
space plus drift flags for early warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from feature_engine import RawBundle, build_features, clean_features
from pca_cluster import ClusterResult, PCAResult, transform

logger = logging.getLogger(__name__)


# =============================================================================
# Snapshot dates
# =============================================================================
def snapshot_dates(
    as_of: pd.Timestamp | None = None,
    n: int = config.TRAJECTORY_QUARTERS,
) -> list[pd.Timestamp]:
    """Return ``n`` evenly spaced quarterly dates ending at (or before) ``as_of``."""
    end = as_of or pd.Timestamp.utcnow().tz_localize(None).normalize()
    # Most recent first, then walk back by ~91 days
    dates = [end - pd.Timedelta(days=91 * k) for k in range(n)]
    return sorted(dates)


# =============================================================================
# Trajectory computation
# =============================================================================
@dataclass
class TrajectoryResult:
    snapshots: list[pd.Timestamp]
    pc_paths: dict[str, pd.DataFrame]     # ticker -> DataFrame (snapshot_date x PC)
    cluster_paths: dict[str, list[int]]   # ticker -> [cluster_ids] per snapshot
    distance_traveled: pd.Series          # ticker -> total PC path length
    two_quarter_drift: pd.Series          # ticker -> euclidean drift last 2 quarters
    cluster_transitions: pd.Series        # ticker -> count of cluster changes
    drift_flags: pd.DataFrame             # per-ticker flags


def compute_trajectories(
    universe_df: pd.DataFrame,
    bundles: dict[str, RawBundle],
    prices: pd.DataFrame,
    pca_result: PCAResult,
    cluster_result: ClusterResult,
    filing_counts_by_date: dict[pd.Timestamp, dict[str, int | None]] | None = None,
    snapshots: list[pd.Timestamp] | None = None,
    tickers_filter: list[str] | None = None,
) -> TrajectoryResult:
    """Compute PC trajectories for every stock with enough history.

    If ``tickers_filter`` is provided, only those tickers are projected through
    PC space at each snapshot — the typical use is portfolio-only trajectory
    mapping, which is far cheaper than re-scoring all 600 names per quarter.
    """
    snapshots = snapshots or snapshot_dates()
    filing_counts_by_date = filing_counts_by_date or {}
    feature_cols = pca_result.feature_cols

    if tickers_filter is not None:
        universe_df = universe_df[universe_df["Ticker"].isin(tickers_filter)].copy()
        logger.info("Trajectory restricted to %d tickers", len(universe_df))

    # per-snapshot scored frames
    per_snap_scores: list[pd.DataFrame] = []
    per_snap_clusters: list[pd.Series] = []

    for snap in snapshots:
        logger.info("Trajectory snapshot: %s", snap.date())
        fc = filing_counts_by_date.get(snap, {})
        feats = build_features(
            universe_df, bundles, prices, filing_counts=fc, as_of=snap
        )
        # Use same-strict cleaning but keep index aligned to current universe
        clean, _ = clean_features(feats)
        scores = transform(pca_result, clean[feature_cols])

        # k-means predict uses the fitted model on current data
        cluster_ids = pd.Series(
            cluster_result.kmeans.predict(scores.to_numpy()),
            index=scores.index,
            name=f"cluster_{snap.date()}",
        )
        per_snap_scores.append(scores)
        per_snap_clusters.append(cluster_ids)

    # Assemble paths per ticker
    all_tickers = set().union(*[s.index for s in per_snap_scores])
    pc_paths: dict[str, pd.DataFrame] = {}
    cluster_paths: dict[str, list[int]] = {}
    distance_traveled = {}
    two_quarter_drift = {}
    cluster_transitions = {}

    n_components = per_snap_scores[-1].shape[1]

    for tk in all_tickers:
        coords = []
        clusters = []
        for snap, scores, cids in zip(snapshots, per_snap_scores, per_snap_clusters):
            if tk in scores.index:
                coords.append(scores.loc[tk].values)
                clusters.append(int(cids.loc[tk]))
            else:
                coords.append([np.nan] * n_components)
                clusters.append(-1)
        arr = np.array(coords)
        path_df = pd.DataFrame(arr,
                               index=pd.Index(snapshots, name="date"),
                               columns=[f"PC{i+1}" for i in range(n_components)])
        pc_paths[tk] = path_df
        cluster_paths[tk] = clusters

        # Valid rows
        valid_mask = ~np.isnan(arr).any(axis=1)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) >= 2:
            dist = 0.0
            for a, b in zip(valid_idx[:-1], valid_idx[1:]):
                dist += float(np.linalg.norm(arr[b] - arr[a]))
            distance_traveled[tk] = dist

            # Last 2 quarters drift: distance between last two valid snapshots
            if len(valid_idx) >= 2:
                a, b = valid_idx[-2], valid_idx[-1]
                two_quarter_drift[tk] = float(np.linalg.norm(arr[b] - arr[a]))
        else:
            distance_traveled[tk] = np.nan
            two_quarter_drift[tk] = np.nan

        transitions = 0
        prev = None
        for c in clusters:
            if c == -1:
                continue
            if prev is not None and c != prev:
                transitions += 1
            prev = c
        cluster_transitions[tk] = transitions

    dt = pd.Series(distance_traveled, name="distance_traveled")
    drift2q = pd.Series(two_quarter_drift, name="two_quarter_drift")
    trans = pd.Series(cluster_transitions, name="cluster_transitions")

    # Drift flags
    sigma = drift2q.std(skipna=True) or 1.0
    mean = drift2q.mean(skipna=True) or 0.0
    threshold = mean + config.DRIFT_SIGMA_THRESHOLD * sigma

    flags = pd.DataFrame(index=list(all_tickers))
    flags["large_2q_drift"] = drift2q >= threshold
    flags["crossed_cluster_last_q"] = False

    for tk in all_tickers:
        cp = [c for c in cluster_paths[tk] if c >= 0]
        if len(cp) >= 2 and cp[-1] != cp[-2]:
            flags.loc[tk, "crossed_cluster_last_q"] = True

    return TrajectoryResult(
        snapshots=snapshots,
        pc_paths=pc_paths,
        cluster_paths=cluster_paths,
        distance_traveled=dt,
        two_quarter_drift=drift2q,
        cluster_transitions=trans,
        drift_flags=flags,
    )


# =============================================================================
# Trajectory direction classification (for each stock)
# =============================================================================
def classify_trajectory(
    trajectory: TrajectoryResult,
    cluster_result: ClusterResult,
    ticker: str,
) -> str:
    """Return 'Improving' / 'Stable' / 'Deteriorating' based on cluster path."""
    cp = [c for c in trajectory.cluster_paths.get(ticker, []) if c >= 0]
    if len(cp) < 2:
        return "Unknown"

    # Risk rank by tier label; TIER_LABELS is ordered safest->riskiest
    tier_labels = config.TIER_LABELS
    rank = {cid: tier_labels.index(lbl) for cid, lbl in cluster_result.tier_labels.items()}
    ranks = [rank[c] for c in cp]

    if ranks[-1] > ranks[0]:
        return "Deteriorating"
    if ranks[-1] < ranks[0]:
        return "Improving"
    return "Stable"


def borderline_stocks(
    scores: pd.DataFrame,
    cluster_result: ClusterResult,
    radius: float = config.CLUSTER_BOUNDARY_RADIUS,
) -> pd.Index:
    """Stocks within ``radius`` of a cluster boundary in PC space."""
    from pca_cluster import nearest_cluster_distance

    dists = nearest_cluster_distance(scores, cluster_result)
    return dists.index[dists["boundary_gap"] < radius]
