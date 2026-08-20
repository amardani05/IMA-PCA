"""Dump pipeline outputs as JSON and copy chart assets into ``webapp/public/``.

The React app reads everything from ``webapp/public/`` at runtime:
    data/*.json       — tables and summary stats
    charts/*.png      — static matplotlib figures
    interactive/*.{html,json}  — plotly interactive figures (HTML to iframe, JSON for react-plotly)
    meta.json         — pipeline metadata (timestamp, universe size, k, ...)

Run standalone via ``python -m webapp_export`` after the pipeline, or let
``main.py`` invoke :func:`export_all` at the end of every run.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

WEBAPP_PUBLIC = config.PROJECT_ROOT / "webapp" / "public"
DATA_DIR = WEBAPP_PUBLIC / "data"
CHART_DIR = WEBAPP_PUBLIC / "charts"
INTERACTIVE_DIR = WEBAPP_PUBLIC / "interactive"


def _ensure_dirs() -> None:
    for d in (DATA_DIR, CHART_DIR, INTERACTIVE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> records, with NaN -> None and numpy scalars -> python."""
    df = df.replace({np.nan: None})
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _write_json(payload, stem: str) -> None:
    path = DATA_DIR / f"{stem}.json"
    path.write_text(json.dumps(payload, default=str, indent=None))
    logger.debug("wrote %s (%d bytes)", path.name, path.stat().st_size)


# =============================================================================
# Individual exporters
# =============================================================================
def export_universe_table(
    features: pd.DataFrame,
    pca_result,
    cluster_result,
    percentile_ranks: pd.DataFrame,
) -> None:
    """Full universe row-per-ticker table with features, PCs, cluster, score."""
    df = features.copy()
    df["Ticker"] = df.index
    for pc in pca_result.scores.columns:
        df[pc] = pca_result.scores[pc]
    df["cluster"] = cluster_result.assignments
    df["cluster_style"] = df["cluster"].map(cluster_result.style_labels)
    df["composite_score"] = percentile_ranks["composite_score"]
    df["score_percentile"] = percentile_ranks["score_percentile"]
    df["risk_tier"] = percentile_ranks["risk_tier"]
    df["is_portfolio"] = df["Ticker"].isin(config.PORTFOLIO)
    df["weight"] = df["Ticker"].map(config.PORTFOLIO).fillna(0.0)

    cols_front = ["Ticker", "Company", "Sector", "Industry", "market_cap",
                  "is_financial", "is_portfolio", "weight", "cluster",
                  "cluster_style", "composite_score", "score_percentile",
                  "risk_tier"]
    pcs = [c for c in df.columns if c.startswith("PC")]
    feats = [c for c in config.FEATURES if c in df.columns]
    cols_order = cols_front + pcs + feats
    df = df[[c for c in cols_order if c in df.columns]]

    _write_json(_to_records(df), "universe")


def export_portfolio(portfolio_report: pd.DataFrame) -> None:
    _write_json(_to_records(portfolio_report), "portfolio")


def export_clusters(cluster_result) -> None:
    char = cluster_result.characterization.reset_index()
    _write_json(_to_records(char), "clusters")
    _write_json({
        "k": cluster_result.k,
        "silhouette": cluster_result.silhouette,
        "style_labels": {int(k): v for k, v in cluster_result.style_labels.items()},
        "risk_rank": {int(k): int(v) for k, v in cluster_result.risk_rank.items()},
        "style_colors": dict(cluster_result.style_colors),
        "centroids": cluster_result.centroids.tolist(),
        "diagnostics": _to_records(cluster_result.diagnostics.reset_index()),
    }, "cluster_meta")


def export_pca(pca_result) -> None:
    loadings = pca_result.loadings.reset_index().rename(columns={"index": "feature"})
    _write_json(_to_records(loadings), "pca_loadings")
    summary = []
    for i, pc in enumerate(pca_result.scores.columns):
        summary.append({
            "pc": pc,
            "variance_explained": float(pca_result.variance_explained[i]),
            "cumulative_variance": float(pca_result.cumulative_variance[i]),
            "label": pca_result.pc_labels[pc],
            "top_loadings": _top_loadings(pca_result.loadings[pc]),
        })
    _write_json(summary, "pca_summary")


def _top_loadings(series: pd.Series, k: int = 4) -> list[dict]:
    top = series.abs().sort_values(ascending=False).head(k)
    return [{"feature": name, "loading": float(series[name])} for name in top.index]


def export_opportunities(opportunities: pd.DataFrame) -> None:
    if opportunities.empty:
        _write_json([], "opportunities")
        return
    df = opportunities.reset_index().rename(columns={"index": "Ticker"})
    _write_json(_to_records(df), "opportunities")


def export_drift(drift_alerts: pd.DataFrame) -> None:
    if drift_alerts.empty:
        _write_json([], "drift_alerts")
        return
    df = drift_alerts.reset_index().rename(columns={"index": "Ticker"})
    _write_json(_to_records(df), "drift_alerts")


def export_trajectory(trajectory) -> None:
    """Per-ticker PC path + cluster path across quarterly snapshots."""
    if trajectory is None:
        _write_json({"snapshots": [], "paths": {}}, "trajectory")
        return

    snapshots = [d.isoformat() for d in trajectory.snapshots]
    paths: dict[str, dict] = {}
    for tk, path in trajectory.pc_paths.items():
        cluster_path = trajectory.cluster_paths.get(tk, [])
        coords = []
        for i, (_, row) in enumerate(path.iterrows()):
            coords.append({
                "date": snapshots[i] if i < len(snapshots) else None,
                "cluster": int(cluster_path[i]) if i < len(cluster_path) else None,
                **{pc: (float(row[pc]) if pd.notna(row[pc]) else None)
                   for pc in path.columns},
            })
        paths[tk] = {
            "coords": coords,
            "distance_traveled": float(trajectory.distance_traveled.get(tk, np.nan))
                if tk in trajectory.distance_traveled.index else None,
            "two_quarter_drift": float(trajectory.two_quarter_drift.get(tk, np.nan))
                if tk in trajectory.two_quarter_drift.index else None,
            "cluster_transitions": int(trajectory.cluster_transitions.get(tk, 0)),
        }
    _write_json({"snapshots": snapshots, "paths": paths}, "trajectory")


def _file_age_meta(path: Path) -> dict | None:
    """Modification timestamp for a cache file, or None when absent."""
    if not path.exists():
        return None
    return {
        "updated_at": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat() + "Z",
    }


def export_meta(pca_result, cluster_result, features: pd.DataFrame) -> None:
    # Cluster style order, safest first (by risk rank)
    style_order = [
        cluster_result.style_labels[cid]
        for cid, _rank in sorted(cluster_result.risk_rank.items(), key=lambda kv: kv[1])
    ]
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "universe_size": int(len(features)),
        "n_portfolio": sum(1 for t in config.PORTFOLIO if t in features.index),
        "financials_count": int(features["is_financial"].sum()),
        "n_features": len(config.FEATURES),
        "benchmark": config.BENCHMARK_TICKER,
        "pca": {
            "n_components": len(pca_result.scores.columns),
            "variance_explained": [float(v) for v in pca_result.variance_explained],
            "cumulative_variance": [float(v) for v in pca_result.cumulative_variance],
            "pc_labels": dict(pca_result.pc_labels),
        },
        "clustering": {
            "k": int(cluster_result.k),
            "silhouette": float(cluster_result.silhouette),
            "style_labels": {int(k): v for k, v in cluster_result.style_labels.items()},
            "risk_rank": {int(k): int(v) for k, v in cluster_result.risk_rank.items()},
        },
        # RISK tiers (score-percentile buckets): stoplight, with a neutral middle.
        "tier_order": list(config.TIER_LABELS),
        "tier_colors": {
            "Low Risk": "#2c7a4b",   # deep green
            "In Line":  "#64748b",   # neutral slate
            "Elevated": "#b3001b",   # deep red
        },
        # Cluster STYLES (descriptive groupings): separate palette so styles
        # never read as a risk verdict.
        "style_order": style_order,
        "style_colors": dict(cluster_result.style_colors),
        # Per-source data freshness for the dashboard banner. Short interest is
        # published by exchanges on a ~2-week lag regardless of fetch time.
        "data_freshness": {
            "prices": _file_age_meta(config.PRICE_CACHE),
            "fundamentals": _file_age_meta(config.FUNDAMENTALS_CACHE),
            "sec_filings": _file_age_meta(config.SEC_FILINGS_CACHE),
            "short_interest_note": "exchange-published on a ~2-week lag",
        },
    }
    (WEBAPP_PUBLIC / "meta.json").write_text(json.dumps(payload, indent=2))
    logger.info("Wrote meta.json")


# =============================================================================
# Backtest export
# =============================================================================
def export_backtest(evaluation: dict) -> None:
    """Write the full backtest evaluation bundle to ``data/backtest.json``.

    ``evaluation`` is the dict returned by ``backtest.evaluate`` — decile/tier
    hit-rates with CIs, IC time series + t-stats, ROC + AUC, tier calibration,
    the IMA hit/miss table, strategy equity curves, event counts, and metadata.
    """
    _ensure_dirs()
    _write_json(evaluation, "backtest")
    logger.info("Wrote backtest.json (%d snapshots, %d events)",
                evaluation.get("metadata", {}).get("n_snapshots", 0),
                evaluation.get("base_rate", {}).get("n_events", 0))


# =============================================================================
# Asset copies
# =============================================================================
CHART_FILES = [
    "cluster_scatter_pc1_pc2.png",
    "cluster_scatter_pc2_pc3.png",
    "cluster_scatter_3d.png",
    "trajectory_map.png",
    "portfolio_risk_dashboard.png",
    "cluster_profiles.png",
    "silhouette_analysis.png",
    "pca_loadings.png",
    "risk_score_distribution.png",
    "sector_risk_comparison.png",
]


def copy_chart_assets() -> list[str]:
    copied = []
    for name in CHART_FILES:
        src = config.OUTPUT_DIR / name
        if not src.exists():
            continue
        dst = CHART_DIR / name
        shutil.copy2(src, dst)
        copied.append(name)
    # Plotly HTML + JSON
    plotly_src = config.OUTPUT_DIR / "interactive"
    if plotly_src.exists():
        for f in plotly_src.iterdir():
            if f.suffix in {".html", ".json"}:
                shutil.copy2(f, INTERACTIVE_DIR / f.name)
                copied.append(f"interactive/{f.name}")
    return copied


# =============================================================================
# Top-level entry point
# =============================================================================
def export_all(
    features: pd.DataFrame,
    pca_result,
    cluster_result,
    percentile_ranks: pd.DataFrame,
    portfolio_report: pd.DataFrame,
    opportunities: pd.DataFrame,
    drift_alerts: pd.DataFrame,
    trajectory=None,
) -> None:
    _ensure_dirs()
    export_universe_table(features, pca_result, cluster_result, percentile_ranks)
    export_portfolio(portfolio_report)
    export_clusters(cluster_result)
    export_pca(pca_result)
    export_opportunities(opportunities)
    export_drift(drift_alerts)
    export_trajectory(trajectory)
    export_meta(pca_result, cluster_result, features)
    copied = copy_chart_assets()
    logger.info("webapp_export: %d JSON docs + %d assets → %s",
                8, len(copied), WEBAPP_PUBLIC)
