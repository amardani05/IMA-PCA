"""Committee-presentable charts for the risk screener.

All charts are saved as PNGs to ``output/``. Style is kept clean and
professional: neutral backgrounds, readable fonts, ticker labels on IMA
holdings, legends describing tiers + counts.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

import config

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

# Two color domains, deliberately distinct so they never get conflated:
#   - RISK tiers (score-percentile buckets) keep a stoplight green/slate/red.
#   - Cluster STYLES are colored from cluster_result.style_colors (a neutral
#     rank-ordered palette from config.CLUSTER_STYLE_PALETTE).
RISK_TIER_COLORS = {
    "Low Risk": "#2c7a4b",   # deep green
    "In Line":  "#64748b",   # neutral slate — mid tier is not a warning
    "Elevated": "#b3001b",   # deep red
}


def _risk_tier_color(tier: str) -> str:
    return RISK_TIER_COLORS.get(tier, "#666666")


def _style_color(cluster_result, style: str) -> str:
    return getattr(cluster_result, "style_colors", {}).get(style, "#666666")


# =============================================================================
# Cluster scatter
# =============================================================================
# =============================================================================
# 3D cluster scatter with trajectories
# =============================================================================
def plot_cluster_scatter_3d(
    scores: pd.DataFrame,
    cluster_result,
    pc_labels: dict[str, str],
    portfolio_tickers: list[str],
    trajectory=None,
    outfile: str = "cluster_scatter_3d.png",
) -> None:
    """3D PCA scatter over PC1/PC2/PC3, with IMA trajectories overlaid.

    - Universe points colored by cluster style, semi-transparent.
    - Cluster centroids marked with a large black 'X'.
    - Portfolio holdings highlighted with ring markers + text labels.
    - If ``trajectory`` is provided, each holding's quarterly path is drawn
      as a 3D polyline; arrow color encodes direction (red = riskier).
    """
    if not {"PC1", "PC2", "PC3"}.issubset(scores.columns):
        logger.warning("plot_cluster_scatter_3d requires PC1/2/3; skipping")
        return

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    assignments = cluster_result.assignments
    style_labels = cluster_result.style_labels
    handles = []

    for cid in sorted(assignments.unique()):
        mask = assignments == cid
        style = style_labels[int(cid)]
        color = _style_color(cluster_result, style)
        ax.scatter(
            scores.loc[mask, "PC1"],
            scores.loc[mask, "PC2"],
            scores.loc[mask, "PC3"],
            s=14, alpha=0.45, color=color, edgecolors="none",
            depthshade=True,
        )
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=color, markersize=9,
                              label=f"C{cid} · {style} (n={int(mask.sum())})"))

    # Centroids
    centroids = cluster_result.centroids
    comps = list(scores.columns)
    ix = [comps.index(c) for c in ("PC1", "PC2", "PC3")]
    ax.scatter(
        centroids[:, ix[0]], centroids[:, ix[1]], centroids[:, ix[2]],
        marker="X", s=180, c="black", edgecolors="white", linewidths=1.5, depthshade=False,
    )

    # Portfolio ring markers + labels
    for tk in portfolio_tickers:
        if tk not in scores.index:
            continue
        x, y, z = scores.loc[tk, ["PC1", "PC2", "PC3"]]
        ax.scatter([x], [y], [z], s=90, facecolor="none", edgecolors="black",
                   linewidths=1.8, depthshade=False)
        ax.text(x, y, z, f"  {tk}", fontsize=8, fontweight="bold")

    # Trajectories
    if trajectory is not None:
        risk_rank = cluster_result.risk_rank
        for tk in portfolio_tickers:
            path = trajectory.pc_paths.get(tk)
            if path is None:
                continue
            valid = path.dropna(subset=["PC1", "PC2", "PC3"])
            if len(valid) < 2:
                continue
            cluster_path = list(trajectory.cluster_paths.get(tk, []))

            for i in range(len(valid) - 1):
                p0 = valid.iloc[i][["PC1", "PC2", "PC3"]].values
                p1 = valid.iloc[i + 1][["PC1", "PC2", "PC3"]].values
                c0 = cluster_path[i] if i < len(cluster_path) else -1
                c1 = cluster_path[i + 1] if (i + 1) < len(cluster_path) else -1
                if c0 >= 0 and c1 >= 0:
                    if risk_rank[c1] > risk_rank[c0]:
                        color = "#b3001b"
                    elif risk_rank[c1] < risk_rank[c0]:
                        color = "#2c7a4b"
                    else:
                        color = "#6c757d"
                else:
                    color = "#6c757d"
                ax.plot(
                    [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=color, lw=1.6, alpha=0.85,
                )

    ax.set_xlabel(f"PC1: {pc_labels.get('PC1', '')}", fontsize=10)
    ax.set_ylabel(f"PC2: {pc_labels.get('PC2', '')}", fontsize=10)
    ax.set_zlabel(f"PC3: {pc_labels.get('PC3', '')}", fontsize=10)
    ax.set_title("S&P 600 Risk Clusters — 3D PCA", fontsize=13)
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.view_init(elev=22, azim=-62)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / outfile)
    plt.close(fig)
    logger.info("Wrote %s", outfile)


def plot_cluster_scatter(
    scores: pd.DataFrame,
    cluster_result,
    pc_x: str,
    pc_y: str,
    pc_labels: dict[str, str],
    portfolio_tickers: list[str],
    outfile: str,
) -> None:
    """PC scatter colored by cluster style."""
    fig, ax = plt.subplots(figsize=(11, 8))

    assignments = cluster_result.assignments
    style_labels = cluster_result.style_labels

    handles = []
    for cid in sorted(assignments.unique()):
        mask = assignments == cid
        style = style_labels[int(cid)]
        color = _style_color(cluster_result, style)
        ax.scatter(
            scores.loc[mask, pc_x],
            scores.loc[mask, pc_y],
            s=22,
            alpha=0.55,
            color=color,
            edgecolors="none",
        )
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=color, markersize=9,
                              label=f"C{cid} · {style} (n={int(mask.sum())})"))

    centroids = cluster_result.centroids
    comps = list(scores.columns)
    x_idx = comps.index(pc_x)
    y_idx = comps.index(pc_y)
    ax.scatter(
        centroids[:, x_idx], centroids[:, y_idx],
        marker="X", s=240, c="black", edgecolors="white", linewidths=1.5, zorder=5,
    )

    port_in = [t for t in portfolio_tickers if t in scores.index]
    for t in port_in:
        ax.scatter(
            scores.loc[t, pc_x], scores.loc[t, pc_y],
            s=160, facecolor="none", edgecolors="black", linewidths=2.0, zorder=6,
        )
        ax.annotate(
            t, (scores.loc[t, pc_x], scores.loc[t, pc_y]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=9, fontweight="bold",
        )

    ax.set_xlabel(f"{pc_x}: {pc_labels.get(pc_x, '')}")
    ax.set_ylabel(f"{pc_y}: {pc_labels.get(pc_y, '')}")
    ax.set_title(f"S&P 600 Risk Clusters — {pc_x} vs {pc_y}")
    ax.axhline(0, color="#cccccc", lw=0.7)
    ax.axvline(0, color="#cccccc", lw=0.7)
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / outfile)
    plt.close(fig)
    logger.info("Wrote %s", outfile)


# =============================================================================
# Trajectory map
# =============================================================================
def plot_trajectory_map(
    current_scores: pd.DataFrame,
    trajectory,
    cluster_result,
    pc_labels: dict[str, str],
    portfolio_tickers: list[str],
    outfile: str = "trajectory_map.png",
) -> None:
    """Draw IMA holdings' quarterly paths through PC1xPC2."""
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(
        current_scores["PC1"], current_scores["PC2"],
        s=14, alpha=0.15, color="#888888", edgecolors="none",
    )

    risk_rank = cluster_result.risk_rank

    for tk in portfolio_tickers:
        path = trajectory.pc_paths.get(tk)
        if path is None:
            continue
        valid = path.dropna(subset=["PC1", "PC2"])
        if len(valid) < 2:
            continue

        cluster_path = list(trajectory.cluster_paths.get(tk, []))

        for i in range(len(valid) - 1):
            x0, y0 = valid.iloc[i]["PC1"], valid.iloc[i]["PC2"]
            x1, y1 = valid.iloc[i + 1]["PC1"], valid.iloc[i + 1]["PC2"]

            c0 = cluster_path[i] if i < len(cluster_path) else -1
            c1 = cluster_path[i + 1] if (i + 1) < len(cluster_path) else -1
            if c0 >= 0 and c1 >= 0:
                if risk_rank[c1] > risk_rank[c0]:
                    color = "#b3001b"
                elif risk_rank[c1] < risk_rank[c0]:
                    color = "#2c7a4b"
                else:
                    color = "#6c757d"
            else:
                color = "#6c757d"

            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle="->", mutation_scale=14,
                color=color, lw=1.6, alpha=0.85,
            )
            ax.add_patch(arrow)

        last = valid.iloc[-1]
        ax.scatter(last["PC1"], last["PC2"], s=70, color="black", zorder=5)
        ax.annotate(tk, (last["PC1"], last["PC2"]),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=10, fontweight="bold")

    ax.set_xlabel(f"PC1: {pc_labels.get('PC1', '')}")
    ax.set_ylabel(f"PC2: {pc_labels.get('PC2', '')}")
    ax.set_title("IMA Portfolio Trajectory Through Risk Space")
    ax.axhline(0, color="#cccccc", lw=0.7)
    ax.axvline(0, color="#cccccc", lw=0.7)

    legend = [
        Line2D([0], [0], color="#b3001b", lw=3, label="→ Riskier cluster"),
        Line2D([0], [0], color="#2c7a4b", lw=3, label="→ Safer cluster"),
        Line2D([0], [0], color="#6c757d", lw=3, label="No change"),
    ]
    ax.legend(handles=legend, loc="upper right")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / outfile)
    plt.close(fig)
    logger.info("Wrote %s", outfile)


# =============================================================================
# Portfolio risk dashboard (horizontal bar charts per holding)
# =============================================================================
def plot_portfolio_dashboard(
    percentile_ranks: pd.DataFrame,
    portfolio_report: pd.DataFrame,
    outfile: str = "portfolio_risk_dashboard.png",
) -> None:
    """20-panel grid; each panel shows the holding's feature percentiles."""
    holdings = [r for r in portfolio_report.to_dict(orient="records")
                if r.get("Cluster", -1) != -1 and r["Ticker"] in percentile_ranks.index]
    if not holdings:
        return

    features = config.DASHBOARD_FEATURES
    nice_names = {
        "altman_z": "Altman Z",
        "asset_growth_yoy": "Asset Growth",
        "short_pct_float": "Short %",
        "momentum_90d": "Mom 90d",
        "net_debt_to_ebitda": "Leverage",
        "volatility_60d": "Vol 60d",
    }

    n = len(holdings)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), squeeze=False)

    for idx, rec in enumerate(holdings):
        ax = axes[idx // ncols][idx % ncols]
        tk = rec["Ticker"]
        row = percentile_ranks.loc[tk]
        vals = [float(row[c]) for c in features]

        colors = ["#b3001b" if v > 60 else ("#f0a202" if v > 40 else "#2c7a4b") for v in vals]
        y = list(range(len(features)))
        ax.barh(y, vals, color=colors, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels([nice_names.get(c, c) for c in features], fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Risk pctile", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)

        tier = rec.get("Risk_Tier", "")
        ax.set_facecolor(_risk_tier_color(tier) + "15")
        ax.set_title(f"{tk} · {tier}", fontsize=10, fontweight="bold")
        ax.invert_yaxis()

    for j in range(len(holdings), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("IMA Portfolio — Risk Feature Percentiles", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(config.OUTPUT_DIR / outfile)
    plt.close(fig)
    logger.info("Wrote %s", outfile)


# =============================================================================
# Cluster feature profiles
# =============================================================================
def plot_cluster_profiles(
    cluster_char: pd.DataFrame,
    cluster_result,
    outfile: str = "cluster_profiles.png",
) -> None:
    """Grouped bar chart: feature means per cluster."""
    feature_cols = [c.replace("_mean", "") for c in cluster_char.columns if c.endswith("_mean")]
    means = cluster_char[[f"{c}_mean" for c in feature_cols]].copy()
    means.columns = feature_cols

    norm = (means - means.mean()) / means.std(ddof=0).replace(0, 1)

    fig, ax = plt.subplots(figsize=(14, 7))
    xs = np.arange(len(feature_cols))
    w = 0.15
    for i, cid in enumerate(cluster_char.index):
        style = cluster_result.style_labels.get(int(cid), "?")
        ax.bar(xs + i * w, norm.loc[cid].values, w,
               label=f"C{cid} · {style}",
               color=_style_color(cluster_result, style), edgecolor="white")

    ax.set_xticks(xs + w * (len(cluster_char) - 1) / 2)
    ax.set_xticklabels(feature_cols, rotation=40, ha="right", fontsize=9)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_ylabel("Feature mean (z-scored across clusters)")
    ax.set_title("Cluster Risk Profiles")
    ax.legend(ncol=len(cluster_char), fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / outfile)
    plt.close(fig)
    logger.info("Wrote %s", outfile)


# =============================================================================
# Silhouette analysis
# =============================================================================
def plot_silhouette_analysis(
    diagnostics: pd.DataFrame,
    chosen_k: int,
    outfile: str = "silhouette_analysis.png",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#b3001b" if k == chosen_k else "#6c757d" for k in diagnostics.index]
    ax.bar(diagnostics.index.astype(int).astype(str),
           diagnostics["silhouette"], color=colors)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"Silhouette by k (selected k={chosen_k})")
    for i, k in enumerate(diagnostics.index):
        ax.text(i, diagnostics.loc[k, "silhouette"],
                f"{diagnostics.loc[k, 'silhouette']:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / outfile)
    plt.close(fig)
    logger.info("Wrote %s", outfile)


# =============================================================================
# PCA loadings heatmap
# =============================================================================
def plot_pca_loadings(
    loadings: pd.DataFrame,
    pc_labels: dict[str, str],
    outfile: str = "pca_loadings.png",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    labels = [f"{c}\n{pc_labels.get(c, '')}" for c in loadings.columns]
    sns.heatmap(
        loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        cbar_kws={"label": "Loading"}, ax=ax,
    )
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_title("PCA Loadings — Feature → PC")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / outfile)
    plt.close(fig)
    logger.info("Wrote %s", outfile)


# =============================================================================
# Risk score distribution
# =============================================================================
def plot_risk_score_distribution(
    scores: pd.DataFrame,
    portfolio_tickers: list[str],
    outfile: str = "risk_score_distribution.png",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(scores["composite_score"], bins=40, color="#4a6fa5",
            edgecolor="white", alpha=0.85)

    # Tier boundaries live on the PERCENTILE scale; translate them back to
    # composite-score values for this cross-section so the lines land where
    # the tiers actually cut.
    edges = []
    for lo, hi, label in config.TIER_PERCENTILE_BUCKETS:
        lo_s = float(scores["composite_score"].quantile(lo / 100.0))
        hi_s = float(scores["composite_score"].quantile(min(hi, 100) / 100.0))
        edges.append((lo_s, hi_s, label))
    for lo_s, hi_s, label in edges[:-1]:
        ax.axvline(hi_s, color="#666666", lw=0.6, ls="--")
    for lo_s, hi_s, label in edges:
        mid = (lo_s + hi_s) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.96, label, ha="center",
                fontsize=9, color=_risk_tier_color(label))

    max_y = ax.get_ylim()[1]
    for tk in portfolio_tickers:
        if tk not in scores.index:
            continue
        s = scores.loc[tk, "composite_score"]
        ax.axvline(s, color="black", lw=0.8, alpha=0.5)
        ax.annotate(tk, (s, max_y * 0.8),
                    rotation=90, fontsize=8, ha="center", va="top")

    ax.set_xlabel("Composite risk score (0=safest, 100=riskiest)")
    ax.set_ylabel("# stocks")
    ax.set_title("S&P 600 Risk Score Distribution · IMA Holdings Marked")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / outfile)
    plt.close(fig)
    logger.info("Wrote %s", outfile)


# =============================================================================
# Sector risk comparison
# =============================================================================
def plot_sector_risk_comparison(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    portfolio_tickers: list[str],
    outfile: str = "sector_risk_comparison.png",
) -> None:
    merged = features[["Sector"]].join(scores[["composite_score"]], how="inner")
    if merged.empty:
        return

    order = merged.groupby("Sector")["composite_score"].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.boxplot(
        data=merged, x="Sector", y="composite_score",
        order=order, color="#a8c5e2", ax=ax,
    )

    port = merged.loc[merged.index.intersection(portfolio_tickers)]
    if not port.empty:
        for tk, row in port.iterrows():
            if row["Sector"] not in order:
                continue
            idx = order.index(row["Sector"])
            ax.scatter(idx, row["composite_score"], s=80, color="#b3001b",
                       zorder=5, edgecolor="white")
            ax.annotate(tk, (idx, row["composite_score"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_ylabel("Composite risk score")
    ax.set_xlabel("")
    ax.set_title("Risk Score Distribution by Sector · IMA Holdings Overlaid")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / outfile)
    plt.close(fig)
    logger.info("Wrote %s", outfile)
