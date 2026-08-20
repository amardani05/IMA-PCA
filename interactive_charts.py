"""Interactive Plotly charts — 2D pairwise PCs and 3D cluster scatter.

Each chart is written as a self-contained HTML file to ``output/interactive/``
(so it can be opened in any browser) and also as a JSON figure spec that the
React webapp consumes via ``react-plotly.js``. Trajectories for IMA holdings
are overlaid on every scatter.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import config

logger = logging.getLogger(__name__)

# Cluster styles are colored from cluster_result.style_colors (rank-ordered
# palette defined in config.CLUSTER_STYLE_PALETTE) — aligned with visualization.py.

INTERACTIVE_DIR = config.OUTPUT_DIR / "interactive"


def _ensure_dir() -> Path:
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    return INTERACTIVE_DIR


def _style_color(cluster_result, style: str) -> str:
    return getattr(cluster_result, "style_colors", {}).get(style, "#666")


def _write(fig: go.Figure, stem: str) -> None:
    out = _ensure_dir()
    html_path = out / f"{stem}.html"
    json_path = out / f"{stem}.json"
    # Full HTML with the plotly.js library inlined so the file works standalone.
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    # Figure-spec JSON for the React app.
    json_path.write_text(pio.to_json(fig))
    logger.info("Wrote %s + %s", html_path.name, json_path.name)


# =============================================================================
# Hover text builder
# =============================================================================
def _hover_text(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    percentile_ranks: pd.DataFrame | None,
    tickers: pd.Index,
    tier_map: dict[str, str],
) -> list[str]:
    rows = []
    for tk in tickers:
        company = features.loc[tk, "Company"] if "Company" in features.columns else tk
        sector = features.loc[tk, "Sector"] if "Sector" in features.columns else ""
        tier = tier_map.get(tk, "?")
        pc_strs = "  ".join(
            f"{pc}={scores.loc[tk, pc]:+.2f}"
            for pc in ("PC1", "PC2", "PC3")
            if pc in scores.columns
        )
        score_str = ""
        if percentile_ranks is not None and tk in percentile_ranks.index:
            score_str = (
                f"<br>Composite score: "
                f"<b>{percentile_ranks.loc[tk, 'composite_score']:.1f}</b>  "
                f"({percentile_ranks.loc[tk, 'risk_tier']})"
            )
        rows.append(
            f"<b>{tk}</b> · {company}<br>"
            f"{sector} · <b>{tier}</b>"
            f"{score_str}<br>"
            f"{pc_strs}"
        )
    return rows


# =============================================================================
# 2D pairwise PC scatter
# =============================================================================
def plot_2d_pc_scatter(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    cluster_result,
    percentile_ranks: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    pc_labels: dict[str, str],
    portfolio_tickers: list[str],
    trajectory=None,
    stem: str | None = None,
) -> None:
    """Interactive PC scatter with hover tooltips + optional trajectories."""
    stem = stem or f"scatter_{pc_x.lower()}_{pc_y.lower()}"
    assignments = cluster_result.assignments
    style_labels = cluster_result.style_labels
    tier_by_ticker = {
        tk: style_labels[int(cid)] for tk, cid in assignments.items()
    }

    fig = go.Figure()

    for cid in sorted(assignments.unique()):
        mask = assignments == cid
        style = style_labels[int(cid)]
        tickers = scores.loc[mask].index
        fig.add_trace(go.Scatter(
            x=scores.loc[mask, pc_x],
            y=scores.loc[mask, pc_y],
            mode="markers",
            name=f"C{cid} · {style} (n={int(mask.sum())})",
            marker=dict(color=_style_color(cluster_result, style), size=7,
                        opacity=0.65, line=dict(width=0)),
            text=_hover_text(features, scores, percentile_ranks, tickers, tier_by_ticker),
            hovertemplate="%{text}<extra></extra>",
        ))

    # Centroids
    centroids = cluster_result.centroids
    comps = list(scores.columns)
    x_idx, y_idx = comps.index(pc_x), comps.index(pc_y)
    fig.add_trace(go.Scatter(
        x=centroids[:, x_idx], y=centroids[:, y_idx],
        mode="markers",
        name="Cluster centroids",
        marker=dict(symbol="x", size=16, color="black",
                    line=dict(color="white", width=1)),
        hoverinfo="skip",
    ))

    # Portfolio overlay
    port_in = [t for t in portfolio_tickers if t in scores.index]
    if port_in:
        fig.add_trace(go.Scatter(
            x=scores.loc[port_in, pc_x],
            y=scores.loc[port_in, pc_y],
            mode="markers+text",
            name="IMA Portfolio",
            marker=dict(size=13, color="rgba(0,0,0,0)",
                        line=dict(color="black", width=2)),
            text=port_in,
            textposition="top center",
            textfont=dict(size=10, color="black"),
            hovertext=_hover_text(features, scores, percentile_ranks,
                                  pd.Index(port_in), tier_by_ticker),
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    # Trajectories
    if trajectory is not None:
        risk_rank = cluster_result.risk_rank
        for tk in portfolio_tickers:
            path = trajectory.pc_paths.get(tk)
            if path is None:
                continue
            valid = path.dropna(subset=[pc_x, pc_y])
            if len(valid) < 2:
                continue
            cluster_path = list(trajectory.cluster_paths.get(tk, []))
            for i in range(len(valid) - 1):
                x0, y0 = valid.iloc[i][[pc_x, pc_y]]
                x1, y1 = valid.iloc[i + 1][[pc_x, pc_y]]
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
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=0.85,
                ))

    fig.update_layout(
        title=f"S&P 600 Risk Clusters — {pc_x} vs {pc_y}",
        xaxis_title=f"{pc_x}: {pc_labels.get(pc_x, '')}",
        yaxis_title=f"{pc_y}: {pc_labels.get(pc_y, '')}",
        hovermode="closest",
        template="plotly_white",
        height=680,
        margin=dict(l=60, r=30, t=70, b=60),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    fig.add_hline(y=0, line_width=0.6, line_color="#bbbbbb")
    fig.add_vline(x=0, line_width=0.6, line_color="#bbbbbb")
    _write(fig, stem)


def plot_all_2d_pc_scatters(
    features, scores, cluster_result, percentile_ranks,
    pc_labels, portfolio_tickers, trajectory=None,
) -> None:
    pairs = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]
    for x, y in pairs:
        if x in scores.columns and y in scores.columns:
            plot_2d_pc_scatter(
                features, scores, cluster_result, percentile_ranks,
                x, y, pc_labels, portfolio_tickers, trajectory,
                stem=f"scatter_{x.lower()}_{y.lower()}",
            )


# =============================================================================
# 3D PCA scatter + trajectories
# =============================================================================
def plot_3d_pc_scatter(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    cluster_result,
    percentile_ranks: pd.DataFrame,
    pc_labels: dict[str, str],
    portfolio_tickers: list[str],
    trajectory=None,
    stem: str = "scatter_3d",
) -> None:
    """Rotatable 3D PC1/PC2/PC3 scatter colored by cluster, with trajectories."""
    if not {"PC1", "PC2", "PC3"}.issubset(scores.columns):
        logger.warning("3D scatter requires PC1/2/3; skipping")
        return

    assignments = cluster_result.assignments
    style_labels = cluster_result.style_labels
    tier_by_ticker = {tk: style_labels[int(cid)] for tk, cid in assignments.items()}

    fig = go.Figure()

    for cid in sorted(assignments.unique()):
        mask = assignments == cid
        style = style_labels[int(cid)]
        tickers = scores.loc[mask].index
        fig.add_trace(go.Scatter3d(
            x=scores.loc[mask, "PC1"],
            y=scores.loc[mask, "PC2"],
            z=scores.loc[mask, "PC3"],
            mode="markers",
            name=f"C{cid} · {style} (n={int(mask.sum())})",
            marker=dict(color=_style_color(cluster_result, style), size=4, opacity=0.6,
                        line=dict(width=0)),
            text=_hover_text(features, scores, percentile_ranks, tickers, tier_by_ticker),
            hovertemplate="%{text}<extra></extra>",
        ))

    # Centroids
    centroids = cluster_result.centroids
    comps = list(scores.columns)
    ix = [comps.index(c) for c in ("PC1", "PC2", "PC3")]
    fig.add_trace(go.Scatter3d(
        x=centroids[:, ix[0]],
        y=centroids[:, ix[1]],
        z=centroids[:, ix[2]],
        mode="markers",
        name="Centroids",
        marker=dict(symbol="x", size=7, color="black",
                    line=dict(color="white", width=1)),
        hoverinfo="skip",
    ))

    # Portfolio rings
    port_in = [t for t in portfolio_tickers if t in scores.index]
    if port_in:
        fig.add_trace(go.Scatter3d(
            x=scores.loc[port_in, "PC1"],
            y=scores.loc[port_in, "PC2"],
            z=scores.loc[port_in, "PC3"],
            mode="markers+text",
            name="IMA Portfolio",
            marker=dict(size=8, color="rgba(0,0,0,0)",
                        line=dict(color="black", width=3)),
            text=port_in,
            textposition="top center",
            textfont=dict(size=10),
            hovertext=_hover_text(features, scores, percentile_ranks,
                                  pd.Index(port_in), tier_by_ticker),
            hovertemplate="%{hovertext}<extra></extra>",
        ))

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
                fig.add_trace(go.Scatter3d(
                    x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                    mode="lines",
                    line=dict(color=color, width=4),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    fig.update_layout(
        title="S&P 600 Risk Clusters — 3D PCA",
        scene=dict(
            xaxis_title=f"PC1 · {pc_labels.get('PC1', '')}",
            yaxis_title=f"PC2 · {pc_labels.get('PC2', '')}",
            zaxis_title=f"PC3 · {pc_labels.get('PC3', '')}",
            aspectmode="cube",
        ),
        template="plotly_white",
        height=760,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    _write(fig, stem)
