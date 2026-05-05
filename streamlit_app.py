"""IMA Principal Component Analysis — Streamlit launch page.

Layout:
  ┌─────────────────────┬───────────────────────────────────────────────┐
  │  SIDEBAR (vertical) │  MAIN                                          │
  │  · Stock selector   │  · 2x2 PC scatter grid (charts up top)         │
  │    — sector tiles   │  · 3D PCA                                       │
  │    as expanders,    │  · PCA decomposition (variance + loadings)      │
  │    portfolio always │                                                 │
  │    visible          │                                                 │
  │  · Filter toggles   │                                                 │
  └─────────────────────┴───────────────────────────────────────────────┘

Run via::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import streamlit_lib as sl

st.set_page_config(
    page_title="IMA PCA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
sl.apply_styles()

if not sl.require_pipeline_outputs():
    st.stop()

meta = sl.load_meta()
universe = sl.load_universe()
portfolio = sl.load_portfolio_report()
trajectory = sl.load_trajectory()
pca_summary = sl.load_pca_summary()


# =============================================================================
# Selection state
# =============================================================================
PORTFOLIO_TICKERS: set[str] = set(portfolio["Ticker"].tolist())

for _tk in universe["Ticker"]:
    _key = f"sel_{_tk}"
    if _key not in st.session_state:
        st.session_state[_key] = _tk in PORTFOLIO_TICKERS


def selected_tickers() -> set[str]:
    return {tk for tk in universe["Ticker"]
            if st.session_state.get(f"sel_{tk}", False)}


def reset_to_portfolio() -> None:
    for tk in universe["Ticker"]:
        st.session_state[f"sel_{tk}"] = tk in PORTFOLIO_TICKERS


def toggle_sector(holdings_keys: list[str]) -> None:
    new_val = not all(st.session_state.get(k, False) for k in holdings_keys)
    for k in holdings_keys:
        st.session_state[k] = new_val


# =============================================================================
# SIDEBAR — stock selector + filter toggles
# =============================================================================
with st.sidebar:
    st.markdown("### Stock selector")

    sb_c1, sb_c2, sb_c3 = st.columns(3)
    with sb_c1:
        if st.button("All", key="ctl_all", use_container_width=True):
            for tk in PORTFOLIO_TICKERS:
                st.session_state[f"sel_{tk}"] = True
    with sb_c2:
        if st.button("None", key="ctl_none", use_container_width=True):
            for tk in universe["Ticker"]:
                st.session_state[f"sel_{tk}"] = False
    with sb_c3:
        if st.button("Reset", key="ctl_reset", use_container_width=True,
                     help="Restore the IMA portfolio default selection"):
            reset_to_portfolio()

    n_sel = sum(1 for tk in universe["Ticker"]
                if st.session_state.get(f"sel_{tk}", False))
    st.caption(f"**{n_sel}** selected")

    # Group portfolio by sector (largest weight first)
    portfolio_by_sector: dict[str, pd.DataFrame] = {
        sec: grp.sort_values("Weight", ascending=False)
        for sec, grp in portfolio.groupby(portfolio["Sector"].fillna("Unknown"))
    }
    sector_order = sorted(
        portfolio_by_sector,
        key=lambda s: portfolio_by_sector[s]["Weight"].sum(),
        reverse=True,
    )

    for sector in sector_order:
        holdings = portfolio_by_sector[sector]
        holding_tickers = holdings["Ticker"].tolist()
        holding_keys = [f"sel_{t}" for t in holding_tickers]
        n_in = sum(1 for k in holding_keys if st.session_state.get(k, False))
        total_w = holdings["Weight"].sum() * 100

        with st.expander(
            f"{sector} · {n_in}/{len(holding_keys)} · {total_w:.1f}%",
            expanded=False,
        ):
            st.button(
                "↻ Toggle sector",
                key=f"sec_toggle_{sector}",
                on_click=toggle_sector,
                args=(holding_keys,),
                use_container_width=True,
            )
            for _, row in holdings.iterrows():
                tier = row.get("Risk_Tier", "?") or "?"
                weight_pct = (row.get("Weight") or 0) * 100
                st.checkbox(
                    f"{row['Ticker']} · {tier} · {weight_pct:.1f}%",
                    key=f"sel_{row['Ticker']}",
                )

            # Universe peers in this sector (no nested expanders — Streamlit
            # forbids them — so peers render inline below the holdings).
            peers = universe[
                (universe["Sector"] == sector)
                & (~universe["Ticker"].isin(holding_tickers))
            ].sort_values("composite_score").head(15)
            if len(peers):
                st.markdown(
                    f"<small style='color:#5a6370'>Peers in sector ({len(peers)})"
                    f"</small>",
                    unsafe_allow_html=True,
                )
                for _, p in peers.iterrows():
                    st.checkbox(
                        f"{p['Ticker']} · {p['cluster_tier']} · "
                        f"score {p['composite_score']:.0f}",
                        key=f"sel_{p['Ticker']}",
                    )

    st.divider()
    st.markdown("**Filters**")
    portfolio_only = st.toggle(
        "Portfolio only", value=False,
        help="Hide non-portfolio dots entirely.",
    )
    show_traj = st.toggle(
        "Show trajectories", value=True,
        help="Quarterly paths for portfolio holdings.",
    )
    tier_filter = st.multiselect(
        "Tiers", options=meta["tier_order"], default=[],
        help="Empty = all tiers",
    )
    sector_filter = st.multiselect(
        "Sectors", options=sorted(universe["Sector"].dropna().unique()),
        default=[], help="Empty = all sectors",
    )


# Snapshot once the sidebar is rendered, then use everywhere
SELECTED: set[str] = selected_tickers()


# =============================================================================
# MAIN — title + charts immediately
# =============================================================================
st.title("IMA Principal Component Analysis")


def filtered_others(df: pd.DataFrame) -> pd.DataFrame:
    out = df[~df["Ticker"].isin(SELECTED)].copy()
    if portfolio_only:
        out = out[out["is_portfolio"] == True]  # noqa: E712
    if tier_filter:
        out = out[out["cluster_tier"].isin(tier_filter)]
    if sector_filter:
        out = out[out["Sector"].isin(sector_filter)]
    return out


def selected_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Ticker"].isin(SELECTED)].copy()


def build_scatter(pc_x: str, pc_y: str, height: int = 460) -> go.Figure:
    fig = go.Figure()
    others = filtered_others(universe)
    sel = selected_rows(universe)

    for tier in meta["tier_order"]:
        bucket = others[others["cluster_tier"] == tier]
        if bucket.empty:
            continue
        fig.add_trace(go.Scatter(
            x=bucket[pc_x], y=bucket[pc_y],
            mode="markers",
            name=f"{tier} (n={len(bucket)})",
            marker=dict(color=sl.TIER_COLORS.get(tier, "#666"),
                        size=6, opacity=0.55, line=dict(width=0)),
            text=bucket.apply(
                lambda r: f"<b>{r['Ticker']}</b><br>{r.get('Company', '')}<br>"
                          f"{r['Sector']} · {r['cluster_tier']}<br>"
                          f"score {r['composite_score']:.1f}",
                axis=1,
            ),
            hovertemplate="%{text}<extra></extra>",
        ))

    if show_traj and trajectory and trajectory.get("paths"):
        for tk in SELECTED:
            path = trajectory["paths"].get(tk)
            if not path:
                continue
            coords = [c for c in path["coords"]
                      if c.get(pc_x) is not None and c.get(pc_y) is not None]
            if len(coords) < 2:
                continue
            fig.add_trace(go.Scatter(
                x=[c[pc_x] for c in coords],
                y=[c[pc_y] for c in coords],
                mode="lines+markers",
                line=dict(color="#6c757d", width=1.5),
                marker=dict(size=5, color="white",
                            line=dict(color="black", width=1)),
                showlegend=False,
                hoverinfo="skip",
            ))

    if not sel.empty:
        fig.add_trace(go.Scatter(
            x=sel[pc_x], y=sel[pc_y],
            mode="markers+text",
            name=f"Selected ({len(sel)})",
            marker=dict(
                size=16,
                color=[sl.TIER_COLORS.get(t, "#666") for t in sel["cluster_tier"]],
                line=dict(color="#0a0a0a", width=2.5),
            ),
            text=sel["Ticker"],
            textposition="top center",
            textfont=dict(size=11, color="#0a0a0a"),
            hovertext=sel.apply(
                lambda r: f"<b>{r['Ticker']}</b> · {r.get('Company', '')}<br>"
                          f"{r['Sector']} · {r['cluster_tier']}<br>"
                          f"score {r['composite_score']:.1f}",
                axis=1,
            ),
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=55, r=20, t=10, b=50),
        xaxis_title=f"{pc_x}: {meta['pca']['pc_labels'].get(pc_x, '')}",
        yaxis_title=f"{pc_y}: {meta['pca']['pc_labels'].get(pc_y, '')}",
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=10)),
        hovermode="closest",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    fig.add_hline(y=0, line_color="#cccccc", line_width=0.7)
    fig.add_vline(x=0, line_color="#cccccc", line_width=0.7)
    return fig


def build_3d(height: int = 520) -> go.Figure:
    fig = go.Figure()
    others = filtered_others(universe)
    sel = selected_rows(universe)

    for tier in meta["tier_order"]:
        bucket = others[others["cluster_tier"] == tier]
        if bucket.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=bucket["PC1"], y=bucket["PC2"], z=bucket["PC3"],
            mode="markers",
            name=f"{tier} (n={len(bucket)})",
            marker=dict(color=sl.TIER_COLORS.get(tier, "#666"),
                        size=3, opacity=0.5),
            text=bucket["Ticker"],
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))

    if not sel.empty:
        fig.add_trace(go.Scatter3d(
            x=sel["PC1"], y=sel["PC2"], z=sel["PC3"],
            mode="markers+text",
            name=f"Selected ({len(sel)})",
            marker=dict(
                size=9,
                color=[sl.TIER_COLORS.get(t, "#666") for t in sel["cluster_tier"]],
                line=dict(color="#0a0a0a", width=3),
            ),
            text=sel["Ticker"],
            textposition="top center",
            textfont=dict(size=10),
        ))

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis_title=f"PC1 · {meta['pca']['pc_labels'].get('PC1', '')}",
            yaxis_title=f"PC2 · {meta['pca']['pc_labels'].get('PC2', '')}",
            zaxis_title=f"PC3 · {meta['pca']['pc_labels'].get('PC3', '')}",
            aspectmode="cube",
            bgcolor="#ffffff",
        ),
        paper_bgcolor="#ffffff",
    )
    return fig


# Charts FIRST — immediately visible after the title
g1, g2 = st.columns(2)
with g1:
    st.markdown("**PC1 vs PC2** — dominant risk-structure view")
    st.plotly_chart(build_scatter("PC1", "PC2"), use_container_width=True)
with g2:
    st.markdown("**PC1 vs PC3** — adds the next dimension")
    st.plotly_chart(build_scatter("PC1", "PC3"), use_container_width=True)

g3, g4 = st.columns(2)
with g3:
    st.markdown("**PC2 vs PC3** — cross-section orthogonal to PC1")
    st.plotly_chart(build_scatter("PC2", "PC3"), use_container_width=True)
with g4:
    st.markdown("**3D PCA** — rotate to inspect cluster geometry")
    st.plotly_chart(build_3d(), use_container_width=True)


# =============================================================================
# PCA decomposition
# =============================================================================
st.markdown("### PCA decomposition")
st.caption(
    "Features are z-scored before PCA. Each PC is auto-labeled by its "
    "dominant-loading feature family — cross-check with the loadings table "
    "before citing the label."
)

dec1, dec2 = st.columns(2)

with dec1:
    st.markdown("**Variance explained**")
    if len(pca_summary):
        df = pca_summary[["pc", "variance_explained", "cumulative_variance", "label"]].copy()
        df["Variance %"] = (df["variance_explained"] * 100).round(2)
        df["Cumulative %"] = (df["cumulative_variance"] * 100).round(2)
        st.dataframe(
            df[["pc", "Variance %", "Cumulative %", "label"]].rename(
                columns={"pc": "PC", "label": "Auto-label"},
            ),
            hide_index=True,
            use_container_width=True,
        )

with dec2:
    st.markdown("**Dominant features per PC**")
    if len(pca_summary):
        for _, row in pca_summary.iterrows():
            pc = row["pc"]
            label = row["label"]
            top_loadings = row.get("top_loadings", [])
            st.markdown(
                f"**{pc}** — *{label}*  "
                + " · ".join(
                    f"`{t['feature']}` "
                    f"<span style='color:{'#2c7a4b' if t['loading'] >= 0 else '#b3001b'}"
                    f";font-weight:600'>{'+' if t['loading'] >= 0 else ''}{t['loading']:.2f}</span>"
                    for t in top_loadings
                ),
                unsafe_allow_html=True,
            )

heatmap_path = sl.CHARTS_DIR / "pca_loadings.png"
if heatmap_path.exists():
    st.markdown("**Loadings heatmap**")
    st.image(str(heatmap_path), use_container_width=True)

st.divider()
st.caption(
    f"Generated {meta['generated_at']} · use the sidebar pages to explore "
    f"macro factor exposures, the portfolio detail, and pitch assessments."
)
