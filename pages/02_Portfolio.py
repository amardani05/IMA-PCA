"""IMA portfolio detail — risk tier, drivers, sector comparison, drift status."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit_lib as sl  # noqa: E402

st.set_page_config(page_title="Portfolio · IMA PCA", layout="wide")
sl.apply_styles()
st.title("IMA portfolio")

if not sl.require_pipeline_outputs():
    st.stop()

portfolio = sl.load_portfolio_report()

# =============================================================================
# Headline stats
# =============================================================================
scored = portfolio[portfolio["Cluster"].notna() & (portfolio["Cluster"] != -1)]
total_w = scored["Weight"].sum() if len(scored) else 0.0
weighted_score = (
    (scored["Composite_Score"] * scored["Weight"]).sum() / total_w
    if total_w > 0 else 0.0
)
elevated_count = int((scored["Risk_Tier"] == "Elevated").sum()) if len(scored) else 0
deteriorating = int((scored["Trajectory"] == "Deteriorating").sum()) if "Trajectory" in scored.columns else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    sl.headline_metric("Positions", str(len(portfolio)), f"{len(scored)} scored")
with c2:
    sl.headline_metric(
        "Weighted score", f"{weighted_score:.1f}",
        "0 = safest · 100 = riskiest",
    )
with c3:
    sl.headline_metric("Elevated tier", str(elevated_count), "positions")
with c4:
    sl.headline_metric("Deteriorating", str(deteriorating), "trajectory classifier")


# =============================================================================
# Holdings table
# =============================================================================
st.subheader("Holdings detail")
display_cols = [
    "Ticker", "Weight", "Sector",
    "Composite_Score", "Risk_Tier", "Cluster_Label",
    "Altman_Z", "Net_Debt_EBITDA", "Short_Pct_Float", "Momentum_90d",
    "Volatility_60d", "Top_Risk_Drivers", "Sector_Comparison", "Trajectory",
]
present_cols = [c for c in display_cols if c in portfolio.columns]
df = portfolio[present_cols].copy()

if "Weight" in df.columns:
    df["Weight"] = (df["Weight"] * 100).round(2)

st.dataframe(
    df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn(width="small"),
        "Weight": st.column_config.NumberColumn("Weight %", format="%.2f%%", width="small"),
        "Composite_Score": st.column_config.NumberColumn("Score", format="%.1f", width="small"),
        "Risk_Tier": st.column_config.TextColumn("Tier", width="small"),
        "Cluster_Label": st.column_config.TextColumn("Cluster", width="small"),
        "Altman_Z": st.column_config.NumberColumn("Altman Z", format="%.2f"),
        "Net_Debt_EBITDA": st.column_config.NumberColumn("ND/EBITDA", format="%.2f"),
        "Short_Pct_Float": st.column_config.NumberColumn("Short %", format="%.1f"),
        "Momentum_90d": st.column_config.NumberColumn("Mom 90d", format="%+.1f%%"),
        "Volatility_60d": st.column_config.NumberColumn("Vol 60d", format="%.1f%%"),
        "Top_Risk_Drivers": st.column_config.TextColumn("Top drivers", width="large"),
        "Sector_Comparison": st.column_config.TextColumn("vs sector", width="medium"),
        "Trajectory": st.column_config.TextColumn(width="small"),
    },
)


# =============================================================================
# Per-stock factor betas (from the most recent macro run, MAX timeframe)
# =============================================================================
sb = sl.load_stock_betas()
factor_meta = sl.load_factor_metadata()
if sb and factor_meta:
    st.subheader("Per-stock macro factor betas")
    st.caption(
        "From the MAX-timeframe v2 (multi-factor residualized) regression. "
        "Use the Macro Exposures page to view a different window."
    )
    by_key = {f["factor"]: f for f in factor_meta["factors"]}
    rows = []
    weight_by_t = dict(zip(portfolio["Ticker"], portfolio["Weight"]))
    for tk in sb["tickers"]:
        row = {
            "Ticker": tk,
            "Weight %": (weight_by_t.get(tk, 0.0) or 0.0) * 100,
        }
        for f in sb["factors"]:
            row[by_key.get(f, {}).get("name", f)] = sb["betas"][tk][f]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Weight %", ascending=False)
    st.dataframe(
        df, hide_index=True, use_container_width=True,
        column_config={
            "Weight %": st.column_config.NumberColumn(format="%.2f%%"),
            **{
                col: st.column_config.NumberColumn(format="%+.3f")
                for col in df.columns if col not in ("Ticker", "Weight %")
            },
        },
    )
