"""Macro factor exposures — timeframe selector + raw/v1/v2 methodology toggle."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make sibling helpers importable when streamlit's CWD is the pages/ dir.
sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit_lib as sl  # noqa: E402

st.set_page_config(page_title="Macro Exposures · IMA PCA", layout="wide")
sl.apply_styles()
st.title("Macro factor exposures")

if not sl.require_pipeline_outputs():
    st.stop()

timeframes = sl.load_macro_timeframes()
metadata = sl.load_factor_metadata()

if not timeframes or not metadata:
    st.error(
        "Macro outputs missing. Run `python main.py` (or pass `--no-macro` to skip "
        "macro analysis explicitly)."
    )
    st.stop()


# =============================================================================
# Timeframe + methodology selectors
# =============================================================================
TF_LABELS = {"ytd": "YTD", "6m": "6M", "1y": "1Y", "2y": "2Y", "max": "MAX"}
available = [tf for tf in timeframes["timeframes"] if tf in timeframes["by_timeframe"]]
default_tf = timeframes.get("default", "max")

c1, c2 = st.columns([3, 2])
with c1:
    tf_code = st.radio(
        "Timeframe",
        options=available,
        index=available.index(default_tf) if default_tf in available else 0,
        format_func=lambda c: TF_LABELS.get(c, c.upper()),
        horizontal=True,
        key="macro_timeframe",
    )
with c2:
    methodology = st.radio(
        "Methodology",
        options=["v2", "v1", "raw"],
        format_func=lambda m: {
            "v2": "Residualized v2 (default)",
            "v1": "v1 (market only)",
            "raw": "Raw OLS",
        }[m],
        horizontal=True,
        key="macro_methodology",
        help=(
            "v2 strips IJR + VIX + HY OAS exposure (recommended). v1 strips "
            "market beta only. Raw OLS doesn't control for shared common "
            "factors and inflates betas."
        ),
    )

active = timeframes["by_timeframe"][tf_code]
result = active[methodology]
factors = active[methodology]["factors"]
betas = active[methodology]["betas"]
factor_meta_by_key = {f["factor"]: f for f in metadata["factors"]}

# Headline stats
n_obs = active["n_obs"]
date_range = active["date_range"]
r_sq = result["r_squared"]
alpha_pct = result["alpha"] * 252 * 100
max_vif = max(result["vifs"].values()) if result.get("vifs") else None
mb = result.get("market_beta")

st.markdown(
    f"<small style='color:#5a6370'>"
    f"<b>{TF_LABELS[tf_code]}</b> · {date_range[0]} → {date_range[1]} · "
    f"n = {n_obs} obs · R² = {r_sq:.3f} · "
    f"α annualized {'+' if alpha_pct >= 0 else ''}{alpha_pct:.2f}% · "
    + (f"market β = {mb:.2f} · " if mb is not None else "")
    + f"max VIF {max_vif:.2f}" + (" ⚠" if max_vif and max_vif > 5 else "") +
    "</small>",
    unsafe_allow_html=True,
)

if methodology == "raw":
    st.warning(
        "Raw OLS doesn't control for the dominant common drivers (market "
        "beta, vol regime, credit cycle). Betas reported here include shared "
        "risk-sentiment exposure — use v2 for the honest answer.",
        icon="⚠️",
    )


# =============================================================================
# Rolling betas chart (sliced to the selected timeframe)
# =============================================================================
rolling = sl.load_rolling_betas()
if rolling and rolling.get("dates"):
    start, end = date_range
    keep_idx = [i for i, d in enumerate(rolling["dates"]) if start <= d <= end]
    dates = [rolling["dates"][i] for i in keep_idx]

    factor_meta_by_key = {f["factor"]: f for f in metadata["factors"]}
    # Default: top 4 by absolute current beta
    ranked = sorted(
        factors,
        key=lambda f: abs(betas[f]["beta"]),
        reverse=True,
    )

    enabled = st.multiselect(
        "Rolling 60-day betas — pick factors to plot",
        options=factors,
        default=ranked[:4],
        format_func=lambda f: factor_meta_by_key.get(f, {}).get("name", f),
        key="rolling_factors",
    )

    if enabled and dates:
        palette = ["#1f3b73", "#b3001b", "#2c7a4b", "#d4a017",
                   "#7e57c2", "#0288d1", "#e57a44"]
        fig = go.Figure()
        for i, f in enumerate(enabled):
            series = rolling["series"].get(f, [])
            y = [series[idx] for idx in keep_idx]
            fig.add_trace(go.Scatter(
                x=dates, y=y, mode="lines",
                name=factor_meta_by_key.get(f, {}).get("name", f),
                line=dict(color=palette[i % len(palette)], width=2),
            ))
        fig.add_hline(y=0, line_color="#999", line_width=0.8)
        fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=50, r=20, t=10, b=40),
            yaxis_title="Beta",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Beta panel + scenarios
# =============================================================================
left, right = st.columns([2, 1])

with left:
    st.subheader("Macro factor betas")
    beta_rows = []
    for f in factors:
        meta = factor_meta_by_key.get(f, {"name": f, "category": "—"})
        b = betas[f]
        stars = ("★★★" if b["p_value"] < 0.01 else
                 "★★"  if b["p_value"] < 0.05 else
                 "★"   if b["p_value"] < 0.10 else "")
        beta_rows.append({
            "Category": meta.get("category", "").replace("_", " ").title(),
            "Factor": meta.get("name", f),
            "β": b["beta"],
            "t": b["t_stat"],
            "p": b["p_value"],
            "Sig.": stars,
        })
    df_betas = pd.DataFrame(beta_rows).sort_values("β", key=lambda s: s.abs(), ascending=False)
    st.dataframe(
        df_betas,
        hide_index=True,
        use_container_width=True,
        column_config={
            "β": st.column_config.NumberColumn("β", format="%+.3f"),
            "t": st.column_config.NumberColumn("t", format="%+.2f"),
            "p": st.column_config.NumberColumn("p", format="%.3f"),
        },
    )

    # Three-way comparison expander
    with st.expander("Raw → v1 → v2 progression", expanded=False):
        comp_rows = active["comparison"]
        comp_df = pd.DataFrame([
            {
                "Factor": factor_meta_by_key.get(r["factor"], {}).get("name", r["factor"]),
                "Raw β": r.get("raw_beta"),
                "v1 β": r.get("v1_beta"),
                "v2 β": r.get("v2_beta"),
                "Interpretation": r.get("interpretation", ""),
            }
            for r in comp_rows
        ])
        st.dataframe(
            comp_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Raw β": st.column_config.NumberColumn(format="%+.3f"),
                "v1 β": st.column_config.NumberColumn(format="%+.3f"),
                "v2 β": st.column_config.NumberColumn(format="%+.3f"),
            },
        )

with right:
    st.subheader("Scenario sensitivity")
    st.caption("β × shock = portfolio impact, all else equal.")
    scenarios = active["scenarios"]
    sc_rows = [
        {
            "Factor": factor_meta_by_key.get(s["factor"], {}).get("name", s["factor"]),
            "Shock": s["label"],
            "Impact": s["impact"] * 100,
        }
        for s in scenarios
    ]
    df_sc = pd.DataFrame(sc_rows)
    st.dataframe(
        df_sc,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Impact": st.column_config.NumberColumn(
                "Impact %", format="%+.2f%%",
            ),
        },
    )


# =============================================================================
# Per-stock × per-factor beta table
# =============================================================================
st.subheader(f"Portfolio × macro factor betas ({TF_LABELS[tf_code]})")
sb = active["stock_betas"]
if sb and sb.get("tickers"):
    # Build a wide DataFrame
    rows = []
    portfolio = sl.load_portfolio_report()
    weight_by_ticker = {r["Ticker"]: r.get("Weight", 0.0) for r in portfolio.to_dict("records")}
    for tk in sb["tickers"]:
        row = {"Ticker": tk, "Weight %": (weight_by_ticker.get(tk, 0.0) or 0.0) * 100}
        for f in sb["factors"]:
            row[factor_meta_by_key.get(f, {}).get("name", f)] = sb["betas"][tk][f]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Weight %", ascending=False)
    st.dataframe(
        df, hide_index=True, use_container_width=True,
        column_config={
            "Weight %": st.column_config.NumberColumn("Weight %", format="%.2f%%"),
            **{
                col: st.column_config.NumberColumn(format="%+.3f")
                for col in df.columns if col not in ("Ticker", "Weight %")
            },
        },
    )
