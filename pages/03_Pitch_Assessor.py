"""Pitch assessor — view a candidate ticker's structured one-pager.

Pitches are pre-generated via ``python main.py --assess TICKER`` (or
``--assess-batch FILE``). The Streamlit app reads from
``webapp/public/data/pitches/`` so no Python regression runs in-browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit_lib as sl  # noqa: E402

st.set_page_config(page_title="Pitch Assessor · IMA PCA", layout="wide")
sl.apply_styles()
st.title("Pitch assessor")

if not sl.require_pipeline_outputs():
    st.stop()

REC_COLORS = {
    "PROCEED":             "#2c7a4b",
    "PROCEED WITH CAVEATS": "#d4a017",
    "QUESTION THESIS":      "#e57a44",
    "AVOID":                "#b3001b",
}


# =============================================================================
# Search bar + recently-generated chips
# =============================================================================
index = sl.load_pitch_index()
ticker_options = sorted({row["ticker"] for row in index})

c1, c2 = st.columns([3, 2])
with c1:
    search = st.text_input(
        "Ticker", placeholder="e.g. CRGY, MYRG, KRYS",
        key="pitch_search",
    )
with c2:
    if ticker_options:
        picked = st.selectbox(
            "or pick from generated assessments", [""] + ticker_options,
            key="pitch_pick",
        )
        if picked:
            search = picked

st.caption(
    "To generate a new pitch, run from the repo root: "
    "`python main.py --assess TICKER`"
)


if not search:
    st.info("Enter a ticker above to view its pitch assessment.")
    if index:
        st.markdown("**Recently generated:**")
        chips = " ".join(
            f"`{r['ticker']}` ({r['recommendation']})"
            for r in sorted(index, key=lambda r: r.get("generated_at", ""), reverse=True)[:10]
        )
        st.markdown(chips)
    st.stop()


# =============================================================================
# Display the pitch
# =============================================================================
ticker = search.strip().upper()
pitch = sl.load_pitch(ticker)
if not pitch:
    st.error(
        f"No pitch assessment found for **{ticker}**. Generate it first:\n\n"
        f"```\npython main.py --assess {ticker}\n```\n\n"
        "If that ticker isn't in the S&P 600 universe (which is what the "
        "screener scopes to), the assess command will fail with a diagnostic "
        "error pointing at the cause."
    )
    st.stop()

rec = pitch.get("recommendation", "")
rec_color = REC_COLORS.get(rec, "#5a6370")
cap = pitch.get("market_cap", 0.0)
cap_str = f"${cap / 1e9:.2f}B" if cap and cap > 0 else "n/a"

# Header card
st.markdown(
    f"""
    <div style="background:#fff;border:1px solid #e4e7eb;border-radius:8px;
                border-left:5px solid {rec_color};padding:16px 18px;margin:8px 0">
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px">
        <div>
          <div style="font-size:24px;font-weight:700">{pitch['ticker']}</div>
          <div style="font-size:14px;color:#5a6370">{pitch['company_name']}</div>
          <div style="font-size:12px;color:#5a6370;margin-top:4px">
            {pitch['sector']} / {pitch.get('industry') or '—'} · Market cap {cap_str}
          </div>
        </div>
        <div style="text-align:right">
          <span style="padding:8px 16px;border-radius:6px;font-weight:700;
                       background:{rec_color};color:#fff;font-size:14px">{rec}</span>
          <div style="margin-top:8px;max-width:460px;font-size:12px;color:#5a6370">
            {pitch.get('recommendation_rationale', '')}
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Key findings
st.subheader("Key findings")
for b in pitch.get("summary_bullets", []):
    st.markdown(f"- {b}")


# Two-column block: neighbors + PC deviations
left, right = st.columns(2)

with left:
    st.subheader("Nearest neighbors")
    n_held = pitch.get("n_neighbors_currently_held", 0)
    if n_held:
        st.caption(f"5 closest in PC space · **{n_held}** currently held.")
    else:
        st.caption("5 closest in PC space.")
    rows = []
    for n in pitch.get("nearest_neighbors", []):
        flag = "HELD" if n.get("is_held") else ("FORMER" if n.get("is_former_hold") else "")
        rows.append({
            "Ticker": n["ticker"],
            "Sector": n.get("sector", ""),
            "Distance": n["distance"],
            "Flag": flag,
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, use_container_width=True,
            column_config={
                "Distance": st.column_config.NumberColumn(format="%.2f"),
                "Flag": st.column_config.TextColumn(width="small"),
            },
        )

with right:
    st.subheader("PC deviation from portfolio centroid")
    div_score = pitch.get("diversification_score", 0)
    st.caption(f"Diversification score: **{div_score:.0f} / 100**")
    pc_rows = []
    deviations = pitch.get("deviations_from_centroid", {})
    centroid = pitch.get("portfolio_centroid", {})
    candidate = pitch.get("candidate_position", {})
    meta = sl.load_meta()
    pc_labels = meta["pca"]["pc_labels"] if meta else {}
    for pc, dev in deviations.items():
        pc_rows.append({
            "PC": pc,
            "Label": pc_labels.get(pc, ""),
            "Candidate": candidate.get(pc, 0),
            "Centroid": centroid.get(pc, 0),
            "σ-deviation": dev,
        })
    if pc_rows:
        st.dataframe(
            pd.DataFrame(pc_rows), hide_index=True, use_container_width=True,
            column_config={
                "Candidate": st.column_config.NumberColumn(format="%.2f"),
                "Centroid": st.column_config.NumberColumn(format="%.2f"),
                "σ-deviation": st.column_config.NumberColumn(format="%+.2fσ"),
            },
        )


# Risk profile
st.subheader("Risk profile")
c1, c2, c3, c4 = st.columns(4)
with c1:
    sl.headline_metric(
        "Cluster", f"C{pitch['cluster_id']}",
        pitch.get("cluster_tier", "?"),
    )
with c2:
    sl.headline_metric(
        "Composite score", f"{pitch.get('composite_risk_score', 0):.0f}/100",
    )
with c3:
    sl.headline_metric(
        "Trajectory", pitch.get("cluster_trajectory", "Unknown"),
    )
with c4:
    sl.headline_metric(
        "vs sector", pitch.get("sector_comparison", ""),
        f"median {pitch.get('sector_median_score', 0):.1f}",
    )


# Risk drivers
drivers = pitch.get("top_risk_drivers", [])
if drivers:
    st.markdown("**Top risk drivers (universe percentile)**")
    for d in drivers:
        feat, pct = (d.get("feature"), d.get("percentile")) if isinstance(d, dict) else d
        color = (
            "#b3001b" if pct > 80 else
            "#e57a44" if pct > 60 else
            "#d4a017" if pct > 40 else
            "#2c7a4b"
        )
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;padding:3px 0">
              <code style="min-width:170px">{feat}</code>
              <div style="flex:1;background:#f0f2f6;border-radius:4px;height:14px">
                <div style="width:{pct:.0f}%;height:100%;background:{color};
                            border-radius:4px"></div>
              </div>
              <span style="min-width:40px;text-align:right;
                           font-variant-numeric:tabular-nums">{pct:.0f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption(f"Generated {pitch.get('generated_at', '')}")
