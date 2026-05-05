"""Interpret the extracted PCs.

- Correlate PC score series against FF5/momentum, sector ETFs, and the benchmark.
- Assign a heuristic label to each PC. These labels are starting points, not
  ground truth — always cross-check against the top/bottom loadings.
"""

from __future__ import annotations

import logging

import pandas as pd

from config import SECTOR_ETFS

logger = logging.getLogger(__name__)


def build_factor_correlation_matrix(
    pc_scores: pd.DataFrame,
    ff_factors: pd.DataFrame,
    sector_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Correlate each PC score series with FF factors, sector ETFs, and benchmark.

    Returns a DataFrame indexed by PC with one column per external factor.
    All series are aligned on common dates before correlating.
    """
    benchmark = benchmark_returns.rename("Benchmark").to_frame()

    # Drop RF when interpreting — it's a risk-free rate level, not a factor return
    ff_cols = [c for c in ff_factors.columns if c != "RF"]
    ff = ff_factors[ff_cols]

    # Rename sector ETFs to human-readable sector names for the heatmap
    sector_named = sector_returns.rename(columns=SECTOR_ETFS)

    combined = ff.join(sector_named, how="outer").join(benchmark, how="outer")
    aligned = pc_scores.join(combined, how="inner").dropna()

    corr = aligned.corr()
    pc_cols = list(pc_scores.columns)
    other_cols = [c for c in aligned.columns if c not in pc_cols]

    return corr.loc[pc_cols, other_cols]


def _label_pc(corr_row: pd.Series) -> str:
    """Heuristic label for a single PC based on its factor correlations.

    Thresholds are rules of thumb; inspect the loadings before citing.
    """
    mkt = corr_row.get("Mkt-RF", 0.0)
    smb = corr_row.get("SMB", 0.0)
    hml = corr_row.get("HML", 0.0)
    mom = corr_row.get("Mom", 0.0)
    rmw = corr_row.get("RMW", 0.0)
    bench = corr_row.get("Benchmark", 0.0)

    # Small-cap PC1 correlates more tightly with IJR than with FF's large-cap-heavy
    # Mkt-RF, so a strong benchmark correlation is the more reliable market-beta tell.
    if abs(mkt) > 0.7 or abs(bench) > 0.7:
        return "Market Beta"
    if abs(hml) > 0.4 and abs(mkt) < 0.5:
        return "Value/Growth"
    if abs(mom) > 0.4:
        return "Momentum"
    if abs(rmw) > 0.4:
        return "Quality/Profitability"
    if abs(smb) > 0.4 and abs(mkt) < 0.5:
        return "Size (within small-cap)"

    sector_cols = list(SECTOR_ETFS.values())
    sector_slice = corr_row.reindex(sector_cols).dropna()
    if not sector_slice.empty:
        top_sector = sector_slice.abs().idxmax()
        top_val = sector_slice[top_sector]
        if abs(top_val) > 0.5:
            sign = "" if top_val > 0 else "Short "
            return f"Sector: {sign}{top_sector}"

    return "Unclassified (inspect loadings)"


def label_pcs(factor_corr: pd.DataFrame) -> pd.Series:
    """Assign a label to each PC. Always verify via top/bottom loadings."""
    labels = factor_corr.apply(_label_pc, axis=1)
    labels.name = "Label"
    return labels


def summarize_top_bottom(
    top_bottom: dict[str, dict[str, pd.Series]],
    universe_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Flatten ``top_bottom_loadings`` output into a tidy DataFrame for export.

    If ``universe_meta`` (Ticker/Sector/Industry) is provided, annotate each
    row with its sector — very helpful when reading the report.
    """
    rows = []
    meta = (
        universe_meta.set_index("Ticker")[["Sector", "Industry"]]
        if universe_meta is not None and "Ticker" in universe_meta.columns
        else None
    )
    for pc, d in top_bottom.items():
        for side, ser in d.items():
            for rank, (ticker, loading) in enumerate(ser.items(), start=1):
                row = {
                    "PC": pc,
                    "Side": side,
                    "Rank": rank,
                    "Ticker": ticker,
                    "Loading": float(loading),
                }
                if meta is not None and ticker in meta.index:
                    row["Sector"] = meta.loc[ticker, "Sector"]
                    row["Industry"] = meta.loc[ticker, "Industry"]
                rows.append(row)
    return pd.DataFrame(rows)
