"""Feature audit — diagnose which features are doing real work in the PCA.

For each feature in ``pca_result.feature_cols`` we report coverage (% non-null
before imputation), imputation rate (% values filled by the cross-sectional
median), cross-sectional standardized variance, range, skewness/kurtosis, and
correlation with each retained PC. Red flags are surfaced for any feature
that's heavily imputed, near-zero-variance, or fails to load on any PC.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def audit_features(
    raw_features: pd.DataFrame,
    clean_features: pd.DataFrame,
    pca_result,
    output_dir: Path,
) -> pd.DataFrame:
    """Per-feature audit. Writes ``feature_audit.csv`` and returns the table."""
    feature_cols = pca_result.feature_cols
    # The clean matrix may have dropped rows with >50% missing. Align raw to
    # clean's index for fair coverage comparison.
    common = raw_features.index.intersection(clean_features.index)
    raw = raw_features.loc[common]
    clean = clean_features.loc[common]
    n = len(common)

    rows = []
    for feat in feature_cols:
        if feat not in raw.columns or feat not in clean.columns:
            continue

        raw_col = raw[feat]
        clean_col = clean[feat]

        coverage = float(raw_col.notna().sum() / n) if n else 0.0
        imputation_rate = float((raw_col.isna() & clean_col.notna()).sum() / n) if n else 0.0

        std = float(clean_col.std(ddof=0))
        if std == 0:
            standardized_std = 0.0
            standardized_range = 0.0
            skewness = 0.0
            kurt = 0.0
        else:
            z = (clean_col - clean_col.mean()) / std
            standardized_std = float(z.std(ddof=0))
            standardized_range = float(z.quantile(0.95) - z.quantile(0.05))
            skewness = float(z.skew())
            kurt = float(z.kurtosis())

        pc_corrs: dict[str, float] = {}
        common_idx = pca_result.scores.index.intersection(clean.index)
        if len(common_idx) > 5:
            for pc in pca_result.scores.columns:
                corr = clean_col.loc[common_idx].corr(pca_result.scores.loc[common_idx, pc])
                pc_corrs[pc] = float(corr) if pd.notna(corr) else 0.0
        else:
            for pc in pca_result.scores.columns:
                pc_corrs[pc] = 0.0

        max_abs_pc_corr = max(abs(c) for c in pc_corrs.values()) if pc_corrs else 0.0

        flags: list[str] = []
        if coverage < 0.50:
            flags.append("LOW_COVERAGE")
        if imputation_rate > 0.30:
            flags.append("HIGH_IMPUTATION")
        if standardized_std < 0.5:
            flags.append("LOW_VARIANCE")
        if max_abs_pc_corr < 0.10:
            flags.append("NOT_LOADING")
        if abs(skewness) > 3:
            flags.append("EXTREME_SKEW")

        raw_dropped = raw_col.dropna()
        row = {
            "feature": feat,
            "coverage_pct": coverage * 100,
            "imputation_rate_pct": imputation_rate * 100,
            "raw_std": float(raw_col.std(ddof=0)) if not raw_dropped.empty else 0.0,
            "raw_p5": float(raw_col.quantile(0.05)) if not raw_dropped.empty else float("nan"),
            "raw_p95": float(raw_col.quantile(0.95)) if not raw_dropped.empty else float("nan"),
            "standardized_std": standardized_std,
            "standardized_range_p5_p95": standardized_range,
            "skewness": skewness,
            "kurtosis": kurt,
            "max_abs_pc_corr": max_abs_pc_corr,
            "red_flags": ",".join(flags),
        }
        for pc, corr in pc_corrs.items():
            row[f"corr_with_{pc}"] = corr
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("max_abs_pc_corr", ascending=False)
    output_path = output_dir / "feature_audit.csv"
    df.to_csv(output_path, index=False)
    logger.info("Wrote feature audit to %s", output_path)

    problematic = df[df["red_flags"] != ""]
    if not problematic.empty:
        logger.warning("=" * 60)
        logger.warning("FEATURES WITH RED FLAGS:")
        for _, row in problematic.iterrows():
            logger.warning(
                "  %-22s flags=%s  coverage=%.0f%%  std=%.2f  max_pc_corr=%.2f",
                row["feature"], row["red_flags"],
                row["coverage_pct"], row["standardized_std"], row["max_abs_pc_corr"],
            )
        logger.warning("=" * 60)
    return df


def write_audit_summary(audit_df: pd.DataFrame, output_dir: Path) -> None:
    """Write a human-readable companion summary."""
    lines = ["FEATURE AUDIT SUMMARY", "=" * 60, ""]

    lines.append("Top 5 features by max PC correlation:")
    for _, row in audit_df.head(5).iterrows():
        lines.append(
            f"  {row['feature']:<24} max_corr={row['max_abs_pc_corr']:.3f}  "
            f"coverage={row['coverage_pct']:.0f}%"
        )
    lines.append("")

    lines.append("Bottom 5 features by max PC correlation:")
    for _, row in audit_df.tail(5).iterrows():
        lines.append(
            f"  {row['feature']:<24} max_corr={row['max_abs_pc_corr']:.3f}  "
            f"coverage={row['coverage_pct']:.0f}%  flags={row['red_flags']}"
        )
    lines.append("")

    low_coverage = audit_df[audit_df["coverage_pct"] < 50]
    if not low_coverage.empty:
        lines.append("Features with LOW COVERAGE (<50%):")
        for _, row in low_coverage.iterrows():
            lines.append(
                f"  {row['feature']}: {row['coverage_pct']:.0f}% coverage "
                f"({row['imputation_rate_pct']:.0f}% imputed)"
            )
        lines.append("")
        lines.append(
            "  → Heavy imputation. The PCA is essentially blind to these "
            "features across most of the universe. Drop or fix the data source."
        )
        lines.append("")

    not_loading = audit_df[audit_df["max_abs_pc_corr"] < 0.10]
    if not not_loading.empty:
        lines.append("Features NOT LOADING on any PC (|corr| < 0.10):")
        for _, row in not_loading.iterrows():
            lines.append(
                f"  {row['feature']}: max_corr={row['max_abs_pc_corr']:.3f}"
            )
        lines.append("")
        lines.append(
            "  → Effectively noise relative to the dominant variance dimensions. "
            "Either zero-variance after standardization, or orthogonal to "
            "the structure PCA found. Consider removing."
        )
        lines.append("")

    (output_dir / "feature_audit_summary.txt").write_text("\n".join(lines))
    logger.info("Wrote feature_audit_summary.txt")
