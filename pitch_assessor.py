"""Pitch Candidate Assessment.

Translates the PCA + clustering output into a structured one-pager so PMs can
defend or attack a name in committee. The assessor doesn't run any new
analysis — it repackages existing pipeline output (PCA scores, cluster tier,
composite risk score, sector context, neighbors in PC space) into a standard
narrative format with a recommendation tag.

Usage::

    from pitch_assessor import assess_pitch

    a = assess_pitch(
        ticker="ABCD",
        pca_result=pca_result,
        cluster_result=cluster_result,
        features=clean_features,
        portfolio=config.PORTFOLIO,
        universe_meta=universe_df,
    )
    print(a.format_text())
    a.export_json("output/pitch_ABCD.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data structure
# =============================================================================
@dataclass
class PitchAssessment:
    # Identification
    ticker: str
    company_name: str
    sector: str
    industry: str
    market_cap: float

    # Statistical neighbors
    nearest_neighbors: list[dict]
    n_neighbors_currently_held: int
    n_neighbors_formerly_held: int
    similarity_verdict: str

    # Differentiation
    portfolio_centroid: dict
    candidate_position: dict
    deviations_from_centroid: dict      # {PC: σ-deviation}
    significant_deviations: list[str]
    diversification_score: float

    # Risk-model context
    cluster_id: int
    cluster_style: str
    risk_tier: str
    composite_risk_score: float
    score_percentile: float
    top_risk_drivers: list[tuple[str, float]]
    cluster_trajectory: str

    # Sector context
    sector_median_score: float
    delta_vs_sector: float
    sector_comparison: str

    # Narrative
    summary_bullets: list[str]
    recommendation: str
    recommendation_rationale: str

    # Metadata
    generated_at: str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())
    n_neighbors: int = 5

    def to_dict(self) -> dict:
        d = asdict(self)
        # Tuples are not JSON-native; convert risk drivers to {"feature": "...", "percentile": x}
        d["top_risk_drivers"] = [
            {"feature": f, "percentile": p} for f, p in self.top_risk_drivers
        ]
        return d

    def export_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def format_text(self) -> str:
        lines = []
        lines.append("=" * 76)
        lines.append(f" PITCH ASSESSMENT: {self.ticker}")
        lines.append(f" {self.company_name}")
        cap_str = f"${self.market_cap/1e9:.2f}B" if self.market_cap > 0 else "n/a"
        lines.append(f" {self.sector} / {self.industry}  |  Market cap: {cap_str}")
        lines.append("=" * 76)
        lines.append("")

        emoji = {"PROCEED": "✓", "PROCEED WITH CAVEATS": "✓⚠",
                 "QUESTION THESIS": "?", "AVOID": "✗"}.get(self.recommendation, "")
        lines.append(f" RECOMMENDATION: {emoji} {self.recommendation}")
        lines.append(f" {self.recommendation_rationale}")
        lines.append("")
        lines.append("-" * 76)

        lines.append(" KEY FINDINGS")
        for b in self.summary_bullets:
            lines.append(f"   • {b}")
        lines.append("")
        lines.append("-" * 76)

        lines.append(f" NEAREST NEIGHBORS ({self.n_neighbors} closest in PC space)")
        for n in self.nearest_neighbors:
            mark = " [HELD]" if n["is_held"] else (" [FORMER HOLD]" if n.get("is_former_hold") else "")
            sect = (n.get("sector") or "")[:24]
            lines.append(f"   {n['ticker']:<6} ({sect:<24}) d={n['distance']:.2f}{mark}")
        lines.append("")

        lines.append(" PORTFOLIO DIFFERENTIATION")
        lines.append(f"   Diversification score: {self.diversification_score:.0f}/100")
        if self.significant_deviations:
            lines.append("   Significant PC deviations:")
            for d in self.significant_deviations:
                lines.append(f"     • {d}")
        else:
            lines.append("   No PC deviations exceed 1σ — sits inside portfolio's typical region")
        lines.append("")

        lines.append(" RISK PROFILE")
        lines.append(f"   Style cluster: {self.cluster_id} ({self.cluster_style})")
        lines.append(f"   Risk tier: {self.risk_tier} "
                     f"({self.score_percentile:.0f}th percentile)")
        lines.append(f"   Composite score: {self.composite_risk_score:.0f}/100")
        lines.append(f"   Sector comparison: {self.sector_comparison}")
        if self.top_risk_drivers:
            lines.append("   Top risk drivers (percentile rank):")
            for f, p in self.top_risk_drivers:
                lines.append(f"     • {f}: {p:.0f}")
        lines.append(f"   Trajectory: {self.cluster_trajectory}")
        lines.append("")
        lines.append("=" * 76)
        return "\n".join(lines)


# =============================================================================
# Recommendation logic
# =============================================================================
def _generate_recommendation(
    n_neighbors_held: int,
    n_neighbors_formerly_held: int,
    risk_tier: str,
    score_percentile: float,
    diversification_score: float,
    sector_delta: float,
) -> tuple[str, str]:
    flags = []

    if risk_tier == "Elevated":
        flags.append(
            f"risk screener places it in the Elevated tier "
            f"({score_percentile:.0f}th percentile of the universe)")

    if n_neighbors_held >= 4:
        flags.append("nearly identical to existing holdings — limited diversification")
    elif diversification_score < 15:
        flags.append("very low statistical differentiation from portfolio")

    if sector_delta > 15:
        flags.append("substantially riskier than sector peers")

    if score_percentile > 90:
        return ("AVOID",
                "Statistical risk indicators are at extreme levels. " + "; ".join(flags))
    if len(flags) >= 2:
        return ("QUESTION THESIS",
                "Multiple concerns require explicit address: " + "; ".join(flags))
    if len(flags) == 1:
        return ("PROCEED WITH CAVEATS",
                "Single concern to acknowledge: " + flags[0])
    return ("PROCEED",
            "No statistical red flags. Standard fundamental due diligence applies.")


def _generate_summary_bullets(
    ticker: str,
    similarity_verdict: str,
    n_held: int,
    nearest_neighbors: list[dict],
    risk_tier: str,
    score_percentile: float,
    drivers: list[tuple[str, float]],
    significant_deviations: list[str],
    sector: str,
    sector_comp: str,
) -> list[str]:
    bullets = []

    held_tickers = [n["ticker"] for n in nearest_neighbors if n["is_held"]]
    if held_tickers:
        bullets.append(
            f"Statistically similar to {len(held_tickers)} current holding(s): "
            f"{', '.join(held_tickers)}. {similarity_verdict.capitalize()}."
        )
    else:
        nbs = ", ".join(n["ticker"] for n in nearest_neighbors[:3])
        bullets.append(
            f"Nearest neighbors in S&P 600: {nbs}. None currently held — "
            f"adds genuinely new statistical exposure."
        )

    if significant_deviations:
        bullets.append(
            "Deviation from portfolio centroid: " + "; ".join(significant_deviations[:2])
        )
    else:
        bullets.append(
            "Sits within 1σ of portfolio centroid on all retained PCs — "
            "limited statistical differentiation"
        )

    if risk_tier == "Elevated":
        driver_str = ", ".join(f"{d[0]} ({d[1]:.0f})" for d in drivers[:2])
        bullets.append(
            f"⚠ Risk screener: Elevated tier — {score_percentile:.0f}th percentile "
            f"of the universe. Top risk drivers: {driver_str}"
        )
    else:
        bullets.append(
            f"Risk screener: {risk_tier} ({score_percentile:.0f}th percentile) — "
            f"no elevated statistical risk signals"
        )

    bullets.append(f"Sector: {sector}, {sector_comp}")
    return bullets


# =============================================================================
# Core entry point
# =============================================================================
def assess_pitch(
    ticker: str,
    pca_result,
    cluster_result,
    features: pd.DataFrame,
    portfolio: dict[str, float],
    portfolio_history: list[str] | None = None,
    n_neighbors: int = 5,
    universe_meta: pd.DataFrame | None = None,
    trajectory=None,
) -> PitchAssessment:
    """Generate a structured assessment for a candidate ticker.

    The candidate must already exist in the PCA universe (i.e., have valid
    features and a PC projection). If it doesn't, raises ``ValueError`` with
    actionable guidance.
    """
    if ticker not in pca_result.scores.index:
        raise ValueError(
            f"{ticker} is not in the PCA universe. Most likely causes:\n"
            "  1. Not in the S&P 600 constituent list\n"
            "  2. Failed feature computation (yfinance returned no fundamentals)\n"
            "  3. Was dropped during cleaning (>50% features missing)\n"
            "Re-run `python main.py --refresh` and check the feature_engine "
            "log lines for this ticker."
        )

    # Local imports keep pitch_assessor decoupled when used in isolation
    from scoring import compute_composite_scores, top_risk_drivers

    # ---------- 1. Nearest neighbors in PC space ----------
    cand_pcs = pca_result.scores.loc[ticker]
    distances = ((pca_result.scores - cand_pcs) ** 2).sum(axis=1).pow(0.5).drop(ticker)
    nearest = distances.nsmallest(n_neighbors)

    held_set = set(portfolio.keys())
    historical_set = set(portfolio_history or [])

    nn_rows = []
    n_held = 0
    n_formerly = 0
    for tk, dist in nearest.items():
        is_held = tk in held_set
        is_former = (tk in historical_set) and not is_held
        if is_held:
            n_held += 1
        if is_former:
            n_formerly += 1
        company = (
            universe_meta.set_index("Ticker").loc[tk, "Company"]
            if universe_meta is not None and tk in universe_meta.set_index("Ticker").index
            else tk
        )
        sector = (
            universe_meta.set_index("Ticker").loc[tk, "Sector"]
            if universe_meta is not None and tk in universe_meta.set_index("Ticker").index
            else ""
        )
        nn_rows.append({
            "ticker": tk,
            "distance": float(dist),
            "is_held": is_held,
            "is_former_hold": is_former,
            "company": str(company) if company is not None else tk,
            "sector": str(sector) if sector is not None else "",
        })

    if n_held >= 3:
        similarity_verdict = "highly similar to existing portfolio"
    elif n_held >= 1 or n_formerly >= 2:
        similarity_verdict = "moderately overlaps with portfolio thesis ground"
    else:
        similarity_verdict = "differentiated from current portfolio"

    # ---------- 2. Centroid + deviation ----------
    held_in_universe = [t for t in portfolio if t in pca_result.scores.index]
    weights = pd.Series({t: portfolio[t] for t in held_in_universe})
    weights = weights / weights.sum() if weights.sum() > 0 else weights

    centroid = (pca_result.scores.loc[held_in_universe].T @ weights).to_dict()
    universe_std = pca_result.scores.std()

    deviations: dict[str, float] = {}
    significant: list[str] = []
    for pc in pca_result.scores.columns:
        std = float(universe_std[pc]) if universe_std[pc] else 1.0
        sigma_dev = (float(cand_pcs[pc]) - float(centroid[pc])) / std
        deviations[pc] = sigma_dev
        if abs(sigma_dev) > 1.0:
            direction = "above" if sigma_dev > 0 else "below"
            label = pca_result.pc_labels.get(pc, pc)
            significant.append(f"{pc} ({label}): {abs(sigma_dev):.1f}σ {direction} portfolio mean")

    avg_abs_dev = sum(abs(d) for d in deviations.values()) / max(len(deviations), 1)
    diversification_score = float(min(100.0, avg_abs_dev * 50.0))

    # ---------- 3. Risk-model context ----------
    cluster_id = int(cluster_result.assignments.get(ticker, -1))
    cluster_style = cluster_result.style_labels.get(cluster_id, "Unknown")
    pct_ranks = compute_composite_scores(features)
    composite_score = float(pct_ranks.loc[ticker, "composite_score"])
    score_percentile = float(pct_ranks.loc[ticker, "score_percentile"])
    risk_tier = str(pct_ranks.loc[ticker, "risk_tier"])
    drivers = top_risk_drivers(pct_ranks, ticker, k=3)

    # ---------- 4. Sector context ----------
    if universe_meta is not None and ticker in universe_meta.set_index("Ticker").index:
        meta_idx = universe_meta.set_index("Ticker")
        sector = str(meta_idx.loc[ticker, "Sector"]) if pd.notna(meta_idx.loc[ticker, "Sector"]) else "Unknown"
        industry = str(meta_idx.loc[ticker, "Industry"]) if pd.notna(meta_idx.loc[ticker, "Industry"]) else ""
        company_name = str(meta_idx.loc[ticker, "Company"]) if pd.notna(meta_idx.loc[ticker, "Company"]) else ticker
    else:
        sector = str(features.loc[ticker, "Sector"]) if "Sector" in features.columns else "Unknown"
        industry = ""
        company_name = (
            str(features.loc[ticker, "Company"])
            if "Company" in features.columns else ticker
        )

    sector_peers = (
        features[features["Sector"] == sector].index
        if "Sector" in features.columns
        else pd.Index([])
    ).intersection(pct_ranks.index).difference([ticker])

    if len(sector_peers) > 0:
        sector_median = float(pct_ranks.loc[sector_peers, "composite_score"].median())
        delta = composite_score - sector_median
        comp_label = (
            f"riskier than sector ({delta:+.1f})" if delta > 0
            else f"safer than sector ({delta:+.1f})"
        )
    else:
        sector_median = 0.0
        delta = 0.0
        comp_label = "no sector peers in universe"

    # ---------- 5. Trajectory ----------
    cluster_trajectory = "Unknown"
    if trajectory is not None:
        try:
            from trajectory import classify_trajectory
            cluster_trajectory = classify_trajectory(trajectory, cluster_result, ticker)
        except Exception as exc:  # noqa: BLE001
            logger.debug("trajectory lookup for %s failed: %s", ticker, exc)

    # ---------- 6. Bullets + recommendation ----------
    bullets = _generate_summary_bullets(
        ticker, similarity_verdict, n_held, nn_rows,
        risk_tier, score_percentile, drivers,
        significant, sector, comp_label,
    )
    rec, rationale = _generate_recommendation(
        n_held, n_formerly, risk_tier, score_percentile,
        diversification_score, delta,
    )

    market_cap = 0.0
    if "market_cap" in features.columns:
        try:
            mc = features.loc[ticker, "market_cap"]
            market_cap = float(mc) if pd.notna(mc) else 0.0
        except (KeyError, ValueError, TypeError):
            market_cap = 0.0

    return PitchAssessment(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        industry=industry,
        market_cap=market_cap,
        nearest_neighbors=nn_rows,
        n_neighbors_currently_held=n_held,
        n_neighbors_formerly_held=n_formerly,
        similarity_verdict=similarity_verdict,
        portfolio_centroid={k: float(v) for k, v in centroid.items()},
        candidate_position={pc: float(cand_pcs[pc]) for pc in pca_result.scores.columns},
        deviations_from_centroid=deviations,
        significant_deviations=significant,
        diversification_score=diversification_score,
        cluster_id=cluster_id,
        cluster_style=cluster_style,
        risk_tier=risk_tier,
        composite_risk_score=composite_score,
        score_percentile=score_percentile,
        top_risk_drivers=drivers,
        cluster_trajectory=cluster_trajectory,
        sector_median_score=sector_median,
        delta_vs_sector=delta,
        sector_comparison=comp_label,
        summary_bullets=bullets,
        recommendation=rec,
        recommendation_rationale=rationale,
        n_neighbors=n_neighbors,
    )


# =============================================================================
# Batch
# =============================================================================
def assess_batch(
    tickers: list[str],
    pca_result,
    cluster_result,
    features: pd.DataFrame,
    portfolio: dict[str, float],
    portfolio_history: list[str] | None = None,
    universe_meta: pd.DataFrame | None = None,
    trajectory=None,
    output_dir: str | Path = "output",
) -> list[PitchAssessment]:
    """Generate one assessment per ticker, persisting JSON for each."""
    out: list[PitchAssessment] = []
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for tk in tickers:
        try:
            a = assess_pitch(
                ticker=tk,
                pca_result=pca_result,
                cluster_result=cluster_result,
                features=features,
                portfolio=portfolio,
                portfolio_history=portfolio_history,
                universe_meta=universe_meta,
                trajectory=trajectory,
            )
            a.export_json(out_path / f"pitch_{tk}.json")
            out.append(a)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pitch assessment failed for %s: %s", tk, exc)
    return out
