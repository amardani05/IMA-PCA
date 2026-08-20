/**
 * Client-side pitch assessment — a faithful TypeScript port of
 * `pitch_assessor.assess_pitch`. Everything it needs is already in the
 * exported pipeline JSON (universe.json + meta.json + trajectory.json), so
 * assessments generate instantly in the browser for ANY universe ticker —
 * no CLI round-trip.
 */
import { Meta, TrajectoryData, UniverseRow, ClusterMeta } from "./types";
import { FEATURE_KEYS, RISK_DIRECTION } from "./glossary";
import { PitchAssessment, PitchNeighbor } from "./pitch";

const PCS = ["PC1", "PC2", "PC3", "PC4"] as const;

// ---------------------------------------------------------------------------
// Per-feature percentile ranks (mirrors scoring.compute_composite_scores)
// ---------------------------------------------------------------------------
export type FeatureRanks = Map<string, Record<string, number>>;

/** pandas-style average percentile rank (ties get the mean rank), flipped for
 * risk direction so 100 = riskiest on every feature. */
export function computeFeatureRanks(universe: UniverseRow[]): FeatureRanks {
  const out: FeatureRanks = new Map(universe.map((u) => [u.Ticker, {}]));
  for (const f of FEATURE_KEYS) {
    const vals: { tk: string; v: number }[] = [];
    for (const u of universe) {
      const v = u[f];
      if (v != null && Number.isFinite(v)) vals.push({ tk: u.Ticker, v });
    }
    if (vals.length === 0) continue;
    vals.sort((a, b) => a.v - b.v);
    const n = vals.length;
    let i = 0;
    while (i < n) {
      let j = i;
      while (j + 1 < n && vals[j + 1].v === vals[i].v) j++;
      const avgRank = (i + j) / 2 + 1; // 1-based average rank across the tie
      let pct = (avgRank / n) * 100;
      if (RISK_DIRECTION[f] === -1) pct = 100 - pct;
      for (let k = i; k <= j; k++) out.get(vals[k].tk)![f] = pct;
      i = j + 1;
    }
  }
  return out;
}

export function topRiskDrivers(
  ranks: FeatureRanks, ticker: string, k = 3,
): { feature: string; percentile: number }[] {
  const row = ranks.get(ticker);
  if (!row) return [];
  return Object.entries(row)
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([feature, percentile]) => ({ feature, percentile }));
}

// ---------------------------------------------------------------------------
// Trajectory classification (mirrors trajectory.classify_trajectory)
// ---------------------------------------------------------------------------
export function classifyTrajectory(
  trajectory: TrajectoryData | null | undefined,
  riskRank: Record<number, number>,
  ticker: string,
): string {
  const path = trajectory?.paths?.[ticker];
  if (!path) return "Unknown";
  const cp = path.coords
    .map((c) => c.cluster)
    .filter((c): c is number => c != null && c >= 0);
  if (cp.length < 2) return "Unknown";
  const first = riskRank[cp[0]];
  const last = riskRank[cp[cp.length - 1]];
  if (first == null || last == null) return "Unknown";
  if (last > first) return "Deteriorating";
  if (last < first) return "Improving";
  return "Steady";
}

// ---------------------------------------------------------------------------
// Full assessment
// ---------------------------------------------------------------------------
export interface AssessmentInputs {
  universe: UniverseRow[];
  meta: Meta;
  clusterMeta: ClusterMeta;
  trajectory?: TrajectoryData | null;
  /** Precomputed once per session for speed; optional. */
  featureRanks?: FeatureRanks;
}

export function buildAssessment(
  tickerRaw: string,
  inputs: AssessmentInputs,
): PitchAssessment | { error: string } {
  const ticker = tickerRaw.trim().toUpperCase();
  const { universe, meta, clusterMeta, trajectory } = inputs;
  const byTicker = new Map(universe.map((u) => [u.Ticker, u]));
  const cand = byTicker.get(ticker);
  if (!cand || cand.PC1 == null) {
    return {
      error:
        `${ticker} is not in the scored universe. It is either not an S&P 600 ` +
        `constituent, or its features could not be computed in the latest run.`,
    };
  }

  const pcsOf = (u: UniverseRow) => PCS.map((pc) => (u[pc] as number) ?? 0);
  const candPcs = pcsOf(cand);

  // ----- 1. Nearest neighbors in PC space -----
  const dists: { u: UniverseRow; d: number }[] = [];
  for (const u of universe) {
    if (u.Ticker === ticker || u.PC1 == null) continue;
    const p = pcsOf(u);
    let s = 0;
    for (let i = 0; i < PCS.length; i++) s += (p[i] - candPcs[i]) ** 2;
    dists.push({ u, d: Math.sqrt(s) });
  }
  dists.sort((a, b) => a.d - b.d);
  const nearest = dists.slice(0, 5);

  let nHeld = 0;
  const neighbors: PitchNeighbor[] = nearest.map(({ u, d }) => {
    if (u.is_portfolio) nHeld++;
    return {
      ticker: u.Ticker, distance: d, is_held: !!u.is_portfolio,
      is_former_hold: false, company: u.Company ?? u.Ticker, sector: u.Sector ?? "",
    };
  });
  const similarityVerdict =
    nHeld >= 3 ? "highly similar to existing portfolio"
    : nHeld >= 1 ? "moderately overlaps with portfolio thesis ground"
    : "differentiated from current portfolio";

  // ----- 2. Portfolio centroid + deviations -----
  const held = universe.filter((u) => u.is_portfolio && u.PC1 != null);
  const wSum = held.reduce((s, u) => s + (u.weight ?? 0), 0) || held.length;
  const centroidArr = PCS.map((pc) =>
    held.reduce((s, u) => s + ((u[pc] as number) ?? 0) * ((u.weight ?? 0) || 1), 0) / wSum,
  );
  // Universe std per PC
  const stdArr = PCS.map((pc, i) => {
    const vals = universe.map((u) => u[pc] as number).filter((v) => v != null);
    const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
    const varr = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length;
    return Math.sqrt(varr) || 1;
  });

  const deviations: Record<string, number> = {};
  const significant: string[] = [];
  PCS.forEach((pc, i) => {
    const dev = (candPcs[i] - centroidArr[i]) / stdArr[i];
    deviations[pc] = dev;
    if (Math.abs(dev) > 1.0) {
      const dir = dev > 0 ? "above" : "below";
      const label = meta.pca.pc_labels[pc] ?? pc;
      significant.push(`${pc} (${label}): ${Math.abs(dev).toFixed(1)}σ ${dir} portfolio mean`);
    }
  });
  const avgAbsDev = PCS.reduce((s, pc) => s + Math.abs(deviations[pc]), 0) / PCS.length;
  const diversification = Math.min(100, avgAbsDev * 50);

  // ----- 3. Risk-model context -----
  const ranks = inputs.featureRanks ?? computeFeatureRanks(universe);
  const drivers = topRiskDrivers(ranks, ticker, 3);
  const clusterStyle = cand.cluster_style ?? clusterMeta.style_labels[cand.cluster] ?? "?";
  const riskTier = cand.risk_tier;
  const pctile = cand.score_percentile;

  // ----- 4. Sector context -----
  const peers = universe.filter((u) => u.Sector === cand.Sector && u.Ticker !== ticker);
  let sectorMedian = 0, delta = 0, sectorComp = "no sector peers in universe";
  if (peers.length > 0) {
    const scores = peers.map((p) => p.composite_score).sort((a, b) => a - b);
    const mid = Math.floor(scores.length / 2);
    sectorMedian = scores.length % 2 ? scores[mid] : (scores[mid - 1] + scores[mid]) / 2;
    delta = cand.composite_score - sectorMedian;
    sectorComp = delta > 0
      ? `riskier than sector (${delta >= 0 ? "+" : ""}${delta.toFixed(1)})`
      : `safer than sector (${delta.toFixed(1)})`;
  }

  // ----- 5. Trajectory -----
  const trajDir = classifyTrajectory(trajectory, clusterMeta.risk_rank, ticker);

  // ----- 6. Bullets + recommendation (mirrors _generate_* in Python) -----
  const bullets: string[] = [];
  const heldNames = neighbors.filter((n) => n.is_held).map((n) => n.ticker);
  if (heldNames.length > 0) {
    bullets.push(
      `Statistically similar to ${heldNames.length} current holding(s): ` +
      `${heldNames.join(", ")}. ${similarityVerdict[0].toUpperCase()}${similarityVerdict.slice(1)}.`,
    );
  } else {
    bullets.push(
      `Nearest neighbors in S&P 600: ${neighbors.slice(0, 3).map((n) => n.ticker).join(", ")}. ` +
      `None currently held — adds genuinely new statistical exposure.`,
    );
  }
  if (significant.length > 0) {
    bullets.push("Deviation from portfolio centroid: " + significant.slice(0, 2).join("; "));
  } else {
    bullets.push("Sits within 1σ of portfolio centroid on all retained PCs — limited statistical differentiation");
  }
  if (riskTier === "Elevated") {
    const dstr = drivers.slice(0, 2).map((d) => `${d.feature} (${d.percentile.toFixed(0)})`).join(", ");
    bullets.push(`⚠ Risk screener: Elevated tier — ${pctile.toFixed(0)}th percentile of the universe. Top risk drivers: ${dstr}`);
  } else {
    bullets.push(`Risk screener: ${riskTier} (${pctile.toFixed(0)}th percentile) — no elevated statistical risk signals`);
  }
  bullets.push(`Sector: ${cand.Sector}, ${sectorComp}`);

  const flags: string[] = [];
  if (riskTier === "Elevated") {
    flags.push(`risk screener places it in the Elevated tier (${pctile.toFixed(0)}th percentile of the universe)`);
  }
  if (nHeld >= 4) flags.push("nearly identical to existing holdings — limited diversification");
  else if (diversification < 15) flags.push("very low statistical differentiation from portfolio");
  if (delta > 15) flags.push("substantially riskier than sector peers");

  let recommendation: string, rationale: string;
  if (pctile > 90) {
    recommendation = "AVOID";
    rationale = "Statistical risk indicators are at extreme levels. " + flags.join("; ");
  } else if (flags.length >= 2) {
    recommendation = "QUESTION THESIS";
    rationale = "Multiple concerns require explicit address: " + flags.join("; ");
  } else if (flags.length === 1) {
    recommendation = "PROCEED WITH CAVEATS";
    rationale = "Single concern to acknowledge: " + flags[0];
  } else {
    recommendation = "PROCEED";
    rationale = "No statistical red flags. Standard fundamental due diligence applies.";
  }

  return {
    ticker,
    company_name: cand.Company ?? ticker,
    sector: cand.Sector ?? "Unknown",
    industry: cand.Industry ?? "",
    market_cap: cand.market_cap ?? 0,
    nearest_neighbors: neighbors,
    n_neighbors_currently_held: nHeld,
    n_neighbors_formerly_held: 0,
    similarity_verdict: similarityVerdict,
    portfolio_centroid: Object.fromEntries(PCS.map((pc, i) => [pc, centroidArr[i]])),
    candidate_position: Object.fromEntries(PCS.map((pc, i) => [pc, candPcs[i]])),
    deviations_from_centroid: deviations,
    significant_deviations: significant,
    diversification_score: diversification,
    cluster_id: cand.cluster,
    cluster_style: clusterStyle,
    risk_tier: riskTier,
    composite_risk_score: cand.composite_score,
    score_percentile: pctile,
    top_risk_drivers: drivers,
    cluster_trajectory: trajDir,
    sector_median_score: sectorMedian,
    delta_vs_sector: delta,
    sector_comparison: sectorComp,
    summary_bullets: bullets,
    recommendation,
    recommendation_rationale: rationale,
    generated_at: meta.generated_at,
    n_neighbors: 5,
  };
}

/** Percentile of a ticker's composite score within its own sector (0-100). */
export function sectorPercentile(universe: UniverseRow[], row: UniverseRow): number | null {
  const peers = universe.filter((u) => u.Sector === row.Sector);
  if (peers.length < 2) return null;
  const below = peers.filter((p) => p.composite_score < row.composite_score).length;
  const equal = peers.filter((p) => p.composite_score === row.composite_score).length;
  return ((below + equal / 2) / peers.length) * 100;
}
