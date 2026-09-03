/**
 * Universe-wide per-stock factor betas — the bridge between the risk screener
 * and the factor library.
 *
 * These are RAW (unconditional) betas, which is exactly what makes them
 * composable: a portfolio's raw beta to factor k is the weight-average of its
 * holdings' raw betas. That identity is what lets the UI answer "what does
 * adding this name do to our exposure?" without re-running any regression.
 *
 * Exposure relative to the benchmark is then `stockBeta - indexBeta`.
 */

export interface UniverseFactorBetas {
  available: boolean;
  tickers: string[];
  factors: string[];
  index_betas: Record<string, number>;
  betas: Record<string, (number | null)[]>;
  p_values: Record<string, (number | null)[]>;
}

export interface AttributionRow {
  factor: string;
  beta: number;
  factor_move: number;
  contribution: number;
  p_value: number;
  significant_10: boolean;
}

export interface Attribution {
  available: boolean;
  window?: [string, string];
  n_obs?: number;
  total_active_return?: number;
  factor_explained?: number;
  selection_residual?: number;
  r_squared?: number;
  contributions?: AttributionRow[];
}

const BASE = "/data/macro";

export async function loadUniverseFactorBetas(): Promise<UniverseFactorBetas | null> {
  try {
    const r = await fetch(`${BASE}/universe_factor_betas.json`);
    if (!r.ok) return null;
    const j = (await r.json()) as UniverseFactorBetas;
    return j.available ? j : null;
  } catch {
    return null;
  }
}

export async function loadAttribution(): Promise<Attribution | null> {
  try {
    const r = await fetch(`${BASE}/attribution.json`);
    if (!r.ok) return null;
    const j = (await r.json()) as Attribution;
    return j.available ? j : null;
  } catch {
    return null;
  }
}

/** One stock's beta to one factor, or null when it wasn't estimable. */
export function betaOf(
  ub: UniverseFactorBetas, ticker: string, factor: string,
): number | null {
  const i = ub.factors.indexOf(factor);
  if (i < 0) return null;
  return ub.betas[ticker]?.[i] ?? null;
}

export function pValueOf(
  ub: UniverseFactorBetas, ticker: string, factor: string,
): number | null {
  const i = ub.factors.indexOf(factor);
  if (i < 0) return null;
  return ub.p_values[ticker]?.[i] ?? null;
}

/** Weight-average of holdings' raw betas = the sleeve's raw beta. */
export function portfolioBeta(
  ub: UniverseFactorBetas,
  weights: { ticker: string; weight: number }[],
  factor: string,
): number | null {
  let wsum = 0;
  let acc = 0;
  for (const { ticker, weight } of weights) {
    const b = betaOf(ub, ticker, factor);
    if (b == null || !Number.isFinite(weight)) continue;
    acc += b * weight;
    wsum += weight;
  }
  return wsum > 0 ? acc / wsum : null;
}

/**
 * What happens to the sleeve's factor exposure if `ticker` is added at
 * `weight` (existing holdings scaled down pro-rata to make room).
 */
export function exposureWithCandidate(
  ub: UniverseFactorBetas,
  holdings: { ticker: string; weight: number }[],
  candidate: string,
  weight: number,
  factor: string,
): { before: number | null; after: number | null; delta: number | null } {
  const before = portfolioBeta(ub, holdings, factor);
  const candBeta = betaOf(ub, candidate, factor);
  if (before == null || candBeta == null) {
    return { before, after: null, delta: null };
  }
  const after = before * (1 - weight) + candBeta * weight;
  return { before, after, delta: after - before };
}
