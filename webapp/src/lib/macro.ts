import {
  ComparisonRow, FactorMetadata, MacroBundle, MacroSummary, MacroTimeframes,
  PortfolioBetas, RollingBetas, FactorReturns, ScenarioRow, StockBetaMatrix,
} from "./macroTypes";

async function getJSON<T>(path: string): Promise<T> {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`Failed to fetch ${path}: ${resp.status}`);
  return (await resp.json()) as T;
}

const BASE = "/data/macro";

export async function loadMacro(): Promise<MacroBundle | null> {
  try {
    const [
      metadata, portfolioBetas, portfolioBetasV1, portfolioBetasRaw,
      portfolioBetasFull, comparison, stockBetas, rollingBetas,
      factorReturns, scenarios, summary, timeframes,
    ] = await Promise.all([
      getJSON<FactorMetadata>(`${BASE}/factor_metadata.json`),
      getJSON<PortfolioBetas>(`${BASE}/portfolio_betas.json`),
      getJSON<PortfolioBetas>(`${BASE}/portfolio_betas_v1.json`).catch(() => null),
      getJSON<PortfolioBetas>(`${BASE}/portfolio_betas_raw.json`).catch(() => null),
      getJSON<PortfolioBetas>(`${BASE}/portfolio_betas_full.json`).catch(() => null),
      getJSON<{ rows: ComparisonRow[] }>(`${BASE}/raw_vs_residualized.json`).catch(() => ({ rows: [] })),
      getJSON<StockBetaMatrix>(`${BASE}/stock_betas.json`),
      getJSON<RollingBetas>(`${BASE}/rolling_betas.json`),
      getJSON<FactorReturns>(`${BASE}/factor_returns.json`),
      getJSON<{ scenarios: ScenarioRow[] }>(`${BASE}/scenarios.json`),
      getJSON<MacroSummary>(`${BASE}/macro_summary.json`),
      getJSON<MacroTimeframes>(`${BASE}/timeframes.json`).catch(() => null),
    ]);
    return {
      metadata,
      portfolioBetas,
      portfolioBetasV1: portfolioBetasV1 as any,
      portfolioBetasRaw: portfolioBetasRaw as any,
      portfolioBetasFull: portfolioBetasFull as any,
      comparison,
      stockBetas, rollingBetas, factorReturns, scenarios, summary,
      timeframes: timeframes as any,
    };
  } catch {
    return null;
  }
}

/**
 * Recompute portfolio betas from per-stock betas given a selection of tickers
 * with renormalized weights. Mathematically exact for the beta point estimate
 * (beta is a linear functional of returns), but residual variance / R² are
 * NOT recomputed — those depend on covariance structure.
 */
export function recomputePortfolioBetas(
  selectedTickers: Set<string>,
  weights: Record<string, number>,
  matrix: StockBetaMatrix,
): { betas: Record<string, number>; effectiveWeights: Record<string, number> } {
  const totalWeight = Array.from(selectedTickers)
    .reduce((sum, t) => sum + (weights[t] ?? 0), 0);

  const effectiveWeights: Record<string, number> = {};
  selectedTickers.forEach((t) => {
    effectiveWeights[t] = totalWeight > 0 ? (weights[t] ?? 0) / totalWeight : 0;
  });

  const portfolioBetas: Record<string, number> = {};
  for (const factor of matrix.factors) {
    let sum = 0;
    selectedTickers.forEach((t) => {
      const beta = matrix.betas[t]?.[factor] ?? 0;
      sum += (effectiveWeights[t] ?? 0) * beta;
    });
    portfolioBetas[factor] = sum;
  }
  return { betas: portfolioBetas, effectiveWeights };
}

export function computeScenarioImpacts(
  betas: Record<string, number>,
  shocks: Record<string, { label: string; shock: number }>,
): { factor: string; label: string; shock: number; beta: number; impact: number }[] {
  const out: { factor: string; label: string; shock: number; beta: number; impact: number }[] = [];
  for (const factor of Object.keys(betas)) {
    const sh = shocks[factor];
    if (!sh) continue;
    const beta = betas[factor];
    out.push({ factor, label: sh.label, shock: sh.shock, beta, impact: beta * sh.shock });
  }
  out.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  return out;
}

export const CATEGORY_ORDER = [
  "rates", "credit", "inflation", "commodities", "currency",
  "volatility_liquidity", "financial_conditions", "thematic", "data_center_proxies",
] as const;

export const CATEGORY_LABELS: Record<string, string> = {
  rates: "Rates",
  credit: "Credit",
  inflation: "Inflation",
  commodities: "Commodities",
  currency: "Currency",
  volatility_liquidity: "Volatility / Liquidity",
  financial_conditions: "Financial Conditions",
  thematic: "Thematic",
  data_center_proxies: "Data Center Proxies",
};

export function significanceStars(p: number): string {
  if (p < 0.01) return "★★★";
  if (p < 0.05) return "★★";
  if (p < 0.10) return "★";
  return "";
}
