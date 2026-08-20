import {
  ClusterMeta,
  ClusterRow,
  DriftRow,
  Meta,
  OpportunityRow,
  PCALoadingRow,
  PCASummaryRow,
  PortfolioRow,
  TrajectoryData,
  UniverseRow,
} from "./types";

async function getJSON<T>(path: string): Promise<T> {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`Failed to fetch ${path}: ${resp.status}`);
  return (await resp.json()) as T;
}

export async function loadAll() {
  const [
    meta,
    universe,
    portfolio,
    clusters,
    clusterMeta,
    pcaSummary,
    pcaLoadings,
    opportunities,
    drift,
    trajectory,
  ] = await Promise.all([
    getJSON<Meta>("/meta.json"),
    getJSON<UniverseRow[]>("/data/universe.json"),
    getJSON<PortfolioRow[]>("/data/portfolio.json"),
    getJSON<ClusterRow[]>("/data/clusters.json"),
    getJSON<ClusterMeta>("/data/cluster_meta.json"),
    getJSON<PCASummaryRow[]>("/data/pca_summary.json"),
    getJSON<PCALoadingRow[]>("/data/pca_loadings.json"),
    getJSON<OpportunityRow[]>("/data/opportunities.json"),
    getJSON<DriftRow[]>("/data/drift_alerts.json"),
    getJSON<TrajectoryData>("/data/trajectory.json"),
  ]);
  return {
    meta, universe, portfolio, clusters, clusterMeta,
    pcaSummary, pcaLoadings, opportunities, drift, trajectory,
  };
}

export async function loadPlotlyFigure(name: string): Promise<any> {
  return getJSON<any>(`/interactive/${name}.json`);
}

// Loaded lazily by BacktestView — absent backtest.json (no `--backtest` run yet)
// is a soft failure (the tab shows an instruction), not a hard app crash.
export async function loadBacktest(): Promise<import("./types").BacktestData> {
  const resp = await fetch("/data/backtest.json");
  if (!resp.ok) throw new Error(`backtest.json not found (${resp.status})`);
  return (await resp.json()) as import("./types").BacktestData;
}

export function tierClass(tier?: string | null): string {
  switch (tier) {
    case "Low Risk": return "tier-pill tier-low";
    case "In Line":  return "tier-pill tier-inline";
    case "Elevated": return "tier-pill tier-elevated";
    default:         return "tier-pill";
  }
}

export function fmt(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(digits)}%`;
}

export function fmtPctRank(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v.toFixed(digits)}`;
}
