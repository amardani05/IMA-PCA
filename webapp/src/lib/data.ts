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

export function tierClass(tier?: string | null): string {
  switch (tier) {
    // Current 3-tier vocabulary
    case "Stable":     return "tier-pill tier-stable";
    case "Mainstream": return "tier-pill tier-mainstream";
    case "Elevated":   return "tier-pill tier-elevated";
    // Legacy 5-tier fallback (older cached payloads)
    case "Low Risk":   return "tier-pill tier-low";
    case "Moderate":   return "tier-pill tier-mod";
    case "High":       return "tier-pill tier-high";
    case "Critical":   return "tier-pill tier-crit";
    default:           return "tier-pill";
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
