import { useMemo, useState } from "react";
import {
  ClusterMeta, ClusterRow, Meta, PCALoadingRow, PCASummaryRow,
  PortfolioRow, TrajectoryData, UniverseRow,
} from "../lib/types";
import { usePortfolioSelection } from "../hooks/usePortfolioSelection";
import { tierClass, fmt } from "../lib/data";
import { PCATree } from "./PCATree";
import {
  InteractivePCAChart,
  InteractivePCAChart3D,
  ChartFilters,
  NO_FILTERS,
} from "./InteractivePCAChart";
import { ChartFilterBar } from "./ChartFilterBar";

interface Props {
  meta: Meta;
  portfolio: PortfolioRow[];
  universe: UniverseRow[];
  clusterMeta: ClusterMeta;
  trajectory?: TrajectoryData | null;
  pcaSummary: PCASummaryRow[];
  pcaLoadings: PCALoadingRow[];
  clusters: ClusterRow[];
}

export function Overview(props: Props) {
  const { meta, portfolio, universe, clusterMeta, trajectory,
          pcaSummary, pcaLoadings, clusters } = props;

  const selection = usePortfolioSelection({
    defaultPortfolio: portfolio,
    universe,
    stockBetas: null,
  });
  const [filters, setFilters] = useState<ChartFilters>(NO_FILTERS);

  const portfolioTickers = useMemo(
    () => new Set(portfolio.map((p) => p.Ticker)),
    [portfolio],
  );

  const sharedChartProps = {
    universe,
    meta,
    clusterMeta,
    selectedTickers: selection.selectedTickers,
    filters,
    trajectory,
    portfolioTickers,
    showCentroids: true,
  };

  return (
    <div>
      <h2 className="section-title">PCA explorer</h2>
      <p className="section-lede">
        Toggle stocks in the sector tiles below to highlight them on every
        chart. Trajectory polylines show portfolio holdings' paths through PC
        space over recent quarters. Use the filter bar to hide non-portfolio
        dots, narrow to specific tiers / sectors, or turn off trajectory overlays.
      </p>

      <PCATree
        portfolio={portfolio}
        universe={universe}
        selectedTickers={selection.selectedTickers}
        toggleStock={selection.toggleStock}
        toggleSector={selection.toggleSector}
        selectAll={selection.selectAll}
        clearAll={selection.clearAll}
        resetToPortfolio={selection.resetToDefault}
      />

      <div style={{ marginTop: 16 }}>
        <ChartFilterBar filters={filters} setFilters={setFilters}
                        meta={meta} universe={universe} />
      </div>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
        gap: 12,
      }}>
        <ChartCard title="PC1 vs PC2"
                   sub="Dominant risk-structure view — most variance captured here.">
          <InteractivePCAChart {...sharedChartProps} pcX="PC1" pcY="PC2" height={460} />
        </ChartCard>

        <ChartCard title="PC1 vs PC3"
                   sub="Adds the next-strongest dimension; separates sentiment from health.">
          <InteractivePCAChart {...sharedChartProps} pcX="PC1" pcY="PC3" height={460} />
        </ChartCard>

        <ChartCard title="PC2 vs PC3"
                   sub="Cross-section orthogonal to PC1.">
          <InteractivePCAChart {...sharedChartProps} pcX="PC2" pcY="PC3" height={460} />
        </ChartCard>

        <ChartCard title="3D PCA"
                   sub="Rotate to inspect cluster geometry across PC1/PC2/PC3.">
          <InteractivePCAChart3D {...sharedChartProps} height={500} />
        </ChartCard>
      </div>

      <ClustersBlock meta={meta} clusters={clusters} clusterMeta={clusterMeta}
                     universe={universe} />

      <PCABlock summary={pcaSummary} loadings={pcaLoadings} />
    </div>
  );
}


function ChartCard({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="card" style={{ padding: 12 }}>
      <h3 style={{ margin: 0, fontSize: 14 }}>{title}</h3>
      {sub && <div className="card-sub" style={{ fontSize: 11 }}>{sub}</div>}
      {children}
    </div>
  );
}


// =============================================================================
// Clusters block (formerly its own tab)
// =============================================================================
function ClustersBlock({
  meta, clusters, clusterMeta, universe,
}: {
  meta: Meta;
  clusters: ClusterRow[];
  clusterMeta: ClusterMeta;
  universe: UniverseRow[];
}) {
  const sectorByCluster = useMemo(() => {
    const m: Record<number, Record<string, number>> = {};
    for (const u of universe) {
      const c = u.cluster;
      const s = u.Sector ?? "Unknown";
      m[c] ??= {};
      m[c][s] = (m[c][s] ?? 0) + 1;
    }
    return m;
  }, [universe]);

  return (
    <div style={{ marginTop: 28 }}>
      <h2 className="section-title">Clusters</h2>
      <p className="section-lede">
        k-means was run for k ∈ {"{3..7}"}. The {meta.clustering.k}-cluster solution
        was selected on a silhouette score of {fmt(meta.clustering.silhouette, 3)}. Tiers are
        assigned by composite-risk rank so the labels stay stable across runs.
      </p>

      <div style={{
        display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12,
      }}>
        <div className="card">
          <h3 style={{ margin: 0 }}>Cluster summary</h3>
          <table className="data">
            <thead>
              <tr><th>Cluster</th><th>Tier</th><th>Count</th><th>Top sectors</th></tr>
            </thead>
            <tbody>
              {clusters.map((c) => {
                const sectors = sectorByCluster[c.cluster] ?? {};
                const topSectors = Object.entries(sectors)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 3)
                  .map(([k, v]) => `${k} (${v})`)
                  .join(", ");
                return (
                  <tr key={c.cluster}>
                    <td><strong>C{c.cluster}</strong></td>
                    <td><span className={tierClass(c.tier)}>{c.tier}</span></td>
                    <td className="num">{c.n_stocks}</td>
                    <td><small>{topSectors || "—"}</small></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3 style={{ margin: 0 }}>k-selection diagnostics</h3>
          <table className="data">
            <thead>
              <tr>
                <th>k</th>
                <th className="num">Silhouette</th>
                <th className="num">Inertia</th>
                <th className="num">Calinski-Harabasz</th>
              </tr>
            </thead>
            <tbody>
              {clusterMeta.diagnostics.map((d) => (
                <tr key={d.k} className={d.k === clusterMeta.k ? "highlight" : ""}>
                  <td><strong>{d.k}</strong></td>
                  <td className="num">{fmt(d.silhouette, 3)}</td>
                  <td className="num">{fmt(d.inertia, 1)}</td>
                  <td className="num">{fmt(d.calinski_harabasz, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <small className="muted">Highlighted row = selected k</small>
        </div>
      </div>

      <div className="card">
        <h3 style={{ margin: 0 }}>Cluster feature profiles</h3>
        <div className="card-sub">
          Z-scored feature means per cluster — makes the cluster signatures explicit.
        </div>
        <div className="static-chart">
          <img src="/charts/cluster_profiles.png" alt="Cluster profiles" />
        </div>
      </div>
    </div>
  );
}


// =============================================================================
// PCA block (formerly its own tab)
// =============================================================================
function PCABlock({
  summary, loadings,
}: {
  summary: PCASummaryRow[];
  loadings: PCALoadingRow[];
}) {
  const pcs = summary.map((s) => s.pc);

  return (
    <div style={{ marginTop: 28 }}>
      <h2 className="section-title">PCA decomposition</h2>
      <p className="section-lede">
        Features are z-scored before PCA. Each PC is auto-labeled by its
        dominant-loading feature family; cross-check the labels against the
        loadings table before citing them.
      </p>

      <div style={{
        display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12,
      }}>
        <div className="card">
          <h3 style={{ margin: 0 }}>Variance explained</h3>
          <table className="data">
            <thead>
              <tr><th>PC</th><th>Variance</th><th>Cumulative</th><th>Label</th></tr>
            </thead>
            <tbody>
              {summary.map((s) => (
                <tr key={s.pc}>
                  <td><strong>{s.pc}</strong></td>
                  <td className="num">{(s.variance_explained * 100).toFixed(2)}%</td>
                  <td className="num">{(s.cumulative_variance * 100).toFixed(2)}%</td>
                  <td>{s.label}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3 style={{ margin: 0 }}>Dominant features per PC</h3>
          {summary.map((s) => (
            <div key={s.pc} style={{ marginBottom: 10 }}>
              <strong>{s.pc}</strong> <small className="muted">({s.label})</small>
              <div style={{ fontSize: 12, marginTop: 2 }}>
                {s.top_loadings.map((t, i) => (
                  <span key={t.feature} style={{ marginRight: 10 }}>
                    {i > 0 && " · "}
                    <code>{t.feature}</code>{" "}
                    <span style={{ color: t.loading >= 0 ? "var(--ok)" : "var(--danger)" }}>
                      {t.loading >= 0 ? "+" : ""}{t.loading.toFixed(2)}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3 style={{ margin: 0 }}>Full loadings matrix</h3>
        <div className="card-sub">Rows are features, columns are PCs. Cell color scales with magnitude.</div>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Feature</th>
                {pcs.map((pc) => <th key={pc} className="num">{pc}</th>)}
              </tr>
            </thead>
            <tbody>
              {loadings.map((row) => (
                <tr key={row.feature}>
                  <td><code>{row.feature}</code></td>
                  {pcs.map((pc) => {
                    const v = row[pc] as number;
                    const mag = Math.min(Math.abs(v ?? 0), 0.7) / 0.7;
                    const bg = v >= 0
                      ? `rgba(44,122,75,${(mag * 0.65).toFixed(2)})`
                      : `rgba(179,0,27,${(mag * 0.65).toFixed(2)})`;
                    return (
                      <td key={pc} className="num" style={{ background: bg }}>
                        {v != null ? (v >= 0 ? "+" : "") + v.toFixed(2) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3 style={{ margin: 0 }}>Static loadings heatmap</h3>
        <div className="static-chart">
          <img src="/charts/pca_loadings.png" alt="PCA loadings" />
        </div>
      </div>
    </div>
  );
}
