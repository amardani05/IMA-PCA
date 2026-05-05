import { useMemo } from "react";
// @ts-ignore
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { ClusterMeta, Meta, TrajectoryData, UniverseRow } from "../lib/types";

const Plot = createPlotlyComponent(Plotly);

export interface ChartFilters {
  portfolioOnly: boolean;            // hide non-portfolio dots entirely
  tiers: Set<string>;                // empty = all tiers
  sectors: Set<string>;              // empty = all sectors
  showTrajectories: boolean;
}

export const NO_FILTERS: ChartFilters = {
  portfolioOnly: false,
  tiers: new Set(),
  sectors: new Set(),
  showTrajectories: true,
};

interface BaseProps {
  universe: UniverseRow[];
  meta: Meta;
  clusterMeta: ClusterMeta;
  selectedTickers: Set<string>;
  filters: ChartFilters;
  trajectory?: TrajectoryData | null;
  portfolioTickers: Set<string>;
  height?: number;
  showCentroids?: boolean;
  title?: string;
}

interface Props2D extends BaseProps {
  pcX: "PC1" | "PC2" | "PC3" | "PC4";
  pcY: "PC1" | "PC2" | "PC3" | "PC4";
}

function applyFilters(
  rows: UniverseRow[],
  filters: ChartFilters,
  portfolioTickers: Set<string>,
): UniverseRow[] {
  return rows.filter((u) => {
    if (filters.portfolioOnly && !portfolioTickers.has(u.Ticker)) return false;
    if (filters.tiers.size > 0 && !filters.tiers.has(u.cluster_tier)) return false;
    if (filters.sectors.size > 0 && !filters.sectors.has(u.Sector)) return false;
    return true;
  });
}

const PC_INDEX: Record<string, number> = { PC1: 0, PC2: 1, PC3: 2, PC4: 3 };

function trajectoryColor(
  prevCluster: number | null | undefined,
  nextCluster: number | null | undefined,
  meta: Meta,
): string {
  if (prevCluster == null || nextCluster == null) return "#6c757d";
  const tierLabels = meta.clustering.tier_labels;
  const order = meta.tier_order;
  const prevTier = tierLabels[prevCluster];
  const nextTier = tierLabels[nextCluster];
  if (!prevTier || !nextTier) return "#6c757d";
  const prevRank = order.indexOf(prevTier);
  const nextRank = order.indexOf(nextTier);
  if (nextRank > prevRank) return "#b3001b";
  if (nextRank < prevRank) return "#2c7a4b";
  return "#6c757d";
}


/**
 * 2D PC scatter, fully reactive to ``selectedTickers`` and the shared filter
 * bar. Trajectory polylines overlay for any selected portfolio holding when
 * ``filters.showTrajectories`` is on.
 */
export function InteractivePCAChart({
  universe, meta, clusterMeta, pcX, pcY, selectedTickers,
  filters, trajectory, portfolioTickers,
  height = 520, showCentroids = true, title,
}: Props2D) {
  const traces = useMemo(() => {
    const tierColors = meta.tier_colors;
    const tierLabels = meta.clustering.tier_labels;

    // Selected tickers ALWAYS render (even if filtered out elsewhere) so users
    // never lose track of the highlighted name. Non-selected rows respect filters.
    const selectedRows = universe.filter((u) => selectedTickers.has(u.Ticker));
    const filteredOthers = applyFilters(
      universe.filter((u) => !selectedTickers.has(u.Ticker)),
      filters, portfolioTickers,
    );

    const buckets = new Map<number, UniverseRow[]>();
    for (const u of filteredOthers) {
      const cid = u.cluster;
      if (!buckets.has(cid)) buckets.set(cid, []);
      buckets.get(cid)!.push(u);
    }

    const out: any[] = [];
    for (const cid of Array.from(buckets.keys()).sort()) {
      const rows = buckets.get(cid)!;
      const tier = tierLabels[cid] ?? "?";
      out.push({
        type: "scatter",
        mode: "markers",
        x: rows.map((r) => r[pcX] ?? null),
        y: rows.map((r) => r[pcY] ?? null),
        name: `C${cid} · ${tier} (n=${rows.length})`,
        marker: { color: tierColors[tier], size: 6, opacity: 0.55 },
        text: rows.map((r) =>
          `<b>${r.Ticker}</b><br>${r.Company ?? ""}<br>` +
          `${r.Sector} · ${r.cluster_tier}<br>` +
          `${pcX}=${r[pcX]?.toFixed(2)} ${pcY}=${r[pcY]?.toFixed(2)}<br>` +
          `score ${r.composite_score?.toFixed(1)}`),
        hovertemplate: "%{text}<extra></extra>",
        showlegend: true,
      });
    }

    if (showCentroids) {
      out.push({
        type: "scatter",
        mode: "markers",
        x: clusterMeta.centroids.map((c) => c[PC_INDEX[pcX]]),
        y: clusterMeta.centroids.map((c) => c[PC_INDEX[pcY]]),
        name: "Centroids",
        marker: { symbol: "x", size: 14, color: "black",
                  line: { color: "white", width: 1 } },
        hoverinfo: "skip",
        showlegend: true,
      });
    }

    // Trajectory polylines for selected portfolio holdings
    if (filters.showTrajectories && trajectory) {
      for (const tk of selectedTickers) {
        const path = trajectory.paths[tk];
        if (!path) continue;
        const coords = path.coords.filter(
          (c) => c[pcX] != null && c[pcY] != null,
        );
        if (coords.length < 2) continue;
        for (let i = 0; i < coords.length - 1; i++) {
          const c0 = coords[i];
          const c1 = coords[i + 1];
          out.push({
            type: "scatter",
            mode: "lines",
            x: [c0[pcX], c1[pcX]],
            y: [c0[pcY], c1[pcY]],
            line: { color: trajectoryColor(c0.cluster, c1.cluster, meta), width: 2 },
            opacity: 0.85,
            showlegend: false,
            hoverinfo: "skip",
          });
        }
        // Quarter-end markers along the path
        out.push({
          type: "scatter",
          mode: "markers",
          x: coords.map((c) => c[pcX]),
          y: coords.map((c) => c[pcY]),
          marker: { size: 6, color: "white", line: { color: "black", width: 1 } },
          name: `${tk} path`,
          hovertext: coords.map((c) => `${tk} @ ${c.date?.slice(0, 10) ?? ""}`),
          hovertemplate: "%{hovertext}<extra></extra>",
          showlegend: false,
        });
      }
    }

    // Selected dots — drawn last so they sit on top
    if (selectedRows.length > 0) {
      out.push({
        type: "scatter",
        mode: "markers+text",
        x: selectedRows.map((r) => r[pcX] ?? null),
        y: selectedRows.map((r) => r[pcY] ?? null),
        name: `Selected (${selectedRows.length})`,
        marker: {
          size: 16,
          color: selectedRows.map((r) => meta.tier_colors[r.cluster_tier] ?? "#666"),
          line: { color: "#0a0a0a", width: 2.5 },
          opacity: 1.0,
        },
        text: selectedRows.map((r) => r.Ticker),
        textposition: "top center",
        textfont: { size: 11, color: "#0a0a0a" },
        hovertext: selectedRows.map((r) =>
          `<b>${r.Ticker}</b> · ${r.Company ?? ""}<br>` +
          `${r.Sector} · ${r.cluster_tier}<br>` +
          `${pcX}=${r[pcX]?.toFixed(2)} ${pcY}=${r[pcY]?.toFixed(2)}<br>` +
          `score ${r.composite_score?.toFixed(1)}`),
        hovertemplate: "%{hovertext}<extra></extra>",
      });
    }

    return out;
  }, [universe, selectedTickers, meta, clusterMeta, pcX, pcY,
      showCentroids, filters, trajectory, portfolioTickers]);

  const layout: any = useMemo(() => ({
    autosize: true,
    height,
    margin: { l: 55, r: 20, t: title ? 36 : 14, b: 50 },
    template: "plotly_white",
    title: title ? { text: title, font: { size: 14 }, x: 0.02 } : undefined,
    xaxis: { title: { text: `${pcX}: ${meta.pca.pc_labels[pcX] ?? ""}` }, zeroline: true },
    yaxis: { title: { text: `${pcY}: ${meta.pca.pc_labels[pcY] ?? ""}` }, zeroline: true },
    hovermode: "closest",
    legend: {
      orientation: "v", x: 1.02, y: 1, bgcolor: "rgba(255,255,255,0.85)",
      font: { size: 10 },
    },
  }), [pcX, pcY, meta, height, title]);

  return (
    <Plot data={traces} layout={layout}
          config={{ displayModeBar: true, responsive: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ["lasso2d", "select2d"] }}
          useResizeHandler
          style={{ width: "100%", height }} />
  );
}


/**
 * 3D variant.
 */
export function InteractivePCAChart3D({
  universe, meta, clusterMeta, selectedTickers, filters, trajectory,
  portfolioTickers, height = 700, title,
}: BaseProps) {
  const traces = useMemo(() => {
    const tierColors = meta.tier_colors;
    const tierLabels = meta.clustering.tier_labels;

    const selectedRows = universe.filter((u) => selectedTickers.has(u.Ticker));
    const filteredOthers = applyFilters(
      universe.filter((u) => !selectedTickers.has(u.Ticker)),
      filters, portfolioTickers,
    );

    const buckets = new Map<number, UniverseRow[]>();
    for (const u of filteredOthers) {
      const cid = u.cluster;
      if (!buckets.has(cid)) buckets.set(cid, []);
      buckets.get(cid)!.push(u);
    }

    const out: any[] = [];
    for (const cid of Array.from(buckets.keys()).sort()) {
      const rows = buckets.get(cid)!;
      const tier = tierLabels[cid] ?? "?";
      out.push({
        type: "scatter3d",
        mode: "markers",
        x: rows.map((r) => r.PC1),
        y: rows.map((r) => r.PC2),
        z: rows.map((r) => r.PC3),
        name: `C${cid} · ${tier} (n=${rows.length})`,
        marker: { color: tierColors[tier], size: 3, opacity: 0.5 },
        text: rows.map((r) => `<b>${r.Ticker}</b> · ${r.Sector}<br>${r.cluster_tier}`),
        hovertemplate: "%{text}<extra></extra>",
      });
    }

    out.push({
      type: "scatter3d",
      mode: "markers",
      x: clusterMeta.centroids.map((c) => c[0]),
      y: clusterMeta.centroids.map((c) => c[1]),
      z: clusterMeta.centroids.map((c) => c[2]),
      name: "Centroids",
      marker: { symbol: "x", size: 6, color: "black", line: { color: "white", width: 1 } },
      hoverinfo: "skip",
    });

    if (filters.showTrajectories && trajectory) {
      for (const tk of selectedTickers) {
        const path = trajectory.paths[tk];
        if (!path) continue;
        const coords = path.coords.filter(
          (c) => c.PC1 != null && c.PC2 != null && c.PC3 != null,
        );
        if (coords.length < 2) continue;
        for (let i = 0; i < coords.length - 1; i++) {
          const c0 = coords[i];
          const c1 = coords[i + 1];
          out.push({
            type: "scatter3d",
            mode: "lines",
            x: [c0.PC1, c1.PC1], y: [c0.PC2, c1.PC2], z: [c0.PC3, c1.PC3],
            line: { color: trajectoryColor(c0.cluster, c1.cluster, meta), width: 4 },
            showlegend: false,
            hoverinfo: "skip",
          });
        }
      }
    }

    if (selectedRows.length > 0) {
      out.push({
        type: "scatter3d",
        mode: "markers+text",
        x: selectedRows.map((r) => r.PC1),
        y: selectedRows.map((r) => r.PC2),
        z: selectedRows.map((r) => r.PC3),
        name: `Selected (${selectedRows.length})`,
        marker: {
          size: 9,
          color: selectedRows.map((r) => meta.tier_colors[r.cluster_tier] ?? "#666"),
          line: { color: "#0a0a0a", width: 3 }, opacity: 1.0,
        },
        text: selectedRows.map((r) => r.Ticker),
        textposition: "top center",
        textfont: { size: 10 },
        hovertext: selectedRows.map((r) =>
          `<b>${r.Ticker}</b> · ${r.Sector}<br>${r.cluster_tier}<br>` +
          `score ${r.composite_score?.toFixed(1)}`),
        hovertemplate: "%{hovertext}<extra></extra>",
      });
    }
    return out;
  }, [universe, selectedTickers, meta, clusterMeta, filters, trajectory, portfolioTickers]);

  const layout: any = {
    autosize: true,
    height,
    margin: { l: 0, r: 0, t: title ? 36 : 0, b: 0 },
    template: "plotly_white",
    title: title ? { text: title, font: { size: 14 }, x: 0.02 } : undefined,
    scene: {
      xaxis: { title: { text: `PC1: ${meta.pca.pc_labels.PC1 ?? ""}` } },
      yaxis: { title: { text: `PC2: ${meta.pca.pc_labels.PC2 ?? ""}` } },
      zaxis: { title: { text: `PC3: ${meta.pca.pc_labels.PC3 ?? ""}` } },
      aspectmode: "cube",
    },
    legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.85)", font: { size: 10 } },
  };

  return (
    <Plot data={traces} layout={layout}
          config={{ displayModeBar: true, responsive: true, displaylogo: false }}
          useResizeHandler style={{ width: "100%", height }} />
  );
}
