import { useMemo } from "react";
import { ChartFilters } from "./InteractivePCAChart";
import { Meta, UniverseRow } from "../lib/types";

interface Props {
  filters: ChartFilters;
  setFilters: (f: ChartFilters) => void;
  meta: Meta;
  universe: UniverseRow[];
}

/**
 * Shared filter bar for the PCA explorer charts.
 * Selected tickers in the tree always render — these filters control which
 * non-selected dots appear and whether trajectory polylines are drawn.
 */
export function ChartFilterBar({ filters, setFilters, meta, universe }: Props) {
  const sectors = useMemo(
    () => Array.from(new Set(universe.map((u) => u.Sector))).sort(),
    [universe],
  );

  const toggleTier = (tier: string) => {
    const next = new Set(filters.tiers);
    if (next.has(tier)) next.delete(tier);
    else next.add(tier);
    setFilters({ ...filters, tiers: next });
  };

  const toggleSector = (sector: string) => {
    const next = new Set(filters.sectors);
    if (next.has(sector)) next.delete(sector);
    else next.add(sector);
    setFilters({ ...filters, sectors: next });
  };

  const tierActive = (t: string) => filters.tiers.size === 0 || filters.tiers.has(t);

  return (
    <div style={{
      background: "#fff",
      border: "1px solid var(--border)",
      borderRadius: 8,
      padding: "10px 14px",
      marginBottom: 12,
      display: "flex", flexWrap: "wrap", alignItems: "center", gap: 14,
    }}>
      <label style={pill}>
        <input type="checkbox"
               checked={filters.portfolioOnly}
               onChange={(e) => setFilters({ ...filters, portfolioOnly: e.target.checked })} />
        <span>Portfolio only</span>
      </label>

      <label style={pill}>
        <input type="checkbox"
               checked={filters.showTrajectories}
               onChange={(e) => setFilters({ ...filters, showTrajectories: e.target.checked })} />
        <span>Show trajectories</span>
      </label>

      <div style={{ borderLeft: "1px solid var(--border)", height: 22, margin: "0 4px" }} />

      <span style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase",
                     letterSpacing: 0.5, fontWeight: 600 }}
            data-hint
            title="Descriptive style clusters — statistical groupings, not risk ratings.">
        Styles
      </span>
      {meta.style_order.map((t) => (
        <button key={t} onClick={() => toggleTier(t)} style={{
          ...chip,
          borderColor: tierActive(t) ? meta.style_colors[t] : "var(--border)",
          background: filters.tiers.has(t) ? meta.style_colors[t] : "#fff",
          color: filters.tiers.has(t) ? "#fff" : "var(--text)",
          opacity: filters.tiers.size === 0 || filters.tiers.has(t) ? 1 : 0.4,
        }}>
          <span style={{
            display: "inline-block", width: 8, height: 8, borderRadius: 4,
            background: meta.style_colors[t], marginRight: 6,
          }} />
          {t}
        </button>
      ))}

      {filters.tiers.size > 0 && (
        <button onClick={() => setFilters({ ...filters, tiers: new Set() })}
                style={clearChip}>clear</button>
      )}

      <div style={{ borderLeft: "1px solid var(--border)", height: 22, margin: "0 4px" }} />

      <details style={{ display: "inline-block" }}>
        <summary style={{ ...chip, cursor: "pointer", listStyle: "none" }}>
          Sectors
          {filters.sectors.size > 0 && (
            <span style={{ marginLeft: 4, fontWeight: 600 }}>· {filters.sectors.size}</span>
          )}
        </summary>
        <div style={{
          position: "absolute", zIndex: 10, marginTop: 4,
          background: "#fff", border: "1px solid var(--border)", borderRadius: 6,
          padding: 8, boxShadow: "0 4px 8px rgba(0,0,0,0.08)",
          maxHeight: 320, overflowY: "auto", minWidth: 260,
        }}>
          {sectors.map((s) => (
            <label key={s} style={{
              display: "flex", alignItems: "center", padding: "3px 4px",
              fontSize: 12, cursor: "pointer", borderRadius: 3,
            }}>
              <input type="checkbox" checked={filters.sectors.has(s)}
                     onChange={() => toggleSector(s)}
                     style={{ marginRight: 6 }} />
              {s}
            </label>
          ))}
          {filters.sectors.size > 0 && (
            <button onClick={() => setFilters({ ...filters, sectors: new Set() })}
                    style={{ ...clearChip, marginTop: 6 }}>clear all</button>
          )}
        </div>
      </details>
    </div>
  );
}

const pill: React.CSSProperties = {
  display: "flex", alignItems: "center", gap: 6,
  fontSize: 13, cursor: "pointer", userSelect: "none",
};

const chip: React.CSSProperties = {
  display: "inline-flex", alignItems: "center",
  padding: "4px 10px", fontSize: 12, fontWeight: 500,
  border: "1px solid var(--border)", borderRadius: 14,
  background: "#fff", cursor: "pointer", userSelect: "none",
};

const clearChip: React.CSSProperties = {
  ...chip, fontSize: 11, padding: "2px 8px",
  background: "#f0f2f6", color: "var(--muted)",
};
