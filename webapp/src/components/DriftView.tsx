import { DriftRow, Meta } from "../lib/types";
import { Column, DataTable } from "./DataTable";
import { fmt } from "../lib/data";
import { TickerLink } from "../lib/tickerContext";

export function DriftView({ drift, meta }: { drift: DriftRow[]; meta: Meta }) {
  const styleChip = (style: string) => (
    <span className="style-chip"
          style={{ background: meta.style_colors?.[style] ?? "#666" }}>
      {style}
    </span>
  );

  const cols: Column<DriftRow>[] = [
    { key: "Ticker", header: "Ticker", accessor: (r) => r.Ticker,
      render: (r) => (
        <span>
          <TickerLink ticker={r.Ticker} />
          {r.is_portfolio && <span style={{ marginLeft: 6, fontSize: 10,
            background: "#1f3b73", color: "#fff",
            padding: "1px 6px", borderRadius: 3 }}>IMA</span>}
        </span>
      ) },
    { key: "assigned_style", header: "Current cluster", accessor: (r) => r.assigned_style,
      render: (r) => styleChip(r.assigned_style) },
    { key: "nearest_other_style", header: "Nearest other", accessor: (r) => r.nearest_other_style,
      render: (r) => styleChip(r.nearest_other_style) },
    { key: "boundary_gap", header: "Boundary gap", numeric: true,
      accessor: (r) => r.boundary_gap, render: (r) => fmt(r.boundary_gap, 2) },
    { key: "two_quarter_drift", header: "2Q drift", numeric: true,
      accessor: (r) => r.two_quarter_drift,
      render: (r) => fmt(r.two_quarter_drift, 2) },
    { key: "flags", header: "Flags",
      accessor: (r) => (r.is_borderline ? 1 : 0) + (r.crossed_cluster_last_q ? 2 : 0) + (r.large_2q_drift ? 4 : 0),
      render: (r) => (
        <span>
          {r.is_borderline && <span style={badge}>borderline</span>}
          {r.crossed_cluster_last_q && <span style={badgeAlert}>crossed last Q</span>}
          {r.large_2q_drift && <span style={badgeWarn}>large 2Q drift</span>}
        </span>
      ) },
  ];

  const portfolioOnly = drift.filter((d) => d.is_portfolio);

  return (
    <div>
      <h2 className="section-title">Drift alerts</h2>
      <p className="section-lede">
        An attention list, not a signal: stocks whose statistical profile is on
        the move — sitting close to a cluster boundary, having crossed into a new
        cluster last quarter, or having moved more than 1.5σ through risk space
        over two quarters. Borderline names can also flip between runs simply
        because cluster boundaries are soft. Portfolio names are pinned to the top;
        click a ticker to see what changed.
      </p>

      <div className="card">
        <h3>IMA portfolio ({portfolioOnly.length})</h3>
        {portfolioOnly.length === 0 ? (
          <p className="muted">No IMA holding is flagged.</p>
        ) : (
          <DataTable rows={portfolioOnly} columns={cols} initialSortKey="boundary_gap"
                     rowClassName={() => "highlight"} />
        )}
      </div>

      <div className="card">
        <h3>Universe ({drift.length})</h3>
        <DataTable rows={drift} columns={cols} initialSortKey="boundary_gap" pageSize={25} />
      </div>
    </div>
  );
}

const badge: React.CSSProperties = {
  display: "inline-block", padding: "1px 8px", background: "#e4e7eb",
  color: "#1a202c", borderRadius: 10, fontSize: 11, marginRight: 4, fontWeight: 600,
};
const badgeAlert: React.CSSProperties = { ...badge, background: "#b3001b", color: "#fff" };
const badgeWarn: React.CSSProperties = { ...badge, background: "#f0a202", color: "#1a202c" };
