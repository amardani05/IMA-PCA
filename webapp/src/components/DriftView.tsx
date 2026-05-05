import { DriftRow } from "../lib/types";
import { Column, DataTable } from "./DataTable";
import { fmt, tierClass } from "../lib/data";

export function DriftView({ drift }: { drift: DriftRow[] }) {
  const cols: Column<DriftRow>[] = [
    { key: "Ticker", header: "Ticker", accessor: (r) => r.Ticker,
      render: (r) => (
        <span>
          <strong>{r.Ticker}</strong>
          {r.is_portfolio && <span style={{ marginLeft: 6, fontSize: 10,
            background: "#1f3b73", color: "#fff",
            padding: "1px 6px", borderRadius: 3 }}>IMA</span>}
        </span>
      ) },
    { key: "assigned_tier", header: "Current tier", accessor: (r) => r.assigned_tier,
      render: (r) => <span className={tierClass(r.assigned_tier)}>{r.assigned_tier}</span> },
    { key: "nearest_other_tier", header: "Nearest other", accessor: (r) => r.nearest_other_tier,
      render: (r) => <span className={tierClass(r.nearest_other_tier)}>{r.nearest_other_tier}</span> },
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
        Stocks that either sit close to a cluster boundary, crossed into a new cluster last quarter,
        or moved more than 1.5σ through PC space over the last two quarters. Portfolio names are
        pinned to the top.
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
