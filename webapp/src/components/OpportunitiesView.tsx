import { OpportunityRow } from "../lib/types";
import { Column, DataTable } from "./DataTable";
import { fmt, fmtPct, tierClass } from "../lib/data";

export function OpportunitiesView({ opportunities }: { opportunities: OpportunityRow[] }) {
  const cols: Column<OpportunityRow>[] = [
    { key: "Ticker", header: "Ticker", accessor: (r) => r.Ticker,
      render: (r) => <strong>{r.Ticker}</strong> },
    { key: "Company", header: "Company", accessor: (r) => r.Company },
    { key: "Sector", header: "Sector", accessor: (r) => r.Sector },
    { key: "altman_z", header: "Altman Z", numeric: true,
      accessor: (r) => r.altman_z, render: (r) => fmt(r.altman_z, 2) },
    { key: "accruals_ratio", header: "OCF/NI", numeric: true,
      accessor: (r) => r.accruals_ratio,
      render: (r) => fmt(r.accruals_ratio, 2) },
    { key: "short_pct_float", header: "Short %", numeric: true,
      accessor: (r) => r.short_pct_float, render: (r) => fmt(r.short_pct_float, 1),
      defaultSortDesc: true },
    { key: "momentum_90d", header: "Mom 90d", numeric: true,
      accessor: (r) => r.momentum_90d, render: (r) => fmtPct(r.momentum_90d) },
    { key: "composite_score", header: "Score", numeric: true,
      accessor: (r) => r.composite_score, render: (r) => fmt(r.composite_score, 1) },
    { key: "risk_tier", header: "Tier", accessor: (r) => r.risk_tier,
      render: (r) => <span className={tierClass(r.risk_tier)}>{r.risk_tier}</span> },
    { key: "contrarian_score", header: "Contrarian", numeric: true,
      accessor: (r) => r.contrarian_score, render: (r) => fmt(r.contrarian_score, 1),
      defaultSortDesc: true },
  ];

  return (
    <div>
      <h2 className="section-title">Opportunity screen</h2>
      <p className="section-lede">
        Contrarian watchlist: stocks with intact fundamentals (Altman Z &gt; 2, Piotroski F ≥ 5)
        where market positioning is bearish (short interest &gt; 8%, negative 90-day momentum).
        Candidates for closer diligence — the market hates them but the numbers don't agree.
      </p>

      <div className="card">
        {opportunities.length === 0 ? (
          <p className="muted">No candidates matched the opportunity criteria in the current run.</p>
        ) : (
          <DataTable rows={opportunities} columns={cols} initialSortKey="contrarian_score" />
        )}
      </div>
    </div>
  );
}
