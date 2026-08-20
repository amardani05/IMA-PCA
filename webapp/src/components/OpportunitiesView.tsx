import { OpportunityRow } from "../lib/types";
import { Column, DataTable } from "./DataTable";
import { fmt, fmtPct, tierClass } from "../lib/data";
import { featureDefinition } from "../lib/glossary";
import { TickerLink } from "../lib/tickerContext";

export function OpportunitiesView({ opportunities }: { opportunities: OpportunityRow[] }) {
  const cols: Column<OpportunityRow>[] = [
    { key: "Ticker", header: "Ticker", accessor: (r) => r.Ticker,
      render: (r) => <TickerLink ticker={r.Ticker} /> },
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
        Contrarian long candidates: names where the balance sheet looks intact
        (Altman Z &gt; 2, operating cash flow &gt; 0.7× net income) but market
        positioning is bearish (short interest &gt; 8% of float, negative 90-day
        momentum). <strong>Treat this as an unvalidated hypothesis, not a buy
        list</strong> — the same profile also describes classic value traps, and
        nothing here distinguishes the two. Start diligence with the question
        "what do the shorts know?"
      </p>

      <div className="card">
        {opportunities.length === 0 ? (
          <p className="muted">No candidates matched the opportunity criteria in the current run.</p>
        ) : (
          <DataTable rows={opportunities} columns={cols} initialSortKey="contrarian_score"
                     headerTitle={(k) => {
                       const d = featureDefinition(k);
                       if (d !== k) return d;
                       if (k === "contrarian_score")
                         return "Ranking of how bearish the positioning is (short-interest rank minus momentum rank). Higher = more hated by the market.";
                       return undefined;
                     }} />
        )}
      </div>
    </div>
  );
}
