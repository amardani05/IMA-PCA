import { Meta, PortfolioRow } from "../lib/types";
import { Column, DataTable } from "./DataTable";
import { fmt, fmtPct, tierClass } from "../lib/data";

interface Props {
  meta: Meta;
  portfolio: PortfolioRow[];
}

export function PortfolioView({ meta, portfolio }: Props) {
  const scored = portfolio.filter((p) => p.Cluster !== undefined && p.Cluster !== -1);
  const totalWeight = scored.reduce((s, p) => s + p.Weight, 0);
  const weightedScore = totalWeight > 0
    ? scored.reduce((s, p) => s + p.Weight * (p.Composite_Score ?? 0), 0) / totalWeight
    : 0;

  const cols: Column<PortfolioRow>[] = [
    { key: "Ticker", header: "Ticker", accessor: (r) => r.Ticker,
      render: (r) => <strong>{r.Ticker}</strong> },
    { key: "Weight", header: "Weight", numeric: true,
      accessor: (r) => r.Weight,
      render: (r) => `${(r.Weight * 100).toFixed(2)}%` },
    { key: "Sector", header: "Sector", accessor: (r) => r.Sector ?? "" },
    { key: "Composite_Score", header: "Score", numeric: true,
      accessor: (r) => r.Composite_Score,
      render: (r) => fmt(r.Composite_Score, 1),
      defaultSortDesc: true },
    { key: "Risk_Tier", header: "Tier", accessor: (r) => r.Risk_Tier,
      render: (r) => r.Risk_Tier
        ? <span className={tierClass(r.Risk_Tier)}>{r.Risk_Tier}</span>
        : "—" },
    { key: "Cluster_Label", header: "Cluster", accessor: (r) => r.Cluster_Label,
      render: (r) => r.Cluster !== undefined && r.Cluster !== -1
        ? `${r.Cluster} · ${r.Cluster_Label}` : "—" },
    { key: "Altman_Z", header: "Altman Z", numeric: true,
      accessor: (r) => r.Altman_Z,
      render: (r) => fmt(r.Altman_Z, 2) },
    { key: "Piotroski_F", header: "Piotroski F", numeric: true,
      accessor: (r) => r.Piotroski_F,
      render: (r) => fmt(r.Piotroski_F, 1) },
    { key: "Short_Pct_Float", header: "Short %", numeric: true,
      accessor: (r) => r.Short_Pct_Float,
      render: (r) => fmt(r.Short_Pct_Float, 1) },
    { key: "Momentum_90d", header: "Mom 90d", numeric: true,
      accessor: (r) => r.Momentum_90d,
      render: (r) => fmtPct(r.Momentum_90d) },
    { key: "Net_Debt_EBITDA", header: "ND/EBITDA", numeric: true,
      accessor: (r) => r.Net_Debt_EBITDA,
      render: (r) => fmt(r.Net_Debt_EBITDA, 2) },
    { key: "Volatility_60d", header: "Vol 60d", numeric: true,
      accessor: (r) => r.Volatility_60d,
      render: (r) => fmtPct(r.Volatility_60d) },
    { key: "Trajectory", header: "Trajectory", accessor: (r) => r.Trajectory,
      render: (r) => {
        const t = r.Trajectory;
        if (!t || t === "N/A" || t === "Unknown") return "—";
        const color = t === "Deteriorating" ? "var(--danger)"
                    : t === "Improving" ? "var(--ok)" : "var(--muted)";
        return <span style={{ color, fontWeight: 600 }}>{t}</span>;
      } },
    { key: "Top_Risk_Drivers", header: "Top risk drivers", accessor: (r) => r.Top_Risk_Drivers ?? "",
      render: (r) => <small>{r.Top_Risk_Drivers ?? "—"}</small> },
    { key: "Sector_Comparison", header: "vs sector", accessor: (r) => r.Sector_Comparison ?? "",
      render: (r) => r.Sector_Comparison ? (
        <small>
          {r.Sector_Comparison} {r.Sector_Delta != null && `(${r.Sector_Delta >= 0 ? "+" : ""}${fmt(r.Sector_Delta, 1)})`}
        </small>
      ) : "—" },
  ];

  return (
    <div>
      <h2 className="section-title">IMA Portfolio</h2>
      <p className="section-lede">
        Risk detail for every IMA holding. Weighted-average composite score is a quick-look summary
        of aggregate portfolio torpedo risk.
      </p>

      <div className="grid grid-4">
        <div className="card stat">
          <div className="stat-label">Positions</div>
          <div className="stat-value">{portfolio.length}</div>
          <small className="muted">{scored.length} scored</small>
        </div>
        <div className="card stat">
          <div className="stat-label">Weighted score</div>
          <div className="stat-value">{fmt(weightedScore, 1)}</div>
          <small className="muted">0 = safest · 100 = riskiest</small>
        </div>
        <div className="card stat">
          <div className="stat-label">High / Critical</div>
          <div className="stat-value">
            {scored.filter((p) => p.Risk_Tier === "High" || p.Risk_Tier === "Critical").length}
          </div>
          <small className="muted">positions</small>
        </div>
        <div className="card stat">
          <div className="stat-label">Deteriorating</div>
          <div className="stat-value">
            {scored.filter((p) => p.Trajectory === "Deteriorating").length}
          </div>
          <small className="muted">per trajectory classifier</small>
        </div>
      </div>

      <div className="card">
        <DataTable
          rows={portfolio}
          columns={cols}
          initialSortKey="Composite_Score"
          rowClassName={(r) =>
            r.Risk_Tier === "Critical" || r.Risk_Tier === "High" ? "highlight" : undefined
          }
        />
      </div>
    </div>
  );
}
