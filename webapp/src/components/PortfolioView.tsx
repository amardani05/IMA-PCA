import { Meta, PortfolioRow } from "../lib/types";
import { Column, DataTable } from "./DataTable";
import { fmt, fmtPct, tierClass } from "../lib/data";
import { featureDefinition, humanizeDriverString } from "../lib/glossary";
import { TickerLink } from "../lib/tickerContext";

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
      render: (r) => <TickerLink ticker={r.Ticker} /> },
    { key: "Weight", header: "Weight", numeric: true,
      accessor: (r) => r.Weight,
      render: (r) => `${(r.Weight * 100).toFixed(2)}%` },
    { key: "Sector", header: "Sector", accessor: (r) => r.Sector ?? "" },
    { key: "Score_Percentile", header: "Risk %ile", numeric: true,
      accessor: (r) => r.Score_Percentile,
      render: (r) => fmt(r.Score_Percentile, 0),
      defaultSortDesc: true },
    { key: "Risk_Tier", header: "Tier", accessor: (r) => r.Risk_Tier,
      render: (r) => r.Risk_Tier
        ? <span className={tierClass(r.Risk_Tier)}>{r.Risk_Tier}</span>
        : "—" },
    { key: "Cluster_Label", header: "Style", accessor: (r) => r.Cluster_Label,
      render: (r) => r.Cluster !== undefined && r.Cluster !== -1 && r.Cluster_Label
        ? <span className="style-chip"
                style={{ background: meta.style_colors?.[r.Cluster_Label] ?? "#666" }}>
            {r.Cluster_Label}
          </span>
        : "—" },
    { key: "Altman_Z", header: "Altman Z", numeric: true,
      accessor: (r) => r.Altman_Z,
      render: (r) => fmt(r.Altman_Z, 2) },
    { key: "Short_Pct_Float", header: "Short %", numeric: true,
      accessor: (r) => r.Short_Pct_Float,
      render: (r) => fmt(r.Short_Pct_Float, 1) },
    { key: "Momentum_90d", header: "Mom 90d", numeric: true,
      accessor: (r) => r.Momentum_90d,
      render: (r) => fmtPct(r.Momentum_90d) },
    { key: "Net_Debt_EBITDA", header: "Net debt/EBITDA", numeric: true,
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
      render: (r) => <small>{humanizeDriverString(r.Top_Risk_Drivers) || "—"}</small> },
    { key: "Sector_Comparison", header: "vs sector", accessor: (r) => r.Sector_Comparison ?? "",
      render: (r) => r.Sector_Comparison ? (
        <small>
          {r.Sector_Comparison} {r.Sector_Delta != null && `(${r.Sector_Delta >= 0 ? "+" : ""}${fmt(r.Sector_Delta, 1)})`}
        </small>
      ) : "—" },
  ];

  const headerTitle = (key: string): string | undefined => {
    switch (key) {
      case "Score_Percentile":
        return "Percentile of the composite risk score within the whole S&P 600 universe (100 = riskiest).";
      case "Risk_Tier":
        return "Calibrated 20/60/20 buckets of the risk percentile — relative to today's universe, not an absolute rating.";
      case "Cluster_Label":
        return "Descriptive style grouping from clustering — what the stock statistically looks like, not a risk rating.";
      case "Trajectory":
        return "Direction of movement through risk space over the last four quarterly snapshots.";
      case "Altman_Z": return featureDefinition("altman_z");
      case "Short_Pct_Float": return featureDefinition("short_pct_float");
      case "Momentum_90d": return featureDefinition("momentum_90d");
      case "Net_Debt_EBITDA": return featureDefinition("net_debt_to_ebitda");
      case "Volatility_60d": return featureDefinition("volatility_60d");
      case "Top_Risk_Drivers":
        return "The three features where this stock ranks riskiest vs the universe (percentile in parentheses).";
      default: return undefined;
    }
  };

  const elevated = scored.filter((p) => p.Risk_Tier === "Elevated");

  return (
    <div>
      <h2 className="section-title">IMA Portfolio</h2>
      <p className="section-lede">
        Statistical risk detail for every IMA holding, ranked against the S&amp;P 600.
        Click a ticker for the single-name view. These are descriptive
        characteristics, not predictions — use them to know what to defend, not
        as a sell signal.
      </p>

      <div className="grid grid-4">
        <div className="card stat">
          <div className="stat-label">Positions</div>
          <div className="stat-value">{portfolio.length}</div>
          <small className="muted">{scored.length} scored</small>
        </div>
        <div className="card stat">
          <div className="stat-label">Weighted risk score</div>
          <div className="stat-value">{fmt(weightedScore, 1)}</div>
          <small className="muted">universe average ≈ 50</small>
        </div>
        <div className="card stat">
          <div className="stat-label">Elevated tier</div>
          <div className="stat-value">{elevated.length}</div>
          <small className="muted">
            {elevated.length > 0
              ? elevated.map((p) => p.Ticker).join(", ")
              : "no holdings in the top risk quintile"}
          </small>
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
          initialSortKey="Score_Percentile"
          headerTitle={headerTitle}
          rowClassName={(r) => (r.Risk_Tier === "Elevated" ? "highlight" : undefined)}
        />
      </div>
    </div>
  );
}
