import { useMemo, useState } from "react";
import { Meta, UniverseRow } from "../lib/types";
import { Column, DataTable } from "./DataTable";
import { fmt, tierClass } from "../lib/data";
import { featureDefinition, featureLabel } from "../lib/glossary";
import { sectorPercentile } from "../lib/assess";
import { TickerLink } from "../lib/tickerContext";

interface Props {
  meta: Meta;
  universe: UniverseRow[];
}

export function UniverseView({ meta, universe }: Props) {
  const [filter, setFilter] = useState("");
  const [tierFilter, setTierFilter] = useState<string>("");
  const [onlyPortfolio, setOnlyPortfolio] = useState(false);

  const sectors = Array.from(new Set(universe.map((u) => u.Sector))).sort();
  const [sectorFilter, setSectorFilter] = useState<string>("");

  // Within-sector risk percentile, computed once
  const secPct = useMemo(() => {
    const m = new Map<string, number | null>();
    for (const u of universe) m.set(u.Ticker, sectorPercentile(universe, u));
    return m;
  }, [universe]);

  const featCol = (key: string, digits = 2, pct = false): Column<UniverseRow> => ({
    key,
    header: featureLabel(key),
    numeric: true,
    accessor: (r) => r[key],
    render: (r) => r[key] != null
      ? (pct ? `${((r[key] as number) * 100).toFixed(1)}%` : fmt(r[key], digits))
      : "—",
  });

  const cols: Column<UniverseRow>[] = [
    { key: "Ticker", header: "Ticker", accessor: (r) => r.Ticker,
      render: (r) => (
        <span>
          <TickerLink ticker={r.Ticker} />
          {r.is_portfolio && <span style={{ marginLeft: 6, fontSize: 10,
                                             background: "#1f3b73", color: "#fff",
                                             padding: "1px 6px", borderRadius: 3 }}>IMA</span>}
        </span>
      ) },
    { key: "Sector", header: "Sector", accessor: (r) => r.Sector },
    { key: "score_percentile", header: "Risk %ile", numeric: true,
      accessor: (r) => r.score_percentile,
      render: (r) => fmt(r.score_percentile, 0),
      defaultSortDesc: true },
    { key: "risk_tier", header: "Tier", accessor: (r) => r.risk_tier,
      render: (r) => <span className={tierClass(r.risk_tier)}>{r.risk_tier}</span> },
    { key: "sector_pct", header: "Sector %ile", numeric: true,
      accessor: (r) => secPct.get(r.Ticker),
      render: (r) => {
        const v = secPct.get(r.Ticker);
        return v != null ? fmt(v, 0) : "—";
      } },
    { key: "cluster_style", header: "Style", accessor: (r) => r.cluster_style,
      render: (r) => (
        <span className="style-chip"
              style={{ background: meta.style_colors?.[r.cluster_style] ?? "#666" }}>
          {r.cluster_style}
        </span>
      ) },
    featCol("altman_z"),
    featCol("asset_growth_yoy", 1, true),
    featCol("short_pct_float", 1),
    featCol("momentum_90d", 1, true),
    featCol("volatility_60d", 1, true),
    featCol("net_debt_to_ebitda"),
  ];

  const filterFn = (r: UniverseRow, t: string): boolean => {
    if (tierFilter && r.risk_tier !== tierFilter) return false;
    if (sectorFilter && r.Sector !== sectorFilter) return false;
    if (onlyPortfolio && !r.is_portfolio) return false;
    if (!t) return true;
    return (
      r.Ticker.toLowerCase().includes(t) ||
      (r.Company ?? "").toLowerCase().includes(t) ||
      (r.Sector ?? "").toLowerCase().includes(t)
    );
  };

  return (
    <div>
      <h2 className="section-title">Universe Explorer</h2>
      <p className="section-lede">
        All {universe.length} S&amp;P 600 stocks. <strong>Risk %ile</strong> ranks aggregate
        statistical risk against today's universe (100 = riskiest); <strong>Sector %ile</strong>{" "}
        re-ranks within the stock's own sector — the fairer comparison, since scores are not
        sector-adjusted. Click a <span style={{ color: "var(--accent)", fontWeight: 700 }}>ticker</span>{" "}
        for the single-name view, hover a column header for its definition. Descriptive, not
        predictive — none of this is a forecast.
      </p>

      <div className="card">
        <div className="filter-bar">
          <input
            type="search"
            placeholder="Search ticker / company / sector…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ flex: "1 1 240px" }}
          />
          <select value={tierFilter} onChange={(e) => setTierFilter(e.target.value)}>
            <option value="">All tiers</option>
            {meta.tier_order.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
          <select value={sectorFilter} onChange={(e) => setSectorFilter(e.target.value)}>
            <option value="">All sectors</option>
            {sectors.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <label>
            <input type="checkbox" checked={onlyPortfolio}
                   onChange={(e) => setOnlyPortfolio(e.target.checked)} />
            IMA only
          </label>
        </div>

        <DataTableWithTooltips
          rows={universe}
          columns={cols}
          filterText={filter}
          filterFn={filterFn}
        />
      </div>
    </div>
  );
}

/** Wraps DataTable, adding title-attribute tooltips on feature headers. */
function DataTableWithTooltips(props: {
  rows: UniverseRow[];
  columns: Column<UniverseRow>[];
  filterText: string;
  filterFn: (r: UniverseRow, t: string) => boolean;
}) {
  const columns = props.columns.map((c) => {
    const def = featureDefinition(c.key);
    if (def === c.key) return c; // not a model feature
    return { ...c, header: c.header };
  });
  return (
    <DataTable
      rows={props.rows}
      columns={columns}
      initialSortKey="score_percentile"
      pageSize={30}
      filterText={props.filterText}
      filterFn={props.filterFn}
      rowClassName={(r) => (r.is_portfolio ? "highlight" : undefined)}
      headerTitle={(key) => {
        const def = featureDefinition(key);
        if (def !== key) return def;
        if (key === "score_percentile") return "Percentile of the composite risk score within the whole universe. 100 = riskiest. Tiers: bottom 20% Low Risk, middle 60% In Line, top 20% Elevated.";
        if (key === "sector_pct") return "Percentile of the composite risk score within the stock's own sector — corrects for sector-level differences the model doesn't adjust for.";
        if (key === "risk_tier") return "Calibrated 20/60/20 buckets of the risk percentile. Relative to today's universe — not an absolute rating.";
        if (key === "cluster_style") return "Descriptive style grouping from clustering (what the stock statistically looks like). NOT a risk ranking.";
        return undefined;
      }}
    />
  );
}
