import { useState } from "react";
import { Meta, UniverseRow } from "../lib/types";
import { Column, DataTable } from "./DataTable";
import { fmt, tierClass } from "../lib/data";

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

  const cols: Column<UniverseRow>[] = [
    { key: "Ticker", header: "Ticker", accessor: (r) => r.Ticker,
      render: (r) => (
        <span>
          <strong>{r.Ticker}</strong>
          {r.is_portfolio && <span style={{ marginLeft: 6, fontSize: 10,
                                             background: "#1f3b73", color: "#fff",
                                             padding: "1px 6px", borderRadius: 3 }}>IMA</span>}
        </span>
      ) },
    { key: "Sector", header: "Sector", accessor: (r) => r.Sector },
    { key: "composite_score", header: "Score", numeric: true,
      accessor: (r) => r.composite_score,
      render: (r) => fmt(r.composite_score, 1),
      defaultSortDesc: true },
    { key: "risk_tier", header: "Tier", accessor: (r) => r.risk_tier,
      render: (r) => <span className={tierClass(r.risk_tier)}>{r.risk_tier}</span> },
    { key: "cluster", header: "Cluster", numeric: true,
      accessor: (r) => r.cluster,
      render: (r) => `${r.cluster} · ${r.cluster_tier}` },
    { key: "altman_z", header: "Altman Z", numeric: true,
      accessor: (r) => r.altman_z,
      render: (r) => fmt(r.altman_z, 2) },
    { key: "asset_growth_yoy", header: "Asset growth", numeric: true,
      accessor: (r) => r.asset_growth_yoy,
      render: (r) => r.asset_growth_yoy != null
        ? `${(r.asset_growth_yoy * 100).toFixed(1)}%` : "—" },
    { key: "short_pct_float", header: "Short %", numeric: true,
      accessor: (r) => r.short_pct_float,
      render: (r) => fmt(r.short_pct_float, 1) },
    { key: "momentum_90d", header: "Mom 90d", numeric: true,
      accessor: (r) => r.momentum_90d,
      render: (r) => r.momentum_90d != null
        ? `${(r.momentum_90d * 100).toFixed(1)}%` : "—" },
    { key: "volatility_60d", header: "Vol 60d", numeric: true,
      accessor: (r) => r.volatility_60d,
      render: (r) => r.volatility_60d != null
        ? `${(r.volatility_60d * 100).toFixed(1)}%` : "—" },
    { key: "net_debt_to_ebitda", header: "ND/EBITDA", numeric: true,
      accessor: (r) => r.net_debt_to_ebitda,
      render: (r) => fmt(r.net_debt_to_ebitda, 2) },
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
        All {universe.length} S&amp;P 600 stocks with computed features. Click any column header to
        sort; use filters to narrow to a tier, sector, or just the IMA portfolio.
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

        <DataTable
          rows={universe}
          columns={cols}
          initialSortKey="composite_score"
          pageSize={30}
          filterText={filter}
          filterFn={filterFn}
          rowClassName={(r) => (r.is_portfolio ? "highlight" : undefined)}
        />
      </div>
    </div>
  );
}
