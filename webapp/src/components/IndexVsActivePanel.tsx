import { useEffect, useState } from "react";
import { FactorMetadata } from "../lib/macroTypes";
import { significanceStars } from "../lib/macro";

interface BetaCell { beta: number; t_stat: number; p_value: number }
interface Row { factor: string; index: BetaCell | null; portfolio_total: BetaCell | null; active: BetaCell | null }
interface Payload { rows: Row[]; note: string }

/**
 * The "is this relative to the index?" answer, permanently on screen:
 * per factor — what IJR itself carries (raw), what the portfolio carries in
 * total (raw), and the ACTIVE exposure (net of index/VIX/credit).
 */
export function IndexVsActivePanel({ metadata }: { metadata: FactorMetadata }) {
  const [data, setData] = useState<Payload | null>(null);

  useEffect(() => {
    fetch("/data/macro/index_vs_active.json")
      .then((r) => (r.ok ? r.json() : null))
      .then(setData)
      .catch(() => setData(null));
  }, []);

  if (!data) return null;

  const nameOf = (f: string) =>
    metadata.factors.find((m) => m.factor === f)?.name ?? f;

  const rows = [...data.rows].sort(
    (a, b) => Math.abs(b.active?.beta ?? 0) - Math.abs(a.active?.beta ?? 0),
  );

  const cell = (c: BetaCell | null) => {
    if (!c) return <td className="num">—</td>;
    const sig = c.p_value < 0.10;
    return (
      <td className="num" style={{ fontWeight: sig ? 700 : 400,
                                   color: sig ? (c.beta >= 0 ? "var(--danger)" : "var(--ok)") : undefined }}>
        {c.beta >= 0 ? "+" : ""}{c.beta.toFixed(3)}
        <small style={{ marginLeft: 3 }}>{significanceStars(c.p_value)}</small>
      </td>
    );
  };

  return (
    <div className="card">
      <h3 style={{ margin: 0 }}>Active vs index — who owns each exposure?</h3>
      <div className="card-sub">
        Total ≈ index-inherited + active. <strong>Index</strong> and{" "}
        <strong>portfolio total</strong> are raw (unconditional) betas;{" "}
        <strong>active</strong> is net of IJR + VIX + HY credit — the part the
        committee can act on. Bold = significant at 10%.
      </div>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Factor</th>
              <th className="num">IJR (index)</th>
              <th className="num">Portfolio total</th>
              <th className="num">Active</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.factor}>
                <td>{nameOf(r.factor)}</td>
                {cell(r.index)}
                {cell(r.portfolio_total)}
                {cell(r.active)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
