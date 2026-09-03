import { useMemo, useState } from "react";
import { PortfolioRow } from "../lib/types";
import { FactorMetadata } from "../lib/macroTypes";
import {
  UniverseFactorBetas, betaOf, exposureWithCandidate,
} from "../lib/factorBetas";

interface Props {
  ub: UniverseFactorBetas;
  metadata: FactorMetadata | null;
  portfolio: PortfolioRow[];
  candidate: string;
}

const DEFAULT_WEIGHT = 0.05;

/**
 * The question a PM actually asks when picking a name: what does adding this
 * do to the sleeve's macro exposures? Uses the identity that a portfolio's raw
 * factor beta is the weight-average of its holdings' raw betas, so the "after"
 * column is exact, not an approximation.
 */
export function SleeveImpactPanel({ ub, metadata, portfolio, candidate }: Props) {
  const [weight, setWeight] = useState(DEFAULT_WEIGHT);

  const holdings = useMemo(
    () => portfolio
      .filter((p) => p.Ticker !== candidate && p.Weight > 0)
      .map((p) => ({ ticker: p.Ticker, weight: p.Weight })),
    [portfolio, candidate],
  );

  const nameOf = (f: string) =>
    metadata?.factors.find((m) => m.factor === f)?.name ?? f;

  const rows = useMemo(() => {
    return ub.factors
      .map((f) => {
        const { before, after, delta } =
          exposureWithCandidate(ub, holdings, candidate, weight, f);
        const idx = ub.index_betas[f] ?? null;
        return { factor: f, before, after, delta, idx,
                 candBeta: betaOf(ub, candidate, f) };
      })
      .filter((r) => r.delta != null)
      .sort((a, b) => Math.abs(b.delta ?? 0) - Math.abs(a.delta ?? 0));
  }, [ub, holdings, candidate, weight]);

  if (rows.length === 0) return null;

  const top = rows.slice(0, 6);

  return (
    <div className="card">
      <h3 style={{ margin: 0 }}>What this does to the sleeve's factor exposure</h3>
      <div className="card-sub" style={{ display: "flex", alignItems: "center",
                                          gap: 10, flexWrap: "wrap" }}>
        <span>
          Adding <strong>{candidate}</strong> at
        </span>
        <input
          type="range" min={1} max={10} step={0.5}
          value={weight * 100}
          onChange={(e) => setWeight(Number(e.target.value) / 100)}
          style={{ width: 140 }}
          aria-label="Candidate position weight"
        />
        <strong style={{ fontVariantNumeric: "tabular-nums" }}>
          {(weight * 100).toFixed(1)}%
        </strong>
        <span>(existing holdings scaled pro-rata). Betas are raw; "vs index"
          compares to what IJR itself carries.</span>
      </div>

      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Factor</th>
              <th className="num">Candidate β</th>
              <th className="num">Index β</th>
              <th className="num">Sleeve now</th>
              <th className="num">Sleeve after</th>
              <th className="num">Change</th>
            </tr>
          </thead>
          <tbody>
            {top.map((r) => {
              const d = r.delta ?? 0;
              const meaningful = Math.abs(d) > 0.002;
              return (
                <tr key={r.factor}>
                  <td>{nameOf(r.factor)}</td>
                  <td className="num">{r.candBeta?.toFixed(3) ?? "—"}</td>
                  <td className="num" style={{ color: "var(--muted)" }}>
                    {r.idx != null ? r.idx.toFixed(3) : "—"}
                  </td>
                  <td className="num">{r.before?.toFixed(3) ?? "—"}</td>
                  <td className="num">{r.after?.toFixed(3) ?? "—"}</td>
                  <td className="num" style={{
                    fontWeight: meaningful ? 700 : 400,
                    color: !meaningful ? "var(--muted)"
                         : d > 0 ? "var(--danger)" : "var(--ok)",
                  }}>
                    {d >= 0 ? "+" : ""}{d.toFixed(3)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <small className="muted">
        Ranked by absolute change. Red = the position increases that exposure,
        green = it reduces it. Use this to size a name against the exposure you
        are trying to add or hedge, not as a standalone reason to buy.
      </small>
    </div>
  );
}
