import { useEffect, useState } from "react";
import { Attribution, loadAttribution } from "../lib/factorBetas";
import { FactorMetadata } from "../lib/macroTypes";
import { significanceStars } from "../lib/macro";

/**
 * Active-return attribution: exposures say what we are exposed to, this says
 * what those exposures actually cost or earned over the window, and how much
 * of the active return was stock selection rather than macro.
 */
export function AttributionPanel({ metadata }: { metadata: FactorMetadata | null }) {
  const [attr, setAttr] = useState<Attribution | null>(null);

  useEffect(() => { loadAttribution().then(setAttr); }, []);

  if (!attr || !attr.contributions) return null;

  const nameOf = (f: string) =>
    metadata?.factors.find((m) => m.factor === f)?.name ?? f;

  const pct = (v: number | undefined) =>
    v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;

  const total = attr.total_active_return ?? 0;
  const explained = attr.factor_explained ?? 0;
  const selection = attr.selection_residual ?? 0;
  const rows = attr.contributions.filter((r) => Math.abs(r.contribution) > 1e-5);
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1e-4);

  return (
    <div className="card">
      <h3 style={{ margin: 0 }}>Active return attribution — what the exposures cost</h3>
      <div className="card-sub">
        Active return (portfolio − IJR) over {attr.window?.[0]} → {attr.window?.[1]}{" "}
        decomposed into each factor's contribution (beta × the factor's realized
        move) plus <strong>selection</strong> — the part the macro factors do not
        explain, i.e. stock picking.
      </div>

      <div className="grid grid-3" style={{ marginBottom: 12 }}>
        <div className="card stat" style={{ marginBottom: 0 }}>
          <div className="stat-label">Total active return</div>
          <div className="stat-value" style={{ color: total >= 0 ? "var(--ok)" : "var(--danger)" }}>
            {pct(total)}
          </div>
          <small className="muted">cumulative, {attr.n_obs} trading days</small>
        </div>
        <div className="card stat" style={{ marginBottom: 0 }}>
          <div className="stat-label">From macro factors</div>
          <div className="stat-value">{pct(explained)}</div>
          <small className="muted">sum of the contributions below</small>
        </div>
        <div className="card stat" style={{ marginBottom: 0 }}>
          <div className="stat-label">From selection</div>
          <div className="stat-value" style={{ color: selection >= 0 ? "var(--ok)" : "var(--danger)" }}>
            {pct(selection)}
          </div>
          <small className="muted">unexplained by the factor library</small>
        </div>
      </div>

      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Factor</th>
              <th className="num">Active β</th>
              <th className="num">Factor move</th>
              <th className="num">Contribution</th>
              <th style={{ width: "34%" }}></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const w = (Math.abs(r.contribution) / maxAbs) * 50;
              return (
                <tr key={r.factor}>
                  <td>{nameOf(r.factor)}</td>
                  <td className="num">
                    {r.beta >= 0 ? "+" : ""}{r.beta.toFixed(3)}
                    <small> {significanceStars(r.p_value)}</small>
                  </td>
                  <td className="num" style={{ color: "var(--muted)" }}>
                    {r.factor_move >= 0 ? "+" : ""}{r.factor_move.toFixed(2)}
                  </td>
                  <td className="num" style={{
                    fontWeight: 600,
                    color: r.contribution >= 0 ? "var(--ok)" : "var(--danger)",
                  }}>
                    {pct(r.contribution)}
                  </td>
                  <td>
                    <div style={{ position: "relative", height: 12,
                                   background: "#f0f2f6", borderRadius: 3 }}>
                      <div style={{ position: "absolute", left: "50%", top: 0,
                                     bottom: 0, width: 1, background: "#c7ced6" }} />
                      <div style={{
                        position: "absolute", top: 0, bottom: 0,
                        left: r.contribution >= 0 ? "50%" : `${50 - w}%`,
                        width: `${w}%`, borderRadius: 3,
                        background: r.contribution >= 0 ? "#2c7a4b" : "#b3001b",
                      }} />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <small className="muted">
        "Factor move" is the summed factor change over the window in its own
        units. Contributions are approximate — they hold betas fixed across the
        window, so a drifting exposure is only captured on average.
      </small>
    </div>
  );
}
