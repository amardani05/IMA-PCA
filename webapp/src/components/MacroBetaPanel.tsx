import { useMemo, useState } from "react";
import { CATEGORY_LABELS, CATEGORY_ORDER, significanceStars } from "../lib/macro";
import { MacroBundle, PortfolioBetas } from "../lib/macroTypes";
import { fmt } from "../lib/data";

interface Props {
  macro: MacroBundle;
  liveBetas: Record<string, number>;          // overrides server betas (from tree selection)
  isModified: boolean;
}

type Mode = "v2" | "v1" | "raw";

export function MacroBetaPanel({ macro, liveBetas, isModified }: Props) {
  const factorMeta = macro.metadata.factors;
  const curated = new Set(macro.metadata.curated_factors);
  const factorByName: Record<string, typeof factorMeta[0]> =
    Object.fromEntries(factorMeta.map((f) => [f.factor, f]));

  // Default to the most-controlled regression that's available.
  // v2 strips IJR + VIX + HY OAS; v1 only IJR; raw strips nothing.
  const defaultMode: Mode = (() => {
    const m = macro.portfolioBetas.methodology;
    if (m === "residualized_v2") return "v2";
    if (m === "residualized_v1" || m === "residualized") return "v1";
    return "raw";
  })();
  const [mode, setMode] = useState<Mode>(defaultMode);

  const activeBetas: PortfolioBetas = (() => {
    if (mode === "raw" && macro.portfolioBetasRaw) return macro.portfolioBetasRaw;
    if (mode === "v1" && macro.portfolioBetasV1) return macro.portfolioBetasV1;
    return macro.portfolioBetas;
  })();

  // Group betas by category
  const grouped = useMemo(() => {
    const m = new Map<string, { factor: string; beta: number; live: number; meta: typeof factorMeta[0] }[]>();
    for (const factor of activeBetas.factors) {
      const meta = factorByName[factor];
      if (!meta) continue;
      const cat = meta.category;
      if (!m.has(cat)) m.set(cat, []);
      const beta = activeBetas.betas[factor].beta;
      const live = liveBetas[factor] ?? beta;
      m.get(cat)!.push({ factor, beta, live, meta });
    }
    return m;
  }, [activeBetas, liveBetas, factorByName]);

  const maxAbs = useMemo(() => {
    let m = 0;
    for (const factor of activeBetas.factors) {
      m = Math.max(m, Math.abs(activeBetas.betas[factor].beta));
      m = Math.max(m, Math.abs(liveBetas[factor] ?? 0));
    }
    return m || 1;
  }, [activeBetas, liveBetas]);

  const isResidualized = mode !== "raw" &&
    (activeBetas.methodology === "residualized" ||
     activeBetas.methodology === "residualized_v1" ||
     activeBetas.methodology === "residualized_v2");
  const marketBeta = isResidualized ? activeBetas.market_beta : null;
  const hasV1 = !!macro.portfolioBetasV1 && !macro.portfolioBetasV1.skipped;
  const hasRaw = !!macro.portfolioBetasRaw && !macro.portfolioBetasRaw.skipped;
  const showToggle = hasRaw || hasV1;

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between",
                     alignItems: "baseline", flexWrap: "wrap", gap: 8 }}>
        <h3 style={{ margin: 0 }}>
          Macro factor betas
          <span title={
            "Residualized: each factor is first regressed on IJR excess "
            + "returns; the portfolio is then regressed on the BENCHMARK + the "
            + "residualized factors. This isolates exposure to each factor "
            + "specifically, rather than picking up shared risk-sentiment beta."
          } style={{ marginLeft: 6, fontSize: 12, cursor: "help",
                     color: "var(--muted)" }}>(?)</span>
        </h3>
        <div style={{ fontSize: 12, color: "var(--muted)" }}>
          {isModified
            ? "live (approximate, from per-stock betas)"
            : `${activeBetas.methodology ?? "raw_ols"} · n=${activeBetas.n_obs}`}
          · R² = {fmt(activeBetas.r_squared, 3)}
          {marketBeta != null && (
            <> · market β = {fmt(marketBeta, 2)}</>
          )}
        </div>
      </div>

      {showToggle && (
        <div style={{ marginTop: 8, display: "flex", gap: 4, flexWrap: "wrap",
                       alignItems: "center" }}>
          <button onClick={() => setMode("v2")}
                  style={{ ...toggle,
                           background: mode === "v2" ? "var(--accent)" : "#fff",
                           color: mode === "v2" ? "#fff" : "var(--text)" }}
                  title="Multi-factor residualized: controls for IJR + VIX + HY OAS">
            Residualized v2 (default)
          </button>
          {hasV1 && (
            <button onClick={() => setMode("v1")}
                    style={{ ...toggle,
                             background: mode === "v1" ? "var(--accent)" : "#fff",
                             color: mode === "v1" ? "#fff" : "var(--text)" }}
                    title="Single-factor residualized: controls for IJR excess only">
              v1 (market only)
            </button>
          )}
          {hasRaw && (
            <button onClick={() => setMode("raw")}
                    style={{ ...toggle,
                             background: mode === "raw" ? "#b3001b" : "#fff",
                             color: mode === "raw" ? "#fff" : "var(--text)" }}
                    title="Raw OLS — no controls, shows the unadjusted artifact">
              Raw OLS
            </button>
          )}
          {mode === "raw" && (
            <small style={{ color: "var(--danger)" }}>
              ⚠ no controls — shared common-factor exposure inflates these betas
            </small>
          )}
          {mode === "v1" && (
            <small style={{ color: "var(--muted)" }}>
              Stripping market beta only — vol-regime + credit-cycle exposure not controlled
            </small>
          )}
          {mode === "v2" && (
            <small style={{ color: "var(--muted)" }}>
              Macro betas orthogonal to IJR + VIX + HY OAS (the three biggest common drivers)
            </small>
          )}
        </div>
      )}

      {isModified && (
        <div style={{ marginTop: 6, fontSize: 12, color: "var(--muted)",
                       background: "#fff9ea", border: "1px solid var(--warn)",
                       padding: "6px 10px", borderRadius: 4 }}>
          <strong>Approximation:</strong> portfolio betas are weighted averages of per-stock
          betas. Re-run <code>python main.py</code> with the new portfolio for an exact regression.
        </div>
      )}

      <div style={{ marginTop: 12 }}>
        {CATEGORY_ORDER.map((cat) => {
          const rows = grouped.get(cat);
          if (!rows || rows.length === 0) return null;
          return (
            <div key={cat} style={{ marginBottom: 14 }}>
              <div style={{
                fontSize: 11, fontWeight: 600, letterSpacing: 0.6,
                textTransform: "uppercase", color: "var(--muted)", marginBottom: 4,
              }}>{CATEGORY_LABELS[cat] || cat}</div>
              {rows.map((r) => (
                <BetaBar key={r.factor} row={r} maxAbs={maxAbs}
                         estimate={activeBetas.betas[r.factor]}
                         curated={curated.has(r.factor)} />
              ))}
            </div>
          );
        })}
      </div>

      {isResidualized && macro.comparison?.rows.length > 0 && (() => {
        const rows = macro.comparison.rows;
        const isThreeWay = rows.some((r) => r.v2_beta != null);
        return (
          <details style={{ marginTop: 12, fontSize: 12 }}>
            <summary style={{ cursor: "pointer", color: "var(--muted)",
                              fontWeight: 600, padding: "4px 0" }}>
              {isThreeWay
                ? `Raw → v1 → v2 progression (${rows.length} factors)`
                : `Raw vs Residualized comparison (${rows.length} factors)`}
            </summary>
            <table className="data" style={{ marginTop: 8 }}>
              <thead>
                {isThreeWay ? (
                  <tr>
                    <th>Factor</th>
                    <th className="num">Raw β</th>
                    <th className="num">v1 β</th>
                    <th className="num">v2 β</th>
                    <th>Interpretation</th>
                  </tr>
                ) : (
                  <tr>
                    <th>Factor</th>
                    <th className="num">Raw β</th>
                    <th className="num">Resid β</th>
                    <th className="num">Δ</th>
                    <th>Interpretation</th>
                  </tr>
                )}
              </thead>
              <tbody>
                {rows.map((r) => isThreeWay ? (
                  <tr key={r.factor}>
                    <td>{factorByName[r.factor]?.name ?? r.factor}</td>
                    <td className="num">{r.raw_beta != null ? r.raw_beta.toFixed(3) : "—"}</td>
                    <td className="num">{r.v1_beta != null ? r.v1_beta.toFixed(3) : "—"}</td>
                    <td className="num" style={{ fontWeight: 600 }}>
                      {r.v2_beta != null ? r.v2_beta.toFixed(3) : "—"}
                    </td>
                    <td><small>{r.interpretation}</small></td>
                  </tr>
                ) : (
                  <tr key={r.factor}>
                    <td>{factorByName[r.factor]?.name ?? r.factor}</td>
                    <td className="num">{r.raw_beta != null ? r.raw_beta.toFixed(3) : "—"}</td>
                    <td className="num">{r.residualized_beta != null ? r.residualized_beta.toFixed(3) : "—"}</td>
                    <td className="num" style={{
                      color: (r.delta ?? 0) > 0 ? "var(--ok)" :
                             (r.delta ?? 0) < 0 ? "var(--danger)" : "inherit",
                    }}>{r.delta != null ? (r.delta >= 0 ? "+" : "") + r.delta.toFixed(3) : "—"}</td>
                    <td><small>{r.interpretation}</small></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        );
      })()}
    </div>
  );
}

const toggle: React.CSSProperties = {
  padding: "4px 12px", fontSize: 12, fontWeight: 500,
  border: "1px solid var(--border)", borderRadius: 4, cursor: "pointer",
};


function BetaBar({ row, maxAbs, estimate, curated }: {
  row: { factor: string; beta: number; live: number; meta: any };
  maxAbs: number;
  estimate: any;
  curated: boolean;
}) {
  const live = row.live;
  const fitted = row.beta;
  const livePct = (Math.abs(live) / maxAbs) * 100;
  const fittedPct = (Math.abs(fitted) / maxAbs) * 100;
  const liveColor = live >= 0 ? "var(--ok)" : "var(--danger)";

  return (
    <div style={{ display: "grid", gridTemplateColumns: "180px 1fr 110px",
                   gap: 10, alignItems: "center", padding: "3px 0", fontSize: 13 }}>
      <div style={{ overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis" }}
           title={`${row.factor} · ${row.meta.transform} · source=${row.meta.source}`}>
        {row.meta.name}
        {!curated && <small className="muted" style={{ marginLeft: 6 }}>(non-curated)</small>}
      </div>
      <div style={{ position: "relative", height: 16, background: "#f0f2f6", borderRadius: 3 }}>
        <div style={{ position: "absolute", top: 0, bottom: 0, left: "50%", width: 1, background: "#999" }} />
        {/* Fitted (faded) */}
        <div style={{
          position: "absolute", top: 2, bottom: 2,
          left: fitted >= 0 ? "50%" : `${50 - fittedPct / 2}%`,
          width: `${fittedPct / 2}%`,
          background: fitted >= 0 ? "rgba(44,122,75,0.25)" : "rgba(179,0,27,0.25)",
          borderRadius: 2,
        }} />
        {/* Live overlay */}
        <div style={{
          position: "absolute", top: 4, bottom: 4,
          left: live >= 0 ? "50%" : `${50 - livePct / 2}%`,
          width: `${livePct / 2}%`,
          background: liveColor, borderRadius: 2,
        }} />
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end",
                     gap: 6, fontVariantNumeric: "tabular-nums" }}>
        <span style={{ color: live >= 0 ? "var(--ok)" : "var(--danger)", fontWeight: 600 }}>
          {live >= 0 ? "+" : ""}{live.toFixed(3)}
        </span>
        <small style={{ color: "var(--muted)" }}>
          {significanceStars(estimate.p_value)}
        </small>
      </div>
    </div>
  );
}
