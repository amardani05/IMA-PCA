import { useMemo, useState } from "react";
import { MacroBundle } from "../lib/macroTypes";
import { CATEGORY_LABELS, CATEGORY_ORDER } from "../lib/macro";
import { PortfolioRow } from "../lib/types";
import { fmt } from "../lib/data";

interface Props {
  macro: MacroBundle;
  portfolio: PortfolioRow[];
}

type FactorScope = "curated" | "all";

/**
 * Per-holding × per-macro-factor beta matrix.
 *
 * Cells render the beta value with a divergent red/blue background scaled to
 * the per-factor robust max-abs (so columns with naturally tiny betas, e.g.
 * VIX in level-changes, still pop). Rows are portfolio holdings sorted by
 * weight; columns are macro factors grouped by category. Significance stars
 * come from the per-stock p-value matrix exported by the pipeline.
 */
export function PortfolioBetaTable({ macro, portfolio }: Props) {
  const [scope, setScope] = useState<FactorScope>("curated");

  const factorMeta = useMemo(
    () => Object.fromEntries(macro.metadata.factors.map((f) => [f.factor, f])),
    [macro],
  );

  const factors = useMemo(() => {
    const all = macro.stockBetas.factors;
    if (scope === "curated") {
      const curated = new Set(macro.metadata.curated_factors);
      return all.filter((f) => curated.has(f));
    }
    return all;
  }, [macro, scope]);

  // Group factors by category for column ordering
  const orderedFactors = useMemo(() => {
    const byCat: Record<string, string[]> = {};
    for (const f of factors) {
      const cat = factorMeta[f]?.category ?? "other";
      byCat[cat] ??= [];
      byCat[cat].push(f);
    }
    const out: string[] = [];
    for (const cat of CATEGORY_ORDER) {
      if (byCat[cat]) out.push(...byCat[cat]);
    }
    // Append any uncategorized
    for (const cat of Object.keys(byCat)) {
      if (!CATEGORY_ORDER.includes(cat as any)) out.push(...byCat[cat]);
    }
    return out;
  }, [factors, factorMeta]);

  // Robust per-factor scaling for cell color (95th-pct of |beta|, fallback small)
  const scales = useMemo(() => {
    const out: Record<string, number> = {};
    for (const f of orderedFactors) {
      const vals = portfolio
        .map((p) => Math.abs(macro.stockBetas.betas[p.Ticker]?.[f] ?? 0))
        .filter((v) => v > 0)
        .sort((a, b) => a - b);
      const n = vals.length;
      out[f] = n > 0 ? Math.max(vals[Math.floor(n * 0.95)], 1e-6) : 1;
    }
    return out;
  }, [orderedFactors, portfolio, macro]);

  // Sort portfolio holdings by weight descending
  const rows = useMemo(() => {
    return [...portfolio]
      .filter((p) => macro.stockBetas.betas[p.Ticker])
      .sort((a, b) => (b.Weight ?? 0) - (a.Weight ?? 0));
  }, [portfolio, macro]);

  // Group factors visually by category — render a category header row
  const factorGroupBoundaries = useMemo(() => {
    const out: { factor: string; category: string; isFirst: boolean }[] = [];
    let prevCat = "";
    for (const f of orderedFactors) {
      const cat = factorMeta[f]?.category ?? "other";
      out.push({ factor: f, category: cat, isFirst: cat !== prevCat });
      prevCat = cat;
    }
    return out;
  }, [orderedFactors, factorMeta]);

  if (rows.length === 0) return null;

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h3 style={{ margin: 0 }}>Portfolio × macro factor betas</h3>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={() => setScope("curated")}
                  style={{ ...toggleBtn,
                           background: scope === "curated" ? "var(--accent)" : "#fff",
                           color: scope === "curated" ? "#fff" : "var(--text)" }}>
            Curated ({macro.metadata.curated_factors.length})
          </button>
          <button onClick={() => setScope("all")}
                  style={{ ...toggleBtn,
                           background: scope === "all" ? "var(--accent)" : "#fff",
                           color: scope === "all" ? "#fff" : "var(--text)" }}>
            All ({macro.stockBetas.factors.length})
          </button>
        </div>
      </div>
      <div className="card-sub">
        Each cell is the per-stock OLS beta on the macro factor (HAC standard
        errors). Hover for t-stat / p-value. Red = positive beta (long that
        factor), blue = negative beta (short). Color magnitude is per-column
        (95th-pct of |beta|).{" "}
        <strong>★</strong> p&lt;0.10 · <strong>★★</strong> p&lt;0.05 ·
        <strong>★★★</strong> p&lt;0.01.
      </div>

      <div className="table-wrap" style={{ marginTop: 8 }}>
        <table className="data" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th colSpan={2} style={{ position: "sticky", left: 0, zIndex: 2, background: "#f0f2f6" }}>
                Holding
              </th>
              {factorGroupBoundaries.map(({ factor, category, isFirst }) => (
                <th key={factor} title={factor}
                    className="num"
                    style={{
                      borderLeft: isFirst ? "2px solid var(--accent)" : undefined,
                      maxWidth: 110, whiteSpace: "normal", lineHeight: 1.2,
                      verticalAlign: "bottom", padding: "8px 6px",
                    }}>
                  {isFirst && (
                    <div style={{
                      fontSize: 9, color: "var(--accent)",
                      textTransform: "uppercase", letterSpacing: 0.4,
                      marginBottom: 2,
                    }}>
                      {CATEGORY_LABELS[category] ?? category}
                    </div>
                  )}
                  <div>{factorMeta[factor]?.name ?? factor}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const tk = row.Ticker;
              return (
                <tr key={tk}>
                  <td style={{ position: "sticky", left: 0, background: "#fff",
                                fontWeight: 600 }}>
                    {tk}
                  </td>
                  <td className="num" style={{ position: "sticky", left: 50, background: "#fff",
                                                 color: "var(--muted)" }}>
                    {((row.Weight ?? 0) * 100).toFixed(2)}%
                  </td>
                  {factorGroupBoundaries.map(({ factor, isFirst }) => {
                    const beta = macro.stockBetas.betas[tk]?.[factor];
                    const p = macro.stockBetas.p_values?.[tk]?.[factor] ?? 1;
                    const se = macro.stockBetas.std_errors?.[tk]?.[factor];
                    if (beta == null) {
                      return <td key={factor} className="num"
                                  style={{ borderLeft: isFirst ? "2px solid var(--accent)" : undefined,
                                            color: "var(--muted)" }}>—</td>;
                    }
                    const scale = scales[factor] || 1;
                    const mag = Math.min(Math.abs(beta) / scale, 1);
                    const bg = beta >= 0
                      ? `rgba(179,0,27,${(mag * 0.55).toFixed(2)})`
                      : `rgba(31,59,115,${(mag * 0.55).toFixed(2)})`;
                    const stars = p < 0.01 ? "★★★" : p < 0.05 ? "★★" : p < 0.10 ? "★" : "";
                    return (
                      <td key={factor} className="num"
                          title={`β=${beta.toFixed(3)}  SE=${fmt(se, 3)}  p=${p.toFixed(3)}`}
                          style={{
                            background: bg,
                            borderLeft: isFirst ? "2px solid var(--accent)" : undefined,
                            fontVariantNumeric: "tabular-nums",
                            color: mag > 0.7 ? "#fff" : "inherit",
                          }}>
                        {beta >= 0 ? "+" : ""}{beta.toFixed(2)}
                        {stars && <span style={{ marginLeft: 2, opacity: 0.7,
                                                  fontSize: 9 }}>{stars}</span>}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const toggleBtn: React.CSSProperties = {
  padding: "4px 12px", fontSize: 12, fontWeight: 500,
  border: "1px solid var(--border)", borderRadius: 4, cursor: "pointer",
};
