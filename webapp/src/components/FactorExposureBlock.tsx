import { UniverseFactorBetas, betaOf, pValueOf } from "../lib/factorBetas";
import { FactorMetadata } from "../lib/macroTypes";

interface Props {
  ub: UniverseFactorBetas;
  metadata: FactorMetadata | null;
  ticker: string;
  /** Show only the N largest |beta − index beta| rows. */
  limit?: number;
  compact?: boolean;
}

/**
 * One stock's macro factor exposures, expressed the way a PM reads them:
 * its own beta, the index's beta, and the difference — i.e. what owning this
 * name instead of the index does to each exposure.
 */
export function FactorExposureBlock({ ub, metadata, ticker, limit = 6, compact }: Props) {
  const nameOf = (f: string) =>
    metadata?.factors.find((m) => m.factor === f)?.name ?? f;

  const rows = ub.factors
    .map((f) => {
      const beta = betaOf(ub, ticker, f);
      const idx = ub.index_betas[f] ?? null;
      const p = pValueOf(ub, ticker, f);
      const vsIndex = beta != null && idx != null ? beta - idx : null;
      return { factor: f, beta, idx, vsIndex, p };
    })
    .filter((r) => r.beta != null)
    .sort((a, b) => Math.abs(b.vsIndex ?? 0) - Math.abs(a.vsIndex ?? 0))
    .slice(0, limit);

  if (rows.length === 0) {
    return <p className="muted" style={{ fontSize: 12 }}>No factor betas estimated for {ticker}.</p>;
  }

  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.vsIndex ?? 0)), 0.001);

  return (
    <>
      {rows.map((r) => {
        const v = r.vsIndex ?? 0;
        const pct = (Math.abs(v) / maxAbs) * 50; // half-width bar either side
        const sig = (r.p ?? 1) < 0.10;
        return (
          <div className="bar-row" key={r.factor}>
            <span className="bar-label" title={`${nameOf(r.factor)} — own beta ${r.beta?.toFixed(3)}, index beta ${r.idx?.toFixed(3)}`}>
              {nameOf(r.factor)}
            </span>
            <div className="bar-track" style={{ position: "relative" }}>
              <div style={{
                position: "absolute", left: "50%", top: 0, bottom: 0,
                width: 1, background: "#c7ced6",
              }} />
              <div style={{
                position: "absolute", top: 0, bottom: 0,
                left: v >= 0 ? "50%" : `${50 - pct}%`,
                width: `${pct}%`,
                background: v >= 0 ? "#b3001b" : "#2c7a4b",
                opacity: sig ? 1 : 0.45,
                borderRadius: 3,
              }} />
            </div>
            <span className="bar-val" style={{ fontWeight: sig ? 700 : 400 }}>
              {v >= 0 ? "+" : ""}{v.toFixed(2)}
            </span>
          </div>
        );
      })}
      {!compact && (
        <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>
          Exposure <em>relative to the index</em> (own beta − IJR beta). Right/red
          = more exposed than the benchmark, left/green = less. Faded bars are
          not statistically significant.
        </div>
      )}
    </>
  );
}
