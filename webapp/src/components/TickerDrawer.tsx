import { useEffect, useMemo } from "react";
import { ClusterMeta, Meta, TrajectoryData, UniverseRow } from "../lib/types";
import { tierClass, fmt } from "../lib/data";
import {
  FEATURE_KEYS, FEATURE_META, featureDefinition, featureLong, ordinal,
} from "../lib/glossary";
import {
  FeatureRanks, classifyTrajectory, sectorPercentile, topRiskDrivers,
} from "../lib/assess";

interface Props {
  ticker: string;
  universe: UniverseRow[];
  meta: Meta;
  clusterMeta: ClusterMeta;
  trajectory?: TrajectoryData | null;
  featureRanks: FeatureRanks;
  onClose: () => void;
  onOpenPitch: (ticker: string) => void;
}

/**
 * Single-name research drawer: everything the dashboard knows about one
 * ticker, in one panel, in plain language. Opened by clicking any ticker.
 */
export function TickerDrawer({
  ticker, universe, meta, clusterMeta, trajectory, featureRanks,
  onClose, onOpenPitch,
}: Props) {
  const row = useMemo(
    () => universe.find((u) => u.Ticker === ticker) ?? null,
    [universe, ticker],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!row) return null;

  const ranks = featureRanks.get(ticker) ?? {};
  const risky = topRiskDrivers(featureRanks, ticker, 3);
  const safest = Object.entries(ranks).sort((a, b) => a[1] - b[1]).slice(0, 3);
  const secPct = sectorPercentile(universe, row);
  const trajDir = classifyTrajectory(trajectory, clusterMeta.risk_rank, ticker);
  const path = trajectory?.paths?.[ticker];
  const styleColor = meta.style_colors?.[row.cluster_style] ?? "#666";
  const cap = row.market_cap ? `$${(row.market_cap / 1e9).toFixed(2)}B` : null;

  // Nearest neighbors (PC distance)
  const neighbors = useMemo(() => {
    if (row.PC1 == null) return [];
    const pcs = ["PC1", "PC2", "PC3", "PC4"] as const;
    const cand = pcs.map((pc) => (row[pc] as number) ?? 0);
    return universe
      .filter((u) => u.Ticker !== ticker && u.PC1 != null)
      .map((u) => {
        let s = 0;
        pcs.forEach((pc, i) => { s += (((u[pc] as number) ?? 0) - cand[i]) ** 2; });
        return { u, d: Math.sqrt(s) };
      })
      .sort((a, b) => a.d - b.d)
      .slice(0, 5);
  }, [universe, row, ticker]);

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label={`${ticker} detail`}>
        <button className="close-btn" onClick={onClose} aria-label="Close">✕</button>

        <h2>
          {row.Ticker}
          {row.is_portfolio && (
            <span style={{ marginLeft: 8, fontSize: 11, verticalAlign: "middle",
                           background: "#1f3b73", color: "#fff",
                           padding: "2px 8px", borderRadius: 4 }}>
              IMA · {(row.weight * 100).toFixed(1)}%
            </span>
          )}
        </h2>
        <div className="drawer-sub">
          {row.Company} · {row.Sector}
          {row.Industry ? ` / ${row.Industry}` : ""}{cap ? ` · ${cap}` : ""}
        </div>

        <section>
          <h4>Risk summary</h4>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <span className={tierClass(row.risk_tier)}>{row.risk_tier}</span>
            <span style={{ fontSize: 13 }}>
              <strong>{ordinal(Math.round(row.score_percentile))}</strong> risk percentile
              of {universe.length} S&amp;P 600 names
            </span>
            <span className="style-chip" style={{ background: styleColor }}
                  data-hint
                  title="Descriptive style grouping from clustering — what the stock statistically looks like, NOT a risk rating.">
              {row.cluster_style}
            </span>
          </div>
          <div style={{ marginTop: 8, fontSize: 12, color: "var(--muted)" }}>
            Relative to today's S&amp;P 600 cross-section — descriptive, not a
            prediction. Composite score {fmt(row.composite_score, 1)}/100
            (equal-weighted mean of 16 feature percentiles).
          </div>
        </section>

        <section>
          <h4>What drives the score</h4>
          {risky.map((d) => (
            <div className="bar-row" key={d.feature}>
              <span className="bar-label" data-hint title={featureDefinition(d.feature)}>
                {featureLong(d.feature)}
              </span>
              <div className="bar-track">
                <div className="bar-fill" style={{
                  width: `${d.percentile}%`,
                  background: d.percentile > 80 ? "var(--danger)" : "#e57a44",
                }} />
              </div>
              <span className="bar-val">{d.percentile.toFixed(0)}</span>
            </div>
          ))}
          {safest.map(([f, p]) => (
            <div className="bar-row" key={f}>
              <span className="bar-label" data-hint title={featureDefinition(f)}>
                {featureLong(f)}
              </span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${p}%`, background: "var(--ok)" }} />
              </div>
              <span className="bar-val">{p.toFixed(0)}</span>
            </div>
          ))}
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>
            Percentile of risk vs the universe (100 = riskiest). Top three
            red flags, then the three strongest points.
          </div>
        </section>

        <section>
          <h4>Sector context</h4>
          <div style={{ fontSize: 13 }}>
            {secPct != null ? (
              <>
                <strong>{ordinal(Math.round(secPct))}</strong> risk percentile within{" "}
                {row.Sector} ({universe.filter((u) => u.Sector === row.Sector).length} names).
                {" "}Sector scores are universe-relative — low-volatility sectors
                rank safer as a group, so compare within sector too.
              </>
            ) : "Too few sector peers in the universe for a within-sector rank."}
          </div>
        </section>

        <section>
          <h4>Statistical neighbors</h4>
          <table className="data">
            <tbody>
              {neighbors.map(({ u, d }) => (
                <tr key={u.Ticker}>
                  <td>
                    <strong>{u.Ticker}</strong>
                    {u.is_portfolio && (
                      <span style={{ marginLeft: 6, fontSize: 9, background: "#1f3b73",
                                     color: "#fff", padding: "1px 5px", borderRadius: 3 }}>
                        IMA
                      </span>
                    )}
                  </td>
                  <td><small>{(u.Company ?? "").slice(0, 22)}</small></td>
                  <td><small>{u.Sector}</small></td>
                  <td className="num">{d.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>
            Closest names in risk-factor space — the stocks the model thinks
            this one statistically resembles.
          </div>
        </section>

        <section>
          <h4>Trajectory</h4>
          <div style={{ fontSize: 13 }}>
            {trajDir === "Unknown" ? (
              row.is_portfolio
                ? "No trajectory available in the latest run."
                : "Trajectories are computed for portfolio holdings only."
            ) : (
              <>
                <span style={{
                  fontWeight: 600,
                  color: trajDir === "Deteriorating" ? "var(--danger)"
                       : trajDir === "Improving" ? "var(--ok)" : "var(--muted)",
                }}>{trajDir}</span>
                {" "}over the last {path?.coords?.length ?? "few"} quarterly snapshots
                {path?.two_quarter_drift != null &&
                  ` · 2-quarter drift ${fmt(path.two_quarter_drift, 2)} PC units`}
              </>
            )}
          </div>
        </section>

        <section style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => onOpenPitch(ticker)}
            style={{
              padding: "9px 16px", fontSize: 13, fontWeight: 600,
              background: "var(--accent)", color: "#fff", border: "none",
              borderRadius: 6, cursor: "pointer",
            }}>
            Full pitch assessment →
          </button>
        </section>
      </aside>
    </>
  );
}
