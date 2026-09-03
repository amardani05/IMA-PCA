import { useEffect, useMemo, useState } from "react";
import { PitchAssessment, recommendationColor } from "../lib/pitch";
import { buildAssessment, computeFeatureRanks } from "../lib/assess";
import { tierClass, fmt } from "../lib/data";
import { featureLong, featureDefinition, ordinal } from "../lib/glossary";
import { ClusterMeta, Meta, PortfolioRow, TrajectoryData, UniverseRow } from "../lib/types";
import { UniverseFactorBetas } from "../lib/factorBetas";
import { FactorMetadata } from "../lib/macroTypes";
import { SleeveImpactPanel } from "./SleeveImpactPanel";

interface Props {
  meta: Meta;
  universe: UniverseRow[];
  clusterMeta: ClusterMeta;
  trajectory?: TrajectoryData | null;
  initialTicker?: string | null;
  portfolio: PortfolioRow[];
  universeBetas?: UniverseFactorBetas | null;
  factorMetadata?: FactorMetadata | null;
}

export function PitchView({ meta, universe, clusterMeta, trajectory, initialTicker,
                            portfolio, universeBetas, factorMetadata }: Props) {
  const [search, setSearch] = useState(initialTicker ?? "");
  const [active, setActive] = useState<string | null>(initialTicker ?? null);

  // Ranks are the only expensive part — compute once per session.
  const featureRanks = useMemo(() => computeFeatureRanks(universe), [universe]);

  useEffect(() => {
    if (initialTicker) {
      setSearch(initialTicker);
      setActive(initialTicker);
    }
  }, [initialTicker]);

  const result = useMemo(() => {
    if (!active) return null;
    return buildAssessment(active, { universe, meta, clusterMeta, trajectory, featureRanks });
  }, [active, universe, meta, clusterMeta, trajectory, featureRanks]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const t = search.trim().toUpperCase();
    if (t) setActive(t);
  };

  // Typeahead: matching universe rows for a partial query
  const suggestions = useMemo(() => {
    const q = search.trim().toUpperCase();
    if (!q || (result && "ticker" in result && result.ticker === q)) return [];
    return universe
      .filter((u) =>
        u.Ticker.startsWith(q) ||
        (u.Company ?? "").toUpperCase().includes(q))
      .slice(0, 8);
  }, [search, universe, result]);

  return (
    <div>
      <h2 className="section-title">Pitch assessor</h2>
      <p className="section-lede">
        Type any S&amp;P 600 ticker to get an instant statistical read on it as a
        <em> candidate</em>: how similar it is to what the fund already owns, where
        it sits in the risk model, and what to address in committee. Assessments
        are computed live from the latest pipeline run — every universe name works.
        This is a screen, not a verdict: it has no view on the business, only on
        the statistics.
      </p>

      <div className="card" style={{ padding: 14 }}>
        <form onSubmit={handleSubmit} style={{ display: "flex", gap: 8 }}>
          <input
            type="search"
            placeholder="Search a ticker or company (e.g. TDS, Shenandoah, IRDM)…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ flex: 1, padding: "8px 12px", fontSize: 14,
                     border: "1px solid var(--border)", borderRadius: 6 }}
          />
          <button type="submit" style={{
            padding: "8px 18px", fontSize: 14, fontWeight: 500,
            background: "var(--accent)", color: "#fff", border: "none",
            borderRadius: 6, cursor: "pointer",
          }}>Assess</button>
        </form>

        {suggestions.length > 0 && (
          <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6 }}>
            {suggestions.map((u) => (
              <button key={u.Ticker}
                      onClick={() => { setSearch(u.Ticker); setActive(u.Ticker); }}
                      style={{
                        padding: "5px 10px", fontSize: 12,
                        border: "1px solid var(--border)", borderRadius: 14,
                        background: "#fff", cursor: "pointer",
                      }}
                      title={`${u.Company ?? ""} · ${u.Sector}`}>
                <strong>{u.Ticker}</strong>
                <span style={{ color: "var(--muted)", marginLeft: 6 }}>
                  {(u.Company ?? "").slice(0, 28)}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {result && "error" in result && (
        <div className="card error" style={{ marginTop: 12 }}>
          <p style={{ margin: 0 }}>{result.error}</p>
        </div>
      )}

      {result && !("error" in result) && (
        <>
          <PitchPage pitch={result} meta={meta} />
          {universeBetas && universeBetas.betas[result.ticker] && (
            <SleeveImpactPanel ub={universeBetas} metadata={factorMetadata ?? null}
                               portfolio={portfolio} candidate={result.ticker} />
          )}
        </>
      )}
    </div>
  );
}


function PitchPage({ pitch, meta }: { pitch: PitchAssessment; meta: Meta }) {
  const recColor = recommendationColor(pitch.recommendation);
  const cap = pitch.market_cap > 0 ? `$${(pitch.market_cap / 1e9).toFixed(2)}B` : "n/a";

  return (
    <div style={{ marginTop: 16 }}>
      {/* Header */}
      <div className="card" style={{ borderLeft: `5px solid ${recColor}`, padding: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between",
                       alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ fontSize: 24, fontWeight: 700 }}>{pitch.ticker}</div>
            <div style={{ fontSize: 14, color: "var(--muted)" }}>
              {pitch.company_name}
            </div>
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>
              {pitch.sector}{pitch.industry ? ` / ${pitch.industry}` : ""}  ·  Market cap {cap}
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <span style={{
              padding: "8px 16px", borderRadius: 6, fontWeight: 700,
              background: recColor, color: "#fff", fontSize: 14,
            }}>{pitch.recommendation}</span>
            <div style={{ marginTop: 8, maxWidth: 460, fontSize: 12, color: "var(--muted)" }}>
              {pitch.recommendation_rationale}
            </div>
          </div>
        </div>
      </div>

      {/* Key findings bullets */}
      <div className="card">
        <h3 style={{ margin: 0 }}>Key findings</h3>
        <ul style={{ margin: "8px 0 0", paddingLeft: 22 }}>
          {pitch.summary_bullets.map((b, i) => (
            <li key={i} style={{ marginBottom: 6 }}>{b}</li>
          ))}
        </ul>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)", gap: 12 }}>
        {/* Neighbors */}
        <div className="card">
          <h3 style={{ margin: 0 }}>Nearest neighbors</h3>
          <div className="card-sub">
            5 statistically closest stocks in PC space.
            {pitch.n_neighbors_currently_held > 0 && (
              <> <strong>{pitch.n_neighbors_currently_held}</strong> currently held.</>
            )}
          </div>
          <table className="data" style={{ marginTop: 6 }}>
            <thead>
              <tr><th>Ticker</th><th>Sector</th><th className="num">Distance</th><th></th></tr>
            </thead>
            <tbody>
              {pitch.nearest_neighbors.map((n) => (
                <tr key={n.ticker} className={n.is_held ? "highlight" : undefined}>
                  <td><strong>{n.ticker}</strong></td>
                  <td><small>{n.sector}</small></td>
                  <td className="num">{n.distance.toFixed(2)}</td>
                  <td>
                    {n.is_held && <span style={badgeHeld}>HELD</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* PC deviations */}
        <div className="card">
          <h3 style={{ margin: 0 }}>Portfolio differentiation</h3>
          <div className="card-sub">
            Diversification score: <strong>{pitch.diversification_score.toFixed(0)}/100</strong>
            {" "}— how far the name sits from the portfolio's center of gravity.
          </div>
          <table className="data" style={{ marginTop: 6 }}>
            <thead>
              <tr><th>PC</th><th>Label</th><th className="num">Candidate</th>
                  <th className="num">Centroid</th><th className="num">σ-deviation</th></tr>
            </thead>
            <tbody>
              {Object.entries(pitch.candidate_position).map(([pc, val]) => {
                const dev = pitch.deviations_from_centroid[pc] ?? 0;
                const cen = pitch.portfolio_centroid[pc] ?? 0;
                const significant = Math.abs(dev) > 1.0;
                return (
                  <tr key={pc} style={significant ? { background: "#fff9ea" } : undefined}>
                    <td><strong>{pc}</strong></td>
                    <td><small>{meta.pca.pc_labels[pc] ?? ""}</small></td>
                    <td className="num">{val.toFixed(2)}</td>
                    <td className="num">{cen.toFixed(2)}</td>
                    <td className="num" style={{
                      fontWeight: significant ? 600 : 400,
                      color: dev > 0 ? "var(--danger)" : "var(--ok)",
                    }}>
                      {dev >= 0 ? "+" : ""}{dev.toFixed(2)}σ
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Risk profile */}
      <div className="card">
        <h3 style={{ margin: 0 }}>Risk profile</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
                       gap: 12, marginTop: 8 }}>
          <Stat label="Risk tier"
                value={pitch.risk_tier}
                sub={`${ordinal(Math.round(pitch.score_percentile))} pctile of universe`}
                tierBadge={pitch.risk_tier} />
          <Stat label="Style cluster" value={pitch.cluster_style}
                sub={`cluster C${pitch.cluster_id} — descriptive, not a risk rank`} />
          <Stat label="Trajectory" value={pitch.cluster_trajectory}
                sub="direction through risk space, last 4 quarters" />
          <Stat label="vs sector" value={pitch.sector_comparison}
                sub={`sector median score ${fmt(pitch.sector_median_score, 1)}`} />
        </div>

        {pitch.top_risk_drivers.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase",
                           letterSpacing: 0.5, fontWeight: 600, marginBottom: 4 }}>
              Top risk drivers (percentile rank in universe — 100 = riskiest)
            </div>
            {pitch.top_risk_drivers.map((d) => (
              <div key={d.feature} style={{ display: "flex", alignItems: "center",
                                              gap: 10, padding: "3px 0" }}>
                <span data-hint title={featureDefinition(d.feature)}
                      style={{ minWidth: 220, fontSize: 13 }}>
                  {featureLong(d.feature)}
                </span>
                <div style={{ flex: 1, background: "#f0f2f6", borderRadius: 4, height: 14 }}>
                  <div style={{
                    width: `${d.percentile}%`, height: "100%",
                    background: d.percentile > 80 ? "var(--danger)"
                              : d.percentile > 60 ? "#e57a44"
                              : d.percentile > 40 ? "var(--warn)"
                              : "var(--ok)",
                    borderRadius: 4,
                  }} />
                </div>
                <span style={{ minWidth: 36, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                  {d.percentile.toFixed(0)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div style={{ fontSize: 11, color: "var(--muted)", textAlign: "right", marginTop: 6 }}>
        Computed live from the pipeline run of {new Date(pitch.generated_at).toLocaleString()}.
        Statistical screen only — validate with fundamental work.
      </div>
    </div>
  );
}


function Stat({ label, value, sub, tierBadge }: { label: string; value: string; sub?: string; tierBadge?: string }) {
  return (
    <div style={{ background: "#f7f8fa", borderRadius: 6, padding: 10 }}>
      <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase",
                     letterSpacing: 0.5, fontWeight: 600 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 600, marginTop: 4 }}>
        {tierBadge ? <span className={tierClass(tierBadge)} style={{ fontSize: 12 }}>{value}</span> : value}
      </div>
      {sub && <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}


const badgeHeld: React.CSSProperties = {
  padding: "1px 6px", background: "var(--accent)", color: "#fff",
  fontSize: 9, fontWeight: 700, borderRadius: 8,
};
