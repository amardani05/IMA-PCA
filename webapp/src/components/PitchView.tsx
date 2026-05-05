import { useEffect, useState } from "react";
import {
  PitchAssessment,
  PitchIndexRow,
  loadPitch,
  loadPitchIndex,
  recommendationColor,
} from "../lib/pitch";
import { tierClass, fmt } from "../lib/data";
import { Meta } from "../lib/types";

interface Props {
  meta: Meta;
}

export function PitchView({ meta }: Props) {
  const [search, setSearch] = useState("");
  const [active, setActive] = useState<string | null>(null);
  const [pitch, setPitch] = useState<PitchAssessment | null>(null);
  const [pitchErr, setPitchErr] = useState<string | null>(null);
  const [index, setIndex] = useState<PitchIndexRow[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadPitchIndex().then((rows) => setIndex(rows ?? []));
  }, []);

  useEffect(() => {
    if (!active) return;
    setLoading(true);
    setPitchErr(null);
    loadPitch(active).then((p) => {
      if (!p) {
        setPitchErr(
          `No pitch assessment found for ${active}. Generate it from the CLI:\n` +
          `  python main.py --assess ${active}`
        );
      }
      setPitch(p);
      setLoading(false);
    });
  }, [active]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const t = search.trim().toUpperCase();
    if (t) setActive(t);
  };

  return (
    <div>
      <h2 className="section-title">Pitch assessor</h2>
      <p className="section-lede">
        Translates the PCA + clustering output into a structured one-pager
        suitable for committee discussion. Pre-generate the assessment from the
        CLI (the webapp can only display assessments that have been written to
        <code> webapp/public/data/pitches/</code>):
        <br />
        <code style={{ background: "#f0f2f6", padding: "2px 6px", borderRadius: 3 }}>
          python main.py --assess TICKER
        </code>
      </p>

      <div className="card" style={{ padding: 14 }}>
        <form onSubmit={handleSubmit} style={{ display: "flex", gap: 8 }}>
          <input
            type="search"
            placeholder="Search a ticker (e.g. CRGY, MYRG, PRDO)…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ flex: 1, padding: "8px 12px", fontSize: 14,
                     border: "1px solid var(--border)", borderRadius: 6 }}
          />
          <button type="submit" style={{
            padding: "8px 18px", fontSize: 14, fontWeight: 500,
            background: "var(--accent)", color: "#fff", border: "none",
            borderRadius: 6, cursor: "pointer",
          }}>Open</button>
        </form>

        {index.length > 0 && (
          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase",
                           letterSpacing: 0.5, fontWeight: 600, marginBottom: 6 }}>
              Recently generated ({index.length})
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {index.slice().sort((a, b) => b.generated_at.localeCompare(a.generated_at)).map((row) => (
                <button key={row.ticker} onClick={() => { setSearch(row.ticker); setActive(row.ticker); }}
                        style={{
                          padding: "5px 10px", fontSize: 12,
                          border: "1px solid var(--border)", borderRadius: 14,
                          background: active === row.ticker ? "var(--accent)" : "#fff",
                          color: active === row.ticker ? "#fff" : "var(--text)",
                          cursor: "pointer",
                        }}
                        title={`${row.company_name} · ${row.sector} · ${row.recommendation}`}>
                  {row.ticker}
                  <span style={{
                    marginLeft: 6, padding: "0 5px", borderRadius: 8, fontSize: 10,
                    background: recommendationColor(row.recommendation),
                    color: "#fff",
                  }}>{row.recommendation.split(" ")[0]}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {pitchErr && (
        <div className="card error" style={{ marginTop: 12 }}>
          <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{pitchErr}</pre>
        </div>
      )}

      {loading && <div className="loading">Loading {active}…</div>}

      {pitch && !loading && <PitchPage pitch={pitch} meta={meta} />}
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
              {pitch.sector} / {pitch.industry || "—"}  ·  Market cap {cap}
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
            5 closest stocks in PC space.
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
                    {n.is_former_hold && <span style={badgeFormer}>FORMER</span>}
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
          <Stat label="Cluster" value={`C${pitch.cluster_id}`} sub={pitch.cluster_tier}
                tierBadge={pitch.cluster_tier} />
          <Stat label="Composite score" value={pitch.composite_risk_score.toFixed(0) + "/100"} />
          <Stat label="Trajectory" value={pitch.cluster_trajectory} />
          <Stat label="vs sector" value={pitch.sector_comparison}
                sub={`median ${fmt(pitch.sector_median_score, 1)}`} />
        </div>

        {pitch.top_risk_drivers.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase",
                           letterSpacing: 0.5, fontWeight: 600, marginBottom: 4 }}>
              Top risk drivers (percentile rank in universe)
            </div>
            {pitch.top_risk_drivers.map((d) => (
              <div key={d.feature} style={{ display: "flex", alignItems: "center",
                                              gap: 10, padding: "3px 0" }}>
                <code style={{ minWidth: 160 }}>{d.feature}</code>
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
        Generated {new Date(pitch.generated_at).toLocaleString()}
      </div>
    </div>
  );
}


function Stat({ label, value, sub, tierBadge }: { label: string; value: string; sub?: string; tierBadge?: string }) {
  return (
    <div style={{ background: "#f7f8fa", borderRadius: 6, padding: 10 }}>
      <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase",
                     letterSpacing: 0.5, fontWeight: 600 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 600, marginTop: 4 }}>
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
const badgeFormer: React.CSSProperties = {
  ...badgeHeld, background: "var(--muted)",
};
