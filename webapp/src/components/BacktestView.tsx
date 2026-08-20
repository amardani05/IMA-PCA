import { useEffect, useMemo, useState } from "react";
// @ts-ignore — plotly.js-dist-min has no types but is the browser bundle
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { BacktestData, HitRateCell } from "../lib/types";
import { loadBacktest, fmt, fmtPct, tierClass } from "../lib/data";
import { Column, DataTable } from "./DataTable";

const Plot = createPlotlyComponent(Plotly);

const TIER_COLOR: Record<string, string> = {
  "Low Risk": "#2c7a4b", "In Line": "#64748b", "Elevated": "#b3001b",
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true } as const;
const BASE_LAYOUT = {
  autosize: true, template: "plotly_white" as any,
  margin: { l: 55, r: 20, t: 30, b: 45 },
};

export function BacktestView() {
  const [data, setData] = useState<BacktestData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    loadBacktest().then(setData).catch((e) => setErr(String(e)));
  }, []);

  if (err) {
    return (
      <div>
        <h2 className="section-title">Backtest</h2>
        <div className="card error">
          <strong>No backtest data yet.</strong>
          <p>{err}</p>
          <p className="muted">
            Run <code>python main.py --backtest</code> from the repo root. With no
            point-in-time store under <code>data/historical/</code> it generates a
            synthetic one so the engine runs end-to-end; feed real parquet files
            (see the README "Historical data" schema) for a meaningful run. The
            export writes <code>webapp/public/data/backtest.json</code>.
          </p>
        </div>
      </div>
    );
  }
  if (!data) return <div className="loading">Loading backtest…</div>;

  return (
    <div>
      <h2 className="section-title">Does the screener actually predict severe drawdowns?</h2>
      <Disclosure data={data} />
      <BaseRateStrip data={data} />
      <DecileChart data={data} />
      <TierChart data={data} />
      <ICChart data={data} />
      <ROCChart data={data} />
      <CalibrationChart data={data} />
      <StrategyChart data={data} />
      <IMASection data={data} />
    </div>
  );
}

// ---------------------------------------------------------------------------
function Disclosure({ data }: { data: BacktestData }) {
  const m = data.metadata;
  return (
    <div className="card" style={{
      borderLeft: "4px solid #1f3b73", background: "#f4f7fb",
    }}>
      <h3 style={{ marginTop: 0 }}>Read this first — what the numbers mean</h3>
      {m.synthetic_store && (
        <div style={{ background: "#fdeeee", border: "1px solid #b3001b",
                      color: "#8c1d18", borderRadius: 6, padding: "10px 14px",
                      fontWeight: 600, marginBottom: 10 }}>
          SYNTHETIC DATA — no real point-in-time fundamentals store has been
          supplied, so every number on this tab comes from a simulated history.
          It validates that the engine works, NOT that the screener predicts
          anything. Do not cite these figures in committee.
        </div>
      )}
      <ul style={{ margin: "6px 0", lineHeight: 1.6 }}>
        <li><strong>Label:</strong> {m.label_definition}</li>
        <li>
          <strong>Survivorship:</strong>{" "}
          {m.survivorship_safe ? (
            <span>point-in-time index membership is used — delisted names are
              retained, so terminal blow-ups are counted, not dropped.</span>
          ) : (
            <span style={{ color: "#b3001b", fontWeight: 600 }}>
              NOT survivorship-safe — no point-in-time universe was supplied, so
              the panel uses current/feature-set membership and event rates are
              biased low. Supply universe.parquet to fix.
            </span>
          )}
        </li>
        <li>
          <strong>Costs:</strong> strategy returns are net of {fmt(m.cost_bps, 0)} bps
          one-way transaction cost applied to turnover; the benchmark is an
          equal-weight universe (IJR proxy) unless a real IJR price series is fed.
        </li>
        <li>
          <strong>Window:</strong> {m.date_range[0]} → {m.date_range[1]} ·{" "}
          {m.n_snapshots} {m.rebalance}-rebalance snapshots ·{" "}
          {data.base_rate.n_events} severe-drawdown events.
        </li>
      </ul>
    </div>
  );
}

function BaseRateStrip({ data }: { data: BacktestData }) {
  const br = data.base_rate;
  const ic = data.information_coefficient;
  const tiers = data.tiers;
  const stat = (label: string, value: string, sub?: string) => (
    <div className="card" style={{ flex: 1, minWidth: 150, textAlign: "center" }}>
      <div className="muted" style={{ fontSize: 12 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700 }}>{value}</div>
      {sub && <div className="muted" style={{ fontSize: 11 }}>{sub}</div>}
    </div>
  );
  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 4 }}>
      {stat("Base rate", fmtPct((br.base_rate ?? 0)), `${br.n_events} events / ${br.n_observations} obs`)}
      {stat("Events / year", fmt(br.events_per_year, 0), `over ${br.n_snapshots} snapshots`)}
      {stat("AUC", data.classification.auc == null ? "—" : fmt(data.classification.auc, 3),
        "score → severe-DD label")}
      {stat("IC vs max-DD", ic.ic_maxdd_mean == null ? "—" : fmt(ic.ic_maxdd_mean, 3),
        `t = ${fmt(ic.ic_maxdd_tstat, 2)} (NW)`)}
      {stat("Monotonic?", tiers.monotonic ? "YES" : "NO",
        `Elevated−Low ${fmtPct(tiers.elevated_minus_low)}`)}
    </div>
  );
}

// ---------------------------------------------------------------------------
function DecileChart({ data }: { data: BacktestData }) {
  const pooled = data.deciles.pooled;
  const x = pooled.map((d) => `D${d.decile}`);
  const trace = {
    x, y: pooled.map((d) => d.rate), type: "bar" as const,
    marker: { color: pooled.map((d) => d.thin ? "#bbb" : "#1f3b73") },
    error_y: {
      type: "data" as const, symmetric: false,
      array: pooled.map((d) => d.ci_high - d.rate),
      arrayminus: pooled.map((d) => d.rate - d.ci_low),
      color: "#444", thickness: 1,
    },
    hovertemplate: "%{x}<br>rate %{y:.1%}<extra></extra>",
  };
  const mono = isMonotone(pooled.map((d) => d.rate));
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Severe-drawdown rate by composite-score decile</h3>
      <div className="card-sub">
        Decile 1 = lowest risk score, Decile 10 = highest. Whiskers are Wilson 95%
        CIs; grey bars are thin-sample cells (&lt; {5} events). This is the core test —
        a working screener rises left→right.{" "}
        <strong style={{ color: mono ? "#2c7a4b" : "#b3001b" }}>
          {mono ? "✓ Monotonically increasing across deciles."
                : "✗ Not monotonic — risk ordering is imperfect."}
        </strong>
      </div>
      <Plot data={[trace]} layout={{
        ...BASE_LAYOUT, height: 380,
        xaxis: { title: { text: "Composite-score decile" } as any },
        yaxis: { title: { text: "Realized 6m severe-DD rate" } as any, tickformat: ".0%" },
        shapes: [hline(data.base_rate.base_rate ?? 0)],
        annotations: [baseRateAnno(data.base_rate.base_rate ?? 0, x[0])],
      }} config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }} />
    </div>
  );
}

function TierChart({ data }: { data: BacktestData }) {
  const pooled = data.tiers.pooled;
  const trace = {
    x: pooled.map((t) => t.tier!), y: pooled.map((t) => t.rate),
    type: "bar" as const,
    marker: { color: pooled.map((t) => TIER_COLOR[t.tier!] ?? "#777") },
    error_y: {
      type: "data" as const, symmetric: false,
      array: pooled.map((t) => t.ci_high - t.rate),
      arrayminus: pooled.map((t) => t.rate - t.ci_low),
      color: "#444", thickness: 1,
    },
    text: pooled.map((t) => `${(t.rate * 100).toFixed(1)}%  (n=${t.n})`),
    textposition: "outside" as const,
    hovertemplate: "%{x}<br>rate %{y:.1%}<extra></extra>",
  };
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Severe-drawdown rate by risk tier</h3>
      <div className="card-sub">
        The committee-facing version of the decile chart.{" "}
        <strong style={{ color: data.tiers.monotonic ? "#2c7a4b" : "#b3001b" }}>
          {data.tiers.monotonic
            ? "✓ Elevated > In Line > Low Risk holds — the tiers are ordered by realized risk."
            : "✗ Tier ordering is violated in-sample."}
        </strong>{" "}
        Elevated−Low spread = {fmtPct(data.tiers.elevated_minus_low)}.
      </div>
      <Plot data={[trace]} layout={{
        ...BASE_LAYOUT, height: 360,
        yaxis: { title: { text: "Realized 6m severe-DD rate" } as any, tickformat: ".0%" },
        shapes: [hline(data.base_rate.base_rate ?? 0)],
      }} config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }} />
    </div>
  );
}

function ICChart({ data }: { data: BacktestData }) {
  const ts = data.information_coefficient.time_series;
  const x = ts.map((p) => p.date);
  const traces = [
    {
      x, y: ts.map((p) => p.ic_maxdd), name: "IC vs forward max-DD",
      type: "scatter" as const, mode: "lines+markers" as const,
      line: { color: "#b3001b", width: 2 },
    },
    {
      x, y: ts.map((p) => p.ic_return), name: "IC vs forward return",
      type: "scatter" as const, mode: "lines+markers" as const,
      line: { color: "#1f3b73", width: 2 },
    },
  ];
  const ic = data.information_coefficient;
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Information coefficient over time</h3>
      <div className="card-sub">
        Per-snapshot Spearman rank correlation of composite score with the forward
        outcome. Positive vs max-DD (high score → deeper drawdown) and negative vs
        return (high score → worse return) are both "working". Fama-MacBeth means
        with Newey-West t-stats:{" "}
        <strong>max-DD {fmt(ic.ic_maxdd_mean, 3)} (t={fmt(ic.ic_maxdd_tstat, 2)})</strong>,{" "}
        return {fmt(ic.ic_return_mean, 3)} (t={fmt(ic.ic_return_tstat, 2)}).
      </div>
      <Plot data={traces} layout={{
        ...BASE_LAYOUT, height: 360,
        yaxis: { title: { text: "Spearman IC" } as any, zeroline: true },
        legend: { orientation: "h", y: -0.2 },
        hovermode: "x unified",
        shapes: [hline(0, "#999")],
      }} config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }} />
    </div>
  );
}

function ROCChart({ data }: { data: BacktestData }) {
  const roc = data.classification.roc;
  if (!roc.length) {
    return (
      <div className="card">
        <h3 style={{ marginTop: 0 }}>ROC curve</h3>
        <p className="muted">ROC undefined (single class in-sample).</p>
      </div>
    );
  }
  const traces = [
    {
      x: roc.map((p) => p.fpr), y: roc.map((p) => p.tpr),
      type: "scatter" as const, mode: "lines" as const,
      line: { color: "#1f3b73", width: 2.5 }, name: "screener",
      hovertemplate: "FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>",
    },
    {
      x: [0, 1], y: [0, 1], type: "scatter" as const, mode: "lines" as const,
      line: { color: "#999", width: 1, dash: "dash" as const }, name: "random",
      hoverinfo: "skip" as const,
    },
  ];
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>
        ROC — composite score as a severe-drawdown classifier (AUC ={" "}
        {fmt(data.classification.auc, 3)})
      </h3>
      <div className="card-sub">
        AUC 0.5 is coin-flip; higher means the score separates future severe
        drawdowns from the rest. Precision / recall / lift at each tier floor below.
      </div>
      <Plot data={traces} layout={{
        ...BASE_LAYOUT, height: 380, width: undefined,
        xaxis: { title: { text: "False positive rate" } as any, range: [0, 1] },
        yaxis: { title: { text: "True positive rate" } as any, range: [0, 1],
                 scaleanchor: "x" as any, scaleratio: 1 },
        legend: { orientation: "h", y: -0.2 },
      }} config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }} />
      <table className="mini-table" style={{ width: "100%", marginTop: 8, fontSize: 13 }}>
        <thead><tr>
          <th style={th}>Tier floor</th><th style={th}>Flagged</th>
          <th style={th}>Precision</th><th style={th}>Recall</th><th style={th}>Lift</th>
        </tr></thead>
        <tbody>
          {data.classification.tier_thresholds.map((t) => (
            <tr key={t.tier}>
              <td style={td}><span className={tierClass(t.tier)}>{t.tier}</span> (≥{t.score_floor})</td>
              <td style={td}>{t.n_flagged}</td>
              <td style={td}>{fmtPct(t.precision)}</td>
              <td style={td}>{fmtPct(t.recall)}</td>
              <td style={td}>{t.lift == null ? "—" : `${fmt(t.lift, 2)}×`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CalibrationChart({ data }: { data: BacktestData }) {
  const rel = data.calibration.reliability;
  const traces = [
    {
      x: rel.map((b) => b.predicted), y: rel.map((b) => b.realized),
      type: "scatter" as const, mode: "lines+markers" as const,
      line: { color: "#2c7a4b", width: 2 }, name: "screener",
      error_y: {
        type: "data" as const, symmetric: false,
        array: rel.map((b) => b.ci_high - b.realized),
        arrayminus: rel.map((b) => b.realized - b.ci_low),
        color: "#888", thickness: 1,
      },
      hovertemplate: "pred %{x:.2f}<br>realized %{y:.2f}<extra></extra>",
    },
    {
      x: [0, Math.max(...rel.map((b) => b.predicted), 0.5)],
      y: [0, Math.max(...rel.map((b) => b.predicted), 0.5)],
      type: "scatter" as const, mode: "lines" as const,
      line: { color: "#999", width: 1, dash: "dash" as const }, name: "perfect",
      hoverinfo: "skip" as const,
    },
  ];
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Reliability / calibration</h3>
      <div className="card-sub">
        Predicted (mean score/100 within each score decile) vs realized event
        rate, with Wilson CIs. Points near the dashed diagonal mean the score is
        well-calibrated as a probability; below it means the score over-warns.
      </div>
      <Plot data={traces} layout={{
        ...BASE_LAYOUT, height: 360,
        xaxis: { title: { text: "Predicted (score / 100)" } as any },
        yaxis: { title: { text: "Realized event rate" } as any, tickformat: ".0%" },
        legend: { orientation: "h", y: -0.2 },
      }} config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }} />
    </div>
  );
}

function StrategyChart({ data }: { data: BacktestData }) {
  const s = data.strategy;
  if (!s.available || !s.equity_curve) {
    return null;
  }
  const c = s.equity_curve;
  const x = c.map((p) => p.date);
  const traces = [
    { x, y: c.map((p) => p.benchmark), name: s.benchmark_label ?? "Benchmark",
      type: "scatter" as const, mode: "lines" as const, line: { color: "#777", width: 2 } },
    { x, y: c.map((p) => p.avoid_top_tier), name: "Avoid top-tier (long-only)",
      type: "scatter" as const, mode: "lines" as const, line: { color: "#2c7a4b", width: 2 } },
    { x, y: c.map((p) => p.long_short), name: "Sector-neutral L/S",
      type: "scatter" as const, mode: "lines" as const, line: { color: "#1f3b73", width: 2 } },
  ];
  const metricRow = (label: string, st?: any) => st && (
    <tr>
      <td style={td}>{label}</td>
      <td style={td}>{fmtPct(st.cagr)}</td>
      <td style={td}>{fmt(st.sharpe, 2)}</td>
      <td style={td}>{fmtPct(st.vol)}</td>
      <td style={td}>{fmtPct(st.max_drawdown)}</td>
      <td style={td}>{st.avg_turnover == null ? "—" : fmt(st.avg_turnover, 2)}</td>
      <td style={td}>{st.hit_rate_vs_bench == null ? "—" : fmtPct(st.hit_rate_vs_bench)}</td>
    </tr>
  );
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Strategy backtest (net of {fmt(s.cost_bps, 0)} bps)</h3>
      <div className="card-sub">
        Long-only "avoid the Elevated tier" and a sector-neutral long-Low-Risk /
        short-Elevated sleeve, {s.rebalance}-rebalanced. Equity curves are growth
        of $1 net of turnover cost. Turnover is reported — read returns with the
        cost + survivorship caveats in the disclosure box above.
      </div>
      <Plot data={traces} layout={{
        ...BASE_LAYOUT, height: 380,
        yaxis: { title: { text: "Growth of $1" } as any },
        legend: { orientation: "h", y: -0.2 }, hovermode: "x unified",
      }} config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }} />
      <table className="mini-table" style={{ width: "100%", marginTop: 8, fontSize: 13 }}>
        <thead><tr>
          <th style={th}>Sleeve</th><th style={th}>CAGR</th><th style={th}>Sharpe</th>
          <th style={th}>Vol</th><th style={th}>Max DD</th><th style={th}>Turnover</th>
          <th style={th}>Hit vs bench</th>
        </tr></thead>
        <tbody>
          {metricRow(s.benchmark_label ?? "Benchmark", s.benchmark)}
          {metricRow("Avoid top-tier", s.avoid_top_tier)}
          {metricRow("Sector-neutral L/S", s.long_short)}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
function IMASection({ data }: { data: BacktestData }) {
  const ima = data.ima;
  if (!ima.available) {
    return (
      <div className="card">
        <h3 style={{ marginTop: 0 }}>IMA portfolio</h3>
        <p className="muted">
          No <code>ima_holdings.parquet</code> in the historical store, so the
          portfolio-specific evaluation is unavailable.
        </p>
      </div>
    );
  }
  const hm = ima.hit_miss ?? [];
  const c = ima.confusion!;
  const cf = ima.counterfactual;

  const cols: Column<NonNullable<typeof ima.hit_miss>[number]>[] = [
    { key: "date", header: "Date", accessor: (r) => r.date },
    { key: "ticker", header: "Ticker", accessor: (r) => r.ticker,
      render: (r) => <strong>{r.ticker}</strong> },
    { key: "score", header: "Score", numeric: true, accessor: (r) => r.score },
    { key: "tier", header: "Tier", accessor: (r) => r.tier,
      render: (r) => <span className={tierClass(r.tier)}>{r.tier}</span> },
    { key: "fwd_maxdd", header: "Fwd max-DD", numeric: true,
      accessor: (r) => r.fwd_maxdd, render: (r) => fmtPct(r.fwd_maxdd) },
    { key: "severe_dd", header: "Severe DD?", accessor: (r) => (r.severe_dd ? 1 : 0),
      render: (r) => (r.severe_dd ? <span style={{ color: "#b3001b", fontWeight: 700 }}>yes</span> : "no") },
    { key: "outcome", header: "Outcome", accessor: (r) => r.outcome,
      render: (r) => <OutcomeBadge outcome={r.outcome} /> },
  ];

  const scoreTS = ima.score_time_series ?? [];
  const scoreTrace = [
    { x: scoreTS.map((p) => p.date), y: scoreTS.map((p) => p.ima_mean_score),
      name: "IMA holdings", type: "scatter" as const, mode: "lines+markers" as const,
      line: { color: "#1f3b73", width: 2 } },
    { x: scoreTS.map((p) => p.date), y: scoreTS.map((p) => p.universe_mean_score),
      name: "Universe", type: "scatter" as const, mode: "lines" as const,
      line: { color: "#999", width: 2, dash: "dot" as const } },
  ];

  return (
    <div>
      <h2 className="section-title" style={{ marginTop: 28 }}>IMA portfolio: did the model flag what later blew up?</h2>

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <ConfusionCard c={c} />
        {cf?.available && (
          <div className="card" style={{ flex: 1, minWidth: 260 }}>
            <h3 style={{ marginTop: 0 }}>Counterfactual: avoid the Elevated holdings</h3>
            <p className="card-sub">
              Drop the model's top-tier names from the sleeve each rebalance and
              renormalize. Effect on the realized sleeve:
            </p>
            <div style={{ display: "flex", gap: 18 }}>
              <Delta label="Δ CAGR" v={cf.delta_cagr} good={(cf.delta_cagr ?? 0) > 0} />
              <Delta label="Δ Max-DD" v={cf.delta_maxdd} good={(cf.delta_maxdd ?? 0) < 0} />
            </div>
            <p className="muted" style={{ fontSize: 11, marginBottom: 0 }}>
              Negative Δ max-DD = the screen would have shrunk the worst drawdown.
            </p>
          </div>
        )}
      </div>

      {scoreTS.length > 0 && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Model score: IMA holdings vs universe</h3>
          <div className="card-sub">
            Average composite score on the holdings the IMA actually owned vs the
            full universe at each date. Above the dotted line = the sleeve was
            carrying more screener-measured risk than the market.
          </div>
          <Plot data={scoreTrace} layout={{
            ...BASE_LAYOUT, height: 320,
            yaxis: { title: { text: "Mean composite score" } as any },
            legend: { orientation: "h", y: -0.25 }, hovermode: "x unified",
          }} config={PLOT_CONFIG} useResizeHandler style={{ width: "100%" }} />
        </div>
      )}

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <NamedList title={`Caught events (${ima.caught_events?.length ?? 0})`}
                   color="#2c7a4b" rows={ima.caught_events ?? []} />
        <NamedList title={`Missed events (${ima.missed_events?.length ?? 0})`}
                   color="#b3001b" rows={ima.missed_events ?? []} />
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Per-holding hit / miss ledger</h3>
        <div className="card-sub">
          Every IMA holding at every rebalance: the model's score/tier at hold-date
          vs the drawdown that followed.
        </div>
        <DataTable rows={hm} columns={cols} initialSortKey="fwd_maxdd" pageSize={20} />
      </div>
    </div>
  );
}

function ConfusionCard({ c }: { c: NonNullable<BacktestData["ima"]["confusion"]> }) {
  const cell = (label: string, v: number, bg: string) => (
    <div style={{ background: bg, padding: "10px 8px", borderRadius: 6, textAlign: "center" }}>
      <div style={{ fontSize: 11 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700 }}>{v}</div>
    </div>
  );
  return (
    <div className="card" style={{ flex: 1, minWidth: 260 }}>
      <h3 style={{ marginTop: 0 }}>Confusion matrix (flagged Elevated × severe drawdown)</h3>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        {cell("Caught (TP)", c.true_positive, "#cdeccd")}
        {cell("False alarm (FP)", c.false_positive, "#fde2c4")}
        {cell("Missed (FN)", c.false_negative, "#f6c5cb")}
        {cell("Cleared (TN)", c.true_negative, "#e4e7eb")}
      </div>
    </div>
  );
}

function NamedList({ title, color, rows }: {
  title: string; color: string;
  rows: { date: string; ticker: string; score: number; fwd_maxdd: number | null }[];
}) {
  return (
    <div className="card" style={{ flex: 1, minWidth: 260 }}>
      <h3 style={{ marginTop: 0, color }}>{title}</h3>
      {rows.length === 0 ? <p className="muted">None.</p> : (
        <table style={{ width: "100%", fontSize: 13 }}>
          <thead><tr>
            <th style={th}>Date</th><th style={th}>Ticker</th>
            <th style={th}>Score</th><th style={th}>Max-DD</th>
          </tr></thead>
          <tbody>
            {rows.slice(0, 12).map((r, i) => (
              <tr key={i}>
                <td style={td}>{r.date}</td>
                <td style={td}><strong>{r.ticker}</strong></td>
                <td style={td}>{fmt(r.score, 0)}</td>
                <td style={td}>{fmtPct(r.fwd_maxdd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function OutcomeBadge({ outcome }: { outcome: string }) {
  const map: Record<string, string> = {
    "CAUGHT (true positive)": "#2c7a4b",
    "MISSED (false negative)": "#b3001b",
    "false alarm": "#d4a017",
    "correctly cleared": "#777",
  };
  return (
    <span style={{
      background: map[outcome] ?? "#777", color: "#fff",
      padding: "1px 8px", borderRadius: 10, fontSize: 11, fontWeight: 600,
    }}>{outcome}</span>
  );
}

function Delta({ label, v, good }: { label: string; v: number | null; good: boolean }) {
  return (
    <div>
      <div className="muted" style={{ fontSize: 11 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: v == null ? "#777" : good ? "#2c7a4b" : "#b3001b" }}>
        {v == null ? "—" : fmtPct(v)}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
function isMonotone(arr: number[]): boolean {
  return arr.every((v, i) => i === 0 || v >= arr[i - 1] - 1e-9);
}
function hline(y: number, color = "#b3001b") {
  return { type: "line" as const, xref: "paper" as const, x0: 0, x1: 1, y0: y, y1: y,
           line: { color, width: 1, dash: "dot" as const } };
}
function baseRateAnno(y: number, x0: string) {
  return { x: x0, y, xref: "x" as const, yref: "y" as const,
           text: `base rate ${(y * 100).toFixed(1)}%`, showarrow: false,
           font: { size: 10, color: "#b3001b" }, yshift: 10, xanchor: "left" as const };
}

const th: React.CSSProperties = {
  textAlign: "left", borderBottom: "1px solid var(--border)", padding: "4px 6px", fontWeight: 600,
};
const td: React.CSSProperties = { padding: "3px 6px", borderBottom: "1px solid #f0f0f0" };
