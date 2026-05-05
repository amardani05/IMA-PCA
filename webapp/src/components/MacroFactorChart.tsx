import { useMemo, useState } from "react";
// @ts-ignore
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { MacroBundle } from "../lib/macroTypes";
import { CATEGORY_LABELS } from "../lib/macro";

const Plot = createPlotlyComponent(Plotly);

interface Props {
  macro: MacroBundle;
}

export function MacroFactorChart({ macro }: Props) {
  const factorMeta = useMemo(
    () => Object.fromEntries(macro.metadata.factors.map((f) => [f.factor, f])),
    [macro],
  );

  // Default: top 4 factors by |abs beta|
  const ranked = useMemo(() => {
    return [...macro.portfolioBetas.factors].sort((a, b) =>
      Math.abs(macro.portfolioBetas.betas[b].beta) -
      Math.abs(macro.portfolioBetas.betas[a].beta)
    );
  }, [macro]);

  const [enabled, setEnabled] = useState<Set<string>>(
    () => new Set(ranked.slice(0, 4)),
  );

  const toggle = (f: string) => {
    setEnabled((prev) => {
      const n = new Set(prev);
      if (n.has(f)) n.delete(f);
      else n.add(f);
      return n;
    });
  };

  const traces = useMemo(() => {
    const dates = macro.rollingBetas.dates;
    const palette = ["#1f3b73", "#b3001b", "#2c7a4b", "#f0a202", "#7e57c2", "#0288d1", "#e57a44"];
    return Array.from(enabled).map((f, i) => ({
      x: dates,
      y: macro.rollingBetas.series[f] ?? [],
      type: "scatter" as const,
      mode: "lines" as const,
      name: factorMeta[f]?.name ?? f,
      line: { color: palette[i % palette.length], width: 2 },
      hovertemplate: "%{x|%Y-%m-%d}<br>β = %{y:.3f}<extra></extra>",
    }));
  }, [enabled, macro, factorMeta]);

  return (
    <div className="card">
      <h3 style={{ margin: 0 }}>Rolling 60-day betas</h3>
      <div className="card-sub">
        Click factors below to add/remove them from the chart. Default: top 4
        by absolute beta magnitude. Use this to see whether exposures are
        stable or drifting.
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 8 }}>
        {ranked.map((f) => {
          const meta = factorMeta[f];
          const active = enabled.has(f);
          return (
            <button key={f} onClick={() => toggle(f)} style={{
              padding: "3px 8px", fontSize: 11,
              border: "1px solid var(--border)", borderRadius: 12, cursor: "pointer",
              background: active ? "var(--accent)" : "#fff",
              color: active ? "#fff" : "var(--text)",
            }}>
              {meta?.name ?? f}
              <small style={{ marginLeft: 4, opacity: 0.7 }}>
                ({CATEGORY_LABELS[meta?.category] ?? "?"})
              </small>
            </button>
          );
        })}
      </div>

      {traces.length === 0 ? (
        <div className="muted" style={{ padding: 30, textAlign: "center" }}>
          Select at least one factor.
        </div>
      ) : (
        <Plot
          data={traces}
          layout={{
            autosize: true,
            height: 540,
            margin: { l: 50, r: 20, t: 10, b: 40 },
            template: "plotly_white" as any,
            shapes: [{
              type: "line", x0: macro.rollingBetas.dates[0],
              x1: macro.rollingBetas.dates[macro.rollingBetas.dates.length - 1],
              y0: 0, y1: 0, line: { color: "#999", width: 0.8 },
            }],
            xaxis: { showgrid: true },
            yaxis: { title: { text: "Beta" } as any, showgrid: true },
            legend: { orientation: "h", y: -0.18 },
            hovermode: "x unified",
          }}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      )}
    </div>
  );
}
