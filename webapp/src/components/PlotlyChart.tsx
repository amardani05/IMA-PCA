import { useEffect, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
// @ts-ignore — plotly.js-dist-min has no types but is the browser bundle
import Plotly from "plotly.js-dist-min";
import { loadPlotlyFigure } from "../lib/data";

const Plot = createPlotlyComponent(Plotly);

interface Props {
  name: string;       // filename stem of the figure, e.g. "scatter_pc1_pc2"
  height?: number;
}

export function PlotlyChart({ name, height = 640 }: Props) {
  const [fig, setFig] = useState<any | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setFig(null);
    setErr(null);
    loadPlotlyFigure(name)
      .then(setFig)
      .catch((e) => setErr(String(e)));
  }, [name]);

  if (err) return <div className="error">Failed to load {name}: {err}</div>;
  if (!fig) return <div className="loading">Loading chart…</div>;

  return (
    <div className="plotly-wrap">
      <Plot
        data={fig.data}
        layout={{ ...fig.layout, autosize: true, height }}
        config={{ displayModeBar: true, responsive: true }}
        style={{ width: "100%", height }}
        useResizeHandler
      />
    </div>
  );
}
