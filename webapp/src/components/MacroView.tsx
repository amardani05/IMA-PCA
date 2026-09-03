import { useEffect, useMemo, useState } from "react";
import { loadMacro } from "../lib/macro";
import { MacroBundle, RollingBetas, TimeframeCode } from "../lib/macroTypes";
import { PortfolioRow, UniverseRow } from "../lib/types";
import { MacroBetaPanel } from "./MacroBetaPanel";
import { MacroFactorChart } from "./MacroFactorChart";
import { ScenarioCard } from "./ScenarioCard";
import { PortfolioBetaTable } from "./PortfolioBetaTable";
import { TimeframeSelector } from "./TimeframeSelector";
import { IndexVsActivePanel } from "./IndexVsActivePanel";
import { FactorPCAPanel } from "./FactorPCAPanel";
import { AttributionPanel } from "./AttributionPanel";
import { fmt } from "../lib/data";

interface Props {
  portfolio: PortfolioRow[];
  universe: UniverseRow[];   // unused now, kept for API compat
}


/**
 * Slice a rolling-betas series to dates within ``[start, end]`` inclusive.
 * The series payload pairs ``dates`` with same-length per-factor arrays,
 * so we filter both in lockstep.
 */
function sliceRollingBetas(
  rolling: RollingBetas,
  startISO: string,
  endISO: string,
): RollingBetas {
  const start = startISO;
  const end = endISO;
  const idx: number[] = [];
  rolling.dates.forEach((d, i) => {
    if (d >= start && d <= end) idx.push(i);
  });
  return {
    dates: idx.map((i) => rolling.dates[i]),
    factors: rolling.factors,
    series: Object.fromEntries(
      rolling.factors.map((f) => [
        f, idx.map((i) => rolling.series[f]?.[i] ?? null),
      ]),
    ),
  };
}


export function MacroView({ portfolio }: Props) {
  const [macro, setMacro] = useState<MacroBundle | null | "loading">("loading");
  const [timeframe, setTimeframe] = useState<TimeframeCode>("max");

  useEffect(() => {
    loadMacro().then((b) => {
      setMacro(b ?? null);
      // Pick the timeframe payload's preferred default if present
      if (b && b.timeframes) setTimeframe(b.timeframes.default);
    });
  }, []);

  // Build the active-bundle view: take the loaded MacroBundle and overlay
  // the selected timeframe's per-period regression results. Keep static
  // assets (factor metadata, factor returns) untouched.
  const activeBundle: MacroBundle | null = useMemo(() => {
    if (!macro || macro === "loading") return null;
    if (!macro.timeframes) return macro;
    const tf = macro.timeframes.by_timeframe[timeframe];
    if (!tf) return macro;
    return {
      ...macro,
      portfolioBetas: tf.v2,
      portfolioBetasV1: tf.v1,
      portfolioBetasRaw: tf.raw,
      stockBetas: tf.stock_betas,
      scenarios: { scenarios: tf.scenarios },
      comparison: { rows: tf.comparison },
      rollingBetas: sliceRollingBetas(
        macro.rollingBetas, tf.date_range[0], tf.date_range[1],
      ),
    };
  }, [macro, timeframe]);

  const fittedBetas = useMemo(() => {
    if (!activeBundle) return {};
    return Object.fromEntries(
      Object.entries(activeBundle.portfolioBetas.betas).map(([f, e]) => [f, e.beta]),
    );
  }, [activeBundle]);

  if (macro === "loading") {
    return (
      <div>
        <h2 className="section-title">Factor exposures</h2>
        <div className="loading">Loading factor analysis…</div>
      </div>
    );
  }

  if (!macro) {
    return (
      <div>
        <h2 className="section-title">Factor exposures</h2>
        <div className="card error">
          <strong>Macro analysis output not available.</strong>
          <p>Run the pipeline with macro analysis enabled:</p>
          <pre style={{ background: "#f0f2f6", padding: 12, borderRadius: 4 }}>
{`# 1. Set FRED_API_KEY in .env at the repo root
echo "FRED_API_KEY=your_key_here" > .env

# 2. Re-run with macro
python main.py`}
          </pre>
          <p className="muted" style={{ marginTop: 8 }}>
            Free key: fred.stlouisfed.org/docs/api/api_key.html. Pass{" "}
            <code>--no-macro</code> to skip when the key isn't available.
          </p>
        </div>
      </div>
    );
  }

  const display = activeBundle!;
  const activeR2 = display.portfolioBetas.r_squared;
  const activeNobs = display.portfolioBetas.n_obs;
  const activeAlphaAnnualized = display.portfolioBetas.alpha * 252;
  const alphaPct = (activeAlphaAnnualized * 100).toFixed(2);
  const maxVif = display.portfolioBetas.vifs
    ? Math.max(...Object.values(display.portfolioBetas.vifs))
    : null;
  const vifFlag = (maxVif ?? 0) > 5;

  return (
    <div>
      <h2 className="section-title">Factor exposures</h2>
      <p className="section-lede">
        The factor library: daily portfolio returns regressed on transformed
        macro factors with HAC (Newey-West, 5-day) errors. Headline betas are
        ACTIVE — net of index, VIX, and credit; the "Active vs index" table
        shows what the benchmark itself carries. Per-stock betas at the bottom
        show which holdings drive each exposure.
        <br />
        <small className="muted">
          n = {activeNobs} obs ·
          R² = {fmt(activeR2, 3)} ·
          α annualized {activeAlphaAnnualized >= 0 ? "+" : ""}{alphaPct}% ·
          max VIF {fmt(maxVif, 2)}
          {vifFlag && <span style={{ color: "var(--danger)" }}> ⚠ multicollinearity</span>}
        </small>
      </p>

      {macro.timeframes && (
        <TimeframeSelector
          timeframes={macro.timeframes}
          selected={timeframe}
          onChange={setTimeframe}
        />
      )}

      <MacroFactorChart macro={display} />

      <div style={{
        display: "grid", gridTemplateColumns: "minmax(0, 2fr) minmax(0, 1fr)",
        gap: 12, marginTop: 12,
      }}>
        <MacroBetaPanel macro={display} liveBetas={fittedBetas} isModified={false} />
        <ScenarioCard macro={display} liveBetas={fittedBetas} />
      </div>

      <div style={{ marginTop: 12 }}>
        <AttributionPanel metadata={macro.metadata} />
      </div>

      <div style={{ marginTop: 12 }}>
        <IndexVsActivePanel metadata={macro.metadata} />
      </div>

      <div style={{ marginTop: 12 }}>
        <FactorPCAPanel />
      </div>

      <div style={{ marginTop: 12 }}>
        <PortfolioBetaTable macro={display} portfolio={portfolio} />
      </div>
    </div>
  );
}
