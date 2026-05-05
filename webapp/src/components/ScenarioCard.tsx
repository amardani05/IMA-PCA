import { useMemo } from "react";
import { computeScenarioImpacts } from "../lib/macro";
import { MacroBundle } from "../lib/macroTypes";

interface Props {
  macro: MacroBundle;
  liveBetas: Record<string, number>;
}

export function ScenarioCard({ macro, liveBetas }: Props) {
  const impacts = useMemo(() => {
    const shocks = macro.metadata.scenario_shocks;
    return computeScenarioImpacts(liveBetas, shocks).slice(0, 8);
  }, [liveBetas, macro]);

  const factorByName = useMemo(
    () => Object.fromEntries(macro.metadata.factors.map((f) => [f.factor, f])),
    [macro],
  );

  return (
    <div className="card">
      <h3 style={{ margin: 0 }}>Scenario sensitivity</h3>
      <div className="card-sb card-sub">
        Beta × shock = portfolio impact, all else equal. Updates live with tree selection.
        These are sensitivities, NOT predictions.
      </div>
      <table className="data" style={{ marginTop: 6 }}>
        <thead>
          <tr><th>Factor</th><th>Shock</th><th className="num">Impact</th></tr>
        </thead>
        <tbody>
          {impacts.map((sc) => {
            const name = factorByName[sc.factor]?.name ?? sc.factor;
            const color = sc.impact >= 0 ? "var(--ok)" : "var(--danger)";
            const pct = (sc.impact * 100).toFixed(2);
            return (
              <tr key={sc.factor}>
                <td>{name}</td>
                <td><small className="muted">{sc.label}</small></td>
                <td className="num" style={{ color, fontWeight: 600 }}>
                  {sc.impact >= 0 ? "+" : ""}{pct}%
                </td>
              </tr>
            );
          })}
          {impacts.length === 0 && (
            <tr><td colSpan={3} className="muted" style={{ textAlign: "center" }}>
              No scenarios available.
            </td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
