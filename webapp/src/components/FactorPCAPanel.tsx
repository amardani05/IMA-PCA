import { useEffect, useState } from "react";
import { significanceStars } from "../lib/macro";
import { fmt } from "../lib/data";

interface ThemeBeta { beta: number; t_stat: number; p_value: number; significant_10: boolean }
interface Component {
  pc: string;
  label: string;
  variance_explained: number;
  top_loadings: { factor: string; name: string; loading: number }[];
  portfolio: ThemeBeta;
  benchmark: ThemeBeta;
  active: ThemeBeta;
}
interface Payload {
  available: boolean;
  n_obs?: number;
  n_factors?: number;
  window?: [string, string];
  cumulative_variance?: number[];
  components?: Component[];
  r_squared?: { portfolio: number; benchmark: number; active: number };
}

/**
 * PCA on the factor library — the "exposures we didn't think we had" panel.
 * Extracts orthogonal macro themes from the daily factor panel and projects
 * portfolio / index / ACTIVE returns onto each. A theme with a significant
 * active beta is a hidden active exposure, whether or not we had a name for it.
 */
export function FactorPCAPanel() {
  const [data, setData] = useState<Payload | null>(null);

  useEffect(() => {
    fetch("/data/macro/factor_pca.json")
      .then((r) => (r.ok ? r.json() : null))
      .then(setData)
      .catch(() => setData(null));
  }, []);

  if (!data || !data.available || !data.components) return null;

  const comps = data.components;
  const cumVar = data.cumulative_variance?.[comps.length - 1] ?? 0;
  const hidden = comps.filter((c) => c.active.significant_10);

  const bp = (b: ThemeBeta) => `${b.beta >= 0 ? "+" : ""}${(b.beta * 1e4).toFixed(1)} bp`;

  return (
    <div className="card">
      <h3 style={{ margin: 0 }}>Macro theme PCA — exposures we didn't name</h3>
      <div className="card-sub">
        PCA across the {data.n_factors}-factor daily panel extracts the
        orthogonal macro <em>themes</em> that actually drive it ({(cumVar * 100).toFixed(0)}%
        of factor variance in {comps.length} themes). Each theme is then
        regressed against portfolio, index, and <strong>active</strong> returns
        (per +1σ daily theme move, HAC errors). This is the complement to the
        named-factor table: it surfaces correlated bundles of macro moves that
        explain active returns whether or not we thought to ask about them.
        {" "}
        {hidden.length > 0 ? (
          <strong style={{ color: "var(--danger)" }}>
            {hidden.length} theme{hidden.length > 1 ? "s" : ""} carr{hidden.length > 1 ? "y" : "ies"} significant
            active exposure: {hidden.map((c) => c.pc).join(", ")}.
          </strong>
        ) : (
          <strong style={{ color: "var(--ok)" }}>
            No theme carries significant active exposure right now.
          </strong>
        )}
      </div>

      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Theme</th>
              <th>Dominant drivers</th>
              <th className="num">Var</th>
              <th className="num">Index β</th>
              <th className="num">Portfolio β</th>
              <th className="num">Active β</th>
            </tr>
          </thead>
          <tbody>
            {comps.map((c) => (
              <tr key={c.pc} className={c.active.significant_10 ? "highlight" : undefined}>
                <td><strong>{c.pc}</strong></td>
                <td>
                  <small>
                    {c.top_loadings.slice(0, 3).map((l, i) => (
                      <span key={l.factor}>
                        {i > 0 && " · "}
                        {l.loading >= 0 ? "+" : "−"}{l.name}
                      </span>
                    ))}
                  </small>
                </td>
                <td className="num">{(c.variance_explained * 100).toFixed(0)}%</td>
                <td className="num">
                  {bp(c.benchmark)}<small> {significanceStars(c.benchmark.p_value)}</small>
                </td>
                <td className="num">
                  {bp(c.portfolio)}<small> {significanceStars(c.portfolio.p_value)}</small>
                </td>
                <td className="num" style={{
                  fontWeight: c.active.significant_10 ? 700 : 400,
                  color: c.active.significant_10
                    ? (c.active.beta >= 0 ? "var(--danger)" : "var(--ok)") : undefined,
                }}>
                  {bp(c.active)}<small> {significanceStars(c.active.p_value)}</small>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <small className="muted">
        Window {data.window?.[0]} → {data.window?.[1]} · n = {data.n_obs} ·
        active-return R² vs themes = {fmt(data.r_squared?.active ?? null, 3)} ·
        thematic/single-name proxies excluded so themes stay macro
      </small>
    </div>
  );
}
