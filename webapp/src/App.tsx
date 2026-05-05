import { useEffect, useState } from "react";
import { loadAll } from "./lib/data";
import {
  ClusterMeta, ClusterRow, DriftRow, Meta, OpportunityRow,
  PCALoadingRow, PCASummaryRow, PortfolioRow, TrajectoryData, UniverseRow,
} from "./lib/types";
import { Overview } from "./components/Overview";
import { UniverseView } from "./components/UniverseView";
import { PortfolioView } from "./components/PortfolioView";
import { OpportunitiesView } from "./components/OpportunitiesView";
import { DriftView } from "./components/DriftView";
import { GalleryView } from "./components/GalleryView";
import { MacroView } from "./components/MacroView";
import { PitchView } from "./components/PitchView";

type Tab =
  | "overview" | "universe" | "portfolio"
  | "macro" | "opportunities" | "drift" | "pitch" | "gallery";

const TABS: { key: Tab; label: string }[] = [
  { key: "overview",      label: "Overview" },
  { key: "portfolio",     label: "Portfolio" },
  { key: "universe",      label: "Universe" },
  { key: "pitch",         label: "Pitch Assessor" },
  { key: "macro",         label: "Macro Exposures" },
  { key: "opportunities", label: "Opportunities" },
  { key: "drift",         label: "Drift Alerts" },
  { key: "gallery",       label: "Chart Gallery" },
];

interface Bundle {
  meta: Meta;
  universe: UniverseRow[];
  portfolio: PortfolioRow[];
  clusters: ClusterRow[];
  clusterMeta: ClusterMeta;
  pcaSummary: PCASummaryRow[];
  pcaLoadings: PCALoadingRow[];
  opportunities: OpportunityRow[];
  drift: DriftRow[];
  trajectory: TrajectoryData;
}

export default function App() {
  const [data, setData] = useState<Bundle | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");

  useEffect(() => {
    loadAll().then(setData).catch((e) => setErr(String(e)));
  }, []);

  if (err) {
    return (
      <div className="app">
        <div className="top-bar"><h1>IMA Principle Component Analysis</h1></div>
        <div className="content">
          <div className="card error">
            <strong>Failed to load pipeline data.</strong>
            <p>{err}</p>
            <p className="muted">
              Run <code>python main.py --no-trajectory</code> from the repo root first,
              then reload this page. The webapp reads from <code>webapp/public/data/*.json</code>,
              which is populated by the <code>webapp_export</code> step.
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="app">
        <div className="top-bar"><h1>IMA Principle Component Analysis</h1></div>
        <div className="loading">Loading pipeline data…</div>
      </div>
    );
  }

  return (
    <div className="app">
      <div className="top-bar">
        <h1>IMA Principle Component Analysis</h1>
        <span className="sub">
          {data.meta.universe_size} S&amp;P 600 stocks ·
          k={data.meta.clustering.k} · silhouette {data.meta.clustering.silhouette.toFixed(3)} ·
          generated {new Date(data.meta.generated_at).toLocaleString()}
        </span>
      </div>
      <nav className="nav">
        {TABS.map((t) => (
          <button key={t.key}
                  className={tab === t.key ? "active" : ""}
                  onClick={() => setTab(t.key)}>
            {t.label}
          </button>
        ))}
      </nav>
      <main className="content">
        {tab === "overview"     && <Overview
                                             meta={data.meta}
                                             portfolio={data.portfolio}
                                             universe={data.universe}
                                             clusterMeta={data.clusterMeta}
                                             trajectory={data.trajectory}
                                             pcaSummary={data.pcaSummary}
                                             pcaLoadings={data.pcaLoadings}
                                             clusters={data.clusters} />}
        {tab === "universe"     && <UniverseView meta={data.meta} universe={data.universe} />}
        {tab === "portfolio"    && <PortfolioView meta={data.meta} portfolio={data.portfolio} />}
        {tab === "macro"        && <MacroView portfolio={data.portfolio} universe={data.universe} />}
        {tab === "pitch"        && <PitchView meta={data.meta} />}
        {tab === "opportunities"&& <OpportunitiesView opportunities={data.opportunities} />}
        {tab === "drift"        && <DriftView drift={data.drift} />}
        {tab === "gallery"      && <GalleryView />}
      </main>
    </div>
  );
}
