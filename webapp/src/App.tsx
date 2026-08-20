import { useEffect, useMemo, useState } from "react";
import { loadAll } from "./lib/data";
import {
  ClusterMeta, ClusterRow, DriftRow, Meta, OpportunityRow,
  PCALoadingRow, PCASummaryRow, PortfolioRow, TrajectoryData, UniverseRow,
} from "./lib/types";
import { TickerOpenContext } from "./lib/tickerContext";
import { computeFeatureRanks } from "./lib/assess";
import { Overview } from "./components/Overview";
import { UniverseView } from "./components/UniverseView";
import { PortfolioView } from "./components/PortfolioView";
import { OpportunitiesView } from "./components/OpportunitiesView";
import { DriftView } from "./components/DriftView";
import { GalleryView } from "./components/GalleryView";
import { MacroView } from "./components/MacroView";
import { PitchView } from "./components/PitchView";
import { BacktestView } from "./components/BacktestView";
import { TickerDrawer } from "./components/TickerDrawer";

type Tab =
  | "overview" | "universe" | "portfolio"
  | "macro" | "opportunities" | "drift" | "pitch" | "backtest" | "gallery";

const TABS: { key: Tab; label: string }[] = [
  { key: "overview",      label: "Overview" },
  { key: "portfolio",     label: "Portfolio" },
  { key: "universe",      label: "Universe" },
  { key: "pitch",         label: "Pitch Assessor" },
  { key: "backtest",      label: "Backtest" },
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

function FreshnessBanner({ meta }: { meta: Meta }) {
  const generated = new Date(meta.generated_at);
  const ageDays = (Date.now() - generated.getTime()) / 86_400_000;
  const stale = ageDays > 4;
  const ageLabel =
    ageDays < 1 ? "today"
    : ageDays < 2 ? "yesterday"
    : `${Math.floor(ageDays)} days ago`;
  return (
    <div className={`freshness${stale ? " stale" : ""}`}>
      <span className="dot" />
      <span>
        Data updated <strong>{ageLabel}</strong>
        {" "}({generated.toLocaleDateString()} {generated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })})
        {stale && " — STALE: the daily refresh has not run for several days; treat scores with caution"}
      </span>
      <span style={{ opacity: 0.75 }}>
        · refreshed automatically each weekday pre-market
        · short interest is exchange-published on a ~2-week lag
      </span>
    </div>
  );
}

export default function App() {
  const [data, setData] = useState<Bundle | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");
  const [drawerTicker, setDrawerTicker] = useState<string | null>(null);
  const [pitchTicker, setPitchTicker] = useState<string | null>(null);

  useEffect(() => {
    loadAll().then(setData).catch((e) => setErr(String(e)));
  }, []);

  // Per-feature risk percentiles, shared by the drawer and pitch tab.
  const featureRanks = useMemo(
    () => (data ? computeFeatureRanks(data.universe) : null),
    [data],
  );

  if (err) {
    return (
      <div className="app">
        <div className="top-bar"><h1>IMA Risk Screener</h1></div>
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
        <div className="top-bar"><h1>IMA Risk Screener</h1></div>
        <div className="loading">Loading pipeline data…</div>
      </div>
    );
  }

  const openPitch = (t: string) => {
    setDrawerTicker(null);
    setPitchTicker(t);
    setTab("pitch");
  };

  return (
    <TickerOpenContext.Provider value={setDrawerTicker}>
      <div className="app">
        <div className="top-bar">
          <h1>IMA Risk Screener</h1>
          <span className="sub">
            S&amp;P 600 drawdown-risk monitor · {data.meta.universe_size} stocks
            · {data.meta.n_portfolio} IMA holdings
          </span>
        </div>
        <FreshnessBanner meta={data.meta} />
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
          {tab === "pitch"        && <PitchView meta={data.meta}
                                               universe={data.universe}
                                               clusterMeta={data.clusterMeta}
                                               trajectory={data.trajectory}
                                               initialTicker={pitchTicker} />}
          {tab === "backtest"     && <BacktestView />}
          {tab === "opportunities"&& <OpportunitiesView opportunities={data.opportunities} />}
          {tab === "drift"        && <DriftView drift={data.drift} meta={data.meta} />}
          {tab === "gallery"      && <GalleryView />}
        </main>

        {drawerTicker && featureRanks && (
          <TickerDrawer
            ticker={drawerTicker}
            universe={data.universe}
            meta={data.meta}
            clusterMeta={data.clusterMeta}
            trajectory={data.trajectory}
            featureRanks={featureRanks}
            onClose={() => setDrawerTicker(null)}
            onOpenPitch={openPitch}
          />
        )}
      </div>
    </TickerOpenContext.Provider>
  );
}
