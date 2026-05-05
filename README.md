# IMA Quantitative Risk Models — S&P 600

This repository hosts two related research projects for IMA. Both use PCA
on the S&P 600 universe, but they answer very different questions:

| Project | Question | Runs against |
| --- | --- | --- |
| **Project 1 — Factor Exposure PCA** | Which latent return factors does our portfolio tilt into? | Daily return series |
| **Project 2 — Torpedo Risk Screener** | Which stocks are at elevated risk of a severe drawdown? | Cross-sectional risk features |

The screener is the currently active project — its entry point is `main.py`.
The factor-exposure modules still live in the repo as reusable Python
components (see "Running Project 1" below).

---

## Repository layout

```
IMA-PCA/
├── config.py                 # Torpedo screener settings (portfolio, features, hyperparams)
├── universe.py               # S&P 600 Wikipedia scraper with fallback (shared)
│
├── feature_engine.py         # [Project 2] 14-feature risk matrix per stock
├── pca_cluster.py            # [Project 2] PCA + k-means + silhouette selection
├── trajectory.py             # [Project 2] Quarterly snapshots through PC space
├── scoring.py                # [Project 2] Composite score + opportunity screen
├── visualization.py          # [Project 2] Committee charts
├── main.py                   # [Project 2] Orchestrator
│
├── data_loader.py            # [Project 1] yfinance prices + FF5 + sector ETFs
├── pca_engine.py             # [Project 1] Correlation-matrix PCA + stability check
├── interpretation.py         # [Project 1] PC-to-factor/sector correlation + labels
├── portfolio_projection.py   # [Project 1] Portfolio PC loadings + variance decomposition
│
├── requirements.txt
├── interactive_charts.py     # [Project 2] Plotly 2D/3D interactive charts (HTML + JSON)
├── webapp_export.py          # [Project 2] Export pipeline outputs as JSON for the webapp
│
├── macro_loader.py           # [Project 4] FRED + yfinance macro factor loader (.env config)
├── macro_regression.py       # [Project 4] OLS/HAC, VIF, per-stock, rolling, scenarios
├── macro_export.py           # [Project 4] JSON dumps for the macro webapp tab
├── webapp/                   # [Project 2/4] React + Vite + TypeScript dashboard
│
├── data/                     # Cached inputs (prices, fundamentals, CIK map, etc.)
└── output/                   # CSVs + PNGs produced by the pipelines
```

---

## Installation

Python 3.11 is what this repo has been validated on.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First-run network calls hit Wikipedia, yfinance, Ken French's data library, and
SEC EDGAR. All outputs are cached under `data/`, so subsequent runs are fast.

---

## Project 2 — Torpedo Risk Screener (active)

### What it does

Screens the S&P 600 for stocks at elevated risk of a severe drawdown and
places them into interpretable risk tiers. The model combines four families of
signals into a single framework:

1. **Financial health** — Altman Z, Piotroski F, current ratio, net debt /
   EBITDA, FCF yield, interest coverage.
2. **Market positioning** — short interest, 30d & 90d momentum, 60d vol, 20/60d
   relative volume.
3. **Valuation stress** — P/E (abs value, capped), EV/EBITDA.
4. **Event risk proxy** — count of SEC 8-K filings in the last 90 days.

All **16 features** are z-scored, reduced to four principal components, and
clustered with k-means. `N_CLUSTERS=3` (the silhouette gap to k=4 was
within tolerance, and k=3 eliminates a cluster whose members were on
average closer to a different cluster's centroid). Each cluster is
tier-labeled by its composite risk score — producing a
**Stable → Mainstream → Elevated** ordering, regardless of which k is picked.

A per-stock 0-100 composite score (mean of risk-direction-flipped percentile
ranks) is produced alongside the cluster label. The two views are reported
together because their disagreements are informative: cluster says "cheap
fundamentals but extreme sentiment"; score says "top decile by sentiment".

The trajectory module re-computes features at several past quarter-ends, runs
them through the *same* scaler + PCA + k-means model, and produces per-stock
paths through PC space. This surfaces drift alerts: stocks that crossed a
cluster boundary last quarter, stocks that moved more than 1.5σ in two
quarters, and stocks within 0.5 PC-units of a boundary.

### How it works (module-by-module)

- **`universe.py`** — Scrapes the Wikipedia S&P 600 table, normalizes tickers
  (Yahoo uses `BRK-B`, not `BRK.B`), caches to `data/sp600_constituents.csv`,
  and falls back to a seeded CSV if the scrape fails.
- **`feature_engine.py`** — Pulls `.info`, quarterly financials, balance
  sheet, and cash flow from yfinance, plus daily prices and volume. Field
  names vary (`Revenue` vs `Total Revenue`, `EBIT` vs `Operating Income`) so
  every lookup goes through `get_field(df, [aliases...])`. Per-ticker errors
  are swallowed and logged. SEC EDGAR is queried via the `submissions.json`
  endpoint for 8-K counts, with a rate limiter well under the 10/s cap.
  Financials (banks, insurance, REITs) get a NaN Altman Z, which is then
  imputed by cross-sectional median — traditional Altman Z is not meaningful
  for balance sheets whose liabilities are structurally large. Stocks with
  >50% of features missing are dropped.
- **`pca_cluster.py`** — `StandardScaler` + `PCA(n_components=4)`. Auto-labels
  each PC by its dominant-loading feature family (Financial Health / Market
  Sentiment / Volatility+Attention / Valuation Stress / Event Activity /
  Leverage). k-means is evaluated over k={3..7} with silhouette,
  Calinski-Harabasz, and inertia. Cluster tiers are assigned by composite
  risk rank, not by cluster id order.
- **`trajectory.py`** — Rebuilds features at each quarterly snapshot using
  the most-recent-quarter-before-snapshot fundamentals and the price history
  up to that date. Uses the SAME scaler/PCA/k-means fitted on the current
  snapshot so all points share one coordinate system.
- **`scoring.py`** — Per-feature percentile rank (flipped for direction),
  averaged into a 0-100 score, bucketed into 3 tiers (**Stable / Mainstream
  / Elevated**). `format_combined_label` pairs the cluster tier with the
  composite-score percentile (e.g. `"Mainstream (47th pct)"`) — the cluster
  label is a coarse summary, the percentile is the granular measure. The
  contrarian "opportunity screen" pulls stocks with `altman_z > 2`,
  `short_pct_float > 8%`, `momentum_90d < 0`.
- **`visualization.py`** — All static matplotlib charts with consistent tier
  coloring (green → yellow → orange → red), including a 3D PCA scatter
  (`mpl_toolkits.mplot3d`) with overlaid IMA trajectories.
- **`interactive_charts.py`** — Plotly versions of the PC scatters (PC1×PC2,
  PC1×PC3, PC2×PC3, and rotatable 3D). Each chart is written as a standalone
  HTML file AND a figure-spec JSON consumed by the React webapp.
- **`webapp_export.py`** — Dumps every pipeline table as JSON and copies all
  chart assets into `webapp/public/`. Invoked as the final pipeline step.
- **`main.py`** — Orchestrator and terminal summary.

### How to run it

Full pipeline, first run (~20 minutes; downloads 600 × fundamentals + prices):

```bash
python main.py
```

After the first run every major input is cached under `data/`, so a re-run
completes in 1-2 minutes:

```bash
python main.py                     # cached re-run
python main.py --no-trajectory     # skip historical snapshots (~30 min → seconds saved)
python main.py --portfolio-only    # only score the 20 IMA holdings (fast daily monitor)
python main.py --clusters 4        # override k
python main.py --refresh           # force re-fetch of ALL cached data
python main.py --refresh-features  # recompute features only (keep price cache)
python main.py -v                  # DEBUG logging
```

### Outputs

Written to `output/`:

| File | Content |
| --- | --- |
| `risk_scores_full.csv` | Every universe stock: features, PCA coords, cluster, tier, score |
| `portfolio_risk_report.csv` | 20 IMA holdings with drivers, sector comparison, trajectory |
| `cluster_summary.csv` | Cluster sizes + per-feature mean/median profiles |
| `cluster_k_diagnostics.csv` | k={3..7} silhouette / inertia / CH |
| `pca_loadings.csv`, `pca_summary.csv` | Feature loadings and variance explained per PC |
| `opportunity_watchlist.csv` | Contrarian candidates (fundamentals intact, sentiment bearish) |
| `trajectory_data.csv` | Per-stock quarterly PC coordinates (if trajectory ran) |
| `drift_alerts.csv` | Borderline / boundary-crossing / large-drift tickers |
| `cluster_scatter_pc1_pc2.png`, `cluster_scatter_pc2_pc3.png` | PC scatters, IMA overlaid |
| `trajectory_map.png` | Arrows tracking IMA holdings through PC space |
| `portfolio_risk_dashboard.png` | 4×5 panel of IMA holdings' feature percentiles |
| `cluster_profiles.png` | Bar chart: each cluster's z-scored feature means |
| `silhouette_analysis.png` | k-selection bar chart |
| `pca_loadings.png` | Feature × PC heatmap |
| `risk_score_distribution.png` | Universe histogram with IMA holdings marked |
| `sector_risk_comparison.png` | Score box plot by sector, IMA holdings overlaid |
| `cluster_scatter_3d.png` | 3D PCA scatter (mpl_toolkits.mplot3d) with IMA trajectories |
| `interactive/scatter_pc1_pc2.html` ,`scatter_pc1_pc3.html`, `scatter_pc2_pc3.html` | Plotly interactive 2D PC scatters (self-contained HTML) |
| `interactive/scatter_3d.html` | Plotly interactive rotatable 3D PC scatter with trajectories |
| `interactive/*.json` | Plotly figure specs consumed by the React webapp |

The terminal prints a committee summary: universe size, PCA variance
explained per PC, cluster sizes and tiers, a table of all 20 IMA holdings
with composite score / tier / cluster / key features / trajectory direction,
top 5 highest-risk holdings with primary risk drivers, the top 10
opportunity-watchlist names, and drift alerts for any IMA holding near or
across a cluster boundary.

### Methodology limitations

A diagnostic suite (`python -m diagnostics.run_all`) audited the pipeline
before the website was built. It surfaced four findings worth being honest
about with the committee:

- **Cluster labels are calibrated to the *current* universe**, not absolute
  risk levels. "Stable" means the stock looks unremarkable on PC space *vs.
  this S&P 600 cohort*. It is **not** a credit rating.
- **About 70% of the universe lands in "Mainstream"** because most S&P 600
  stocks don't exhibit extreme risk signatures. The cluster taxonomy adds
  granularity at the tails (Stable + Elevated) but doesn't cleanly separate
  the middle of the distribution. The composite **score percentile** (0-100)
  is the more granular measure; tier labels are summary categories.
  `format_combined_label` shows both at once (e.g. `"Mainstream (47th pct)"`).
- **Cluster ARI under 80% subsampling = 0.671** (moderate-stable). PCs are
  highly stable (cosine similarity 0.82–0.97). Borderline names near a
  cluster boundary may shift between runs; the drift-alerts table flags them.
- **Three features were dropped** in the 2026-05-03 cleanup because they
  contributed nothing measurable (leave-one-out ARI ≥ 0.989):
  - `piotroski_f` — yfinance only returns ~4 quarters of fundamentals; the
    YoY signals require 8.
  - `earnings_drag` — `yfinance.earnings_dates` returned no historical data
    for our universe.
  - `insider_net_sell_ratio` — the v1 implementation only counted Form-4
    filings without parsing transaction codes (P/S/A/M); the proxy was too
    coarse to load on any PC. Reinstating requires Form-4 XML parsing.

To restore any dropped feature, fix the data source, then add it back to
`config.FEATURES` and `config.RISK_DIRECTION`.

### Design notes worth knowing

- **Rate limits.** SEC EDGAR caps at 10 req/sec; the screener rate-limits to
  8/sec to leave margin. yfinance fundamentals are requested with 20-ticker
  batches and a 3-second inter-batch sleep.
- **Leverage sign / cap.** `net_debt_to_ebitda` is capped at ±20 and set to
  +20 when TTM EBITDA ≤ 0 (maximum risk, not missing data).
- **Negative earnings.** `pe_ratio` uses `abs(P/E)` capped at 200, with a
  separate `earnings_negative` boolean flag so the sign isn't lost.
- **Partial Piotroski.** If fewer than 9 of the 9 signals are computable, the
  score is scaled proportionally and tagged as partial.
- **Financials.** Altman Z is set to NaN for `Financials` / `Financial
  Services` / `Real Estate` sectors (structurally high liabilities) and then
  imputed. Acceptable for clustering but flagged.

---

### Web dashboard (React + Vite + TypeScript)

A full-featured browser dashboard lives under [`webapp/`](webapp/). Every pipeline
run automatically exports its outputs to `webapp/public/data/*.json` and copies
the PNGs / Plotly JSON into `webapp/public/charts` and `webapp/public/interactive`,
so the dashboard is always up-to-date with the latest run.

Sections:

- **Overview** — headline stats, portfolio tier distribution, top 5 risky holdings, PC summary.
- **Interactive PCA** — rotatable 3D plot and all three 2D PC pair scatters with hover tooltips
  and IMA trajectory overlays.
- **Portfolio** — sortable, filterable table of every IMA holding with drivers, sector
  comparison, trajectory classifier, and weighted composite score.
- **Universe** — the full 600-row risk-scores table with search, tier / sector / IMA filters,
  and column-click sorting.
- **Clusters** — cluster sizes, tier assignments, k-selection diagnostics, static feature
  profile chart.
- **PCA** — variance explained, auto-generated PC labels, and a color-coded loadings matrix.
- **Opportunities** — contrarian watchlist.
- **Macro Exposures** — interactive portfolio tree, live macro betas with significance stars, rolling-beta time series, scenario sensitivities. (Project 4.)
- **Drift Alerts** — boundary-crossing / borderline stocks, IMA names pinned up top.
- **Chart Gallery** — every static PNG at full resolution.

Run it:

```bash
cd webapp
npm install        # first time only
npm run dev        # dev server at http://localhost:5173
npm run build      # static bundle to webapp/dist/ for deployment
```

If you see "Failed to load pipeline data", run `python main.py --no-trajectory`
once from the repo root — that populates `webapp/public/data/` and
`webapp/public/charts/` via the `webapp_export` step. Pass `--no-webapp` to
main.py to skip the export step on slow runs.

---

---

## Project 4 — Macro Factor Exposures + Interactive Portfolio Builder

### What it does

Answers the committee question "are we accidentally long oil / inflation /
credit spreads / the long bond?" by regressing daily portfolio (and
per-holding) returns on a curated set of stationarity-transformed macro
factors. Every regression uses HAC (Newey-West, 5-day) standard errors so
significance isn't overstated by autocorrelated residuals.

The webapp Macro tab adds a hierarchical portfolio tree where stocks can be
toggled on/off live; betas update instantly via a weighted-average of the
exported per-stock beta matrix. Re-run the Python pipeline for an exact
regression on the new portfolio (the webapp shows an explicit warning when
the displayed betas are approximate).

### How it works

- **`macro_loader.py`** — Pulls FRED series (yields, spreads, breakevens,
  commodities, USD, financial-conditions indices) and yfinance series (gold,
  copper, MOVE, thematic ETFs, data-center proxies). Applies
  `level_change` to rates / spreads / VIX (which are I(1)) and `log_return`
  to prices. Aligns on business-day frequency, forward-fills small gaps up to
  5 days, and caches every series to `data/macro/`. Validation reports
  per-factor coverage and flags any series with >10% missing or
  zero-variance.
- **`macro_regression.py`** — `statsmodels.OLS` with HAC errors, two modes:
  - `mode="curated"` (default): one representative factor per category (10Y-2Y
    spread, HY OAS, 5Y breakeven, WTI, gold, copper, trade-weighted USD, VIX,
    NFCI). Designed to keep VIF < 5.
  - `mode="full"`: kitchen-sink across every defined factor; reports VIF.
  Also: per-stock regressions producing a ticker × factor beta matrix,
  rolling 60-day window regressions for stability monitoring, and scenario
  shock impacts (`beta × shock`).
- **`macro_export.py`** — Dumps `portfolio_betas.json`, `stock_betas.json`,
  `rolling_betas.json`, `factor_returns.json`, `scenarios.json`,
  `factor_contributions.json`, `factor_metadata.json`, and `macro_summary.json`
  into `webapp/public/data/macro/`.
- **Webapp** — `MacroView` integrates `PortfolioTree` (sector-grouped,
  indeterminate checkboxes, candidate add-from-universe), `MacroBetaPanel`
  (grouped horizontal bars with significance stars and live-vs-fitted overlay),
  `MacroFactorChart` (rolling 60-day betas, multi-factor toggleable lines),
  and `ScenarioCard` (live scenario impacts). The
  `usePortfolioSelection` hook recomputes portfolio betas client-side from the
  per-stock matrix as users toggle holdings.

### How to run it

```bash
# 1. Get a free FRED key at https://fred.stlouisfed.org/docs/api/api_key.html
echo "FRED_API_KEY=your_key_here" > .env

# 2. Run the full pipeline (caches both fundamentals and macro series)
python main.py --no-trajectory

# 3. To skip macro on a run (e.g., when key is unavailable):
python main.py --no-macro

# 4. Launch the dashboard
cd webapp
npm run dev   # http://localhost:5173 → "Macro Exposures" tab
```

### Outputs

In `webapp/public/data/macro/`:

| File | Content |
| --- | --- |
| `factor_metadata.json` | Factor names, categories, transforms, source, curated flag |
| `portfolio_betas.json` | Curated set: betas, t-stats, p-values, CIs, VIFs, R² |
| `portfolio_betas_full.json` | Kitchen-sink regression for diagnostics |
| `stock_betas.json` | Per-holding ticker × factor beta matrix (drives live recomputation) |
| `rolling_betas.json` | Time series of rolling 60-day portfolio betas |
| `factor_returns.json` | Time series of transformed factor returns |
| `portfolio_returns.json` | Daily portfolio return series |
| `scenarios.json` | Beta × shock impact decomposition |
| `factor_contributions.json` | Cumulative return contribution per factor |
| `macro_summary.json` | Generated-at, R², annualized α, max VIF |

### Critical notes

- **Stationarity**: yields and spreads are I(1) — the loader takes first
  differences before regression. VIX is technically stationary but persistent;
  first-differencing is the safer default. Prices use log returns. Get this
  wrong and every t-stat is spurious.
- **HAC standard errors**: `cov_type="HAC"` with `maxlags=5` corrects for
  both heteroskedasticity AND autocorrelation in residuals. Don't use HC0/HC1
  on daily return data.
- **Multicollinearity**: the curated set (default) keeps VIF < 5 across the
  board. If `--no-macro` is off and the kitchen-sink mode reports VIF > 10 on
  any factor, treat that beta as unreliable.
- **Live tree edits**: the webapp's portfolio tree recomputes betas as a
  weighted average of per-stock betas. This is mathematically exact for the
  point estimate but does NOT recompute residual variance / R² / SEs. Re-run
  `python main.py` for exact figures on a modified portfolio.

---

## Project 1 — Factor Exposure PCA (reference / research)

### What it does

Runs PCA on the correlation matrix of daily returns for the S&P 600 universe
(plus IMA holdings), retains `N_COMPONENTS` eigenvectors as "factors", and
measures the IMA portfolio's loading on each one. Each retained PC is
interpreted by correlating its score series against:

- the Fama-French 5 factors + momentum (Ken French's data library);
- the 11 sector SPDR ETFs (`XLF`, `XLK`, ...);
- the S&P 600 benchmark ETF (`IJR`).

Heuristic labels (`Market Beta`, `Value/Growth`, `Momentum`, ...) come from
the dominant correlation. Stability is checked by running PCA on two
non-overlapping halves of the window and comparing eigenvectors.

### How it works

- **`data_loader.py`** — Batched yfinance downloads with retry, parquet
  caching, and Fama-French ZIP parsing.
- **`pca_engine.py`** — Correlation (not covariance) PCA via `numpy.linalg.eigh`
  for numerical stability. Includes a Marchenko-Pastur noise threshold so
  eigenvalues from pure noise can be separated from real structure.
- **`interpretation.py`** — Correlates each PC score series against external
  factors, assigns a heuristic label, and summarizes top / bottom loadings.
- **`portfolio_projection.py`** — Weighted-average PC loading for the
  portfolio; active exposure vs universe average; variance decomposition with
  a residual bucket for variance living outside the retained PCs; rolling
  portfolio-PC beta.

### How to run it

**Status:** `main.py`, `config.py`, and `visualization.py` in this repo were
rewritten for the torpedo screener (Project 2). To run Project 1 you need to
invoke its modules directly (Python REPL / a notebook) or restore its
orchestrator. A minimal Python driver looks like:

```python
from data_loader import (
    load_universe_returns, load_benchmark_returns,
    load_sector_etf_returns, load_fama_french,
)
from pca_engine import run_pca, stability_check, top_bottom_loadings
from interpretation import (
    build_factor_correlation_matrix, label_pcs, summarize_top_bottom,
)
from portfolio_projection import (
    active_pc_exposure, portfolio_variance_decomposition,
)
from universe import get_sp600_universe

universe_df = get_sp600_universe()
tickers = sorted(set(universe_df["Ticker"]) | set(PORTFOLIO))

returns = load_universe_returns(tickers, lookback_days=504)
benchmark = load_benchmark_returns("IJR", lookback_days=504)
sectors = load_sector_etf_returns(lookback_days=504)
ff = load_fama_french()

pca = run_pca(returns, n_components=6)
stab = stability_check(returns, n_components=6)
corr = build_factor_correlation_matrix(pca.scores, ff, sectors, benchmark)
labels = label_pcs(corr)
exposure = active_pc_exposure(PORTFOLIO, pca)
decomp = portfolio_variance_decomposition(PORTFOLIO, pca)
```

`data_loader.py` still imports from the old `config.py` constants
(`FF5_URL`, `SECTOR_ETFS`, `BENCHMARK_TICKER`, etc.). Those were removed from
the current `config.py`, so running Project 1 requires either reinstating
those constants at the bottom of `config.py` or extracting them into a
dedicated `config_factor.py`. If you want the prior `main.py` restored
verbatim, the conversation history has it.

### Outputs it was designed to produce

- `pca_summary.csv`, `pc_loadings.csv`, `portfolio_exposure.csv`
- `pc_factor_correlations.csv`, `pc_top_bottom_loadings.csv`
- `portfolio_variance_decomposition.csv`, `rolling_pc_exposure.csv`
- Charts: scree, factor-correlation heatmap, portfolio PC exposure, variance
  decomposition, PC1×PC2 / PC1×PC3 scatter with holdings overlaid, rolling
  exposure.

---

## Troubleshooting

- **`ImportError: numpy.core.multiarray failed to import`** — NumPy 2 vs
  pre-built 1.x wheels. Upgrade `scikit-learn`, `pyarrow`, and `matplotlib`
  (`pip install -U scikit-learn pyarrow matplotlib`) or pin `numpy<2`.
- **Wikipedia scrape returns few tickers** — the scraper raises if it finds
  <400 names and falls back to `data/sp600_fallback.csv`. Seed that file from
  a known-good run.
- **SEC EDGAR 403 / 429** — the `User-Agent` header must include a real
  contact email. Update `SEC_USER_AGENT` in `config.py`.
- **yfinance "possibly delisted; no price data found"** — expected for
  constituent changes; those tickers are logged and skipped, not fatal.
