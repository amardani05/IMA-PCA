/**
 * Plain-English glossary for every model feature. This is the single source
 * the UI uses for column headers, tooltips, and driver names, so a PM never
 * has to decode `net_debt_to_ebitda(87)`.
 *
 * `direction` mirrors config.RISK_DIRECTION in the Python pipeline:
 *   +1 → a HIGHER raw value means MORE risk; -1 → a HIGHER value means LESS.
 */
export interface FeatureMeta {
  label: string;        // short human name for table headers
  long: string;         // full name for the drawer / pitch page
  definition: string;   // one-sentence tooltip
  direction: 1 | -1;
  fmt: "num" | "pct" | "x" | "count";
}

export const FEATURE_META: Record<string, FeatureMeta> = {
  altman_z: {
    label: "Altman Z", long: "Altman Z-score",
    definition: "Bankruptcy-risk composite from balance-sheet ratios. Higher is safer; below ~1.8 signals distress. Not meaningful for banks/REITs (imputed).",
    direction: -1, fmt: "num",
  },
  current_ratio: {
    label: "Current ratio", long: "Current ratio",
    definition: "Current assets ÷ current liabilities. Higher means more short-term liquidity.",
    direction: -1, fmt: "num",
  },
  net_debt_to_ebitda: {
    label: "Net debt/EBITDA", long: "Net debt / EBITDA",
    definition: "Leverage: years of EBITDA needed to repay net debt. Higher is riskier; capped at ±20 (set to +20 when EBITDA is negative).",
    direction: 1, fmt: "x",
  },
  fcf_yield: {
    label: "FCF yield", long: "Free-cash-flow yield",
    definition: "Trailing free cash flow ÷ market cap. Higher means more cash generation per dollar of price.",
    direction: -1, fmt: "pct",
  },
  interest_coverage: {
    label: "Int. coverage", long: "Interest coverage",
    definition: "EBIT ÷ interest expense. Higher means earnings cover debt service more comfortably.",
    direction: -1, fmt: "x",
  },
  accruals_ratio: {
    label: "OCF/NI", long: "Cash conversion (OCF / Net income)",
    definition: "Operating cash flow ÷ net income. Low values mean earnings aren't backed by cash (Sloan accruals red flag).",
    direction: -1, fmt: "x",
  },
  asset_growth_yoy: {
    label: "Asset growth", long: "Total asset growth (YoY)",
    definition: "Year-over-year balance-sheet growth. Rapid asset growth historically precedes weak returns (the asset-growth anomaly).",
    direction: 1, fmt: "pct",
  },
  net_issuance_yoy: {
    label: "Share issuance", long: "Net share issuance (YoY)",
    definition: "Change in shares outstanding. Positive = dilution; buybacks are negative.",
    direction: 1, fmt: "pct",
  },
  short_pct_float: {
    label: "Short %", long: "Short interest (% of float)",
    definition: "Percent of tradable shares sold short (exchange-published, ~2-week lag). High values mean the market is betting against the name.",
    direction: 1, fmt: "num",
  },
  momentum_30d: {
    label: "Mom 30d", long: "30-day price momentum",
    definition: "Trailing 30-day price return. Falling prices are treated as risk-increasing.",
    direction: -1, fmt: "pct",
  },
  momentum_90d: {
    label: "Mom 90d", long: "90-day price momentum",
    definition: "Trailing 90-day price return. Falling prices are treated as risk-increasing.",
    direction: -1, fmt: "pct",
  },
  volatility_60d: {
    label: "Vol 60d", long: "60-day realized volatility",
    definition: "Annualized standard deviation of daily returns over 60 sessions. Higher = larger price swings.",
    direction: 1, fmt: "pct",
  },
  relative_volume: {
    label: "Rel. volume", long: "Relative volume (20d/60d)",
    definition: "Recent average volume vs its own 60-day baseline. Spikes mean unusual attention.",
    direction: 1, fmt: "x",
  },
  pe_ratio: {
    label: "P/E", long: "Price / earnings (|abs|, capped at 200)",
    definition: "Absolute P/E, capped at 200. High values flag valuation stress — note loss-makers and expensive growers look alike here.",
    direction: 1, fmt: "x",
  },
  ev_to_ebitda: {
    label: "EV/EBITDA", long: "Enterprise value / EBITDA",
    definition: "Capital-structure-neutral valuation multiple. Higher = more expensive.",
    direction: 1, fmt: "x",
  },
  filing_count_90d: {
    label: "8-Ks (90d)", long: "SEC 8-K filings, last 90 days",
    definition: "Count of 8-K material-event filings. A crude event-risk proxy — routine earnings 8-Ks count too.",
    direction: 1, fmt: "count",
  },
};

export function featureLabel(key: string): string {
  return FEATURE_META[key]?.label ?? key;
}

export function featureLong(key: string): string {
  return FEATURE_META[key]?.long ?? key;
}

export function featureDefinition(key: string): string {
  const m = FEATURE_META[key];
  if (!m) return key;
  const dir = m.direction === 1 ? "Higher = riskier." : "Higher = safer.";
  return `${m.definition} ${dir}`;
}

/** Mirror of config.RISK_DIRECTION for client-side percentile computation. */
export const RISK_DIRECTION: Record<string, 1 | -1> = Object.fromEntries(
  Object.entries(FEATURE_META).map(([k, v]) => [k, v.direction]),
) as Record<string, 1 | -1>;

export const FEATURE_KEYS: string[] = Object.keys(FEATURE_META);

/** "short_pct_float(92); momentum_90d(88)" → "Short interest 92nd pct; 90-day momentum 88th pct" */
export function humanizeDriverString(drivers?: string): string {
  if (!drivers) return "";
  return drivers
    .split(";")
    .map((part) => {
      const m = part.trim().match(/^([a-z0-9_]+)\((\d+)\)$/i);
      if (!m) return part.trim();
      return `${featureLabel(m[1])} ${ordinal(Number(m[2]))} pct`;
    })
    .join(" · ");
}

export function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] ?? s[v] ?? s[0]);
}
