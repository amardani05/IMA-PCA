export type FactorCategory =
  | "rates" | "credit" | "inflation" | "commodities"
  | "currency" | "volatility_liquidity" | "financial_conditions"
  | "thematic" | "data_center_proxies";

export interface FactorMeta {
  factor: string;          // e.g. "rates_T10Y2Y"
  category: FactorCategory;
  series_id: string;
  name: string;
  source: "fred" | "yfinance" | "derived" | "manual";
  transform: "level_change" | "neg_level_change" | "log_return" | "pct_change" | "return_spread";
  in_curated: boolean;
  frequency?: "monthly";
}

export interface FactorMetadata {
  factors: FactorMeta[];
  categories: FactorCategory[];
  curated_factors: string[];
  scenario_shocks: Record<string, { label: string; shock: number }>;
}

export interface BetaEstimate {
  factor: string;
  beta: number;
  std_err: number;
  t_stat: number;
  p_value: number;
  ci_low: number;
  ci_high: number;
  significant_05: boolean;
  significant_01: boolean;
}

export interface PortfolioBetas {
  label: string;
  methodology?: "raw_ols" | "residualized" | "residualized_v1" | "residualized_v2";
  frequency?: "daily" | "weekly";
  market_beta?: number | null;
  market_beta_t?: number | null;
  market_beta_p?: number | null;
  control_betas?: Record<string, number>;
  factors: string[];
  n_obs: number;
  r_squared: number;
  adj_r_squared: number;
  alpha: number;
  alpha_t: number;
  alpha_p: number;
  residual_std: number;
  bonferroni_threshold: number;
  betas: Record<string, BetaEstimate>;
  vifs: Record<string, number>;
  contributions: Record<string, number>;
  skipped?: boolean;
}

export interface ComparisonRow {
  factor: string;
  // Three-way (current): raw / v1 / v2
  raw_beta?: number | null;
  raw_p?: number | null;
  v1_beta?: number | null;
  v1_p?: number | null;
  v2_beta?: number | null;
  v2_p?: number | null;
  // Legacy two-way fields (older payloads)
  residualized_beta?: number | null;
  residualized_p?: number | null;
  delta?: number | null;
  interpretation: string;
}

export interface StockBetaMatrix {
  tickers: string[];
  factors: string[];
  betas: Record<string, Record<string, number>>;       // ticker -> factor -> beta
  p_values: Record<string, Record<string, number>>;
  std_errors: Record<string, Record<string, number>>;
}

export interface RollingBetas {
  dates: string[];
  factors: string[];
  series: Record<string, (number | null)[]>;
}

export interface FactorReturns {
  dates: string[];
  factors: string[];
  series: Record<string, (number | null)[]>;
}

export interface ScenarioRow {
  factor: string;
  label: string;
  shock: number;
  beta: number;
  impact: number;
  significant: boolean;
  p_value: number;
}

export interface MacroSummary {
  generated_at: string;
  methodology?: "raw_ols" | "residualized";
  frequency?: "daily" | "weekly";
  market_beta?: number | null;
  n_factors_curated: number;
  n_obs: number;
  r_squared: number;
  alpha_annualized: number;
  max_vif: number | null;
}

export interface MacroBundle {
  metadata: FactorMetadata;
  portfolioBetas: PortfolioBetas;            // v2 (multi-factor residualized) when available
  portfolioBetasV1: PortfolioBetas | null;   // single-factor residualized (market only)
  portfolioBetasRaw: PortfolioBetas | null;  // raw OLS — diagnostic toggle
  portfolioBetasFull: PortfolioBetas | null;
  comparison: { rows: ComparisonRow[] };
  stockBetas: StockBetaMatrix;
  rollingBetas: RollingBetas;
  factorReturns: FactorReturns;
  scenarios: { scenarios: ScenarioRow[] };
  summary: MacroSummary;
  timeframes: MacroTimeframes | null;
}

export type TimeframeCode = "ytd" | "6m" | "1y" | "2y" | "max";

export interface TimeframeBundle {
  code: TimeframeCode;
  label: string;
  n_obs: number;
  date_range: [string, string];
  v2: PortfolioBetas;
  v1: PortfolioBetas;
  raw: PortfolioBetas;
  comparison: ComparisonRow[];
  scenarios: ScenarioRow[];
  stock_betas: StockBetaMatrix;
}

export interface MacroTimeframes {
  timeframes: TimeframeCode[];
  default: TimeframeCode;
  by_timeframe: Record<TimeframeCode, TimeframeBundle>;
}
