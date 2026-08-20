export type RiskTier = "Low Risk" | "In Line" | "Elevated";

/** Descriptive cluster style name, e.g. "Core", "High-Multiple Growth". */
export type ClusterStyle = string;

export interface Meta {
  generated_at: string;
  universe_size: number;
  n_portfolio: number;
  financials_count: number;
  n_features: number;
  benchmark: string;
  pca: {
    n_components: number;
    variance_explained: number[];
    cumulative_variance: number[];
    pc_labels: Record<string, string>;
  };
  clustering: {
    k: number;
    silhouette: number;
    style_labels: Record<number, ClusterStyle>;
    risk_rank: Record<number, number>;
  };
  tier_order: RiskTier[];
  tier_colors: Record<RiskTier, string>;
  style_order: ClusterStyle[];
  style_colors: Record<ClusterStyle, string>;
  data_freshness?: {
    prices?: { updated_at: string } | null;
    fundamentals?: { updated_at: string } | null;
    sec_filings?: { updated_at: string } | null;
    short_interest_note?: string;
  };
}

export interface UniverseRow {
  Ticker: string;
  Company: string;
  Sector: string;
  is_financial: boolean;
  is_portfolio: boolean;
  weight: number;
  Industry?: string | null;
  market_cap?: number | null;
  cluster: number;
  cluster_style: ClusterStyle;
  composite_score: number;
  score_percentile: number;
  risk_tier: RiskTier;
  PC1?: number;
  PC2?: number;
  PC3?: number;
  PC4?: number;
  [feature: string]: any;
}

export interface PortfolioRow {
  Ticker: string;
  Weight: number;
  Sector?: string;
  Composite_Score?: number;
  Score_Percentile?: number;
  Risk_Tier?: RiskTier;
  Cluster?: number;
  Cluster_Label?: ClusterStyle;
  Altman_Z?: number | null;
  Piotroski_F?: number | null;
  Short_Pct_Float?: number | null;
  Momentum_90d?: number | null;
  Net_Debt_EBITDA?: number | null;
  Volatility_60d?: number | null;
  Top_Risk_Drivers?: string;
  Sector_Comparison?: string;
  Sector_Delta?: number | null;
  Trajectory?: string;
  Two_Q_Drift?: number | null;
  Status?: string;
}

export interface ClusterRow {
  cluster: number;
  style: ClusterStyle;
  n_stocks: number;
  [key: string]: any;
}

export interface ClusterMeta {
  k: number;
  silhouette: number;
  style_labels: Record<number, ClusterStyle>;
  risk_rank: Record<number, number>;
  style_colors: Record<ClusterStyle, string>;
  centroids: number[][];
  diagnostics: { k: number; silhouette: number; inertia: number; calinski_harabasz: number }[];
}

export interface PCASummaryRow {
  pc: string;
  variance_explained: number;
  cumulative_variance: number;
  label: string;
  top_loadings: { feature: string; loading: number }[];
}

export interface PCALoadingRow {
  feature: string;
  [pc: string]: any;
}

export interface OpportunityRow {
  Ticker: string;
  Company: string;
  Sector: string;
  altman_z: number;
  accruals_ratio: number;
  short_pct_float: number;
  momentum_90d: number;
  composite_score: number;
  risk_tier: RiskTier;
  contrarian_score: number;
}

export interface DriftRow {
  Ticker: string;
  assigned: number;
  assigned_style: ClusterStyle;
  nearest_other: number;
  nearest_other_style: ClusterStyle;
  boundary_gap: number;
  is_borderline: boolean;
  crossed_cluster_last_q: boolean;
  large_2q_drift: boolean;
  two_quarter_drift: number | null;
  is_portfolio: boolean;
  alert: boolean;
}

export interface TrajectoryCoord {
  date: string | null;
  cluster: number | null;
  PC1?: number | null;
  PC2?: number | null;
  PC3?: number | null;
  PC4?: number | null;
}

export interface TrajectoryData {
  snapshots: string[];
  paths: Record<string, {
    coords: TrajectoryCoord[];
    distance_traveled: number | null;
    two_quarter_drift: number | null;
    cluster_transitions: number;
  }>;
}

// =============================================================================
// Backtest (backtest.json) — produced by backtest.evaluate / webapp_export
// =============================================================================
export interface HitRateCell {
  n: number;
  events: number;
  rate: number;
  ci_low: number;
  ci_high: number;
  thin: boolean;
  decile?: number;
  tier?: string;
  mean_score?: number;
}

export interface BacktestData {
  metadata: {
    date_range: [string | null, string | null];
    n_snapshots: number;
    rebalance: string;
    horizon_months: number;
    dd_threshold: number;
    label_definition: string;
    survivorship_safe: boolean;
    cost_bps: number;
    synthetic_store?: boolean;
  };
  base_rate: {
    n_observations: number;
    n_events: number;
    base_rate: number | null;
    events_per_year: number;
    per_year: { year: number; events: number; n: number; rate: number | null }[];
    n_snapshots: number;
    survivorship_safe: boolean;
    dd_threshold: number;
    horizon_months: number;
  };
  deciles: {
    pooled: HitRateCell[];
    per_year: Record<string, HitRateCell[]>;
  };
  tiers: {
    pooled: HitRateCell[];
    per_year: Record<string, HitRateCell[]>;
    monotonic: boolean;
    elevated_minus_low: number;
    tier_order: string[];
  };
  information_coefficient: {
    time_series: { date: string; ic_return: number | null; ic_maxdd: number | null }[];
    ic_return_mean: number | null;
    ic_return_tstat: number | null;
    ic_maxdd_mean: number | null;
    ic_maxdd_tstat: number | null;
    horizon_months: number;
    n_cross_sections: number;
  };
  classification: {
    base_rate: number;
    n: number;
    roc: { fpr: number; tpr: number; threshold: number | null }[];
    auc: number | null;
    tier_thresholds: {
      tier: string; score_floor: number;
      precision: number | null; recall: number | null;
      lift: number | null; n_flagged: number;
    }[];
  };
  calibration: {
    reliability: {
      bin: number; mean_score: number; predicted: number; realized: number;
      ci_low: number; ci_high: number; n: number;
    }[];
    tier_calibration: HitRateCell[];
  };
  ima: {
    available: boolean;
    hit_miss?: {
      date: string; ticker: string; score: number; tier: string;
      flagged_elevated: boolean; fwd_maxdd: number | null;
      severe_dd: boolean; outcome: string;
    }[];
    score_time_series?: {
      date: string; ima_mean_score: number; universe_mean_score: number;
      n_holdings: number;
    }[];
    caught_events?: { date: string; ticker: string; score: number; tier: string; fwd_maxdd: number | null }[];
    missed_events?: { date: string; ticker: string; score: number; tier: string; fwd_maxdd: number | null }[];
    confusion?: { true_positive: number; false_positive: number; false_negative: number; true_negative: number };
    counterfactual?: {
      available: boolean;
      actual: StrategyStats;
      avoid_top_tier: StrategyStats;
      delta_cagr: number | null;
      delta_maxdd: number | null;
      equity_curve: { date: string; actual: number; avoid_top_tier: number }[];
    };
  };
  strategy: {
    available: boolean;
    benchmark_label?: string;
    cost_bps?: number;
    rebalance?: string;
    benchmark?: StrategyStats;
    avoid_top_tier?: StrategyStats;
    long_short?: StrategyStats;
    equity_curve?: { date: string; benchmark: number; avoid_top_tier: number; long_short: number }[];
  };
}

export interface StrategyStats {
  cagr: number | null;
  vol: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  total_return: number | null;
  equity: number[];
  hit_rate_vs_bench?: number;
  avg_turnover?: number;
}
