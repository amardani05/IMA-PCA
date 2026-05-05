export type RiskTier = "Low Risk" | "Moderate" | "Elevated" | "High" | "Critical";

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
    tier_labels: Record<number, RiskTier>;
  };
  tier_order: RiskTier[];
  tier_colors: Record<RiskTier, string>;
}

export interface UniverseRow {
  Ticker: string;
  Company: string;
  Sector: string;
  is_financial: boolean;
  is_portfolio: boolean;
  weight: number;
  cluster: number;
  cluster_tier: RiskTier;
  composite_score: number;
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
  Risk_Tier?: RiskTier;
  Cluster?: number;
  Cluster_Label?: RiskTier;
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
  tier: RiskTier;
  n_stocks: number;
  [key: string]: any;
}

export interface ClusterMeta {
  k: number;
  silhouette: number;
  tier_labels: Record<number, RiskTier>;
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
  assigned_tier: RiskTier;
  nearest_other: number;
  nearest_other_tier: RiskTier;
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
