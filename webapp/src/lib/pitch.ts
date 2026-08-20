export interface PitchNeighbor {
  ticker: string;
  distance: number;
  is_held: boolean;
  is_former_hold: boolean;
  company: string;
  sector: string;
}

export interface PitchAssessment {
  ticker: string;
  company_name: string;
  sector: string;
  industry: string;
  market_cap: number;

  nearest_neighbors: PitchNeighbor[];
  n_neighbors_currently_held: number;
  n_neighbors_formerly_held: number;
  similarity_verdict: string;

  portfolio_centroid: Record<string, number>;
  candidate_position: Record<string, number>;
  deviations_from_centroid: Record<string, number>;
  significant_deviations: string[];
  diversification_score: number;

  cluster_id: number;
  cluster_style: string;
  risk_tier: string;
  composite_risk_score: number;
  score_percentile: number;
  top_risk_drivers: { feature: string; percentile: number }[];
  cluster_trajectory: string;

  sector_median_score: number;
  delta_vs_sector: number;
  sector_comparison: string;

  summary_bullets: string[];
  recommendation: "PROCEED" | "PROCEED WITH CAVEATS" | "QUESTION THESIS" | "AVOID" | string;
  recommendation_rationale: string;

  generated_at: string;
  n_neighbors: number;
}

export function recommendationColor(rec: string): string {
  switch (rec) {
    case "PROCEED": return "var(--ok)";
    case "PROCEED WITH CAVEATS": return "var(--warn)";
    case "QUESTION THESIS": return "#e57a44";
    case "AVOID": return "var(--danger)";
    default: return "var(--muted)";
  }
}
