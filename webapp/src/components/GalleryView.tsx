const CHARTS: { file: string; title: string; desc: string }[] = [
  { file: "cluster_scatter_pc1_pc2.png", title: "Clusters · PC1 vs PC2",
    desc: "Static matplotlib scatter. IMA holdings ringed with ticker labels." },
  { file: "cluster_scatter_pc2_pc3.png", title: "Clusters · PC2 vs PC3",
    desc: "Alternate 2D view." },
  { file: "cluster_scatter_3d.png", title: "Clusters · 3D",
    desc: "mpl_toolkits.mplot3d rendering with trajectories." },
  { file: "trajectory_map.png", title: "Trajectory map",
    desc: "IMA holdings' quarterly paths through PC space. Red arrows = riskier, green = safer." },
  { file: "portfolio_risk_dashboard.png", title: "Portfolio dashboard",
    desc: "Per-holding feature percentile bars, tier-colored background." },
  { file: "cluster_profiles.png", title: "Cluster feature profiles",
    desc: "Each cluster's characteristic feature signature (z-scored means)." },
  { file: "silhouette_analysis.png", title: "Silhouette by k",
    desc: "Which k best separates the data." },
  { file: "pca_loadings.png", title: "PCA loadings heatmap",
    desc: "Feature → PC loadings, diverging colormap." },
  { file: "risk_score_distribution.png", title: "Risk score distribution",
    desc: "Universe composite-score histogram with IMA holdings marked." },
  { file: "sector_risk_comparison.png", title: "Sector risk comparison",
    desc: "Boxplot of risk scores per GICS sector with IMA overlaid." },
];

export function GalleryView() {
  return (
    <div>
      <h2 className="section-title">Static chart gallery</h2>
      <p className="section-lede">
        PNG charts produced by the matplotlib pipeline — these are the same figures attached to
        the committee deck. Open in a new tab for full resolution.
      </p>

      <div className="grid grid-2">
        {CHARTS.map((c) => (
          <div key={c.file} className="card">
            <h3>{c.title}</h3>
            <div className="card-sub">{c.desc}</div>
            <div className="static-chart">
              <a href={`/charts/${c.file}`} target="_blank" rel="noreferrer">
                <img src={`/charts/${c.file}`} alt={c.title} />
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
