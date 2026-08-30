"""PCA on the factor library itself — "exposures we didn't think we had".

Where the screener runs PCA across STOCKS (cross-sectional features), this
module runs PCA across the FACTOR PANEL (daily time series): it extracts the
orthogonal macro themes that actually drive the factor library, then projects
portfolio, benchmark, and ACTIVE (portfolio − benchmark) returns onto each
theme with HAC errors.

Why this complements the per-factor betas: named-factor regressions answer
"are we exposed to X?" for the X's we thought to ask about. Theme PCA answers
the reverse question — it finds whatever correlated bundle of macro moves
explains our active returns, whether or not we had a name for it. A theme with
a significant ACTIVE beta is a hidden active exposure.

Scope choices:
  * only macro categories enter (rates, credit, inflation, commodities,
    debasement, currency, vol/liquidity, financial conditions, growth) —
    thematic equity ETFs and single-name proxies are excluded so themes stay
    macro, not equity-beta in disguise;
  * legs of derived factors are excluded (the spread stays, XLI/XLP go);
  * monthly manual series are excluded (a monthly step ffilled across days
    has no daily variance to decompose).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

from macro_loader import MACRO_FACTORS

logger = logging.getLogger(__name__)

MACRO_THEME_CATEGORIES: tuple[str, ...] = (
    "rates", "credit", "inflation", "commodities", "debasement",
    "currency", "volatility_liquidity", "financial_conditions", "growth",
)

MIN_COVERAGE = 0.70          # a factor must have >=70% of days present
DEFAULT_N_COMPONENTS = 6


def _display_names() -> dict[str, str]:
    out = {}
    for category, defs in MACRO_FACTORS.items():
        for series_id, defn in defs.items():
            out[f"{category}_{series_id}"] = defn["name"]
    return out


def _derived_leg_columns() -> set[str]:
    legs: set[str] = set()
    for defs in MACRO_FACTORS.values():
        for defn in defs.values():
            if defn.get("source") == "derived":
                legs.update(defn.get("legs", []))
    return legs


def _monthly_columns() -> set[str]:
    out: set[str] = set()
    for category, defs in MACRO_FACTORS.items():
        for series_id, defn in defs.items():
            if defn.get("frequency") == "monthly":
                out.add(f"{category}_{series_id}")
    return out


@dataclass
class ThemeBeta:
    beta: float          # return per +1σ daily theme move
    t_stat: float
    p_value: float

    def to_dict(self) -> dict:
        return {"beta": self.beta, "t_stat": self.t_stat, "p_value": self.p_value,
                "significant_10": self.p_value < 0.10,
                "significant_05": self.p_value < 0.05}


@dataclass
class FactorPCAResult:
    n_obs: int
    window: tuple[str, str]
    factors: list[str]                       # columns that entered
    variance_explained: list[float]
    cumulative_variance: list[float]
    labels: list[str]                        # auto theme label per PC
    top_loadings: list[list[dict]]           # per PC: [{factor, name, loading}]
    betas: dict[str, list[ThemeBeta]]        # "portfolio" / "benchmark" / "active"
    r_squared: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_obs": self.n_obs,
            "window": list(self.window),
            "n_factors": len(self.factors),
            "factors": self.factors,
            "variance_explained": self.variance_explained,
            "cumulative_variance": self.cumulative_variance,
            "components": [
                {
                    "pc": f"PC{i+1}",
                    "label": self.labels[i],
                    "variance_explained": self.variance_explained[i],
                    "top_loadings": self.top_loadings[i],
                    "portfolio": self.betas["portfolio"][i].to_dict(),
                    "benchmark": self.betas["benchmark"][i].to_dict(),
                    "active": self.betas["active"][i].to_dict(),
                }
                for i in range(len(self.labels))
            ],
            "r_squared": self.r_squared,
        }


def _hac_betas(y: pd.Series, scores: pd.DataFrame, lags: int = 5) -> tuple[list[ThemeBeta], float]:
    """Regress y on all PC scores jointly (orthogonal, so joint == marginal)."""
    df = pd.concat([y.rename("y"), scores], axis=1).dropna()
    X = sm.add_constant(df[scores.columns])
    model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = [
        ThemeBeta(
            beta=float(model.params[pc]),
            t_stat=float(model.tvalues[pc]),
            p_value=float(model.pvalues[pc]),
        )
        for pc in scores.columns
    ]
    return out, float(model.rsquared)


def _auto_label(loading_row: pd.Series, names: dict[str, str], k: int = 2) -> str:
    top = loading_row.abs().sort_values(ascending=False).head(k)
    parts = []
    for col in top.index:
        sign = "+" if loading_row[col] >= 0 else "−"
        parts.append(f"{sign}{names.get(col, col)}")
    return " / ".join(parts)


def run_factor_pca(
    factors: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> FactorPCAResult | None:
    names = _display_names()
    legs = _derived_leg_columns()
    monthly = _monthly_columns()

    candidates = [
        c for c in factors.columns
        if any(c.startswith(cat + "_") for cat in MACRO_THEME_CATEGORIES)
        and c not in legs and c not in monthly
    ]

    sub = factors[candidates]
    coverage = sub.notna().mean()
    kept = coverage[coverage >= MIN_COVERAGE].index.tolist()
    dropped = sorted(set(candidates) - set(kept))
    if dropped:
        logger.info("factor PCA: dropped %d low-coverage factors: %s",
                    len(dropped), ", ".join(dropped))
    sub = sub[kept].dropna(how="any")
    if len(sub) < 60 or len(kept) < 4:
        logger.warning("factor PCA: insufficient data (%d obs × %d factors); skipping",
                       len(sub), len(kept))
        return None

    # Z-score each factor, then PCA
    z = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
    z = z.dropna(axis=1, how="any")
    k = min(n_components, z.shape[1])
    pca = PCA(n_components=k, random_state=42)
    scores = pca.fit_transform(z.to_numpy())
    loadings = pd.DataFrame(pca.components_.T, index=z.columns,
                            columns=[f"PC{i+1}" for i in range(k)])

    # Sign convention: flip each PC so its dominant factor loads positive
    for i, pc in enumerate(loadings.columns):
        dom = loadings[pc].abs().idxmax()
        if loadings.loc[dom, pc] < 0:
            loadings[pc] = -loadings[pc]
            scores[:, i] = -scores[:, i]

    # Standardize scores to unit variance so betas read "per 1σ theme move"
    scores = scores / scores.std(axis=0, ddof=0)
    scores_df = pd.DataFrame(scores, index=z.index,
                             columns=loadings.columns)

    labels = [_auto_label(loadings[pc], names) for pc in loadings.columns]
    top_loadings = [
        [
            {"factor": col, "name": names.get(col, col),
             "loading": float(loadings.loc[col, pc])}
            for col in loadings[pc].abs().sort_values(ascending=False).head(5).index
        ]
        for pc in loadings.columns
    ]

    active = (portfolio_returns - benchmark_returns).dropna()
    betas: dict[str, list[ThemeBeta]] = {}
    r2: dict[str, float] = {}
    for key, series in (("portfolio", portfolio_returns),
                        ("benchmark", benchmark_returns),
                        ("active", active)):
        betas[key], r2[key] = _hac_betas(series, scores_df)

    logger.info(
        "factor PCA: %d factors → %d themes (%.0f%% var); active-R² %.3f; "
        "significant active themes: %s",
        z.shape[1], k, 100 * float(np.sum(pca.explained_variance_ratio_)),
        r2["active"],
        ", ".join(f"PC{i+1}" for i, b in enumerate(betas["active"])
                  if b.p_value < 0.10) or "none",
    )

    return FactorPCAResult(
        n_obs=int(len(scores_df)),
        window=(z.index.min().date().isoformat(), z.index.max().date().isoformat()),
        factors=list(z.columns),
        variance_explained=[float(v) for v in pca.explained_variance_ratio_],
        cumulative_variance=[float(v) for v in np.cumsum(pca.explained_variance_ratio_)],
        labels=labels,
        top_loadings=top_loadings,
        betas=betas,
        r_squared=r2,
    )
