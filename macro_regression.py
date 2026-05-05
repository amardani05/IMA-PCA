"""Multi-factor macro regression engine for IMA portfolio + per-stock analysis.

Uses statsmodels OLS with HAC (Newey-West) standard errors at 5-day lag —
appropriate for daily-frequency macro data which typically has both
heteroskedasticity and autocorrelation in the residuals.

Two modes:

* ``mode="curated"`` (default) — one representative factor per category. VIFs
  should stay below 5. Use this for committee-presentable output.
* ``mode="full"`` — kitchen-sink regression on every factor in
  ``MACRO_FACTORS``. Reports VIFs and warns about multicollinearity. Useful for
  diagnostics, NOT for inference.

Public entry points
-------------------
* :func:`run_macro_regression` — single regression
* :func:`per_stock_macro_betas` — beta matrix across portfolio holdings
* :func:`rolling_macro_betas` — 60-day rolling window
* :func:`compute_scenarios` — beta × shock impact decomposition
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from macro_loader import CURATED_FACTORS, MACRO_FACTORS, SCENARIO_SHOCKS

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================
@dataclass
class FactorEstimate:
    factor: str
    beta: float
    std_err: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    significant_05: bool
    significant_01: bool


@dataclass
class RegressionResult:
    """Output bundle from :func:`run_macro_regression` and
    :func:`residualized_macro_betas`.

    For the residualized variant the ``market_beta`` and ``factor_residuals``
    fields are populated and ``methodology`` is set to ``"residualized"``;
    for the raw OLS variant they remain ``None`` / ``"raw_ols"``.
    """
    factors: list[str]
    estimates: dict[str, FactorEstimate]
    alpha: float
    alpha_t: float
    alpha_p: float
    r_squared: float
    adj_r_squared: float
    n_obs: int
    residual_std: float
    vifs: dict[str, float]
    contributions: dict[str, float]
    bonferroni_threshold: float

    methodology: str = "raw_ols"
    frequency: str = "daily"
    market_beta: float | None = None        # legacy: alias for the IJR-excess control beta
    market_beta_t: float | None = None
    market_beta_p: float | None = None
    control_betas: dict[str, float] = field(default_factory=dict)   # all stage-2 control betas
    factor_residuals: pd.DataFrame | None = field(default=None, repr=False)
    sm_result: object | None = field(default=None, repr=False)


# =============================================================================
# Helpers
# =============================================================================
def _select_factors(factors: pd.DataFrame, mode: str) -> list[str]:
    if mode == "curated":
        return [f for f in CURATED_FACTORS if f in factors.columns]
    if mode == "full":
        return list(factors.columns)
    raise ValueError(f"unknown mode: {mode}")


def compute_vifs(X: pd.DataFrame) -> dict[str, float]:
    """Variance inflation factor for each column. Excludes the intercept column."""
    if X.shape[1] < 2:
        return {col: 1.0 for col in X.columns}
    Xc = sm.add_constant(X, has_constant="add").to_numpy()
    out = {}
    cols = ["__const__"] + list(X.columns)
    for i, col in enumerate(cols):
        if col == "__const__":
            continue
        try:
            v = variance_inflation_factor(Xc, i)
        except Exception:
            v = float("nan")
        out[col] = float(v)
    return out


def _align_returns_factors(
    returns: pd.Series, factors: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """Inner-join, drop NaN rows, return aligned (y, X)."""
    df = pd.concat([returns.rename("__ret__"), factors], axis=1).dropna()
    y = df["__ret__"]
    X = df.drop(columns=["__ret__"])
    return y, X


# =============================================================================
# Frequency conversion helpers
# =============================================================================
def to_weekly_returns(daily_returns: pd.Series, end_day: str = "FRI") -> pd.Series:
    """Compound daily simple returns into weekly returns ending Friday."""
    if daily_returns is None or daily_returns.empty:
        return pd.Series(dtype=float)
    weekly = (1.0 + daily_returns).resample(f"W-{end_day}").prod() - 1.0
    return weekly.dropna()


def to_weekly_factor_changes(daily_factors: pd.DataFrame, end_day: str = "FRI") -> pd.DataFrame:
    """Aggregate daily factor changes into weekly.

    For both ``level_change`` and ``log_return`` factors, the weekly value is
    the sum of daily values (additive in both cases). For multiplicative
    ``pct_change`` we'd need to compound, but the loader only emits the two
    additive transforms today.
    """
    if daily_factors is None or daily_factors.empty:
        return pd.DataFrame()
    return daily_factors.resample(f"W-{end_day}").sum().dropna(how="all")


# =============================================================================
# Single regression
# =============================================================================
def run_macro_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    factor_names: list[str] | None = None,
    mode: str = "curated",
    hac_lags: int = 5,
    frequency: str = "daily",
) -> RegressionResult:
    """OLS with HAC (Newey-West) errors.

    ``frequency="weekly"`` aggregates returns and factor changes to Friday
    bars before running the regression and uses ``hac_lags=2`` (weekly
    autocorrelation persists for fewer lags than daily).
    """
    selected = factor_names or _select_factors(factors, mode)
    if frequency == "weekly":
        returns = to_weekly_returns(returns)
        factors = to_weekly_factor_changes(factors[selected])
        hac_lags = min(hac_lags, 2)
    X = factors[selected].copy() if frequency == "daily" else factors.copy()
    y, X = _align_returns_factors(returns, X)

    if len(y) < 30:
        raise ValueError(f"too few observations ({len(y)}) for regression")

    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, Xc, missing="drop")
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    params = fit.params
    bse = fit.bse
    tstats = fit.tvalues
    pvalues = fit.pvalues
    conf = fit.conf_int(alpha=0.05)
    if isinstance(conf, pd.DataFrame):
        conf.columns = ["ci_low", "ci_high"]

    bonferroni = 0.05 / max(len(selected), 1)

    estimates: dict[str, FactorEstimate] = {}
    for f in selected:
        beta = float(params[f])
        se = float(bse[f])
        t = float(tstats[f])
        p = float(pvalues[f])
        lo, hi = float(conf.loc[f, "ci_low"]), float(conf.loc[f, "ci_high"])
        estimates[f] = FactorEstimate(
            factor=f, beta=beta, std_err=se, t_stat=t, p_value=p,
            ci_low=lo, ci_high=hi,
            significant_05=p < 0.05,
            significant_01=p < 0.01,
        )

    contributions = {f: float(params[f] * X[f].sum()) for f in selected}

    return RegressionResult(
        factors=selected,
        estimates=estimates,
        alpha=float(params.get("const", 0.0)),
        alpha_t=float(tstats.get("const", 0.0)),
        alpha_p=float(pvalues.get("const", 1.0)),
        r_squared=float(fit.rsquared),
        adj_r_squared=float(fit.rsquared_adj),
        n_obs=int(fit.nobs),
        residual_std=float(np.sqrt(fit.scale)),
        vifs=compute_vifs(X),
        contributions=contributions,
        bonferroni_threshold=bonferroni,
        methodology="raw_ols",
        frequency=frequency,
        sm_result=fit,
    )


# =============================================================================
# Residualized regression (controls for benchmark / risk-sentiment exposure)
# =============================================================================
def residualized_macro_betas(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    control_factors: pd.DataFrame | pd.Series,
    factor_names: list[str] | None = None,
    mode: str = "curated",
    hac_lags: int = 5,
    frequency: str = "daily",
) -> RegressionResult:
    """Two-stage regression that strips multi-factor common-driver exposure.

    Stage 1 — for every macro factor F_k, regress F_k on ALL control factors.
    The residual is the part of F_k orthogonal to every control.

    Stage 2 — regress portfolio returns on the controls + the Stage-1
    residuals with HAC standard errors. The macro-factor betas from Stage 2
    are economically interpretable as "exposure to factor X holding all the
    common drivers constant."

    ``control_factors`` accepts either a Series (legacy single-factor "v1"
    residualization, i.e. against IJR excess returns only) or a DataFrame
    (v2: e.g. IJR excess + VIX change + HY OAS change). The methodology
    string in the result distinguishes ``residualized_v1`` from
    ``residualized_v2``.

    Why this matters: small-caps and many macro variables co-move via a
    common risk-sentiment factor. Naive OLS treats macro variables as
    exogenous and attributes the shared move to a spurious factor beta
    (the +1.0 USD beta artifact that motivated this rewrite). v1 strips
    market-beta-shared moves; v2 adds VIX and HY OAS, which are the next
    two biggest common drivers in daily small-cap data.
    """
    selected = factor_names or _select_factors(factors, mode)

    # Normalize controls to DataFrame
    if isinstance(control_factors, pd.Series):
        controls_df = control_factors.rename(
            control_factors.name or "benchmark_excess_return"
        ).to_frame()
    else:
        controls_df = control_factors.copy()
    control_cols = list(controls_df.columns)
    methodology = "residualized_v1" if len(control_cols) == 1 else "residualized_v2"

    # Frequency conversion
    if frequency == "weekly":
        portfolio_returns = to_weekly_returns(portfolio_returns)
        controls_df = pd.DataFrame({c: to_weekly_returns(controls_df[c])
                                    for c in control_cols})
        factors = to_weekly_factor_changes(factors[selected])
        hac_lags = min(hac_lags, 2)
    else:
        factors = factors[selected]

    # Inner-join: portfolio + controls + factors
    df = pd.concat([
        portfolio_returns.rename("__portfolio__"),
        controls_df,
        factors,
    ], axis=1).dropna()

    if len(df) < 30:
        raise ValueError(f"residualized regression: only {len(df)} obs after alignment")

    Xc_controls = sm.add_constant(df[control_cols], has_constant="add")

    # Stage 1: residualize each factor against ALL controls
    residualized = pd.DataFrame(index=df.index)
    for f in selected:
        s1 = sm.OLS(df[f], Xc_controls).fit()
        residualized[f] = s1.resid

    # Stage 2: portfolio returns ~ controls + residualized factors
    X = pd.concat([df[control_cols], residualized], axis=1)
    Xc = sm.add_constant(X, has_constant="add")
    y = df["__portfolio__"]

    fit = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    params = fit.params
    bse = fit.bse
    tstats = fit.tvalues
    pvalues = fit.pvalues
    conf = fit.conf_int(alpha=0.05)
    if isinstance(conf, pd.DataFrame):
        conf.columns = ["ci_low", "ci_high"]

    bonferroni = 0.05 / max(len(selected), 1)

    estimates: dict[str, FactorEstimate] = {}
    for f in selected:
        beta = float(params[f])
        se = float(bse[f])
        t = float(tstats[f])
        p = float(pvalues[f])
        lo, hi = float(conf.loc[f, "ci_low"]), float(conf.loc[f, "ci_high"])
        estimates[f] = FactorEstimate(
            factor=f, beta=beta, std_err=se, t_stat=t, p_value=p,
            ci_low=lo, ci_high=hi,
            significant_05=p < 0.05,
            significant_01=p < 0.01,
        )

    contributions = {f: float(params[f] * residualized[f].sum()) for f in selected}

    control_betas = {c: float(params[c]) for c in control_cols}
    # Pick the IJR-excess column as the canonical "market beta" if present,
    # for backward-compat with the existing webapp UI.
    market_beta_key = next(
        (c for c in control_cols if "benchmark" in c.lower() or "ijr" in c.lower()),
        control_cols[0] if control_cols else None,
    )
    market_beta = float(params[market_beta_key]) if market_beta_key else None
    market_beta_t = float(tstats[market_beta_key]) if market_beta_key else None
    market_beta_p = float(pvalues[market_beta_key]) if market_beta_key else None

    return RegressionResult(
        factors=selected,
        estimates=estimates,
        alpha=float(params.get("const", 0.0)),
        alpha_t=float(tstats.get("const", 0.0)),
        alpha_p=float(pvalues.get("const", 1.0)),
        r_squared=float(fit.rsquared),
        adj_r_squared=float(fit.rsquared_adj),
        n_obs=int(fit.nobs),
        residual_std=float(np.sqrt(fit.scale)),
        vifs=compute_vifs(X),
        contributions=contributions,
        bonferroni_threshold=bonferroni,
        methodology=methodology,
        frequency=frequency,
        market_beta=market_beta,
        market_beta_t=market_beta_t,
        market_beta_p=market_beta_p,
        control_betas=control_betas,
        factor_residuals=residualized,
        sm_result=fit,
    )


# =============================================================================
# Standard control set (Project 9 v2)
# =============================================================================
def build_control_factors(
    benchmark_excess_return: pd.Series,
    macro_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Build the standard 3-factor control set for v2 residualization.

    - ``benchmark_excess_return`` (IJR excess) — proxy for market beta
    - VIX daily change — proxy for volatility regime
    - HY OAS daily change — proxy for credit cycle

    These three are the dominant common-factor drivers in daily small-cap
    returns. Macro betas measured orthogonal to all three are the cleanest
    available estimate of true economic exposure.

    Returns a DataFrame indexed on dates where every control is present.
    """
    controls = pd.DataFrame(index=benchmark_excess_return.index)
    controls["benchmark_excess_return"] = benchmark_excess_return

    vix_col = next(
        (c for c in macro_factors.columns if "VIXCLS" in c or c.lower().endswith("_vix")),
        None,
    )
    if vix_col:
        controls["vix_change"] = macro_factors[vix_col]
    else:
        logger.warning(
            "VIX not in macro factors; control set will exclude vol regime"
        )

    hy_col = next(
        (c for c in macro_factors.columns
         if "BAMLH0A0HYM2" in c or "hy_oas" in c.lower()),
        None,
    )
    if hy_col:
        controls["hy_spread_change"] = macro_factors[hy_col]
    else:
        logger.warning(
            "HY OAS not in macro factors; control set will exclude credit cycle"
        )

    return controls.dropna()


def _interpret_delta(raw_beta: float, res_beta: float, res_pval: float) -> str:
    """Human-readable note on what the change between raw and residualized means."""
    if abs(raw_beta) > 0.3 and abs(res_beta) < 0.1 and res_pval > 0.10:
        return "Raw beta was risk-sentiment artifact (residualized → no real exposure)"
    if abs(raw_beta - res_beta) < 0.05:
        return "Genuine factor exposure (robust to risk-sentiment control)"
    if (raw_beta > 0) != (res_beta > 0):
        return "Sign flipped after residualization (severe contamination)"
    if abs(raw_beta - res_beta) > 0.2:
        return "Substantial risk-sentiment contamination"
    return "Partial risk-sentiment contamination"


def compare_raw_vs_residualized(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    benchmark_excess_return: pd.Series,
    factor_names: list[str] | None = None,
    mode: str = "curated",
    frequency: str = "daily",
) -> pd.DataFrame:
    """Two-way (raw vs single-factor residualized) comparison. Kept for
    back-compat; new code should use :func:`compare_residualization_approaches`."""
    raw = run_macro_regression(
        portfolio_returns, factors, factor_names=factor_names,
        mode=mode, frequency=frequency,
    )
    res = residualized_macro_betas(
        portfolio_returns, factors, benchmark_excess_return,
        factor_names=factor_names, mode=mode, frequency=frequency,
    )

    rows = []
    for fname in raw.factors:
        rb = raw.estimates[fname].beta
        rb_p = raw.estimates[fname].p_value
        sb = res.estimates[fname].beta if fname in res.estimates else None
        sb_p = res.estimates[fname].p_value if fname in res.estimates else None
        rows.append({
            "factor": fname,
            "raw_beta": rb,
            "raw_p": rb_p,
            "residualized_beta": sb,
            "residualized_p": sb_p,
            "delta": (sb - rb) if (sb is not None) else None,
            "interpretation": _interpret_delta(rb, sb, sb_p) if sb is not None else "",
        })
    return pd.DataFrame(rows)


# =============================================================================
# Three-way comparison: raw / v1 (market only) / v2 (market + vol + credit)
# =============================================================================
def _interpret_progression(raw, v1, v2, v2_p) -> str:
    if v2 is None:
        return ""
    if abs(v2) < 0.05 and v2_p > 0.10:
        return "Common-factor artifact (no real exposure after multi-factor control)"
    if raw is not None and abs(raw - v2) < 0.05:
        return "Robust exposure (stable across all residualization approaches)"
    if raw is not None and abs(raw) > abs(v2) * 2:
        return "Substantial common-factor contamination (multi-factor reveals smaller true exposure)"
    if raw is not None and ((raw > 0) != (v2 > 0)):
        return "Sign reversal (raw OLS picked up wrong direction; multi-resid corrects)"
    return "Modest contamination (multi-resid materially changes magnitude)"


def compare_residualization_approaches(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    benchmark_excess_return: pd.Series,
    controls: pd.DataFrame,
    factor_names: list[str] | None = None,
    mode: str = "curated",
    frequency: str = "daily",
) -> tuple[pd.DataFrame, "RegressionResult", "RegressionResult", "RegressionResult"]:
    """Run THREE versions of the macro regression and return a progression table:

    1. Raw OLS (no residualization) — what the user sees if you don't think.
    2. Single-factor residualized v1 — controls for IJR-excess only.
    3. Multi-factor residualized v2 — controls for IJR + VIX + HY OAS.

    A factor with stable beta across all three is a genuine exposure. A
    factor whose beta shrinks toward zero as controls accumulate was just
    capturing common-factor confound.

    Returns ``(comparison_df, raw_result, v1_result, v2_result)`` so callers
    can persist all three regressions to JSON without re-running them.
    """
    raw = run_macro_regression(
        portfolio_returns, factors, factor_names=factor_names,
        mode=mode, frequency=frequency,
    )
    v1 = residualized_macro_betas(
        portfolio_returns, factors, benchmark_excess_return,
        factor_names=factor_names, mode=mode, frequency=frequency,
    )
    v2 = residualized_macro_betas(
        portfolio_returns, factors, controls,
        factor_names=factor_names, mode=mode, frequency=frequency,
    )

    rows = []
    for fname in raw.factors:
        rb = raw.estimates[fname].beta
        rb_p = raw.estimates[fname].p_value
        v1b = v1.estimates[fname].beta if fname in v1.estimates else None
        v1p = v1.estimates[fname].p_value if fname in v1.estimates else None
        v2b = v2.estimates[fname].beta if fname in v2.estimates else None
        v2p = v2.estimates[fname].p_value if fname in v2.estimates else None
        rows.append({
            "factor": fname,
            "raw_beta": rb, "raw_p": rb_p,
            "v1_beta": v1b, "v1_p": v1p,
            "v2_beta": v2b, "v2_p": v2p,
            "interpretation": _interpret_progression(rb, v1b, v2b, v2p),
        })
    return pd.DataFrame(rows), raw, v1, v2


# =============================================================================
# Per-stock betas
# =============================================================================
def per_stock_macro_betas(
    stock_returns: pd.DataFrame,
    factors: pd.DataFrame,
    mode: str = "curated",
    hac_lags: int = 5,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Run the same regression on each column of ``stock_returns``.

    Returns a long-form DataFrame (rows = ticker × factor) with beta, t-stat,
    p-value, plus a wide pivot is exposed via :func:`pivot_betas`.
    """
    selected = _select_factors(factors, mode)
    X = factors[selected].copy()

    rows = []
    for tk in stock_returns.columns:
        y = stock_returns[tk].dropna()
        if len(y) < 30:
            continue
        try:
            res = run_macro_regression(y, X, factor_names=selected,
                                       mode=mode, hac_lags=hac_lags,
                                       frequency=frequency)
        except Exception as exc:  # noqa: BLE001
            logger.warning("per-stock regression failed for %s: %s", tk, exc)
            continue
        for f, est in res.estimates.items():
            rows.append({
                "Ticker": tk,
                "factor": f,
                "beta": est.beta,
                "std_err": est.std_err,
                "t_stat": est.t_stat,
                "p_value": est.p_value,
                "significant_05": est.significant_05,
            })

    return pd.DataFrame(rows)


def pivot_betas(per_stock: pd.DataFrame, value: str = "beta") -> pd.DataFrame:
    """Wide pivot: rows = Ticker, cols = factor, values = beta (or t/p)."""
    if per_stock.empty:
        return pd.DataFrame()
    return per_stock.pivot(index="Ticker", columns="factor", values=value)


# =============================================================================
# Rolling betas
# =============================================================================
def rolling_macro_betas(
    returns: pd.Series,
    factors: pd.DataFrame,
    mode: str = "curated",
    window: int = 60,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Rolling-window OLS at the portfolio level. Returns DataFrame (date × factor)."""
    selected = _select_factors(factors, mode)
    if frequency == "weekly":
        returns = to_weekly_returns(returns)
        factors = to_weekly_factor_changes(factors[selected])
        # Rolling window in weekly bars: shrink default 60d to 12w (~quarter)
        if window >= 60:
            window = 12
    X = factors[selected].copy() if frequency == "daily" else factors.copy()
    y, X = _align_returns_factors(returns, X)
    Xc = sm.add_constant(X, has_constant="add")

    if len(y) < window:
        return pd.DataFrame(columns=selected)

    rows = []
    for i in range(window, len(y) + 1):
        y_win = y.iloc[i - window:i]
        X_win = Xc.iloc[i - window:i]
        try:
            fit = sm.OLS(y_win, X_win).fit()
            row = {"date": y.index[i - 1]}
            for f in selected:
                row[f] = float(fit.params[f])
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.debug("rolling window @ %s failed: %s", y.index[i - 1], exc)
            continue

    if not rows:
        return pd.DataFrame(columns=selected)

    df = pd.DataFrame(rows).set_index("date")
    return df


# =============================================================================
# Scenarios
# =============================================================================
def compute_scenarios(
    result: RegressionResult,
    shocks: dict[str, tuple[str, float]] | None = None,
) -> list[dict]:
    """For each curated factor, ``beta × shock = portfolio impact``."""
    shocks = shocks or SCENARIO_SHOCKS
    out = []
    for f in result.factors:
        if f not in shocks:
            continue
        label, size = shocks[f]
        est = result.estimates[f]
        impact = est.beta * size
        out.append({
            "factor": f,
            "label": label,
            "shock": size,
            "beta": est.beta,
            "impact": impact,
            "significant": est.significant_05,
            "p_value": est.p_value,
        })
    out.sort(key=lambda r: abs(r["impact"]), reverse=True)
    return out


# =============================================================================
# Excess returns
# =============================================================================
def excess_returns(returns: pd.Series, rf_daily: pd.Series | None = None) -> pd.Series:
    """Subtract a daily risk-free rate. If ``rf_daily`` is None, returns are used as-is."""
    if rf_daily is None:
        return returns
    aligned = returns.subtract(rf_daily.reindex(returns.index).fillna(0.0))
    return aligned


# =============================================================================
# Helpers consumed by main.py
# =============================================================================
def factor_metadata() -> list[dict]:
    """Flat metadata list for export (category, series_id, name, transform, source)."""
    rows = []
    for category, defs in MACRO_FACTORS.items():
        for series_id, defn in defs.items():
            rows.append({
                "factor": f"{category}_{series_id}",
                "category": category,
                "series_id": series_id,
                "name": defn["name"],
                "source": defn["source"],
                "transform": defn["transform"],
                "in_curated": f"{category}_{series_id}" in CURATED_FACTORS,
            })
    return rows
