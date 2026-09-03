"""Export macro-factor regression outputs as JSON for the React webapp.

Files written into ``webapp/public/data/macro/`` so the dashboard can load them
without a server. The ``stock_betas.json`` matrix is the key payload — the
webapp uses it to recompute portfolio betas client-side as the user toggles
holdings on and off in the portfolio tree.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from macro_loader import CURATED_FACTORS, MACRO_FACTORS, SCENARIO_SHOCKS
from macro_regression import (
    RegressionResult,
    factor_metadata,
    pivot_betas,
)

logger = logging.getLogger(__name__)

MACRO_OUT_DIR: Path = config.PROJECT_ROOT / "webapp" / "public" / "data" / "macro"


def _ensure_dir() -> None:
    MACRO_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_jsonable(obj):
    """Recursive NaN/Timestamp -> JSON-safe."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


def _write_json(payload, name: str) -> Path:
    _ensure_dir()
    path = MACRO_OUT_DIR / name
    path.write_text(json.dumps(_to_jsonable(payload), default=str, separators=(",", ":")))
    logger.info("wrote %s (%d bytes)", path.name, path.stat().st_size)
    return path


# =============================================================================
# Individual exporters
# =============================================================================
def export_factor_metadata(category_overrides: dict | None = None) -> None:
    meta = factor_metadata()
    payload = {
        "factors": meta,
        "categories": list(MACRO_FACTORS.keys()),
        "curated_factors": CURATED_FACTORS,
        "scenario_shocks": {
            f: {"label": label, "shock": shock}
            for f, (label, shock) in SCENARIO_SHOCKS.items()
        },
    }
    _write_json(payload, "factor_metadata.json")


def _result_to_dict(result: RegressionResult, label: str) -> dict:
    return {
        "label": label,
        "methodology": getattr(result, "methodology", "raw_ols"),
        "frequency": getattr(result, "frequency", "daily"),
        "market_beta": getattr(result, "market_beta", None),
        "market_beta_t": getattr(result, "market_beta_t", None),
        "market_beta_p": getattr(result, "market_beta_p", None),
        "factors": result.factors,
        "n_obs": result.n_obs,
        "r_squared": result.r_squared,
        "adj_r_squared": result.adj_r_squared,
        "alpha": result.alpha,
        "alpha_t": result.alpha_t,
        "alpha_p": result.alpha_p,
        "residual_std": result.residual_std,
        "bonferroni_threshold": result.bonferroni_threshold,
        "betas": {
            f: {
                "factor": e.factor,
                "beta": e.beta,
                "std_err": e.std_err,
                "t_stat": e.t_stat,
                "p_value": e.p_value,
                "ci_low": e.ci_low,
                "ci_high": e.ci_high,
                "significant_05": bool(e.significant_05),
                "significant_01": bool(e.significant_01),
            }
            for f, e in result.estimates.items()
        },
        "vifs": result.vifs,
        "contributions": result.contributions,
    }


def export_portfolio_betas(curated: RegressionResult) -> None:
    _write_json(_result_to_dict(curated, "Curated"), "portfolio_betas.json")


def export_portfolio_betas_full(full: RegressionResult | None) -> None:
    if full is None:
        _write_json({"label": "Full", "skipped": True}, "portfolio_betas_full.json")
        return
    _write_json(_result_to_dict(full, "Kitchen-sink"), "portfolio_betas_full.json")


def _stock_betas_payload(per_stock_long: pd.DataFrame) -> dict:
    """Wide-form payload for the webapp's StockBetaMatrix consumer."""
    if per_stock_long.empty:
        return {"tickers": [], "factors": [], "betas": {}, "p_values": {}, "std_errors": {}}

    beta_wide = pivot_betas(per_stock_long, "beta").fillna(0.0)
    p_wide = pivot_betas(per_stock_long, "p_value").fillna(1.0)
    se_wide = pivot_betas(per_stock_long, "std_err").fillna(0.0)

    return {
        "tickers": list(beta_wide.index),
        "factors": list(beta_wide.columns),
        "betas": {
            tk: {f: float(beta_wide.loc[tk, f]) for f in beta_wide.columns}
            for tk in beta_wide.index
        },
        "p_values": {
            tk: {f: float(p_wide.loc[tk, f]) for f in p_wide.columns}
            for tk in p_wide.index
        },
        "std_errors": {
            tk: {f: float(se_wide.loc[tk, f]) for f in se_wide.columns}
            for tk in se_wide.index
        },
    }


def export_stock_betas(per_stock_long: pd.DataFrame) -> None:
    """Per-stock beta matrix used by the webapp for live recomputation."""
    _write_json(_stock_betas_payload(per_stock_long), "stock_betas.json")


def export_timeframes(timeframe_results: dict[str, dict] | None) -> None:
    """Pre-computed per-timeframe regression bundles (YTD / 6M / 1Y / 2Y / MAX).

    Each entry has its own raw / v1 / v2 result, scenarios, comparison rows,
    and stock-level betas — so the webapp can swap them in instantly without
    re-running OLS in the browser.
    """
    if not timeframe_results:
        _write_json({"timeframes": [], "default": "max", "by_timeframe": {}},
                    "timeframes.json")
        return
    payload = {
        "timeframes": list(timeframe_results.keys()),
        "default": "max" if "max" in timeframe_results else next(iter(timeframe_results)),
        "by_timeframe": timeframe_results,
    }
    _write_json(payload, "timeframes.json")


def export_rolling_betas(rolling: pd.DataFrame) -> None:
    if rolling is None or rolling.empty:
        _write_json({"dates": [], "factors": [], "series": {}}, "rolling_betas.json")
        return
    payload = {
        "dates": [d.isoformat() for d in rolling.index],
        "factors": list(rolling.columns),
        "series": {
            f: [None if pd.isna(v) else float(v) for v in rolling[f].values]
            for f in rolling.columns
        },
    }
    _write_json(payload, "rolling_betas.json")


def export_factor_returns(factors: pd.DataFrame, n_recent: int | None = 504) -> None:
    """Time series of (transformed) factor returns. Truncated to ``n_recent`` to
    keep the JSON small; default ~2 trading years."""
    df = factors.tail(n_recent) if n_recent else factors
    payload = {
        "dates": [d.isoformat() for d in df.index],
        "factors": list(df.columns),
        "series": {
            f: [None if pd.isna(v) else float(v) for v in df[f].values]
            for f in df.columns
        },
    }
    _write_json(payload, "factor_returns.json")


def export_portfolio_returns(returns: pd.Series, n_recent: int | None = 504) -> None:
    s = returns.tail(n_recent) if n_recent else returns
    payload = {
        "dates": [d.isoformat() for d in s.index],
        "returns": [None if pd.isna(v) else float(v) for v in s.values],
    }
    _write_json(payload, "portfolio_returns.json")


def export_scenarios(scenarios: list[dict]) -> None:
    _write_json({"scenarios": scenarios}, "scenarios.json")


def export_factor_contributions(curated: RegressionResult) -> None:
    rows = []
    for f, contrib in curated.contributions.items():
        est = curated.estimates[f]
        rows.append({
            "factor": f,
            "beta": est.beta,
            "contribution": contrib,
            "significant_05": bool(est.significant_05),
        })
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    _write_json({"contributions": rows}, "factor_contributions.json")


# =============================================================================
# Top-level entry
# =============================================================================
def export_portfolio_betas_raw(raw: RegressionResult | None) -> None:
    """The raw OLS regression — exposed for transparency and the diagnostic toggle."""
    if raw is None:
        _write_json({"label": "Raw OLS", "skipped": True}, "portfolio_betas_raw.json")
        return
    _write_json(_result_to_dict(raw, "Raw OLS"), "portfolio_betas_raw.json")


def export_portfolio_betas_v1(v1: RegressionResult | None) -> None:
    """The single-factor (IJR-only) residualized regression."""
    if v1 is None:
        _write_json({"label": "Residualized v1", "skipped": True}, "portfolio_betas_v1.json")
        return
    _write_json(_result_to_dict(v1, "Residualized v1 (market only)"), "portfolio_betas_v1.json")


def export_comparison(comparison: pd.DataFrame | None) -> None:
    """Three-way (raw / v1 / v2) comparison — also kept under the v1-vs-v2
    legacy filename for the existing webapp code path."""
    if comparison is None or comparison.empty:
        _write_json({"rows": []}, "raw_vs_residualized.json")
        return
    df = comparison.replace({np.nan: None})
    payload = {"rows": df.to_dict(orient="records")}
    _write_json(payload, "raw_vs_residualized.json")


def export_index_vs_active(
    index_raw: RegressionResult | None,
    portfolio_raw: RegressionResult | None,
    active_v2: RegressionResult,
) -> None:
    """Side-by-side per-factor table: what the INDEX carries (raw), what the
    portfolio carries in total (raw), and the ACTIVE exposure (v2, net of
    index/VIX/credit). total ≈ index-inherited + active."""
    rows = []
    for f in active_v2.factors:
        row = {"factor": f}
        for key, res in (("index", index_raw), ("portfolio_total", portfolio_raw),
                         ("active", active_v2)):
            est = res.estimates.get(f) if res is not None else None
            row[key] = (
                {"beta": est.beta, "t_stat": est.t_stat, "p_value": est.p_value}
                if est is not None else None
            )
        rows.append(row)
    _write_json({"rows": rows,
                 "note": "index & portfolio_total are raw OLS; active is v2 "
                         "residualized (net of IJR + VIX + HY OAS)"},
                "index_vs_active.json")


def export_universe_factor_betas(
    per_stock_long: pd.DataFrame,
    index_betas: dict[str, float] | None,
) -> None:
    """Factor betas for EVERY scored stock — the bridge that turns the screener
    into a factor tool.

    These are raw (unconditional) per-stock betas, which is what makes them
    composable: a portfolio's raw beta is the weight-average of its holdings'
    raw betas, so the webapp can answer "what does adding this name do to our
    exposure?" exactly. ``index_betas`` (IJR raw) travels alongside so the UI
    can show each stock's exposure *relative to the benchmark*.
    """
    if per_stock_long.empty:
        _write_json({"available": False, "tickers": [], "factors": [],
                     "betas": {}, "index_betas": {}}, "universe_factor_betas.json")
        return

    beta_wide = pivot_betas(per_stock_long, "beta")
    p_wide = pivot_betas(per_stock_long, "p_value")
    factors = list(beta_wide.columns)

    payload = {
        "available": True,
        "tickers": list(beta_wide.index),
        "factors": factors,
        "index_betas": {f: float(v) for f, v in (index_betas or {}).items() if f in factors},
        # rounded to keep the payload small — 4dp is far finer than the
        # standard errors on these estimates
        "betas": {
            tk: [None if pd.isna(beta_wide.loc[tk, f]) else round(float(beta_wide.loc[tk, f]), 4)
                 for f in factors]
            for tk in beta_wide.index
        },
        "p_values": {
            tk: [None if pd.isna(p_wide.loc[tk, f]) else round(float(p_wide.loc[tk, f]), 3)
                 for f in factors]
            for tk in p_wide.index
        },
    }
    _write_json(payload, "universe_factor_betas.json")


def export_attribution(attribution: dict | None) -> None:
    """Active-return attribution (factor contributions + selection)."""
    _write_json(attribution or {"available": False}, "attribution.json")


def export_factor_pca(result) -> None:
    """``result`` is a factor_pca.FactorPCAResult (or None to clear)."""
    payload = result.to_dict() if result is not None else {"available": False}
    if result is not None:
        payload["available"] = True
    _write_json(payload, "factor_pca.json")


def export_macro_to_webapp(
    portfolio_curated: RegressionResult,
    stock_betas: pd.DataFrame,
    rolling_betas: pd.DataFrame,
    scenarios: list[dict],
    macro_factors: pd.DataFrame,
    portfolio_returns: pd.Series,
    portfolio_full: RegressionResult | None = None,
    portfolio_raw: RegressionResult | None = None,
    portfolio_v1: RegressionResult | None = None,
    comparison: pd.DataFrame | None = None,
    timeframe_results: dict[str, dict] | None = None,
) -> None:
    _ensure_dir()
    export_factor_metadata()
    export_portfolio_betas(portfolio_curated)        # the v2 (or best available) result
    export_portfolio_betas_v1(portfolio_v1)
    export_portfolio_betas_full(portfolio_full)
    export_portfolio_betas_raw(portfolio_raw)
    export_comparison(comparison)
    export_stock_betas(stock_betas)
    export_rolling_betas(rolling_betas)
    export_factor_returns(macro_factors)
    export_portfolio_returns(portfolio_returns)
    export_scenarios(scenarios)
    export_factor_contributions(portfolio_curated)
    export_timeframes(timeframe_results)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "methodology": getattr(portfolio_curated, "methodology", "raw_ols"),
        "frequency": getattr(portfolio_curated, "frequency", "daily"),
        "market_beta": getattr(portfolio_curated, "market_beta", None),
        "n_factors_curated": len(portfolio_curated.factors),
        "n_obs": portfolio_curated.n_obs,
        "r_squared": portfolio_curated.r_squared,
        "alpha_annualized": portfolio_curated.alpha * 252,
        "max_vif": max(portfolio_curated.vifs.values()) if portfolio_curated.vifs else None,
    }
    _write_json(summary, "macro_summary.json")
    logger.info("Macro JSON exports complete in %s", MACRO_OUT_DIR)
