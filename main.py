"""IMA Torpedo Screener — orchestrator.

Pipeline:
  1. Fetch S&P 600 constituent list.
  2. Download/cache prices and quarterly fundamentals for the universe.
  3. Fetch SEC 8-K counts.
  4. Compute the 14-feature risk matrix; impute; drop sparse rows.
  5. Z-score → PCA (N_PCA_COMPONENTS) with auto-labeled PCs.
  6. k-means over k={3..7} with silhouette-based selection.
  7. Cluster characterization and tier-labeling by composite risk.
  8. Per-stock composite score (0-100 percentile mean) with tier buckets.
  9. IMA portfolio report and contrarian opportunity screen.
 10. (Optional) Trajectory mapping through PC space over recent quarters.
 11. All charts and CSVs into ``output/``. Terminal-committee summary.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import interactive_charts as ichart
import visualization as viz
import webapp_export
from feature_engine import (
    build_features,
    clean_features,
    fetch_filing_counts,
    fetch_fundamentals,
    fetch_prices,
    save_features,
)
from pca_cluster import (
    nearest_cluster_distance,
    run_clustering,
    run_pca,
)
from scoring import (
    build_portfolio_report,
    compute_composite_scores,
    opportunity_screen,
)
from trajectory import compute_trajectories
from universe import get_sp600_universe

logger = logging.getLogger("torpedo")


# =============================================================================
# Logging
# =============================================================================
def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("yfinance", "urllib3", "matplotlib", "PIL", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# =============================================================================
# Pipeline
# =============================================================================
def run_pipeline(args: argparse.Namespace) -> None:
    t_start = time.time()

    # 1. Universe
    logger.info("=== STEP 1: S&P 600 universe ===")
    universe_df = get_sp600_universe(force_refresh=args.refresh)
    tickers = sorted(set(universe_df["Ticker"].tolist()) | set(config.PORTFOLIO))
    universe_df = universe_df[universe_df["Ticker"].isin(tickers)].copy()
    # Ensure every portfolio ticker has a row
    missing = [t for t in config.PORTFOLIO if t not in universe_df["Ticker"].values]
    if missing:
        pad = pd.DataFrame({
            "Ticker": missing, "Company": missing, "Sector": "Unknown", "Industry": "Unknown"
        })
        universe_df = pd.concat([universe_df, pad], ignore_index=True)
    logger.info("Universe size: %d (incl. %d portfolio names)",
                len(universe_df), len(config.PORTFOLIO))

    if args.portfolio_only:
        tickers = list(config.PORTFOLIO.keys())
        universe_df = universe_df[universe_df["Ticker"].isin(tickers)]
        logger.info("Portfolio-only mode: %d tickers", len(tickers))

    # 2. Prices
    logger.info("=== STEP 2: Prices ===")
    prices = fetch_prices(
        tickers,
        lookback_days=config.PRICE_LOOKBACK_DAYS,
        force_refresh=args.refresh and not args.refresh_features,
    )

    # 3. Fundamentals
    logger.info("=== STEP 3: Fundamentals ===")
    bundles = fetch_fundamentals(tickers, force_refresh=args.refresh)

    # 4. SEC 8-K counts
    logger.info("=== STEP 4: SEC EDGAR 8-K counts ===")
    try:
        filing_counts = fetch_filing_counts(tickers, days=90)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SEC filing count fetch failed: %s", exc)
        filing_counts = {}

    # 5. Feature matrix
    logger.info("=== STEP 5: Feature computation ===")
    raw_features = build_features(
        universe_df, bundles, prices,
        filing_counts=filing_counts,
    )
    clean, _imputed = clean_features(raw_features)
    save_features(clean)
    logger.info("Final feature matrix: %d stocks × %d features",
                len(clean), len(config.FEATURES))

    # 6. PCA
    logger.info("=== STEP 6: PCA ===")
    pca_result = run_pca(clean)

    # 7. Clustering
    logger.info("=== STEP 7: Clustering ===")
    cluster_result = run_clustering(clean, pca_result, override_k=args.clusters)

    # 8. Composite scoring
    logger.info("=== STEP 8: Composite scoring ===")
    percentile_ranks = compute_composite_scores(clean)

    # Short-circuit for pitch assessments — everything downstream of this
    # (trajectory / charts / macro / webapp export / drift table) is irrelevant
    # for a single-ticker assessment and only adds latency.
    if args.assess or args.assess_batch:
        _run_pitch_mode(args, universe_df, clean, pca_result, cluster_result)
        return

    # 9. Trajectory (portfolio-only — fast)
    trajectory = None
    if not args.no_trajectory:
        logger.info("=== STEP 9: Trajectory (portfolio-only) ===")
        trajectory = compute_trajectories(
            universe_df, bundles, prices, pca_result, cluster_result,
            tickers_filter=list(config.PORTFOLIO.keys()),
        )

    # 10. Portfolio report
    logger.info("=== STEP 10: Portfolio report ===")
    port_report = build_portfolio_report(
        clean, percentile_ranks, cluster_result.assignments,
        cluster_result.tier_labels, trajectory,
    )

    # 11. Opportunity screen
    logger.info("=== STEP 11: Opportunity screen ===")
    opportunities = opportunity_screen(clean, percentile_ranks, limit=25)

    # 12. Drift alerts
    logger.info("=== STEP 12: Drift alerts ===")
    drift_alerts = _build_drift_alerts(pca_result, cluster_result, trajectory)

    # 13. CSV exports
    logger.info("=== STEP 13: CSV exports ===")
    _export_csvs(
        clean, pca_result, cluster_result, percentile_ranks,
        port_report, opportunities, trajectory, drift_alerts,
    )

    # 14. Charts (static matplotlib)
    logger.info("=== STEP 14: Static charts ===")
    _make_charts(
        clean, pca_result, cluster_result, percentile_ranks,
        port_report, trajectory,
    )

    # 15. Interactive plotly charts
    logger.info("=== STEP 15: Interactive charts ===")
    try:
        ichart.plot_all_2d_pc_scatters(
            clean, pca_result.scores, cluster_result, percentile_ranks,
            pca_result.pc_labels, list(config.PORTFOLIO.keys()), trajectory,
        )
        ichart.plot_3d_pc_scatter(
            clean, pca_result.scores, cluster_result, percentile_ranks,
            pca_result.pc_labels, list(config.PORTFOLIO.keys()), trajectory,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Interactive chart generation failed: %s", exc)

    # 16. Macro factor regression
    macro_summary = None
    if not args.no_macro:
        logger.info("=== STEP 16: Macro factor regression ===")
        try:
            macro_summary = _run_macro_pipeline(prices, force_refresh=args.refresh)
        except EnvironmentError as exc:
            logger.warning("Skipping macro analysis: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Macro regression failed: %s", exc)

    # 17. Webapp JSON + asset export
    if not args.no_webapp:
        logger.info("=== STEP 17: Webapp export ===")
        try:
            webapp_export.export_all(
                clean, pca_result, cluster_result, percentile_ranks,
                port_report, opportunities, drift_alerts, trajectory,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Webapp export failed: %s", exc)

    # 18. Terminal report
    _print_report(
        clean, pca_result, cluster_result, percentile_ranks,
        port_report, opportunities, drift_alerts, trajectory,
    )
    if macro_summary is not None:
        _print_macro_report(macro_summary)

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1fs", elapsed)


# =============================================================================
# Pitch assessment short-circuit
# =============================================================================
def _run_pitch_mode(
    args: argparse.Namespace,
    universe_df: pd.DataFrame,
    clean: pd.DataFrame,
    pca_result,
    cluster_result,
) -> None:
    """Generate pitch assessment(s) and write JSON to output/, then exit."""
    from pitch_assessor import assess_batch, assess_pitch

    output_dir = config.OUTPUT_DIR
    webapp_pitch_dir = config.PROJECT_ROOT / "webapp" / "public" / "data" / "pitches"
    webapp_pitch_dir.mkdir(parents=True, exist_ok=True)

    if args.assess:
        try:
            a = assess_pitch(
                ticker=args.assess.upper(),
                pca_result=pca_result,
                cluster_result=cluster_result,
                torpedo_features=clean,
                portfolio=config.PORTFOLIO,
                universe_meta=universe_df,
            )
        except ValueError as exc:
            print(f"\n{exc}\n")
            return
        print(a.format_text())
        a.export_json(output_dir / f"pitch_{a.ticker}.json")
        a.export_json(webapp_pitch_dir / f"{a.ticker}.json")
        _refresh_pitch_index(webapp_pitch_dir)
        logger.info("Wrote %s and webapp copy", output_dir / f"pitch_{a.ticker}.json")

    if args.assess_batch:
        path = Path(args.assess_batch)
        if not path.exists():
            print(f"Batch file not found: {path}")
            return
        tickers = [
            line.strip().upper() for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        logger.info("Assessing %d tickers from %s", len(tickers), path)
        results = assess_batch(
            tickers=tickers,
            pca_result=pca_result,
            cluster_result=cluster_result,
            torpedo_features=clean,
            portfolio=config.PORTFOLIO,
            universe_meta=universe_df,
            output_dir=output_dir,
        )
        # Webapp copies
        for a in results:
            a.export_json(webapp_pitch_dir / f"{a.ticker}.json")
        _refresh_pitch_index(webapp_pitch_dir)

        # Print a one-line summary per assessment
        print("\n" + "=" * 76)
        print(f" BATCH PITCH ASSESSMENTS ({len(results)} succeeded)")
        print("=" * 76)
        for a in results:
            print(f"  {a.ticker:<6}  {a.recommendation:<22}  "
                  f"score={a.composite_risk_score:>4.0f}  "
                  f"tier={a.cluster_tier:<10}  "
                  f"div={a.diversification_score:>4.0f}  "
                  f"({a.sector})")
        print("=" * 76)


def _refresh_pitch_index(pitch_dir: Path) -> None:
    """Build/refresh ``pitches/index.json`` listing every available assessment."""
    import json as _json
    rows = []
    for p in sorted(pitch_dir.glob("*.json")):
        if p.name == "index.json":
            continue
        try:
            payload = _json.loads(p.read_text())
            rows.append({
                "ticker": payload.get("ticker"),
                "company_name": payload.get("company_name"),
                "sector": payload.get("sector"),
                "recommendation": payload.get("recommendation"),
                "composite_risk_score": payload.get("composite_risk_score"),
                "cluster_tier": payload.get("cluster_tier"),
                "generated_at": payload.get("generated_at"),
            })
        except Exception:
            continue
    (pitch_dir / "index.json").write_text(_json.dumps({"pitches": rows}, indent=2))


# =============================================================================
# Macro pipeline
# =============================================================================
def _build_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Daily simple returns for ``tickers`` from the cached price MultiIndex frame."""
    out = {}
    for tk in tickers:
        if (tk, "Close") not in prices.columns:
            continue
        s = prices[(tk, "Close")].dropna().sort_index()
        if len(s) < 30:
            continue
        out[tk] = s.pct_change().dropna()
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).dropna(how="all")


def _portfolio_returns(stock_returns: pd.DataFrame) -> pd.Series:
    """Weighted-sum portfolio return using PORTFOLIO weights (renormalized)."""
    weights = pd.Series(config.PORTFOLIO).reindex(stock_returns.columns).dropna()
    if weights.empty:
        return pd.Series(dtype=float)
    weights = weights / weights.sum()
    rets = stock_returns[weights.index].fillna(0.0) @ weights
    rets.name = "portfolio_return"
    return rets


def _benchmark_excess_returns(
    benchmark_ticker: str,
    aligned_to: pd.DatetimeIndex,
    force_refresh: bool = False,
) -> pd.Series:
    """Fetch IJR (or whatever ``BENCHMARK_TICKER`` is) and DGS3MO, return
    daily excess return series aligned to the portfolio's index."""
    from feature_engine import fetch_prices
    from macro_loader import FredClient, _fred_api_key

    bench_prices = fetch_prices(
        [benchmark_ticker],
        lookback_days=config.PRICE_LOOKBACK_DAYS,
        force_refresh=force_refresh,
    )
    if bench_prices is None or bench_prices.empty:
        raise RuntimeError(f"could not fetch {benchmark_ticker} prices")
    if (benchmark_ticker, "Close") not in bench_prices.columns:
        raise RuntimeError(f"{benchmark_ticker} Close not in price frame")
    bench = bench_prices[(benchmark_ticker, "Close")].dropna().sort_index()
    bench_ret = bench.pct_change().dropna()

    # Subtract daily risk-free (3-month T-bill annualized) if FRED is available.
    rf_daily = None
    if _fred_api_key(required=False):
        try:
            fred = FredClient()
            rf_annual_pct = fred.get_series(
                "DGS3MO", aligned_to.min().date().isoformat(), None,
            )
            rf_daily = (rf_annual_pct / 100.0 / 252.0).reindex(bench_ret.index).ffill(limit=5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DGS3MO fetch failed: %s — using zero risk-free", exc)
            rf_daily = None

    if rf_daily is not None:
        excess = bench_ret.subtract(rf_daily.fillna(0.0), fill_value=0.0)
    else:
        excess = bench_ret  # ~5%/252 is small enough to ignore for residualization

    return excess.reindex(aligned_to).dropna()


def _run_macro_pipeline(prices: pd.DataFrame, force_refresh: bool = False) -> dict:
    """Build returns, fetch macro factors, run all regressions, and export JSON.

    Returns a small summary dict for the terminal report.
    """
    # Imports here so the rest of the pipeline still works without statsmodels/fredapi.
    from macro_export import export_macro_to_webapp
    from macro_loader import (
        CURATED_FACTORS, MACRO_FACTORS, SCENARIO_SHOCKS,
        load_all_macro_factors, validate_macro_data,
    )
    from macro_regression import (
        build_control_factors, compare_residualization_approaches,
        compute_scenarios, per_stock_macro_betas, pivot_betas,
        residualized_macro_betas, rolling_macro_betas, run_macro_regression,
    )

    holdings = list(config.PORTFOLIO.keys())
    stock_rets = _build_returns(prices, holdings)
    if stock_rets.empty:
        raise RuntimeError("no portfolio price history; rerun with --refresh")

    port_ret = _portfolio_returns(stock_rets)
    logger.info("Portfolio returns: %d obs, mean=%.4f%%, vol=%.2f%%",
                len(port_ret), port_ret.mean() * 100, port_ret.std() * 100 * (252 ** 0.5))

    # Benchmark excess return for residualization
    try:
        bench_excess = _benchmark_excess_returns(
            config.BENCHMARK_TICKER, port_ret.index, force_refresh=force_refresh,
        )
        port_excess = port_ret.subtract(
            (bench_excess - bench_excess.mean()).reindex(port_ret.index).fillna(0.0) * 0,
            fill_value=0.0,
        )
        # We want portfolio excess returns too — same RF subtraction.
        # Easiest: subtract the implied RF (bench_ret - bench_excess) from port_ret.
        from feature_engine import fetch_prices as _fp  # noqa: F401
        # Recover RF as bench_ret - bench_excess:
        bench_prices = (
            prices[(config.BENCHMARK_TICKER, "Close")]
            if (config.BENCHMARK_TICKER, "Close") in prices.columns
            else None
        )
        if bench_prices is None:
            from feature_engine import fetch_prices
            bp = fetch_prices([config.BENCHMARK_TICKER],
                              lookback_days=config.PRICE_LOOKBACK_DAYS)
            bench_prices = bp[(config.BENCHMARK_TICKER, "Close")] if not bp.empty else None
        if bench_prices is not None:
            bench_ret = bench_prices.dropna().pct_change().dropna()
            rf_implied = bench_ret.reindex(bench_excess.index) - bench_excess
            port_excess = port_ret.subtract(
                rf_implied.reindex(port_ret.index).fillna(0.0), fill_value=0.0,
            )
        else:
            port_excess = port_ret
    except Exception as exc:  # noqa: BLE001
        logger.warning("benchmark fetch failed (%s) — residualization will be skipped", exc)
        bench_excess = None
        port_excess = port_ret

    macro = load_all_macro_factors(
        start=port_ret.index.min(),
        end=port_ret.index.max(),
        force_refresh=force_refresh,
    )
    validation = validate_macro_data(macro)
    logger.info("macro validation: %d factors, %d with warnings",
                len(validation), int((validation["warnings"] != "").sum()))

    # ---- Three-way regression: raw / v1 (market only) / v2 (market+vol+credit) ----
    raw_curated = None
    residualized_v1 = None
    residualized_v2 = None
    comparison = None

    if bench_excess is not None:
        try:
            controls = build_control_factors(bench_excess, macro)
            logger.info(
                "macro v2 controls: %s (n=%d)",
                ", ".join(controls.columns), len(controls),
            )
            comparison_df, raw_curated, residualized_v1, residualized_v2 = (
                compare_residualization_approaches(
                    portfolio_returns=port_excess,
                    factors=macro,
                    benchmark_excess_return=bench_excess,
                    controls=controls,
                    mode="curated",
                )
            )
            comparison = comparison_df
        except Exception as exc:  # noqa: BLE001
            logger.warning("v2 residualization failed (%s); falling back to v1", exc)

    if raw_curated is None:
        raw_curated = run_macro_regression(port_ret, macro, mode="curated")
    if residualized_v1 is None and bench_excess is not None:
        try:
            residualized_v1 = residualized_macro_betas(
                portfolio_returns=port_excess,
                factors=macro,
                control_factors=bench_excess,
                mode="curated",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("v1 fallback regression failed: %s", exc)

    # Default "curated" output: v2 → v1 → raw, preferring the most-controlled
    # estimate that is actually available.
    curated = residualized_v2 or residualized_v1 or raw_curated

    full = None
    try:
        full = run_macro_regression(port_ret, macro, mode="full")
    except Exception as exc:  # noqa: BLE001
        logger.warning("kitchen-sink regression failed: %s", exc)

    per_stock = per_stock_macro_betas(stock_rets, macro, mode="curated")
    rolling = rolling_macro_betas(port_ret, macro, mode="curated", window=60)
    scenarios = compute_scenarios(curated)

    # Per-timeframe regressions (YTD / 6M / 1Y / 2Y / MAX) — pre-computed
    # so the dashboard can swap between them with no client-side regression.
    timeframe_results = _compute_timeframe_results(
        port_ret=port_ret, port_excess=port_excess,
        bench_excess=bench_excess, macro=macro, stock_rets=stock_rets,
    )

    export_macro_to_webapp(
        portfolio_curated=curated,
        stock_betas=per_stock,
        rolling_betas=rolling,
        scenarios=scenarios,
        macro_factors=macro,
        portfolio_returns=port_ret,
        portfolio_full=full,
        portfolio_raw=raw_curated,
        portfolio_v1=residualized_v1,
        comparison=comparison,
        timeframe_results=timeframe_results,
    )

    # Top per-stock contributors per significant factor
    sig_factors = [f for f, e in curated.estimates.items() if e.significant_05]
    contributors: dict[str, list[dict]] = {}
    if not per_stock.empty:
        beta_wide = pivot_betas(per_stock, "beta")
        weights = pd.Series(config.PORTFOLIO).reindex(stock_rets.columns).dropna()
        weights = weights / weights.sum()
        for f in sig_factors:
            if f not in beta_wide.columns:
                continue
            wb = beta_wide[f].reindex(weights.index).fillna(0.0) * weights
            top = wb.reindex(wb.abs().sort_values(ascending=False).index).head(3)
            contributors[f] = [
                {"ticker": tk, "weighted_beta": float(v),
                 "share": float(v / curated.estimates[f].beta)
                 if abs(curated.estimates[f].beta) > 1e-9 else 0.0}
                for tk, v in top.items()
            ]

    return {
        "n_obs": curated.n_obs,
        "date_range": (port_ret.index.min(), port_ret.index.max()),
        "r_squared": curated.r_squared,
        "alpha_annualized": curated.alpha * 252,
        "alpha_p": curated.alpha_p,
        "methodology": curated.methodology,
        "market_beta": curated.market_beta,
        "comparison": comparison.to_dict(orient="records") if comparison is not None else None,
        "estimates": curated.estimates,
        "scenarios": scenarios,
        "max_vif": max(curated.vifs.values()) if curated.vifs else 0.0,
        "top_contributors": contributors,
    }


TIMEFRAMES: list[str] = ["ytd", "6m", "1y", "2y", "max"]


def _timeframe_window(end_date: pd.Timestamp, timeframe: str) -> tuple[pd.Timestamp, str]:
    """Return ``(start_date, label)`` for a named timeframe, where ``end_date``
    is the most recent observation. ``max`` returns ``Timestamp.min``."""
    end = pd.Timestamp(end_date)
    if timeframe == "ytd":
        return pd.Timestamp(year=end.year, month=1, day=1), "YTD"
    if timeframe == "6m":
        return end - pd.DateOffset(months=6), "6M"
    if timeframe == "1y":
        return end - pd.DateOffset(years=1), "1Y"
    if timeframe == "2y":
        return end - pd.DateOffset(years=2), "2Y"
    if timeframe == "max":
        return pd.Timestamp("1900-01-01"), "MAX"
    raise ValueError(f"unknown timeframe: {timeframe}")


def _compute_timeframe_results(
    port_ret: pd.Series,
    port_excess: pd.Series,
    bench_excess: pd.Series | None,
    macro: pd.DataFrame,
    stock_rets: pd.DataFrame,
) -> dict[str, dict]:
    """Run raw / v1 / v2 + per-stock + scenarios for each named timeframe.

    Returns a dict keyed on timeframe code (``"ytd"``..``"max"``). Each entry
    has the same shape as the main result block so the webapp can drop it in
    as the active set when the user toggles timeframe.
    """
    from macro_export import _result_to_dict, _stock_betas_payload
    from macro_regression import (
        build_control_factors, compare_residualization_approaches,
        compute_scenarios, per_stock_macro_betas,
    )

    if port_ret.empty or bench_excess is None:
        return {}

    out: dict[str, dict] = {}
    end = port_ret.index.max()

    for tf in TIMEFRAMES:
        start, label = _timeframe_window(end, tf)
        pr = port_ret.loc[port_ret.index >= start]
        pe = port_excess.loc[port_excess.index >= start]
        be = bench_excess.loc[bench_excess.index >= start]
        macro_tf = macro.loc[macro.index >= start]
        stock_tf = stock_rets.loc[stock_rets.index >= start]

        if len(pr) < 30 or be.empty:
            logger.info("timeframe %s: only %d obs, skipping", label, len(pr))
            continue

        try:
            controls_tf = build_control_factors(be, macro_tf)
            comp_df, raw, v1, v2 = compare_residualization_approaches(
                portfolio_returns=pe,
                factors=macro_tf,
                benchmark_excess_return=be,
                controls=controls_tf,
                mode="curated",
            )
            per_stock_tf = per_stock_macro_betas(stock_tf, macro_tf, mode="curated")
            scenarios_tf = compute_scenarios(v2)

            out[tf] = {
                "code": tf,
                "label": label,
                "n_obs": int(v2.n_obs),
                "date_range": [pr.index.min().date().isoformat(),
                               pr.index.max().date().isoformat()],
                "v2": _result_to_dict(v2, "Residualized v2"),
                "v1": _result_to_dict(v1, "Residualized v1"),
                "raw": _result_to_dict(raw, "Raw OLS"),
                "comparison": comp_df.replace({float("nan"): None})
                                     .to_dict(orient="records"),
                "scenarios": scenarios_tf,
                "stock_betas": _stock_betas_payload(per_stock_tf),
            }
            logger.info("timeframe %s: n=%d obs, R²=%.3f",
                        label, v2.n_obs, v2.r_squared)
        except Exception as exc:  # noqa: BLE001
            logger.warning("timeframe %s failed: %s", label, exc)

    return out


def _print_macro_report(summary: dict) -> None:
    print()
    print("=" * 92)
    print(" PORTFOLIO MACRO FACTOR EXPOSURES")
    print("=" * 92)
    d0, d1 = summary["date_range"]
    print(f" Analysis window     : {d0.date()} → {d1.date()} ({summary['n_obs']} obs)")
    print(f" Methodology         : {summary.get('methodology', 'raw_ols')}"
          + (f"  (market β={summary['market_beta']:.2f})"
             if summary.get('market_beta') is not None else ""))
    print(f" R² (curated)        : {summary['r_squared']:.3f}")
    alpha_str = f"{summary['alpha_annualized']*100:+.2f}%"
    print(f" Alpha (annualized)  : {alpha_str}  (p={summary['alpha_p']:.3f})")
    print(f" Max VIF             : {summary['max_vif']:.2f}  "
          f"{'⚠ multicollinearity' if summary['max_vif'] > 5 else ''}")
    if summary.get("comparison"):
        print("-" * 92)
        print(" Raw OLS → v1 (market only) → v2 (market + VIX + HY OAS)")
        if "v2_beta" in (summary["comparison"][0] if summary["comparison"] else {}):
            # Three-way table
            print(f"   {'Factor':<32} {'Raw β':>8} {'v1 β':>8} {'v2 β':>8}  Interpretation")
            for row in summary["comparison"]:
                rb = row.get("raw_beta")
                v1b = row.get("v1_beta")
                v2b = row.get("v2_beta")
                if rb is None or v2b is None:
                    continue
                v1_str = f"{v1b:>+8.3f}" if v1b is not None else "      —"
                print(f"   {row['factor']:<32}{rb:>+8.3f}{v1_str}{v2b:>+8.3f}  "
                      f"{row.get('interpretation', '')}")
        else:
            # Legacy two-way fallback
            print(f"   {'Factor':<32} {'Raw β':>9} {'Resid β':>9} {'Δ':>7}  Interpretation")
            for row in summary["comparison"]:
                rb = row.get("raw_beta")
                sb = row.get("residualized_beta")
                d = row.get("delta") or 0.0
                if rb is None or sb is None:
                    continue
                print(f"   {row['factor']:<32}{rb:>+9.3f}{sb:>+9.3f}{d:>+7.3f}  "
                      f"{row.get('interpretation', '')}")
    print("-" * 92)
    print(" Significant exposures (|beta| in raw factor units; * = p<0.10, ** = p<0.05, *** = p<0.01)")
    sig = sorted(summary["estimates"].values(),
                 key=lambda e: abs(e.beta), reverse=True)
    for est in sig:
        if est.p_value > 0.10:
            continue
        stars = "***" if est.p_value < 0.01 else ("**" if est.p_value < 0.05 else "*")
        print(f"   {est.factor:<32} β={est.beta:+.3f}  t={est.t_stat:+.2f}  {stars}")
    print("-" * 92)
    print(" Scenario impacts (beta × shock)")
    for sc in summary["scenarios"][:8]:
        marker = "★" if sc["significant"] else " "
        print(f"   {marker} {sc['factor']:<32} {sc['label']:<24}"
              f"  → portfolio {sc['impact']*100:+.2f}%")
    if summary["top_contributors"]:
        print("-" * 92)
        for f, rows in summary["top_contributors"].items():
            print(f" Top contributors to {f}:")
            for r in rows:
                pct = r["share"] * 100 if r["share"] is not None else 0.0
                print(f"   {r['ticker']:<6}  weighted β={r['weighted_beta']:+.3f}  "
                      f"({pct:+.0f}% of factor exposure)")
    print("=" * 92)


# =============================================================================
# Drift alerts
# =============================================================================
def _build_drift_alerts(pca_result, cluster_result, trajectory) -> pd.DataFrame:
    dists = nearest_cluster_distance(pca_result.scores, cluster_result)
    dists["assigned_tier"] = dists["assigned"].map(cluster_result.tier_labels)
    dists["nearest_other_tier"] = dists["nearest_other"].map(cluster_result.tier_labels)
    dists["is_borderline"] = dists["boundary_gap"] < config.CLUSTER_BOUNDARY_RADIUS

    alerts = dists.copy()
    if trajectory is not None:
        alerts["two_quarter_drift"] = trajectory.two_quarter_drift.reindex(alerts.index)
        alerts["crossed_cluster_last_q"] = trajectory.drift_flags.get(
            "crossed_cluster_last_q", pd.Series(False, index=alerts.index)
        ).reindex(alerts.index).fillna(False)
        alerts["large_2q_drift"] = trajectory.drift_flags.get(
            "large_2q_drift", pd.Series(False, index=alerts.index)
        ).reindex(alerts.index).fillna(False)
    else:
        alerts["two_quarter_drift"] = np.nan
        alerts["crossed_cluster_last_q"] = False
        alerts["large_2q_drift"] = False

    alerts["is_portfolio"] = alerts.index.isin(config.PORTFOLIO)

    # Flag if any condition is true
    alerts["alert"] = (
        alerts["is_borderline"]
        | alerts["crossed_cluster_last_q"]
        | alerts["large_2q_drift"]
    )
    return alerts[alerts["alert"]].sort_values(
        ["is_portfolio", "boundary_gap"], ascending=[False, True]
    )


# =============================================================================
# CSV exports
# =============================================================================
def _export_csvs(clean, pca_result, cluster_result, percentile_ranks,
                 port_report, opportunities, trajectory, drift_alerts) -> None:
    out = config.OUTPUT_DIR

    # Full universe
    full = clean.copy()
    for pc in pca_result.scores.columns:
        full[pc] = pca_result.scores[pc]
    full["cluster"] = cluster_result.assignments
    full["cluster_tier"] = full["cluster"].map(cluster_result.tier_labels)
    full["composite_score"] = percentile_ranks["composite_score"]
    full["risk_tier"] = percentile_ranks["risk_tier"]
    full.to_csv(out / "risk_scores_full.csv")

    port_report.to_csv(out / "portfolio_risk_report.csv", index=False)
    cluster_result.characterization.to_csv(out / "cluster_summary.csv")
    cluster_result.diagnostics.to_csv(out / "cluster_k_diagnostics.csv")
    pca_result.loadings.to_csv(out / "pca_loadings.csv")

    pca_summary = pd.DataFrame({
        "PC": list(pca_result.scores.columns),
        "VarExplained": pca_result.variance_explained,
        "CumVarExplained": pca_result.cumulative_variance,
        "Label": [pca_result.pc_labels[pc] for pc in pca_result.scores.columns],
    })
    pca_summary.to_csv(out / "pca_summary.csv", index=False)

    if not opportunities.empty:
        opportunities.to_csv(out / "opportunity_watchlist.csv")

    if trajectory is not None:
        rows = []
        for tk, path in trajectory.pc_paths.items():
            cp = trajectory.cluster_paths.get(tk, [])
            for (date, row), cid in zip(path.iterrows(), cp):
                out_row = {"Ticker": tk, "Date": date, "Cluster": cid}
                out_row.update(row.to_dict())
                rows.append(out_row)
        pd.DataFrame(rows).to_csv(out / "trajectory_data.csv", index=False)

    drift_alerts.to_csv(out / "drift_alerts.csv")
    logger.info("CSVs written to %s", out)


# =============================================================================
# Charts
# =============================================================================
def _make_charts(clean, pca_result, cluster_result, percentile_ranks,
                 port_report, trajectory) -> None:
    port_tickers = list(config.PORTFOLIO.keys())

    viz.plot_cluster_scatter(
        pca_result.scores, cluster_result, "PC1", "PC2",
        pca_result.pc_labels, port_tickers,
        "cluster_scatter_pc1_pc2.png",
    )
    if "PC3" in pca_result.scores.columns:
        viz.plot_cluster_scatter(
            pca_result.scores, cluster_result, "PC2", "PC3",
            pca_result.pc_labels, port_tickers,
            "cluster_scatter_pc2_pc3.png",
        )

    viz.plot_portfolio_dashboard(percentile_ranks, port_report)
    viz.plot_cluster_profiles(cluster_result.characterization, cluster_result.tier_labels)
    viz.plot_silhouette_analysis(cluster_result.diagnostics, cluster_result.k)
    viz.plot_pca_loadings(pca_result.loadings, pca_result.pc_labels)
    viz.plot_risk_score_distribution(percentile_ranks, port_tickers)
    viz.plot_sector_risk_comparison(clean, percentile_ranks, port_tickers)

    if trajectory is not None:
        viz.plot_trajectory_map(
            pca_result.scores, trajectory, cluster_result,
            pca_result.pc_labels, port_tickers,
        )

    # 3D scatter (always, trajectories only if available)
    viz.plot_cluster_scatter_3d(
        pca_result.scores, cluster_result, pca_result.pc_labels,
        port_tickers, trajectory=trajectory,
    )


# =============================================================================
# Terminal report
# =============================================================================
def _print_report(clean, pca_result, cluster_result, percentile_ranks,
                  port_report, opportunities, drift_alerts, trajectory) -> None:
    print()
    print("=" * 92)
    print(" IMA TORPEDO SCREENER — COMMITTEE REPORT")
    print("=" * 92)

    print(f" Universe scored          : {len(clean)} stocks with valid features")
    print(f" Benchmark                : {config.BENCHMARK_TICKER} (iShares S&P Small-Cap 600)")
    print(f" Features                 : {len(config.FEATURES)}")
    print(f" Financials flagged       : {int(clean['is_financial'].sum())} "
          f"(Altman Z imputed)")
    print("-" * 92)

    print(" PCA — variance explained")
    for i, pc in enumerate(pca_result.scores.columns):
        print(f"   {pc}  {pca_result.variance_explained[i]*100:5.2f}%  "
              f"(cum {pca_result.cumulative_variance[i]*100:5.2f}%)  "
              f"→ {pca_result.pc_labels[pc]}")
    print(f"   Top 3 loadings on PC1: "
          f"{_top_loadings_str(pca_result.loadings['PC1'])}")
    print("-" * 92)

    print(" Clustering")
    print(f"   Selected k = {cluster_result.k}  "
          f"(silhouette = {cluster_result.silhouette:.3f})")
    for cid in sorted(cluster_result.assignments.unique()):
        n = int((cluster_result.assignments == cid).sum())
        tier = cluster_result.tier_labels[int(cid)]
        print(f"   Cluster {cid} [{tier:<10}]  n={n:3d}")
    print("-" * 92)

    print(" IMA Portfolio — risk summary")
    header = (f"   {'Ticker':<7}{'Score':>6}{'Tier':>11}{'Cluster':>9}  "
              f"{'AltZ':>6}{'NDtoE':>7}{'Short%':>8}{'Mom90':>8}  Trajectory")
    print(header)
    for _, row in port_report.iterrows():
        if row.get("Cluster", -1) == -1:
            print(f"   {row['Ticker']:<7}  (not in universe)")
            continue
        print(
            f"   {row['Ticker']:<7}"
            f"{row['Composite_Score']:>6.1f}"
            f"{row['Risk_Tier']:>11}"
            f"{int(row['Cluster']):>9}  "
            f"{_fmt(row['Altman_Z']):>6}"
            f"{_fmt(row['Net_Debt_EBITDA']):>7}"
            f"{_fmt(row['Short_Pct_Float']):>8}"
            f"{_fmt_pct(row['Momentum_90d']):>8}  "
            f"{row.get('Trajectory', 'N/A')}"
        )

    # Top 5 risky holdings + primary driver
    valid = port_report[port_report["Cluster"] != -1].copy()
    top5 = valid.sort_values("Composite_Score", ascending=False).head(5)
    print("-" * 92)
    print(" Top 5 highest-risk IMA holdings")
    for _, row in top5.iterrows():
        print(f"   {row['Ticker']:<6}  score={row['Composite_Score']:.1f}  "
              f"tier={row['Risk_Tier']:<10}  drivers: {row['Top_Risk_Drivers']}")
    print("-" * 92)

    if not opportunities.empty:
        print(" Opportunity watchlist (fundamentals intact, sentiment bearish) — top 10")
        for tk, row in opportunities.head(10).iterrows():
            print(f"   {tk:<6}  Z={row['altman_z']:.2f}  OCF/NI={row['accruals_ratio']:.2f}  "
                  f"short={row['short_pct_float']:.1f}%  mom90={row['momentum_90d']*100:+.1f}%  "
                  f"{row['Sector']}")
        print("-" * 92)

    port_drift = drift_alerts[drift_alerts["is_portfolio"]]
    if not port_drift.empty:
        print(" Drift alerts — IMA holdings near a cluster boundary or recently transitioned")
        for tk, row in port_drift.iterrows():
            flag_bits = []
            if row["is_borderline"]:
                flag_bits.append(f"borderline(gap={row['boundary_gap']:.2f})")
            if row.get("crossed_cluster_last_q", False):
                flag_bits.append("CROSSED last Q")
            if row.get("large_2q_drift", False):
                flag_bits.append(f"2Q drift {row['two_quarter_drift']:.2f}")
            print(f"   {tk:<6} {row['assigned_tier']:<10} → "
                  f"{row['nearest_other_tier']:<10}  {' | '.join(flag_bits)}")
        print("-" * 92)

    print(f" Outputs: {config.OUTPUT_DIR}")
    print("=" * 92)
    print()


def _top_loadings_str(series: pd.Series, k: int = 3) -> str:
    top = series.abs().sort_values(ascending=False).head(k).index
    parts = [f"{name} ({series[name]:+.2f})" for name in top]
    return ", ".join(parts)


def _fmt(v, ndigits: int = 2) -> str:
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{float(v):.{ndigits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v) -> str:
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{float(v) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "—"


# =============================================================================
# CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IMA torpedo-risk screener")
    ap.add_argument("--refresh", action="store_true",
                    help="Force re-fetch of ALL cached data")
    ap.add_argument("--refresh-features", action="store_true",
                    help="Recompute features only (keep price cache)")
    ap.add_argument("--clusters", type=int, default=None,
                    help="Override number of k-means clusters")
    ap.add_argument("--no-trajectory", action="store_true",
                    help="Skip trajectory computation (much faster)")
    ap.add_argument("--portfolio-only", action="store_true",
                    help="Only score IMA holdings (skip universe clustering detail)")
    ap.add_argument("--no-webapp", action="store_true",
                    help="Skip JSON + asset export into webapp/public/")
    ap.add_argument("--no-macro", action="store_true",
                    help="Skip macro factor regression (faster; useful when FRED_API_KEY is unset)")
    ap.add_argument("--assess", type=str, default=None, metavar="TICKER",
                    help="Generate pitch assessment for one ticker, then exit "
                         "(prints + writes output/pitch_<TICKER>.json)")
    ap.add_argument("--assess-batch", type=str, default=None, metavar="FILE",
                    help="File path with one ticker per line; assess all and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    run_pipeline(args)


if __name__ == "__main__":
    main()
