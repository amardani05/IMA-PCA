"""Project the IMA portfolio onto the universe PCs.

A stock's PC loading comes from the eigenvectors of the universe correlation
matrix. The portfolio's loading on PC_k is the weighted average of its
holdings' loadings on PC_k. We do NOT re-run PCA on the 20-name subset — doing
so would estimate a different basis and defeat the point of measuring our
position in universe-factor space.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pca_engine import PCAResult

logger = logging.getLogger(__name__)


def _align_portfolio_to_universe(
    portfolio: dict[str, float], universe_tickers: list[str]
) -> tuple[pd.Series, list[str]]:
    """Restrict portfolio to names present in the universe and renormalize."""
    held = {t: w for t, w in portfolio.items() if t in universe_tickers}
    missing = [t for t in portfolio if t not in universe_tickers]
    if missing:
        logger.warning(
            "Portfolio holdings not in PCA universe (dropped): %s", ", ".join(missing)
        )
    if not held:
        raise RuntimeError("No portfolio holdings found in PCA universe")

    w = pd.Series(held)
    w = w / w.sum()  # renormalize
    return w, missing


def portfolio_pc_loadings(
    portfolio: dict[str, float], pca: PCAResult
) -> pd.Series:
    """Weighted-sum portfolio loading on each retained PC."""
    w, _ = _align_portfolio_to_universe(portfolio, pca.tickers)
    L = pca.loadings.reindex(w.index)
    loadings = L.T @ w
    loadings.name = "PortfolioLoading"
    return loadings


def universe_benchmark_loadings(pca: PCAResult) -> pd.Series:
    """Equal-weighted average PC loading across the universe.

    Benchmarks the portfolio's PC exposure against a neutral "own the universe"
    stance. Mathematically this is tiny for every PC (eigenvectors are roughly
    mean-zero by construction), but tiny *active* differences are meaningful.
    """
    ew = pca.loadings.mean(axis=0)
    ew.name = "UniverseAverage"
    return ew


def active_pc_exposure(
    portfolio: dict[str, float], pca: PCAResult
) -> pd.DataFrame:
    """Active exposure = portfolio loading - universe average, with z-score.

    The z-score uses the cross-sectional stdev of individual stock loadings on
    each PC as the scale (i.e. "how many typical single-stock loadings does
    our active exposure move us?"). Above 1.0 sigma is flagged as a tilt.
    """
    port = portfolio_pc_loadings(portfolio, pca)
    uni = universe_benchmark_loadings(pca)
    sigma = pca.loadings.std(axis=0, ddof=1)
    active = port - uni
    z = active / sigma

    df = pd.DataFrame(
        {
            "PortfolioLoading": port,
            "UniverseAverage": uni,
            "ActiveExposure": active,
            "CrossSectionStd": sigma,
            "ActiveZ": z,
            "Tilt": z.abs() > 1.0,
        }
    )
    return df


def portfolio_variance_decomposition(
    portfolio: dict[str, float], pca: PCAResult
) -> pd.DataFrame:
    """Share of portfolio variance explained by each retained PC.

    var_contribution_k = loading_k^2 * eigenvalue_k / sum_j loading_j^2 * eigenvalue_j

    Note: this is the share within the *retained* PCs. Any remaining share
    of real variance lives in PCs we didn't keep (residual/idiosyncratic);
    we label it ``Residual`` in the output.
    """
    port_loadings = portfolio_pc_loadings(portfolio, pca)

    eigvals = pca.eigenvalues
    contribs = (port_loadings.values ** 2) * eigvals
    total_retained = contribs.sum()

    # Total portfolio variance (z-scored return space): weights' * Sigma * weights,
    # where Sigma = PCA correlation matrix. Using loadings basis this is
    # sum_k (L_k' w)^2 * eigval_k over ALL eigenvalues. We only kept K, so the
    # residual captures what the retained PCs don't.
    w, _ = _align_portfolio_to_universe(portfolio, pca.tickers)
    # Total portfolio variance in z-score space = w' Corr w, which also equals
    # sum over ALL k of (V_k' w)^2 * lambda_k. We compute it directly from the
    # z-scored returns to capture variance that lives in PCs we didn't retain.
    w_vec = w.reindex(pca.tickers).fillna(0.0).to_numpy()
    z = pca.returns.reindex(columns=pca.tickers).to_numpy()
    corr = np.corrcoef(z, rowvar=False)
    total_all = float(w_vec @ corr @ w_vec)
    residual = max(total_all - total_retained, 0.0)

    shares = contribs / total_all if total_all > 0 else contribs * 0
    residual_share = residual / total_all if total_all > 0 else 0.0

    df = pd.DataFrame(
        {
            "PC": list(port_loadings.index) + ["Residual"],
            "Loading": list(port_loadings.values) + [np.nan],
            "Eigenvalue": list(eigvals) + [np.nan],
            "VarShare": list(shares) + [residual_share],
        }
    )
    return df


def rolling_portfolio_pc_loadings(
    portfolio: dict[str, float],
    pca: PCAResult,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling portfolio PC score using fixed universe loadings.

    We hold the eigenvectors constant (estimated on the full window) and roll
    the weighted PC score to show how the portfolio's realized factor beta
    drifts through time. This is intentionally simpler than re-estimating the
    basis each day — it isolates weight drift from model drift.
    """
    w, _ = _align_portfolio_to_universe(portfolio, pca.tickers)
    # Per-date weighted portfolio return in z-score space
    returns_z = pca.returns.reindex(columns=w.index)
    port_series = returns_z @ w  # T-length series

    # Project each rolling window of portfolio returns on each PC series via
    # rolling correlation * stdev ratio == rolling regression beta.
    out = {}
    for pc in pca.scores.columns:
        pc_series = pca.scores[pc]
        aligned = pd.concat([port_series.rename("P"), pc_series.rename("F")], axis=1).dropna()
        cov = aligned["P"].rolling(window).cov(aligned["F"])
        var = aligned["F"].rolling(window).var()
        beta = cov / var
        out[pc] = beta
    df = pd.DataFrame(out)
    return df.dropna(how="all")
