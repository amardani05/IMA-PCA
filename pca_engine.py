"""Core PCA on the S&P 600 correlation matrix.

We deliberately use the correlation matrix rather than covariance so that
high-vol names don't dominate the leading components. Eigendecomposition is
done via ``numpy.linalg.eigh`` (exact for symmetric matrices, numerically
stable) so the committee can follow the math step-by-step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PCAResult:
    """Output bundle from ``run_pca``.

    Attributes:
        eigenvalues: (N,) array of eigenvalues, sorted descending.
        eigenvectors: (N, N) matrix of eigenvectors (columns), same ordering.
        loadings: (N, K) DataFrame indexed by ticker; each column a PC.
        scores: (T, K) DataFrame indexed by date; daily "returns" of each PC.
        variance_explained: (K,) array, fraction explained by each retained PC.
        cumulative_variance: (K,) array, cumulative fraction retained.
        all_eigenvalues: (N,) full eigenvalue spectrum (for scree plots).
        mp_threshold: Marchenko-Pastur upper bound for noise eigenvalues.
        n_components: K (number of retained components).
        tickers: list of tickers in loadings order.
        returns: the (T x N) returns matrix used to fit.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    loadings: pd.DataFrame
    scores: pd.DataFrame
    variance_explained: np.ndarray
    cumulative_variance: np.ndarray
    all_eigenvalues: np.ndarray
    mp_threshold: float
    n_components: int
    tickers: list[str]
    returns: pd.DataFrame


def marchenko_pastur_max(n_stocks: int, n_obs: int) -> float:
    """Largest eigenvalue expected under the Marchenko-Pastur null.

    For a correlation matrix of IID noise with aspect ratio q = N/T < 1,
    eigenvalues of the sample correlation matrix are bounded above by
    ``(1 + sqrt(q))^2``. Eigenvalues above this are evidence of structure.
    """
    q = n_stocks / n_obs
    return float((1.0 + np.sqrt(q)) ** 2)


def _fill_and_align(returns: pd.DataFrame) -> pd.DataFrame:
    """Drop all-NaN columns and fill small gaps with 0 for correlation estimation."""
    r = returns.dropna(axis=1, how="all").copy()
    # A small number of isolated NaNs (untraded days) should be zero-return, not NaN
    r = r.fillna(0.0)
    return r


def run_pca(returns: pd.DataFrame, n_components: int) -> PCAResult:
    """Run correlation-matrix PCA on a T x N returns frame."""
    r = _fill_and_align(returns)
    tickers = list(r.columns)
    T, N = r.shape
    if n_components > N:
        raise ValueError(f"n_components ({n_components}) exceeds N ({N})")

    logger.info("Running PCA on %d obs x %d tickers", T, N)

    # Standardize each column (correlation-matrix PCA is PCA on z-scored returns)
    mu = r.mean(axis=0)
    sigma = r.std(axis=0, ddof=1).replace(0.0, np.nan)
    z = (r - mu) / sigma
    z = z.dropna(axis=1, how="any")  # drop any zero-variance stocks
    tickers = list(z.columns)
    N = z.shape[1]

    corr = np.corrcoef(z.values, rowvar=False)
    # eigh returns eigenvalues in ascending order
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Sign convention: make the largest-absolute-loading positive for each PC
    for k in range(eigvecs.shape[1]):
        j = int(np.argmax(np.abs(eigvecs[:, k])))
        if eigvecs[j, k] < 0:
            eigvecs[:, k] *= -1

    loadings = pd.DataFrame(
        eigvecs[:, :n_components],
        index=tickers,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )

    # PC scores = z-scored returns projected onto eigenvectors
    scores_vals = z.values @ eigvecs[:, :n_components]
    scores = pd.DataFrame(
        scores_vals,
        index=z.index,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )

    total_var = eigvals.sum()
    variance_explained = eigvals[:n_components] / total_var
    cumulative_variance = np.cumsum(variance_explained)

    mp_max = marchenko_pastur_max(N, T)

    return PCAResult(
        eigenvalues=eigvals[:n_components],
        eigenvectors=eigvecs[:, :n_components],
        loadings=loadings,
        scores=scores,
        variance_explained=variance_explained,
        cumulative_variance=cumulative_variance,
        all_eigenvalues=eigvals,
        mp_threshold=mp_max,
        n_components=n_components,
        tickers=tickers,
        returns=z,
    )


def stability_check(
    returns: pd.DataFrame, n_components: int
) -> pd.DataFrame:
    """Run PCA on two non-overlapping halves and compare eigenvectors.

    Stability for PC_i is ``|v1_i . v2_i|``. Values >0.7 are robust; <0.5 noise.
    Only tickers with data in *both* halves are used.
    """
    r = _fill_and_align(returns)
    mid = len(r) // 2
    r1 = r.iloc[:mid]
    r2 = r.iloc[mid:]

    # Restrict to tickers with nonzero variance in both halves
    common = r1.columns.intersection(r2.columns)
    r1 = r1[common]
    r2 = r2[common]
    common = [
        c
        for c in common
        if r1[c].std(ddof=1) > 0 and r2[c].std(ddof=1) > 0
    ]
    r1 = r1[common]
    r2 = r2[common]

    res1 = run_pca(r1, n_components)
    res2 = run_pca(r2, n_components)

    # Align by ticker
    L1 = res1.loadings.reindex(common).values
    L2 = res2.loadings.reindex(common).values

    stabilities = []
    for k in range(n_components):
        v1 = L1[:, k]
        v2 = L2[:, k]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        sim = abs(float(v1 @ v2) / denom) if denom > 0 else float("nan")
        stabilities.append(sim)

    df = pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(n_components)],
            "Stability": stabilities,
            "Robust": [s > 0.7 for s in stabilities],
            "NoiseLike": [s < 0.5 for s in stabilities],
        }
    )
    return df


def top_bottom_loadings(
    loadings: pd.DataFrame, k: int = 10
) -> dict[str, dict[str, pd.Series]]:
    """For each PC, return the top-k and bottom-k stocks by loading."""
    out: dict[str, dict[str, pd.Series]] = {}
    for pc in loadings.columns:
        s = loadings[pc].sort_values(ascending=False)
        out[pc] = {"top": s.head(k), "bottom": s.tail(k)}
    return out
