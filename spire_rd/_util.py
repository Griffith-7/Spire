"""Internal shared utilities for spire_rd.

Convex hull helpers, mutual information computation, and cost matrix builders.
All functions are pure numpy; no external dependencies beyond numpy.
"""
from __future__ import annotations

import numpy as np


def lower_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower convex hull of (D, R) points, sorted by D.

    Given a set of (distortion, rate) operating points, computes the lower
    convex hull — the Pareto-optimal frontier where no point simultaneously
    has both lower distortion and lower rate than a convex combination of
    its neighbors.

    Args:
        points: List of (distortion, rate) tuples.

    Returns:
        Sorted list of (D, R) tuples on the lower convex hull.
    """
    pts = sorted(points)
    hull: list[tuple[float, float]] = []
    for d, r in pts:
        while len(hull) >= 2:
            d0, r0 = hull[-2]
            d1, r1 = hull[-1]
            if (r1 - r0) * (d - d1) >= (r - r1) * (d1 - d0):
                hull.pop()
            else:
                break
        hull.append((d, r))
    return hull


def hull_at(
    hull: list[tuple[float, float]], d: float, tol: float = 5e-6
) -> float | None:
    """Linear interpolation of lower hull at abscissa *d*.

    Args:
        hull: Lower convex hull as (D, R) list from :func:`lower_hull`.
        d: Distortion value at which to interpolate.
        tol: Tolerance for matching hull endpoints (matches 6-dp CSV storage).

    Returns:
        Interpolated rate at distortion *d*, or ``None`` if *d* is outside
        the hull's support.
    """
    if d < hull[0][0] - tol or d > hull[-1][0] + tol:
        return None
    for i in range(len(hull) - 1):
        d0, r0 = hull[i]
        d1, r1 = hull[i + 1]
        if d0 - tol <= d <= d1 + tol:
            if d1 - d0 < 1e-15:
                return min(r0, r1)
            return r0 + (r1 - r0) * (d - d0) / (d1 - d0)
    return hull[-1][1]


def physical_rate(
    px: np.ndarray, P_rows: np.ndarray
) -> float:
    """Compute end-to-end mutual information I(X; x_hat) in bits.

    Given source priors *px* and the induced channel rows P[k|w] (one row
    per codeword, selected by a deterministic map), computes the physical
    rate in bits per source symbol.

    Args:
        px: Source prior distribution, shape ``(K,)``, sums to 1.
        P_rows: Induced channel rows for the chosen codewords, shape ``(K, K_out)``.

    Returns:
        Rate in bits per source symbol.
    """
    pk = px @ P_rows
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(
            P_rows > 0,
            np.log(np.maximum(P_rows, 1e-300))
            - np.log(np.maximum(pk[None, :], 1e-300)),
            0.0,
        )
    return float(px @ (P_rows * log_ratio).sum(axis=1) / np.log(2))


def physical_rate_from_q(
    px: np.ndarray, q: np.ndarray, P: np.ndarray
) -> float:
    """Compute I(X; x_hat) from an encoder distribution q(w|x) and channel P(k|w).

    Args:
        px: Source prior, shape ``(K,)``.
        q: Encoder distribution q(w|x), shape ``(K, T)`` rows sum to 1.
        P: Induced channel P(k|w), shape ``(T, K_out)``.

    Returns:
        Rate in bits per source symbol.
    """
    Q = q @ P
    Qm = px @ Q
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(
            Q > 0,
            Q * (
                np.log(np.maximum(Q, 1e-300))
                - np.log(np.maximum(Qm[None, :], 1e-300))
            ),
            0.0,
        )
    return float(px @ t.sum(axis=1) / np.log(2))


def mse_cost_matrix(K: int, bin_centers: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Build MSE distortion cost matrix c[x, w] = E[(x_hat - x)^2 | w].

    Args:
        K: Number of source bins.
        bin_centers: Center of each bin, shape ``(K,)``.
        P: Induced channel P(k|w), shape ``(K, K_out)``.

    Returns:
        Cost matrix of shape ``(K, K_out)``.
    """
    c = np.zeros((K, P.shape[0]))
    for x in range(K):
        c[x] = ((bin_centers[x] - bin_centers) ** 2) @ P.T
    return c


def task_cost_matrix(
    K: int, M: int, P: np.ndarray
) -> np.ndarray:
    """Build 0-1 task loss cost matrix for M-ary classification.

    Args:
        K: Number of source bins.
        M: Number of task classes.
        P: Induced channel P(k|w), shape ``(K, K_out)``.

    Returns:
        Cost matrix of shape ``(K, K_out)`` where ``c[x, w]`` = P(decoded
        outside x's class | codeword w).
    """
    oc = np.minimum((np.arange(K) * M) // K, M - 1)
    mism = (oc[:, None] != oc[None, :]).astype(float)
    return mism @ P.T  # type: ignore[no-any-return]
