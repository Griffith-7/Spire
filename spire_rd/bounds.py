"""Rate-distortion bounds computation (T2).

Computes the information-theoretic R(D) ceiling for the spike channel using
two methods:

1. **Blahut-Arimoto capacity** — standard capacity iteration on the induced
   discrete channel.
2. **Direct I(X; x_hat) minimization** — exponentiated-gradient mirror descent
   that minimizes the true end-to-end rate (no surrogate), producing the
   optimal R(D) frontier.

The direct solver is the canonical method from SPIRE T2 v1.2.1, fixing the
surrogate-rate flaw of earlier BA-on-cost approaches (MEMORY I13).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._util import hull_at, lower_hull, mse_cost_matrix, physical_rate_from_q, task_cost_matrix
from .channel import SpikeChannel


@dataclass
class RDBound:
    """R(D) bound computation for a spike channel.

    Args:
        channel: The spike channel to compute bounds for.
        K: Number of source bins.
        nsamp: Monte-Carlo samples per codeword for induced channel estimation.
        seed: Random seed for induced channel computation.

    Example::

        ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        bound = RDBound(ch, K=64, nsamp=4000)
        result = bound.compute(sigma=1.0, metric="mse")
        print(f"capacity = {result.capacity:.3f} bits/spike")
    """

    channel: SpikeChannel
    K: int = 64
    nsamp: int = 4000
    seed: int = 0

    def capacity(
        self,
        sigma: float | None = None,
        n_per_pattern: int = 2500,
        iters: int = 500,
        tol: float = 1e-8,
    ) -> float:
        """Compute channel capacity via Blahut-Arimoto iteration.

        Args:
            sigma: Jitter sigma (defaults to channel sigma).
            n_per_pattern: MC samples per codeword for induced channel.
            iters: Max BA iterations.
            tol: Convergence tolerance.

        Returns:
            Capacity in bits per spike.
        """
        s = sigma if sigma is not None else self.channel.sigma
        pats = [np.array([t]) for t in range(self.channel.T)]
        P = self.channel.induced_channel(
            pats, self.K, n_per_pattern=n_per_pattern, seed=self.seed, sigma=s
        )
        return float(_ba_capacity(P, iters=iters, tol=tol))

    def compute(
        self,
        sigma: float | None = None,
        metric: str = "mse",
        M: int = 2,
        betas: np.ndarray | None = None,
        eta: float = 0.2,
        iters: int = 25000,
        tol: float = 1e-12,
    ) -> RDResult:
        """Compute the R(D) frontier via direct I(X; x_hat) minimization.

        Args:
            sigma: Jitter sigma (defaults to channel sigma).
            metric: ``"mse"`` for mean-squared error, or ``"task"`` for 0-1
                classification loss.
            M: Number of task classes (only used when ``metric="task"``).
            betas: Lagrange multiplier sweep.  Defaults to 60 points in
                ``[1e-2, 1e4]`` on log-scale.
            eta: Mirror-descent step size.
            iters: Max iterations per beta.
            tol: Convergence tolerance.

        Returns:
            An :class:`RDResult` with the computed frontier and metadata.
        """
        s = sigma if sigma is not None else self.channel.sigma
        K = self.K
        pats = [np.array([t]) for t in range(self.channel.T)]
        P = self.channel.induced_channel(
            pats, K, n_per_pattern=self.nsamp, seed=self.seed, sigma=s
        )
        bin_centers = (np.arange(K) + 0.5) / K

        if metric == "mse":
            c = mse_cost_matrix(K, bin_centers, P)
        elif metric == "task":
            c = task_cost_matrix(K, M, P)
        else:
            raise ValueError(f"unknown metric '{metric}'")

        if betas is None:
            betas = np.logspace(-2, 4.0, 60)

        px = np.full(K, 1.0 / K)
        q = None
        pts: list[tuple[float, float, float]] = []

        for b in betas:
            D, R, q, _used = _solve_direct(c, P, px, b, q0=q, eta=eta, iters=iters, tol=tol)
            pts.append((D, R, b))

        hull = lower_hull([(d, r) for d, r, _b in pts])

        cap = _ba_capacity(P)

        return RDResult(
            sigma=s,
            metric=metric,
            points=[(d, r) for d, r, _b in pts],
            hull=hull,
            capacity=float(cap),
            K=K,
            T=self.channel.T,
            p_d=self.channel.p_d,
            p_i=self.channel.p_i,
        )

    def feasible_points(
        self,
        sigma: float | None = None,
        metric: str = "mse",
        M: int = 2,
    ) -> list[tuple[str, float, float]]:
        """Compute deterministic reference operating points (TTFS, greedy).

        Args:
            sigma: Jitter sigma.
            metric: ``"mse"`` or ``"task"``.
            M: Number of task classes.

        Returns:
            List of ``(label, distortion, rate)`` tuples.
        """
        s = sigma if sigma is not None else self.channel.sigma
        K = self.K
        T = self.channel.T
        pats = [np.array([t]) for t in range(T)]
        P = self.channel.induced_channel(
            pats, K, n_per_pattern=self.nsamp, seed=self.seed, sigma=s
        )
        bin_centers = (np.arange(K) + 0.5) / K

        if metric == "mse":
            c = mse_cost_matrix(K, bin_centers, P)
        elif metric == "task":
            c = task_cost_matrix(K, M, P)
        else:
            raise ValueError(f"unknown metric '{metric}'")

        px = np.full(K, 1.0 / K)
        nom_pos = np.round((1.0 - bin_centers) * (T - 1)).astype(int)
        greedy = np.argmin(c, axis=1)

        out = []
        for label, wmap in [("ttfs", nom_pos), ("greedy", greedy)]:
            R = physical_rate_from_q(px, np.eye(T)[wmap], P)
            D = float((px * c[np.arange(K), wmap]).sum())
            out.append((label, D, R))
        return out


@dataclass
class RDResult:
    """Result of an R(D) frontier computation.

    Attributes:
        sigma: Jitter sigma used.
        metric: Distortion metric (``"mse"`` or ``"task"``).
        points: Raw ``(D, R)`` operating points from the beta sweep.
        hull: Lower convex hull of the points.
        capacity: Channel capacity in bits per spike.
        K: Number of source bins.
        T: Number of clock steps.
        p_d: Deletion probability.
        p_i: Insertion probability.
    """

    sigma: float
    metric: str
    points: list[tuple[float, float]]
    hull: list[tuple[float, float]]
    capacity: float
    K: int
    T: int
    p_d: float
    p_i: float

    def rate_at(self, D: float) -> float | None:
        """Interpolate the frontier rate at a given distortion.

        Args:
            D: Target distortion.

        Returns:
            Minimum rate at distortion D, or None if outside hull support.
        """
        return hull_at(self.hull, D)

    def gain_vs_ttfs(self, ttfs_D: float, ttfs_R: float) -> float | None:
        """Matched-D rate saving vs a TTFS reference point.

        Args:
            ttfs_D: TTFS distortion.
            ttfs_R: TTFS rate.

        Returns:
            Percentage rate saving ``(ttfs_R - R_frontier) / ttfs_R * 100``,
            or None if TTFS point is outside hull support.
        """
        R_star = self.rate_at(ttfs_D)
        if R_star is None:
            return None
        return 100.0 * (ttfs_R - R_star) / ttfs_R


# ---------------------------------------------------------------------------
# Internal solvers
# ---------------------------------------------------------------------------

def _solve_direct(
    c: np.ndarray,
    P: np.ndarray,
    px: np.ndarray,
    beta: float,
    q0: np.ndarray | None = None,
    eta: float = 0.2,
    iters: int = 25000,
    tol: float = 1e-12,
) -> tuple[float, float, np.ndarray, int]:
    """Mirror descent on L = I(X;x_hat) + beta * <c, q>.

    Returns ``(D, R, q, iterations_used)``.
    """
    K = c.shape[0]
    q = q0.copy() if q0 is not None else np.full((K, P.shape[0]), 1.0 / P.shape[0])
    used = iters
    for it in range(iters):
        Q = q @ P
        Qm = px @ Q
        with np.errstate(divide="ignore", invalid="ignore"):
            LOGR = np.where(
                Q > 0,
                np.log(np.maximum(Q, 1e-300))
                - np.log(np.maximum(Qm[None, :], 1e-300)),
                0.0,
            )
        g = LOGR @ P.T + beta * c
        g -= g.min(axis=1, keepdims=True)
        qn = q * np.exp(-eta * g)
        qn /= qn.sum(axis=1, keepdims=True)
        if np.abs(qn - q).max() < tol:
            q = qn
            used = it + 1
            break
        q = qn
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
    R = float(px @ t.sum(axis=1) / np.log(2))
    D = float((q * c).sum() / K)
    return D, R, q, used


def _ba_capacity(
    P: np.ndarray, iters: int = 500, tol: float = 1e-8
) -> float:
    """Blahut-Arimoto capacity of a discrete channel P(k|w).

    Args:
        P: Channel transition matrix ``(M, K_out)``, rows sum to 1.
        iters: Max iterations.
        tol: Convergence tolerance.

    Returns:
        Capacity in bits per channel use.
    """
    M, K_out = P.shape
    pw = np.ones(M) / M
    for _ in range(iters):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(np.maximum(P, 1e-300)) - np.log(
                np.maximum(pw @ P, 1e-300)
            )
        num = pw * np.exp(log_ratio.max(axis=1) + np.log(
            np.exp(log_ratio - log_ratio.max(axis=1, keepdims=True)).sum(axis=1)
        ))
        pw_new = num / num.sum()
        if np.abs(pw_new - pw).max() < tol:
            pw = pw_new
            break
        pw = pw_new
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(np.maximum(P, 1e-300)) - np.log(
            np.maximum(pw @ P, 1e-300)
        )
    C = float(pw @ (P * log_ratio).sum(axis=1) / np.log(2))
    return C
