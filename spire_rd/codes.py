"""Multi-spike code family and deterministic encoder (C1).

Constructs a parametric encoder family that maps source values to spike-train
codewords, with a tunable redundancy knob (lambda) that trades off distortion
for spike count.

The encoder uses a deterministic assignment:

    map_lambda(x) = argmin_w [ c(x, w) + lambda * spikes(w) ]

where ``c(x, w)`` is the task or MSE cost and ``spikes(w)`` is the number of
spikes in codeword *w*.  The family interpolates between single-spike latency
codes (lambda -> inf) and high-redundancy multi-spike codes (lambda -> 0).

Codeword geometries include TTFS (single spike), burst (spread spikes), and
time-repetition (coincident multi-spikes) -- the channel-matched geometry
discovered by SPIRE H2 v5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._util import lower_hull, mse_cost_matrix, physical_rate, task_cost_matrix
from .channel import SpikeChannel


def _physical_rate_rows(px: np.ndarray, rows: np.ndarray) -> float:
    """Compute I(X; x_hat) for a deterministic map whose induced channel rows
    are given (rows[k] = P(k | codeword chosen for source bin k))."""
    return physical_rate(px, rows)


@dataclass
class SpikeEncoder:
    """Parametric multi-spike code family encoder.

    Args:
        channel: The spike channel.
        K: Number of value bins.
        m_set: List of spike-count options per codeword (e.g. ``[1, 2, 4]``).
        h_half: Burst half-width in clock steps.
        nsamp: MC samples per codeword for induced channel.
        seed: Random seed.

    Example::

        ch = SpikeChannel(T=32, sigma=1.0)
        enc = SpikeEncoder(ch, K=64, m_set=[1, 2, 4], h_half=3)
        result = enc.compute(sigma=1.0, metric="mse")
        print(f"TTFS: D={result.ttfs_D:.4f}, R={result.ttfs_R:.3f}")
    """

    channel: SpikeChannel
    K: int = 64
    m_set: list[int] = field(default_factory=lambda: [1, 2, 4])
    h_half: int = 3
    nsamp: int = 4000
    seed: int = 0

    def build_codewords(self) -> tuple[list[np.ndarray], np.ndarray]:
        """Build the full codeword dictionary.

        Returns:
            Tuple of ``(patterns, lengths)`` where *patterns* is a list of
            integer arrays (one per codeword) and *lengths* is an array of
            spike counts.
        """
        T = self.channel.T
        pats: list[np.ndarray] = []
        lens: list[int] = []
        for tau in range(T):
            for m in self.m_set:
                offs = np.round(np.linspace(-self.h_half, self.h_half, m)).astype(int)
                pats.append(np.clip(tau + offs, 0, T - 1))
                lens.append(m)
        return pats, np.array(lens)

    def compute(
        self,
        sigma: float | None = None,
        metric: str = "mse",
        M: int = 2,
        lambdas: np.ndarray | None = None,
    ) -> EncoderResult:
        """Compute the encoder's operating curve over a lambda sweep.

        Args:
            sigma: Jitter sigma (defaults to channel sigma).
            metric: ``"mse"`` or ``"task"``.
            M: Number of task classes.
            lambdas: Redundancy penalty values to sweep.  Defaults to 200
                log-spaced points plus endpoints.

        Returns:
            An :class:`EncoderResult` with the Pareto curve and baselines.
        """
        s = sigma if sigma is not None else self.channel.sigma
        K = self.K
        T = self.channel.T

        pats_all, lens_all = self.build_codewords()
        P_all = self.channel.induced_channel(
            pats_all, K, n_per_pattern=self.nsamp, seed=self.seed
        )

        bin_centers = (np.arange(K) + 0.5) / K

        if metric == "mse":
            c = mse_cost_matrix(K, bin_centers, P_all)
        elif metric == "task":
            c = task_cost_matrix(K, M, P_all)
        else:
            raise ValueError(f"unknown metric '{metric}'")

        if lambdas is None:
            lambdas = np.logspace(-4, 1.0, 200)
            lambdas = np.concatenate([[0.0], lambdas, [10.0]])

        px = np.full(K, 1.0 / K)
        pts: list[tuple[float, float, float, float]] = []
        seen_maps: set[tuple[int, ...]] = set()

        for lam in lambdas:
            total_cost = c + lam * lens_all[None, :]
            w_map = np.argmin(total_cost, axis=1)
            map_tuple = tuple(w_map)
            if map_tuple not in seen_maps:
                seen_maps.add(map_tuple)
                D = float(c[np.arange(K), w_map].mean())
                R = _physical_rate_rows(px, P_all[w_map])
                E = float(lens_all[w_map].mean())
                pts.append((D, R, E, lam))

        nom_pos = np.round((1.0 - bin_centers) * (T - 1)).astype(int)
        min_m = int(lens_all.min())
        idx_smallest = [i for i, cnt in enumerate(lens_all) if cnt == min_m]
        smallest_positions = np.array([pats_all[i][0] for i in idx_smallest])
        ttfs_map = np.array(idx_smallest)[
            np.abs(smallest_positions[None, :] - nom_pos[:, None]).argmin(axis=1)
        ]
        D_ttfs = float(c[np.arange(K), ttfs_map].mean())
        R_ttfs = _physical_rate_rows(px, P_all[ttfs_map])

        c_smallest = c[:, idx_smallest]
        greedy_smallest_idx = np.argmin(c_smallest, axis=1)
        greedy_smallest_map = np.array(idx_smallest)[greedy_smallest_idx]
        D_greedy = float(c[np.arange(K), greedy_smallest_map].mean())
        R_greedy = _physical_rate_rows(px, P_all[greedy_smallest_map])

        hull = lower_hull([(d, r) for d, r, _e, _lam in pts])

        return EncoderResult(
            sigma=s,
            metric=metric,
            points=pts,
            hull=hull,
            ttfs_D=D_ttfs,
            ttfs_R=R_ttfs,
            greedy_D=D_greedy,
            greedy_R=R_greedy,
            K=K,
            T=T,
            m_set=list(self.m_set),
            h_half=self.h_half,
        )


@dataclass
class EncoderResult:
    """Result of an encoder operating-curve computation.

    Attributes:
        sigma: Jitter sigma used.
        metric: Distortion metric.
        points: List of ``(D, R, E, lambda)`` operating points.
        hull: Lower convex hull of ``(D, R)`` points.
        ttfs_D, ttfs_R: TTFS baseline distortion and rate.
        greedy_D, greedy_R: Greedy single-spike baseline distortion and rate.
        K, T: Source bins and clock steps.
        m_set: Spike-count options used.
        h_half: Burst half-width used.
    """

    sigma: float
    metric: str
    points: list[tuple[float, float, float, float]]
    hull: list[tuple[float, float]]
    ttfs_D: float
    ttfs_R: float
    greedy_D: float
    greedy_R: float
    K: int
    T: int
    m_set: list[int]
    h_half: int

    @property
    def num_distinct_maps(self) -> int:
        """Number of distinct encoding maps found across the lambda sweep."""
        return len(self.points)
