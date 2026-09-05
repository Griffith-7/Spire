"""Soft decoder for spike trains (C2).

Provides channel-aware probabilistic decoding via intensity pseudo-likelihood
(PL) and, for the special case of m=2 spikes, a closed-form exact likelihood.

The PL decoder evaluates ``log lambda_x(t)`` for each received spike and
accumulates evidence across the full spike train.  It is O(|S| x |W|) and
event-driven, making it suitable for neuromorphic hardware mapping.

The exact m=2 likelihood uses the full T1 generative model (survivors,
jittered + clipped, deletions, uniform insertions) and is Bayes-optimal for
that observable.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.special import softmax as _softmax

from .channel import SpikeChannel


class SoftDecoder:
    """Channel-aware soft decoder for spike trains.

    Args:
        channel: A :class:`~spire_rd.channel.SpikeChannel` instance.
        K: Number of value bins for the source alphabet.

    Example::

        ch = SpikeChannel(T=32, sigma=1.0)
        dec = SoftDecoder(ch, K=64)
        log_L = dec.intensity_matrix(w_map)
        values = dec.decode(bufs, log_L)
    """

    def __init__(self, channel: SpikeChannel, K: int = 64) -> None:
        self.channel = channel
        self.K = int(K)
        self.T = channel.T
        self.p_d = channel.p_d
        self.p_i = channel.p_i

    def source_marginals(self, pattern: np.ndarray, sigma: float | None = None) -> np.ndarray:
        """Per-source landing distribution q_i(t) with clip-boundary folding.

        Args:
            pattern: Spike positions for one codeword.
            sigma: Jitter sigma (defaults to channel sigma).

        Returns:
            Array of shape ``(m, T)`` where ``m = len(pattern)``.
        """
        s = sigma if sigma is not None else self.channel.sigma
        offs, kern = self.channel.jitter_kernel(s)
        qs = []
        for u in pattern:
            q = np.zeros(self.T)
            for o, pr in zip(offs, kern, strict=True):
                t = int(np.clip(u + o, 0, self.T - 1))
                q[t] += pr
            qs.append(q)
        return np.array(qs)

    def intensity_matrix(self, w_map: list[np.ndarray], sigma: float | None = None) -> np.ndarray:
        """Compute log-intensity matrix log lambda_x(t) for every codeword.

        ``lambda_x(t) = sum_i (1 - p_d) * q_i(t) + p_i`` where the sum is
        over surviving spikes of codeword *x*.

        Args:
            w_map: List of spike-position arrays, one per codeword.
            sigma: Jitter sigma (defaults to channel sigma).

        Returns:
            Log-intensity matrix of shape ``(K, T)``.
        """
        L = np.zeros((len(w_map), self.T))
        for x, pat in enumerate(w_map):
            for q in self.source_marginals(pat, sigma):
                L[x] += (1.0 - self.p_d) * q
        L += self.p_i
        return np.log(np.maximum(L, 1e-300))  # type: ignore[no-any-return]

    def pl_scores(self, bufs: np.ndarray, log_L: np.ndarray) -> np.ndarray:
        """Pseudo-likelihood scores: sum over received spikes of log lambda_x(t).

        Args:
            bufs: Position buffer ``(n, L)`` from channel simulation.
            log_L: Log-intensity matrix ``(K, T)`` from :meth:`intensity_matrix`.

        Returns:
            Score matrix of shape ``(n, K)``.
        """
        pl = np.zeros((bufs.shape[0], log_L.shape[0]))
        for j in range(bufs.shape[1]):
            col = bufs[:, j]
            ok = ~np.isnan(col)
            idx = np.where(ok)[0]
            if len(idx):
                pl[idx] += log_L[:, col[idx].astype(int)].T
        return pl

    def decode(
        self, bufs: np.ndarray, log_L: np.ndarray, method: str = "pl"
    ) -> np.ndarray:
        """Decode received buffers to estimated values.

        Args:
            bufs: Position buffer ``(n, L)`` from channel simulation.
            log_L: Log-intensity matrix ``(K, T)`` from :meth:`intensity_matrix`.
            method: ``"pl"`` for pseudo-likelihood (default), ``"median"`` for
                median decoding.

        Returns:
            Estimated values in [0, 1], shape ``(n,)``.
        """
        bin_centers = (np.arange(self.K) + 0.5) / self.K
        if method == "pl":
            scores = self.pl_scores(bufs, log_L)
            values = _softmax(scores, axis=1) @ bin_centers
            silent = np.all(np.isnan(bufs), axis=1)
            values[silent] = 0.5
            return values  # type: ignore[no-any-return]
        elif method == "median":
            est = self.channel.decode(bufs, decoder="median")
            return self.channel.pos_to_value(est)
        else:
            raise ValueError(f"unknown decode method '{method}'")

    def counts_from_buffer(self, bufs: np.ndarray) -> np.ndarray:
        """Convert NaN-padded position buffer to count vectors.

        Args:
            bufs: Position buffer ``(n, L)``.

        Returns:
            Count matrix ``(n, T)`` of integer spike counts per timestep.
        """
        C = np.zeros((bufs.shape[0], self.T), dtype=np.int64)
        for i in range(bufs.shape[0]):
            v = bufs[i][~np.isnan(bufs[i])].astype(int)
            if len(v):
                C[i] = np.bincount(v, minlength=self.T)
        return C

    def exact_logG(
        self,
        Cmat: np.ndarray,
        Qs: list[np.ndarray],
        surv_pmf: list[float],
    ) -> np.ndarray:
        """Exact marginal log-likelihood for m=2 spikes.

        Computes ``log G(x, c)`` for every candidate codeword and every
        sample, where ``G`` is the marginal likelihood of the observed count
        vector under the full T1 generative model (jittered+clipped survivors
        with position marginals ``Q``, deletions, and uniform insertions).

        Only valid for ``m=2`` codewords.  The likelihood marginalizes over
        the number of surviving spikes ``k = 0, 1, 2``::

            G(x, c) = sum_k surv_pmf[k] * P(B = n - k) * (1/T)^(n-k)
                      * (n - k)! / prod_t c_t! * W_k(x, c)

        where ``B ~ Binom(T, p_i)`` is the insertion count, ``n = sum_t c_t``,
        and ``W_k`` accounts for the survivor position marginals:

        - ``W_0 = 1``,
        - ``W_1 = (q1 + q2) . c / C(2, 1)``,
        - ``W_2 = (q1 . c)(q2 . c) - q1 q2 . c``.

        This is a correctly normalized likelihood (it sums to 1 over all
        observable count vectors), unlike a raw unordered count of survivor
        assignments.

        Args:
            Cmat: Count vectors ``(N, T)``.
            Qs: List of ``(2, T)`` source marginal arrays, one per codeword.
            surv_pmf: Survival probability weights ``[P(k=0), P(k=1), P(k=2)]``
                for exactly ``k`` of the two spikes surviving.

        Returns:
            Log-likelihood matrix ``(N, n_candidates)``.
        """
        N = Cmat.shape[0]
        n = Cmat.sum(axis=1)
        c = Cmat.astype(np.float64)
        T = self.T
        log_invT = np.full(N, np.log(1.0 / T))
        log_fact_c = gammaln(c + 1).sum(axis=1)
        bs = [float(s) for s in surv_pmf]

        logG = np.full((N, len(Qs)), -np.inf)
        for xi, Q in enumerate(Qs):
            q1, q2 = Q[0], Q[1]
            G = np.zeros(N)
            # k = 0 survivors: insertions only
            b0 = n
            m0 = (b0 >= 0) & (b0 <= T)
            if m0.any():
                bi = b0[m0]
                lb0 = self._log_bp(bi)
                L0 = lb0 + bi * log_invT[m0] + gammaln(bi + 1) - log_fact_c[m0]
                G[m0] += bs[0] * np.exp(L0)
            # k = 1 survivor: (q1 + q2) . c , halved for the two source patterns
            b1 = n - 1
            m1 = (b1 >= 0) & (b1 <= T)
            if m1.any():
                bi = b1[m1]
                L1 = (
                    self._log_bp(bi)
                    + bi * log_invT[m1]
                    + gammaln(bi + 1)
                    - log_fact_c[m1]
                )
                w1 = (q1 + q2) @ c.T
                G[m1] += bs[1] * 0.5 * np.exp(L1) * w1[m1]
            # k = 2 survivors: (q1.c)(q2.c) - q1 q2 . c
            b2 = n - 2
            m2 = (b2 >= 0) & (b2 <= T)
            if m2.any():
                bi = b2[m2]
                L2 = (
                    self._log_bp(bi)
                    + bi * log_invT[m2]
                    + gammaln(bi + 1)
                    - log_fact_c[m2]
                )
                w2 = (q1 @ c.T) * (q2 @ c.T) - ((q1 * q2) @ c.T)
                G[m2] += bs[2] * np.exp(L2) * w2[m2]
            with np.errstate(divide="ignore"):
                logG[:, xi] = np.log(np.maximum(G, 1e-300))
        return logG

    def _log_bp(self, b: np.ndarray) -> np.ndarray:
        """Log P(B = b) for B ~ Binom(T, p_i); -inf outside 0..T."""
        b = np.asarray(b, dtype=float)
        out = np.full(b.shape, -np.inf)
        m_ok = (b >= 0) & (b <= self.T)
        if m_ok.any():
            bb = b[m_ok]
            out[m_ok] = (
                bb * np.log(max(self.p_i, 1e-300))
                + (self.T - bb) * np.log(max(1 - self.p_i, 1e-300))
                + self._logcomb_vec(self.T, bb)
            )
        return out  # type: ignore[no-any-return]

    @staticmethod
    def _logcomb_vec(n_: float | np.ndarray, k_: float | np.ndarray) -> np.ndarray:
        """Log binomial coefficient via log-gamma."""
        n_ = np.asarray(n_, dtype=float)
        k_ = np.asarray(k_, dtype=float)
        return gammaln(n_ + 1) - gammaln(k_ + 1) - gammaln(n_ - k_ + 1)  # type: ignore[no-any-return]
