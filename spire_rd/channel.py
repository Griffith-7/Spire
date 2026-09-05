"""Canonical spike channel model (T1 spec v1.1).

Models a discrete-time LIF-compatible spike channel with timing jitter,
spike deletion, and spike insertion over a finite clock window.

Model (fixed composition order):
    y = Insert(p_i) o Delete(p_d) o Jitter(sigma)

applied to spike positions.  Deletion commutes with per-spike i.i.d. jitter
(Proposition 1 in the SPIRE channel spec); insertion is additive and
independent.

Default parameters match the SPIRE benchmark configuration:
    T=32 clock steps, sigma=1.0, p_d=0.05, p_i=0.01.
"""
from __future__ import annotations

from typing import Literal

import numpy as np


class SpikeChannel:
    """Canonical noisy spike channel.

    Simulates the effect of timing jitter, deletion, and insertion on
    discrete-time spike trains, and provides utilities for decoding and
    computing the induced channel.

    Args:
        T: Number of clock steps in the simulation window.
        sigma: Jitter standard deviation (in clock steps).
        p_d: Per-spike deletion probability.
        p_i: Per-step insertion probability.
        boundary: How to handle spikes that land outside [0, T-1] after
            jitter: ``"clip"`` (saturate at edge), ``"drop"`` (absorb),
            or ``"wrap"`` (modular).

    Example::

        ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        buf = ch.simulate(np.array([10, 20]), n=1000, seed=0)
        est = ch.decode(buf)
    """

    def __init__(
        self,
        T: int = 32,
        sigma: float = 1.0,
        p_d: float = 0.05,
        p_i: float = 0.01,
        boundary: Literal["clip", "drop", "wrap"] = "clip",
    ) -> None:
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        if not (0.0 <= p_d <= 1.0):
            raise ValueError(f"p_d must be in [0, 1], got {p_d}")
        if not (0.0 <= p_i <= 1.0):
            raise ValueError(f"p_i must be in [0, 1], got {p_i}")
        if boundary not in ("clip", "drop", "wrap"):
            raise ValueError(f"boundary must be 'clip', 'drop', or 'wrap', got '{boundary}'")
        self.T = int(T)
        self.sigma = float(sigma)
        self.p_d = float(p_d)
        self.p_i = float(p_i)
        self.boundary = boundary

    def jitter_kernel(self, sigma: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Discretized Gaussian kernel over integer offsets.

        Args:
            sigma: Jitter std.  Defaults to ``self.sigma``.

        Returns:
            Tuple of ``(offsets, probabilities)`` arrays.
        """
        s = sigma if sigma is not None else self.sigma
        half = max(1, int(np.ceil(3.0 * max(s, 1e-9))))
        js = np.arange(-half, half + 1)
        if s <= 0:
            k = np.zeros_like(js, float)
            k[js == 0] = 1.0
            return js, k
        k = np.exp(-(js ** 2) / (2 * s ** 2))
        return js, k / k.sum()

    def _insertion_buffer_cap(self) -> int:
        """Analytic high quantile for Binomial(T, p_i): mean + 9 sigma."""
        m = self.T * self.p_i
        sd = (self.T * self.p_i * (1 - self.p_i)) ** 0.5
        return int(np.ceil(m + 9 * sd))

    def simulate(
        self,
        pattern: np.ndarray,
        n: int,
        seed: int | np.random.Generator = 0,
        max_extra: int | None = None,
        sigma: float | None = None,
        p_d: float | None = None,
        p_i: float | None = None,
    ) -> np.ndarray:
        """Simulate one codeword pattern through the channel *n* times.

        Args:
            pattern: Spike positions as integer array, e.g. ``np.array([10, 20])``.
            n: Number of independent channel realizations.
            seed: Random seed or ``np.random.Generator``.
            max_extra: Override for insertion buffer size (for advanced use).
            sigma: Jitter sigma to use (defaults to channel sigma).
            p_d: Deletion probability override (defaults to channel ``p_d``).
            p_i: Insertion rate override (defaults to channel ``p_i``).

        Returns:
            Buffer of shape ``(n, L)`` where ``L = len(pattern) + extra_cap``,
            filled with received positions and NaN-padded for deleted/absent spikes.
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        pat = np.asarray(pattern, dtype=int)
        s = sigma if sigma is not None else self.sigma
        pd = p_d if p_d is not None else self.p_d
        pi = p_i if p_i is not None else self.p_i
        offs, kern = self.jitter_kernel(s)
        disp = rng.choice(offs, size=(n, len(pat)), p=kern)
        raw = pat[None, :] + disp
        if self.boundary == "clip":
            raw = np.clip(raw, 0, self.T - 1)
        elif self.boundary == "drop":
            raw = np.where((raw < 0) | (raw > self.T - 1), np.nan, raw)
        elif self.boundary == "wrap":
            raw = np.mod(raw, self.T)
        alive = rng.random((n, len(pat))) >= pd
        buf = np.where(alive, raw, np.nan)
        n_ins = rng.binomial(self.T, pi, size=n)
        extra = max_extra if max_extra is not None else self._insertion_buffer_cap()
        out = np.full((n, len(pat) + extra), np.nan)
        out[:, :len(pat)] = buf
        for i in range(len(pat), len(pat) + extra):
            m = n_ins >= (i - len(pat) + 1)
            out[m, i] = rng.integers(0, self.T, size=int(m.sum()))
        return out

    def decode(
        self,
        buf: np.ndarray,
        decoder: Literal["mean", "median"] = "median",
    ) -> np.ndarray:
        """Decode received-position buffer to estimated positions.

        Rows with no surviving spikes decode to the window center.

        Args:
            buf: Position buffer ``(n, L)`` from :meth:`simulate`.
            decoder: ``"mean"`` or ``"median"``.

        Returns:
            Estimated positions, shape ``(n,)``.
        """
        n_rows = buf.shape[0]
        with np.errstate(all="ignore"):
            has_data = ~np.isnan(buf).all(axis=1)
            est = np.full(n_rows, np.nan)
            if has_data.any():
                if decoder == "mean":
                    est[has_data] = np.nanmean(buf[has_data], axis=1)
                elif decoder == "median":
                    est[has_data] = np.nanmedian(buf[has_data], axis=1)
                else:
                    raise ValueError(f"unknown decoder '{decoder}'")
        return np.where(np.isnan(est), (self.T - 1) / 2, est)

    def pos_to_value(self, est_pos: np.ndarray) -> np.ndarray:
        """Map estimated positions to values in [0, 1].

        Args:
            est_pos: Estimated positions from :meth:`decode`.

        Returns:
            Values in [0, 1].
        """
        if self.T <= 1:
            return np.zeros_like(est_pos, dtype=float)
        return 1.0 - est_pos / (self.T - 1)

    def induced_channel(
        self,
        patterns: list[np.ndarray],
        K: int,
        n_per_pattern: int = 2500,
        decoder: Literal["mean", "median"] = "median",
        seed: int = 0,
        sigma: float | None = None,
        p_d: float | None = None,
        p_i: float | None = None,
    ) -> np.ndarray:
        """Compute the Monte-Carlo induced discrete channel P[decoded-bin | codeword].

        Args:
            patterns: List of spike-position arrays (one per codeword).
            K: Number of value bins.
            n_per_pattern: Channel realizations per codeword.
            decoder: Decoding method.
            seed: Random seed.
            sigma: Jitter sigma to use (defaults to channel sigma).
            p_d: Deletion probability override.
            p_i: Insertion rate override.

        Returns:
            Transition matrix of shape ``(M, K)`` where ``M = len(patterns)``,
            rows sum to 1.
        """
        rng = np.random.default_rng(seed)
        P = np.zeros((len(patterns), K))
        for wi, pat in enumerate(patterns):
            est = self.decode(
                self.simulate(pat, n_per_pattern, seed=rng, sigma=sigma, p_d=p_d, p_i=p_i),
                decoder=decoder,
            )
            vals = self.pos_to_value(est)
            kbins = np.clip((vals * K).astype(int), 0, K - 1)
            np.add.at(P[wi], kbins, 1)
        row_sums = P.sum(axis=1, keepdims=True)
        P = np.where(row_sums > 0, P / row_sums, 1.0 / K)
        return P

    def self_test(self, verbose: bool = True) -> bool:
        """Run sanity checks: noiseless identity + composition invariance.

        Args:
            verbose: Print results to stdout.

        Returns:
            ``True`` if all checks pass.
        """
        K = 64
        pats = [np.array([t]) for t in range(self.T)]
        ok = True
        P = self.induced_channel(pats, K, n_per_pattern=500, seed=1, sigma=0.0, p_d=0.0, p_i=0.0)
        nom_bin = np.clip(
            ((1.0 - np.arange(self.T) / (self.T - 1)) * K).astype(int), 0, K - 1
        )
        diag_mass = P[np.arange(self.T), nom_bin]
        if verbose:
            print(
                f"[selftest] noiseless diagonal mass: min={diag_mass.min():.4f} "
                f"(all > {1 - 1/K:.3f} required)"
            )
        ok &= bool((diag_mass > 1 - 1 / K).all())
        a = self.simulate(np.array([10]), n=20000, seed=2, sigma=1.5, p_d=0.4, p_i=0.0)
        offs_b, kern_b = self.jitter_kernel(1.5)
        rng_b = np.random.default_rng(3)
        b_pos = rng_b.choice(offs_b, size=(20000, 1), p=kern_b) + 10
        b_alive = np.random.default_rng(4).random((20000, 1)) >= 0.4
        b = np.where(b_alive, np.clip(b_pos, 0, self.T - 1), np.nan)
        ma, mb = np.nanmean(a[:, :1]), np.nanmean(b[:, :1])
        close = abs(ma - mb) < 0.15
        if verbose:
            print(
                f"[selftest] delete/jitter commutation means: {ma:.3f} vs {mb:.3f} "
                f"-> {'PASS' if close else 'FAIL'}"
            )
        ok &= bool(close)
        if verbose:
            print(f"[selftest] overall: {'PASS' if ok else 'FAIL'}")
        return bool(ok)
