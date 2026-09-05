"""spire_rd -- Information-theoretic rate-distortion bounds and optimal spike codes.

A production-grade plugin for computing R(D) ceilings on noisy spike channels
and building constructive codes that approach them.

Quick start::

    from spire_rd import SpikeChannel, RDBound, SpikeEncoder, SoftDecoder

    ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
    bound = RDBound(ch, K=64)
    result = bound.compute(sigma=1.0)
    print(f"capacity = {result.capacity:.3f} bits/spike")

Core modules:

- :class:`SpikeChannel` -- canonical noisy spike channel (jitter, deletion, insertion)
- :class:`RDBound` -- R(D) frontiers via direct I(X; x_hat) minimization
- :class:`SpikeEncoder` -- multi-spike code family with tunable redundancy
- :class:`SoftDecoder` -- channel-aware pseudo-likelihood decoder

Optional torch wrappers (requires ``pip install spire-rd[torch]``)::

    from spire_rd.torch import SpikeChannelLayer, SoftDecoderLayer
"""
from __future__ import annotations

import numpy as np

__version__ = "1.0.0"
__author__ = "Sumith Kumar"

from .bounds import RDBound, RDResult
from .channel import SpikeChannel
from .codes import EncoderResult, SpikeEncoder
from .decoder import SoftDecoder

__all__ = [
    "SpikeChannel",
    "RDBound",
    "RDResult",
    "SpikeEncoder",
    "EncoderResult",
    "SoftDecoder",
    "self_test",
]


def self_test(verbose: bool = True) -> bool:
    """Run end-to-end sanity checks on the core modules.

    Checks:
    1. Noiseless channel identity (diagonal mass > 1 - 1/K).
    2. R(D) capacity is positive and finite.
    3. Encoder builds correct codeword count.
    4. Decoder PL scores have correct shape.

    Args:
        verbose: Print results.

    Returns:
        ``True`` if all checks pass.
    """
    ok = True

    # 1. Channel self-test
    ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
    if not ch.self_test(verbose=verbose):
        ok = False

    # 2. Capacity
    bound = RDBound(ch, K=16, nsamp=500)
    cap = bound.capacity(sigma=0.5)
    if verbose:
        print(f"[selftest] capacity @ sigma=0.5: {cap:.3f} bits/spike")
    ok &= cap > 0 and np.isfinite(cap)

    # 3. Encoder codeword count
    enc = SpikeEncoder(ch, K=16, m_set=[1, 2], h_half=2)
    pats, lens = enc.build_codewords()
    expected = ch.T * len([1, 2])
    if verbose:
        print(f"[selftest] codeword count: {len(pats)} (expected {expected})")
    ok &= len(pats) == expected

    # 4. Decoder shape
    dec = SoftDecoder(ch, K=16)
    pat = np.array([5, 10])
    log_L = dec.intensity_matrix([pat, pat])
    if verbose:
        print(f"[selftest] intensity matrix shape: {log_L.shape} (expected (2, {ch.T}))")
    ok &= log_L.shape == (2, ch.T)

    if verbose:
        print(f"[selftest] overall: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


if __name__ == "__main__":
    self_test()
