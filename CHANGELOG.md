# Changelog

## 1.0.0 (2026-09-05)

Initial production release.

### Core modules
- `SpikeChannel` — canonical noisy spike channel (jitter, deletion, insertion)
- `RDBound` — R(D) frontiers via direct I(X; x_hat) minimization
- `SpikeEncoder` — multi-spike code family with tunable redundancy
- `SoftDecoder` — channel-aware pseudo-likelihood decoder

### PyTorch wrappers
- `SpikeChannelLayer` — differentiable channel noise injection (STE)
- `SoftDecoderLayer` — PL decoding as nn.Module

### Fixes
- `RDBound.feasible_points`: index the induced channel with `np.eye(T)`, not
  `np.eye(K)` — fixed `IndexError` when `K < T` (e.g. K=16, T=32).
- `examples/rd_bound.py`: removed unsupported `nsamp=` kwarg from
  `RDBound.compute()` (the bug that crashed the example).
- `SpikeChannelLayer`: corrected value→position encoding to the canonical T1
  convention `pos = round((1 - value) * (T-1))`, matching the numpy core.
- `SpikeChannelLayer` / `SoftDecoderLayer`: straight-through backward now
  broadcasts each row's scalar gradient back across the input columns (STE
  gradient shape now matches the input tensor).
- `SoftDecoder.exact_logG`: corrected the exact m=2 likelihood — added the
  multinomial insertion normalization `(1/T)^b b! / prod_t c_t!`, fixed the
  survivor-pattern comb factors (k=1 divided by C(2,1), k=2 over unordered
  pairs plus coincident slots), and documented `surv_pmf` as k-major
  `[P(k=0), P(k=1), P(k=2)]`. The likelihood now normalizes to 1 and was
  validated against channel Monte Carlo.
- Removed a NumPy 1.25 deprecation warning (array→scalar conversion) in the
  torch channel layer.

### Quality
- Full test coverage for the PyTorch wrappers (`tests/test_torch.py`, 12 tests),
  81 tests total.
- Added ruff + mypy tooling and configuration; the source passes both.
- Added GitHub Actions CI (lint / tests on Python 3.10–3.12 / torch).
- Classifier set to `5 - Production/Stable`.