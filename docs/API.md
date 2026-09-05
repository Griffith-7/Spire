# API Reference

## `spire_rd.channel.SpikeChannel`

```python
SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01, boundary="clip")
```

Canonical noisy spike channel.

**Methods:**

- `jitter_kernel(sigma)` -> `(offsets, probs)`
- `simulate(pattern, n, seed=0, max_extra=None)` -> `ndarray (n, L)`
- `decode(buf, decoder="median")` -> `ndarray (n,)`
- `pos_to_value(est_pos)` -> `ndarray (n,)`
- `induced_channel(patterns, K, n_per_pattern=2500, decoder="median", seed=0)` -> `ndarray (M, K)`
- `self_test(verbose=True)` -> `bool`

---

## `spire_rd.bounds.RDBound`

```python
RDBound(channel, K=64, nsamp=4000, seed=0)
```

R(D) bound computation.

**Methods:**

- `capacity(sigma=None, n_per_pattern=2500, iters=500, tol=1e-8)` -> `float`
- `compute(sigma=None, metric="mse", M=2, betas=None, eta=0.2, iters=25000, tol=1e-12)` -> `RDResult`
- `feasible_points(sigma=None, metric="mse", M=2)` -> `list[(label, D, R)]`

---

## `spire_rd.bounds.RDResult`

```python
RDResult(sigma, metric, points, hull, capacity, K, T, p_d, p_i)
```

**Methods:**

- `rate_at(D)` -> `float | None`
- `gain_vs_ttfs(ttfs_D, ttfs_R)` -> `float | None`

**Attributes:** `sigma`, `metric`, `points`, `hull`, `capacity`, `K`, `T`, `p_d`, `p_i`

---

## `spire_rd.codes.SpikeEncoder`

```python
SpikeEncoder(channel, K=64, m_set=[1, 2, 4], h_half=3, nsamp=4000, seed=0)
```

Multi-spike code family encoder.

**Methods:**

- `build_codewords()` -> `(patterns, lengths)`
- `compute(sigma=None, metric="mse", M=2, lambdas=None)` -> `EncoderResult`

---

## `spire_rd.codes.EncoderResult`

```python
EncoderResult(sigma, metric, points, hull, ttfs_D, ttfs_R, greedy_D, greedy_R, K, T, m_set, h_half)
```

**Attributes:** `points` (list of `(D, R, E, lambda)`), `hull`, `ttfs_D`, `ttfs_R`, `greedy_D`, `greedy_R`, `num_distinct_maps`

---

## `spire_rd.decoder.SoftDecoder`

```python
SoftDecoder(channel, K=64)
```

Channel-aware soft decoder.

**Methods:**

- `source_marginals(pattern, sigma=None)` -> `ndarray (m, T)`
- `intensity_matrix(w_map, sigma=None)` -> `ndarray (K, T)`
- `pl_scores(bufs, log_L)` -> `ndarray (n, K)`
- `decode(bufs, log_L, method="pl")` -> `ndarray (n,)`
- `counts_from_buffer(bufs)` -> `ndarray (n, T)`
- `exact_logG(Cmat, Qs, surv_pmf)` -> `ndarray (N, n_candidates)` (m=2 only)
  - `surv_pmf` = `[P(k=0), P(k=1), P(k=2)]` for exactly `k` of the two
    spikes surviving; the returned likelihood is normalized.

---

## PyTorch wrappers (`spire_rd.torch`)

### `SpikeChannelLayer`

```python
SpikeChannelLayer(T=32, sigma=1.0, p_d=0.05, p_i=0.01, seed=0)
```

`nn.Module` that injects channel noise with STE gradients.

### `SoftDecoderLayer`

```python
SoftDecoderLayer(channel, K=64)
```

`nn.Module` for PL decoding with STE gradients.

- `precompute_log_L(w_map, sigma=None)` -> `Tensor (K, T)`
- `forward(bufs, log_L)` -> `Tensor (n,)`
