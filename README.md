# spire

Information-theoretic rate-distortion bounds and optimal spike codes for spiking neural networks.

Computes the R(D)/capacity ceiling of a noisy spike channel, then builds
constructive codes that approach it.  The "LDPC moment" for spike codes.

## Installation

```bash
pip install -e .

# with PyTorch wrappers:
pip install -e ".[torch]"
```

Requires Python 3.10+, NumPy, SciPy.

## Quick start

```python
import numpy as np
from spire_rd import SpikeChannel, RDBound, SpikeEncoder, SoftDecoder

# Define the noisy spike channel
ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)

# Compute the R(D) ceiling
bound = RDBound(ch, K=64, nsamp=4000)
result = bound.compute(sigma=1.0)
print(f"capacity = {result.capacity:.3f} bits/spike")
print(f"frontier points: {len(result.hull)}")

# Build multi-spike codes
enc = SpikeEncoder(ch, K=64, m_set=[1, 2, 4], h_half=3)
enc_result = enc.compute(sigma=1.0)
print(f"TTFS: D={enc_result.ttfs_D:.4f}, R={enc_result.ttfs_R:.3f}")

# Decode with pseudo-likelihood
K = 64
dec = SoftDecoder(ch, K=K)
bin_centers = (np.arange(K) + 0.5) / K
w_map = [np.array([int(np.round((1 - bc) * (ch.T - 1)))]) for bc in bin_centers]
bufs = ch.simulate(np.array([10, 20]), n=64, seed=0)
log_L = dec.intensity_matrix(w_map)
values = dec.decode(bufs, log_L, method="pl")
```

## Modules

| Module | What it does |
|--------|-------------|
| `SpikeChannel` | Canonical noisy spike channel: jitter, deletion, insertion |
| `RDBound` | R(D) frontiers via direct I(X; x_hat) minimization |
| `SpikeEncoder` | Multi-spike code family with tunable redundancy knob |
| `SoftDecoder` | Channel-aware pseudo-likelihood decoder |

## Channel model

The spike channel applies three operations in fixed order:

1. **Jitter**: each spike position is displaced by a discretized Gaussian kernel
2. **Deletion**: each surviving spike is independently removed with probability `p_d`
3. **Insertion**: each clock step independently generates a spurious spike with probability `p_i`

Boundary handling after jitter: `clip` (default), `drop`, or `wrap`.

## R(D) computation

The `RDBound` class computes the information-theoretic rate-distortion ceiling
using exponentiated-gradient mirror descent that minimizes the **true** end-to-end
rate I(X; x_hat) — no surrogate, no looseness.

```python
result = bound.compute(
    sigma=1.0,       # jitter
    metric="mse",    # or "task" for classification
    betas=np.logspace(-2, 4, 60),
)
# result.hull gives the Pareto-optimal (D, R) frontier
# result.capacity gives the channel capacity
```

## Optional PyTorch integration

```python
from spire_rd.torch import SpikeChannelLayer, SoftDecoderLayer

# Inject channel noise during training
noise_layer = SpikeChannelLayer(T=32, sigma=2.0, p_d=0.1, p_i=0.02)
x_noisy = noise_layer(x_clean)

# PL decoding layer
dec_layer = SoftDecoderLayer(ch, K=64)
values = dec_layer(bufs, log_L)
```

## Tests

```bash
pytest tests/
```

## License

MIT
