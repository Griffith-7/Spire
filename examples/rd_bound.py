"""Compute R(D) bounds for a noisy spike channel."""
import numpy as np

from spire_rd import RDBound, SpikeChannel

# 1. Define the channel
ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)

# 2. Compute capacity
bound = RDBound(ch, K=16, nsamp=1000)
cap = bound.capacity(sigma=1.0)
print(f"Channel capacity: {cap:.3f} bits/spike")

# 3. Compute R(D) frontier (fast sweep with fewer betas)
result = bound.compute(
    sigma=1.0,
    metric="mse",
    betas=np.logspace(-1, 3, 20),  # 20 points for speed
)
print(f"\nR(D) frontier ({len(result.points)} points):")
for D, R in result.hull:
    print(f"  D={D:.6f}  R={R:.4f} bits")

# 4. Compare with deterministic baselines
fps = bound.feasible_points(sigma=1.0)
print("\nDeterministic baselines:")
for label, D, R in fps:
    gain = result.gain_vs_ttfs(D, R)
    gain_str = f"  gain={gain:.1f}%" if gain is not None else "  (outside hull)"
    print(f"  {label}: D={D:.6f}  R={R:.4f}{gain_str}")
