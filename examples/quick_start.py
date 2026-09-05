"""Quick start example: encode, transmit through channel, decode, measure."""
import numpy as np

from spire_rd import SoftDecoder, SpikeChannel

# 1. Set up channel
ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)

# 2. Encode: map values to spike positions
K = 16
bin_centers = (np.arange(K) + 0.5) / K
w_map = [np.array([int(np.round((1.0 - bc) * (ch.T - 1)))]) for bc in bin_centers]

# 3. Transmit through noisy channel
rng = np.random.default_rng(42)
true_values = bin_centers  # one sample per bin
true_bins = np.arange(K)
bufs = np.vstack([ch.simulate(w_map[b], n=1, seed=rng) for b in true_bins])

# 4. Decode with PL
dec = SoftDecoder(ch, K=K)
log_L = dec.intensity_matrix(w_map)
decoded = dec.decode(bufs, log_L, method="pl")

# 5. Measure MSE
mse = np.mean((true_values - decoded) ** 2)
print(f"MSE (PL decode): {mse:.6f}")

# 6. Compare with median decode
decoded_median = dec.decode(bufs, None, method="median")
mse_median = np.mean((true_values - decoded_median) ** 2)
print(f"MSE (median):    {mse_median:.6f}")
print(f"PL advantage:    {(mse_median - mse) / mse_median * 100:.1f}%")
