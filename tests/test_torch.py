"""Tests for the optional PyTorch wrappers (spire_rd.torch).

These are only run when torch is installed.  They verify forward shapes,
STE gradient flow, output range, and consistency with the numpy core.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spire_rd import SpikeChannel  # noqa: E402
from spire_rd.torch import SoftDecoderLayer, SpikeChannelLayer  # noqa: E402


class TestSpikeChannelLayer:
    def test_shapes(self):
        # Each row is one codeword; output is one value per codeword.
        layer = SpikeChannelLayer(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        x = torch.full((4, 3), 0.5)
        out = layer(x)
        assert out.shape == (4,)
        assert out.dtype == x.dtype

    def test_extreme_inputs_stay_bounded(self):
        layer = SpikeChannelLayer(T=32, sigma=3.0, p_d=0.3, p_i=0.05)
        x = torch.tensor([[0.3], [0.7]])
        out = layer(x)
        assert torch.isfinite(out).all()
        assert (out >= 0).all() and (out <= 1).all()

    def test_noiseless_approximates_single_spike(self):
        layer = SpikeChannelLayer(T=32, sigma=0.0, p_d=0.0, p_i=0.0)
        vals = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        x = torch.tensor(vals)
        out = layer(x)
        np.testing.assert_allclose(out.numpy(), vals[:, 0], atol=0.05)

    def test_autograd_flows(self):
        layer = SpikeChannelLayer(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        x = torch.full((3, 5), 0.5, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        # STE broadcasts each row's scalar gradient across its spikes.
        np.testing.assert_allclose(x.grad.numpy(), np.ones((3, 5)))

    def test_repeated_calls_advance_seed(self):
        layer = SpikeChannelLayer(T=16, sigma=2.0, p_d=0.2, p_i=0.05)
        x = torch.tensor([[0.5]])
        a = layer(x).item()
        b = layer(x).item()
        assert a != b  # different noise per forward call

    def test_extra_repr(self):
        layer = SpikeChannelLayer(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        assert "T=32" in layer.extra_repr()
        assert "sigma" in layer.extra_repr()


class TestSoftDecoderLayer:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.ch = SpikeChannel(T=16, sigma=1.0, p_d=0.05, p_i=0.01)
        self.K = 16
        self.layer = SoftDecoderLayer(self.ch, K=self.K)
        # One codeword per value bin (the invariant log_L.shape[0] == K).
        self.w_map = [
            np.array([int(np.round((1 - (i + 0.5) / self.K) * 15))]) for i in range(self.K)
        ]

    def test_precompute_log_L_shape(self):
        log_L = self.layer.precompute_log_L(self.w_map)
        assert log_L.shape == (self.K, self.ch.T)

    def test_forward_shape(self):
        bufs = torch.full((5, 16), np.nan, dtype=torch.float64)
        log_L = self.layer.precompute_log_L(self.w_map)
        out = self.layer(bufs, log_L)
        assert out.shape == (5,)

    def test_decoded_values_in_range(self):
        layer = self.layer
        rng = np.random.default_rng(0)
        pat = np.array([5, 10])
        buf = self.ch.simulate(pat, n=20, seed=rng)
        bufs = torch.tensor(buf, dtype=torch.float64)
        log_L = layer.precompute_log_L(self.w_map)
        out = layer(bufs, log_L)
        assert torch.isfinite(out).all()
        assert (out >= 0).all() and (out <= 1).all()

    def test_autograd_flows(self):
        layer = self.layer
        log_L = layer.precompute_log_L(self.w_map)
        rng = np.random.default_rng(1)
        buf = self.ch.simulate(np.array([5, 10]), n=10, seed=rng)
        bufs = torch.tensor(buf, dtype=torch.float64, requires_grad=True)
        layer(bufs, log_L).sum().backward()
        assert bufs.grad is not None
        assert bufs.grad.shape == bufs.shape
        # STE: gradient is distributed to the non-NaN (received) spikes.
        assert torch.isfinite(bufs.grad).any()

    def test_silent_samples_decode_to_midpoint(self):
        layer = self.layer
        bufs = torch.full((2, 8), np.nan, dtype=torch.float64)
        log_L = layer.precompute_log_L(self.w_map)
        out = layer(bufs, log_L)
        np.testing.assert_allclose(out.numpy(), 0.5, atol=1e-9)

    def test_extra_repr(self):
        assert "K=16" in self.layer.extra_repr()
