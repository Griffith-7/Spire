"""Tests for spire_rd.channel -- SpikeChannel."""
import numpy as np
import pytest

from spire_rd.channel import SpikeChannel


class TestSpikeChannelInit:
    def test_defaults(self):
        ch = SpikeChannel()
        assert ch.T == 32
        assert ch.sigma == 1.0
        assert ch.p_d == 0.05
        assert ch.p_i == 0.01
        assert ch.boundary == "clip"

    def test_custom_params(self):
        ch = SpikeChannel(T=16, sigma=2.0, p_d=0.1, p_i=0.02, boundary="drop")
        assert ch.T == 16
        assert ch.sigma == 2.0

    def test_invalid_T(self):
        with pytest.raises(ValueError, match="T must be >= 1"):
            SpikeChannel(T=0)

    def test_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            SpikeChannel(sigma=-1)

    def test_invalid_p_d(self):
        with pytest.raises(ValueError, match="p_d must be in"):
            SpikeChannel(p_d=1.5)

    def test_invalid_boundary(self):
        with pytest.raises(ValueError, match="boundary must be"):
            SpikeChannel(boundary="invalid")


class TestJitterKernel:
    def test_noiseless(self):
        ch = SpikeChannel(T=32, sigma=0.0)
        offs, kern = ch.jitter_kernel(0.0)
        assert kern.sum() == pytest.approx(1.0)
        assert kern[np.argmax(offs == 0)] == 1.0

    def test_symmetry(self):
        ch = SpikeChannel(T=32, sigma=1.0)
        offs, kern = ch.jitter_kernel()
        assert kern.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(kern, kern[::-1], atol=1e-10)

    def test_support_scales_with_sigma(self):
        ch = SpikeChannel(T=32)
        _, k1 = ch.jitter_kernel(0.5)
        _, k2 = ch.jitter_kernel(2.0)
        assert len(k2) > len(k1)


class TestSimulate:
    def test_shape(self):
        ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        buf = ch.simulate(np.array([10, 20]), n=100, seed=0)
        assert buf.shape[0] == 100
        assert buf.shape[1] >= 2  # at least len(pattern)

    def test_noiseless_near_identity(self):
        ch = SpikeChannel(T=32, sigma=0.0, p_d=0.0, p_i=0.0)
        pat = np.array([10])
        buf = ch.simulate(pat, n=500, seed=0)
        est = ch.decode(buf)
        np.testing.assert_allclose(est, 10.0, atol=0.1)

    def test_boundary_clip(self):
        ch = SpikeChannel(T=8, sigma=5.0, p_d=0.0, p_i=0.0, boundary="clip")
        buf = ch.simulate(np.array([0]), n=2000, seed=0)
        assert np.all((buf >= 0) | np.isnan(buf))

    def test_boundary_drop(self):
        ch = SpikeChannel(T=8, sigma=5.0, p_d=0.0, p_i=0.0, boundary="drop")
        buf = ch.simulate(np.array([0]), n=2000, seed=0)
        valid = buf[~np.isnan(buf)]
        assert np.all(valid >= 0)

    def test_boundary_wrap(self):
        ch = SpikeChannel(T=8, sigma=5.0, p_d=0.0, p_i=0.0, boundary="wrap")
        buf = ch.simulate(np.array([0]), n=2000, seed=0)
        valid = buf[~np.isnan(buf)]
        assert np.all(valid >= 0)
        assert np.all(valid < 8)


class TestDecode:
    def test_median_decode(self):
        ch = SpikeChannel(T=32)
        buf = np.array([[5.0, 6.0, 7.0], [10.0, np.nan, np.nan]])
        est = ch.decode(buf, decoder="median")
        assert est[0] == pytest.approx(6.0)
        assert est[1] == pytest.approx(10.0)

    def test_mean_decode(self):
        ch = SpikeChannel(T=32)
        buf = np.array([[5.0, 6.0, 7.0]])
        est = ch.decode(buf, decoder="mean")
        assert est[0] == pytest.approx(6.0)

    def test_silent_window(self):
        ch = SpikeChannel(T=32)
        buf = np.array([[np.nan, np.nan, np.nan]])
        est = ch.decode(buf)
        assert est[0] == pytest.approx(15.5)

    def test_invalid_decoder(self):
        ch = SpikeChannel(T=32)
        buf = np.array([[5.0]])
        with pytest.raises(ValueError, match="unknown decoder"):
            ch.decode(buf, decoder="invalid")


class TestPosToValue:
    def test_endpoints(self):
        ch = SpikeChannel(T=32)
        assert ch.pos_to_value(np.array([0.0])) == pytest.approx(1.0)
        assert ch.pos_to_value(np.array([31.0])) == pytest.approx(0.0)

    def test_center(self):
        ch = SpikeChannel(T=32)
        val = ch.pos_to_value(np.array([15.5]))
        assert val == pytest.approx(0.5)


class TestInducedChannel:
    def test_shape_and_rowsum(self):
        ch = SpikeChannel(T=16, sigma=1.0, p_d=0.05, p_i=0.01)
        pats = [np.array([t]) for t in range(16)]
        P = ch.induced_channel(pats, K=16, n_per_pattern=500, seed=0)
        assert P.shape == (16, 16)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-10)

    def test_noiseless_diagonal(self):
        ch = SpikeChannel(T=16, sigma=0.0, p_d=0.0, p_i=0.0)
        pats = [np.array([t]) for t in range(16)]
        K = 16
        P = ch.induced_channel(pats, K, n_per_pattern=200, seed=0)
        nom_bin = np.clip(
            ((1.0 - np.arange(16) / (16 - 1)) * K).astype(int), 0, K - 1
        )
        diag = P[np.arange(16), nom_bin]
        assert diag.min() > 0.9


class TestSelfTest:
    def test_passes(self):
        ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        assert ch.self_test(verbose=False) is True
