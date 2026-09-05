"""Edge case tests for spire_rd core modules.

Covers: T=1, p_d=1.0, sigma=0, -inf log-intensity, TTFS non-standard m_set,
empty patterns, and decoder numerical stability.
"""
import numpy as np
import pytest

from spire_rd.bounds import RDBound
from spire_rd.channel import SpikeChannel
from spire_rd.codes import SpikeEncoder
from spire_rd.decoder import SoftDecoder


class TestChannelT1:
    def test_simulate_works(self):
        ch = SpikeChannel(T=1, sigma=1.0)
        buf = ch.simulate(np.array([0]), n=10, seed=0)
        assert buf.shape[0] == 10

    def test_decode_returns_zero(self):
        ch = SpikeChannel(T=1, sigma=1.0)
        buf = ch.simulate(np.array([0]), n=10, seed=0)
        est = ch.decode(buf)
        assert est.shape == (10,)

    def test_pos_to_value_returns_zero(self):
        ch = SpikeChannel(T=1, sigma=1.0)
        val = ch.pos_to_value(np.array([0.0, 0.5]))
        np.testing.assert_array_equal(val, 0.0)

    def test_induced_channel_works(self):
        ch = SpikeChannel(T=1, sigma=0.0, p_d=0.0, p_i=0.0)
        P = ch.induced_channel([np.array([0])], K=1, n_per_pattern=100, seed=0)
        assert P.shape == (1, 1)
        assert P[0, 0] == pytest.approx(1.0)

    def test_jitter_kernel_works(self):
        ch = SpikeChannel(T=1, sigma=1.0)
        offs, kern = ch.jitter_kernel()
        assert kern.sum() == pytest.approx(1.0)


class TestChannelPD1:
    def test_induced_channel_no_nan(self):
        ch = SpikeChannel(T=16, sigma=1.0, p_d=1.0, p_i=0.0)
        pats = [np.array([t]) for t in range(16)]
        P = ch.induced_channel(pats, K=8, n_per_pattern=200, seed=0)
        assert not np.any(np.isnan(P))

    def test_induced_channel_rows_sum_to_one(self):
        ch = SpikeChannel(T=16, sigma=1.0, p_d=1.0, p_i=0.0)
        pats = [np.array([t]) for t in range(16)]
        P = ch.induced_channel(pats, K=8, n_per_pattern=200, seed=0)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-10)

    def test_simulate_all_deleted(self):
        ch = SpikeChannel(T=16, sigma=1.0, p_d=1.0, p_i=0.0)
        buf = ch.simulate(np.array([10]), n=100, seed=0)
        assert np.all(np.isnan(buf[:, :1]))


class TestChannelSigma0:
    def test_noiseless_capacity_high(self):
        ch = SpikeChannel(T=16, sigma=0.0, p_d=0.0, p_i=0.0)
        bound = RDBound(ch, K=16, nsamp=500)
        cap = bound.capacity(sigma=0.0)
        assert cap > 2.0

    def test_noiseless_simulation_identity(self):
        ch = SpikeChannel(T=16, sigma=0.0, p_d=0.0, p_i=0.0)
        buf = ch.simulate(np.array([8]), n=50, seed=0)
        est = ch.decode(buf)
        np.testing.assert_allclose(est, 8.0, atol=0.1)


class TestDecoderLogIntensity:
    def test_sigma0_pi0_no_nan(self):
        ch = SpikeChannel(T=16, sigma=0.0, p_d=0.0, p_i=0.0)
        dec = SoftDecoder(ch, K=16)
        w_map = [np.array([t]) for t in range(16)]
        log_L = dec.intensity_matrix(w_map)
        assert np.all(np.isfinite(log_L))

    def test_sigma0_pi0_no_nan_in_scores(self):
        ch = SpikeChannel(T=16, sigma=0.0, p_d=0.0, p_i=0.0)
        dec = SoftDecoder(ch, K=16)
        w_map = [np.array([t]) for t in range(16)]
        log_L = dec.intensity_matrix(w_map)
        buf = ch.simulate(np.array([8]), n=10, seed=0)
        scores = dec.pl_scores(buf, log_L)
        assert np.all(np.isfinite(scores))


class TestTTFSNonStandardMSet:
    def test_ttfs_map_works(self):
        ch = SpikeChannel(T=16, sigma=1.0, p_d=0.05, p_i=0.01)
        enc = SpikeEncoder(ch, K=16, m_set=[2, 4], h_half=2, nsamp=500)
        result = enc.compute(sigma=1.0, lambdas=np.array([1.0]))
        assert result.ttfs_D >= 0
        assert result.ttfs_R >= 0
        assert result.greedy_D >= 0
        assert result.greedy_R >= 0

    def test_ttfs_map_with_m_set_starting_at_2(self):
        ch = SpikeChannel(T=8, sigma=1.0, p_d=0.05, p_i=0.01)
        enc = SpikeEncoder(ch, K=8, m_set=[2, 3], h_half=1, nsamp=500)
        result = enc.compute(sigma=1.0, lambdas=np.array([1.0]))
        assert result.ttfs_D >= 0
        assert result.ttfs_R >= 0


class TestEmptyPattern:
    def test_simulate_empty_pattern(self):
        ch = SpikeChannel(T=16, sigma=1.0)
        buf = ch.simulate(np.array([], dtype=int), n=10, seed=0)
        assert buf.shape[0] == 10
        est = ch.decode(buf)
        assert est.shape == (10,)

    def test_decode_all_nan(self):
        ch = SpikeChannel(T=16, sigma=1.0)
        buf = np.full((5, 3), np.nan)
        est = ch.decode(buf)
        np.testing.assert_allclose(est, 7.5)


class TestDecoderPLNumerical:
    def test_pl_decode_silent_sample(self):
        ch = SpikeChannel(T=16, sigma=1.0, p_d=0.05, p_i=0.01)
        dec = SoftDecoder(ch, K=16)
        w_map = [np.array([t]) for t in range(16)]
        log_L = dec.intensity_matrix(w_map)
        buf = np.full((1, 5), np.nan)
        values = dec.decode(buf, log_L, method="pl")
        assert values.shape == (1,)
        assert np.isfinite(values[0])
        assert 0 <= values[0] <= 1

    def test_exact_logG_finite(self):
        ch = SpikeChannel(T=16, sigma=1.0, p_d=0.05, p_i=0.01)
        dec = SoftDecoder(ch, K=8)
        w_map = [np.array([5, 10]), np.array([3, 8])]
        Qs = [dec.source_marginals(pat) for pat in w_map]
        surv_pmf = [ch.p_d ** 2, 2 * ch.p_d * (1 - ch.p_d), (1 - ch.p_d) ** 2]
        buf = ch.simulate(np.array([5, 10]), n=20, seed=0)
        C = dec.counts_from_buffer(buf)
        logG = dec.exact_logG(C, Qs, surv_pmf)
        assert np.all(np.isfinite(logG))
