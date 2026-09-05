"""Tests for spire_rd.decoder -- SoftDecoder."""
import itertools

import numpy as np
import pytest

from spire_rd.channel import SpikeChannel
from spire_rd.decoder import SoftDecoder


class TestSourceMarginals:
    def test_row_sums_to_one(self):
        ch = SpikeChannel(T=32, sigma=1.0)
        dec = SoftDecoder(ch, K=32)
        pat = np.array([10, 20])
        Q = dec.source_marginals(pat)
        assert Q.shape == (2, 32)
        np.testing.assert_allclose(Q.sum(axis=1), 1.0, atol=1e-9)

    def test_noiseless_peak(self):
        ch = SpikeChannel(T=32, sigma=0.0)
        dec = SoftDecoder(ch, K=32)
        Q = dec.source_marginals(np.array([10]))
        assert Q[0, 10] == pytest.approx(1.0)


class TestIntensityMatrix:
    def test_shape(self):
        ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        dec = SoftDecoder(ch, K=16)
        w_map = [np.array([5]), np.array([10]), np.array([15])]
        L = dec.intensity_matrix(w_map)
        assert L.shape == (3, 32)

    def test_log_intensity_finite(self):
        ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        dec = SoftDecoder(ch, K=16)
        w_map = [np.array([5]), np.array([10])]
        L = dec.intensity_matrix(w_map)
        assert np.all(np.isfinite(L))

    def test_insertion_adds_floor(self):
        ch = SpikeChannel(T=32, sigma=0.0, p_d=0.0, p_i=0.01)
        dec = SoftDecoder(ch, K=16)
        L = dec.intensity_matrix([np.array([10])])
        expected_log = np.log(0.01)
        assert L[0, 10] > expected_log


class TestPLScores:
    def test_shape(self):
        ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        dec = SoftDecoder(ch, K=16)
        w_map = [np.array([i * 2]) for i in range(8)]
        log_L = dec.intensity_matrix(w_map)
        buf = ch.simulate(np.array([10]), n=50, seed=0)
        scores = dec.pl_scores(buf, log_L)
        assert scores.shape == (50, 8)


class TestDecode:
    def test_pl_shape(self):
        ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        dec = SoftDecoder(ch, K=16)
        w_map = [np.array([i * 2]) for i in range(16)]
        log_L = dec.intensity_matrix(w_map)
        buf = ch.simulate(np.array([10]), n=50, seed=0)
        values = dec.decode(buf, log_L, method="pl")
        assert values.shape == (50,)
        assert np.all(values >= 0)
        assert np.all(values <= 1)

    def test_median_decode(self):
        ch = SpikeChannel(T=32, sigma=0.0, p_d=0.0, p_i=0.0)
        dec = SoftDecoder(ch, K=16)
        buf = ch.simulate(np.array([10]), n=10, seed=0)
        values = dec.decode(buf, None, method="median")
        assert values.shape == (10,)

    def test_invalid_method(self):
        ch = SpikeChannel(T=32)
        dec = SoftDecoder(ch, K=16)
        buf = np.array([[5.0]])
        with pytest.raises(ValueError, match="unknown decode method"):
            dec.decode(buf, None, method="invalid")


class TestCountsFromBuffer:
    def test_basic(self):
        ch = SpikeChannel(T=8)
        dec = SoftDecoder(ch, K=8)
        buf = np.array([[1.0, 3.0, np.nan], [5.0, 5.0, 5.0]])
        C = dec.counts_from_buffer(buf)
        assert C.shape == (2, 8)
        assert C[0, 1] == 1
        assert C[0, 3] == 1
        assert C[1, 5] == 3


class TestExactLogG:
    def test_shape(self):
        ch = SpikeChannel(T=16, sigma=1.0, p_d=0.05, p_i=0.01)
        dec = SoftDecoder(ch, K=8)
        w_map = [np.array([5, 10]), np.array([3, 8])]
        Qs = [dec.source_marginals(pat) for pat in w_map]
        surv_pmf = [ch.p_d ** 2, 2 * ch.p_d * (1 - ch.p_d), (1 - ch.p_d) ** 2]
        buf = ch.simulate(np.array([5, 10]), n=20, seed=0)
        C = dec.counts_from_buffer(buf)
        logG = dec.exact_logG(C, Qs, surv_pmf)
        assert logG.shape == (20, 2)
        assert np.all(np.isfinite(logG))

    def test_normalizes_to_one(self):
        ch = SpikeChannel(T=6, sigma=1.0, p_d=0.1, p_i=0.05)
        dec = SoftDecoder(ch, K=4)
        patterns = [[2, 3], [1, 4], [0, 5], [1, 1]]
        Qs = [dec.source_marginals(np.array(p)) for p in patterns]
        surv_pmf = [ch.p_d ** 2, 2 * ch.p_d * (1 - ch.p_d), (1 - ch.p_d) ** 2]
        rows = []
        for nn in range(0, ch.T + 3):
            for combo in itertools.combinations_with_replacement(range(ch.T), nn):
                c = [0] * ch.T
                for t in combo:
                    c[t] += 1
                rows.append(c)
        Cmat = np.array(rows)
        logG = dec.exact_logG(Cmat, Qs, surv_pmf)
        np.testing.assert_allclose(np.exp(logG).sum(axis=0), 1.0, atol=1e-9)

    def test_zero_sum_counts_finite(self):
        ch = SpikeChannel(T=8, sigma=1.0, p_d=0.1, p_i=0.05)
        dec = SoftDecoder(ch, K=4)
        Qs = [dec.source_marginals(np.array([2, 3]))]
        surv_pmf = [ch.p_d ** 2, 2 * ch.p_d * (1 - ch.p_d), (1 - ch.p_d) ** 2]
        logG = dec.exact_logG(np.zeros((3, 8), dtype=int), Qs, surv_pmf)
        assert np.all(np.isfinite(logG))
