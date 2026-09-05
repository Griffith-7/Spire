"""Tests for spire_rd.bounds -- RDBound."""
import numpy as np

from spire_rd.bounds import RDBound, RDResult
from spire_rd.channel import SpikeChannel


class TestRDBoundInit:
    def test_defaults(self):
        ch = SpikeChannel(T=16, sigma=1.0)
        bound = RDBound(ch, K=16)
        assert bound.K == 16
        assert bound.nsamp == 4000


class TestCapacity:
    def test_positive(self):
        ch = SpikeChannel(T=16, sigma=1.0, p_d=0.05, p_i=0.01)
        bound = RDBound(ch, K=16, nsamp=500)
        cap = bound.capacity(sigma=0.5)
        assert cap > 0
        assert np.isfinite(cap)

    def test_monotone_in_sigma(self):
        ch = SpikeChannel(T=16, sigma=1.0, p_d=0.05, p_i=0.01)
        bound = RDBound(ch, K=16, nsamp=500)
        cap_low = bound.capacity(sigma=0.5)
        cap_high = bound.capacity(sigma=2.0)
        assert cap_low > cap_high


class TestCompute:
    def test_returns_rdresult(self):
        ch = SpikeChannel(T=8, sigma=1.0, p_d=0.05, p_i=0.01)
        bound = RDBound(ch, K=8, nsamp=500)
        result = bound.compute(sigma=1.0, betas=np.logspace(-1, 2, 10))
        assert isinstance(result, RDResult)
        assert result.sigma == 1.0
        assert result.capacity > 0
        assert len(result.points) == 10
        assert len(result.hull) >= 2

    def test_hull_is_decreasing(self):
        ch = SpikeChannel(T=8, sigma=1.0, p_d=0.05, p_i=0.01)
        bound = RDBound(ch, K=8, nsamp=500)
        result = bound.compute(sigma=1.0, betas=np.logspace(-1, 2, 10))
        rates = [r for _d, r in result.hull]
        for i in range(1, len(rates)):
            assert rates[i] <= rates[i - 1] + 1e-6


class TestRDResult:
    def test_rate_at(self):
        ch = SpikeChannel(T=8, sigma=1.0, p_d=0.05, p_i=0.01)
        bound = RDBound(ch, K=8, nsamp=500)
        result = bound.compute(sigma=1.0, betas=np.logspace(-1, 2, 10))
        d_mid = (result.hull[0][0] + result.hull[-1][0]) / 2
        rate = result.rate_at(d_mid)
        assert rate is not None
        assert rate >= 0


class TestFeasiblePoints:
    def test_ttfs_and_greedy(self):
        ch = SpikeChannel(T=16, sigma=1.0, p_d=0.05, p_i=0.01)
        bound = RDBound(ch, K=16, nsamp=500)
        fps = bound.feasible_points(sigma=1.0)
        labels = [fp[0] for fp in fps]
        assert "ttfs" in labels
        assert "greedy" in labels
        for _label, D, R in fps:
            assert D >= 0
            assert R >= 0

    def test_ttfs_and_greedy_k_smaller_than_T(self):
        # Regression: feasible_points raised IndexError (np.eye(K)[wmap])
        # when the codeword indices ranged over T but the eye matrix used K.
        ch = SpikeChannel(T=32, sigma=1.0, p_d=0.05, p_i=0.01)
        bound = RDBound(ch, K=16, nsamp=500)
        fps = bound.feasible_points(sigma=1.0)
        assert len(fps) == 2
        for _label, D, R in fps:
            assert np.isfinite(D) and np.isfinite(R)

    def test_task_metric(self):
        ch = SpikeChannel(T=8, sigma=1.0, p_d=0.05, p_i=0.01)
        bound = RDBound(ch, K=8, nsamp=500)
        fps = bound.feasible_points(sigma=1.0, metric="task", M=2)
        for _label, D, R in fps:
            assert D >= 0
            assert np.isfinite(R)
