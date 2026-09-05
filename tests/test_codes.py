"""Tests for spire_rd.codes -- SpikeEncoder."""
import numpy as np

from spire_rd.channel import SpikeChannel
from spire_rd.codes import EncoderResult, SpikeEncoder


class TestBuildCodewords:
    def test_count(self):
        ch = SpikeChannel(T=16)
        enc = SpikeEncoder(ch, K=16, m_set=[1, 2], h_half=2)
        pats, lens = enc.build_codewords()
        expected = 16 * len([1, 2])
        assert len(pats) == expected
        assert len(lens) == expected

    def test_spike_counts_match(self):
        ch = SpikeChannel(T=16)
        enc = SpikeEncoder(ch, K=16, m_set=[1, 2, 4], h_half=2)
        pats, lens = enc.build_codewords()
        for pat, m in zip(pats, lens, strict=True):
            assert len(pat) == m

    def test_positions_in_range(self):
        ch = SpikeChannel(T=16)
        enc = SpikeEncoder(ch, K=16, m_set=[1, 2, 4], h_half=2)
        pats, _ = enc.build_codewords()
        for pat in pats:
            assert np.all(pat >= 0)
            assert np.all(pat < 16)


class TestCompute:
    def test_returns_encoder_result(self):
        ch = SpikeChannel(T=8, sigma=1.0, p_d=0.05, p_i=0.01)
        enc = SpikeEncoder(ch, K=8, m_set=[1, 2], h_half=2, nsamp=500)
        result = enc.compute(sigma=1.0, lambdas=np.logspace(-2, 1, 5))
        assert isinstance(result, EncoderResult)
        assert len(result.points) >= 2
        assert result.ttfs_D >= 0
        assert result.ttfs_R >= 0
        assert result.greedy_D >= 0
        assert result.greedy_R >= 0

    def test_hull_exists(self):
        ch = SpikeChannel(T=8, sigma=1.0, p_d=0.05, p_i=0.01)
        enc = SpikeEncoder(ch, K=8, m_set=[1, 2], h_half=2, nsamp=500)
        result = enc.compute(sigma=1.0, lambdas=np.logspace(-2, 1, 5))
        assert len(result.hull) >= 2


class TestEncoderResult:
    def test_num_distinct_maps(self):
        ch = SpikeChannel(T=8, sigma=1.0, p_d=0.05, p_i=0.01)
        enc = SpikeEncoder(ch, K=8, m_set=[1, 2], h_half=2, nsamp=500)
        result = enc.compute(sigma=1.0, lambdas=np.logspace(-2, 1, 5))
        assert result.num_distinct_maps >= 2
