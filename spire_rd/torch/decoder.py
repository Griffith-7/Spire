"""PyTorch layer wrapping the soft decoder.

Provides ``SoftDecoderLayer``, an ``nn.Module`` that applies channel-aware
pseudo-likelihood decoding as a differentiable operation.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.autograd import Function
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if not _HAS_TORCH:
    raise ImportError(
        "PyTorch is required for spire_rd.torch. "
        "Install with: pip install spire-rd[torch]"
    )

from ..channel import SpikeChannel
from ..decoder import SoftDecoder


class _PLDecodeFn(Function):
    """Pseudo-likelihood decoding with straight-through gradient."""

    @staticmethod
    def forward(ctx, bufs_tensor, log_L_tensor, K, T):
        ctx.K = int(K)
        ctx.T = int(T)
        ctx.n_cols = int(bufs_tensor.shape[1])
        bufs_np = bufs_tensor.detach().cpu().numpy()
        log_L_np = log_L_tensor.detach().cpu().numpy().astype(np.float64)
        from scipy.special import softmax as _softmax
        bin_centers = (np.arange(int(K)) + 0.5) / int(K)
        scores = np.zeros((bufs_np.shape[0], log_L_np.shape[0]))
        for j in range(bufs_np.shape[1]):
            col = bufs_np[:, j]
            ok = ~np.isnan(col)
            idx = np.where(ok)[0]
            if len(idx):
                scores[idx] += log_L_np[:, col[idx].astype(int)].T
        values = _softmax(scores, axis=1) @ bin_centers
        silent = np.all(np.isnan(bufs_np), axis=1)
        values[silent] = 0.5
        return torch.tensor(values, dtype=bufs_tensor.dtype, device=bufs_tensor.device)

    @staticmethod
    def backward(ctx, grad_output):
        # Input shape (n, L); output shape (n,).  Distribute each row's
        # scalar gradient back across the buffer's columns (STE).
        return grad_output[:, None].expand(-1, ctx.n_cols), None, None, None


class SoftDecoderLayer(nn.Module):
    """Channel-aware pseudo-likelihood decoding layer.

    Takes a raw spike-train buffer and a precomputed log-intensity matrix,
    and produces decoded values via intensity pseudo-likelihood.  Gradients
    flow through via STE.

    Args:
        channel: A :class:`~spire_rd.channel.SpikeChannel` instance.
        K: Number of value bins.

    Example::

        ch = SpikeChannel(T=32, sigma=1.0)
        dec_layer = SoftDecoderLayer(ch, K=64)
        log_L = dec_layer.precompute_log_L(w_map)
        values = dec_layer(bufs, log_L)
    """

    def __init__(self, channel: SpikeChannel, K: int = 64) -> None:
        super().__init__()
        self.decoder = SoftDecoder(channel, K)
        self.K = int(K)
        self.T = channel.T

    def precompute_log_L(
        self, w_map: list[np.ndarray], sigma: float | None = None
    ) -> torch.Tensor:
        """Precompute log-intensity matrix for a codebook.

        Args:
            w_map: List of spike-position arrays.
            sigma: Jitter sigma (defaults to channel sigma).

        Returns:
            Log-intensity matrix as float64 tensor, shape ``(K, T)``.
        """
        log_L = self.decoder.intensity_matrix(w_map, sigma)
        return torch.tensor(log_L, dtype=torch.float64)

    def forward(
        self, bufs: torch.Tensor, log_L: torch.Tensor
    ) -> torch.Tensor:
        """Decode spike-train buffers to values.

        Args:
            bufs: Position buffer tensor ``(n, L)`` with NaN for absent spikes.
            log_L: Precomputed log-intensity matrix ``(K, T)``.

        Returns:
            Decoded values in [0, 1], shape ``(n,)``.
        """
        return _PLDecodeFn.apply(bufs, log_L, self.K, self.T)  # type: ignore[no-any-return]

    def extra_repr(self) -> str:
        return f"K={self.K}, T={self.T}"
