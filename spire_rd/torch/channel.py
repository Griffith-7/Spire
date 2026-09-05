"""PyTorch layer wrapping the spike channel for differentiable noise injection.

This module provides ``SpikeChannelLayer``, an ``nn.Module`` that injects
T1 channel noise (jitter, deletion, insertion) between layers during
training using a straight-through estimator (STE) for gradients.
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


class _ChannelNoiseFn(Function):
    """Straight-through estimator for discrete spike channel noise."""

    @staticmethod
    def forward(ctx, x_tensor, sigma, p_d, p_i, T, seed):
        ctx.T = int(T)
        ctx.sigma = float(sigma)
        ctx.p_d = float(p_d)
        ctx.p_i = float(p_i)
        ctx.n_spikes = int(x_tensor.shape[1]) if x_tensor.dim() > 1 else 1
        ch = SpikeChannel(T=int(T), sigma=float(sigma), p_d=float(p_d), p_i=float(p_i))
        rng = np.random.default_rng(int(seed))
        x_np = x_tensor.detach().cpu().numpy()
        T_int = int(T)
        # Canonical T1 convention: value -> 1 - pos/(T-1), so a high value
        # maps to an early (low) spike position.
        pats = np.clip(np.round((1.0 - x_np) * (T_int - 1)).astype(int), 0, T_int - 1)
        out = np.empty(x_np.shape[0], dtype=np.float64)
        for i in range(x_np.shape[0]):
            buf = ch.simulate(pats[i], n=1, seed=rng)
            est = ch.decode(buf, decoder="median")
            out[i] = ch.pos_to_value(est)[0]
        return torch.tensor(out, dtype=x_tensor.dtype, device=x_tensor.device)

    @staticmethod
    def backward(ctx, grad_output):
        # Input shape (n, n_spikes); output shape (n,).  Distribute each
        # row's scalar gradient back across every spike in that row.
        return grad_output[:, None].expand(-1, ctx.n_spikes), None, None, None, None, None


class SpikeChannelLayer(nn.Module):
    """Applies T1 channel noise as a differentiable layer.

    During forward pass, input values in [0, 1] are quantized to spike
    positions, passed through the noisy channel, and decoded back to values.
    Gradients pass through via STE.

    Args:
        T: Number of clock steps.
        sigma: Jitter standard deviation.
        p_d: Deletion probability.
        p_i: Insertion probability.
        seed: Base random seed (incremented per forward call).

    Shape:
        - Input: ``(n, n_spikes)`` — each row is one codeword (values in [0, 1]).
        - Output: ``(n,)`` — one decoded value per codeword.

    Example::

        layer = SpikeChannelLayer(T=32, sigma=2.0, p_d=0.1, p_i=0.02)
        x_noisy = layer(x_clean)  # injects channel noise
        loss = criterion(model(x_noisy), targets)
        loss.backward()  # STE gradients flow through
    """

    def __init__(
        self,
        T: int = 32,
        sigma: float = 1.0,
        p_d: float = 0.05,
        p_i: float = 0.01,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.T = int(T)
        self.sigma = float(sigma)
        self.p_d = float(p_d)
        self.p_i = float(p_i)
        self._seed = int(seed)
        self.register_buffer("_call_count", torch.tensor(0, dtype=torch.long))
        self._call_count: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._call_count.add_(1)
        return _ChannelNoiseFn.apply(  # type: ignore[no-any-return]
            x, self.sigma, self.p_d, self.p_i, self.T, self._seed + self._call_count.item()
        )

    def extra_repr(self) -> str:
        return (
            f"T={self.T}, sigma={self.sigma:.2f}, "
            f"p_d={self.p_d:.2f}, p_i={self.p_i:.2f}"
        )
