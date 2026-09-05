"""Optional PyTorch wrappers for spire_rd.

Import this submodule only if you need ``torch.nn.Module`` integration::

    from spire_rd.torch import SpikeChannelLayer, SoftDecoderLayer

These wrappers call the numpy core algorithms under the hood and provide
autograd-compatible interfaces for integration with PyTorch training loops.
"""
from __future__ import annotations

import importlib.util

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    from .channel import SpikeChannelLayer as SpikeChannelLayer
    from .decoder import SoftDecoderLayer as SoftDecoderLayer
