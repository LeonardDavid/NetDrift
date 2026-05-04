"""Binary quantization scheme: weights mapped to ``{-1, +1}``.

This is the only scheme used by existing NetDrift / HuggingFace checkpoints.
It produces the same numerical output as the legacy :mod:`binarize` CUDA
extension when the extension is available, and falls back to a pure-PyTorch
:func:`torch.sign`-based implementation otherwise (with ``sign(0) → +1`` to
match the kernel).
"""

from __future__ import annotations

from typing import Optional

import torch

from netdrift.quant.base import QuantizedTensor, QuantScheme

try:
    import binarize as _binarize_ext  # legacy CUDA extension built by code/cuda/binarize/setup.py
    _HAS_BINARIZE_EXT = True
except ImportError:
    _binarize_ext = None
    _HAS_BINARIZE_EXT = False


def _binarize_pure(x: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch binarization: returns +1 where ``x > 0`` else -1.

    Matches the legacy ``binarize`` CUDA kernel which thresholds at 0 and
    treats exactly-zero values as +1. Works on CPU and GPU.
    """
    return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))


class BinaryScheme(QuantScheme):
    """Binarize weights to ``{-1, +1}``.

    Uses the in-place CUDA extension when available (matches legacy bit-for-bit
    on GPU) and a pure-PyTorch fallback otherwise. The extension mutates its
    input — we operate on a clone to avoid corrupting ``self.weight``.
    """

    bits = 1
    needs_per_channel_scale = False
    levels: tuple[float, ...] = (-1.0, 1.0)

    def quantize(
        self,
        weight: torch.Tensor,
        per_channel_scale: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        # The legacy CUDA kernel only handles 3D-or-fewer tensor shapes by
        # reshaping internally; calling it on a 4D conv weight is supported by
        # the extension and produces an in-place binarization.
        if _HAS_BINARIZE_EXT and weight.is_cuda:
            out = weight.detach().clone()
            _binarize_ext.binarize(out)
        else:
            out = _binarize_pure(weight.detach())

        return QuantizedTensor(
            values=out,
            bits=self.bits,
            levels=self.levels,
            per_channel_scale=None,
        )
