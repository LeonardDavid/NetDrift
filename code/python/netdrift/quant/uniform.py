"""Uniform symmetric quantization on ``[-1, 1]`` for activations.

Used in the W1A_n_ recipe: weights stay binary (``BinaryScheme``), while every
post-htanh activation is snapped to one of ``2**bits`` evenly-spaced levels on
``[-1, 1]``. Forward = clamp + round-to-grid. Backward = identity STE (the
surrounding :class:`~netdrift.quant.layers._STEQuantize` handles gradients).

The scheme assumes a `Hardtanh` upstream — VGG3/VGG7 satisfy this. Architectures
without a clipping non-linearity (ReLU-based ResNet etc.) would need a
PACT/LSQ-style learned clip, which is deliberately not implemented here.
"""

from __future__ import annotations

from typing import Optional

import torch

from netdrift.quant.base import QuantizedTensor, QuantScheme


class IntUniformActScheme(QuantScheme):
    """Symmetric uniform activation quantization to ``2**bits`` levels on ``[-1, 1]``.

    Level set ``{-1 + 2k/(2^bits - 1) : k = 0..2^bits - 1}``. ``bits=1`` reduces
    to the binary sign quantizer (``{-1, +1}``).
    """

    needs_per_channel_scale = False

    def __init__(self, bits: int) -> None:
        if bits < 1:
            raise ValueError(f"IntUniformActScheme requires bits >= 1, got {bits}")
        self.bits = bits
        self.n_levels = 1 << bits  # 2**bits
        # step between adjacent levels; n_levels==2 ⇒ step=2.0 (i.e. {-1, +1})
        self.step = 2.0 / (self.n_levels - 1) if self.n_levels > 1 else 2.0
        self.levels = tuple(-1.0 + k * self.step for k in range(self.n_levels))

    def quantize(
        self,
        x: torch.Tensor,
        per_channel_scale: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        # Defensive clamp: Hardtanh upstream already enforces this, but applying
        # it here keeps the scheme self-contained for callers without htanh.
        x_c = x.clamp(-1.0, 1.0)

        if self.n_levels == 2:
            # Binary fast path: sign with sign(0) -> +1, matching BinaryScheme.
            ones = torch.ones_like(x_c)
            out = torch.where(x_c >= 0, ones, -ones)
        else:
            # Snap to the nearest grid point. Round-to-nearest-even on ties is
            # fine — STE backward is identity so the rounding rule does not
            # affect gradients.
            out = torch.round((x_c + 1.0) / self.step) * self.step - 1.0

        return QuantizedTensor(
            values=out,
            bits=self.bits,
            levels=self.levels,
            per_channel_scale=None,
        )
