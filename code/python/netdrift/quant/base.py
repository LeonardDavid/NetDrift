"""QuantScheme abstract base class and supporting types.

A scheme converts a real-valued weight tensor into a :class:`QuantizedTensor`
whose values lie on a discrete level set. The forward pass of every quantized
layer calls :meth:`QuantScheme.quantize`. Backward gradients flow through the
straight-through estimator (the surrounding autograd ``Function`` in
:mod:`netdrift.quant.layers` handles this — schemes themselves do not need to
manage gradients).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import torch


# Scale init policy for multi-bit schemes (used by :class:`IntUniformScheme` etc.)
# ``max_abs``      → ``scale = max(|W_c|) / (2^(bits-1) - 1)`` (per output channel)
# ``quantile_<x>`` → ``scale = quantile(|W_c|, x) / (2^(bits-1) - 1)`` (robust to outliers)
ScaleInit = Literal["max_abs", "quantile_999", "quantile_99", "quantile_95"]


@dataclass
class QuantizedTensor:
    """A weight tensor along with metadata describing its quantization.

    The :class:`~netdrift.storage.base.WeightLayout` consumes this rather than a
    raw tensor so that storage strategies can lay out racetracks correctly for
    any bit-width (Phase 3).

    Attributes:
        values:        The tensor with values on the discrete level set. Same shape as the source weight.
        bits:          Number of bits per value (1 for binary, 2 for ternary, etc.).
        levels:        The full discrete level set, e.g. ``[-1, 1]`` for binary.
                       For per-channel schemes this is the per-channel range —
                       still a fixed-length tuple per channel.
        per_channel_scale: Optional scale per output channel (multi-bit only).
                       Shape ``(out_channels,)``. ``None`` for binary/ternary.
    """

    values: torch.Tensor
    bits: int
    levels: tuple[float, ...]
    per_channel_scale: Optional[torch.Tensor] = None


class QuantScheme(ABC):
    """Stateless quantization scheme.

    Subclasses implement :meth:`quantize` (forward) and declare ``bits``,
    ``levels``, and whether they need per-channel scales. The scheme itself
    holds no per-layer state — scales live on the layer as
    ``nn.Parameter``\\ s so they show up in ``state_dict``.
    """

    bits: int
    """Bits per value (1 for binary, 2 for ternary, 4 for INT4, ...)."""

    needs_per_channel_scale: bool = False
    """Whether the scheme requires per-output-channel learnable scales."""

    @abstractmethod
    def quantize(
        self,
        weight: torch.Tensor,
        per_channel_scale: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        """Quantize ``weight`` to the scheme's discrete levels.

        Args:
            weight:            Real-valued weight tensor.
            per_channel_scale: Per-output-channel scale ``(out_channels,)``.
                               Required if :attr:`needs_per_channel_scale`.

        Returns:
            A :class:`QuantizedTensor` whose ``values`` have the same dtype/device/shape as ``weight``.
        """

    def init_scale(
        self,
        weight: torch.Tensor,
        policy: ScaleInit = "max_abs",
    ) -> Optional[torch.Tensor]:
        """Compute initial per-channel scale from an FP32 weight tensor.

        Used by the checkpoint adapter when warm-starting from FP32. Returns
        ``None`` for schemes that don't use per-channel scales.

        Args:
            weight: FP32 weight tensor as it would be loaded from a checkpoint.
            policy: Scale-init policy (``max_abs``, ``quantile_999``, ...).

        Returns:
            Tensor of shape ``(out_channels,)`` or ``None``.
        """
        if not self.needs_per_channel_scale:
            return None

        # Reduce over all dims except the first (out_channels)
        flat = weight.reshape(weight.shape[0], -1).abs()
        if policy == "max_abs":
            stat = flat.max(dim=1).values
        elif policy.startswith("quantile_"):
            q = float("0." + policy.split("_", 1)[1])
            stat = torch.quantile(flat, q, dim=1)
        else:
            raise ValueError(f"Unknown scale-init policy: {policy}")

        denom = float(2 ** (self.bits - 1) - 1)
        if denom <= 0:
            raise ValueError(f"init_scale requires bits >= 2, got bits={self.bits}")
        return stat / denom
