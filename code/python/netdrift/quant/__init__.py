"""Quantization schemes.

A :class:`QuantScheme` defines how a continuous weight tensor is mapped to a
discrete level set during the forward pass. Schemes are stateless objects
attached to each :class:`~netdrift.quant.layers.QuantizedConv2d` /
:class:`~netdrift.quant.layers.QuantizedLinear`. Per-layer parameters such as
``scale_per_channel`` live on the layer (so they appear in the state dict),
not on the scheme.

Available schemes:

* :class:`BinaryScheme`              — values in ``{-1, +1}``, 1 bit/value.
* :class:`TernaryScheme`             — values in ``{-1, 0, +1}`` (Phase 3).
* :class:`IntUniformScheme`          — uniform INT2/4/8 with per-channel scale (Phase 3).
* :class:`MixedPrecisionScheme`      — per-channel manual bit assignment (Phase 3).
"""

from netdrift.quant.base import QuantScheme, QuantizedTensor, ScaleInit
from netdrift.quant.binary import BinaryScheme
from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear, QuantizedActivation
from netdrift.quant.uniform import IntUniformActScheme

__all__ = [
    "QuantScheme",
    "QuantizedTensor",
    "ScaleInit",
    "BinaryScheme",
    "IntUniformActScheme",
    "QuantizedConv2d",
    "QuantizedLinear",
    "QuantizedActivation",
]
