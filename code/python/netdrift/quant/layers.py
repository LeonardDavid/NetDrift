"""Thin quantized layers delegating to a :class:`QuantScheme` + optional :class:`FaultModel`.

These replace the ~600-line legacy ``QuantizedNN.QuantizedLinear`` /
``QuantizedConv2d`` with a uniform forward pass:

.. code-block:: python

    qw = self.scheme.quantize(self.weight, self.scale_per_channel)
    if not self.protected and self.fault_model is not None:
        qw_t, self.fault_state, stats = self.fault_model.inject(qw.values, self.fault_state, ctx)
        self.metrics.record_stats(stats)
        qw = QuantizedTensor(qw_t, qw.bits, qw.levels, qw.per_channel_scale)
    return F.linear(input, qw.values)   # or F.conv2d

Layer protection moved from a model-level ``protectLayers`` array to a
per-module ``protected`` flag, set by
:func:`netdrift.models.protection.apply_protection_policy`.

State-dict compatibility:

* The layer subclasses ``nn.Linear``/``nn.Conv2d``, so ``self.weight`` (and
  optional ``self.bias``) appear under their natural names. This preserves
  existing checkpoints.
* For multi-bit schemes (Phase 3+), an extra ``scale_per_channel`` parameter
  is registered. It is **not** present for binary schemes, so binary
  checkpoints continue to load with ``strict=True``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from netdrift.faults.base import FaultCtx, FaultModel, FaultState
from netdrift.metrics.layer import LayerMetrics
from netdrift.quant.base import QuantizedTensor, QuantScheme


class _STEQuantize(Function):
    """Straight-through estimator wrapping a :class:`QuantScheme`.

    Forward: scheme.quantize(weight). Backward: identity. This matches the
    legacy behaviour (``Quantize`` autograd function in QuantizedNN.py:13).
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, scheme: QuantScheme,
                per_channel_scale: Optional[torch.Tensor]) -> torch.Tensor:
        qt = scheme.quantize(weight, per_channel_scale)
        return qt.values

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


def _ste_quantize(weight: torch.Tensor, scheme: QuantScheme,
                  per_channel_scale: Optional[torch.Tensor]) -> torch.Tensor:
    return _STEQuantize.apply(weight, scheme, per_channel_scale)


class _QuantizedMixin:
    """Shared init/state for QuantizedLinear and QuantizedConv2d.

    Held as a mixin (not a separate base class) so the resulting layers can
    still inherit cleanly from ``nn.Linear`` / ``nn.Conv2d`` and reuse their
    weight/bias parameter registration, which preserves state-dict keys.
    """

    layer_id: int = 0
    layer_name: str = ""
    scheme: Optional[QuantScheme] = None
    fault_model: Optional[FaultModel] = None
    fault_state: Optional[FaultState] = None
    metrics: LayerMetrics
    protected: bool = False
    nr_run: int = 0
    rt_mapping: Optional[str] = None
    kernel_mapping: Optional[str] = None

    def _init_quant(self, scheme: Optional[QuantScheme] = None) -> None:
        self.scheme = scheme
        self.fault_model = None
        self.fault_state = None
        self.metrics = LayerMetrics()
        self.protected = False
        self.nr_run = 0
        self.layer_id = 0
        self.layer_name = ""
        self.rt_mapping = None
        self.kernel_mapping = None
        # Per-channel scale parameter is created on demand by attach_scheme()
        # so that binary schemes (which don't need it) leave the state_dict
        # unchanged.

    def attach_scheme(self, scheme: QuantScheme) -> None:
        """Bind a quantization scheme. Allocates ``scale_per_channel`` if needed."""
        self.scheme = scheme
        if scheme.needs_per_channel_scale:
            out_channels = self.weight.shape[0]  # type: ignore[attr-defined]
            if not hasattr(self, "scale_per_channel"):
                self.register_parameter(  # type: ignore[attr-defined]
                    "scale_per_channel",
                    nn.Parameter(torch.ones(out_channels, dtype=self.weight.dtype)),  # type: ignore[attr-defined]
                )

    def attach_fault_model(self, fault_model: Optional[FaultModel]) -> None:
        """Bind a fault model. State is allocated lazily on the first forward."""
        self.fault_model = fault_model
        self.fault_state = None  # reallocated on next inject

    def _maybe_init_fault_state(self) -> None:
        if self.fault_model is None or self.fault_state is not None:
            return
        scheme_bits = self.scheme.bits if self.scheme is not None else 1
        ctx = FaultCtx(
            layer_id=self.layer_id,
            layer_name=self.layer_name,
            nr_run=self.nr_run + 1,
            training=bool(getattr(self, "training", False)),
            bits=scheme_bits,
            extra={
                "rt_mapping": self.rt_mapping,
                "kernel_mapping": self.kernel_mapping,
                "kernel_size": self._kernel_size_for_state(),
            },
        )
        self.fault_state = self.fault_model.init_state(tuple(self.weight.shape), ctx)  # type: ignore[attr-defined]

    def _kernel_size_for_state(self) -> Optional[int]:
        # Convs override; linear has no kernel.
        return None

    def _build_ctx(self) -> FaultCtx:
        scheme_bits = self.scheme.bits if self.scheme is not None else 1
        return FaultCtx(
            layer_id=self.layer_id,
            layer_name=self.layer_name,
            nr_run=self.nr_run + 1,
            training=bool(getattr(self, "training", False)),
            bits=scheme_bits,
            extra={
                "rt_mapping": self.rt_mapping,
                "kernel_mapping": self.kernel_mapping,
                "kernel_size": self._kernel_size_for_state(),
            },
        )

    def _quantize_then_inject(self, weight: torch.Tensor) -> torch.Tensor:
        """Run the scheme then (if attached and not protected) the fault model."""
        if self.scheme is None:
            qw = weight  # raw FP32, e.g. before any scheme is attached (warm start)
        else:
            scale = getattr(self, "scale_per_channel", None)
            qw = _ste_quantize(weight, self.scheme, scale)

        if self.fault_model is None or self.protected:
            return qw

        self._maybe_init_fault_state()
        self.nr_run += 1
        ctx = self._build_ctx()
        new_w, new_state, stats = self.fault_model.inject(qw, self.fault_state, ctx)  # type: ignore[arg-type]
        self.fault_state = new_state
        self.metrics.record_stats(stats)
        return new_w


class QuantizedLinear(nn.Linear, _QuantizedMixin):
    """Linear layer with optional quantization + fault injection.

    Constructor mirrors ``nn.Linear``; ``scheme``, ``fault_model``,
    ``rt_mapping`` etc. are attached after construction (they have to be —
    ``replace_with_quantized`` swaps the module first, then binds the scheme/
    fault model).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self._init_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        w = self._quantize_then_inject(self.weight)
        return F.linear(x, w, self.bias)


class QuantizedConv2d(nn.Conv2d, _QuantizedMixin):
    """Conv2d layer with optional quantization + fault injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
        )
        self._init_quant()

    def _kernel_size_for_state(self) -> Optional[int]:
        # Conv kernel_size is a tuple (kh, kw); we report the side length.
        kh, kw = self.kernel_size  # type: ignore[misc]
        if kh != kw:
            raise NotImplementedError(f"non-square kernels not supported: {self.kernel_size}")
        return int(kh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        w = self._quantize_then_inject(self.weight)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantizedActivation(nn.Module):
    """Activation-side quantization (binarize the output).

    Used after ``htanh`` in the legacy VGG/ResNet topologies. Carries no fault
    model — only forward-side faults on weights are simulated. The legacy
    ``BinarizeFIModel`` activation-side error injection is intentionally not
    ported (priority 1 says: clean :class:`FaultModel` interface; the legacy
    activation-side path was binary-only and rarely used).
    """

    def __init__(self, scheme: Optional[QuantScheme] = None) -> None:
        super().__init__()
        self.scheme = scheme

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scheme is None:
            return x
        return _ste_quantize(x, self.scheme, None)
