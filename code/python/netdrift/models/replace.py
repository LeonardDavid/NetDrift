"""In-place layer replacement for quantization.

The strategy is: build a vanilla (FP32) model from the registry, then walk
its module tree swapping ``nn.Conv2d`` → :class:`QuantizedConv2d` and
``nn.Linear`` → :class:`QuantizedLinear` via ``setattr(parent, attr, new)``.
This preserves attribute paths (``conv1``, ``layer1.0.conv1``, ``fc.weight``,
...) and therefore preserves state-dict keys, which is the hard constraint
for HuggingFace checkpoint compatibility.

Layer ordering is the iteration order of ``model.named_modules()``, which is
deterministic and consistent across runs. Layers are auto-numbered as they
are swapped, so ``layer.layer_id`` matches the position in that traversal.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch.nn as nn

from netdrift.faults.base import FaultModel
from netdrift.quant.base import QuantScheme
from netdrift.quant.layers import QuantizedActivation, QuantizedConv2d, QuantizedLinear


# Module types that are *never* replaced. Useful when the user wants to keep
# the first/last layer at full precision (a common BNN recipe), or skip
# auxiliary classifier heads, etc.
DEFAULT_SKIP_TYPES: tuple[type, ...] = ()


def _swap_module(parent: nn.Module, attr: str, new: nn.Module) -> None:
    setattr(parent, attr, new)


def _new_qconv2d_from(src: nn.Conv2d) -> QuantizedConv2d:
    """Build a QuantizedConv2d with the same hyperparameters and weights as ``src``."""
    new = QuantizedConv2d(
        in_channels=src.in_channels,
        out_channels=src.out_channels,
        kernel_size=src.kernel_size,
        stride=src.stride,
        padding=src.padding,
        dilation=src.dilation,
        groups=src.groups,
        bias=src.bias is not None,
    )
    # Copy weights so a freshly-built (random) torchvision model preserves
    # its parameters when warm-started; for from-scratch use this is
    # effectively a no-op since we'll overwrite via load_state_dict anyway.
    with __import__("torch").no_grad():
        new.weight.copy_(src.weight)
        if src.bias is not None and new.bias is not None:
            new.bias.copy_(src.bias)
    return new


def _new_qlinear_from(src: nn.Linear) -> QuantizedLinear:
    """Build a QuantizedLinear with the same hyperparameters and weights as ``src``."""
    new = QuantizedLinear(
        in_features=src.in_features,
        out_features=src.out_features,
        bias=src.bias is not None,
    )
    with __import__("torch").no_grad():
        new.weight.copy_(src.weight)
        if src.bias is not None and new.bias is not None:
            new.bias.copy_(src.bias)
    return new


def replace_with_quantized(
    model: nn.Module,
    scheme: Optional[QuantScheme] = None,
    *,
    skip_first: bool = False,
    skip_last: bool = False,
    skip_modules: Optional[Iterable[str]] = None,
) -> nn.Module:
    """Walk ``model`` and swap ``Conv2d``/``Linear`` for their quantized cousins.

    Args:
        model:        Root module. Mutated in place.
        scheme:       Quantization scheme bound to every replaced layer. Pass
                      ``None`` to leave layers unbound (FP32 warm-start mode);
                      a scheme can be attached later via
                      ``layer.attach_scheme(scheme)``.
        skip_first:   Leave the first encountered ``Conv2d``/``Linear`` at full precision.
        skip_last:    Leave the last encountered ``Conv2d``/``Linear`` at full precision.
        skip_modules: Iterable of dotted names to skip (e.g. ``["fc"]``).

    Returns:
        The same model object, with quantized layers swapped in place.
    """
    skip_set = set(skip_modules or ())

    # First pass: enumerate all candidate swaps in iteration order so we can
    # honour skip_first/skip_last without traversing twice.
    candidates: list[tuple[nn.Module, str, nn.Module, str]] = []
    for name, module in model.named_modules():
        for attr, child in module.named_children():
            full = f"{name}.{attr}" if name else attr
            if full in skip_set:
                continue
            if isinstance(child, (nn.Conv2d, nn.Linear)) and not isinstance(
                child, (QuantizedConv2d, QuantizedLinear)
            ):
                candidates.append((module, attr, child, full))

    if not candidates:
        return model

    if skip_first:
        candidates = candidates[1:]
    if skip_last and candidates:
        candidates = candidates[:-1]

    for layer_idx, (parent, attr, child, full_name) in enumerate(candidates, start=1):
        if isinstance(child, nn.Conv2d):
            new = _new_qconv2d_from(child)
        else:
            new = _new_qlinear_from(child)
        new.layer_id = layer_idx
        new.layer_name = full_name
        if scheme is not None:
            new.attach_scheme(scheme)
        _swap_module(parent, attr, new)

    return model


def attach_activation_scheme(
    model: nn.Module,
    scheme: Optional[QuantScheme],
) -> nn.Module:
    """Bind ``scheme`` to every :class:`QuantizedActivation` in ``model``.

    With ``scheme=None`` the activations pass through unchanged (FP / plain-BNN
    behaviour). With a scheme (e.g. :class:`IntUniformActScheme`) every qact
    site quantizes its input. The scheme is a plain attribute, not a registered
    submodule parameter, so state-dict keys are unchanged.
    """
    for _, module in model.named_modules():
        if isinstance(module, QuantizedActivation):
            module.scheme = scheme
    return model


def attach_fault_model(
    model: nn.Module,
    fault_model: Optional[FaultModel],
    *,
    rt_mapping_fn: Optional[callable] = None,  # type: ignore[type-arg]
    kernel_mapping: Optional[str] = None,
    base_layout: Optional[str] = None,
) -> nn.Module:
    """Bind a fault model to every quantized layer in ``model``.

    Args:
        model:          Root module (after :func:`replace_with_quantized`).
        fault_model:    Fault model to bind. Pass ``None`` to clear bindings.
        rt_mapping_fn:  Callable ``(layer) -> str`` returning ``"ROW"`` /
                        ``"COL"`` for each layer. Defaults to ``"ROW"`` for
                        every layer when not given.
        kernel_mapping: Conv kernel mapping (``"ROW"``/``"COL"``/``"CLW"``/``"ACW"``).
                        Linear layers ignore this.
        base_layout:    ROW/COL base segmentation used underneath the BLOCK
                        mapping. ``None`` for ROW/COL storage layouts;
                        ``"ROW"``/``"COL"`` when ``storage.layout=="block"``.

    Returns:
        The same model.
    """
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            module.attach_fault_model(fault_model)
            module.rt_mapping = "ROW" if rt_mapping_fn is None else rt_mapping_fn(module)
            module.kernel_mapping = (
                kernel_mapping if isinstance(module, QuantizedConv2d) else None
            )
            module.base_layout = base_layout
    return model
