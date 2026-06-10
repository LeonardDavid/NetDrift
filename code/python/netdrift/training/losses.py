"""Loss functions used by NetDrift.

Phase 1 only needs the binary multiclass hinge loss used by the legacy
training code. Phase 2 adds knowledge-distillation and fault-sensitivity
regularization losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from netdrift.faults.layout import _layout_weight_for_racetrack


def binary_hingeloss(yhat: torch.Tensor, y: torch.Tensor, b: float = 128.0) -> torch.Tensor:
    """Binary multiclass hinge loss (BNN training default).

    Encodes targets as ±1 and clips ``b - y_enc * yhat`` from below at zero.
    Returns a per-sample tensor (caller takes ``.mean()``).
    """
    y_enc = 2 * torch.nn.functional.one_hot(y, yhat.shape[-1]) - 1.0
    return ((b - y_enc * yhat).clamp(min=0).mean(dim=1)) / b


class BinaryHingeLoss:
    """Callable wrapper for :func:`binary_hingeloss` so it composes like nn losses."""

    def __init__(self, b: float = 128.0) -> None:
        self.b = b

    def __call__(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return binary_hingeloss(yhat, y, self.b)


def build_criterion(name: str, hinge_b: float = 128.0):
    """Return a loss callable selected by ``name``.

    ``"hinge"``         → :class:`BinaryHingeLoss` (the modified hinge loss, MHL,
                          per Yayla et al.) with the given ``hinge_b``.
    ``"cross_entropy"`` → :class:`torch.nn.CrossEntropyLoss`.

    Both callables work at the existing ``loss_fn(out, target).mean()`` call
    sites: ``BinaryHingeLoss`` returns a per-sample tensor, ``CrossEntropyLoss``
    returns a scalar, and ``.mean()`` on a scalar tensor is a no-op.
    """
    if name == "hinge":
        return BinaryHingeLoss(b=hinge_b)
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(
        f"unknown criterion {name!r}; expected 'hinge' or 'cross_entropy'"
    )


def run_length_penalty(
    model: nn.Module,
    *,
    beta: float,
    rt_size: int,
    layout: str,
    kernel_mapping: str,
) -> torch.Tensor:
    """Adjacent sign-agreement penalty over racetrack-aligned weights.

    For every unprotected quantized layer, lay the *latent* weight into its
    racetrack-aligned 2D view (the exact order the nanowire stores bits, via
    the shared layout helper), then for each adjacent pair ``(w_i, w_{i+1})``
    that lies INSIDE the same ``rt_size`` block, accumulate

        agree = tanh(beta*w_i) * tanh(beta*w_{i+1})    # +1 same sign, -1 opposite

    and return ``-mean(agree)`` over all such pairs across all contributing
    layers. Minimizing this loss lengthens same-sign runs along racetracks,
    which is what the RTM fault model and endlen care about. Pairs straddling a
    racetrack boundary are excluded (a sign change there costs nothing in the
    fault model). Returns a 0-dim tensor; ``0.0`` when no layer contributes.
    """
    # Imported here to avoid a hard dependency at module import time.
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    layout = layout.upper()
    kernel_mapping = kernel_mapping.upper()
    agree_terms: list[torch.Tensor] = []
    for _, module in model.named_modules():
        if not isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            continue
        if getattr(module, "protected", False):
            continue
        if getattr(module, "scheme", None) is None:
            continue
        weight = module.weight  # latent FP weight (a Parameter; keeps grad)
        if weight.numel() == 0:
            continue
        km = kernel_mapping if weight.dim() == 4 else None
        w_2d, _undo = _layout_weight_for_racetrack(
            weight, rt_mapping=layout, kernel_mapping=km
        )
        rows, cols = w_2d.shape
        n_full = cols // rt_size  # floor: ignore trailing partial track (endlen convention)
        if n_full == 0:
            continue
        # Reshape full tracks to (rows, n_full, rt_size); pairs are adjacent
        # within the last axis, so no pair crosses a block boundary.
        block = w_2d[:, : n_full * rt_size].reshape(rows, n_full, rt_size)
        s = torch.tanh(beta * block)
        agree = s[..., :-1] * s[..., 1:]  # (rows, n_full, rt_size-1)
        agree_terms.append(agree.reshape(-1))

    if not agree_terms:
        return torch.zeros((), dtype=torch.float32)
    all_agree = torch.cat(agree_terms)
    return -all_agree.mean()
