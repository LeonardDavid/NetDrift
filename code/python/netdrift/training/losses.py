"""Loss functions used by NetDrift.

Phase 1 only needs the binary multiclass hinge loss used by the legacy
training code. Phase 2 adds knowledge-distillation and fault-sensitivity
regularization losses.
"""

from __future__ import annotations

import torch


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
