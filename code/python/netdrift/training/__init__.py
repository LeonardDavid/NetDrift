"""Training and test loops.

Phase 1 ships clean train/test loops with the standard binary hinge loss
and the legacy ``Clippy`` (gradient-clipping Adam) optimizer. Phase 2 adds
fault-aware modes: ``ste_inject``, ``kd``, ``regularization``.
"""

from netdrift.training.losses import BinaryHingeLoss, binary_hingeloss
from netdrift.training.optim import Clippy
from netdrift.training.train import (
    evaluate_clean,
    evaluate_with_faults,
    train_one_epoch,
)

__all__ = [
    "BinaryHingeLoss",
    "binary_hingeloss",
    "Clippy",
    "train_one_epoch",
    "evaluate_clean",
    "evaluate_with_faults",
]
