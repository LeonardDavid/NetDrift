"""build_criterion factory: name → loss callable. CPU-only."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def test_build_criterion_hinge_default():
    from netdrift.training.losses import BinaryHingeLoss, build_criterion

    crit = build_criterion("hinge")
    assert isinstance(crit, BinaryHingeLoss)
    assert crit.b == 128.0


def test_build_criterion_hinge_custom_b():
    from netdrift.training.losses import BinaryHingeLoss, build_criterion

    crit = build_criterion("hinge", hinge_b=64.0)
    assert isinstance(crit, BinaryHingeLoss)
    assert crit.b == 64.0


def test_build_criterion_cross_entropy():
    from netdrift.training.losses import build_criterion

    crit = build_criterion("cross_entropy")
    assert isinstance(crit, nn.CrossEntropyLoss)


def test_build_criterion_invalid_raises():
    from netdrift.training.losses import build_criterion

    with pytest.raises(ValueError):
        build_criterion("focal")


def test_both_criteria_compose_with_mean_call_site():
    """Both losses must work at the existing ``loss_fn(out, target).mean()`` site.

    BinaryHingeLoss returns a per-sample tensor; CrossEntropyLoss returns a
    scalar. ``.mean()`` on a scalar is a no-op, so both yield a 0-dim tensor.
    """
    from netdrift.training.losses import build_criterion

    out = torch.randn(8, 10)
    target = torch.randint(0, 10, (8,))
    for name in ("hinge", "cross_entropy"):
        loss = build_criterion(name)(out, target).mean()
        assert loss.dim() == 0
        assert torch.isfinite(loss)
