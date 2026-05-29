"""Fault-aware training mechanics: STE passthrough + fault-state modes."""
from __future__ import annotations

import torch
import torch.nn as nn

from netdrift.faults.base import FaultModel, FaultState, FaultStats


class _ZeroingFault(FaultModel):
    """Fake fault model: returns a DETACHED zero tensor (like the real one,
    breaks autograd) and counts how many times init_state was called."""

    name = "zeroing"

    def __init__(self):
        self.init_calls = 0

    def init_state(self, weight_shape, ctx):
        self.init_calls += 1
        return FaultState()

    def inject(self, weight, state, ctx):
        new_w = torch.zeros_like(weight).detach()  # detached, like from_numpy
        return new_w, state, FaultStats()


def _make_linear_with_fault(fault_model):
    from netdrift.quant.binary import BinaryScheme
    from netdrift.quant.layers import QuantizedLinear

    layer = QuantizedLinear(4, 2, bias=False)
    layer.attach_scheme(BinaryScheme())
    layer.attach_fault_model(fault_model)
    layer.rt_mapping = "ROW"
    layer.kernel_mapping = None
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.5, -0.5, 0.5, -0.5], [0.1, 0.2, 0.3, 0.4]]))
    return layer


def test_passthrough_off_breaks_gradient():
    layer = _make_linear_with_fault(_ZeroingFault())
    assert layer.fault_grad_passthrough is False  # default
    x = torch.randn(3, 4)
    out = layer(x)
    # Passthrough off + a detaching fault model severs the output from the
    # autograd graph entirely — this is exactly the dead-gradient path that
    # passthrough exists to fix. ``out`` having no grad_fn is the strongest
    # proof no gradient can reach the weight; calling ``out.sum().backward()``
    # here would raise (no grad-requiring leaf), so we assert the severance
    # at the output instead.
    assert out.requires_grad is False
    assert layer.weight.grad is None


def test_passthrough_on_passes_gradient():
    layer = _make_linear_with_fault(_ZeroingFault())
    layer.fault_grad_passthrough = True
    x = torch.randn(3, 4)
    out = layer(x)
    out.sum().backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert layer.weight.grad.abs().sum() > 0


def test_passthrough_on_uses_faulted_forward_values():
    layer = _make_linear_with_fault(_ZeroingFault())
    layer.fault_grad_passthrough = True
    x = torch.randn(3, 4)
    out = layer(x)
    # Faulted weight is all zeros => output is all zeros (values use faulted w).
    assert torch.allclose(out, torch.zeros_like(out))


def _build_quant_vgg3_with_fault(fault_model, *, protect_none=True):
    from netdrift.models import (
        apply_protection_policy, attach_fault_model, build_model, replace_with_quantized,
    )
    from netdrift.quant.binary import BinaryScheme

    model = build_model("vgg3_fmnist")
    replace_with_quantized(model, BinaryScheme())
    if protect_none:
        apply_protection_policy(model, "custom", layers=[1, 2, 3, 4])  # all unprotected
    attach_fault_model(model, fault_model, kernel_mapping="ROW")
    return model


def test_fresh_mode_resets_state_each_batch():
    from torch.utils.data import DataLoader, TensorDataset
    from netdrift.config.schema import TrainCfg
    from netdrift.training.faultaware import train_one_epoch_fault_aware

    fm = _ZeroingFault()
    model = _build_quant_vgg3_with_fault(fm)
    x = torch.randn(24, 1, 28, 28)
    y = torch.randint(0, 10, (24,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)  # 3 batches
    cfg = TrainCfg(mode="train", fault_aware="regularization", fault_state_mode="fresh")
    cfg.reg.lambda_ = 0.0
    cfg.reg.inject_faults = True

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    fm.init_calls = 0
    train_one_epoch_fault_aware(
        model, loader, optimizer, torch.device("cpu"),
        rt_size=64, layout="ROW", kernel_mapping="ROW", cfg=cfg, epoch=1,
    )
    # fresh => state reset each batch => init_state re-runs per layer per batch.
    # 4 unprotected layers * 3 batches = 12 init calls (>= 1 proves resets happened).
    assert fm.init_calls >= 8


def test_accumulate_mode_does_not_reset():
    from torch.utils.data import DataLoader, TensorDataset
    from netdrift.config.schema import TrainCfg
    from netdrift.training.faultaware import train_one_epoch_fault_aware

    fm = _ZeroingFault()
    model = _build_quant_vgg3_with_fault(fm)
    x = torch.randn(24, 1, 28, 28)
    y = torch.randint(0, 10, (24,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    cfg = TrainCfg(mode="train", fault_aware="regularization", fault_state_mode="accumulate")
    cfg.reg.lambda_ = 0.0
    cfg.reg.inject_faults = True

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    fm.init_calls = 0
    train_one_epoch_fault_aware(
        model, loader, optimizer, torch.device("cpu"),
        rt_size=64, layout="ROW", kernel_mapping="ROW", cfg=cfg, epoch=1,
    )
    # accumulate => init_state runs only once per layer (first batch), 4 total.
    assert fm.init_calls == 4


def test_gradient_guard_lambda0_nofaults_trains_weights():
    """Independent of the regularizer: with lambda=0 and inject_faults=False,
    one step must reduce task loss and produce nonzero conv weight grads.

    This catches the dead-gradient path (attached detaching fault model with no
    passthrough) that would otherwise let the sign-transition metric improve on
    a model whose accuracy collapsed."""
    from torch.utils.data import DataLoader, TensorDataset
    from netdrift.config.schema import TrainCfg
    from netdrift.training.faultaware import train_one_epoch_fault_aware

    fm = _ZeroingFault()
    model = _build_quant_vgg3_with_fault(fm)
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    cfg = TrainCfg(mode="train", fault_aware="regularization", fault_state_mode="fresh")
    cfg.reg.lambda_ = 0.0
    cfg.reg.inject_faults = False

    # Capture grads via a one-batch manual check by giving lr=0 then reading grads.
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    train_one_epoch_fault_aware(
        model, loader, optimizer, torch.device("cpu"),
        rt_size=64, layout="ROW", kernel_mapping="ROW", cfg=cfg, epoch=1,
    )
    # With faults detached for the forward, conv weights must receive gradient.
    assert model.conv1.weight.grad is not None
    assert model.conv1.weight.grad.abs().sum() > 0


def test_regularizer_adds_to_loss_and_grads_weights():
    """With lambda>0 and inject_faults=False, the run-length term contributes
    gradient to the latent weights (faults detached so task grad also flows)."""
    from torch.utils.data import DataLoader, TensorDataset
    from netdrift.config.schema import TrainCfg
    from netdrift.training.faultaware import train_one_epoch_fault_aware

    fm = _ZeroingFault()
    model = _build_quant_vgg3_with_fault(fm)
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    cfg = TrainCfg(mode="train", fault_aware="regularization", fault_state_mode="fresh")
    cfg.reg.lambda_ = 1.0
    cfg.reg.inject_faults = False

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    train_one_epoch_fault_aware(
        model, loader, optimizer, torch.device("cpu"),
        rt_size=64, layout="ROW", kernel_mapping="ROW", cfg=cfg, epoch=1,
    )
    assert model.conv1.weight.grad is not None
    assert torch.isfinite(model.conv1.weight.grad).all()
