"""Pattern-preserving recalibration (Capability #1).

After the endlen encoder flips bits to lengthen same-sign racetrack runs, the
network's clean accuracy typically drops a little. Recalibration recovers that
gap WITHOUT moving any binary weight sign: it re-fits the parts downstream of
the frozen weights — BatchNorm running stats / affine params and the learnable
output ``Scale``.

No fault injection happens here (the caller detaches the fault model first):
this targets the clean-accuracy loss endlen introduces, not accumulating-fault
robustness. Because BN and Scale live after ``F.linear``/``F.conv2d``, updating
them cannot change a single weight sign — the endlen bit-pattern is preserved.

Two sub-steps:

* ``bn_stats``    — put the model in ``train()`` and run forward passes so BN
                    re-estimates running mean/var (no backward).
* ``tune_affine`` — a short backprop fine-tune of BN gamma/beta + Scale only.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
from netdrift.training.losses import build_criterion


def _set_recal_trainable(model: nn.Module) -> list[nn.Parameter]:
    """Freeze everything except BN affine params + the output Scale.

    Returns the list of trainable parameters (BN gamma/beta + Scale.scale).
    Quantized-layer latent weights are frozen so their signs cannot move.
    """
    trainable: list[nn.Parameter] = []
    # Freeze all parameters first.
    for p in model.parameters():
        p.requires_grad_(False)
    # Unfreeze BN affine params.
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                module.weight.requires_grad_(True)
                trainable.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                trainable.append(module.bias)
    # Unfreeze the output Scale (attribute name 'scale' is a Scale module with
    # a 'scale' Parameter; see topologies.Scale).
    scale_mod = getattr(model, "scale", None)
    if scale_mod is not None and hasattr(scale_mod, "scale"):
        scale_mod.scale.requires_grad_(True)
        trainable.append(scale_mod.scale)
    return trainable


@torch.no_grad()
def _reestimate_bn_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: Optional[int],
) -> None:
    """Reset BN running stats, then forward N batches in train() to re-estimate.

    Resetting momentum to None makes BN use a cumulative moving average, so the
    estimate reflects exactly the batches seen here regardless of prior history.
    """
    bns = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    saved_momentum = []
    for m in bns:
        m.reset_running_stats()
        saved_momentum.append(m.momentum)
        m.momentum = None  # cumulative moving average
    model.train()
    for i, (data, _target) in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        model(data.to(device))
    for m, mom in zip(bns, saved_momentum):
        m.momentum = mom


def _tune_affine(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    trainable: list[nn.Parameter],
    criterion: str = "hinge",
    hinge_b: float = 128.0,
) -> None:
    """Short backprop fine-tune of BN affine + Scale only.

    Uses the baseline-training criterion (``criterion``/``hinge_b``); default
    is the modified hinge loss with b=128, matching plain BNN training.
    """
    if epochs <= 0 or not trainable:
        return
    optimizer = torch.optim.Adam(trainable, lr=lr)
    loss_fn = build_criterion(criterion, hinge_b)
    model.train()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, target).mean()
            loss.backward()
            optimizer.step()


def recalibrate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg,
    *,
    criterion: str = "hinge",
    hinge_b: float = 128.0,
) -> None:
    """Run the configured recalibration sub-steps in place.

    ``cfg`` is a :class:`netdrift.config.schema.RecalibrateCfg`. ``criterion`` /
    ``hinge_b`` select the loss for sub-step B (``tune_affine``) — the caller
    passes the baseline-training criterion (``cfg.training.criterion`` /
    ``hinge_b``); default is the modified hinge loss with b=128. The caller is
    responsible for detaching the fault model BEFORE calling this (no faults
    during recalibration) and restoring it / calling ``model.eval()`` after.
    """
    trainable = _set_recal_trainable(model)
    if cfg.tune_affine:
        # Keep gamma/beta trainable for sub-step B; otherwise re-freeze them so
        # only running stats move.
        pass
    else:
        for p in trainable:
            p.requires_grad_(False)

    if cfg.bn_stats:
        _reestimate_bn_stats(model, loader, device, cfg.num_batches)
    if cfg.tune_affine:
        _tune_affine(
            model, loader, device,
            epochs=cfg.epochs, lr=cfg.lr, trainable=trainable,
            criterion=criterion, hinge_b=hinge_b,
        )
    model.eval()
