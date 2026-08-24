"""Fault-aware training loop (Capabilities #2 and #3).

Why this exists separately from ``train_one_epoch``: the RTM fault model is
non-differentiable (Numba CUDA → numpy → ``torch.from_numpy`` returns a
detached tensor) and is attached to every unprotected layer. A plain training
loop through such a model gets ZERO gradient to the weights. This loop handles
the gradient path explicitly:

* ``reg.inject_faults == False`` — detach the fault model for the loop so the
  quantized weight (which IS differentiable via the existing weight STE) flows
  to the latent weight; restore the fault model afterward.
* ``reg.inject_faults == True``  — keep the fault model attached but set
  ``module.fault_grad_passthrough = True`` on every contributing layer so the
  layer returns ``qw + (faulted - qw).detach()`` (STE residual): faulted VALUES
  in the forward, gradient through ``qw``.

Fault-state modes (only relevant when faults are injected):

* ``fresh``      — reset every layer's ``fault_state`` to None at the START of
                   each batch, so a NEW misalignment realization is sampled per
                   batch. Acts as augmentation over the fault distribution; the
                   trained weights generalize across realizations.
                   **RECOMMENDED for training.**
* ``accumulate`` — never reset within the epoch, so faults persist across
                   batches (stuck-stays-stuck), matching the eval-sweep
                   semantics. Faithful to one deployment scenario but risks
                   overfitting BN/weights to a single fault realization.

``fault_aware`` dispatch: ``regularization`` adds ``lambda_ *`` the penalty named
by ``reg.objective`` (``run_length_penalty`` by default, ``ppm_count_penalty``
for PPM window sign-counts) to the task loss; ``ste_inject`` trains the task loss on faulted weights with no
run-length term; ``kd`` additionally distills from a clean teacher (deferred —
raises NotImplementedError until implemented).
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
from netdrift.training.losses import (
    build_criterion,
    ppm_count_penalty,
    run_length_penalty,
)

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False

    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


def _effective_lambda(reg, epoch: int) -> float:
    """``reg.lambda_``, linearly ramped over ``reg.lambda_warmup_epochs``.

    ``epoch`` is 1-based (as in runner/run.py's loop), so a 4-epoch warm-up
    applies 1/4, 2/4, 3/4 and then the full weight. With
    ``lambda_warmup_epochs == 0`` this is the identity.

    The ramp exists because a converged model cannot absorb its non-conforming
    windows being fixed all at once: doing exactly that post-hoc took vgg7 from
    88.19% to 10.00% (see faults/ppm_align.py).
    """
    warmup = int(getattr(reg, "lambda_warmup_epochs", 0) or 0)
    if warmup <= 0:
        return float(reg.lambda_)
    return float(reg.lambda_) * min(1.0, epoch / warmup)


def _reg_penalty(model, reg, *, rt_size: int, layout: str, kernel_mapping: str):
    """The penalty ``reg.lambda_`` scales, selected by ``reg.objective``.

    ``run_length`` (default) is the adjacent sign-agreement surrogate over the
    training layout. ``ppm_count`` targets PPM's window sign-counts and takes its
    view from ``reg.ppm_base_layout`` rather than the training ``layout``: the
    objective describes the DEPLOYMENT layout, so a model trained under
    ``storage.layout=col`` can be optimised for evaluation under ``polarity``
    (which cannot itself be the training layout — the packing paths cache their
    structure from the first forward, see ``_validate_block_layout_combo``).
    """
    if reg.objective == "ppm_count":
        return ppm_count_penalty(
            model, beta=reg.ppm_beta, rt_size=rt_size,
            base_layout=reg.ppm_base_layout, window=reg.ppm_window,
            kernel_mapping=kernel_mapping,
        )
    return run_length_penalty(
        model, beta=reg.beta, rt_size=rt_size, layout=layout,
        kernel_mapping=kernel_mapping,
    )


def _contributing_layers(model: nn.Module):
    """Unprotected quantized layers — the ones exposed to faults / regularized."""
    for _, m in model.named_modules():
        if isinstance(m, (QuantizedConv2d, QuantizedLinear)) and not getattr(m, "protected", False):
            yield m


def _set_passthrough(model: nn.Module, value: bool) -> None:
    for m in _contributing_layers(model):
        m.fault_grad_passthrough = value


def _reset_fault_state(model: nn.Module) -> None:
    for m in _contributing_layers(model):
        m.fault_state = None
        m.nr_run = 0


def train_one_epoch_fault_aware(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    rt_size: int,
    layout: str,
    kernel_mapping: str,
    cfg,
    epoch: int,
) -> float:
    """One fault-aware training epoch. ``cfg`` is a TrainCfg.

    Returns the mean per-batch total loss. Gradient-path handling and
    fault-state mode are driven by ``cfg.reg.inject_faults`` and
    ``cfg.fault_state_mode``; the run-length term by ``cfg.fault_aware`` +
    ``cfg.reg.lambda_``.
    """
    if cfg.fault_aware == "kd":
        raise NotImplementedError(
            "fault_aware='kd' is not implemented yet (schema/dispatch are in "
            "place; implement the teacher-distillation body before using it)."
        )

    # ``ste_inject`` is *defined* by training on faulted weights, so it implies
    # injection regardless of reg.inject_faults. ``regularization`` injects only
    # when the user opts in via reg.inject_faults.
    inject = bool(cfg.reg.inject_faults) or cfg.fault_aware == "ste_inject"
    use_reg = cfg.fault_aware == "regularization" and cfg.reg.lambda_ > 0

    # Gradient-path setup. Either detach faults entirely, or keep them attached
    # with STE passthrough so the latent weights still receive gradient.
    saved_fault_models = {}
    if inject:
        _set_passthrough(model, True)
    else:
        # Detach the fault model for the forward passes; restore afterward.
        for name, m in model.named_modules():
            if isinstance(m, (QuantizedConv2d, QuantizedLinear)):
                saved_fault_models[name] = m.fault_model
                m.attach_fault_model(None)

    loss_fn = build_criterion(cfg.fault_aware_criterion, cfg.fault_aware_hinge_b)
    model.train()
    running = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=f"fa-epoch {epoch}", leave=False, file=sys.stdout)
    try:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            if inject and cfg.fault_state_mode == "fresh":
                _reset_fault_state(model)  # new realization this batch
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, target).mean()
            if use_reg:
                loss = loss + _effective_lambda(cfg.reg, epoch) * _reg_penalty(
                    model, cfg.reg, rt_size=rt_size,
                    layout=layout, kernel_mapping=kernel_mapping,
                )
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
            if _HAS_TQDM:
                pbar.set_postfix(loss=f"{running / max(n_batches,1):.4f}")
    finally:
        if inject:
            _set_passthrough(model, False)
        else:
            for name, m in model.named_modules():
                if isinstance(m, (QuantizedConv2d, QuantizedLinear)) and name in saved_fault_models:
                    m.attach_fault_model(saved_fault_models[name])
        if _HAS_TQDM:
            pbar.close()
    return running / n_batches if n_batches > 0 else 0.0
