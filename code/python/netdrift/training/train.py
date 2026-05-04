"""Train / test loops.

Three entry points:

* :func:`train_one_epoch`     — one epoch of standard training.
* :func:`evaluate_clean`      — single test pass without fault injection
                                 (caller usually clears the fault model first).
* :func:`evaluate_with_faults`— inference loop with fault injection enabled.

Phase 2 will add fault-aware training modes (``ste_inject``, ``kd``,
``regularization``) by routing through a strategy object configured from the
``training.fault_aware`` field.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    epoch: int,
    *,
    log_interval: int = 10,
    log_fn: Callable[[str], None] = print,
) -> None:
    """One training epoch with optional batch-level logging."""
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target).mean()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            log_fn(
                f"Train Epoch {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} "
                f"({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    log_fn: Optional[Callable[[str], None]] = print,
) -> float:
    """Plain test pass returning top-1 accuracy %.

    Caller is responsible for ensuring the fault model is detached if a
    no-fault baseline is desired.
    """
    model.eval()
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.numel()
    accuracy = 100.0 * correct / total
    if log_fn is not None:
        log_fn(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def evaluate_with_faults(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    loops: int = 1,
    log_fn: Callable[[str], None] = print,
) -> list[float]:
    """Run ``loops`` consecutive inference passes; return per-pass accuracies.

    ``index_offset`` accumulates across passes (legacy behaviour: a stuck
    racetrack stays stuck across iterations until reset). The returned list
    has length ``loops``.
    """
    accuracies: list[float] = []
    for i in range(loops):
        log_fn(f"Inference {i + 1}/{loops}")
        start = time.perf_counter()
        acc = evaluate_clean(model, loader, device, log_fn=None)
        elapsed = time.perf_counter() - start
        log_fn(f"  acc={acc:.2f}%  elapsed={elapsed:.2f}s")
        accuracies.append(acc)
    return accuracies
