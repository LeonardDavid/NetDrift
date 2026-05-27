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

import sys
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:  # tqdm not installed — fall back to a no-op shim
    _HAS_TQDM = False

    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


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
) -> float:
    """One training epoch with a tqdm progress bar (loss in the postfix).

    Returns the mean per-batch loss over the epoch (0.0 if the loader was empty).
    """
    model.train()
    pbar = tqdm(
        loader,
        desc=f"epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    )
    running_loss = 0.0
    n_batches = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target).mean()
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
        n_batches += 1
        if _HAS_TQDM:
            pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}")
    if _HAS_TQDM:
        pbar.close()
    if not _HAS_TQDM and n_batches > 0:
        # Fallback when tqdm isn't installed: one summary line per epoch.
        log_fn(f"Train Epoch {epoch}  avg_loss={running_loss / n_batches:.6f}")
    return running_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    log_fn: Optional[Callable[[str], None]] = print,
    desc: Optional[str] = None,
) -> float:
    """Plain test pass returning top-1 accuracy %.

    Caller is responsible for ensuring the fault model is detached if a
    no-fault baseline is desired. Pass ``desc`` to label the tqdm progress bar
    (e.g. ``"rt_error=1e-5 loop 1/2"``); ``None`` shows just batch counts.
    """
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(
        loader,
        desc=desc or "infer",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.numel()
        if _HAS_TQDM:
            pbar.set_postfix(acc=f"{100.0 * correct / total:.2f}%")
    if _HAS_TQDM:
        pbar.close()
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
    desc_prefix: str = "",
) -> list[float]:
    """Run ``loops`` consecutive inference passes; return per-pass accuracies.

    ``index_offset`` accumulates across passes (legacy behaviour: a stuck
    racetrack stays stuck across iterations until reset). The returned list
    has length ``loops``.
    """
    accuracies: list[float] = []
    for i in range(loops):
        loop_label = f"loop {i + 1}/{loops}"
        desc = f"{desc_prefix} {loop_label}".strip() if desc_prefix else loop_label
        log_fn(f"  Inference {i + 1}/{loops}")
        start = time.perf_counter()
        acc = evaluate_clean(model, loader, device, log_fn=None, desc=desc)
        elapsed = time.perf_counter() - start
        log_fn(f"    acc={acc:.2f}%   elapsed={elapsed:.2f}s")
        accuracies.append(acc)
    return accuracies
