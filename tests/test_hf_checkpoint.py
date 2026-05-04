"""End-to-end verification with a real HuggingFace checkpoint.

These tests load a pre-trained NetDrift checkpoint and assert:

* ``strict=True`` succeeds — the refactored model has the same state-dict keys.
* Clean test accuracy on the configured dataset reproduces the published
  baseline within a small tolerance (default 0.1%).

Auto-skipped when no checkpoint path is configured. Set the environment
variable ``NETDRIFT_HF_CKPT`` (and optionally ``NETDRIFT_HF_DATASET`` —
defaults to ``cifar10``) to run this on demand. CUDA is required for fast
inference; skipped otherwise.
"""

from __future__ import annotations

import os

import pytest
import torch

from netdrift.data import build_datasets
from netdrift.models import build_model, replace_with_quantized
from netdrift.models.checkpoint import load_checkpoint
from netdrift.quant import BinaryScheme
from netdrift.training import evaluate_clean


CKPT_ENV = "NETDRIFT_HF_CKPT"
MODEL_ENV = "NETDRIFT_HF_MODEL"
DATASET_ENV = "NETDRIFT_HF_DATASET"
EXPECTED_ACC_ENV = "NETDRIFT_HF_ACC"


def _ckpt_path() -> str:
    path = os.environ.get(CKPT_ENV)
    if not path:
        pytest.skip(f"set {CKPT_ENV}=<path> to enable this test")
    if not os.path.exists(path):
        pytest.skip(f"checkpoint not found at {path}")
    return path


@pytest.mark.cuda
def test_hf_checkpoint_strict_load() -> None:
    """A real HF checkpoint loads strictly into the refactored model."""
    path = _ckpt_path()
    model_name = os.environ.get(MODEL_ENV, "vgg7_cifar10")

    model = build_model(model_name)
    replace_with_quantized(model, BinaryScheme())
    report = load_checkpoint(model, path, mode="strict")
    assert not report.missing_keys, f"missing keys: {report.missing_keys}"
    assert not report.unexpected_keys, f"unexpected keys: {report.unexpected_keys}"


@pytest.mark.cuda
def test_hf_checkpoint_baseline_accuracy() -> None:
    """Clean inference accuracy reproduces the published baseline within tolerance."""
    path = _ckpt_path()
    model_name = os.environ.get(MODEL_ENV, "vgg7_cifar10")
    dataset_name = os.environ.get(DATASET_ENV, "cifar10")
    expected = float(os.environ.get(EXPECTED_ACC_ENV, "0"))

    if expected <= 0:
        pytest.skip(
            f"set {EXPECTED_ACC_ENV}=<published_pct> to enable accuracy check"
        )

    device = torch.device("cuda")
    model = build_model(model_name)
    replace_with_quantized(model, BinaryScheme())
    load_checkpoint(model, path, mode="strict", map_location=str(device))
    model.to(device)

    _, test_ds, _ = build_datasets(dataset_name)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=1, pin_memory=True
    )
    acc = evaluate_clean(model, loader, device, log_fn=None)

    tolerance = 0.1  # %
    assert abs(acc - expected) <= tolerance, (
        f"accuracy diverged: got {acc:.2f}%, expected {expected:.2f}% (±{tolerance}%)"
    )
