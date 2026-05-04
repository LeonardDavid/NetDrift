"""CheckpointAdapter: strict / fp32_warmstart / scheme_transfer modes.

Covers:

* Strict load: BNN→BNN exact-key match.
* FP32 warm-start: ``*.bias`` keys dropped when target has ``bias=False``.
* FP32 warm-start: torchvision ResNet ``fc.*`` → NetDrift ``linear.*`` rename.
* Round-trip: warm-start → save → strict reload succeeds.

CPU-safe — uses tiny synthetic checkpoints, no GPU work.
"""

from __future__ import annotations

from pathlib import Path
from typing import OrderedDict

import torch
import torch.nn as nn

from netdrift.models import build_model, replace_with_quantized
from netdrift.models.checkpoint import load_checkpoint
from netdrift.quant import BinaryScheme


def _save(state_dict: dict, path: Path) -> None:
    torch.save(state_dict, path)


def test_strict_load_round_trip(tmp_path: Path) -> None:
    """Save a quantized VGG3's state_dict and reload it strictly."""
    qmodel = build_model("vgg3_mnist")
    replace_with_quantized(qmodel, BinaryScheme())

    ckpt_path = tmp_path / "model.pt"
    _save(qmodel.state_dict(), ckpt_path)

    fresh = build_model("vgg3_mnist")
    replace_with_quantized(fresh, BinaryScheme())
    report = load_checkpoint(fresh, str(ckpt_path), mode="strict")
    assert not report.missing_keys
    assert not report.unexpected_keys

    # Weights match
    assert torch.equal(fresh.conv1.weight, qmodel.conv1.weight)
    assert torch.equal(fresh.fc1.weight, qmodel.fc1.weight)


def test_fp32_warmstart_drops_unexpected_biases(tmp_path: Path) -> None:
    """An FP32 checkpoint with conv.bias keys loads cleanly into a bias=False target."""
    # Target: torchvision-style ResNet18 with no biases on conv layers and
    # ``fc`` as the classifier head (no rename needed).
    target = build_model("resnet18_cifar10")
    replace_with_quantized(target, BinaryScheme())
    target_sd = target.state_dict()

    # Build an FP32-like source by copying the target state_dict and adding a
    # bogus bias key under conv1 (the target's conv1 has no bias).
    src_sd: dict[str, torch.Tensor] = {k: v.clone() for k, v in target_sd.items()}
    src_sd["conv1.bias"] = torch.zeros(target_sd["conv1.weight"].shape[0])

    ckpt_path = tmp_path / "fp32.pt"
    _save(src_sd, ckpt_path)

    fresh = build_model("resnet18_cifar10")
    replace_with_quantized(fresh, BinaryScheme())
    report = load_checkpoint(
        fresh, str(ckpt_path),
        mode="fp32_warmstart",
        scheme=BinaryScheme(),
    )

    # The bogus bias should have been dropped.
    assert "conv1.bias" in report.dropped_keys
    # No renames should fire (torchvision target has ``fc``, not ``linear``).
    assert report.renamed_keys == []
    # And the conv weight should have been transferred.
    assert torch.equal(fresh.conv1.weight, target_sd["conv1.weight"])


def test_fp32_warmstart_renames_fc_to_linear_for_legacy_target(tmp_path: Path) -> None:
    """When the target uses the legacy ``linear`` name, ``fc.*`` keys are remapped."""
    # Synthesize a tiny target model whose classifier is named ``linear``
    # (mimicking the legacy NetDrift custom ResNet). Just enough to exercise
    # the rename rule.
    class _LegacyResnetLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
            self.linear = nn.Linear(4, 10, bias=False)

    target = _LegacyResnetLike()
    src_sd = {
        "conv1.weight": torch.randn(4, 3, 3, 3),
        "fc.weight": torch.randn(10, 4),
    }
    ckpt_path = tmp_path / "src.pt"
    _save(src_sd, ckpt_path)

    report = load_checkpoint(
        target, str(ckpt_path),
        mode="fp32_warmstart",
        scheme=BinaryScheme(),
    )
    # The rename should be in the report
    rename_targets = {dst for _, dst in report.renamed_keys}
    assert "linear.weight" in rename_targets
    # And the weight should have been transferred under the new name.
    assert torch.equal(target.linear.weight, src_sd["fc.weight"])


def test_fp32_warmstart_round_trip(tmp_path: Path) -> None:
    """FP32 → warm-start → save → strict reload should succeed."""
    # 1) Build an FP32 source state_dict (vanilla VGG3, no quantization)
    fp32 = build_model("vgg3_mnist")
    src_path = tmp_path / "fp32.pt"
    _save(fp32.state_dict(), src_path)

    # 2) Warm-start into a quantized VGG3
    target = build_model("vgg3_mnist")
    replace_with_quantized(target, BinaryScheme())
    load_checkpoint(target, str(src_path), mode="fp32_warmstart", scheme=BinaryScheme())

    # 3) Save the quantized checkpoint
    qpath = tmp_path / "quantized.pt"
    _save(target.state_dict(), qpath)

    # 4) Reload strictly into a new quantized VGG3
    fresh = build_model("vgg3_mnist")
    replace_with_quantized(fresh, BinaryScheme())
    report = load_checkpoint(fresh, str(qpath), mode="strict")
    assert not report.missing_keys
    assert not report.unexpected_keys


def test_legacy_training_state_checkpoint_is_unwrapped(tmp_path: Path) -> None:
    """A checkpoint with ``model_state_dict`` wrapper is auto-unwrapped (legacy format)."""
    qmodel = build_model("vgg3_mnist")
    replace_with_quantized(qmodel, BinaryScheme())
    wrapped = {"epoch": 5, "model_state_dict": qmodel.state_dict()}
    path = tmp_path / "wrapped.pt"
    _save(wrapped, path)

    fresh = build_model("vgg3_mnist")
    replace_with_quantized(fresh, BinaryScheme())
    report = load_checkpoint(fresh, str(path), mode="strict")
    assert not report.missing_keys
    assert not report.unexpected_keys


def test_adapter_report_summary_is_formatted() -> None:
    """The summary string mentions every category that has entries."""
    from netdrift.models.checkpoint import AdapterReport
    rep = AdapterReport(
        mode="fp32_warmstart",
        dropped_keys=["a.bias"],
        renamed_keys=[("fc.weight", "linear.weight")],
        initialized_keys=[],
        missing_keys=[],
        unexpected_keys=[],
    )
    s = rep.summary()
    assert "fp32_warmstart" in s
    assert "dropped" in s
    assert "renamed" in s
