"""Diagnostic tests for the endlen ``mode=once`` pipeline.

These tests check the parts of the once-mode flow that the existing
``test_weight_encoders.py`` and ``test_rtm_fault.py`` do not exercise:

1. The layout helper :func:`_layout_weight_for_racetrack` round-trips a
   conv weight bit-for-bit (no encoder involved). If this is broken the
   encoder operates on misaligned data.
2. :func:`apply_weight_encoder_to_model` actually mutates layer weights
   with a non-trivial bit count for a realistic VGG-style conv layer
   filled with high-entropy ±1 data.
3. After ``apply_weight_encoder_to_model``, the model's forward output
   differs from the pre-encoded output (i.e. accuracy degradation is
   physically possible — the model isn't quantizing away the changes).
4. (GPU) End-to-end: the saved post-encode state_dict reloads cleanly,
   produces the same forward output as the in-memory encoded model, and
   that output differs from the original-checkpoint output.

Tests 1–3 are CPU-safe. Test 4 requires CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from netdrift.faults.rtm_misalignment import _layout_weight_for_racetrack
from netdrift.faults.weight_encoders import (
    EndlenEncoder,
    apply_weight_encoder_to_model,
)
from netdrift.models import build_model, replace_with_quantized
from netdrift.quant import BinaryScheme


# ---------------------------------------------------------------------------
# 1. Layout helper round-trips
# ---------------------------------------------------------------------------


def test_layout_round_trip_linear_row() -> None:
    w = torch.randn(8, 64)
    w_2d, undo = _layout_weight_for_racetrack(w, rt_mapping="ROW", kernel_mapping=None)
    assert w_2d.shape == (8, 64)
    assert torch.equal(undo(w_2d), w)


def test_layout_round_trip_linear_col() -> None:
    w = torch.randn(8, 64)
    w_2d, undo = _layout_weight_for_racetrack(w, rt_mapping="COL", kernel_mapping=None)
    assert w_2d.shape == (64, 8)  # transposed
    assert torch.equal(undo(w_2d), w)


def test_layout_round_trip_conv_row_mapping() -> None:
    w = torch.randn(16, 4, 3, 3)
    w_2d, undo = _layout_weight_for_racetrack(
        w, rt_mapping="ROW", kernel_mapping="ROW"
    )
    assert w_2d.shape == (16, 36)
    assert torch.equal(undo(w_2d), w)


@pytest.mark.parametrize("kmap", ["ROW", "COL", "CLW", "ACW"])
def test_layout_round_trip_conv_all_kernel_mappings(kmap: str) -> None:
    w = torch.randn(8, 4, 3, 3)
    w_2d, undo = _layout_weight_for_racetrack(
        w, rt_mapping="ROW", kernel_mapping=kmap
    )
    out = undo(w_2d)
    assert out.shape == w.shape
    assert torch.equal(out, w), f"kernel mapping {kmap} did not round-trip"


# ---------------------------------------------------------------------------
# 2. apply_weight_encoder_to_model actually mutates weights
# ---------------------------------------------------------------------------


def _populate_with_high_entropy_pm1(model: torch.nn.Module, seed: int = 0) -> None:
    """Overwrite every quantized layer's weight with random ±1 noise.

    The point is to make sure each racetrack contains many sign changes,
    so endlen has plenty of merge candidates. Without this, weights from a
    freshly-built model are FP32 noise — re-quantization to ±1 would still
    leave the per-racetrack bit patterns predictable but at least non-trivial.
    """
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    g = torch.Generator().manual_seed(seed)
    for _, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            bits = torch.randint(
                0, 2, module.weight.shape, generator=g, dtype=torch.float32
            )
            module.weight.data.copy_(bits * 2 - 1)  # {0,1} → {-1,+1}


@pytest.mark.cuda
def test_apply_mutates_layer_weights() -> None:
    """apply_weight_encoder_to_model must flip a substantial number of bits
    on high-entropy ±1 weights — far more than zero.
    """
    device = torch.device("cuda")
    model = build_model("vgg3_mnist").to(device)
    replace_with_quantized(model, BinaryScheme())
    _populate_with_high_entropy_pm1(model, seed=42)
    model.to(device)

    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
    pre = {
        n: m.weight.detach().clone()
        for n, m in model.named_modules()
        if isinstance(m, (QuantizedConv2d, QuantizedLinear))
    }

    report = apply_weight_encoder_to_model(
        model, EndlenEncoder(),
        rt_size=8,  # small rt_size so each layer's flat-dim covers many racetracks
        rt_mapping="ROW",
        kernel_mapping_default="ROW",
    )

    assert report, "apply_weight_encoder_to_model returned an empty report"
    total = sum(report.values())
    assert total > 0, (
        f"encoder did not flip any bits across {len(report)} layers — "
        f"report={report}"
    )

    # Per-layer sanity: every layer with weight numel >= 8 should have at
    # least one flip (high-entropy random ±1 with rt_size=8 reliably has
    # at least one merge candidate).
    for name, changed in report.items():
        if pre[name].numel() < 8:
            continue
        assert changed > 0, f"layer {name} had zero bit flips ({changed=})"


@pytest.mark.cuda
def test_apply_respects_protection_policy() -> None:
    """Protected layers must be skipped by apply_weight_encoder_to_model.

    Sets ``protected=True`` on a subset of layers, runs the encoder, asserts
    that those layers are absent from the report and their weights unchanged.
    """
    from netdrift.models import apply_protection_policy
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    device = torch.device("cuda")
    model = build_model("vgg3_mnist").to(device)
    replace_with_quantized(model, BinaryScheme())
    _populate_with_high_entropy_pm1(model, seed=42)
    model.to(device)

    # vgg3_mnist has layer_ids conv1=1, conv2=2, fc1=3, fc2=4.
    # custom policy: leave only conv2 and fc1 unprotected.
    apply_protection_policy(model, "custom", layers=[2, 3])

    pre = {
        n: m.weight.detach().clone()
        for n, m in model.named_modules()
        if isinstance(m, (QuantizedConv2d, QuantizedLinear))
    }

    report = apply_weight_encoder_to_model(
        model, EndlenEncoder(),
        rt_size=8, rt_mapping="ROW", kernel_mapping_default="ROW",
    )

    # Only the two unprotected layers should appear in the report.
    assert set(report.keys()) == {"conv2", "fc1"}, (
        f"report should only contain unprotected layers, got {sorted(report)}"
    )

    # Protected layers' weights must be untouched.
    for name in ("conv1", "fc2"):
        post = next(
            m for n, m in model.named_modules() if n == name
        ).weight.detach()
        assert torch.equal(pre[name], post), (
            f"protected layer {name} was modified by the encoder"
        )

    # Unprotected layers should have nonzero flips on high-entropy ±1 input.
    for name in report:
        assert report[name] > 0, (
            f"unprotected layer {name} had zero flips ({report[name]})"
        )


@pytest.mark.cuda
def test_apply_changes_model_outputs() -> None:
    """After encoding, the model's output on a fixed input must differ from
    the pre-encode output. If they're identical, the encoder's writes are
    being overwritten somewhere in the forward path (or the layout is
    misaligned and the encoder is a no-op on every racetrack).
    """
    device = torch.device("cuda")
    torch.manual_seed(123)
    model = build_model("vgg3_mnist").to(device)
    replace_with_quantized(model, BinaryScheme())
    _populate_with_high_entropy_pm1(model, seed=42)
    model.to(device).eval()

    x = torch.randn(2, 1, 28, 28, device=device)
    with torch.no_grad():
        y_pre = model(x).detach().clone()

    apply_weight_encoder_to_model(
        model, EndlenEncoder(),
        rt_size=8,
        rt_mapping="ROW",
        kernel_mapping_default="ROW",
    )

    with torch.no_grad():
        y_post = model(x).detach().clone()

    # We are NOT asserting accuracy degradation here — that depends on the
    # task. We are asserting that the forward computation is sensitive to
    # the encoder's writes at all. If y_pre == y_post then the encoder is
    # being silently discarded.
    assert not torch.equal(y_pre, y_post), (
        "model output is identical pre- and post-encoding — the encoder's "
        "writes to module.weight.data are not flowing through the forward "
        "pass (likely re-quantized away or wrong layout)."
    )


def test_apply_on_already_pm1_weights_is_meaningful() -> None:
    """Even without the cuda stub, the CPU reference applied directly to a
    layer's 2D-flattened weight changes a substantial bit count.

    Decouples the algorithm correctness from the apply_to_model glue.
    """
    from netdrift.faults.weight_encoders.endlen import _endlen_cpu_reference

    rng = np.random.default_rng(0)
    # Simulate a conv1 of shape (out=16, in=3, 3, 3) flattened to (16, 27).
    # rt_size=8 → ceil(27/8) = 4 racetracks per row, last one short.
    weight = np.where(rng.random((16, 27)) > 0.5, 1, -1).astype(np.int32)
    pre = weight.copy()

    _endlen_cpu_reference(weight, 8)

    changed = int((pre != weight).sum())
    assert changed > 0, "CPU reference did not flip any bits on random ±1 input"
    # On 16×27 random ±1 with rt_size=8, we expect tens of flips. Lower
    # bound is loose; the real signal is just "not zero".
    assert changed >= 4, f"only {changed} bits flipped on random ±1 input"


# ---------------------------------------------------------------------------
# 4. End-to-end GPU test: save/reload preserves encoded weights and outputs
# ---------------------------------------------------------------------------


@pytest.mark.cuda
def test_once_mode_on_real_checkpoint(tmp_path) -> None:
    """Mirror the runner's mode=once path on a real checkpoint, if available.

    Skipped unless ``NETDRIFT_REAL_CKPT`` (path to a BNN ``.pt``) and
    ``NETDRIFT_REAL_MODEL`` (registry name, e.g. ``vgg7_cifar10``) are set.
    When run, this asserts the same property as ``test_apply_changes_model_outputs``
    but on the actual production weights — catching the "trained BNN happens
    to be near-fixed-point for endlen" case the synthetic test cannot.

    Failure modes this catches:

    * Total bit-flip count across the whole model is suspiciously small
      (< 0.1 %% of total quantized weight bits) — endlen had nothing to do.
    * Forward output on a deterministic random input is unchanged after
      encoding, despite a meaningful flip count — the runner's apply args
      don't match what the forward pass expects.
    """
    import os
    ckpt_path = os.environ.get("NETDRIFT_REAL_CKPT")
    model_name = os.environ.get("NETDRIFT_REAL_MODEL")
    if not (ckpt_path and model_name):
        pytest.skip(
            "set NETDRIFT_REAL_CKPT=<path> NETDRIFT_REAL_MODEL=<name> to enable"
        )
    if not os.path.exists(ckpt_path):
        pytest.skip(f"checkpoint not found: {ckpt_path}")

    rt_size = int(os.environ.get("NETDRIFT_REAL_RT_SIZE", "64"))
    rt_mapping = os.environ.get("NETDRIFT_REAL_RT_MAPPING", "ROW").upper()
    kernel_mapping = os.environ.get("NETDRIFT_REAL_KERNEL_MAPPING", "ROW").upper()

    from netdrift.models.checkpoint import load_checkpoint
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    device = torch.device("cuda")
    model = build_model(model_name)
    replace_with_quantized(model, BinaryScheme())
    load_checkpoint(model, ckpt_path, mode="strict", map_location=str(device))
    model.to(device).eval()

    # Deterministic random input matching the model's expected shape. We
    # introspect the first conv to figure out channels/spatial.
    first_conv = next(
        m for _, m in model.named_modules() if isinstance(m, QuantizedConv2d)
    )
    in_ch = first_conv.in_channels
    spatial = 32 if "cifar" in model_name else 28
    torch.manual_seed(2026)
    x = torch.randn(2, in_ch, spatial, spatial, device=device)
    with torch.no_grad():
        y_pre = model(x).detach().clone()

    # Snapshot pre-encode for the bit-flip count.
    pre = {
        n: m.weight.detach().clone()
        for n, m in model.named_modules()
        if isinstance(m, (QuantizedConv2d, QuantizedLinear))
    }
    total_bits = sum(t.numel() for t in pre.values())

    report = apply_weight_encoder_to_model(
        model, EndlenEncoder(),
        rt_size=rt_size,
        rt_mapping=rt_mapping,
        kernel_mapping_default=kernel_mapping,
    )
    flipped = sum(report.values())

    # Diagnostic floor: real BNN weights are structured, but on a model with
    # millions of quantized weights endlen almost always finds *something* —
    # we expect at least 1 flip per 10000 weights. If we're below 1 flip per
    # 100000 (0.001 %%), endlen is effectively a no-op on this checkpoint and
    # the user should know.
    floor = max(1, total_bits // 100000)
    assert flipped >= floor, (
        f"endlen flipped only {flipped} of {total_bits} weight bits "
        f"({100*flipped/total_bits:.5f}%%) — below diagnostic floor of {floor}. "
        f"Per-layer breakdown: {report}"
    )

    with torch.no_grad():
        y_post = model(x).detach().clone()
    assert not torch.equal(y_pre, y_post), (
        f"flipped {flipped} bits but forward output is unchanged — apply args "
        f"likely don't match the forward path (rt_mapping={rt_mapping}, "
        f"kernel_mapping={kernel_mapping})."
    )


@pytest.mark.cuda
def test_once_mode_save_reload_round_trip(tmp_path) -> None:
    """The state_dict saved after mode=once reloads strictly and yields the
    same forward output as the in-memory encoded model. The output must
    also differ from the *original* (pre-encode) checkpoint's output.

    Catches: save format bugs; double-encoding bugs; layout drift between
    the encoder pass and the model forward.
    """
    device = torch.device("cuda")
    torch.manual_seed(7)

    model = build_model("vgg3_mnist").to(device)
    replace_with_quantized(model, BinaryScheme())
    _populate_with_high_entropy_pm1(model, seed=11)
    model.to(device).eval()

    x = torch.randn(2, 1, 28, 28, device=device)
    with torch.no_grad():
        y_pre = model(x).detach().clone()

    # Save pre-encode for the differs-from-original check.
    pre_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), pre_path)

    apply_weight_encoder_to_model(
        model, EndlenEncoder(),
        rt_size=64,
        rt_mapping="ROW",
        kernel_mapping_default="ROW",
    )
    with torch.no_grad():
        y_post = model(x).detach().clone()
    assert not torch.equal(y_pre, y_post), "encoder did not affect forward output"

    enc_path = tmp_path / "model_endlen.pt"
    torch.save(model.state_dict(), enc_path)

    # Build a fresh model, load the encoded checkpoint strictly, compare.
    fresh = build_model("vgg3_mnist").to(device)
    replace_with_quantized(fresh, BinaryScheme())
    fresh.load_state_dict(torch.load(enc_path, map_location=device), strict=True)
    fresh.to(device).eval()
    with torch.no_grad():
        y_reload = fresh(x).detach().clone()

    assert torch.equal(y_post, y_reload), (
        "reloaded encoded model produces a different output than the "
        "in-memory encoded model — save/load is not preserving weights"
    )
