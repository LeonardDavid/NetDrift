"""Snapshot capture across layers and before/after deltas."""
from __future__ import annotations

import torch
import torch.nn as nn


def _tiny_quant_model():
    """A 2-layer quantized model with a known weight pattern, on CPU."""
    from netdrift.quant.binary import BinaryScheme
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    conv = QuantizedConv2d(1, 2, kernel_size=3, bias=False)
    lin = QuantizedLinear(8, 4, bias=False)
    for layer, name, lid in ((conv, "conv", 1), (lin, "lin", 2)):
        layer.attach_scheme(BinaryScheme())
        layer.rt_mapping = "ROW"
        layer.kernel_mapping = "ROW"
        layer.layer_name = name
        layer.layer_id = lid
    model = nn.Sequential()
    model.add_module("conv", conv)
    model.add_module("lin", lin)
    return model


def test_capture_snapshot_covers_all_quant_layers():
    from netdrift.metrics.snapshots import capture_snapshot

    model = _tiny_quant_model()
    snap = capture_snapshot(model, label="trained", rt_size=4)
    assert snap.label == "trained"
    assert set(snap.per_layer) == {"conv", "lin"}
    assert snap.totals["block_count"]["total"] > 0


def test_capture_snapshot_honors_col_rt_mapping():
    # Guards the layout bug: a snapshot taken while rt_mapping="COL" must use
    # the COL layout, not fall back to ROW. (The runner's placement fix relies
    # on rt_mapping being correct AT CAPTURE TIME — this asserts the contract.)
    from netdrift.faults.layout import compute_index_offset_shape
    from netdrift.metrics.snapshots import capture_snapshot

    model = _tiny_quant_model()
    lin = dict(model.named_modules())["lin"]  # weight (4, 8)
    lin.rt_mapping = "COL"
    snap = capture_snapshot(model, label="trained", rt_size=4)
    expected = compute_index_offset_shape((4, 8), rt_size=4, rt_mapping="COL")
    assert tuple(snap.per_layer["lin"].n_racetracks) == expected
    # COL shape differs from ROW for this weight, proving no ROW fallback.
    assert expected != compute_index_offset_shape((4, 8), rt_size=4, rt_mapping="ROW")


def test_compute_deltas_counts_sign_flips_and_threshold_movement():
    from netdrift.metrics.snapshots import capture_snapshot, compute_deltas

    model = _tiny_quant_model()
    # Set 'lin' weights to known magnitudes so the threshold-movement is exact.
    lin = dict(model.named_modules())["lin"]
    with torch.no_grad():
        lin.weight.fill_(0.5)  # all +0.5 -> signed dist +0.5
    before = capture_snapshot(model, label="before_encoder", rt_size=4)
    # Flip the sign of every weight in 'lin' -> bitflips == numel; each weight
    # moves from +0.5 to -0.5, a per-weight threshold change of 1.0.
    with torch.no_grad():
        lin.weight.mul_(-1.0)
    after = capture_snapshot(model, label="after_encoder", rt_size=4)

    deltas = compute_deltas(before, after)
    lin_numel = lin.weight.numel()
    assert deltas["per_layer"]["lin"]["bitflips"] == lin_numel
    assert deltas["per_layer"]["conv"]["bitflips"] == 0
    assert deltas["total"]["bitflips"] == lin_numel
    # Per-weight |Δ signed-distance-to-threshold| is 1.0 for every flipped weight.
    assert abs(deltas["per_layer"]["lin"]["abs_dist_to_threshold_change_mean"] - 1.0) < 1e-6
    assert abs(deltas["per_layer"]["conv"]["abs_dist_to_threshold_change_mean"]) < 1e-6


def test_recal_param_deltas():
    import torch.nn as nn
    from netdrift.metrics.snapshots import capture_recal_params, recal_deltas

    model = nn.Module()
    model.bn = nn.BatchNorm2d(3)
    scale_mod = nn.Module()
    scale_mod.scale = nn.Parameter(torch.tensor([2.0]))
    model.scale = scale_mod

    before = capture_recal_params(model)
    with torch.no_grad():
        model.bn.weight.fill_(2.0)         # gamma 1 -> 2, abs delta 1.0 per elem
        model.bn.running_mean.fill_(0.5)   # 0 -> 0.5, abs delta 0.5 (effective boundary)
        model.scale.scale.fill_(5.0)       # 2 -> 5, abs delta 3.0
    after = capture_recal_params(model)

    d = recal_deltas(before, after)
    assert abs(d["bn"]["bn.weight"]["abs_mean_change"] - 1.0) < 1e-6
    assert abs(d["bn"]["bn.running_mean"]["abs_mean_change"] - 0.5) < 1e-6
    assert abs(d["scale"]["scale.scale"]["abs_mean_change"] - 3.0) < 1e-6
