"""Wire-purity reaches the artifacts: static metrics, meta block, model totals.

Companion to ``test_wire_purity.py`` (the metric itself). Nothing here needs a
GPU: ``_metrics_meta`` / ``_total_wire_purity`` are called directly on a bare
``QuantizedLinear``, the same trick
``test_run_units_wiring.py::test_metrics_meta_units_branch_reports_packed_wire_count``
uses. The full end-to-end JSON is covered by ``test_metrics_integration.py``.
"""

from __future__ import annotations

import torch

from netdrift.faults.purity import wire_purity

RT = 4

TOY = torch.tensor([[+1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1],
                    [-1, +1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1],
                    [+1, +1, +1, +1, +1, +1, -1, -1, -1, +1, -1, -1]],
                   dtype=torch.float32)


def _cfg(path, *overrides):
    from netdrift.config.loader import load
    return load(path, overrides=[f"storage.rt_size={RT}", *overrides])


def _toy_layer(rt_mapping, base_layout="ROW"):
    """A bare QuantizedLinear holding TOY, tagged the way the runner tags layers."""
    from netdrift.quant.layers import QuantizedLinear

    layer = QuantizedLinear(in_features=TOY.shape[1], out_features=TOY.shape[0],
                            bias=False)
    with torch.no_grad():
        layer.weight.copy_(TOY)
    layer.rt_mapping = rt_mapping
    layer.base_layout = base_layout
    return layer


# --------------------------------------------------------------------------
# metrics/static.py — per-layer field
# --------------------------------------------------------------------------

def test_static_metrics_carry_wire_purity_for_a_dense_layer():
    """``compute_static_metrics`` must expose the purity of the layer it measured.

    Breaks if the field is dropped from ``StaticLayerMetrics`` or computed on a
    different view than the block/run metrics beside it.
    """
    from netdrift.metrics.static import compute_static_metrics

    m = compute_static_metrics(TOY, rt_mapping="ROW", kernel_mapping=None,
                               rt_size=RT)

    assert m.wire_purity == wire_purity(TOY, RT, "ROW")
    assert m.wire_purity.total == m.n_racetracks[0] * m.n_racetracks[1]


def test_static_metrics_wire_purity_for_polarity_is_all_pure():
    from netdrift.metrics.static import compute_static_metrics

    m = compute_static_metrics(TOY, rt_mapping="POLARITY", kernel_mapping=None,
                               rt_size=RT, base_layout="ROW",
                               polarity_params=(0, True))

    assert m.wire_purity.mixed == 0
    assert m.wire_purity.total == m.n_racetracks[0]


def test_static_metrics_wire_purity_for_unpadded_polarity_finds_mixed_wires():
    """The ablation arm must be visibly non-immune in the reported metric."""
    from netdrift.metrics.static import compute_static_metrics

    m = compute_static_metrics(TOY, rt_mapping="POLARITY", kernel_mapping=None,
                               rt_size=RT, base_layout="ROW",
                               polarity_params=(0, False))

    assert m.wire_purity.mixed == 3
    assert m.wire_purity.total == m.n_racetracks[0]


# --------------------------------------------------------------------------
# metrics/snapshots.py — model totals
# --------------------------------------------------------------------------

def test_snapshot_totals_sum_wire_purity_over_layers():
    """Snapshot totals are per-layer sums, so purity must aggregate too.

    Breaks if the totals block forgets the new key, which would leave the
    encoder/recal purity delta unobservable.
    """
    import torch.nn as nn

    from netdrift.metrics.snapshots import capture_snapshot
    from netdrift.quant.binary import BinaryScheme

    model = nn.Sequential()
    for i in range(2):
        layer = _toy_layer("ROW")
        layer.attach_scheme(BinaryScheme())
        layer.layer_name = f"lin{i}"
        layer.layer_id = i + 1
        model.add_module(f"lin{i}", layer)

    snap = capture_snapshot(model, "trained", rt_size=RT)

    one = wire_purity(TOY, RT, "ROW")
    assert snap.totals["wire_purity"] == (one + one).as_dict()
    assert snap.totals["wire_purity"]["mixed"] == 2 * one.mixed


# --------------------------------------------------------------------------
# runner/run.py — meta block, storage block, model total
# --------------------------------------------------------------------------

def test_metrics_meta_reports_wire_purity_per_layer():
    from netdrift.runner.run import _metrics_meta

    cfg = _cfg("configs/modes/vgg7_cifar10_w1a1_baseline_col.yaml")
    layer = _toy_layer("ROW")

    meta = _metrics_meta(cfg, layer, category=None, subcategory=None)

    lm = meta["layers"][0]
    assert lm["wire_purity"] == wire_purity(TOY, RT, "ROW").as_dict()
    # The two cost/robustness numbers are reported side by side and must agree.
    assert lm["wire_purity"]["total"] == lm["n_racetracks"][0] * lm["n_racetracks"][1]


def test_metrics_meta_polarity_layer_is_all_pure_and_matches_n_racetracks():
    from netdrift.runner.run import _metrics_meta

    cfg = _cfg("configs/modes/vgg7_cifar10_w1a1_polarity.yaml",
               "storage.base_layout=row")
    layer = _toy_layer("POLARITY", base_layout="ROW")

    meta = _metrics_meta(cfg, layer, category=None, subcategory=None)

    lm = meta["layers"][0]
    assert lm["wire_purity"]["mixed"] == 0
    assert lm["wire_purity"]["mixed_frac"] == 0.0
    assert lm["wire_purity"]["total"] == lm["n_racetracks"][0]


def test_metrics_meta_storage_block_records_base_layout_and_partition():
    """The window/base_layout gap: an aggregator joining on meta.storage must
    be able to tell the PPM window arms apart.

    Breaks if the storage block goes back to rt_size/layout/kernel_mapping only.
    """
    from netdrift.runner.run import _metrics_meta

    cfg = _cfg("configs/modes/vgg7_cifar10_w1a1_polarity.yaml",
               "storage.partition.window=2")
    layer = _toy_layer("POLARITY", base_layout="COL")

    storage = _metrics_meta(cfg, layer, category=None, subcategory=None)["storage"]

    assert storage["base_layout"] == "col"
    assert storage["partition"] == {"window": 2, "pad": True}


def test_metrics_meta_storage_omits_partition_for_non_polarity_layouts():
    """``storage.partition`` is schema-ignored outside layout=polarity; logging
    its defaults on a dense run would imply PPM is in play — the same
    omit-when-irrelevant rule the ``units_*`` W&B keys follow.
    """
    from netdrift.runner.run import _metrics_meta

    cfg = _cfg("configs/modes/vgg7_cifar10_w1a1_baseline_col.yaml")
    layer = _toy_layer("ROW")

    storage = _metrics_meta(cfg, layer, category=None, subcategory=None)["storage"]

    assert "partition" not in storage
    # base_layout is still logged (it is what block/units/polarity segment on);
    # for a dense col run the schema default "row" is the resolved value.
    assert storage["base_layout"] == cfg.storage.base_layout == "row"


def test_total_wire_purity_sums_over_quantized_layers():
    """The model-level number that lands in summary.json and the W&B config."""
    import torch.nn as nn

    from netdrift.runner.run import _total_wire_purity

    # layout=row, so the laid-out view is TOY itself (a col layout would
    # transpose it and the hand-computed expectation would not apply).
    cfg = _cfg("configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml")
    model = nn.Sequential()
    for i in range(3):
        model.add_module(f"lin{i}", _toy_layer("ROW"))

    total = _total_wire_purity(cfg, model)

    one = wire_purity(TOY, RT, "ROW")
    assert total == one + one + one


def test_total_wire_purity_is_protection_invariant():
    """Protection is a fault-injection policy, not a layout change — purity of
    the storage must not depend on it (same contract as ``_total_racetracks``).
    """
    import torch.nn as nn

    from netdrift.runner.run import _total_wire_purity

    model = nn.Sequential()
    model.add_module("lin0", _toy_layer("ROW"))
    cfg_path = "configs/modes/vgg7_cifar10_w1a1_baseline_col.yaml"
    a = _total_wire_purity(_cfg(cfg_path), model)
    b = _total_wire_purity(_cfg(cfg_path, "fault.protection.policy=all"), model)

    assert a == b


def test_total_wire_purity_total_equals_total_racetracks_for_conv_layers():
    """The ``purity.total == n_racetracks`` invariant, on the path where the two
    are computed differently.

    ``_total_racetracks`` goes through ``compute_index_offset_shape`` with an
    explicit ``kernel_size``; ``_total_wire_purity`` lays the conv weight out via
    ``_layout_weight_for_racetrack`` (which applies the kernel permutation) and
    segments the result. Those are two independent derivations of the same wire
    count, and only a 4D weight exercises the difference — every other test here
    uses a linear layer, where both reduce to the same arithmetic.
    """
    import torch.nn as nn

    from netdrift.quant.layers import QuantizedConv2d
    from netdrift.runner.run import _total_racetracks, _total_wire_purity

    for path in ("configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml",
                 "configs/modes/vgg7_cifar10_w1a1_baseline_col.yaml"):
        cfg = _cfg(path)
        conv = QuantizedConv2d(2, 4, kernel_size=3, bias=False)
        model = nn.Sequential()
        model.add_module("conv", conv)

        assert _total_wire_purity(cfg, model).total == _total_racetracks(cfg, model), path


def test_total_wire_purity_handles_a_non_row_kernel_mapping():
    """A kernel permutation reorders cells along the wire, so it can change which
    wires are pure — but never how many wires there are.
    """
    import torch.nn as nn

    from netdrift.quant.layers import QuantizedConv2d
    from netdrift.runner.run import _total_racetracks, _total_wire_purity

    cfg = _cfg("configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml",
               "storage.kernel_mapping=clw")
    model = nn.Sequential()
    model.add_module("conv", QuantizedConv2d(2, 4, kernel_size=3, bias=False))

    total = _total_wire_purity(cfg, model)

    assert total.total == _total_racetracks(cfg, model)
    assert total.total_weights == 4 * 2 * 3 * 3
