from types import SimpleNamespace

import pytest

from netdrift.runner.run import _rt_mapping_fn_for_layout


def test_units_layout_maps_to_units():
    assert _rt_mapping_fn_for_layout("units")(None) == "UNITS"


def test_existing_layouts_unchanged():
    assert _rt_mapping_fn_for_layout("row")(None) == "ROW"
    assert _rt_mapping_fn_for_layout("col")(None) == "COL"
    assert _rt_mapping_fn_for_layout("block")(None) == "BLOCK"


def test_unknown_layout_still_raises():
    with pytest.raises(NotImplementedError):
        _rt_mapping_fn_for_layout("interleaved")


def test_units_configs_parse():
    from netdrift.config.loader import load
    for name, thresh, mp, pg in [
        ("configs/modes/vgg7_cifar10_w1a1_units_t2.yaml", 2, 2, 1),
        ("configs/modes/vgg7_cifar10_w1a1_units_t4.yaml", 4, 1, 0),
    ]:
        cfg = load(name)
        assert cfg.storage.layout == "units"
        assert cfg.storage.units.threshold == thresh
        assert cfg.storage.units.max_period == mp
        assert cfg.storage.units.pool_guard == pg
        assert cfg.fault.weight_encoder is None


def test_nested_units_override_reaches_the_dataclass():
    # project_comparison_database records a nested override-path gotcha; the
    # sweep driver depends on this working, so pin it here.
    from netdrift.config.loader import load
    cfg = load("configs/modes/vgg7_cifar10_w1a1_units_t4.yaml",
               overrides=["storage.units.threshold=8"])
    assert cfg.storage.units.threshold == 8


def _units_cfg(threshold=4, max_period=1, pool_guard=0,
               weight_encoder=None, fault_aware="none"):
    """Stub matching tests/test_config_block.py's _cfg, plus storage.units."""
    from netdrift.config.schema import UnitsCfg
    return SimpleNamespace(
        storage=SimpleNamespace(
            layout="units",
            units=UnitsCfg(threshold=threshold, max_period=max_period,
                           pool_guard=pool_guard),
        ),
        fault=SimpleNamespace(weight_encoder=weight_encoder),
        training=SimpleNamespace(fault_aware=fault_aware),
    )


def test_units_rejects_weight_encoder():
    from netdrift.runner.run import _validate_block_layout_combo
    with pytest.raises(ValueError, match="weight encoder"):
        _validate_block_layout_combo(_units_cfg(weight_encoder="endlen"))


def test_units_rejects_fault_aware_training():
    from netdrift.runner.run import _validate_block_layout_combo
    with pytest.raises(ValueError):
        _validate_block_layout_combo(_units_cfg(fault_aware="regularization"))


def test_units_accepts_clean_config():
    from netdrift.runner.run import _validate_block_layout_combo
    _validate_block_layout_combo(_units_cfg())  # must not raise


def test_metrics_meta_units_branch_reports_packed_wire_count():
    """``_metrics_meta``'s UNITS branch must report the actual packed-wire
    count, not just avoid raising ``ValueError: invalid rt_mapping: UNITS``
    (which is what happens if a layer falls through to the ROW/COL
    rectangular-shape branch instead of the dedicated UNITS one).

    ``netdrift/metrics/`` (specifically ``compute_static_metrics`` in
    ``netdrift/metrics/static.py``) now has a dedicated UNITS branch too, and
    ``scripts/sweep_units.py`` passes ``--metrics offline`` by default, but
    that end-to-end path needs a real GPU run to exercise (this file has no
    CUDA marker). This test covers ``_metrics_meta`` directly instead, without
    a full runner invocation. It builds a real
    ``QuantizedLinear`` (no fault model attached, no full model needed),
    tags it ``rt_mapping="UNITS"`` / ``base_layout="ROW"`` the way the runner
    does after ``attach_fault_model``, and loads the real units_t4 overlay so
    ``_wandb_config_with_category`` gets a genuine, fully-populated config.
    """
    import torch

    from netdrift.config.loader import load
    from netdrift.faults.layout import _layout_weight_for_racetrack, compute_index_offset_shape
    from netdrift.faults.packing import build_unit_buckets
    from netdrift.quant.layers import QuantizedLinear
    from netdrift.runner.run import _metrics_meta

    cfg = load("configs/modes/vgg7_cifar10_w1a1_units_t4.yaml")
    rt_size = cfg.storage.rt_size  # 64
    threshold = cfg.storage.units.threshold  # 4
    max_period = cfg.storage.units.max_period  # 1
    pool_guard = cfg.storage.units.pool_guard  # 0

    # Two identical rows: a length-5 run (>= threshold=4, isolated onto its
    # own wire) followed by 59 alternating length-1 runs (all below
    # threshold, pooled). This guarantees more physical wires than the single
    # dense ROW racetrack per row this shape would otherwise use, so the
    # ">dense" bound below is a real check, not a tautology.
    tail = [(-1) ** (i + 1) for i in range(rt_size - 5)]
    row = [1] * 5 + tail
    weight = torch.tensor([row, row], dtype=torch.float32)
    assert weight.shape == (2, rt_size)

    layer = QuantizedLinear(in_features=rt_size, out_features=2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.rt_mapping = "UNITS"
    layer.base_layout = "ROW"

    # A bare QuantizedLinear is itself a valid nn.Module root: named_modules()
    # yields exactly the one quantized layer _metrics_meta walks for.
    meta = _metrics_meta(cfg, layer, category=None, subcategory=None)
    assert len(meta["layers"]) == 1
    layer_meta = meta["layers"][0]
    assert layer_meta["rt_mapping"] == "UNITS"

    # Independently compute the expected wire count the same way the branch
    # does, so this test would catch a branch that silently returns the wrong
    # quantity (e.g. block-bucket count, or a stale cached value) rather than
    # merely one that runs without raising.
    w_2d, _ = _layout_weight_for_racetrack(layer.weight, rt_mapping="ROW", kernel_mapping=None)
    buckets = build_unit_buckets(
        w_2d, rt_size, threshold=threshold, max_period=max_period, pool_guard=pool_guard,
    )
    expected_n_wires = int(sum(b.weight_grid.shape[0] for b in buckets.values()))

    assert layer_meta["n_racetracks"] == [expected_n_wires, 1]

    dense_shape = compute_index_offset_shape(
        tuple(layer.weight.shape), rt_size=rt_size, rt_mapping="ROW", kernel_size=None,
    )
    dense_total = dense_shape[0] * dense_shape[1]
    assert expected_n_wires > dense_total
    assert expected_n_wires <= layer.weight.numel()
