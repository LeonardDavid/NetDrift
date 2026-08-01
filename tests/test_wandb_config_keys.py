"""Tests for the flat W&B config keys added in the design-space sweep spec
(``docs/superpowers/specs/2026-07-31-design-space-weekend-sweep.md`` §6):
``base_layout``/``edge_mode``/``ap_position`` (always present), the
``units_*`` triple (present only for ``storage.layout == "units"``), and
``n_racetracks`` (the sweep's protection-invariant cost x-axis).

Style follows ``tests/test_run_units_wiring.py`` / ``tests/test_config_units.py``:
configs are built through the real loader, layers are bare ``QuantizedLinear``
instances (a bare quantized layer is itself a valid ``nn.Module`` root — see
``test_metrics_meta_units_branch_reports_packed_wire_count`` in
test_run_units_wiring.py), and nothing here touches a GPU, numba, or a real
checkpoint.
"""

from __future__ import annotations


def _dense_col_cfg(*overrides: str):
    """A real dense (layout=col) config via the loader.

    Base file already has ``storage.layout: col`` and
    ``fault.protection: {policy: custom, layers: [2,3,4,5,6,7]}``; extra
    ``overrides`` (``key.path=value`` strings) let individual tests tweak it
    without hand-rolling a dataclass tree.
    """
    from netdrift.config.loader import load
    return load(
        "configs/modes/vgg7_cifar10_w1a1_baseline_col.yaml",
        overrides=list(overrides) or None,
    )


def _units_cfg(*overrides: str):
    from netdrift.config.loader import load
    return load(
        "configs/modes/vgg7_cifar10_w1a1_units_t4.yaml",
        overrides=list(overrides) or None,
    )


def test_dense_config_has_always_present_keys_and_no_units_keys():
    """layout=col: base_layout/edge_mode/ap_position present; units_* absent.

    ``units_*`` describe ``storage.units``, which the schema documents as
    "ignored" outside ``layout=="units"`` — logging its defaults on a dense
    run would misleadingly suggest units packing is in play, so the keys must
    be omitted entirely (not just null), matching the existing
    omit-when-unset pattern used for ``category``/``subcategory``.
    """
    from netdrift.quant.layers import QuantizedLinear
    from netdrift.runner.run import _wandb_config

    cfg = _dense_col_cfg()
    assert cfg.storage.layout == "col"
    layer = QuantizedLinear(in_features=8, out_features=4, bias=False)

    out = _wandb_config(cfg, layer)

    # Literal checks alongside the cfg-echo checks: this must actually be the
    # resolved value, not merely whatever the function happened to read back
    # (a hardcoded "row" would also satisfy `out["base_layout"] ==
    # cfg.storage.base_layout` if both were wrong the same way).
    assert out["base_layout"] == cfg.storage.base_layout == "row"
    assert out["edge_mode"] == cfg.fault.edge_mode == "saturate"
    assert out["ap_position"] == cfg.fault.ap_position is None

    for key in ("units_threshold", "units_max_period", "units_pool_guard"):
        assert key not in out, f"{key} must be absent on a non-units run"


def test_units_config_reports_configured_units_values():
    """layout=units: all three units_* keys present with the configured values."""
    from netdrift.quant.layers import QuantizedLinear
    from netdrift.runner.run import _wandb_config

    cfg = _units_cfg()
    assert cfg.storage.layout == "units"
    layer = QuantizedLinear(in_features=8, out_features=4, bias=False)

    out = _wandb_config(cfg, layer)

    assert out["units_threshold"] == cfg.storage.units.threshold == 4
    assert out["units_max_period"] == cfg.storage.units.max_period == 1
    assert out["units_pool_guard"] == cfg.storage.units.pool_guard == 0
    # Always-present keys still show up alongside the units_* triple. Literal
    # "row" alongside the cfg-echo, same reasoning as the dense test above.
    assert out["base_layout"] == cfg.storage.base_layout == "row"
    assert out["edge_mode"] == cfg.fault.edge_mode == "saturate"


def test_n_racetracks_omitted_when_none_present_when_given():
    """n_racetracks is a convenience add-on: absent by default, present when
    a caller (main()) supplies a precomputed value — never recomputed inside
    the config builder itself."""
    from netdrift.quant.layers import QuantizedLinear
    from netdrift.runner.run import _wandb_config, _wandb_config_with_category

    cfg = _dense_col_cfg()
    layer = QuantizedLinear(in_features=8, out_features=4, bias=False)

    omitted = _wandb_config(cfg, layer)
    assert "n_racetracks" not in omitted

    present = _wandb_config(cfg, layer, n_racetracks=12345)
    assert present["n_racetracks"] == 12345

    # _wandb_config_with_category must thread the same param through.
    omitted_cat = _wandb_config_with_category(cfg, layer, None, None)
    assert "n_racetracks" not in omitted_cat

    present_cat = _wandb_config_with_category(
        cfg, layer, "catA", "subA", n_racetracks=999
    )
    assert present_cat["n_racetracks"] == 999
    assert present_cat["category"] == "catA"
    assert present_cat["subcategory"] == "subA"


def test_total_racetracks_is_protection_invariant():
    """The regression this design exists to prevent: ``_total_racetracks``
    must give the SAME total under ``policy=all`` and
    ``policy=custom, layers=[2..7]`` for the identical layout, because it is
    the sweep's shared cost x-axis (spec §1). ``prot-2to7`` and ``prot-1to8``
    joining the same figure at different x-values would silently mix two cost
    definitions.

    The model here is deliberately built to *disagree* with cfg: one layer is
    tagged the way ``attach_fault_model`` tags an unprotected layer
    (``rt_mapping="COL"``), the other is left exactly as
    ``_QuantizedMixin._init_quant`` constructs it — ``rt_mapping=None`` — the
    same state a *protected* layer can be left in. A correct implementation
    reads ``cfg.storage.layout``/``base_layout`` for every layer and ignores
    both attributes, so it must report the COL total (64 wires/layer) for
    BOTH layers regardless of protection policy. An implementation that fell
    back to ``mod.rt_mapping or "ROW"`` (the pattern ``_metrics_meta`` uses
    for its purely descriptive per-layer listing) would silently count the
    untagged layer under the ROW formula (8 wires) instead — a different,
    smaller total that this test catches.
    """
    from netdrift.faults.layout import compute_index_offset_shape
    from netdrift.quant.layers import QuantizedLinear
    from netdrift.runner.run import _total_racetracks
    import torch.nn as nn

    cfg_all = _dense_col_cfg(
        "storage.layout=col", "storage.base_layout=col",
        "fault.protection.policy=all",
    )
    cfg_custom = _dense_col_cfg(
        "storage.layout=col", "storage.base_layout=col",
        "fault.protection.policy=custom", "fault.protection.layers=[2,3,4,5,6,7]",
    )
    assert cfg_all.fault.protection.policy == "all"
    assert cfg_custom.fault.protection.policy == "custom"

    # in_features != out_features so ROW and COL give visibly different
    # per-layer totals (8 vs 64) — a formula mixup cannot hide behind symmetry.
    rt_size = cfg_all.storage.rt_size
    model = nn.Module()
    model.tagged = QuantizedLinear(in_features=64, out_features=8, bias=False)
    model.tagged.rt_mapping = "COL"
    model.tagged.base_layout = "COL"
    model.untagged = QuantizedLinear(in_features=64, out_features=8, bias=False)
    # Left untouched: rt_mapping/base_layout are None, as _init_quant leaves
    # them — simulating a layer attach_fault_model never tagged.
    assert model.untagged.rt_mapping is None

    total_all = _total_racetracks(cfg_all, model)
    total_custom = _total_racetracks(cfg_custom, model)
    assert total_all == total_custom, (
        "n_racetracks must not depend on fault.protection.policy"
    )

    col_shape = compute_index_offset_shape(
        (8, 64), rt_size=rt_size, rt_mapping="COL", kernel_size=None
    )
    row_shape = compute_index_offset_shape(
        (8, 64), rt_size=rt_size, rt_mapping="ROW", kernel_size=None
    )
    col_wires = col_shape[0] * col_shape[1]
    row_wires = row_shape[0] * row_shape[1]
    assert col_wires == 64 and row_wires == 8  # sanity: formulas really differ

    # Both layers must count under COL — a cfg-not-module implementation
    # cannot see the untagged layer's (nonexistent) rt_mapping, so it applies
    # the same cfg-derived mapping to every quantized layer uniformly.
    assert total_all == 2 * col_wires
    # And NOT the mixed total a mod.rt_mapping-or-"ROW" fallback would give.
    assert total_all != col_wires + row_wires


def test_total_racetracks_units_branch_uses_cfg_kernel_mapping():
    """UNITS branch smoke test, doubling as a regression guard on the
    ``kernel_mapping``-from-cfg decision.

    ``_layout_weight_for_racetrack`` (layout.py) defaults an unset
    ``kernel_mapping`` to ``"ROW"`` internally (``km = kernel_mapping or
    "ROW"``) — the exact same fallback shape as the ``rt_mapping``/
    ``base_layout`` issue this design fixes, and just as real for conv
    layers: a protected conv layer's ``mod.kernel_mapping`` can be left
    ``None``, and reading it instead of ``cfg.storage.kernel_mapping`` would
    silently flatten the kernel in the wrong order. That changes which cells
    are sign-adjacent, which changes the UNITS packer's isolate/pool split —
    a data-dependent wrong answer, not just a formula swap.

    The 3x3, 1-in/1-out kernel below is hand-picked so ROW-order flattening
    ([+,+,+,+,+,-,-,-,-]) produces two runs of length >= threshold=4 (both
    isolated -> 2 wires), while COL-order flattening ([0,3,6,1,4,7,2,5,8] ->
    [+,+,-,+,+,-,+,-,-]) produces six runs all < 4 (all pooled -> 1 wire).
    The two orderings are thus deterministically distinguishable, not just
    "probably different for random data" — this test doesn't just avoid a
    crash in a branch nothing else in ``tests/`` executes, it pins the value.
    """
    import torch

    from netdrift.faults.layout import _layout_weight_for_racetrack
    from netdrift.faults.packing import build_unit_buckets
    from netdrift.quant.layers import QuantizedConv2d
    from netdrift.runner.run import _total_racetracks

    cfg = _units_cfg("storage.kernel_mapping=col")
    assert cfg.storage.layout == "units"
    assert cfg.storage.kernel_mapping == "col"
    assert cfg.storage.units.threshold == 4

    layer = QuantizedConv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[
            [1., 1., 1.],
            [1., 1., -1.],
            [-1., -1., -1.],
        ]]]))
    assert layer.kernel_mapping is None  # never tagged, as on a protected layer

    # A bare QuantizedConv2d is itself a valid nn.Module root (same pattern as
    # the QuantizedLinear cases above and in test_run_units_wiring.py).
    actual = _total_racetracks(cfg, layer)

    w_2d, _ = _layout_weight_for_racetrack(layer.weight, rt_mapping="ROW", kernel_mapping="COL")
    buckets = build_unit_buckets(
        w_2d, cfg.storage.rt_size,
        threshold=cfg.storage.units.threshold,
        max_period=cfg.storage.units.max_period,
        pool_guard=cfg.storage.units.pool_guard,
    )
    expected_col = int(sum(b.weight_grid.shape[0] for b in buckets.values()))
    assert expected_col == 1  # all six runs < threshold=4 -> one pooled wire

    w_2d_row, _ = _layout_weight_for_racetrack(layer.weight, rt_mapping="ROW", kernel_mapping="ROW")
    buckets_row = build_unit_buckets(
        w_2d_row, cfg.storage.rt_size,
        threshold=cfg.storage.units.threshold,
        max_period=cfg.storage.units.max_period,
        pool_guard=cfg.storage.units.pool_guard,
    )
    expected_row = int(sum(b.weight_grid.shape[0] for b in buckets_row.values()))
    assert expected_row == 2  # both runs >= threshold=4 -> two isolated wires

    assert actual == expected_col
    assert actual != expected_row
