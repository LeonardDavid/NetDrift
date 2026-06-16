"""Protection-policy override behaviour of the comparison-DB sweep drivers.

CPU-only, arg-assembly only (no training, no GPU).

Regression guard for the silent-clobber bug: the drivers used to ALWAYS emit
``fault.protection.policy=custom`` + ``fault.protection.layers=[2,3]`` (a
hardcoded VGG3 default), overriding whatever the config said — so a VGG7 sweep
launched without ``--protection-layers`` silently ran with VGG3's 2-layer set
instead of the config's ``[2,3,4,5,6,7]``.

New contract:
  * protection params default to ``None`` in the drivers,
  * when ``None`` the driver emits NO ``fault.protection.*`` override, so the
    runner uses the config's own values (and the schema halts on a bad
    ``custom`` + missing ``layers``),
  * when explicitly provided, the override IS emitted (escape hatch preserved).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

CURVE = [1e-7, 3e-7]


def _has_protection_override(argv: list[str]) -> bool:
    s = " ".join(argv)
    return "fault.protection.policy" in s or "fault.protection.layers" in s


# ---------------------------------------------------------------------------
# base_overrides (shared by both drivers' test/eval phase)
# ---------------------------------------------------------------------------

def test_base_overrides_omits_protection_when_none():
    from comparison_common import base_overrides

    argv = base_overrides(curve=CURVE, loops=10)
    assert not _has_protection_override(argv), (
        "base_overrides must NOT emit a protection override by default — "
        "the config's protection.policy/layers must survive."
    )
    # the other shared overrides are still present
    s = " ".join(argv)
    assert "fault.rt_error=" in s
    assert "training.loops=10" in s


def test_base_overrides_emits_protection_when_explicit():
    from comparison_common import base_overrides

    argv = base_overrides(
        curve=CURVE, loops=10,
        protection_policy="custom", protection_layers=[2, 3, 4, 5, 6, 7],
    )
    s = " ".join(argv)
    assert "fault.protection.policy=custom" in s
    # json_list serializes without spaces, e.g. [2,3,4,5,6,7]
    assert "fault.protection.layers=[2,3,4,5,6,7]" in s


def test_base_overrides_layers_alone_implies_custom():
    """Passing layers (the common case) is enough — policy defaults to custom."""
    from comparison_common import base_overrides

    argv = base_overrides(curve=CURVE, loops=10, protection_layers=[2, 3])
    s = " ".join(argv)
    assert "fault.protection.policy=custom" in s
    assert "fault.protection.layers=[2,3]" in s


# ---------------------------------------------------------------------------
# sweep_regularizer: train + test phases (cat5/cat8)
# ---------------------------------------------------------------------------

def _one_cat5_cell():
    import sweep_regularizer as reg
    cells = reg.build_cells(lambdas=[0.05], seeds=[707],
                            include_ste=False, inject_lambdas=[])
    return reg, next(c for c in cells if c["config_key"] == "lam0p05")


def test_reg_train_argv_omits_protection_when_none():
    reg, cell = _one_cat5_cell()
    argv = reg._train_argv(
        cfg_path=Path("configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml"),
        cell=cell, base_stem="vgg7_cifar10_w1a1_rtm",
        save_root=Path("/tmp/reg"), epochs=10, train_lr=0.001, beta=4.0,
        protection_layers=None, wdb_args=[], crit_tok="crit-h128p0",
        fault_aware_criterion="hinge", fault_aware_hinge_b=128.0,
    )
    assert not _has_protection_override(argv)


def test_reg_train_argv_emits_protection_when_explicit():
    reg, cell = _one_cat5_cell()
    argv = reg._train_argv(
        cfg_path=Path("configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml"),
        cell=cell, base_stem="vgg7_cifar10_w1a1_rtm",
        save_root=Path("/tmp/reg"), epochs=10, train_lr=0.001, beta=4.0,
        protection_layers=[2, 3, 4, 5, 6, 7], wdb_args=[], crit_tok="crit-h128p0",
        fault_aware_criterion="hinge", fault_aware_hinge_b=128.0,
    )
    s = " ".join(argv)
    assert "fault.protection.policy=custom" in s
    assert "fault.protection.layers=[2,3,4,5,6,7]" in s


def test_reg_test_argv_omits_protection_when_none():
    reg, cell = _one_cat5_cell()
    argv = reg._test_argv(
        cfg_path=Path("configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml"),
        cell=cell, base_stem="vgg7_cifar10_w1a1_rtm",
        save_root=Path("/tmp/reg"), curve=CURVE, loops=10,
        protection_layers=None, wdb_args=[], crit_tok="crit-h128p0",
    )
    assert not _has_protection_override(argv)


# ---------------------------------------------------------------------------
# sweep_recalibration: cell argv (cat4a/4b/6)
# ---------------------------------------------------------------------------

def test_recal_cell_argv_omits_protection_when_none():
    import sweep_recalibration as recal
    cells = recal._build_cells(
        Path("configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml"),
        seeds=[707], reg_checkpoint=None, categories={"4b"},
    )
    cell = next(c for c in cells if c["category"] == "4b")
    argv = recal._cell_argv(
        cell, Path("configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml"),
        CURVE, 10, None, None, None,
    )
    assert not _has_protection_override(argv)


def test_recal_cell_argv_emits_protection_when_explicit():
    import sweep_recalibration as recal
    cells = recal._build_cells(
        Path("configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml"),
        seeds=[707], reg_checkpoint=None, categories={"4b"},
    )
    cell = next(c for c in cells if c["category"] == "4b")
    argv = recal._cell_argv(
        cell, Path("configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml"),
        CURVE, 10, [2, 3, 4, 5, 6, 7], None, None,
    )
    s = " ".join(argv)
    assert "fault.protection.policy=custom" in s
    assert "fault.protection.layers=[2,3,4,5,6,7]" in s


# ---------------------------------------------------------------------------
# Schema + model-aware validation: bad protection halts execution
# ---------------------------------------------------------------------------

def test_schema_custom_without_layers_raises():
    import pytest
    from netdrift.config.schema import ProtectionCfg

    with pytest.raises(ValueError):
        ProtectionCfg(policy="custom", layers=None)
    with pytest.raises(ValueError):
        ProtectionCfg(policy="custom", layers=[])
    with pytest.raises(ValueError):
        ProtectionCfg(policy="custom", layers=[0, 2])  # 1-based, id<1 invalid
    with pytest.raises(ValueError):
        ProtectionCfg(policy="indiv", indiv_layer=None)
    # valid forms must NOT raise
    ProtectionCfg(policy="all")
    ProtectionCfg(policy="custom", layers=[2, 3, 4, 5, 6, 7])
    ProtectionCfg(policy="indiv", indiv_layer=4)


def test_apply_protection_rejects_out_of_range_layer_id():
    """A layer id beyond the model's layers (e.g. VGG3's [2,3] used where the
    model has fewer layers, or a too-high id) halts instead of silently
    mis-protecting."""
    import pytest
    from netdrift.models import build_model, replace_with_quantized
    from netdrift.models.protection import apply_protection_policy
    from netdrift.quant.binary import BinaryScheme

    model = build_model("vgg3_fmnist")  # 4 quantized layers → valid ids 1..4
    replace_with_quantized(model, BinaryScheme())

    # id 5 does not exist on vgg3 → must raise
    with pytest.raises(ValueError):
        apply_protection_policy(model, "custom", layers=[2, 3, 4, 5, 6, 7])
    # the correct vgg3 set applies cleanly
    apply_protection_policy(model, "custom", layers=[2, 3])
