"""Cell enumeration + override wiring for the software-category paper sweep.

These are the assertions that catch a miswired category BEFORE it costs GPU
hours: that each category loads the checkpoint it claims to, that only the
recalibrating categories carry a recalibrate block (with the right trigger),
that endlen is on for exactly one category, and that the structural axes
(layout, protection, loops, rt curve) are identical across all six — which is
what makes the comparison a comparison.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import sweep_paper_sw as sw  # noqa: E402


def _args(**kw):
    """A finalized Namespace for one category, without touching the disk."""
    argv = ["--category", kw.pop("category", "baseline")]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        argv += [flag] + ([str(x) for x in v] if isinstance(v, list) else [str(v)])
    return sw.parse_args(argv)


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def test_one_cell_per_seed_rt_curve_is_swept_inside():
    """rt_error is NOT a cell axis — the runner sweeps the whole curve in one
    invocation, so three seeds is three cells, not nine."""
    args = _args(category="reg", seeds=[707, 808, 909])
    cells = sw.build_cells(args)
    assert len(cells) == 3
    assert [c["seed"] for c in cells] == [707, 808, 909]


def test_subcategory_names_the_protection_range_and_seed():
    vgg = sw.build_cells(_args(category="reg", model="vgg7_cifar10"))
    assert vgg[0]["subcategory"] == "prot-2to7_seed707"
    res = sw.build_cells(_args(category="reg", model="resnet18_imagenette"))
    assert res[0]["subcategory"] == "prot-3to21_seed707"


def test_every_category_is_reachable_and_uniquely_tokened():
    tokens = [c["token"] for c in sw.CATEGORIES.values()]
    assert len(tokens) == len(set(tokens)) == 6


# ---------------------------------------------------------------------------
# Per-category wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category,ckpt_dir", [
    ("baseline", "model_best.pt"),
    ("endlen-recal", "model_best.pt"),
    ("reg", "cat5_col/model.pt"),
    ("reg-recal", "cat5_col/model.pt"),
    ("ste", "cat8_col/model.pt"),
    ("reg-ste-recal", "cat68_col/model.pt"),
])
def test_category_loads_its_own_checkpoint(category, ckpt_dir):
    """cat6 has no checkpoint of its own — it is cat5's weights plus this
    driver's re-fit — so reg and reg-recal deliberately share one file."""
    args = _args(category=category)
    cell = sw.build_cells(args)[0]
    ov = sw.cell_overrides(cell, args)
    ckpt = [o for o in ov if o.startswith("model.checkpoint=")][0]
    assert ckpt.endswith(ckpt_dir), ckpt


@pytest.mark.parametrize("category,trigger", [
    ("baseline", None),
    ("endlen-recal", "endlen"),
    ("reg", None),
    ("reg-recal", "always"),
    ("ste", None),
    ("reg-ste-recal", "always"),
])
def test_recalibration_is_on_for_exactly_the_recal_categories(category, trigger):
    """`on=endlen` fires because an encoder ran; the encoder-free categories
    need `always` or the re-fit silently never happens."""
    args = _args(category=category)
    ov = sw.cell_overrides(sw.build_cells(args)[0], args)
    enabled = [o for o in ov if o.startswith("training.recalibrate.enabled=")]
    if trigger is None:
        assert enabled == []
    else:
        assert enabled == ["training.recalibrate.enabled=true"]
        assert f"training.recalibrate.on={trigger}" in ov
        assert "training.recalibrate.bn_stats=true" in ov
        assert "training.recalibrate.tune_affine=true" in ov


def test_endlen_is_on_for_exactly_one_category():
    on = []
    for category in sw.CATEGORIES:
        args = _args(category=category)
        ov = sw.cell_overrides(sw.build_cells(args)[0], args)
        if "fault.weight_encoder=endlen" in ov:
            on.append(category)
        else:
            assert "fault.weight_encoder=null" in ov, category
    assert on == ["endlen-recal"]


def test_endlen_is_unbudgeted():
    """1.0/1.0 is vanilla endlen; a budget would make it cat3, not cat4."""
    args = _args(category="endlen-recal")
    ov = sw.cell_overrides(sw.build_cells(args)[0], args)
    assert "fault.global_bitflip_budget=1.0" in ov
    assert "fault.local_bitflip_budget=1.0" in ov
    assert "fault.weight_encoder_mode=once" in ov


# ---------------------------------------------------------------------------
# What must be IDENTICAL across categories (or the comparison is not one)
# ---------------------------------------------------------------------------

_STRUCTURAL = [
    "storage.layout=col",
    "storage.base_layout=col",
    "storage.rt_size=64",
    "storage.kernel_mapping=row",
    "fault.edge_mode=saturate",
    "fault.protection.policy=custom",
    "fault.protection.layers=[2,3,4,5,6,7]",
    "fault.ap_position=0",
    "training.mode=test",
    "training.fault_aware=none",
    "training.loops=100",
]


@pytest.mark.parametrize("category", list(sw.CATEGORIES))
def test_structural_axes_are_pinned_identically(category):
    args = _args(category=category, model="vgg7_cifar10")
    ov = sw.cell_overrides(sw.build_cells(args)[0], args)
    for expected in _STRUCTURAL:
        assert expected in ov, f"{category} missing {expected}"


@pytest.mark.parametrize("category", list(sw.CATEGORIES))
def test_rt_curve_is_identical_and_test_mode_never_trains(category):
    args = _args(category=category)
    ov = sw.cell_overrides(sw.build_cells(args)[0], args)
    assert "fault.rt_error=[0.0001,4.55e-05,1e-05]" in ov
    # Nothing in this driver may train weights. The recal categories tune BN and
    # Scale only, which is a different (and explicitly signed-off) thing.
    assert not any(o.startswith("training.fault_aware=regularization") for o in ov)
    assert not any(o.startswith("training.mode=train") for o in ov)


def test_resnet18_gets_its_own_protection_range():
    args = _args(category="reg", model="resnet18_imagenette")
    ov = sw.cell_overrides(sw.build_cells(args)[0], args)
    assert ("fault.protection.layers="
            "[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]") in ov


def test_resnet18_rejects_non_row_kernel_mapping():
    """The 1x1 shortcut convs make every non-ROW mapping raise deep in the fault
    path; the driver must refuse up front."""
    with pytest.raises(SystemExit):
        _args(category="reg", model="resnet18_imagenette", kernel_mapping="col")


# ---------------------------------------------------------------------------
# Paths + provenance
# ---------------------------------------------------------------------------

def test_experiment_name_and_run_dir():
    assert sw.experiment_name("vgg7_cifar10") == "paper-sw_vgg7_cifar10"
    args = _args(category="reg-ste-recal", model="vgg7_cifar10")
    cell = sw.build_cells(args)[0]
    d = sw.cell_run_dir(cell, args)
    assert d.parts[-3:] == ("paper-sw_vgg7_cifar10", "cat68_reg_ste_recal",
                            "prot-2to7_seed707")


def test_layout_flag_retargets_the_checkpoints():
    """--layout row must load the ROW-trained checkpoints, not the COL ones —
    the whole reason the dirs carry a layout suffix."""
    args = _args(category="reg-ste-recal", layout="row")
    ov = sw.cell_overrides(sw.build_cells(args)[0], args)
    assert any(o.endswith("cat68_row/model.pt") for o in ov
               if o.startswith("model.checkpoint=")), ov
    assert "storage.layout=row" in ov


def test_ap_position_can_be_unset_with_minus_one():
    args = _args(category="baseline", ap_position=-1)
    ov = sw.cell_overrides(sw.build_cells(args)[0], args)
    assert not any(o.startswith("fault.ap_position") for o in ov)


def test_passthrough_override_wins():
    args = _args(category="baseline")
    args.override = ["training.loops=7"]
    ov = sw.cell_overrides(sw.build_cells(args)[0], args)
    assert ov.index("training.loops=7") > ov.index("training.loops=100")


def test_wandb_category_matches_the_table():
    args = _args(category="reg-ste-recal")
    argv = sw.cell_argv(sw.build_cells(args)[0], args)
    assert "--wandb-category" in argv
    assert argv[argv.index("--wandb-category") + 1] == "cat68_reg_ste_recal"
    assert argv[argv.index("--wandb-project") + 1] == "netdrift-paper-runs-sw"
