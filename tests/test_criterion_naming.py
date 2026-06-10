"""Criterion visibility: wandb flat fields + name/path encoding across drivers.

CPU-only. Exercises the cell-building / arg-assembly logic of the comparison-DB
drivers (no training, no GPU) plus the shared crit_token helpers and the wandb
config. Verifies the two asks:

  (a) criterion is a filterable wandb config field (baseline + fault-aware), and
  (b) criterion is embedded in run names / checkpoint paths for the categories
      that actually train with it (cat4b / cat5 / cat8 / cat6), via a name PREFIX
      and a save_dir PATH-LEVEL segment, leaving the cat5 leaf dir criterion-free
      so cat6 discovery still parses it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


# ---------------------------------------------------------------------------
# 1. crit_token <-> parse_crit_token round-trip
# ---------------------------------------------------------------------------

def test_crit_token_format():
    from comparison_common import crit_token

    assert crit_token("cross_entropy") == "crit-ce"
    assert crit_token("hinge", 128.0) == "crit-h128p0"
    assert crit_token("hinge", 64.0) == "crit-h64p0"
    with pytest.raises(ValueError):
        crit_token("focal")


def test_parse_crit_token_roundtrip():
    from comparison_common import crit_token, parse_crit_token

    assert parse_crit_token("crit-ce") == ("cross_entropy", 128.0)
    assert parse_crit_token("crit-h128p0") == ("hinge", 128.0)
    assert parse_crit_token("crit-h64p0") == ("hinge", 64.0)
    # None / missing → default hinge/128 (so an untagged old dir is treated as MHL)
    assert parse_crit_token(None) == ("hinge", 128.0)
    assert parse_crit_token("not-a-token") == ("hinge", 128.0)
    # round-trip both directions
    for name, b in [("hinge", 128.0), ("hinge", 32.0), ("cross_entropy", 128.0)]:
        assert parse_crit_token(crit_token(name, b)) == (name, b)


# ---------------------------------------------------------------------------
# 2. wandb config carries the four flat criterion fields
# ---------------------------------------------------------------------------

def test_wandb_config_has_flat_criterion_fields():
    from netdrift.config.schema import ExperimentConfig
    from netdrift.models import build_model, replace_with_quantized
    from netdrift.quant.binary import BinaryScheme
    from netdrift.runner.run import _wandb_config

    cfg = ExperimentConfig()
    cfg.training.criterion = "cross_entropy"
    cfg.training.fault_aware_criterion = "hinge"
    cfg.training.fault_aware_hinge_b = 64.0
    model = build_model("vgg3_fmnist")
    replace_with_quantized(model, BinaryScheme())

    wc = _wandb_config(cfg, model)
    # flat (top-level) keys so the W&B runs table can filter on them directly
    assert wc["criterion"] == "cross_entropy"
    assert wc["hinge_b"] == 128.0
    assert wc["fault_aware_criterion"] == "hinge"
    assert wc["fault_aware_hinge_b"] == 64.0


# ---------------------------------------------------------------------------
# 3. cat5/cat8 (sweep_regularizer): name PREFIX + save_dir PATH SEGMENT
# ---------------------------------------------------------------------------

def _one_cat5_cell():
    import sweep_regularizer as reg
    cells = reg.build_cells(lambdas=[0.05], seeds=[707],
                            include_ste=False, inject_lambdas=[])
    # the core lam0p05 no-inject cell
    return reg, next(c for c in cells if c["config_key"] == "lam0p05")


def test_cat5_name_and_save_dir_encode_criterion_ce():
    reg, cell = _one_cat5_cell()
    crit_tok = "crit-ce"
    train_argv = reg._train_argv(
        cfg_path=Path("configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml"),
        cell=cell, base_stem="vgg3_fmnist_w1a1_rtm",
        save_root=Path("/tmp/reg"), epochs=10, train_lr=0.001, beta=4.0,
        protection_layers=[2, 3], wdb_args=[], crit_tok=crit_tok,
        fault_aware_criterion="cross_entropy", fault_aware_hinge_b=128.0,
    )
    s = " ".join(train_argv)
    # name is a PREFIX segment: <stem>__crit-ce__cat5_..._train
    assert "experiment.name=vgg3_fmnist_w1a1_rtm__crit-ce__cat5_lam0p05_seed707_train" in s
    # save_dir is a PATH-LEVEL segment; the cat5 leaf stays criterion-free
    assert "training.save_dir=/tmp/reg/crit-ce/cat5_lam0p05_seed707" in s
    # the criterion override is passed to the runner
    assert "training.fault_aware_criterion=cross_entropy" in s


def test_cat5_test_phase_loads_from_criterion_path():
    reg, cell = _one_cat5_cell()
    test_argv = reg._test_argv(
        cfg_path=Path("configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml"),
        cell=cell, base_stem="vgg3_fmnist_w1a1_rtm",
        save_root=Path("/tmp/reg"), curve=[1e-6], loops=10,
        protection_layers=[2, 3], wdb_args=[], crit_tok="crit-ce",
    )
    s = " ".join(test_argv)
    assert "model.checkpoint=/tmp/reg/crit-ce/cat5_lam0p05_seed707/model.pt" in s
    assert "experiment.name=vgg3_fmnist_w1a1_rtm__crit-ce__cat5_lam0p05_seed707_test" in s
    assert "fault.weight_encoder=null" in s


def test_cat5_hinge_default_token():
    reg, cell = _one_cat5_cell()
    train_argv = reg._train_argv(
        cfg_path=Path("c.yaml"), cell=cell, base_stem="stem",
        save_root=Path("/tmp/r"), epochs=1, train_lr=0.001, beta=4.0,
        protection_layers=[2, 3], wdb_args=[], crit_tok="crit-h128p0",
        fault_aware_criterion="hinge", fault_aware_hinge_b=128.0,
    )
    s = " ".join(train_argv)
    assert "stem__crit-h128p0__cat5_lam0p05_seed707_train" in s
    assert "training.save_dir=/tmp/r/crit-h128p0/cat5_lam0p05_seed707" in s


# ---------------------------------------------------------------------------
# 4. cat6 (sweep_recalibration): discover criterion segment + inherit 1:1
# ---------------------------------------------------------------------------

def _make_fake_cat5_tree(root: Path, stem: str, crit_seg: str | None, config_key: str, seed: int):
    """Create <root>/<stem>[/<crit_seg>]/cat5_<config_key>_seed<seed>/model.pt."""
    leaf = f"cat5_{config_key}_seed{seed}"
    if crit_seg:
        d = root / stem / crit_seg / leaf
    else:
        d = root / stem / leaf
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_text("")  # empty stand-in; never loaded in dry build
    return d / "model.pt"


def test_cat6_discovers_criterion_segment_and_inherits(tmp_path):
    import sweep_recalibration as recal

    stem = "vgg3_fmnist_w1a1_rtm"
    # two cat5 checkpoints under different criterion path segments
    _make_fake_cat5_tree(tmp_path, stem, "crit-ce", "lam0p05", 1)
    _make_fake_cat5_tree(tmp_path, stem, "crit-h128p0", "lam0p1", 42)

    found = recal._discover_cat5_checkpoints(tmp_path, stem)
    by_key = {c["config_key"]: c for c in found}
    assert by_key["lam0p05"]["crit_tok"] == "crit-ce"
    assert by_key["lam0p05"]["seed"] == 1
    assert by_key["lam0p1"]["crit_tok"] == "crit-h128p0"

    cells = recal._build_cells(
        Path(f"configs/x/{stem}.yaml"), seeds=[707], reg_checkpoint=None,
        categories={"6"}, reg_save_root=tmp_path,
    )
    cells_by_key = {c["config_key"]: c for c in cells}
    # cat6 cell mirrors the source: name prefix carries the crit token, recal
    # criterion is INHERITED 1:1 from the checkpoint's segment.
    ce = cells_by_key["lam0p05"]
    assert ce["exp_name"] == f"{stem}__crit-ce__cat6_lam0p05_seed1"
    assert ce["criterion"] == "cross_entropy"
    assert ce["seed"] == 1
    assert ce["subcategory"] == "cat6_reg-recal_lam0p05_crit-ce"
    hi = cells_by_key["lam0p1"]
    assert hi["exp_name"] == f"{stem}__crit-h128p0__cat6_lam0p1_seed42"
    assert hi["criterion"] == "hinge"


def test_cat6_argv_overrides_inherited_criterion(tmp_path):
    import sweep_recalibration as recal

    stem = "vgg3_fmnist_w1a1_rtm"
    _make_fake_cat5_tree(tmp_path, stem, "crit-ce", "lam0p05", 1)
    cells = recal._build_cells(
        Path(f"configs/x/{stem}.yaml"), seeds=[707], reg_checkpoint=None,
        categories={"6"}, reg_save_root=tmp_path,
    )
    argv = recal._cell_argv(
        cells[0], Path("c.yaml"), curve=[1e-6], loops=10,
        protection_layers=[2, 3], wandb_project=None, wandb_entity=None,
        baseline_criterion="hinge", baseline_hinge_b=128.0,  # cat6 must IGNORE this
    )
    s = " ".join(argv)
    # cat6 uses the inherited cross_entropy, NOT the baseline hinge
    assert "training.criterion=cross_entropy" in s
    assert "training.criterion=hinge" not in s


def test_cat6_source_criterion_filter(tmp_path):
    """--source-criterion (→ crit_filter) restricts cat6 to one criterion's cat5
    checkpoints even when the save-root holds multiple criteria."""
    import sweep_recalibration as recal

    stem = "vgg3_fmnist_w1a1_rtm"
    _make_fake_cat5_tree(tmp_path, stem, "crit-ce", "lam0p05", 1)
    _make_fake_cat5_tree(tmp_path, stem, "crit-h128p0", "lam0p05", 1)
    _make_fake_cat5_tree(tmp_path, stem, "crit-ce", "lam0p1", 42)

    # no filter → all 3
    allc = recal._discover_cat5_checkpoints(tmp_path, stem)
    assert len(allc) == 3

    # filter to CE → only the 2 crit-ce ones
    ce = recal._discover_cat5_checkpoints(tmp_path, stem, crit_filter="crit-ce")
    assert len(ce) == 2
    assert all(c["crit_tok"] == "crit-ce" for c in ce)

    # filter to hinge → only the 1 crit-h128p0 one
    hi = recal._discover_cat5_checkpoints(tmp_path, stem, crit_filter="crit-h128p0")
    assert len(hi) == 1 and hi[0]["crit_tok"] == "crit-h128p0"

    # end-to-end via _build_cells: CE filter → 2 cat6 cells, all CE
    cells = recal._build_cells(
        Path(f"configs/x/{stem}.yaml"), seeds=[707], reg_checkpoint=None,
        categories={"6"}, reg_save_root=tmp_path, source_crit_filter="crit-ce",
    )
    assert len(cells) == 2
    assert all(c["criterion"] == "cross_entropy" for c in cells)


def test_cat4b_base_checkpoint_override():
    """--base-checkpoint overrides the cat4b model.checkpoint (e.g. CEL baseline);
    cat6 is unaffected (it loads its discovered cat5 checkpoint)."""
    import sweep_recalibration as recal

    stem = "vgg3_fmnist_w1a1_rtm"
    cells = recal._build_cells(
        Path(f"configs/x/{stem}.yaml"), seeds=[707], reg_checkpoint=None,
        categories={"4b"}, reg_save_root=None,
    )
    argv = recal._cell_argv(
        cells[0], Path("c.yaml"), curve=[1e-6], loops=10, protection_layers=[2, 3],
        wandb_project=None, wandb_entity=None,
        base_checkpoint="models/w1a1_cel/vgg3_fmnist/model_best.pt",
    )
    s = " ".join(argv)
    assert "model.checkpoint=models/w1a1_cel/vgg3_fmnist/model_best.pt" in s
    assert "model.checkpoint_mode=strict" in s


def test_cat5_base_checkpoint_warmstart():
    """--base-checkpoint sets the cat5 TRAIN-phase warm-start model.checkpoint."""
    reg, cell = _one_cat5_cell()
    train_argv = reg._train_argv(
        cfg_path=Path("c.yaml"), cell=cell, base_stem="stem",
        save_root=Path("/tmp/r"), epochs=1, train_lr=0.001, beta=4.0,
        protection_layers=[2, 3], wdb_args=[], crit_tok="crit-ce",
        fault_aware_criterion="cross_entropy", fault_aware_hinge_b=128.0,
        base_checkpoint="models/w1a1_cel/vgg3_fmnist/model_best.pt",
    )
    s = " ".join(train_argv)
    assert "model.checkpoint=models/w1a1_cel/vgg3_fmnist/model_best.pt" in s
    # default (no base_checkpoint) must NOT inject a model.checkpoint override
    train_argv2 = reg._train_argv(
        cfg_path=Path("c.yaml"), cell=cell, base_stem="stem",
        save_root=Path("/tmp/r"), epochs=1, train_lr=0.001, beta=4.0,
        protection_layers=[2, 3], wdb_args=[], crit_tok="crit-ce",
        fault_aware_criterion="cross_entropy", fault_aware_hinge_b=128.0,
    )
    assert "model.checkpoint=" not in " ".join(train_argv2)


def test_cat6_old_layout_no_criterion_segment_back_compat(tmp_path):
    """A cat5 dir WITHOUT a criterion segment (old layout) → no crit prefix,
    inherited criterion defaults to hinge."""
    import sweep_recalibration as recal

    stem = "vgg3_fmnist_w1a1_rtm"
    _make_fake_cat5_tree(tmp_path, stem, None, "lam0p05", 7)
    cells = recal._build_cells(
        Path(f"configs/x/{stem}.yaml"), seeds=[707], reg_checkpoint=None,
        categories={"6"}, reg_save_root=tmp_path,
    )
    c = cells[0]
    assert c["exp_name"] == f"{stem}__cat6_lam0p05_seed7"   # no crit prefix
    assert c["criterion"] == "hinge"


# ---------------------------------------------------------------------------
# 5. cat4b (sweep_recalibration): baseline criterion in name + argv
# ---------------------------------------------------------------------------

def test_cat4b_name_and_argv_encode_baseline_criterion():
    import sweep_recalibration as recal

    stem = "vgg3_fmnist_w1a1_rtm"
    cells = recal._build_cells(
        Path(f"configs/x/{stem}.yaml"), seeds=[707], reg_checkpoint=None,
        categories={"4b"}, reg_save_root=None, baseline_crit_tok="crit-ce",
    )
    c = cells[0]
    assert c["exp_name"] == f"{stem}__crit-ce__cat4_recal-bn-affine_seed707"

    argv = recal._cell_argv(
        c, Path("c.yaml"), curve=[1e-6], loops=10, protection_layers=[2, 3],
        wandb_project=None, wandb_entity=None,
        baseline_criterion="cross_entropy", baseline_hinge_b=128.0,
    )
    assert "training.criterion=cross_entropy" in " ".join(argv)


# ---------------------------------------------------------------------------
# 6. backfill infer_subcategory appends the crit token (live/backfill agree)
# ---------------------------------------------------------------------------

def test_infer_subcategory_appends_crit_token():
    import wandb_tag_categories as wtc

    # cat5 test run with a criterion prefix + the runner's -rt suffix
    name5 = "vgg3_fmnist_w1a1_rtm__crit-ce__cat5_lam0p05_seed1_test-rt1e-07"
    assert wtc.infer_category(name5) == "cat5_regularizer"
    assert wtc.infer_subcategory(name5, "cat5_regularizer") == \
        "cat5_regularizer_lam0p05_crit-ce"

    # cat6 with criterion prefix
    name6 = "vgg3_fmnist_w1a1_rtm__crit-h128p0__cat6_lam0p1_seed42-rt1e-06"
    assert wtc.infer_category(name6) == "cat6_reg_recal"
    assert wtc.infer_subcategory(name6, "cat6_reg_recal") == \
        "cat6_reg-recal_lam0p1_crit-h128p0"

    # cat5 WITHOUT a criterion prefix (old run) → no token appended
    name5_old = "vgg3_fmnist_w1a1_rtm__cat5_lam0p05_seed1_test-rt1e-07"
    assert wtc.infer_subcategory(name5_old, "cat5_regularizer") == \
        "cat5_regularizer_lam0p05"
