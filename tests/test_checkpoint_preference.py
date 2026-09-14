"""Downstream stages must load ``model_best.pt``, falling back to ``model.pt``.

``training.save_dir`` receives BOTH files: ``model.pt`` is the final epoch and
``model_best.pt`` is the best-clean-accuracy epoch (runner/run.py saves the
latter inside the epoch loop and copies both at the end). Every driver that
consumes a trained checkpoint as the INPUT to a later stage should prefer the
best one, but must not break on the older checkpoint trees that only ever
contained ``model.pt``.

The resolution is deliberately conservative: it only ever swaps a path whose
basename is exactly ``model.pt``, so a deliberately-named file (an endlen
artifact, a hand-picked epoch) is never silently replaced.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from comparison_common import prefer_best_checkpoint  # noqa: E402


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    return p


def test_prefers_model_best_when_present(tmp_path):
    d = tmp_path / "cat5_col"
    _touch(d / "model.pt")
    _touch(d / "model_best.pt")
    assert prefer_best_checkpoint(d / "model.pt") == str(d / "model_best.pt")


def test_falls_back_to_model_when_best_absent(tmp_path):
    """Older checkpoint trees (and any run that never beat epoch 0) have only
    model.pt — those must keep resolving to exactly what was asked for."""
    d = tmp_path / "cat5_row"
    _touch(d / "model.pt")
    assert prefer_best_checkpoint(d / "model.pt") == str(d / "model.pt")


def test_missing_directory_returns_input_unchanged(tmp_path):
    """Nothing on disk yet: return the requested path verbatim so the driver's
    own missing-checkpoint check reports the path the user configured, rather
    than a model_best.pt they never asked for."""
    p = tmp_path / "nope" / "model.pt"
    assert prefer_best_checkpoint(p) == str(p)


def test_already_model_best_is_untouched(tmp_path):
    d = tmp_path / "cat8_col"
    _touch(d / "model_best.pt")
    assert prefer_best_checkpoint(d / "model_best.pt") == str(d / "model_best.pt")


@pytest.mark.parametrize("name", ["model_endlen.pt", "model_best_ppm.pt", "epoch7.pt"])
def test_other_basenames_are_never_swapped(tmp_path, name):
    """Only the literal ``model.pt`` is a candidate. A driver that asked for a
    specific artifact means it — ppm_align/endlen outputs and hand-picked epochs
    must survive resolution even with a model_best.pt sitting beside them."""
    d = tmp_path / "special"
    _touch(d / name)
    _touch(d / "model_best.pt")
    assert prefer_best_checkpoint(d / name) == str(d / name)


def test_relative_path_stays_relative(tmp_path, monkeypatch):
    """Driver checkpoint paths are repo-root-relative and go straight into
    ``--override model.checkpoint=...``. Resolution must check under the repo
    root but hand back a path of the SAME form, or the override would suddenly
    carry an absolute path from the driver's machine."""
    import comparison_common

    monkeypatch.setattr(comparison_common, "REPO_ROOT", tmp_path)
    rel = "models/w1a1/vgg7_cifar10/cat5_col/model.pt"
    _touch(tmp_path / rel)
    _touch(tmp_path / "models/w1a1/vgg7_cifar10/cat5_col/model_best.pt")
    assert (comparison_common.prefer_best_checkpoint(rel)
            == "models/w1a1/vgg7_cifar10/cat5_col/model_best.pt")


def test_relative_path_falls_back_relative(tmp_path, monkeypatch):
    import comparison_common

    monkeypatch.setattr(comparison_common, "REPO_ROOT", tmp_path)
    rel = "models/w1a1/vgg7_cifar10/cat5_row/model.pt"
    _touch(tmp_path / rel)
    assert comparison_common.prefer_best_checkpoint(rel) == rel


def test_relative_resolution_never_crosses_roots(tmp_path, monkeypatch):
    """REGRESSION (caught on the GPU host, 2026-09-14).

    A relative path is checked against the repo root AND the CWD, because the
    drivers' own existence guards accept either. But the two FILES must be
    compared inside the same tree. The original helper asked
    ``(REPO_ROOT / best).exists() or best.exists()``, so when REPO_ROOT held only
    ``model.pt`` and the CWD tree happened to hold a ``model_best.pt`` at the
    same relative path — which is exactly what an in-progress training run
    creates — it returned the CWD's best and silently loaded a different model's
    weights.
    """
    import os

    import comparison_common

    repo, cwd = tmp_path / "repo", tmp_path / "cwd"
    rel = "models/w1a1/vgg7_cifar10/cat8_col/model.pt"
    _touch(repo / rel)                                    # only model.pt here
    _touch(cwd / "models/w1a1/vgg7_cifar10/cat8_col/model_best.pt")   # decoy

    monkeypatch.setattr(comparison_common, "REPO_ROOT", repo)
    monkeypatch.chdir(cwd)
    assert comparison_common.prefer_best_checkpoint(rel) == rel
    assert os.getcwd() == str(cwd)  # the decoy really was reachable


def test_relative_resolution_uses_cwd_when_repo_root_has_nothing(tmp_path, monkeypatch):
    """The CWD fallback still works — it is only ever consulted as a whole tree,
    never mixed with the repo root's answer."""
    import comparison_common

    repo, cwd = tmp_path / "repo", tmp_path / "cwd"
    repo.mkdir()
    rel = "models/w1a1/vgg7_cifar10/cat5_col/model.pt"
    _touch(cwd / rel)
    _touch(cwd / "models/w1a1/vgg7_cifar10/cat5_col/model_best.pt")

    monkeypatch.setattr(comparison_common, "REPO_ROOT", repo)
    monkeypatch.chdir(cwd)
    assert (comparison_common.prefer_best_checkpoint(rel)
            == "models/w1a1/vgg7_cifar10/cat5_col/model_best.pt")


def test_placeholder_template_is_left_alone():
    """``ckpt_defaults`` hands out templates like ``cat5_{base_layout}/model.pt``
    that are formatted per cell. Resolving one before substitution must not
    touch it — the real resolution happens after ``.format``."""
    tpl = "models/w1a1/vgg7_cifar10/cat5_{base_layout}/model.pt"
    assert prefer_best_checkpoint(tpl) == tpl


# ---------------------------------------------------------------------------
# Call sites
# ---------------------------------------------------------------------------

def test_sweep_paper_runs_resolve_ckpt_prefers_best(tmp_path, monkeypatch):
    """sweep_paper_runs funnels every variant through resolve_ckpt, so wiring it
    there covers cell_overrides AND check_checkpoints in one place."""
    import comparison_common
    import sweep_paper_runs

    monkeypatch.setattr(comparison_common, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sweep_paper_runs, "REPO_ROOT", tmp_path)
    rel = "models/w1a1/vgg7_cifar10/cat5_col/model.pt"
    _touch(tmp_path / rel)
    _touch(tmp_path / "models/w1a1/vgg7_cifar10/cat5_col/model_best.pt")

    got = sweep_paper_runs.resolve_ckpt(
        "models/w1a1/vgg7_cifar10/cat5_{base_layout}/model.pt",
        variant="cat6", seed=707, base_layout="col",
    )
    assert got == "models/w1a1/vgg7_cifar10/cat5_col/model_best.pt"


def test_sweep_paper_runs_resolve_ckpt_falls_back(tmp_path, monkeypatch):
    import comparison_common
    import sweep_paper_runs

    monkeypatch.setattr(comparison_common, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sweep_paper_runs, "REPO_ROOT", tmp_path)
    rel = "models/w1a1/vgg7_cifar10/cat8_row/model.pt"
    _touch(tmp_path / rel)

    got = sweep_paper_runs.resolve_ckpt(
        "models/w1a1/vgg7_cifar10/cat8_{base_layout}/model.pt",
        variant="cat8", seed=707, base_layout="row",
    )
    assert got == rel


def test_cat5_discovery_prefers_best(tmp_path):
    """cat6 recalibrates a cat5 checkpoint; it should recalibrate the BEST one."""
    import sweep_recalibration as sr

    root = tmp_path / "stem" / "lay-col" / "crit-ce" / "cat5_lam0p05_seed707"
    _touch(root / "model.pt")
    _touch(root / "model_best.pt")
    found = sr._discover_cat5_checkpoints(tmp_path, "stem")
    assert len(found) == 1
    assert found[0]["path"] == str(root / "model_best.pt")
    assert found[0]["config_key"] == "lam0p05"
    assert found[0]["seed"] == 707


def test_cat5_discovery_falls_back_to_model(tmp_path):
    import sweep_recalibration as sr

    root = tmp_path / "stem" / "lay-row" / "crit-ce" / "cat5_lam0p05_seed1"
    _touch(root / "model.pt")
    found = sr._discover_cat5_checkpoints(tmp_path, "stem")
    assert len(found) == 1
    assert found[0]["path"] == str(root / "model.pt")


def test_cat5_discovery_finds_best_only_dir(tmp_path):
    """A cell whose model.pt was cleaned up but whose model_best.pt survives is
    still a usable cat5 checkpoint — the glob must not require model.pt."""
    import sweep_recalibration as sr

    root = tmp_path / "stem" / "lay-col" / "crit-ce" / "cat5_lam0p1_seed42"
    _touch(root / "model_best.pt")
    found = sr._discover_cat5_checkpoints(tmp_path, "stem")
    assert len(found) == 1
    assert found[0]["path"] == str(root / "model_best.pt")


def test_cat5_discovery_does_not_double_count(tmp_path):
    """model.pt + model_best.pt in one dir is ONE checkpoint, not two cells."""
    import sweep_recalibration as sr

    for seed in (707, 1):
        root = tmp_path / "stem" / "lay-col" / "crit-ce" / f"cat5_lam0p05_seed{seed}"
        _touch(root / "model.pt")
        _touch(root / "model_best.pt")
    found = sr._discover_cat5_checkpoints(tmp_path, "stem")
    assert len(found) == 2
    assert sorted(f["seed"] for f in found) == [1, 707]


def test_collect_records_the_resolved_checkpoint(tmp_path, monkeypatch):
    """A sweep launched while its checkpoints are still training can resolve
    model.pt for early cells and model_best.pt for later ones. The CSV is the
    only surviving record of which file each cell actually loaded."""
    import csv as _csv

    import comparison_common
    import sweep_paper_runs as spr

    monkeypatch.setattr(comparison_common, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(spr, "REPO_ROOT", tmp_path)
    # chdir into the fixture too: the relative checkpoint paths below would
    # otherwise also be reachable under the real repo, where an in-progress
    # training run may have just written a model_best.pt.
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "models/w1a1/vgg7_cifar10/cat5_col/model.pt")
    _touch(tmp_path / "models/w1a1/vgg7_cifar10/cat5_col/model_best.pt")
    _touch(tmp_path / "models/w1a1/vgg7_cifar10/cat8_col/model.pt")  # no _best

    ns = _ns(spr, tmp_path)
    cells = [
        {"arm": "col", "subcategory": "var-cat6", "variant": "cat6",
         "base_layout": "col", "pad": None, "policy": "custom", "seed": 707,
         "immune": False},
        {"arm": "col", "subcategory": "var-cat8", "variant": "cat8",
         "base_layout": "col", "pad": None, "policy": "custom", "seed": 707,
         "immune": False},
    ]
    out = tmp_path / "out"
    csv_path = spr.collect(cells, ns, out)
    rows = list(_csv.DictReader(open(csv_path)))
    assert rows[0]["checkpoint"].endswith("cat5_col/model_best.pt")
    assert rows[1]["checkpoint"].endswith("cat8_col/model.pt")


def _ns(spr, tmp_path):
    """Minimal Namespace for collect(): it only reads these fields."""
    import argparse
    return argparse.Namespace(
        model="vgg7_cifar10",
        layout="col",
        experiment_name="paper-sw_vgg7_cifar10",
        output_dir="runs/paper-runs/",
        rt_curve=[1e-6],
        ckpt_base="models/w1a1/vgg7_cifar10/model_best.pt",
        ckpt_cat6="models/w1a1/vgg7_cifar10/cat5_{base_layout}/model.pt",
        ckpt_cat8="models/w1a1/vgg7_cifar10/cat8_{base_layout}/model.pt",
        ckpt_ppmreg="models/w1a1/vgg7_cifar10/ppmreg_{base_layout}/model.pt",
    )


# ---------------------------------------------------------------------------
# --checkpoint-select: the choice is NOT neutral for ppmreg
# ---------------------------------------------------------------------------

def test_checkpoint_select_final_pins_model_pt(tmp_path, monkeypatch):
    """ppmreg's published conformance / wire-ratio numbers describe the FINAL
    weights (runner/run.py writes the ppm objective report from model.pt), while
    model_best.pt is selected on clean accuracy ALONE and need not be conforming.
    Without an explicit opt-out there is no way to ask for the final epoch: the
    resolver swaps any path whose basename is model.pt, including one passed by
    hand on --ckpt-ppmreg."""
    import argparse

    import comparison_common
    import sweep_paper_runs as spr

    monkeypatch.setattr(comparison_common, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(spr, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    rel = "models/w1a1/vgg7_cifar10/ppmreg_col/model.pt"
    _touch(tmp_path / rel)
    _touch(tmp_path / "models/w1a1/vgg7_cifar10/ppmreg_col/model_best.pt")

    tpl = "models/w1a1/vgg7_cifar10/ppmreg_{base_layout}/model.pt"
    assert spr.resolve_ckpt(tpl, variant="ppmreg", seed=707, base_layout="col",
                            prefer_best=True).endswith("model_best.pt")
    assert spr.resolve_ckpt(tpl, variant="ppmreg", seed=707, base_layout="col",
                            prefer_best=False) == rel

    cell = {"arm": "polarity-reg", "variant": "ppmreg", "base_layout": "col",
            "seed": 707}
    for select, expect in (("best", "model_best.pt"), ("final", "model.pt")):
        ns = argparse.Namespace(ckpt_base=None, ckpt_cat6=None, ckpt_cat8=None,
                                ckpt_ppmreg=tpl, checkpoint_select=select)
        assert spr.cell_checkpoint(cell, ns).endswith(expect)


def test_checkpoint_select_defaults_to_best():
    """Absent the attribute entirely (older callers), behave as 'best'."""
    import argparse

    import sweep_paper_runs as spr

    cell = {"arm": "col", "variant": "base", "base_layout": "col", "seed": 707}
    ns = argparse.Namespace(ckpt_base="models/x/model_best.pt", ckpt_cat6=None,
                            ckpt_cat8=None, ckpt_ppmreg=None)
    assert spr.cell_checkpoint(cell, ns) == "models/x/model_best.pt"


# ---------------------------------------------------------------------------
# --pads
# ---------------------------------------------------------------------------

def _paper_runs_cells(argv):
    import sweep_paper_runs as spr

    captured = {}
    real = spr.build_cells

    def capture(args):
        captured["cells"] = real(args)
        raise SystemExit(0)

    spr.build_cells = capture
    try:
        spr.main(argv)
    except SystemExit:
        # A parser error also exits — re-raise it rather than reporting a
        # confusing KeyError, so the negative tests observe what they assert on.
        if "cells" not in captured:
            raise
    finally:
        spr.build_cells = real
    return captured["cells"]


def test_pads_false_drops_the_immune_half():
    """pad=true is the fault-immune half; skipping it must leave only pad=false
    cells, and therefore no immune cells (which is what the seed reduction and
    the flatness audit key off)."""
    cells = _paper_runs_cells([
        "--layout", "polarity", "--base-layouts", "col", "--pads", "false",
        "--seeds", "707", "808", "909", "--no-check-checkpoints",
    ])
    assert {c["pad"] for c in cells} == {False}
    assert not any(c["immune"] for c in cells)
    assert len(cells) == 18   # 3 variants x 1 base x 3 seeds x 2 protections


def test_pads_default_keeps_both():
    cells = _paper_runs_cells([
        "--layout", "polarity", "--base-layouts", "col",
        "--seeds", "707", "808", "909", "--no-check-checkpoints",
    ])
    assert {c["pad"] for c in cells} == {True, False}


def test_pads_false_on_the_ppmreg_arm():
    cells = _paper_runs_cells([
        "--layout", "polarity-reg", "--base-layouts", "col", "--pads", "false",
        "--seeds", "707", "808", "909", "--no-check-checkpoints",
    ])
    assert len(cells) == 6    # 1 variant x 1 base x 3 seeds x 2 protections
    assert not any(c["immune"] for c in cells)


def test_pads_rejected_for_non_partitioning_arms():
    """row/col/block never set storage.partition.pad — accepting --pads there
    would silently do nothing."""
    with pytest.raises(SystemExit):
        _paper_runs_cells(["--layout", "row", "--pads", "false",
                           "--no-check-checkpoints"])
