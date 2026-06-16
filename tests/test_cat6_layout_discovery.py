"""cat6 must discover the COL cat5 checkpoints, not the ROW ones.

sweep_regularizer writes cat5 checkpoints at
``<base_stem>/<lay-tok>/<crit-tok>/cat5_<key>_seed<N>/model.pt`` (layout
outermost). When cat6 runs with the col config, it must recalibrate ONLY the
col-trained checkpoints — recalibrating a row checkpoint and labelling it col
would silently corrupt the comparison. The discovery helper therefore takes a
``layout_filter`` analogous to the existing ``crit_filter``.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _make_ckpt(root: Path, *segments: str) -> None:
    d = root.joinpath(*segments)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_text("dummy")


def test_layout_filter_scopes_discovery_to_one_layout(tmp_path):
    from sweep_recalibration import _discover_cat5_checkpoints

    stem = "vgg7_cifar10_w1a1_rtm"
    root = tmp_path
    # Two layouts under the same save-root + criterion.
    _make_ckpt(root, stem, "lay-row", "crit-h128p0", "cat5_lam0p05_seed707")
    _make_ckpt(root, stem, "lay-col", "crit-h128p0", "cat5_lam0p05_seed707")

    col = _discover_cat5_checkpoints(root, stem, layout_filter="lay-col")
    assert len(col) == 1
    assert "lay-col" in col[0]["path"]
    assert col[0]["config_key"] == "lam0p05"
    assert col[0]["seed"] == 707
    assert col[0]["crit_tok"] == "crit-h128p0"
    assert col[0]["layout_tok"] == "lay-col"

    row = _discover_cat5_checkpoints(root, stem, layout_filter="lay-row")
    assert len(row) == 1
    assert "lay-row" in row[0]["path"]


def test_layout_and_crit_filters_compose(tmp_path):
    from sweep_recalibration import _discover_cat5_checkpoints

    stem = "vgg7_cifar10_w1a1_rtm"
    root = tmp_path
    _make_ckpt(root, stem, "lay-col", "crit-h128p0", "cat5_lam0p05_seed1")
    _make_ckpt(root, stem, "lay-col", "crit-ce", "cat5_lam0p05_seed1")
    _make_ckpt(root, stem, "lay-row", "crit-ce", "cat5_lam0p05_seed1")

    got = _discover_cat5_checkpoints(
        root, stem, crit_filter="crit-ce", layout_filter="lay-col"
    )
    assert len(got) == 1
    assert got[0]["crit_tok"] == "crit-ce"
    assert got[0]["layout_tok"] == "lay-col"


def test_no_layout_filter_returns_all_layouts(tmp_path):
    from sweep_recalibration import _discover_cat5_checkpoints

    stem = "vgg7_cifar10_w1a1_rtm"
    root = tmp_path
    _make_ckpt(root, stem, "lay-row", "crit-h128p0", "cat5_lam0p05_seed1")
    _make_ckpt(root, stem, "lay-col", "crit-h128p0", "cat5_lam0p05_seed1")

    got = _discover_cat5_checkpoints(root, stem)  # no filters
    assert len(got) == 2
    assert {c["layout_tok"] for c in got} == {"lay-row", "lay-col"}


def test_legacy_layout_free_paths_still_discovered(tmp_path):
    """Old checkpoints written before the layout segment (just <crit>/cat5_*)."""
    from sweep_recalibration import _discover_cat5_checkpoints

    stem = "vgg7_cifar10_w1a1_rtm"
    root = tmp_path
    _make_ckpt(root, stem, "crit-h128p0", "cat5_lam0p05_seed1")  # no lay- segment

    got = _discover_cat5_checkpoints(root, stem)
    assert len(got) == 1
    assert got[0]["crit_tok"] == "crit-h128p0"
    assert got[0]["layout_tok"] is None  # absent in legacy layout
