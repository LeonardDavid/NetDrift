"""Layout-token helpers in scripts/comparison_common.py.

These disambiguate ROW vs COL comparison-DB artifacts the same way crit_token
disambiguates criteria: a name prefix + a save_dir path segment + a wandb
subcategory suffix. Without them, a COL sweep reusing the same --save-root and
configs would overwrite the ROW checkpoints/runs and corrupt the comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_layout_token_row_and_col():
    from comparison_common import layout_token

    assert layout_token("row") == "lay-row"
    assert layout_token("col") == "lay-col"


def test_layout_token_is_case_insensitive():
    from comparison_common import layout_token

    # configs may write COL/Col; the runner upper-cases internally, so the token
    # must normalize too (so a COL config and a col config share one tree).
    assert layout_token("COL") == "lay-col"
    assert layout_token("Row") == "lay-row"


def test_layout_token_rejects_unsupported_layout():
    from comparison_common import layout_token

    # mix/interleaved are schema placeholders not wired into the fault model;
    # the DB only compares row vs col, so anything else is a hard error rather
    # than a silently-colliding token.
    import pytest

    with pytest.raises(ValueError):
        layout_token("interleaved")


def test_layout_from_cfg_reads_storage_layout(tmp_path):
    from comparison_common import layout_from_cfg

    cfg = tmp_path / "vgg7_col.yaml"
    cfg.write_text(
        "experiment:\n"
        "  name: x\n"
        "storage:\n"
        "  layout: col\n"
        "  rt_size: 64\n"
    )
    assert layout_from_cfg(cfg) == "col"


def test_layout_from_cfg_defaults_to_row_when_absent(tmp_path):
    from comparison_common import layout_from_cfg

    # A config with no storage.layout key uses the schema default (row).
    cfg = tmp_path / "vgg7.yaml"
    cfg.write_text("experiment:\n  name: x\n")
    assert layout_from_cfg(cfg) == "row"
