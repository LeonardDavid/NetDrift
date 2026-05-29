"""storage.layout must drive the fault model's per-layer rt_mapping."""
from __future__ import annotations

import pytest


def test_rt_mapping_fn_from_layout_row_col():
    from netdrift.runner.run import _rt_mapping_fn_for_layout

    fn_row = _rt_mapping_fn_for_layout("row")
    fn_col = _rt_mapping_fn_for_layout("col")
    assert fn_row(object()) == "ROW"
    assert fn_col(object()) == "COL"


def test_unsupported_layout_raises():
    from netdrift.runner.run import _rt_mapping_fn_for_layout

    with pytest.raises(NotImplementedError):
        _rt_mapping_fn_for_layout("interleaved")
    with pytest.raises(NotImplementedError):
        _rt_mapping_fn_for_layout("mix")
