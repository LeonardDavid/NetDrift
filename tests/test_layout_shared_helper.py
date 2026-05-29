"""The layout helpers must round-trip and import without numba/CUDA."""
from __future__ import annotations

import importlib
import sys

import torch


def test_layout_module_imports_without_numba():
    # Ensure numba is not already imported via this module's import chain.
    mod = importlib.import_module("netdrift.faults.layout")
    # The layout module itself must not import numba at module load.
    assert "numba" not in mod.__dict__, "layout.py must not import numba"


def test_linear_row_roundtrip():
    from netdrift.faults.layout import _layout_weight_for_racetrack

    w = torch.randn(8, 20)
    w_2d, undo = _layout_weight_for_racetrack(w, rt_mapping="ROW", kernel_mapping=None)
    assert torch.allclose(undo(w_2d), w)


def test_linear_col_roundtrip():
    from netdrift.faults.layout import _layout_weight_for_racetrack

    w = torch.randn(8, 20)
    w_2d, undo = _layout_weight_for_racetrack(w, rt_mapping="COL", kernel_mapping=None)
    assert torch.allclose(undo(w_2d), w)


def test_conv_roundtrip_all_kernel_mappings():
    from netdrift.faults.layout import _layout_weight_for_racetrack

    w = torch.randn(4, 3, 3, 3)
    for rt_mapping in ("ROW", "COL"):
        for km in ("ROW", "COL", "CLW", "ACW"):
            w_2d, undo = _layout_weight_for_racetrack(
                w, rt_mapping=rt_mapping, kernel_mapping=km
            )
            restored = undo(w_2d)
            assert restored.shape == w.shape
            assert torch.allclose(restored, w), f"{rt_mapping}/{km} failed roundtrip"


def test_compute_index_offset_shape_importable_from_layout():
    from netdrift.faults.layout import compute_index_offset_shape

    # linear (8, 20), rt_size 16, ROW -> (8, ceil(20/16)=2)
    assert compute_index_offset_shape((8, 20), rt_size=16, rt_mapping="ROW") == (8, 2)


def test_rtm_misalignment_reexports_layout():
    # Back-compat: apply.py imports _layout_weight_for_racetrack from here.
    from netdrift.faults.rtm_misalignment import _layout_weight_for_racetrack  # noqa: F401
    from netdrift.faults.rtm_misalignment import compute_index_offset_shape  # noqa: F401
