"""``fault.ap_position`` is accepted for POLARITY, still rejected for BLOCK/units.

Why polarity is different (2026-09-14): ``build_polarity_buckets`` returns a
SINGLE bucket whose wires are all exactly ``rt_size`` — a ragged tail is filled
with the wire's sign rather than shortened — so one absolute access-port index
means the same thing on every wire, exactly as on dense ROW/COL. BLOCK and units
build one bucket PER padded length ``P`` and resolve the port per bucket as
``P//2 - 1``, where an absolute index genuinely is meaningless.

This matters for the paper runs: without it the polarity arms sat at mid-wire
while the row/col arms sat at ap0, so any polarity-vs-dense accuracy gap carried
an access-port term on top of the layout difference.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "python"))

from netdrift.faults.rtm_misalignment import RTMConfig  # noqa: E402


def _cfg(**kw):
    return RTMConfig(rt_size=64, rt_error=1e-5, edge_mode="saturate", **kw)


def test_polarity_accepts_absolute_ap_position():
    cfg = _cfg(polarity_mapping=True, ap_position=0)
    assert cfg.ap_position == 0


@pytest.mark.parametrize("ap", [0, 1, 31, 63])
def test_polarity_accepts_any_in_range_index(ap):
    assert _cfg(polarity_mapping=True, ap_position=ap).ap_position == ap


@pytest.mark.parametrize("mapping", ["block_mapping", "units_mapping"])
def test_block_and_units_still_reject(mapping):
    """These really do have heterogeneous per-bucket lengths."""
    with pytest.raises(ValueError, match="BLOCK or units"):
        _cfg(ap_position=0, **{mapping: True})


def test_rejection_message_no_longer_claims_polarity():
    with pytest.raises(ValueError) as e:
        _cfg(block_mapping=True, ap_position=0)
    assert "polarity" not in str(e.value).lower()


def test_negative_ap_position_still_rejected_everywhere():
    with pytest.raises(ValueError, match="ap_position must be >= 0"):
        _cfg(polarity_mapping=True, ap_position=-1)


def test_dense_paths_unaffected():
    assert _cfg(ap_position=0).ap_position == 0
    assert _cfg().ap_position is None


def test_polarity_wires_are_uniform_rt_size():
    """The property the relaxation rests on: one bucket, every wire rt_size
    long, so an absolute port index is well defined. If this ever stops holding,
    the guard above must come back."""
    torch = pytest.importorskip("torch")
    from netdrift.faults.partitioning import build_polarity_buckets

    rt_size = 8
    for pad in (True, False):
        w = torch.where(torch.randn(12, 40) > 0, 1.0, -1.0)
        buckets = build_polarity_buckets(w, rt_size, window=0, pad=pad)
        assert list(buckets) == [rt_size], f"pad={pad}: expected ONE bucket"
        grid = buckets[rt_size].weight_grid
        assert grid.shape[1] == rt_size, f"pad={pad}: wires must be rt_size wide"
        # Filler carries the wire's sign, so every cell is a readable ±1 — which
        # is what lets the read kernel treat the whole rt_size window as real.
        assert set(map(float, set(grid.reshape(-1).tolist()))) <= {1.0, -1.0}


def test_driver_marks_polarity_as_ap_supported():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import sweep_paper_runs as spr

    assert spr.ap_position_supported("polarity") is True
    assert spr.ap_position_supported("row") is True
    assert spr.ap_position_supported("col") is True
    assert spr.ap_position_supported("block") is False
    assert spr.ap_position_supported("units") is False
