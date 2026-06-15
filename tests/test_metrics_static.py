"""Static-weight metrics: block/run/transition correctness and layout dependence."""
from __future__ import annotations

import torch


def test_block_and_runlength_single_racetrack():
    from netdrift.metrics.static import _block_runlength_for_rows

    # One racetrack (one row), rt_size covers the whole row: [+1,+1,-1,+1]
    # Runs along the wire: [+1,+1] (len 2, pos), [-1] (len 1, neg), [+1] (len 1, pos)
    rows = torch.tensor([[1.0, 1.0, -1.0, 1.0]])
    pos, neg, transitions, runlen_hist = _block_runlength_for_rows(rows, rt_size=4)
    assert pos == 2          # two positive blocks
    assert neg == 1          # one negative block
    assert transitions == 2  # +→- and -→+
    assert runlen_hist == {2: 1, 1: 2}  # one run of length 2, two runs of length 1


def test_runs_do_not_span_rt_size_boundary():
    from netdrift.metrics.static import _block_runlength_for_rows

    # rt_size=2 splits the row into two physical segments: [+1,+1] | [+1,+1].
    # Each segment is its own racetrack; runs must NOT merge across the split.
    rows = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    pos, neg, transitions, runlen_hist = _block_runlength_for_rows(rows, rt_size=2)
    assert pos == 2                 # one run per segment, both positive
    assert neg == 0
    assert transitions == 0         # no sign change within either segment
    assert runlen_hist == {2: 2}    # two runs of length 2


def test_runlength_ragged_segment_shorter_than_rt_size():
    from netdrift.metrics.static import _block_runlength_for_rows

    # n_cols (3) is NOT a multiple of rt_size (64) — a single ragged segment
    # shorter than the racetrack. This is the conv1 layout case in real models
    # (e.g. (64, 9) with rt_size 64). Must count the partial segment, not error.
    rows = torch.tensor([[1.0, 1.0, -1.0]])
    pos, neg, transitions, runlen_hist = _block_runlength_for_rows(rows, rt_size=64)
    assert pos == 1            # [+1,+1]
    assert neg == 1            # [-1]
    assert transitions == 1
    assert runlen_hist == {2: 1, 1: 1}


def test_magnitude_and_threshold_stats():
    from netdrift.metrics.static import _magnitude_stats

    w = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    stats = _magnitude_stats(w, per_channel_scale=None, n_bins=4, hist_max=2.0)
    # |w| = [2,1,1,2] -> mean 1.5
    assert abs(stats["mean"] - 1.5) < 1e-6
    assert abs(stats["std"] - w.abs().std(unbiased=False).item()) < 1e-6
    # histogram has n_bins entries and they sum to the element count
    assert len(stats["histogram"]) == 4
    assert sum(stats["histogram"].values()) == 4


def test_threshold_distance_uses_effective_scale():
    from netdrift.metrics.static import _magnitude_stats

    w = torch.tensor([2.0, 2.0])
    # With a per-channel scale of 2, the effective threshold distance halves.
    scale = torch.tensor([2.0, 2.0])
    stats = _magnitude_stats(w, per_channel_scale=scale, n_bins=4, hist_max=2.0)
    assert abs(stats["mean"] - 1.0) < 1e-6  # |2/2| = 1


def test_compute_static_metrics_layout_dependence():
    from netdrift.metrics.static import compute_static_metrics

    # A weight whose row-runs and column-runs differ.
    #   rows:    [+,+]  and [-,-]   -> 2 runs total along ROW
    #   columns: [+,-]  and [+,-]   -> 4 runs total along COL
    w = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])

    row = compute_static_metrics(w, rt_mapping="ROW", kernel_mapping=None, rt_size=2)
    col = compute_static_metrics(w, rt_mapping="COL", kernel_mapping=None, rt_size=2)

    assert row.block_count["total"] == 2
    assert col.block_count["total"] == 4
    # transitions differ accordingly
    assert row.sign_transitions == 0
    assert col.sign_transitions == 2


def test_compute_static_metrics_returns_dataclass_fields():
    from netdrift.metrics.static import compute_static_metrics, StaticLayerMetrics

    w = torch.randn(4, 6)
    m = compute_static_metrics(w, rt_mapping="ROW", kernel_mapping=None, rt_size=3)
    assert isinstance(m, StaticLayerMetrics)
    assert set(m.block_count) == {"pos", "neg", "total"}
    assert isinstance(m.run_length_histogram, dict)
    assert "mean" in m.weight_magnitude
    assert "mean" in m.dist_to_threshold
    assert m.raw_sign_transitions is None  # want_raw defaults False
