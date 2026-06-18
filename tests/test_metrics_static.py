"""Static-weight metrics: block/run/transition correctness and layout dependence."""
from __future__ import annotations

import torch


def test_block_and_runlength_single_racetrack():
    from netdrift.metrics.static import _block_runlength_for_rows

    # One racetrack (one row), rt_size covers the whole row: [+1,+1,-1,+1]
    # Runs along the wire: [+1,+1] (len 2, pos), [-1] (len 1, neg), [+1] (len 1, pos)
    rows = torch.tensor([[1.0, 1.0, -1.0, 1.0]])
    pos, neg, transitions, runlen_hist, alt_hist = _block_runlength_for_rows(rows, rt_size=4)
    assert pos == 2          # two positive blocks
    assert neg == 1          # one negative block
    assert transitions == 2  # +→- and -→+
    assert runlen_hist == {2: 1, 1: 2}  # one run of length 2, two runs of length 1
    assert alt_hist == {2: 1}  # the trailing [-1, +1] is one alternating seq of length 2


def test_runs_do_not_span_rt_size_boundary():
    from netdrift.metrics.static import _block_runlength_for_rows

    # rt_size=2 splits the row into two physical segments: [+1,+1] | [+1,+1].
    # Each segment is its own racetrack; runs must NOT merge across the split.
    rows = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    pos, neg, transitions, runlen_hist, alt_hist = _block_runlength_for_rows(rows, rt_size=2)
    assert pos == 2                 # one run per segment, both positive
    assert neg == 0
    assert transitions == 0         # no sign change within either segment
    assert runlen_hist == {2: 2}    # two runs of length 2
    assert alt_hist == {}           # no alternation in either segment


def test_runlength_ragged_segment_shorter_than_rt_size():
    from netdrift.metrics.static import _block_runlength_for_rows

    # n_cols (3) is NOT a multiple of rt_size (64) — a single ragged segment
    # shorter than the racetrack. This is the conv1 layout case in real models
    # (e.g. (64, 9) with rt_size 64). Must count the partial segment, not error.
    rows = torch.tensor([[1.0, 1.0, -1.0]])
    pos, neg, transitions, runlen_hist, alt_hist = _block_runlength_for_rows(rows, rt_size=64)
    assert pos == 1            # [+1,+1]
    assert neg == 1            # [-1]
    assert transitions == 1
    assert runlen_hist == {2: 1, 1: 1}
    assert alt_hist == {}      # only one length-1 run, not a >=2 alternating group


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
    assert isinstance(m.alternating_seq_histogram, dict)
    assert "mean" in m.weight_magnitude
    assert "mean" in m.dist_to_threshold
    assert m.raw_sign_transitions is None  # want_raw defaults False
    assert m.raw_alternating_lengths is None  # want_raw defaults False


def test_alternating_sequence_canonical_example():
    from netdrift.metrics.static import _block_runlength_for_rows

    # The locked example: runs [3,1,1,3] -> the two middle length-1 runs form
    # one alternating sequence of length 2. Edge bits of the length-3 blocks
    # stay in their blocks (never counted toward the alternating sequence).
    rows = torch.tensor([[1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0]])
    *_, alt_hist = _block_runlength_for_rows(rows, rt_size=8)
    assert alt_hist == {2: 1}


def test_alternating_sequence_full_segment():
    from netdrift.metrics.static import _block_runlength_for_rows

    # Whole segment alternates: runs [1,1,1,1,1,1] -> one group of 6.
    rows = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
    *_, alt_hist = _block_runlength_for_rows(rows, rt_size=6)
    assert alt_hist == {6: 1}


def test_alternating_sequence_at_segment_start():
    from netdrift.metrics.static import _block_runlength_for_rows

    # Alternating at the segment start (no left flank): runs [1,1,1,3] -> {3:1}.
    rows = torch.tensor([[-1.0, 1.0, -1.0, 1.0, 1.0, 1.0]])
    *_, alt_hist = _block_runlength_for_rows(rows, rt_size=6)
    assert alt_hist == {3: 1}


def test_alternating_sequence_at_segment_end():
    from netdrift.metrics.static import _block_runlength_for_rows

    # Alternating at the segment end (no right flank): runs [3,1,1,1] -> {3:1}.
    rows = torch.tensor([[1.0, 1.0, 1.0, -1.0, 1.0, -1.0]])
    *_, alt_hist = _block_runlength_for_rows(rows, rt_size=6)
    assert alt_hist == {3: 1}


def test_alternating_sequence_none_when_no_alternation():
    from netdrift.metrics.static import _block_runlength_for_rows

    rows = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    *_, alt_hist = _block_runlength_for_rows(rows, rt_size=4)
    assert alt_hist == {}


def test_alternating_sequence_solitary_length1_is_not_counted():
    from netdrift.metrics.static import _block_runlength_for_rows

    # runs [3,1,3]: the single length-1 run is a block, NOT an alternating seq.
    rows = torch.tensor([[1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]])
    *_, alt_hist = _block_runlength_for_rows(rows, rt_size=7)
    assert alt_hist == {}


def test_alternating_sequence_does_not_span_rt_size_boundary():
    from netdrift.metrics.static import _block_runlength_for_rows

    # rt_size=2 splits [1,-1,1,-1] into two segments [1,-1] | [1,-1]. Each is its
    # own alternating sequence of length 2 -> {2:2}, NOT one merged {4:1}.
    rows = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    *_, alt_hist = _block_runlength_for_rows(rows, rt_size=2)
    assert alt_hist == {2: 2}


def test_alternating_overlay_does_not_change_block_metrics():
    from netdrift.metrics.static import _block_runlength_for_rows

    # [1,-1] on a 2-bit racetrack: two length-1 blocks AND one alt-seq of len 2.
    # Overlay = the same bits are described by BOTH metrics, intentionally.
    rows = torch.tensor([[1.0, -1.0]])
    pos, neg, transitions, runlen, alt_hist = _block_runlength_for_rows(
        rows, rt_size=2
    )
    assert pos == 1 and neg == 1                 # block metrics unchanged
    assert transitions == 1
    assert runlen == {1: 2}
    assert alt_hist == {2: 1}                     # new metric overlaid


def test_compute_static_metrics_has_alternating_histogram():
    from netdrift.metrics.static import compute_static_metrics

    # rows [1,1,1,-1,1,-1,-1,-1] in one racetrack -> one alt-seq of length 2.
    w = torch.tensor([[1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0]])
    m = compute_static_metrics(w, rt_mapping="ROW", kernel_mapping=None, rt_size=8)
    assert m.alternating_seq_histogram == {2: 1}
    # Overlay: block metrics unchanged (runs [3,1,1,3] -> 4 blocks).
    assert m.block_count["total"] == 4
    assert m.raw_alternating_lengths is None  # want_raw defaults False


def test_compute_static_metrics_alternating_layout_dependence():
    from netdrift.metrics.static import compute_static_metrics

    # ROW reads [+,+] and [-,-] (no alternation); COL reads [+,-] and [+,-]
    # (each column fully alternates) -> different alternating histograms.
    w = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    row = compute_static_metrics(w, rt_mapping="ROW", kernel_mapping=None, rt_size=2)
    col = compute_static_metrics(w, rt_mapping="COL", kernel_mapping=None, rt_size=2)
    assert row.alternating_seq_histogram == {}      # no alternation along rows
    assert col.alternating_seq_histogram == {2: 2}  # two columns, each [+,-]


def test_compute_static_metrics_raw_alternating_lengths_when_want_raw():
    import numpy as np
    from netdrift.metrics.static import compute_static_metrics

    # Two rows, rt_size 4 (one segment each). Row 0 = [1,-1,1,-1] -> 1 alt-seq;
    # row 1 = [1,1,1,1] -> 0 alt-seqs. raw array is per-(row, segment) count.
    w = torch.tensor([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, 1.0]])
    m = compute_static_metrics(
        w, rt_mapping="ROW", kernel_mapping=None, rt_size=4, want_raw=True
    )
    assert m.raw_alternating_lengths is not None
    arr = np.asarray(m.raw_alternating_lengths)
    assert arr.shape == (2, 1)                       # 2 rows, 1 segment
    assert arr[0, 0] == 1 and arr[1, 0] == 0
