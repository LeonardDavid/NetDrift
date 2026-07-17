"""Block-size histogram and padding metrics from run-length histograms."""
from __future__ import annotations

from netdrift.metrics.static import block_size_histogram_from_runs


def test_block_size_histogram():
    # runs: two of length 3 (->pad4), one of length 1 (->pad1), one of 8 (->pad8)
    rl = {3: 2, 1: 1, 8: 1}
    hist = block_size_histogram_from_runs(rl)
    assert hist == {4: 2, 1: 1, 8: 1}


def test_block_size_histogram_merges_same_pad():
    # length 3 (->4) and length 4 (->4) both map to padded 4
    rl = {3: 1, 4: 2}
    assert block_size_histogram_from_runs(rl) == {4: 3}
