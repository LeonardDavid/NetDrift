import pytest
import torch

from netdrift.faults.layout import extract_blocks
from netdrift.faults.units import parse_units, Fragment


def test_threshold_1_isolates_every_run():
    # +,-,+,+  -> runs [1,1,2]; threshold=1 isolates all, pool is empty.
    w = torch.tensor([[1.0, -1.0, 1.0, 1.0]])
    iso, frags = parse_units(w, rt_size=4, threshold=1)
    assert [b.length for b in iso] == [1, 1, 2]
    assert [b.sign for b in iso] == [1, -1, 1]
    assert frags == []


def test_threshold_2_pools_only_length1_runs():
    # +,-,+,+,+,-  -> runs [1,1,3,1]. threshold=2 isolates the len-3 run,
    # pools the len-1 runs into two fragments (split by the isolated run).
    w = torch.tensor([[1.0, -1.0, 1.0, 1.0, 1.0, -1.0]])
    iso, frags = parse_units(w, rt_size=6, threshold=2)
    assert [b.length for b in iso] == [3]
    assert [b.sign for b in iso] == [1]
    assert [f.cells for f in frags] == [[(0, 0), (0, 1)], [(0, 5)]]
    assert [f.signs for f in frags] == [[1, -1], [-1]]


def test_fragment_at_threshold_2_alternates_internally():
    # A fragment is a maximal group of consecutive length-1 runs, so at
    # threshold=2 its signs must strictly alternate.
    w = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, 1.0]])
    _iso, frags = parse_units(w, rt_size=6, threshold=2)
    assert len(frags) == 1
    s = frags[0].signs
    assert all(s[i] != s[i + 1] for i in range(len(s) - 1))


def test_threshold_4_pool_is_not_alternating():
    # runs [2,1,4]: threshold=4 pools the len-2 and len-1 runs into ONE
    # fragment whose signs do NOT alternate cell-to-cell (+,+,-).
    w = torch.tensor([[1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0]])
    iso, frags = parse_units(w, rt_size=7, threshold=4)
    assert [b.length for b in iso] == [4]
    assert len(frags) == 1
    assert frags[0].signs == [1, 1, -1]


def test_runs_never_span_segment_boundary():
    # All +1 across 8 cols with rt_size=4 -> two separate len-4 runs.
    w = torch.ones(1, 8)
    iso, frags = parse_units(w, rt_size=4, threshold=1)
    assert [b.length for b in iso] == [4, 4]
    assert frags == []


def test_zero_weight_is_negative_sign():
    # extract_blocks uses w > 0, so exact zero must map to -1.
    w = torch.tensor([[0.0, 0.0, 1.0]])
    iso, _frags = parse_units(w, rt_size=3, threshold=1)
    assert [(b.sign, b.length) for b in iso] == [(-1, 2), (1, 1)]


def test_ragged_tail_segment_included():
    # 5 cols, rt_size=4 -> segments [0..3] and [4]; the tail must be parsed.
    w = torch.ones(1, 5)
    iso, _frags = parse_units(w, rt_size=4, threshold=1)
    assert [b.length for b in iso] == [4, 1]


def test_threshold1_isolated_equals_extract_blocks():
    """At threshold=1 nothing is pooled, so the parse must reproduce
    extract_blocks exactly -- same order, same rows/cols/padded_len."""
    torch.manual_seed(0)
    for shape, rt_size in [((4, 64), 64), ((7, 37), 8), ((1, 5), 4),
                           ((16, 129), 64), ((3, 1), 64), ((5, 2), 2)]:
        w = torch.where(torch.rand(*shape) < 0.5, 1.0, -1.0)
        iso, frags = parse_units(w, rt_size, threshold=1)
        ref = extract_blocks(w, rt_size)
        assert frags == []
        assert len(iso) == len(ref), (shape, rt_size)
        for a, b in zip(iso, ref):
            assert (a.sign, a.length, a.padded_len, a.rows, a.cols) == \
                   (b.sign, b.length, b.padded_len, b.rows, b.cols), (shape, rt_size)


def test_threshold1_isolated_equals_extract_blocks_skewed_density():
    """Uniform p=0.5 signs under-exercise long runs; check skewed densities
    (mostly-one-sign matrices) on an (8, 64) matrix still match exactly."""
    torch.manual_seed(0)
    shape, rt_size = (8, 64), 64
    for p in (0.1, 0.9):
        w = torch.where(torch.rand(*shape) < p, 1.0, -1.0)
        iso, frags = parse_units(w, rt_size, threshold=1)
        ref = extract_blocks(w, rt_size)
        assert frags == []
        assert len(iso) == len(ref), p
        for a, b in zip(iso, ref):
            assert (a.sign, a.length, a.padded_len, a.rows, a.cols) == \
                   (b.sign, b.length, b.padded_len, b.rows, b.cols), p


def test_fragments_never_span_segment_boundary():
    """A fragment is a maximal group of consecutive pooled runs WITHIN one
    segment; it must never merge pooled runs from two different segments."""
    # rt_size=4. Row is + - + -  | + - + -  -> each segment is one all-len-1
    # fragment, so we must get TWO fragments of 4 cells, not one of 8.
    w = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
    iso, frags = parse_units(w, rt_size=4, threshold=2)
    assert iso == []
    assert len(frags) == 2, [f.cells for f in frags]
    assert [f.cells for f in frags] == [
        [(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 4), (0, 5), (0, 6), (0, 7)]]


def test_fragments_never_span_rows():
    """Two rows of all-alternating signs must give one fragment per row."""
    w = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    iso, frags = parse_units(w, rt_size=2, threshold=2)
    assert iso == []
    assert [f.cells for f in frags] == [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]


def test_threshold_below_1_raises():
    w = torch.tensor([[1.0, -1.0]])
    with pytest.raises(ValueError):
        parse_units(w, rt_size=2, threshold=0)
