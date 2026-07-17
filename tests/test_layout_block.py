import pytest
import numpy as np
import torch
from netdrift.faults.layout import next_pow2, extract_blocks, build_block_buckets, compute_index_offset_shape, _ap_reads_for_mapping


def test_next_pow2():
    assert next_pow2(1) == 1
    assert next_pow2(2) == 2
    assert next_pow2(3) == 4
    assert next_pow2(5) == 8
    assert next_pow2(64) == 64


def test_uniform_row_is_one_block():
    # One row, all +1, length 4, rt_size 4 -> a single block of length 4.
    w = torch.ones(1, 4)
    blocks = extract_blocks(w, rt_size=4)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.sign == 1
    assert b.length == 4
    assert b.padded_len == 4
    assert b.rows == [0, 0, 0, 0]
    assert b.cols == [0, 1, 2, 3]


def test_alternating_row_is_all_length1():
    # +,-,+,- -> four length-1 blocks (padded_len 1 each).
    w = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    blocks = extract_blocks(w, rt_size=4)
    assert len(blocks) == 4
    assert all(b.length == 1 and b.padded_len == 1 for b in blocks)
    assert [b.sign for b in blocks] == [1, -1, 1, -1]


def test_mixed_padding():
    # +,+,+,-,-  -> block len3 (pad4), block len2 (pad2). rt_size 5.
    w = torch.tensor([[1.0, 1.0, 1.0, -1.0, -1.0]])
    blocks = extract_blocks(w, rt_size=5)
    assert [(b.length, b.padded_len, b.sign) for b in blocks] == [(3, 4, 1), (2, 2, -1)]


def test_runs_do_not_span_segment_boundary():
    # All +1, length 6, rt_size 3 -> two blocks (one per segment), each len 3.
    w = torch.ones(1, 6)
    blocks = extract_blocks(w, rt_size=3)
    assert len(blocks) == 2
    assert all(b.length == 3 for b in blocks)
    assert blocks[0].cols == [0, 1, 2]
    assert blocks[1].cols == [3, 4, 5]


def test_zero_binarizes_negative():
    # w == 0 must count as -1 (BinaryScheme convention).
    w = torch.tensor([[0.0, 0.0]])
    blocks = extract_blocks(w, rt_size=2)
    assert len(blocks) == 1
    assert blocks[0].sign == -1


def test_over_64_run_splits():
    # A single run of length 100 in one segment (rt_size 128) -> split into
    # ceil(100/64)=2 racetracks: len 64 (pad 64) + len 36 (pad 64).
    w = torch.ones(1, 100)
    blocks = extract_blocks(w, rt_size=128)
    assert len(blocks) == 2
    assert (blocks[0].length, blocks[0].padded_len) == (64, 64)
    assert (blocks[1].length, blocks[1].padded_len) == (36, 64)
    assert blocks[0].cols == list(range(64))
    assert blocks[1].cols == list(range(64, 100))


def test_buckets_group_by_padded_len():
    # row: +,+,+,-,-  -> block len3(pad4) and len2(pad2). rt_size 5.
    w = torch.tensor([[1.0, 1.0, 1.0, -1.0, -1.0]])
    buckets = build_block_buckets(w, rt_size=5)
    assert set(buckets.keys()) == {4, 2}
    b4 = buckets[4]
    assert b4.weight_grid.shape == (1, 4)
    # real cells 0..2 are +1; padding cell 3 is the block sign (+1).
    np.testing.assert_array_equal(b4.weight_grid[0], np.array([1, 1, 1, 1], dtype=np.float32))
    np.testing.assert_array_equal(b4.scatter_cols[0], np.array([0, 1, 2, -1]))
    assert int(b4.length[0]) == 3
    b2 = buckets[2]
    # len2 block of -1: both real, no padding.
    np.testing.assert_array_equal(b2.weight_grid[0], np.array([-1, -1], dtype=np.float32))
    np.testing.assert_array_equal(b2.scatter_cols[0], np.array([3, 4]))


def test_bucket_padding_holds_block_sign():
    # single -1 of length 5 -> pad 8: cells 5..7 must be -1 (block sign), not +1.
    w = -torch.ones(1, 5)
    buckets = build_block_buckets(w, rt_size=5)
    assert set(buckets.keys()) == {8}
    grid = buckets[8].weight_grid[0]
    np.testing.assert_array_equal(grid, np.array([-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32))
    np.testing.assert_array_equal(buckets[8].scatter_cols[0], np.array([0, 1, 2, 3, 4, -1, -1, -1]))


def test_scalar_apis_reject_block():
    with pytest.raises(ValueError, match="BLOCK"):
        compute_index_offset_shape((8, 8), rt_size=64, rt_mapping="BLOCK")
    with pytest.raises(ValueError, match="BLOCK"):
        _ap_reads_for_mapping(64, "BLOCK")
