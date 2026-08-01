import numpy as np
import torch

from netdrift.faults.layout import build_block_buckets
from netdrift.faults.packing import build_unit_buckets


def _canonical(bucket):
    """Sort a bucket's rows so wire ordering cannot cause a false mismatch."""
    key = [tuple(r) for r in bucket.scatter_rows.tolist()]
    key2 = [tuple(c) for c in bucket.scatter_cols.tolist()]
    order = sorted(range(len(key)), key=lambda i: (key[i], key2[i]))
    return (bucket.weight_grid[order], bucket.scatter_rows[order],
            bucket.scatter_cols[order], bucket.length[order])


def test_threshold1_maxperiod1_equals_block_buckets():
    torch.manual_seed(0)
    for shape, rt_size in [((4, 64), 64), ((7, 37), 8), ((1, 5), 4),
                           ((16, 129), 64), ((3, 1), 64)]:
        w = torch.where(torch.rand(*shape) < 0.5, 1.0, -1.0)
        blk = build_block_buckets(w, rt_size)
        uni = build_unit_buckets(w, rt_size, threshold=1, max_period=1,
                                 pool_guard=0)
        assert set(blk) == set(uni), (shape, rt_size, sorted(blk), sorted(uni))
        for p in blk:
            b_grid, b_r, b_c, b_len = _canonical(blk[p])
            u_grid, u_r, u_c, u_len = _canonical(uni[p])
            assert np.array_equal(b_grid, u_grid), (shape, rt_size, p, "grid")
            assert np.array_equal(b_r, u_r), (shape, rt_size, p, "rows")
            assert np.array_equal(b_c, u_c), (shape, rt_size, p, "cols")
            assert np.array_equal(b_len, u_len), (shape, rt_size, p, "length")


def test_equivalence_holds_for_skewed_sign_densities():
    torch.manual_seed(2)
    for prob in (0.1, 0.9):
        w = torch.where(torch.rand(8, 64) < prob, 1.0, -1.0)
        blk = build_block_buckets(w, 64)
        uni = build_unit_buckets(w, 64, threshold=1, max_period=1, pool_guard=0)
        assert set(blk) == set(uni)
        for p in blk:
            assert np.array_equal(_canonical(blk[p])[0], _canonical(uni[p])[0])
