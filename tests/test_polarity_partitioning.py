"""Polarity-partitioned mapping (PPM) — packing/structure tests.

These cover the mapping algorithm itself (pure host code, no CUDA). The
end-to-end fault-path behaviour lives in ``test_polarity_immunity.py``.
"""

import math

import pytest
import torch

from netdrift.faults.partitioning import (
    CHANNEL_ALIGNED,
    build_polarity_buckets,
    count_polarity_racetracks,
    polarity_wire_plan,
)

WINDOWS = [CHANNEL_ALIGNED, 1, 2, 4]


def _rand_signs(rows, cols, seed=0):
    torch.manual_seed(seed)
    return torch.where(torch.rand(rows, cols) > 0.5, 1.0, -1.0)


def _is_pure(row_values):
    return len(set((row_values > 0).tolist())) == 1


# --------------------------------------------------------------------------
# Round-trip: the permutation must be a bijection onto the real cells
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window", WINDOWS)
def test_every_cell_mapped_exactly_once(window):
    w = _rand_signs(6, 40)
    seen = set()
    for wire in polarity_wire_plan(w, 8, window=window, pad=True):
        assert wire["length"] == len(wire["cols"]) == len(wire["rows"])
        for rc in zip(wire["rows"], wire["cols"]):
            assert rc not in seen, f"cell {rc} emitted twice"
            seen.add(rc)
    assert len(seen) == w.numel()


@pytest.mark.parametrize("window", WINDOWS)
def test_scatter_indices_recover_original_values(window):
    """grid[real cells] must equal the original weights at their scatter idx."""
    w = _rand_signs(5, 33, seed=3)
    bucket = build_polarity_buckets(w, 8, window=window, pad=True)[8]
    mask = bucket.scatter_cols >= 0
    rows = bucket.scatter_rows[mask]
    cols = bucket.scatter_cols[mask]
    assert torch.allclose(
        torch.from_numpy(bucket.weight_grid[mask]),
        w[rows, cols].float(),
    )


# --------------------------------------------------------------------------
# Sign purity — the immunity precondition
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("rt_size", [2, 8, 64])
def test_padded_wires_are_all_sign_pure(window, rt_size):
    """With pad=True EVERY wire is same-sign, filler cells included.

    This is the whole mechanism: a same-sign wire reads identically at any
    offset under saturate, so it cannot be corrupted.
    """
    w = _rand_signs(4, 100, seed=1)
    bucket = build_polarity_buckets(w, rt_size, window=window, pad=True)[rt_size]
    for i, row in enumerate(bucket.weight_grid):
        assert _is_pure(row), f"wire {i} is mixed-sign: {row}"


def test_filler_cells_carry_the_wire_sign():
    # 3 positives at rt_size=4 -> one wire, 1 filler cell holding +1 (not 0).
    w = torch.tensor([[1.0, -1.0, 1.0, 1.0]])
    bucket = build_polarity_buckets(w, 4, window=CHANNEL_ALIGNED, pad=True)[4]
    pos = [i for i, L in enumerate(bucket.length) if bucket.weight_grid[i][0] > 0][0]
    assert bucket.length[pos] == 3
    assert bucket.weight_grid[pos][3] == 1.0
    assert bucket.scatter_cols[pos][3] == -1  # filler is not scattered back


def test_unpadded_ablation_produces_mixed_wires():
    """pad=False is the ablation arm and MUST leave the boundary wire mixed.

    If this ever passes as sign-pure the ablation has stopped being an
    ablation, and the padding claim would be untestable.
    """
    w = _rand_signs(6, 40, seed=2)
    bucket = build_polarity_buckets(w, 8, window=CHANNEL_ALIGNED, pad=False)[8]
    mixed = sum(1 for row in bucket.weight_grid if not _is_pure(row))
    assert mixed > 0


# --------------------------------------------------------------------------
# Sign-pure inputs must cost nothing extra
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value", [1.0, -1.0])
def test_uniform_layer_costs_exactly_dense(value):
    """A layer of one sign has no boundary, so padding adds zero wires."""
    w = torch.full((4, 32), value)
    n = count_polarity_racetracks(w, 8, window=CHANNEL_ALIGNED, pad=True)
    assert n == 4 * (32 // 8)


def test_zero_maps_to_negative_matching_extract_blocks():
    """Sign convention is ``w > 0 -> +1``; exactly-zero must group with -1."""
    w = torch.tensor([[0.0, 0.0, -1.0, -1.0]])
    n = count_polarity_racetracks(w, 4, window=CHANNEL_ALIGNED, pad=True)
    assert n == 1  # all four cells are one negative group


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------

def test_padding_adds_at_most_one_wire_per_window():
    """The (1 + 1/K) cost claim, stated exactly: <= 1 extra wire per window."""
    w = _rand_signs(4, 64, seed=5)
    rt_size, k = 8, 2
    dense = w.shape[0] * math.ceil(w.shape[1] / rt_size)
    n_windows = w.shape[0] * math.ceil(w.shape[1] / (k * rt_size))
    n = count_polarity_racetracks(w, rt_size, window=k, pad=True)
    assert dense <= n <= dense + n_windows


def test_channel_aligned_is_cheapest_window():
    """Fewer, wider windows mean fewer boundaries, hence fewer padded wires."""
    w = _rand_signs(6, 128, seed=6)
    counts = {k: count_polarity_racetracks(w, 8, window=k, pad=True)
              for k in (1, 2, 4, CHANNEL_ALIGNED)}
    assert counts[CHANNEL_ALIGNED] <= counts[4] <= counts[2] <= counts[1]


def test_count_helper_agrees_with_built_buckets():
    w = _rand_signs(5, 47, seed=7)
    for window in WINDOWS:
        n = count_polarity_racetracks(w, 8, window=window, pad=True)
        built = build_polarity_buckets(w, 8, window=window, pad=True)[8]
        assert n == built.weight_grid.shape[0]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_negative_window_rejected():
    with pytest.raises(ValueError, match="window must be >= 0"):
        polarity_wire_plan(_rand_signs(2, 8), 4, window=-1)


def test_rt_size_below_one_rejected():
    with pytest.raises(ValueError, match="rt_size must be >= 1"):
        polarity_wire_plan(_rand_signs(2, 8), 0)
