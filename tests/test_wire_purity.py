"""Wire-purity metric: counting sign-pure vs mixed-sign racetracks.

A wire is *pure* iff all of its PHYSICAL cells (real weights plus any
padding/guard filler, because ``simulate_racetrack_kernel`` reads those too)
hold the same sign. Under ``edge_mode=saturate`` that is exactly the immunity
criterion, so ``mixed == 0`` means "this layer cannot bitflip".

Pure host code — no CUDA, no checkpoint. The end-to-end JSON/W&B wiring is
covered by ``test_metrics_integration.py`` (CUDA) and
``test_wandb_config_keys.py`` (CPU).
"""

from __future__ import annotations

import math

import pytest
import torch

from netdrift.faults.purity import WirePurity, wire_purity

RT = 4

# 3x12 sign matrix reused across mappings. Under dense ROW at rt_size=4 it has
# exactly two sign-pure segments (r0c4..r0c7 and r2c0..r2c3) out of nine.
TOY = torch.tensor([[+1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1],
                    [-1, +1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1],
                    [+1, +1, +1, +1, +1, +1, -1, -1, -1, +1, -1, -1]],
                   dtype=torch.float32)


def _uniform(rows=3, cols=12, value=1.0):
    return torch.full((rows, cols), value, dtype=torch.float32)


# --------------------------------------------------------------------------
# Dense (ROW/COL): rt_size-wide segments of the laid-out view
# --------------------------------------------------------------------------

def test_dense_counts_the_two_pure_segments_of_the_toy():
    """Hand-counted ground truth: a dense segment is pure only by luck.

    Breaks if the segmentation drifts (e.g. segmenting columns instead of
    rows, or judging a wire on its first cell only).
    """
    p = wire_purity(TOY, RT, "ROW")

    assert (p.pure, p.mixed, p.total) == (2, 7, 9)
    assert (p.weights_pure, p.weights_mixed) == (8, 28)


def test_dense_ragged_tail_is_judged_on_its_real_cells():
    """A short trailing segment is a wire of its own, judged on what it holds.

    Row is 6 wide at rt_size=4: segment [+ - + +] is mixed, tail [- -] is pure.
    Breaks if the tail is padded with zeros (sign -1) or folded into the
    previous wire.
    """
    w = torch.tensor([[+1, -1, +1, +1, -1, -1]], dtype=torch.float32)

    p = wire_purity(w, RT, "ROW")

    assert (p.pure, p.mixed, p.total) == (1, 1, 2)
    assert (p.weights_pure, p.weights_mixed) == (2, 4)


def test_dense_col_view_is_segmented_row_wise_like_the_caller_laid_it_out():
    """``wire_purity`` takes an ALREADY laid-out view, so COL == ROW here.

    The COL transpose happens in ``_layout_weight_for_racetrack`` before this
    function sees the matrix (same contract as ``compute_static_metrics``).
    Breaks if the function starts transposing internally, double-transposing
    every COL caller.
    """
    assert wire_purity(TOY, RT, "COL") == wire_purity(TOY, RT, "ROW")


def test_exactly_zero_weight_counts_as_negative():
    """Sign convention must match ``extract_blocks``/``BinaryScheme``: w>0 -> +1.

    An all-zero wire is therefore pure(-), not mixed. Breaks if the
    implementation uses ``torch.sign`` (which maps 0 -> 0, a third value).
    """
    w = torch.zeros(1, RT, dtype=torch.float32)

    p = wire_purity(w, RT, "ROW")

    assert (p.pure, p.mixed) == (1, 0)


# --------------------------------------------------------------------------
# POLARITY: padding is the mechanism, so padded => zero mixed wires
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window", [0, 1, 2, 4])
def test_padded_polarity_has_no_mixed_wires(window):
    """The immunity precondition, measured rather than assumed.

    Breaks if the planner ever emits a wire spanning both sign groups — the
    same regression ``test_padded_wires_are_all_sign_pure`` guards on the
    grid, checked here through the metric that will be reported.
    """
    p = wire_purity(TOY, RT, "POLARITY", polarity_params=(window, True))

    assert p.mixed == 0
    assert p.pure == p.total
    assert p.weights_mixed == 0


def test_unpadded_polarity_has_one_mixed_wire_per_mixed_window():
    """The ablation arm: sorted but packed back-to-back leaves the boundary wire mixed.

    Channel-aligned on the toy = one window per row = 3 mixed wires of 9.
    Breaks if ``pad=False`` silently starts padding (which would make the
    padding-is-load-bearing control vacuous).
    """
    p = wire_purity(TOY, RT, "POLARITY", polarity_params=(0, False))

    assert (p.pure, p.mixed, p.total) == (6, 3, 9)
    assert p.weights_mixed == 12          # 3 mixed wires x 4 real cells


def test_unpadded_polarity_gets_worse_as_the_window_narrows():
    """More windows => more boundaries => more mixed wires (never fewer).

    This is the ordering the ablation results are read against, so pin it.
    """
    wide = wire_purity(TOY, RT, "POLARITY", polarity_params=(0, False))
    narrow = wire_purity(TOY, RT, "POLARITY", polarity_params=(2, False))

    assert narrow.mixed > wide.mixed


def test_polarity_total_matches_the_racetrack_counter():
    """Purity total must equal ``count_polarity_racetracks`` for the same args.

    The two numbers are reported side by side (``n_racetracks`` and
    ``wire_purity.total``); a disagreement means one of the two metrics
    dispatches has drifted.
    """
    from netdrift.faults.partitioning import count_polarity_racetracks

    for window in (0, 1, 2, 4):
        for pad in (True, False):
            p = wire_purity(TOY, RT, "POLARITY", polarity_params=(window, pad))
            assert p.total == count_polarity_racetracks(
                TOY, RT, window=window, pad=pad)


# --------------------------------------------------------------------------
# BLOCK / UNITS
# --------------------------------------------------------------------------

def test_block_wires_are_all_pure():
    """Every BLOCK wire is one maximal same-sign run plus sign-carrying padding.

    Breaks if the guard-band padding stops carrying the block's sign (which
    would make BLOCK fault-vulnerable at a ragged tail).
    """
    p = wire_purity(TOY, RT, "BLOCK")

    assert p.mixed == 0
    assert p.weights_mixed == 0


def test_block_total_matches_the_extracted_block_count():
    from netdrift.faults.layout import extract_blocks

    p = wire_purity(TOY, RT, "BLOCK")

    assert p.total == len(extract_blocks(TOY, RT))


def test_units_pooled_wires_are_mixed():
    """UNITS pools sub-threshold runs onto shared wires, which mixes signs.

    A row of alternating single-weight runs has NO run >= threshold, so every
    wire is a pooled (mixed) one. Breaks if pooling ever silently isolates by
    sign — the property that separates UNITS from PPM.
    """
    row = [(-1) ** i for i in range(RT * 2)]
    w = torch.tensor([row], dtype=torch.float32)

    p = wire_purity(w, RT, "UNITS", units_params=(2, 1, 0))

    assert p.mixed > 0
    assert p.weights_mixed > 0


def test_units_total_matches_the_packed_wire_count():
    from netdrift.faults.packing import build_unit_buckets

    p = wire_purity(TOY, RT, "UNITS", units_params=(2, 1, 0))
    buckets = build_unit_buckets(TOY, RT, threshold=2, max_period=1, pool_guard=0)

    assert p.total == int(sum(b.weight_grid.shape[0] for b in buckets.values()))


# --------------------------------------------------------------------------
# Cross-mapping invariants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mapping,kwargs", [
    ("ROW", {}),
    ("COL", {}),
    ("BLOCK", {}),
    ("UNITS", {"units_params": (2, 1, 0)}),
    ("POLARITY", {"polarity_params": (0, True)}),
    ("POLARITY", {"polarity_params": (2, False)}),
])
def test_every_weight_is_attributed_to_exactly_one_wire(mapping, kwargs):
    """weights_pure + weights_mixed == numel, for every mapping.

    Breaks if filler cells are counted as weights, or if a wire's real cells
    are missed (the units guard-slot case, where real cells are not a
    contiguous prefix).
    """
    p = wire_purity(TOY, RT, mapping, **kwargs)

    assert p.weights_pure + p.weights_mixed == TOY.numel()


@pytest.mark.parametrize("mapping,kwargs", [
    ("ROW", {}),
    ("BLOCK", {}),
    ("UNITS", {"units_params": (2, 1, 0)}),
    ("POLARITY", {"polarity_params": (0, True)}),
    ("POLARITY", {"polarity_params": (0, False)}),
])
def test_a_uniform_layer_is_all_pure_under_every_mapping(mapping, kwargs):
    """With one sign in the layer there is nothing to mix, whatever the packing."""
    p = wire_purity(_uniform(), RT, mapping, **kwargs)

    assert p.mixed == 0
    assert p.weights_mixed == 0


def test_dense_pure_fraction_is_lower_than_padded_polarity():
    """The headline comparison the metric exists to report."""
    dense = wire_purity(TOY, RT, "ROW")
    ppm = wire_purity(TOY, RT, "POLARITY", polarity_params=(0, True))

    assert dense.mixed_frac > 0.0
    assert ppm.mixed_frac == 0.0


# --------------------------------------------------------------------------
# WirePurity value object
# --------------------------------------------------------------------------

def test_fractions_are_shares_of_the_wire_and_weight_totals():
    p = WirePurity(pure=3, mixed=1, weights_pure=30, weights_mixed=10)

    assert p.total == 4
    assert p.total_weights == 40
    assert p.mixed_frac == pytest.approx(0.25)
    assert p.pure_frac == pytest.approx(0.75)
    assert p.weights_mixed_frac == pytest.approx(0.25)


def test_fractions_of_an_empty_layer_are_zero_not_a_division_error():
    p = WirePurity(pure=0, mixed=0, weights_pure=0, weights_mixed=0)

    assert p.mixed_frac == 0.0
    assert p.pure_frac == 0.0
    assert p.weights_mixed_frac == 0.0


def test_adding_purities_aggregates_layers_into_a_model_total():
    """Snapshot/summary totals are the sum over layers."""
    a = WirePurity(pure=1, mixed=2, weights_pure=10, weights_mixed=20)
    b = WirePurity(pure=3, mixed=4, weights_pure=30, weights_mixed=40)

    assert a + b == WirePurity(pure=4, mixed=6, weights_pure=40, weights_mixed=60)
    assert sum([a, b], WirePurity()) == a + b


def test_as_dict_carries_absolute_counts_and_percentages():
    """The JSON block: absolutes AND fractions, so no consumer has to divide."""
    d = WirePurity(pure=3, mixed=1, weights_pure=30, weights_mixed=10).as_dict()

    assert d == {
        "pure": 3, "mixed": 1, "total": 4, "mixed_frac": 0.25, "pure_frac": 0.75,
        "weights_pure": 30, "weights_mixed": 10, "weights_mixed_frac": 0.25,
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_units_without_params_raises_rather_than_reporting_dense_numbers():
    """Same loud-failure rule as ``compute_static_metrics``: wrong data > missing data."""
    with pytest.raises(ValueError, match="units_params"):
        wire_purity(TOY, RT, "UNITS")


def test_polarity_without_params_raises():
    with pytest.raises(ValueError, match="polarity_params"):
        wire_purity(TOY, RT, "POLARITY")


def test_unknown_mapping_raises():
    with pytest.raises(ValueError, match="rt_mapping"):
        wire_purity(TOY, RT, "INTERLEAVED")


def test_rt_size_below_one_rejected():
    with pytest.raises(ValueError, match="rt_size"):
        wire_purity(TOY, 0, "ROW")


def test_larger_case_agrees_with_a_brute_force_reference():
    """Independent O(cells) reference for dense, on a shape with a ragged tail.

    Guards the vectorised segment scan against an off-by-one at the tail.
    """
    torch.manual_seed(3)
    w = torch.where(torch.rand(5, 47) > 0.5, 1.0, -1.0)
    rt = 8

    p = wire_purity(w, rt, "ROW")

    pure = mixed = wp = wm = 0
    for r in range(w.shape[0]):
        for s in range(0, w.shape[1], rt):
            seg = [1 if v > 0 else -1 for v in w[r, s:s + rt].tolist()]
            if len(set(seg)) == 1:
                pure += 1
                wp += len(seg)
            else:
                mixed += 1
                wm += len(seg)
    assert (p.pure, p.mixed, p.weights_pure, p.weights_mixed) == (pure, mixed, wp, wm)
    assert p.total == w.shape[0] * math.ceil(w.shape[1] / rt)
