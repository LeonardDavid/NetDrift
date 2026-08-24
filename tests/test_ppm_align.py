"""Greedy sign alignment: the minimum-flip baseline for PPM window counts.

Verified in the feasibility probe: a PPM sort window contributes a mixed wire
(``pad=False``) or one extra padded wire (``pad=True``) exactly when its positive
count ``p`` is not a multiple of ``rt_size``. The cheapest repair flips
``min(p mod R, R - p mod R)`` weights, choosing the smallest-magnitude latent
weights on the side that needs to give way — which is provably minimal and is
what a regularizer can at best match.

Pure host code: no CUDA, no checkpoint.
"""

from __future__ import annotations

import math

import pytest
import torch

from netdrift.faults.ppm_align import align_model_for_ppm, align_windows_to_multiple
from netdrift.faults.purity import wire_purity

R = 4


def _positives_per_window(w_2d, rt_size, window):
    nr, nc = w_2d.shape
    span = nc if window == 0 else window * rt_size
    out = []
    for r in range(nr):
        for s in range(0, nc, span):
            out.append(int((w_2d[r, s:s + span] > 0).sum()))
    return out


def _min_flips(w_2d, rt_size, window):
    return sum(min(p % rt_size, (-p) % rt_size)
               for p in _positives_per_window(w_2d, rt_size, window))


# --------------------------------------------------------------------------
# The core guarantee
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window", [0, 1, 2])
def test_every_window_count_becomes_a_multiple_of_rt_size(window):
    """The objective itself. Breaks if a window is left non-conforming."""
    torch.manual_seed(0)
    w = torch.randn(6, 8)

    aligned, _ = align_windows_to_multiple(w, R, window=window)

    for p in _positives_per_window(aligned, R, window):
        assert p % R == 0


def test_flip_count_is_the_theoretical_minimum():
    """Greedy is optimal here: each window is independent and needs exactly
    ``min(p mod R, R - p mod R)`` flips. Breaks if it flips a whole group, or
    picks the far side of the modulus.
    """
    torch.manual_seed(1)
    w = torch.randn(8, 12)

    _, flips = align_windows_to_multiple(w, R, window=0)

    assert flips == _min_flips(w, R, window=0)


def test_it_flips_the_smallest_magnitude_weights_of_the_giving_side():
    """Cheapest-first: a BNN pays least for flipping weights nearest 0.

    Row has 5 positives (needs 1 flip down to 4): the flipped one must be the
    smallest positive, 0.1 — not the largest, and not a negative.
    """
    w = torch.tensor([[0.9, 0.1, 0.5, 0.7, 0.3, -0.2, -0.8, -0.4]])

    aligned, flips = align_windows_to_multiple(w, R, window=0)

    assert flips == 1
    assert aligned[0, 1] < 0                      # the 0.1 gave way
    assert torch.equal(aligned[0, [0, 2, 3, 4]], w[0, [0, 2, 3, 4]])
    assert torch.equal(aligned[0, 5:], w[0, 5:])  # negatives untouched


def test_it_grows_the_positive_side_when_that_is_cheaper():
    """p mod R > R/2 means adding positives is fewer flips than removing them.

    7 positives of 8 at R=4: 3 flips down to 4, or 1 flip up to 8. Must pick 1,
    flipping the smallest-magnitude NEGATIVE.
    """
    w = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, -0.05]])

    aligned, flips = align_windows_to_multiple(w, R, window=0)

    assert flips == 1
    assert aligned[0, 7] > 0
    assert int((aligned > 0).sum()) == 8


def test_negation_preserves_magnitude():
    """Flips negate the latent weight, so |w| (and thus the weight's confidence
    and its distance-to-threshold statistics) is preserved. Breaks if a flip
    zeroes or clamps the weight instead.
    """
    w = torch.tensor([[0.9, 0.1, 0.5, 0.7, 0.3, -0.2, -0.8, -0.4]])

    aligned, _ = align_windows_to_multiple(w, R, window=0)

    assert torch.equal(aligned.abs(), w.abs())


def test_alignment_is_idempotent():
    torch.manual_seed(2)
    w = torch.randn(5, 16)

    once, first = align_windows_to_multiple(w, R, window=2)
    twice, second = align_windows_to_multiple(once, R, window=2)

    assert second == 0
    assert torch.equal(once, twice)


# --------------------------------------------------------------------------
# The payoff, in the real metric
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window", [0, 2])
def test_aligned_weights_make_unpadded_ppm_immune(window):
    """The point of the exercise: zero mixed wires without any padding."""
    torch.manual_seed(3)
    w = torch.randn(6, 16)

    aligned, _ = align_windows_to_multiple(w, R, window=window)

    assert wire_purity(w, R, "POLARITY", polarity_params=(window, False)).mixed > 0
    assert wire_purity(aligned, R, "POLARITY",
                       polarity_params=(window, False)).mixed == 0


def test_aligned_weights_make_padded_ppm_cost_exactly_dense():
    """The safe payoff: padding overhead goes to zero, immunity never at risk."""
    torch.manual_seed(4)
    w = torch.randn(6, 16)
    dense = w.shape[0] * math.ceil(w.shape[1] / R)

    aligned, _ = align_windows_to_multiple(w, R, window=0)

    assert wire_purity(w, R, "POLARITY", polarity_params=(0, True)).total > dense
    assert wire_purity(aligned, R, "POLARITY", polarity_params=(0, True)).total == dense


# --------------------------------------------------------------------------
# Windows that cannot be satisfied
# --------------------------------------------------------------------------

def test_windows_shorter_than_rt_size_are_left_alone():
    """A window of W < R can only reach p == 0 by going all-negative, which
    destroys the layer (this is vgg7's fc2, W=10 at rt_size=64). Skip it and
    report it rather than flattening the weights.
    """
    w = torch.tensor([[0.5, -0.2, 0.7]])          # W=3 < R=4

    aligned, flips = align_windows_to_multiple(w, R, window=0)

    assert flips == 0
    assert torch.equal(aligned, w)


def test_a_ragged_trailing_window_shorter_than_rt_size_is_skipped_not_forced():
    """Same rule for the short tail of an unevenly divided row: cols 0-3 are
    aligned, the 2-wide tail is left as it is.
    """
    w = torch.tensor([[0.9, 0.1, 0.5, -0.7, 0.3, -0.2]])

    aligned, _ = align_windows_to_multiple(w, R, window=1)

    assert int((aligned[0, :4] > 0).sum()) % R == 0
    assert torch.equal(aligned[0, 4:], w[0, 4:])


def test_an_all_positive_window_is_already_conforming_when_w_is_a_multiple():
    w = torch.ones(1, 8)

    aligned, flips = align_windows_to_multiple(w, R, window=0)

    assert flips == 0
    assert torch.equal(aligned, w)


# --------------------------------------------------------------------------
# Model level
# --------------------------------------------------------------------------

def test_align_model_reports_flips_per_layer_and_respects_protection():
    """Protected layers are excluded, matching ``run_length_penalty``'s rule:
    flipping them costs accuracy for no robustness gain.
    """
    import torch.nn as nn

    from netdrift.quant.layers import QuantizedLinear

    torch.manual_seed(5)
    model = nn.Sequential()
    for i in range(2):
        layer = QuantizedLinear(in_features=16, out_features=8, bias=False)
        layer.layer_name = f"lin{i}"
        model.add_module(f"lin{i}", layer)
    model.lin1.protected = True
    before = model.lin1.weight.detach().clone()

    report = align_model_for_ppm(model, rt_size=R, base_layout="ROW", window=0)

    assert report["lin0"]["flips"] > 0
    assert "lin1" not in report
    assert torch.equal(model.lin1.weight.detach(), before)


def test_align_model_honours_base_layout():
    """col aligns counts down the transposed view, so it flips a different set
    than row — the layout must be a real parameter, not a label.
    """
    import copy

    import torch.nn as nn

    from netdrift.quant.layers import QuantizedLinear

    torch.manual_seed(6)
    base = nn.Sequential()
    layer = QuantizedLinear(in_features=16, out_features=8, bias=False)
    layer.layer_name = "lin"
    base.add_module("lin", layer)

    row_model, col_model = copy.deepcopy(base), copy.deepcopy(base)
    align_model_for_ppm(row_model, rt_size=R, base_layout="ROW", window=0)
    align_model_for_ppm(col_model, rt_size=R, base_layout="COL", window=0)

    assert not torch.equal(row_model.lin.weight.detach(), col_model.lin.weight.detach())
    # ...and each is conforming in ITS OWN view
    for model, layout in ((row_model, "ROW"), (col_model, "COL")):
        from netdrift.faults.layout import _layout_weight_for_racetrack
        v, _ = _layout_weight_for_racetrack(model.lin.weight.detach(),
                                            rt_mapping=layout, kernel_mapping=None)
        assert wire_purity(v, R, "POLARITY", polarity_params=(0, False)).mixed == 0


def test_align_model_handles_conv_weights():
    """4D weights go through the kernel permutation like everything else."""
    import torch.nn as nn

    from netdrift.faults.layout import _layout_weight_for_racetrack
    from netdrift.quant.layers import QuantizedConv2d

    torch.manual_seed(7)
    model = nn.Sequential()
    conv = QuantizedConv2d(4, 8, kernel_size=3, bias=False)
    conv.layer_name = "conv"
    model.add_module("conv", conv)

    align_model_for_ppm(model, rt_size=R, base_layout="COL", window=0)

    v, _ = _layout_weight_for_racetrack(model.conv.weight.detach(),
                                        rt_mapping="COL", kernel_mapping="ROW")
    assert wire_purity(v, R, "POLARITY", polarity_params=(0, False)).mixed == 0


# --------------------------------------------------------------------------
# Driver script
# --------------------------------------------------------------------------

def test_aligned_checkpoint_path_encodes_layout_and_window():
    """The marker must carry (layout, window): an alignment is only valid for
    the pairing it was computed for, and evaluating the wrong pairing silently
    reports unaligned cost. Mirrors the ``_endlen`` marker convention.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from ppm_align_checkpoint import aligned_checkpoint_path

    got = aligned_checkpoint_path("models/w1a1/vgg7_cifar10/model_best.pt", "col", 0)

    assert got == Path("models/w1a1/vgg7_cifar10/model_best_ppmalign-col-w0.pt")
    assert aligned_checkpoint_path("a/m.pt", "ROW", 2).name == "m_ppmalign-row-w2.pt"


def test_cheap_to_align_flags_widths_that_are_not_multiples_of_twice_rt_size():
    """The flip budget collapses unless W % 2R == 0: the balanced sign-count mode
    W/2 must itself be a multiple of R. vgg3's conv2 (col width 64 at rt_size=64)
    is the real case that costs ~45% of its weights; vgg7's widths (128..1024)
    are all fine.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from ppm_align_checkpoint import cheap_to_align

    assert cheap_to_align(128, 64) and cheap_to_align(256, 64) and cheap_to_align(1024, 64)
    assert not cheap_to_align(64, 64)      # W/2 = 32, i.e. R/2 from any multiple
    assert not cheap_to_align(192, 64)     # 3R: W/2 = 96
    assert not cheap_to_align(10, 64)      # narrower than a wire: unfixable
    assert cheap_to_align(8, 4) and not cheap_to_align(4, 4)


def test_a_ragged_tail_wider_than_rt_size_is_genuinely_fixed_not_just_skipped():
    """Regression: the align/penalty skip rule is a WIDTH threshold, while
    ``polarity_wire_plan`` has none — so a tail between rt_size and 2*rt_size is
    charged by one and laid out by the other. It must still come out immune.

    22 cols at R=4, K=2 gives windows [0,8) [8,16) [16,22): the 6-wide tail is
    above the threshold, so it is aligned rather than skipped, and the pos/neg
    boundary has to land on a wire boundary for the wire count to be right.
    """
    import math

    torch.manual_seed(11)
    w = torch.randn(4, 22)
    dense = w.shape[0] * math.ceil(w.shape[1] / R)

    aligned, _ = align_windows_to_multiple(w, R, window=2)

    assert wire_purity(aligned, R, "POLARITY", polarity_params=(2, False)).mixed == 0
    assert wire_purity(aligned, R, "POLARITY", polarity_params=(2, True)).total == dense


def test_the_penalty_and_the_greedy_baseline_agree_on_a_ragged_layer():
    """The two implementations of one objective must not disagree about which
    windows are charged, or a regularizer-trained model would report conforming
    while the metric still shows overhead.
    """
    import torch.nn as nn

    from netdrift.quant.binary import BinaryScheme
    from netdrift.quant.layers import QuantizedLinear
    from netdrift.training.losses import ppm_count_penalty

    torch.manual_seed(12)
    w = torch.randn(4, 22)
    aligned, _ = align_windows_to_multiple(w, R, window=2)

    layer = QuantizedLinear(22, 4, bias=False)
    layer.attach_scheme(BinaryScheme())
    layer.rt_mapping = "ROW"
    layer.kernel_mapping = None
    with torch.no_grad():
        layer.weight.copy_(aligned)
    model = nn.Module()
    model.fc = layer

    assert ppm_count_penalty(model, beta=20.0, rt_size=R, base_layout="ROW",
                             window=2, kernel_mapping="ROW").item() == 0.0


def test_report_includes_how_deep_into_the_weight_distribution_the_flips_cut():
    """Damage proxy: flipping a weight at the 2nd |w| percentile is nearly free
    for a BNN, flipping one at the 40th is not. The report must expose that, or
    the flip COUNT alone gives no read on the accuracy risk.
    """
    import torch.nn as nn

    from netdrift.quant.layers import QuantizedLinear

    torch.manual_seed(9)
    model = nn.Sequential()
    layer = QuantizedLinear(in_features=16, out_features=8, bias=False)
    layer.layer_name = "lin"
    model.add_module("lin", layer)

    report = align_model_for_ppm(model, rt_size=R, base_layout="ROW", window=0)

    r = report["lin"]
    assert 0.0 <= r["flip_abs_pctl_mean"] <= r["flip_abs_pctl_max"] <= 100.0
    # Cheapest-first means the flips must come from the low end, not the middle.
    assert r["flip_abs_pctl_mean"] < 50.0


def test_percentiles_are_zero_when_nothing_is_flipped():
    w = torch.ones(1, 8)
    import torch.nn as nn

    from netdrift.quant.layers import QuantizedLinear

    model = nn.Sequential()
    layer = QuantizedLinear(in_features=8, out_features=1, bias=False)
    layer.layer_name = "lin"
    with torch.no_grad():
        layer.weight.copy_(w)
    model.add_module("lin", layer)

    r = align_model_for_ppm(model, rt_size=R, base_layout="ROW", window=0)["lin"]

    assert r["flips"] == 0
    assert r["flip_abs_pctl_mean"] == 0.0 and r["flip_abs_pctl_max"] == 0.0


def test_align_model_changes_signs_only_never_magnitudes_or_positions():
    """Decisive guard on the layout round-trip: after alignment every weight must
    keep its exact |w| AT ITS OWN POSITION, so the only possible change is a sign.

    A bug in the ``undo`` path (transpose or kernel-reshape mismatch) would
    permute weights instead of negating them — preserving the |w| multiset, the
    sign statistics and therefore the wire counts, while destroying the network.
    That failure is indistinguishable from "the flips were too damaging" unless
    it is checked here.
    """
    import copy

    import torch.nn as nn

    from netdrift.quant.binary import BinaryScheme
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    torch.manual_seed(13)
    model = nn.Sequential()
    conv = QuantizedConv2d(8, 16, kernel_size=3, bias=False)
    conv.attach_scheme(BinaryScheme())
    conv.layer_name = "conv"
    model.add_module("conv", conv)
    lin = QuantizedLinear(32, 16, bias=False)
    lin.attach_scheme(BinaryScheme())
    lin.layer_name = "lin"
    model.add_module("lin", lin)
    before = copy.deepcopy(model)

    report = align_model_for_ppm(model, rt_size=R, base_layout="COL", window=0,
                                 kernel_mapping="ROW")

    for name in ("conv", "lin"):
        b = getattr(before, name).weight.detach()
        a = getattr(model, name).weight.detach()
        assert a.shape == b.shape
        # elementwise |w| identical => no permutation, no magnitude change
        assert torch.equal(a.abs(), b.abs()), f"{name}: magnitudes moved"
        n_sign_changes = int(((a > 0) != (b > 0)).sum())
        assert n_sign_changes == report[name]["flips"]
