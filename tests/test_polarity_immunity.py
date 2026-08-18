"""Polarity-partitioned mapping (PPM) — end-to-end fault-path behaviour.

Requires CUDA (the RTM kernels are numba-cuda). The pure host-side packing
tests live in ``test_polarity_partitioning.py`` and need no GPU.

The central claim: with ``pad=True`` every wire is sign-pure, and a sign-pure
wire reads identically at every offset under ``edge_mode=saturate``. PPM is
therefore fault-immune for the same reason BLOCK is
(``test_ap_saturate.py::test_block_saturate_is_fault_immune``) — but at roughly
dense wire count rather than one racetrack per sign-run.
"""

import pytest
import torch

from netdrift.faults.base import FaultCtx
from netdrift.faults.rtm_misalignment import RTMConfig, RTMMisalignmentFault


def _make_ctx(rt_mapping="POLARITY", kernel_mapping=None, kernel_size=None,
              nr_run=1, base_layout="col"):
    """Mirrors ``test_ap_saturate.py::_make_ctx`` so the two suites agree."""
    extra = {
        "rt_mapping": rt_mapping,
        "kernel_mapping": kernel_mapping,
        "kernel_size": kernel_size,
    }
    if base_layout is not None:
        extra["base_layout"] = base_layout
    return FaultCtx(
        layer_id=1, layer_name="conv1", nr_run=nr_run, training=False, bits=1,
        extra=extra,
    )


def _signs(rows, cols, device="cuda"):
    return torch.where(torch.randn(rows, cols, device=device) > 0,
                       torch.ones(rows, cols, device=device),
                       -torch.ones(rows, cols, device=device))


def _run(weight, *, window=0, pad=True, rt_error=0.5, base_layout="col",
         rt_size=64, **cfg_kw):
    cfg = RTMConfig(rt_size=rt_size, rt_error=rt_error, edge_mode="saturate",
                    polarity_mapping=True, polarity_window=window,
                    polarity_pad=pad, track_bitflips=True, **cfg_kw)
    fault = RTMMisalignmentFault(cfg)
    ctx = _make_ctx(rt_mapping="POLARITY", base_layout=base_layout)
    state = fault.init_state(tuple(weight.shape), ctx)
    return fault.inject(weight, state, ctx)


# --------------------------------------------------------------------------
# Immunity
# --------------------------------------------------------------------------

@pytest.mark.cuda
@pytest.mark.parametrize("window", [0, 1, 2, 8])
def test_polarity_saturate_is_fault_immune(window):
    """pad=True + saturate => zero bitflips at any rt_error, for every window."""
    weight = _signs(64, 128)
    new_w, _, stats = _run(weight, window=window, rt_error=0.5)
    assert torch.equal(new_w, weight), f"PPM(window={window}) must be immune"
    assert stats.bitflips == 0


@pytest.mark.cuda
@pytest.mark.parametrize("base_layout", ["row", "col"])
def test_immune_under_both_base_layouts(base_layout):
    weight = _signs(32, 96)
    new_w, _, stats = _run(weight, base_layout=base_layout)
    assert torch.equal(new_w, weight)
    assert stats.bitflips == 0


@pytest.mark.cuda
def test_immunity_persists_across_repeated_injections():
    """Misalignment accumulates across runs; a sign-pure wire stays immune."""
    weight = _signs(32, 64)
    cfg = RTMConfig(rt_size=64, rt_error=0.5, edge_mode="saturate",
                    polarity_mapping=True, track_bitflips=True)
    fault = RTMMisalignmentFault(cfg)
    ctx = _make_ctx(rt_mapping="POLARITY", base_layout="col")
    state = fault.init_state(tuple(weight.shape), ctx)
    for run in range(1, 6):  # nr_run is 1-based
        ctx = _make_ctx(rt_mapping="POLARITY", base_layout="col", nr_run=run)
        weight_out, state, stats = fault.inject(weight, state, ctx)
        assert torch.equal(weight_out, weight), f"drifted on run {run}"
        assert stats.bitflips == 0


@pytest.mark.cuda
def test_conv_weights_are_immune():
    """4D conv weights go through the kernel-permutation path first."""
    w = torch.where(torch.randn(16, 8, 3, 3, device="cuda") > 0,
                    torch.ones(16, 8, 3, 3, device="cuda"),
                    -torch.ones(16, 8, 3, 3, device="cuda"))
    new_w, _, stats = _run(w)
    assert torch.equal(new_w, w)
    assert stats.bitflips == 0


# --------------------------------------------------------------------------
# The padding ablation — padding is the mechanism, not an optimisation
# --------------------------------------------------------------------------

@pytest.mark.cuda
def test_unpadded_polarity_is_NOT_immune():
    """pad=False leaves the boundary wire mixed-sign, so faults get through.

    This is the control for the immunity claim: if it ever passes as immune,
    then padding is not what is doing the work and the design's central
    argument (design doc section 3) is wrong.
    """
    weight = _signs(64, 128)
    _, _, stats = _run(weight, window=4, pad=False, rt_error=0.5)
    assert stats.bitflips > 0, "unpadded PPM must NOT be immune"


# --------------------------------------------------------------------------
# Functional invariance — PPM relocates weights, it never changes them
# --------------------------------------------------------------------------

@pytest.mark.cuda
@pytest.mark.parametrize("window", [0, 2])
def test_zero_rt_error_is_exact_identity(window):
    """The preflight invariant: no faults => bit-exact passthrough.

    Sharper for PPM than for other layouts because PPM changes no weight
    VALUE at all -- a mismatch here means the permutation lost or duplicated
    a weight, not that the fault model misbehaved.
    """
    weight = _signs(32, 100)
    new_w, _, stats = _run(weight, window=window, rt_error=0.0)
    assert torch.equal(new_w, weight)
    assert stats.bitflips == 0


# --------------------------------------------------------------------------
# Config rejections
# --------------------------------------------------------------------------

def test_polarity_rejects_ap_position():
    with pytest.raises(ValueError, match="ap_position is not supported"):
        RTMConfig(rt_size=64, polarity_mapping=True, ap_position=3)


def test_polarity_rejects_negative_window():
    with pytest.raises(ValueError, match="polarity_window must be >= 0"):
        RTMConfig(rt_size=64, polarity_mapping=True, polarity_window=-1)


@pytest.mark.parametrize("other", ["block_mapping", "units_mapping"])
def test_polarity_mutually_exclusive_with_other_packings(other):
    with pytest.raises(ValueError, match="mutually exclusive"):
        RTMConfig(rt_size=64, polarity_mapping=True, **{other: True})


def test_polarity_rejects_per_forward_encoder():
    with pytest.raises(ValueError, match="incompatible with weight_encoder_mode"):
        RTMConfig(rt_size=64, polarity_mapping=True, weight_encoder=object(),
                  weight_encoder_mode="per_forward")
