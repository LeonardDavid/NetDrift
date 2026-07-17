"""GPU end-to-end tests for the BLOCK racetrack mapping.

Exercises the full ``RTMMisalignmentFault`` BLOCK path (bucketed kernels +
scatter read-back) on a real CUDA device. Skipped when no GPU is present.
"""
import numpy as np
import pytest
import torch

numba_cuda = pytest.importorskip("numba.cuda")
if not numba_cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from netdrift.faults.base import FaultCtx
from netdrift.faults.rtm_misalignment import RTMConfig, RTMMisalignmentFault


def _ctx(base="ROW"):
    return FaultCtx(
        layer_id=1, layer_name="l", nr_run=1, training=False,
        extra={"rt_mapping": "BLOCK", "kernel_mapping": None,
               "kernel_size": None, "base_layout": base},
    )


def _binary(*shape, seed=0):
    torch.manual_seed(seed)
    w = torch.sign(torch.randn(*shape))
    w[w == 0] = -1.0
    return w


def test_block_bit_exact_at_zero_error():
    """rt_error=0 must leave every weight unchanged (no misalignment)."""
    w = _binary(16, 40, seed=0)
    cfg = RTMConfig(rt_size=64, rt_error=0.0, track_bitflips=True, block_mapping=True)
    fm = RTMMisalignmentFault(cfg)
    ctx = _ctx("ROW")
    state = fm.init_state(tuple(w.shape), ctx)
    new_w, new_state, stats = fm.inject(w, state, ctx)
    assert torch.equal(new_w, w), "rt_error=0 must be bit-exact"
    assert stats.bitflips == 0


def test_block_shape_and_counts_populate():
    """A faulted forward returns the right shape, caches structure, persists offsets."""
    w = _binary(16, 40, seed=1)
    cfg = RTMConfig(
        rt_size=64, rt_error=0.3,
        track_bitflips=True, track_misalign_faults=True,
        track_affected_units=True, track_wrong_reads=True,
        block_mapping=True,
    )
    fm = RTMMisalignmentFault(cfg)
    ctx = _ctx("ROW")
    state = fm.init_state(tuple(w.shape), ctx)
    new_w, new_state, stats = fm.inject(w, state, ctx)
    assert new_w.shape == w.shape
    assert stats.misalign_faults is not None
    assert stats.bitflips is not None
    assert stats.extra.get("wrong_bits_read") is not None
    # structure cached; a second forward reuses it (not rebuilt) and offsets persist
    assert new_state.block_buckets is not None
    new_w2, new_state2, _ = fm.inject(w, new_state, ctx)
    assert new_state2.block_buckets is new_state.block_buckets
    assert new_w2.shape == w.shape


def test_block_col_base_also_runs():
    """The COL base segmentation is a valid BLOCK base and produces a valid forward."""
    w = _binary(16, 40, seed=2)
    cfg = RTMConfig(rt_size=64, rt_error=0.2, track_bitflips=True, block_mapping=True)
    fm = RTMMisalignmentFault(cfg)
    ctx = _ctx("COL")
    state = fm.init_state(tuple(w.shape), ctx)
    new_w, new_state, stats = fm.inject(w, state, ctx)
    assert new_w.shape == w.shape
    assert new_state.base_mapping == "COL"


def test_block_guard_band_protects_small_shift():
    """A uniform block padded to >=2 has guard-band cells of the block's sign;
    with rt_error=0 the read is exact regardless (sanity floor for the guard band)."""
    # One +1 run of length 5 -> padded to 8: cells 5..7 are guard-band +1.
    w = torch.ones(1, 5)
    cfg = RTMConfig(rt_size=8, rt_error=0.0, track_bitflips=True, block_mapping=True)
    fm = RTMMisalignmentFault(cfg)
    ctx = _ctx("ROW")
    state = fm.init_state(tuple(w.shape), ctx)
    new_w, _, stats = fm.inject(w, state, ctx)
    assert torch.equal(new_w, w)
    assert stats.bitflips == 0


def test_block_conv_weight_roundtrips_at_zero_error():
    """4D conv weights go through kernel-mapping + BLOCK and are bit-exact at rt_error=0."""
    torch.manual_seed(3)
    w = torch.sign(torch.randn(4, 3, 3, 3))
    w[w == 0] = -1.0
    cfg = RTMConfig(rt_size=64, rt_error=0.0, track_bitflips=True, block_mapping=True)
    fm = RTMMisalignmentFault(cfg)
    ctx = FaultCtx(
        layer_id=1, layer_name="c", nr_run=1, training=False,
        extra={"rt_mapping": "BLOCK", "kernel_mapping": "ROW",
               "kernel_size": 3, "base_layout": "ROW"},
    )
    state = fm.init_state(tuple(w.shape), ctx)
    new_w, _, stats = fm.inject(w, state, ctx)
    assert new_w.shape == w.shape
    assert torch.equal(new_w, w)
    assert stats.bitflips == 0
