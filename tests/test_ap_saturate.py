"""Fixed access-port + saturating edge reads (``edge_mode="saturate"``).

The default edge model places the access port at a fixed position on the
nanowire and clamps every read to the racetrack's real-data window, so:

* no random ±1 value is ever produced (contrast the legacy ``random`` mode),
* a read that would drift past a data edge returns the nearest real cell,
* a partially-filled racetrack replicates its edge value into the ragged tail,
* the offset accumulator obeys an asymmetric clamp derived from the AP and
  recovers from saturation when the opposite fault occurs.

Direct-kernel tests assert exact values (they double as sign-convention guards:
the read is ``k - offset``, not ``k + offset``). Fault-model tests cover the
config plumbing, BLOCK fault-immunity, and ragged-tail replication end to end.

GPU-required (Numba CUDA kernels run on the device).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from netdrift.faults import (
    FaultCtx,
    RTMConfig,
    RTMMisalignmentFault,
)


def _make_ctx(rt_mapping: str = "ROW", kernel_mapping: str | None = None,
              kernel_size: int | None = None, nr_run: int = 1,
              base_layout: str | None = None) -> FaultCtx:
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


# ---------------------------------------------------------------------------
# Config validation (no GPU) — plumbing + guards.
# ---------------------------------------------------------------------------

def test_edge_mode_defaults_to_saturate() -> None:
    cfg = RTMConfig(rt_size=64, rt_error=0.0)
    assert cfg.edge_mode == "saturate"
    assert cfg.ap_position is None


def test_invalid_edge_mode_raises() -> None:
    with pytest.raises(ValueError, match="edge_mode"):
        RTMConfig(rt_size=64, edge_mode="bogus")


def test_negative_ap_position_raises() -> None:
    with pytest.raises(ValueError, match="ap_position"):
        RTMConfig(rt_size=64, ap_position=-1)


def test_ap_position_with_block_raises() -> None:
    with pytest.raises(ValueError, match="BLOCK"):
        RTMConfig(rt_size=64, ap_position=8, block_mapping=True)


# ---------------------------------------------------------------------------
# Direct simulate-kernel value assertions (sign-convention discriminators).
# ---------------------------------------------------------------------------

@pytest.mark.cuda
def test_saturate_positive_offset_output_and_wrong_reads() -> None:
    """rt_size=4, qin=[1,-1,1,-1], offset=+1, saturate:

    slot k reads real index k-1, clamped to [0,3]:
      k=0 -> raw -1 -> clamp to 0 -> qin[0]= 1  (identity, NOT a wrong read)
      k=1 -> raw  0 -> qin[0]= 1
      k=2 -> raw  1 -> qin[1]=-1
      k=3 -> raw  2 -> qin[2]= 1
    => output [1, 1, -1, 1], wrong_read = 3 (k=1,2,3).

    This fails if the kernel computes k+offset instead of k-offset.
    """
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from netdrift.faults.kernels.rtm_numba import simulate_racetrack_kernel

    rt_size = 4
    qin = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    qout = np.zeros_like(qin)
    offset = np.array([[1]], dtype=np.int32)
    wrong = np.zeros_like(offset)
    rng = create_xoroshiro128p_states(1, seed=1)

    d_out = cuda.to_device(qout)
    d_wrong = cuda.to_device(wrong)
    simulate_racetrack_kernel[(1, 1), (1, 1)](
        rng, cuda.to_device(qin), d_out, cuda.to_device(offset), rt_size,
        d_wrong, 1, 1,  # track_wrong=1, edge_mode=1 (saturate)
    )
    cuda.synchronize()
    assert np.array_equal(d_out.copy_to_host(), np.array([[1.0, 1.0, -1.0, 1.0]]))
    assert d_wrong.copy_to_host().sum() == 3


@pytest.mark.cuda
def test_saturate_negative_offset_output_and_wrong_reads() -> None:
    """rt_size=4, qin=[1,-1,1,-1], offset=-1, saturate:

    slot k reads real index k+1, clamped to [0,3]:
      k=0 -> raw 1 -> qin[1]=-1
      k=1 -> raw 2 -> qin[2]= 1
      k=2 -> raw 3 -> qin[3]=-1
      k=3 -> raw 4 -> clamp to 3 -> qin[3]=-1  (identity, NOT a wrong read)
    => output [-1, 1, -1, -1], wrong_read = 3 (k=0,1,2).
    """
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from netdrift.faults.kernels.rtm_numba import simulate_racetrack_kernel

    rt_size = 4
    qin = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    qout = np.zeros_like(qin)
    offset = np.array([[-1]], dtype=np.int32)
    wrong = np.zeros_like(offset)
    rng = create_xoroshiro128p_states(1, seed=1)

    d_out = cuda.to_device(qout)
    d_wrong = cuda.to_device(wrong)
    simulate_racetrack_kernel[(1, 1), (1, 1)](
        rng, cuda.to_device(qin), d_out, cuda.to_device(offset), rt_size,
        d_wrong, 1, 1,
    )
    cuda.synchronize()
    assert np.array_equal(d_out.copy_to_host(), np.array([[-1.0, 1.0, -1.0, -1.0]]))
    assert d_wrong.copy_to_host().sum() == 3


@pytest.mark.cuda
def test_saturate_zero_offset_is_identity() -> None:
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from netdrift.faults.kernels.rtm_numba import simulate_racetrack_kernel

    rt_size = 4
    qin = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    qout = np.zeros_like(qin)
    offset = np.array([[0]], dtype=np.int32)
    wrong = np.zeros_like(offset)
    rng = create_xoroshiro128p_states(1, seed=1)

    d_out = cuda.to_device(qout)
    d_wrong = cuda.to_device(wrong)
    simulate_racetrack_kernel[(1, 1), (1, 1)](
        rng, cuda.to_device(qin), d_out, cuda.to_device(offset), rt_size,
        d_wrong, 1, 1,
    )
    cuda.synchronize()
    assert np.array_equal(d_out.copy_to_host(), qin)
    assert d_wrong.copy_to_host().sum() == 0


@pytest.mark.cuda
def test_saturate_never_produces_out_of_alphabet_values() -> None:
    """Under saturate mode every output value is one that appears in the input
    racetrack — never a fresh random value (the whole point of the mode)."""
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from netdrift.faults.kernels.rtm_numba import simulate_racetrack_kernel

    rt_size = 8
    # A racetrack whose real cells are ALL +1: any saturating/shifted read must
    # still return +1 — a random-mode kernel would emit some -1s at the edges.
    qin = np.ones((1, rt_size), dtype=np.float32)
    qout = np.zeros_like(qin)
    offset = np.array([[3]], dtype=np.int32)  # large positive shift
    wrong = np.zeros_like(offset)
    rng = create_xoroshiro128p_states(1, seed=7)

    d_out = cuda.to_device(qout)
    simulate_racetrack_kernel[(1, 1), (1, 1)](
        rng, cuda.to_device(qin), d_out, cuda.to_device(offset), rt_size,
        cuda.to_device(wrong), 0, 1,  # edge_mode=1
    )
    cuda.synchronize()
    assert np.all(d_out.copy_to_host() == 1.0), "saturate must not invent values"


# ---------------------------------------------------------------------------
# Direct calc-kernel: asymmetric clamp + recovery from saturation.
# ---------------------------------------------------------------------------

@pytest.mark.cuda
def test_calc_asymmetric_clamp_bounds() -> None:
    """With ap_position=p, offset is clamped to [-(rt_size-1-p), p].

    Drive rt_error=1.0 so every access-port read faults; force the direction by
    NOT relying on the draw — instead run many reads and assert the offset never
    leaves the asymmetric window, and that both bounds are reachable across a
    population of racetracks.
    """
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from netdrift.faults.kernels.rtm_numba import calc_index_offset_kernel

    rt_size = 8
    p = rt_size // 2 - 1          # 3
    lo = -(rt_size - 1 - p)       # -4
    hi = p                        # 3
    n = 256
    offset = np.zeros((n, 1), dtype=np.int32)
    misalign = np.zeros((1, 1), dtype=np.int32)  # tracking off
    ap_reads = rt_size * rt_size
    rng = create_xoroshiro128p_states(n, seed=3)

    d_off = cuda.to_device(offset)
    calc_index_offset_kernel[(n, 1), (min(n, 32), 1)](
        rng, d_off, cuda.to_device(misalign), rt_size, ap_reads, 1.0, p, 1,
    )
    cuda.synchronize()
    out = d_off.copy_to_host()
    assert out.min() >= lo, f"offset {out.min()} below lo={lo}"
    assert out.max() <= hi, f"offset {out.max()} above hi={hi}"
    # With rt_error=1.0 and 64 reads, the random walk should reach both bounds
    # somewhere in a population of 256 racetracks.
    assert out.min() == lo, f"lo bound {lo} never reached (min={out.min()})"
    assert out.max() == hi, f"hi bound {hi} never reached (max={out.max()})"


@pytest.mark.cuda
def test_calc_recovers_from_saturation() -> None:
    """A racetrack pre-saturated at +hi must be able to move back toward zero.

    Seed offset at hi=p, then run the calc kernel: because the direction is
    drawn *before* the room check, some opposite draws will decrement it, so the
    minimum observed offset over a population must drop below hi.
    """
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from netdrift.faults.kernels.rtm_numba import calc_index_offset_kernel

    rt_size = 8
    p = rt_size // 2 - 1  # 3 == hi
    n = 256
    offset = np.full((n, 1), p, dtype=np.int32)  # start saturated at +hi
    misalign = np.zeros((1, 1), dtype=np.int32)
    ap_reads = rt_size * rt_size
    rng = create_xoroshiro128p_states(n, seed=5)

    d_off = cuda.to_device(offset)
    calc_index_offset_kernel[(n, 1), (min(n, 32), 1)](
        rng, d_off, cuda.to_device(misalign), rt_size, ap_reads, 1.0, p, 1,
    )
    cuda.synchronize()
    out = d_off.copy_to_host()
    assert out.max() <= p, "must not exceed hi"
    assert out.min() < p, "a saturated racetrack must be able to recover downward"


@pytest.mark.cuda
def test_calc_saturated_still_counts_misalign_faults() -> None:
    """A drawn fault in the already-saturated direction still counts.

    Start every racetrack at +hi and force rt_error=1.0. Even reads whose draw
    is blocked by the clamp must increment misalign_faults (a fault event
    occurred; only the physical shift is suppressed).
    """
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from netdrift.faults.kernels.rtm_numba import calc_index_offset_kernel

    rt_size = 8
    p = rt_size // 2 - 1
    n = 8
    offset = np.full((n, 1), p, dtype=np.int32)
    misalign = np.zeros((n, 1), dtype=np.int32)  # tracking ON (shape matches)
    ap_reads = rt_size * rt_size  # 64 -> 63 reads, all fault at rt_error=1.0
    rng = create_xoroshiro128p_states(n, seed=2)

    d_mis = cuda.to_device(misalign)
    calc_index_offset_kernel[(n, 1), (min(n, 32), 1)](
        rng, cuda.to_device(offset), d_mis, rt_size, ap_reads, 1.0, p, 1,
    )
    cuda.synchronize()
    mis = d_mis.copy_to_host()
    # Every one of the 63 reads faults regardless of whether the wire moved.
    assert np.all(mis == ap_reads - 1), f"expected {ap_reads-1} faults each, got {mis.ravel()}"


# ---------------------------------------------------------------------------
# Fault-model level: identity, ragged-tail replication, BLOCK immunity.
# ---------------------------------------------------------------------------

@pytest.mark.cuda
def test_saturate_zero_error_identity() -> None:
    cfg = RTMConfig(rt_size=64, rt_error=0.0, edge_mode="saturate", track_bitflips=True)
    fault = RTMMisalignmentFault(cfg)
    weight = torch.where(torch.randn(64, 128, device="cuda") > 0,
                         torch.ones(64, 128, device="cuda"),
                         -torch.ones(64, 128, device="cuda"))
    ctx = _make_ctx(rt_mapping="ROW")
    state = fault.init_state(tuple(weight.shape), ctx)
    new_w, _, stats = fault.inject(weight, state, ctx)
    assert torch.equal(new_w, weight)
    assert stats.bitflips == 0


@pytest.mark.cuda
def test_saturate_ragged_tail_replicates_edge_value() -> None:
    """A sub-rt_size layer (width 27 with rt_size 64) is one partial racetrack.

    Every real cell is +1; under saturate mode a shifted read into the ragged
    tail (positions 27..63) replicates the nearest real cell (+1), so the output
    is unchanged even at a large error rate. A random-mode read would flip some.
    """
    cfg = RTMConfig(rt_size=64, rt_error=0.5, edge_mode="saturate",
                    track_bitflips=True)
    fault = RTMMisalignmentFault(cfg)
    # 8 output channels, in=3, k=3 -> reshaped width 27 < 64 (single partial rt).
    weight = torch.ones(8, 3, 3, 3, device="cuda")
    ctx = _make_ctx(rt_mapping="ROW", kernel_mapping="ROW", kernel_size=3)
    state = fault.init_state(tuple(weight.shape), ctx)
    new_w, _, stats = fault.inject(weight, state, ctx)
    # All weights identical (+1): saturation can only ever return +1.
    assert torch.equal(new_w, weight), "ragged tail must replicate the edge value"
    assert stats.bitflips == 0


@pytest.mark.cuda
def test_block_saturate_is_fault_immune() -> None:
    """BLOCK + saturate: same-sign blocks + saturating reads => zero bitflips
    at any rt_error (documented, intended behaviour)."""
    cfg = RTMConfig(rt_size=64, rt_error=0.5, edge_mode="saturate",
                    block_mapping=True, track_bitflips=True)
    fault = RTMMisalignmentFault(cfg)
    weight = torch.where(torch.randn(64, 128, device="cuda") > 0,
                         torch.ones(64, 128, device="cuda"),
                         -torch.ones(64, 128, device="cuda"))
    ctx = _make_ctx(rt_mapping="BLOCK", base_layout="row")
    state = fault.init_state(tuple(weight.shape), ctx)
    new_w, _, stats = fault.inject(weight, state, ctx)
    assert torch.equal(new_w, weight), "BLOCK+saturate must be fault-immune"
    assert stats.bitflips == 0


@pytest.mark.cuda
def test_block_random_still_faults() -> None:
    """The random mode remains available for a non-trivial BLOCK curve."""
    cfg = RTMConfig(rt_size=64, rt_error=0.5, edge_mode="random",
                    block_mapping=True, track_bitflips=True)
    fault = RTMMisalignmentFault(cfg)
    weight = torch.where(torch.randn(64, 128, device="cuda") > 0,
                         torch.ones(64, 128, device="cuda"),
                         -torch.ones(64, 128, device="cuda"))
    ctx = _make_ctx(rt_mapping="BLOCK", base_layout="row")
    state = fault.init_state(tuple(weight.shape), ctx)
    new_w, _, stats = fault.inject(weight, state, ctx)
    assert not torch.equal(new_w, weight), "BLOCK+random should still corrupt"
    assert stats.bitflips is not None and stats.bitflips > 0
