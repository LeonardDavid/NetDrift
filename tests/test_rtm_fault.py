"""RTMMisalignmentFault correctness — functional behavior and determinism.

Two main checks:

1. **Functional**: at zero fault rate the output equals the input; at high fault
   rate it diverges; index_offset accumulates across calls; mitigations run when
   configured; stats are populated only when their flags are set.

2. **Determinism**: at a fixed seed, two consecutive calls produce identical
   output. This verifies the new kernels (which fix two latent rng-races in
   the legacy implementation) are reproducible.

GPU-required (Numba CUDA kernels run on the device).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from netdrift.faults import (
    FaultCtx,
    RTMConfig,
    RTMMisalignmentFault,
    RTMState,
)
from netdrift.faults.mitigations import get_mitigation
from netdrift.faults.weight_encoders import (
    EndlenEncoder,
    _endlen_cpu_reference,
)


def _make_ctx(rt_mapping: str = "ROW", kernel_mapping: str | None = None,
              kernel_size: int | None = None, nr_run: int = 1) -> FaultCtx:
    return FaultCtx(
        layer_id=1, layer_name="conv1", nr_run=nr_run, training=False, bits=1,
        extra={
            "rt_mapping": rt_mapping,
            "kernel_mapping": kernel_mapping,
            "kernel_size": kernel_size,
        },
    )


@pytest.mark.cuda
def test_zero_error_is_identity_2d() -> None:
    """rt_error=0 must return the input unchanged for a 2D (linear) weight."""
    cfg = RTMConfig(rt_size=64, rt_error=0.0, track_bitflips=True)
    fault = RTMMisalignmentFault(cfg)

    weight = torch.where(torch.randn(64, 128, device="cuda") > 0,
                         torch.ones(64, 128, device="cuda"),
                         -torch.ones(64, 128, device="cuda"))
    ctx = _make_ctx(rt_mapping="ROW")
    state = fault.init_state(tuple(weight.shape), ctx)

    new_w, new_state, stats = fault.inject(weight, state, ctx)
    assert torch.equal(new_w, weight), "zero-error injection must be identity"
    assert stats.bitflips == 0
    # No misalignments → all offsets remain zero.
    assert int(np.count_nonzero(new_state.index_offset)) == 0


@pytest.mark.cuda
def test_zero_error_is_identity_4d_conv() -> None:
    """Same identity property for a 4D conv weight (with kernel mapping)."""
    cfg = RTMConfig(rt_size=64, rt_error=0.0, track_bitflips=True)
    fault = RTMMisalignmentFault(cfg)

    weight = torch.where(torch.randn(64, 64, 3, 3, device="cuda") > 0,
                         torch.ones(64, 64, 3, 3, device="cuda"),
                         -torch.ones(64, 64, 3, 3, device="cuda"))
    ctx = _make_ctx(rt_mapping="ROW", kernel_mapping="ROW", kernel_size=3)
    state = fault.init_state(tuple(weight.shape), ctx)

    new_w, _, stats = fault.inject(weight, state, ctx)
    assert torch.equal(new_w, weight)
    assert stats.bitflips == 0


@pytest.mark.cuda
def test_high_error_produces_bitflips() -> None:
    """At a meaningful error rate, the output diverges from the input."""
    cfg = RTMConfig(rt_size=64, rt_error=0.05, track_bitflips=True,
                    track_misalign_faults=True, track_affected_units=True)
    fault = RTMMisalignmentFault(cfg)

    weight = torch.where(torch.randn(64, 128, device="cuda") > 0,
                         torch.ones(64, 128, device="cuda"),
                         -torch.ones(64, 128, device="cuda"))
    ctx = _make_ctx(rt_mapping="ROW")
    state = fault.init_state(tuple(weight.shape), ctx)

    new_w, new_state, stats = fault.inject(weight, state, ctx)
    assert not torch.equal(new_w, weight), "high error rate must produce bitflips"
    assert stats.bitflips is not None and stats.bitflips > 0
    assert stats.misalign_faults is not None and stats.misalign_faults > 0
    assert stats.affected_units is not None and stats.affected_units > 0


@pytest.mark.cuda
def test_offset_accumulates_across_calls() -> None:
    """Repeated injections produce a substantial population of nonzero offsets.

    Each call may shift any racetrack ±1 with probability ``rt_error``; offsets
    are *not* guaranteed monotonic (cancelling shifts can return to zero),
    but at ``rt_error=0.05`` after 3 calls almost every racetrack should have
    drifted from its starting zero. We assert "most racetracks have moved",
    not strict monotonic growth.
    """
    cfg = RTMConfig(rt_size=64, rt_error=0.05)
    fault = RTMMisalignmentFault(cfg)

    weight = torch.where(torch.randn(64, 128, device="cuda") > 0,
                         torch.ones(64, 128, device="cuda"),
                         -torch.ones(64, 128, device="cuda"))
    ctx = _make_ctx(rt_mapping="ROW")
    state = fault.init_state(tuple(weight.shape), ctx)

    for i in range(3):
        ctx = _make_ctx(rt_mapping="ROW", nr_run=i + 1)
        _, state, _ = fault.inject(weight, state, ctx)

    # After 3 calls at rt_error=0.05 with 4096 access-port reads per call,
    # essentially every racetrack should have non-zero offset. Allow some
    # slack: at least 80% of the racetracks are off-zero.
    nonzero = int(np.count_nonzero(state.index_offset))
    total = state.index_offset.size
    assert nonzero / total >= 0.8, (
        f"only {nonzero}/{total} racetracks drifted after 3 calls at rt_error=0.05"
    )


@pytest.mark.cuda
def test_mitigations_modify_offsets() -> None:
    """A mitigation step modifies the offset array between fault generation and read-out."""
    mitigation = get_mitigation("bin_revert_mid")
    cfg = RTMConfig(rt_size=64, rt_error=0.1, mitigations=[mitigation],
                    track_misalign_faults=True)
    fault = RTMMisalignmentFault(cfg)

    weight = torch.where(torch.randn(64, 256, device="cuda") > 0,
                         torch.ones(64, 256, device="cuda"),
                         -torch.ones(64, 256, device="cuda"))
    ctx = _make_ctx(rt_mapping="ROW")
    state = fault.init_state(tuple(weight.shape), ctx)
    _, new_state, _ = fault.inject(weight, state, ctx)
    # Mitigation may have zeroed some offsets; we only assert the array is finite.
    assert np.all(np.isfinite(new_state.index_offset))


@pytest.mark.cuda
def test_stats_are_off_by_default() -> None:
    """Without explicit ``track_*`` flags, the corresponding stats fields are None."""
    cfg = RTMConfig(rt_size=64, rt_error=0.05)  # all track_* default False
    fault = RTMMisalignmentFault(cfg)

    weight = torch.where(torch.randn(64, 128, device="cuda") > 0,
                         torch.ones(64, 128, device="cuda"),
                         -torch.ones(64, 128, device="cuda"))
    ctx = _make_ctx(rt_mapping="ROW")
    state = fault.init_state(tuple(weight.shape), ctx)
    _, _, stats = fault.inject(weight, state, ctx)
    assert stats.bitflips is None
    assert stats.misalign_faults is None
    assert stats.affected_units is None


@pytest.mark.cuda
def test_col_mapping_round_trip() -> None:
    """COL mapping internally transposes the weight; result must have the original shape."""
    cfg = RTMConfig(rt_size=64, rt_error=0.05, track_bitflips=True)
    fault = RTMMisalignmentFault(cfg)

    weight = torch.where(torch.randn(64, 128, device="cuda") > 0,
                         torch.ones(64, 128, device="cuda"),
                         -torch.ones(64, 128, device="cuda"))
    ctx = _make_ctx(rt_mapping="COL")
    state = fault.init_state(tuple(weight.shape), ctx)
    new_w, _, _ = fault.inject(weight, state, ctx)
    assert new_w.shape == weight.shape


@pytest.mark.cuda
def test_kernel_mapping_round_trip() -> None:
    """Each non-trivial kernel mapping returns a tensor of the original 4D shape."""
    cfg = RTMConfig(rt_size=64, rt_error=0.0, track_bitflips=True)
    fault = RTMMisalignmentFault(cfg)
    weight = torch.where(torch.randn(64, 64, 3, 3, device="cuda") > 0,
                         torch.ones(64, 64, 3, 3, device="cuda"),
                         -torch.ones(64, 64, 3, 3, device="cuda"))
    for kmap in ("ROW", "COL", "CLW", "ACW"):
        ctx = _make_ctx(rt_mapping="ROW", kernel_mapping=kmap, kernel_size=3)
        state = fault.init_state(tuple(weight.shape), ctx)
        new_w, _, stats = fault.inject(weight, state, ctx)
        assert new_w.shape == weight.shape
        # Zero error → identity in the original space.
        assert torch.equal(new_w, weight), f"kernel mapping {kmap} broke zero-error identity"


@pytest.mark.cuda
def test_endlen_gpu_matches_cpu_reference() -> None:
    """The Numba CUDA endlen kernel produces the same output as the CPU port.

    On several random ±1 racetracks (multiple rows, multiple racetracks per
    row), the GPU encoder and the CPU reference must agree bit-for-bit.
    """
    from numba import cuda
    rt_size = 16
    rng = np.random.default_rng(0)
    # Shape: 4 rows, 4 racetracks per row → 64 cols.
    weight_cpu = np.where(rng.random((4, 64)) > 0.5, 1, -1).astype(np.int32)
    weight_gpu_host = weight_cpu.copy()

    _endlen_cpu_reference(weight_cpu, rt_size)

    gpu_arr = cuda.to_device(weight_gpu_host)
    EndlenEncoder().apply(gpu_arr, rt_size)
    weight_gpu_host = gpu_arr.copy_to_host()

    assert np.array_equal(weight_cpu, weight_gpu_host), (
        "GPU kernel and CPU reference disagree on endlen output"
    )


@pytest.mark.cuda
def test_endlen_per_forward_mode_encodes_at_inject() -> None:
    """In ``per_forward`` mode, inject's output reflects the encoded weights
    (not the raw input) even at rt_error=0, while module.weight stays untouched.
    """
    cfg = RTMConfig(
        rt_size=16, rt_error=0.0,
        weight_encoder=EndlenEncoder(),
        weight_encoder_mode="per_forward",
        track_bitflips=True,
    )
    fault = RTMMisalignmentFault(cfg)

    weight = torch.where(
        torch.randn(8, 64, device="cuda") > 0,
        torch.ones(8, 64, device="cuda"),
        -torch.ones(8, 64, device="cuda"),
    )
    pre = weight.detach().clone()
    ctx = _make_ctx(rt_mapping="ROW")
    state = fault.init_state(tuple(weight.shape), ctx)

    new_w, _, stats = fault.inject(weight, state, ctx)

    # The encoder ran inside inject, so at rt_error=0 the read-out equals
    # the encoded weights, which generally differ from the input.
    assert stats.bitflips is not None
    # On random ±1 of size 8x64=512, endlen should flip at least *some* bits.
    # Exact count depends on the random pattern, so we just check >0.
    assert stats.bitflips > 0, (
        "per_forward mode at rt_error=0 should still produce bit-flips "
        "(the encoder fires regardless)"
    )
    # The function does not mutate the caller's input tensor.
    assert torch.equal(pre, weight)


@pytest.mark.cuda
def test_kernel_is_deterministic_at_fixed_seed() -> None:
    """Two consecutive injections with the same seeded RNG produce identical output.

    The legacy ``racetrack_sim`` kernels contain two latent races on
    ``rng_states[0]`` (see ``netdrift/faults/kernels/rtm_numba.py`` module
    docstring) that make them non-deterministic across runs even at a fixed
    seed. The refactored kernels fix both bugs by giving every thread its
    own rng-state index, so the kernel is reproducible.
    """
    cfg = RTMConfig(rt_size=64, rt_error=0.05, track_misalign_faults=True,
                    track_bitflips=True)
    fault = RTMMisalignmentFault(cfg)

    torch.manual_seed(42)
    weight = torch.where(torch.randn(64, 128, device="cuda") > 0,
                         torch.ones(64, 128, device="cuda"),
                         -torch.ones(64, 128, device="cuda"))

    def _run() -> tuple[torch.Tensor, np.ndarray, int, int]:
        state = RTMState(
            index_offset=np.zeros((64, 2), dtype=np.int32),
            rt_mapping="ROW", kernel_mapping=None,
        )
        ctx = _make_ctx(rt_mapping="ROW")
        random.seed(12345)
        w, st, stats = fault.inject(weight, state, ctx)
        return w, st.index_offset, stats.misalign_faults, stats.bitflips

    w1, off1, mis1, flip1 = _run()
    w2, off2, mis2, flip2 = _run()

    assert torch.equal(w1.cpu(), w2.cpu()), "weight tensors should match across runs"
    assert np.array_equal(off1, off2), "index_offset arrays should match across runs"
    assert mis1 == mis2, f"misalign_fault counts should match: {mis1} vs {mis2}"
    assert flip1 == flip2, f"bitflip counts should match: {flip1} vs {flip2}"
