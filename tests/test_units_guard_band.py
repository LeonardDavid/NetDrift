# tests/test_units_guard_band.py
"""Validation gate 4: what the period-p guard band does and does not buy.

A period-p wire reads position k correctly iff off = 0 (mod p) AND no clamping
occurs. When k < off the index clamps to 0 and returns s[0], correct only when
k = 0 (mod p). At p=1 every cell shares a sign so clamping is harmless and the
wire is fully immune -- that is BLOCK. At p=2 the clamped prefix contributes
about off/2 wrong bits EVEN AT EVEN OFFSETS.
"""
import numpy as np
import pytest


def _read(seq, off, P):
    """Mirror simulate_racetrack_kernel's saturate branch for one wire."""
    out = []
    for k in range(P):
        idx = k - off
        idx = 0 if idx < 0 else (P - 1 if idx > P - 1 else idx)
        out.append(seq[idx])
    return out


def test_period1_wire_is_immune_at_every_offset():
    P = 8
    seq = [1] * P                      # a same-sign run plus its guard band
    for off in range(-(P - 1), P):
        assert _read(seq, off, P) == seq, off


def test_period2_wire_is_not_immune_at_even_offsets():
    P = 8
    seq = [1 if j % 2 == 0 else -1 for j in range(P)]
    # off=0 is exact
    assert _read(seq, 0, P) == seq
    # even off > 0: the clamped prefix k < off is half wrong
    for off in (2, 4, 6):
        got = _read(seq, off, P)
        wrong = [k for k in range(P) if got[k] != seq[k]]
        assert wrong, f"expected clamp-induced errors at off={off}"
        assert all(k < off for k in wrong), (off, wrong)
        assert len(wrong) == off // 2, (off, wrong)


def test_period2_wire_interior_is_wrong_at_odd_offsets():
    P = 8
    seq = [1 if j % 2 == 0 else -1 for j in range(P)]
    for off in (1, 3, 5):
        got = _read(seq, off, P)
        wrong = [k for k in range(P) if got[k] != seq[k]]
        # every unclamped position (k >= off) is wrong at odd offset
        assert all(k in wrong for k in range(off, P)), (off, wrong)


def test_negative_offsets_clamp_at_the_far_end():
    P = 8
    seq = [1] * P
    assert _read(seq, -3, P) == seq          # period 1: still immune
    alt = [1 if j % 2 == 0 else -1 for j in range(P)]
    got = _read(alt, -2, P)
    wrong = [k for k in range(P) if got[k] != alt[k]]
    assert all(k >= P - 2 for k in wrong), wrong


@pytest.mark.cuda
@pytest.mark.parametrize("seq_kind,off", [
    ("period1", 1), ("period1", 3), ("period1", -2),
    ("period2", 0), ("period2", 2), ("period2", 3), ("period2", -2),
])
def test_kernel_matches_the_reference_read(seq_kind, off):
    """The Python mirror above must match the real CUDA kernel.

    Drives simulate_racetrack_kernel directly with a pre-seeded offset, exactly
    as tests/test_ap_saturate.py does, so no fault-rate randomness is involved.
    """
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states

    from netdrift.faults.kernels.rtm_numba import simulate_racetrack_kernel

    P = 8
    if seq_kind == "period1":
        seq = [1] * P                                  # isolated run + guard band
    else:
        seq = [1 if j % 2 == 0 else -1 for j in range(P)]   # guarded period-2 pool

    qin = np.array([[float(v) for v in seq]], dtype=np.float32)
    qout = np.zeros_like(qin)
    offset = np.array([[off]], dtype=np.int32)
    wrong = np.zeros_like(offset)
    rng = create_xoroshiro128p_states(1, seed=1)

    d_out = cuda.to_device(qout)
    d_wrong = cuda.to_device(wrong)
    simulate_racetrack_kernel[(1, 1), (1, 1)](
        rng, cuda.to_device(qin), d_out, cuda.to_device(offset), P,
        d_wrong, 1, 1,  # track_wrong=1, edge_mode=1 (saturate)
    )
    cuda.synchronize()

    expected = np.array([[float(v) for v in _read(seq, off, P)]], dtype=np.float32)
    assert np.array_equal(d_out.copy_to_host(), expected), (seq_kind, off)


@pytest.mark.cuda
def test_period1_wire_is_immune_on_the_real_kernel():
    """The claim BLOCK rests on, asserted against the kernel rather than a mirror."""
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states

    from netdrift.faults.kernels.rtm_numba import simulate_racetrack_kernel

    P = 8
    qin = np.ones((1, P), dtype=np.float32)
    rng = create_xoroshiro128p_states(1, seed=1)
    for off in range(-(P - 1), P):
        d_out = cuda.to_device(np.zeros_like(qin))
        d_wrong = cuda.to_device(np.zeros((1, 1), dtype=np.int32))
        simulate_racetrack_kernel[(1, 1), (1, 1)](
            rng, cuda.to_device(qin), d_out,
            cuda.to_device(np.array([[off]], dtype=np.int32)), P, d_wrong, 1, 1,
        )
        cuda.synchronize()
        assert np.array_equal(d_out.copy_to_host(), qin), off
