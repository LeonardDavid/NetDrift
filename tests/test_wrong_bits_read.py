"""Kernel wrong-read counter: correctness + numeric invariance."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.cuda
def test_wrong_read_counts_nonidentity_reads():
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from netdrift.faults.kernels.rtm_numba import simulate_racetrack_kernel

    rt_size = 4
    # One racetrack lane (1 row, 4 cols), forced offset of +1 -> every read shifts.
    # NOTE: the offset grid is (1,1) here — counting must still work because the
    # kernel gates on the explicit track_wrong flag, not the array shape.
    qin = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    qout = np.zeros_like(qin)
    offset = np.array([[1]], dtype=np.int32)
    rng = create_xoroshiro128p_states(1, seed=1)

    wrong_full = np.zeros_like(offset)
    d_wrong = cuda.to_device(wrong_full)
    simulate_racetrack_kernel[(1, 1), (1, 1)](
        rng, cuda.to_device(qin), cuda.to_device(qout),
        cuda.to_device(offset), rt_size, d_wrong, 1, 0,  # track_wrong=1, edge_mode=0 (random)
    )
    cuda.synchronize()
    # With offset=+1, positions k=0..3 map to in_index k-1: k=0 reads OOB (left),
    # k=1,2,3 read neighbours — all 4 reads are non-identity (q_in_index != q_out_index).
    # Asserted under edge_mode=0 (legacy random): the OOB k=0 read still counts.
    assert d_wrong.copy_to_host().sum() == 4


@pytest.mark.cuda
def test_track_wrong_flag_off_skips_counting():
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from netdrift.faults.kernels.rtm_numba import simulate_racetrack_kernel

    rt_size = 4
    qin = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    qout = np.zeros_like(qin)
    offset = np.array([[1]], dtype=np.int32)  # non-zero -> would count if enabled
    wrong = np.zeros((1, 1), dtype=np.int32)
    rng = create_xoroshiro128p_states(1, seed=1)

    d_wrong = cuda.to_device(wrong)
    simulate_racetrack_kernel[(1, 1), (1, 1)](
        rng, cuda.to_device(qin), cuda.to_device(qout),
        cuda.to_device(offset), rt_size, d_wrong, 0, 0,  # track_wrong=0, edge_mode=0
    )
    cuda.synchronize()
    assert d_wrong.copy_to_host().sum() == 0  # disabled despite non-zero offset


@pytest.mark.cuda
def test_zero_offset_yields_zero_wrong_reads_and_identity_output():
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from netdrift.faults.kernels.rtm_numba import simulate_racetrack_kernel

    rt_size = 4
    qin = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    qout = np.zeros_like(qin)
    offset = np.array([[0]], dtype=np.int32)
    wrong_full = np.zeros_like(offset)
    rng = create_xoroshiro128p_states(1, seed=1)

    d_out = cuda.to_device(qout)
    d_wrong = cuda.to_device(wrong_full)
    simulate_racetrack_kernel[(1, 1), (1, 1)](
        rng, cuda.to_device(qin), d_out, cuda.to_device(offset), rt_size,
        d_wrong, 1, 0,  # track_wrong=1, edge_mode=0 -> genuinely zero because offset is 0
    )
    cuda.synchronize()
    assert d_wrong.copy_to_host().sum() == 0
    assert np.array_equal(d_out.copy_to_host(), qin)  # output identical to input
