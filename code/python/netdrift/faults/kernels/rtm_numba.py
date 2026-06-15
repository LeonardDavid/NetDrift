"""Numba CUDA kernels for racetrack-memory misalignment simulation.

Two kernels:

* :func:`calc_index_offset_kernel` — generate per-racetrack misalignment
  offsets stochastically, one thread per racetrack.
* :func:`simulate_racetrack_kernel` — read the (shifted) value at each
  racetrack position, returning random ±1 for out-of-bounds reads.

Kept Numba-based intentionally: the racetrack simulation does not participate
in autograd (weights are detached before injection), so JIT-compiled device
code is the simplest path. New fault models that need autograd participation
should use PyTorch C++ extensions instead — see ``code/cuda/`` for examples.

**Bugfixes vs. the legacy ``code/python/legacy_cuda/racetrack.py``**

Two latent races in the legacy implementation are fixed here, so the new
kernels are deterministic at a fixed seed.

1. *Direction draw race.* The legacy passed ``rand`` (a ``float32`` in
   ``[0, 1)``) as the *index* argument to the second
   ``xoroshiro128p_uniform_float32`` call. Numba truncates floats to int,
   which collapses every thread's direction draw onto ``rng_states[0]`` —
   producing a write-write race across all racetracks and non-deterministic
   output. Fixed by reusing the per-thread state index ``i*shape[1]+j``.

2. *Out-of-bounds read race.* The legacy used ``q_out_index`` (which
   collides across rows of racetracks at the same column) as the rng index
   in :func:`simulate_racetrack_kernel`. Multiple threads at the same
   column then race on the same RNG state. Fixed by deriving a unique index
   per ``(i, j, k)`` triple.

These changes are intentional behavioral changes. Aggregate statistics
(mean misalignment count, mean bitflip count) should remain very close to
the legacy under matched seeds; results just become reproducible.
"""

from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


@cuda.jit
def calc_index_offset_kernel(rng_states, index_offset, misalign_faults, rt_size, ap_reads, rt_error):  # noqa: E501
    """Generate per-racetrack misalignment offsets.

    For each racetrack ``(i, j)``, simulate ``ap_reads - 1`` access-port
    reads. Each read may incur a misalignment fault with probability
    ``rt_error``; on a fault the offset shifts ±1 (50/50) up to the buffer
    bound ``rt_size/2``.

    Args:
        rng_states:      Pre-seeded xoroshiro128p PRNG states (one per thread).
        index_offset:    2D int32 array, in/out, shape ``(num_racetracks_x, num_racetracks_y)``.
        misalign_faults: 2D int array. If shape matches ``index_offset``, per-racetrack fault counts are accumulated; otherwise (e.g. shape ``(1,1)``) the kernel skips the bookkeeping.
        rt_size:         Bits per racetrack.
        ap_reads:        Number of access-port reads simulated (``rt_size`` for COL mapping, ``rt_size**2`` for ROW).
        rt_error:        Per-read fault probability in ``[0, 1]``.

    Notes:
        Each thread uses a single rng-state index (``i*shape[1] + j``) for
        both the misalignment-probability draw and the direction draw. This
        makes the kernel deterministic per seed (legacy was not — see module docstring).
    """
    i, j = cuda.grid(2)

    if i < index_offset.shape[0] and j < index_offset.shape[1]:
        rng_idx = i * index_offset.shape[1] + j
        for k in range(1, ap_reads):
            rand = xoroshiro128p_uniform_float32(rng_states, rng_idx)
            if rand < rt_error:
                if misalign_faults.shape[0] > 1:  # i.e. CALC_MISALIGN_FAULTS-equivalent
                    misalign_faults[i, j] += 1

                if abs(index_offset[i, j]) < rt_size / 2:
                    rand2 = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                    if rand2 > 0.5:
                        index_offset[i, j] += 1
                    else:
                        index_offset[i, j] -= 1


@cuda.jit
def simulate_racetrack_kernel(rng_states, qweight_in, qweight_out, index_offset, rt_size, wrong_read, track_wrong):  # noqa: E501
    """Read each racetrack at its (shifted) position.

    For every racetrack ``(i, j)`` and every position ``k`` in ``[0, rt_size)``,
    fetch the value at ``j*rt_size + k - index_offset[i, j]``. Reads that fall
    outside the racetrack's own ``[j*rt_size, (j+1)*rt_size)`` slot return a
    random ±1 value, modelling reading from "buffer" positions outside the
    valid wire window.

    Args:
        rng_states:   xoroshiro128p PRNG states.
        qweight_in:   2D float input weights, contiguous, shape ``(_, n_cols)``.
        qweight_out:  2D float output weights, same shape.
        index_offset: 2D int32 offset array, shape compatible with the racetrack grid.
        rt_size:      Bits per racetrack.
        wrong_read:   2D int array, same shape as ``index_offset`` when tracking,
                      else a ``(1, 1)`` dummy. Per-racetrack counts of non-identity
                      reads (``q_in_index != q_out_index``) are accumulated when
                      ``track_wrong`` is truthy.
        track_wrong:  ``1`` to accumulate per-racetrack wrong-read counts, ``0`` to
                      skip. Passed explicitly (not inferred from ``wrong_read``'s
                      shape) so a genuine ``(1, 1)`` racetrack grid is still counted.

    Notes:
        Each thread uses its own rng-state index (``i*shape[1] + j``) for
        all out-of-bounds draws. This makes the kernel deterministic per
        seed (legacy collided across rows on the same column — see module docstring).
    """
    i, j = cuda.grid(2)

    if i < index_offset.shape[0] and j < index_offset.shape[1]:
        rng_idx = i * index_offset.shape[1] + j
        for k in range(rt_size):
            q_out_index = j * rt_size + k
            q_in_index = q_out_index - index_offset[i, j]

            if q_out_index < qweight_out.shape[1]:
                if track_wrong and q_in_index != q_out_index:
                    wrong_read[i, j] += 1
                if j * rt_size <= q_in_index <= (j + 1) * rt_size - 1 and q_in_index < qweight_in.shape[1]:  # noqa: E501
                    qweight_out[i, q_out_index] = qweight_in[i, q_in_index]
                else:
                    rand = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                    qweight_out[i, q_out_index] = 1 if rand > 0.5 else -1
