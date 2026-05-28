"""Endlen (block-hypothesis) write-time weight encoder.

Ported from ``metrics/blockhyp_endlen/blockhyp_endlen.py``. The algorithm
walks each racetrack with a sliding 3-window over bitgroups (maximal runs of
identical sign), emits a tuple per window scoring it by

    endlen = len(L) + len(M) + len(R)        # combined run if M is flipped
    flips  = len(M)                          # bits to flip to merge L and R

sorts tuples by (-endlen, +flips), and greedily flips the middle bitgroup M
in place. Picking a tuple invalidates its two neighbors (their windows
overlap M), so only non-overlapping merges are applied.

The CUDA kernel and the CPU reference are line-for-line ports of the legacy
versions, with the same hardcoded 64-element tuple buffer. They are kept
together so the GPU/CPU parity test can compare them on small inputs.
"""

from __future__ import annotations

import math

import numba
import numpy as np
from numba import cuda

from netdrift.faults.weight_encoders.base import WeightEncoder, register_encoder


_MAX_RT_SIZE = 64
"""Hardcoded local-array size in the legacy kernel. rt_size > this is rejected."""


@cuda.jit
def _endlen_kernel(qweight, rt_size):
    """Endlen encoder, 1 CUDA thread per (row, racetrack-block) cell.

    Mutates ``qweight`` in place. Direct port of
    ``blockhyp_endlen_algorithm_parallel_kernel``.
    """
    rt_i, rt_j = cuda.grid(2)

    if rt_i < qweight.shape[0] and rt_j < qweight.shape[1] // rt_size:
        w_i = rt_i

        count = 0  # number of sign changes seen so far
        k = 0      # index of the next tuple to emit
        tuple_counts = 0
        current_sign = qweight[w_i, rt_j * rt_size]

        rt_tuples = cuda.local.array((_MAX_RT_SIZE, 4), dtype=numba.int64)
        bitgroup = cuda.local.array(3, dtype=numba.int32)
        j_mid = cuda.local.array(3, dtype=numba.int32)

        for x in range(rt_size):
            for y in range(4):
                rt_tuples[x, y] = 0

        for idx in range(3):
            bitgroup[idx] = 0
            j_mid[idx] = 0

        for bit_index in range(rt_size):
            w_j = rt_j * rt_size + bit_index

            if w_j < qweight.shape[1] and qweight[w_i, w_j] != current_sign:
                j_mid[(count + 1) % 3] = w_j
                current_sign = -current_sign
                count += 1

                if count > 2:
                    if k < rt_size:
                        rt_tuples[k, 0] = k
                        rt_tuples[k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
                        rt_tuples[k, 2] = bitgroup[(k + 1) % 3]
                        rt_tuples[k, 3] = j_mid[(k + 1) % 3]
                        k += 1

                    bitgroup[count % 3] = 0

            if w_j < qweight.shape[1]:
                bitgroup[count % 3] += 1

        if k < rt_size:
            rt_tuples[k, 0] = k
            rt_tuples[k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
            rt_tuples[k, 2] = bitgroup[(k + 1) % 3]
            rt_tuples[k, 3] = j_mid[(k + 1) % 3]
            tuple_counts = k + 1

        # Bubble sort by (-endlen, +flips). Naive but bounded by rt_size <= 64.
        for _ in range(tuple_counts):
            for id in range(tuple_counts - 1):
                endlen_j = rt_tuples[id, 1]
                flips_j = rt_tuples[id, 2]
                endlen_j1 = rt_tuples[id + 1, 1]
                flips_j1 = rt_tuples[id + 1, 2]
                if (endlen_j < endlen_j1) or (
                    endlen_j == endlen_j1 and flips_j > flips_j1
                ):
                    for k in range(4):
                        temp = rt_tuples[id, k]
                        rt_tuples[id, k] = rt_tuples[id + 1, k]
                        rt_tuples[id + 1, k] = temp

        # Greedy: pick top tuple, flip its middle bitgroup, null self+neighbors.
        processed = 0
        while True:
            if processed >= tuple_counts:
                break
            idx_tuple = processed

            tuple_index = rt_tuples[idx_tuple, 0]
            endlen = rt_tuples[idx_tuple, 1]
            flips = rt_tuples[idx_tuple, 2]
            start_index_mid = rt_tuples[idx_tuple, 3]

            if endlen == 0:
                processed += 1
                continue

            final_index_mid = start_index_mid + flips
            for w_jx in range(start_index_mid, final_index_mid):
                if w_jx < qweight.shape[1]:
                    qweight[w_i, w_jx] = -qweight[w_i, w_jx]

            for id in range(tuple_counts):
                tj_index = rt_tuples[id, 0]
                if (
                    tj_index == tuple_index - 1
                    or tj_index == tuple_index
                    or tj_index == tuple_index + 1
                ):
                    rt_tuples[id, 1] = 0

            processed += 1


def _endlen_cpu_reference(weight_2d: np.ndarray, rt_size: int) -> None:
    """CPU reference port of ``blockhyp_endlen_algorithm_cpu_old``.

    Mutates ``weight_2d`` in place. Used by tests as the algorithmic ground
    truth; not invoked at runtime.

    Args:
        weight_2d: 2D ``±1`` array, racetracks laid along the row dimension
                   in blocks of ``rt_size`` columns.
        rt_size:   Bits per racetrack.
    """
    rows, cols = weight_2d.shape
    n_rt_per_row = math.ceil(cols / rt_size)

    for w_i in range(rows):
        for rt_j in range(n_rt_per_row):
            rt_tuples: list[tuple[int, int, int, int]] = []
            count = 0
            k = 0
            bitgroup = [0, 0, 0]
            j_mid = [0, 0, 0]
            start = rt_j * rt_size
            end = min(start + rt_size, cols)
            current_sign = weight_2d[w_i, start]

            for bit_index in range(rt_size):
                w_j = start + bit_index
                if w_j >= end:
                    # Pad the trailing racetrack by treating out-of-bounds bits
                    # as a no-op — kernel checks ``w_j < qweight.shape[1]``
                    # before reading or counting.
                    continue

                if weight_2d[w_i, w_j] != current_sign:
                    j_mid[(count + 1) % 3] = w_j
                    current_sign = -current_sign
                    count += 1

                    if count > 2:
                        rt_tuples.append(
                            (
                                k,
                                sum(bitgroup),
                                bitgroup[(k + 1) % 3],
                                j_mid[(k + 1) % 3],
                            )
                        )
                        k += 1
                        bitgroup[count % 3] = 0

                bitgroup[count % 3] += 1

            rt_tuples.append(
                (k, sum(bitgroup), bitgroup[(k + 1) % 3], j_mid[(k + 1) % 3])
            )

            rt_tuples.sort(key=lambda t: (-t[1], t[2]))

            while rt_tuples:
                tuple_index, _endlen, flips, start_mid = rt_tuples[0]
                final_mid = start_mid + flips
                for idx in range(start_mid, final_mid):
                    if idx < cols:
                        weight_2d[w_i, idx] = -weight_2d[w_i, idx]

                rt_tuples = [
                    t
                    for t in rt_tuples
                    if t[0] not in (tuple_index - 1, tuple_index, tuple_index + 1)
                ]


@cuda.jit
def _endlen_emit_kernel(qweight, rt_size, out_starts, out_nflips, out_gain, out_count):
    """Like :func:`_endlen_kernel` but RECORDS merges instead of applying them.

    Verbatim copy of ``_endlen_kernel``'s scan / bubble-sort / greedy body; the
    ONLY change is the inner apply loop, which — instead of negating
    ``qweight`` — writes ``(start, n_flips, endlen_gain)`` for each processed
    tuple into the per-thread output rows. ``out_count[i, j]`` holds how many
    merges thread (i, j) recorded. Coverage matches ``_endlen_kernel`` exactly
    (floor division over full racetracks), so emitted candidates correspond
    one-to-one with the in-place kernel's flips.
    """
    rt_i, rt_j = cuda.grid(2)

    if rt_i < qweight.shape[0] and rt_j < qweight.shape[1] // rt_size:
        w_i = rt_i

        count = 0
        k = 0
        tuple_counts = 0
        current_sign = qweight[w_i, rt_j * rt_size]

        rt_tuples = cuda.local.array((_MAX_RT_SIZE, 4), dtype=numba.int64)
        bitgroup = cuda.local.array(3, dtype=numba.int32)
        j_mid = cuda.local.array(3, dtype=numba.int32)

        for x in range(rt_size):
            for y in range(4):
                rt_tuples[x, y] = 0

        for idx in range(3):
            bitgroup[idx] = 0
            j_mid[idx] = 0

        for bit_index in range(rt_size):
            w_j = rt_j * rt_size + bit_index

            if w_j < qweight.shape[1] and qweight[w_i, w_j] != current_sign:
                j_mid[(count + 1) % 3] = w_j
                current_sign = -current_sign
                count += 1

                if count > 2:
                    if k < rt_size:
                        rt_tuples[k, 0] = k
                        rt_tuples[k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
                        rt_tuples[k, 2] = bitgroup[(k + 1) % 3]
                        rt_tuples[k, 3] = j_mid[(k + 1) % 3]
                        k += 1

                    bitgroup[count % 3] = 0

            if w_j < qweight.shape[1]:
                bitgroup[count % 3] += 1

        if k < rt_size:
            rt_tuples[k, 0] = k
            rt_tuples[k, 1] = bitgroup[0] + bitgroup[1] + bitgroup[2]
            rt_tuples[k, 2] = bitgroup[(k + 1) % 3]
            rt_tuples[k, 3] = j_mid[(k + 1) % 3]
            tuple_counts = k + 1

        for _ in range(tuple_counts):
            for id in range(tuple_counts - 1):
                endlen_j = rt_tuples[id, 1]
                flips_j = rt_tuples[id, 2]
                endlen_j1 = rt_tuples[id + 1, 1]
                flips_j1 = rt_tuples[id + 1, 2]
                if (endlen_j < endlen_j1) or (
                    endlen_j == endlen_j1 and flips_j > flips_j1
                ):
                    for k in range(4):
                        temp = rt_tuples[id, k]
                        rt_tuples[id, k] = rt_tuples[id + 1, k]
                        rt_tuples[id + 1, k] = temp

        processed = 0
        while True:
            if processed >= tuple_counts:
                break
            idx_tuple = processed

            tuple_index = rt_tuples[idx_tuple, 0]
            endlen = rt_tuples[idx_tuple, 1]
            flips = rt_tuples[idx_tuple, 2]
            start_index_mid = rt_tuples[idx_tuple, 3]

            if endlen == 0:
                processed += 1
                continue

            # RECORD instead of flip. Clamp the span to the array width and
            # only record when it covers ≥1 bit (matches the CPU reference).
            final_index_mid = start_index_mid + flips
            if final_index_mid > qweight.shape[1]:
                final_index_mid = qweight.shape[1]
            n_flips = final_index_mid - start_index_mid
            if n_flips > 0:
                n = out_count[rt_i, rt_j]
                out_starts[rt_i, rt_j, n] = w_i * qweight.shape[1] + start_index_mid
                out_nflips[rt_i, rt_j, n] = n_flips
                out_gain[rt_i, rt_j, n] = endlen
                out_count[rt_i, rt_j] = n + 1

            for id in range(tuple_counts):
                tj_index = rt_tuples[id, 0]
                if (
                    tj_index == tuple_index - 1
                    or tj_index == tuple_index
                    or tj_index == tuple_index + 1
                ):
                    rt_tuples[id, 1] = 0

            processed += 1


def emit_candidates_gpu(weight_2d_np, latent_2d_np, *, rt_size: int, layer_idx: int) -> list:
    """Run the emit kernel and assemble :class:`MergeCandidate` objects.

    ``min_latent_magnitude`` is computed host-side from ``latent_2d_np`` over
    each recorded span (kept off-device to avoid float reductions in-kernel).
    Coverage follows the kernel (full racetracks only), matching
    :func:`_endlen_emit_cpu_reference` on multiple-of-``rt_size`` widths.
    """
    from netdrift.faults.weight_encoders.candidates import MergeCandidate

    rows, cols = weight_2d_np.shape
    n_rt = cols // rt_size  # floor: kernel skips trailing partial track
    w_gpu = cuda.to_device(np.ascontiguousarray(weight_2d_np))
    starts = cuda.to_device(np.zeros((rows, max(n_rt, 1), _MAX_RT_SIZE), dtype=np.int64))
    nflips = cuda.to_device(np.zeros((rows, max(n_rt, 1), _MAX_RT_SIZE), dtype=np.int64))
    gain = cuda.to_device(np.zeros((rows, max(n_rt, 1), _MAX_RT_SIZE), dtype=np.int64))
    count = cuda.to_device(np.zeros((rows, max(n_rt, 1)), dtype=np.int64))

    threads = (min(rows, 32), min(max(n_rt, 1), 32))
    blocks = (
        math.ceil(rows / threads[0]),
        math.ceil(max(n_rt, 1) / threads[1]),
    )
    _endlen_emit_kernel[blocks, threads](w_gpu, rt_size, starts, nflips, gain, count)
    cuda.synchronize()

    starts_h = starts.copy_to_host()
    nflips_h = nflips.copy_to_host()
    gain_h = gain.copy_to_host()
    count_h = count.copy_to_host()

    out: list[MergeCandidate] = []
    for i in range(rows):
        for j in range(n_rt):
            for m in range(int(count_h[i, j])):
                s = int(starts_h[i, j, m])
                nf = int(nflips_h[i, j, m])
                col_start = s - i * cols
                span_end = min(col_start + nf, cols)
                min_mag = (
                    float(np.min(np.abs(latent_2d_np[i, col_start:span_end])))
                    if span_end > col_start else 0.0
                )
                out.append(
                    MergeCandidate(
                        layer_idx=layer_idx,
                        unit_id_layer=0,
                        unit_id_racetrack=i * n_rt + j,
                        unit_id_channel=i,
                        start_idx=s,
                        n_flips=nf,
                        endlen_gain=int(gain_h[i, j, m]),
                        min_latent_magnitude=min_mag,
                    )
                )
    return out


def _endlen_emit_cpu_reference(
    *,
    got_view: np.ndarray,
    latent_view: np.ndarray,
    rt_size: int,
    layer_idx: int,
) -> list:
    """Emit the merges in-place endlen WOULD apply, without mutating anything.

    A line-for-line mirror of :func:`_endlen_cpu_reference` — same windowing,
    same ``(-endlen, +flips)`` sort, same neighbor-invalidation — but instead of
    negating the weight it records a :class:`MergeCandidate` for every processed
    tuple that flips at least one bit. Applying every returned candidate (in any
    order, since they are non-overlapping) reproduces the in-place result exactly;
    this parity is asserted in the tests.

    Args:
        got_view:    2D ``±1`` racetrack-aligned weight view (NOT mutated here).
        latent_view: Same-shaped latent FP weight; used for the per-merge
                     ``min_latent_magnitude`` (magnitude-aware ranking).
        rt_size:     Bits per racetrack.
        layer_idx:   Layer index stamped on every emitted candidate.

    Returns:
        List of :class:`MergeCandidate`, one per applied merge.
    """
    from netdrift.faults.weight_encoders.candidates import MergeCandidate

    rows, cols = got_view.shape
    n_rt_per_row = math.ceil(cols / rt_size)
    out: list[MergeCandidate] = []

    for w_i in range(rows):
        for rt_j in range(n_rt_per_row):
            rt_tuples: list[tuple[int, int, int, int]] = []
            count = 0
            k = 0
            bitgroup = [0, 0, 0]
            j_mid = [0, 0, 0]
            start = rt_j * rt_size
            end = min(start + rt_size, cols)
            current_sign = got_view[w_i, start]

            for bit_index in range(rt_size):
                w_j = start + bit_index
                if w_j >= end:
                    continue

                if got_view[w_i, w_j] != current_sign:
                    j_mid[(count + 1) % 3] = w_j
                    current_sign = -current_sign
                    count += 1

                    if count > 2:
                        rt_tuples.append(
                            (
                                k,
                                sum(bitgroup),
                                bitgroup[(k + 1) % 3],
                                j_mid[(k + 1) % 3],
                            )
                        )
                        k += 1
                        bitgroup[count % 3] = 0

                bitgroup[count % 3] += 1

            rt_tuples.append(
                (k, sum(bitgroup), bitgroup[(k + 1) % 3], j_mid[(k + 1) % 3])
            )

            rt_tuples.sort(key=lambda t: (-t[1], t[2]))

            while rt_tuples:
                tuple_index, endlen_gain, flips, start_mid = rt_tuples[0]
                final_mid = min(start_mid + flips, cols)
                if final_mid > start_mid:
                    span = slice(start_mid, final_mid)
                    min_mag = float(np.min(np.abs(latent_view[w_i, span])))
                    out.append(
                        MergeCandidate(
                            layer_idx=layer_idx,
                            unit_id_layer=0,
                            unit_id_racetrack=w_i * n_rt_per_row + rt_j,
                            unit_id_channel=w_i,
                            start_idx=w_i * cols + start_mid,
                            n_flips=int(final_mid - start_mid),
                            endlen_gain=int(endlen_gain),
                            min_latent_magnitude=min_mag,
                        )
                    )

                rt_tuples = [
                    t
                    for t in rt_tuples
                    if t[0] not in (tuple_index - 1, tuple_index, tuple_index + 1)
                ]

    return out


@register_encoder("endlen")
class EndlenEncoder(WeightEncoder):
    """Block-hypothesis weight encoder using the endlen heuristic.

    Stateless. Operates on the 2D racetrack-aligned view of a layer's
    weights; the caller handles kernel-mapping and rt_mapping reshapes.
    """

    def apply(self, weight_2d_gpu, rt_size: int) -> None:
        if rt_size > _MAX_RT_SIZE:
            raise ValueError(
                f"endlen requires rt_size <= {_MAX_RT_SIZE} "
                f"(legacy kernel uses a fixed-size local buffer); got rt_size={rt_size}"
            )

        rows, cols = weight_2d_gpu.shape
        n_rt_per_row = math.ceil(cols / rt_size)

        threads = (min(rows, 32), min(n_rt_per_row, 32))
        blocks = (
            math.ceil(rows / threads[0]),
            math.ceil(n_rt_per_row / threads[1]),
        )

        _endlen_kernel[blocks, threads](weight_2d_gpu, rt_size)
        cuda.synchronize()
