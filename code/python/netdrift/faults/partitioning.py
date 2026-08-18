"""Polarity-partitioned racetrack mapping (PPM).

Sorts each window of the racetrack-aligned weight view so that all ``+1``
weights precede all ``-1`` weights, then lays the sorted sequence out densely at
``rt_size``. Because a wire is fault-immune under ``edge_mode=saturate`` exactly
when all its cells share a sign, and a sorted window has only ONE sign boundary,
rounding each sign group up to a wire boundary makes *every* wire immune.

Contrast with the two existing data-dependent mappings:

* ``BLOCK`` (``layout.build_block_buckets``) isolates every maximal sign-run
  **in place**. A run is by definition maximal, so in-place segmentation yields
  millions of groups, and describing them is what costs BLOCK its
  ``table_bits_per_weight_bit = 1.000``.
* ``UNITS`` (``packing.build_unit_buckets``) isolates only long runs and pools
  the remainder onto ``rt_size``-long mixed-sign wires.

PPM instead exploits that "same sign" is a **global** property: sorting produces
exactly two groups per window regardless of how the signs were interleaved. The
permutation is admissible because weights are frozen at inference -- it is fixed
interconnect, not a stored runtime table (see the design doc for the limits of
that assumption).

Unlike BLOCK/UNITS the grid stays **rectangular** -- every wire is exactly
``rt_size`` long -- so PPM is a single :class:`~netdrift.faults.layout.BlockBucket`
at ``P = rt_size`` and needs no new CUDA kernels.

Sign convention matches ``layout.extract_blocks`` exactly: ``w > 0 -> +1``, so
an exactly-zero weight maps to ``-1``. Partitioning is **stable** (relative
order within each sign group is preserved), so the permutation is deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

from netdrift.faults.layout import BlockBucket

__all__ = [
    "CHANNEL_ALIGNED",
    "polarity_wire_plan",
    "build_polarity_buckets",
    "count_polarity_racetracks",
]

#: ``window`` sentinel: one window per row of the racetrack-aligned 2D view, so
#: a window never spans an output channel. The physically deployable instance --
#: a filter's weights are already fetched together.
CHANNEL_ALIGNED = 0


def _window_cols(n_cols: int, rt_size: int, window: int) -> list[tuple[int, int]]:
    """Split ``[0, n_cols)`` into ``[start, stop)`` window spans.

    ``window == CHANNEL_ALIGNED`` yields a single span covering the whole row;
    ``window == K`` yields spans of ``K * rt_size`` columns (the last is short
    whenever the row does not divide evenly).
    """
    if window == CHANNEL_ALIGNED:
        return [(0, n_cols)] if n_cols else []
    span = window * rt_size
    return [(s, min(s + span, n_cols)) for s in range(0, n_cols, span)]


def polarity_wire_plan(
    w_2d: "torch.Tensor",
    rt_size: int,
    window: int = CHANNEL_ALIGNED,
    pad: bool = True,
) -> list[dict]:
    """Plan the wires for one layer. Pure structure -- no weight values.

    Returns one dict per wire: ``{"sign", "rows", "cols", "length"}`` where
    ``rows``/``cols`` index the base 2D view and ``length == len(cols)`` is the
    real-cell count (``rt_size - length`` filler cells carry ``sign``).

    With ``pad=True`` each sign group within a window starts on a fresh wire, so
    every wire is sign-pure. With ``pad=False`` the sorted window is packed
    back-to-back, so the wire containing the +/- boundary is MIXED -- the
    ablation arm, expected to reproduce the units failure mode (design doc §3).
    """
    if rt_size < 1:
        raise ValueError(f"rt_size must be >= 1, got {rt_size}")
    if window < 0:
        raise ValueError(
            f"window must be >= 0 ({CHANNEL_ALIGNED} = channel-aligned), got {window}"
        )

    import torch

    signs = torch.where(w_2d > 0, 1, -1).cpu().tolist()
    n_rows = len(signs)
    n_cols = len(signs[0]) if n_rows else 0
    wires: list[dict] = []

    for r in range(n_rows):
        row = signs[r]
        for start, stop in _window_cols(n_cols, rt_size, window):
            pos = [c for c in range(start, stop) if row[c] > 0]
            neg = [c for c in range(start, stop) if row[c] <= 0]

            if pad:
                # Each group gets its own wires => every wire sign-pure.
                for sign, cols in ((1, pos), (-1, neg)):
                    for s in range(0, len(cols), rt_size):
                        chunk = cols[s:s + rt_size]
                        wires.append({
                            "sign": sign, "rows": [r] * len(chunk),
                            "cols": chunk, "length": len(chunk),
                        })
            else:
                # Ablation: concatenate, then cut on wire boundaries. The wire
                # straddling the boundary holds both signs.
                cols = pos + neg
                for s in range(0, len(cols), rt_size):
                    chunk = cols[s:s + rt_size]
                    # Filler sign only matters for a ragged tail; use the last
                    # real cell's sign so an all-positive tail stays pure.
                    wires.append({
                        "sign": 1 if row[chunk[-1]] > 0 else -1,
                        "rows": [r] * len(chunk), "cols": chunk,
                        "length": len(chunk),
                    })
    return wires


def build_polarity_buckets(
    w_2d: "torch.Tensor",
    rt_size: int,
    window: int = CHANNEL_ALIGNED,
    pad: bool = True,
) -> dict:
    """Return ``{rt_size: BlockBucket}`` -- a single bucket, all wires ``rt_size``.

    Mirrors ``layout.build_block_buckets`` / ``packing.build_unit_buckets`` so
    the existing per-bucket fault path consumes it unchanged. Filler cells carry
    the wire's sign (the same guard-band trick BLOCK uses), so a ragged tail
    reads that sign at any offset.
    """
    import numpy as np

    plan = polarity_wire_plan(w_2d, rt_size, window=window, pad=pad)
    w_host = w_2d.detach().cpu().float().numpy()

    n = len(plan)
    grid = np.zeros((n, rt_size), dtype=np.float32)
    srows = np.full((n, rt_size), -1, dtype=np.int64)
    scols = np.full((n, rt_size), -1, dtype=np.int64)
    lengths = np.zeros((n,), dtype=np.int32)

    for i, wire in enumerate(plan):
        L = wire["length"]
        lengths[i] = L
        rows, cols = wire["rows"], wire["cols"]
        grid[i, :L] = w_host[rows, cols]
        srows[i, :L] = rows
        scols[i, :L] = cols
        if rt_size > L:
            grid[i, L:] = float(wire["sign"])

    return {rt_size: BlockBucket(grid, srows, scols, lengths)}


def count_polarity_racetracks(
    w_2d: "torch.Tensor",
    rt_size: int,
    window: int = CHANNEL_ALIGNED,
    pad: bool = True,
) -> int:
    """Wire count for one layer, without materialising the grids.

    Used by the static-metrics path, which needs the cost axis but not the
    values. Counting via the plan (rather than a closed form) keeps this exact
    for ragged rows and sign-pure windows, both of which the ``(1 + 1/K)``
    estimate only approximates.
    """
    return len(polarity_wire_plan(w_2d, rt_size, window=window, pad=pad))
