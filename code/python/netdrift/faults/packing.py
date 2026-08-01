"""Pack layout units onto physical racetracks (numba-free).

Isolated runs each take their own wire of physical length ``next_pow2(L)``,
whose trailing guard cells replicate the run's own sign -- a *period-1*
extension. Because every cell on such a wire shares a sign, a misaligned read
that saturates anywhere inside it returns the correct value; that is the source
of BLOCK's fault immunity.

Pooled fragments are packed densely into ``rt_size``-cell wires in layout order.
When ``max_period == 2`` (valid only at ``threshold == 2``) one guard cell is
inserted at each fragment junction so the whole wire alternates: a fragment ends
with the sign opposite the isolated run that split it, and the next fragment
begins with that same sign, so an unguarded junction is *always* a phase break.
The trailing pad continues the alternation.

Guard cells are charged per fragment junction, never per pooled run.

Output is the existing :class:`~netdrift.faults.layout.BlockBucket` shape, so the
unchanged rectangular CUDA kernels can be driven per bucket with ``rt_size=P``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from netdrift.faults.layout import BlockBucket
from netdrift.faults.units import parse_units


@dataclass
class WireRecord:
    """One physical racetrack.

    ``rows``/``cols``/``fill`` all have length ``padded_len``. A slot with
    ``rows[j] == -1`` is a filler (guard band or interior phase guard) whose
    stored value is ``fill[j]``; otherwise the slot holds the weight at
    ``(rows[j], cols[j])`` and ``fill[j]`` is unused.

    ``period`` is 1 for a same-sign isolated run, 2 for a guarded alternating
    pool, and 0 for an unguarded pool with no periodicity guarantee.
    """
    period: int
    rows: list = field(default_factory=list)
    cols: list = field(default_factory=list)
    fill: list = field(default_factory=list)


def build_unit_wires(
    w_2d: "torch.Tensor",
    rt_size: int,
    *,
    threshold: int,
    max_period: int,
    pool_guard: int,
) -> list:
    """Parse ``w_2d`` into units and lay them out as :class:`WireRecord`\\ s."""
    if max_period not in (1, 2):
        raise ValueError(f"max_period must be 1 or 2, got {max_period}")
    if max_period == 2 and threshold != 2:
        raise ValueError(
            "max_period=2 (guarded period-2 pooling) requires threshold=2; at "
            "threshold>2 the pool contains runs of length >= 2 and is not "
            "period-2. See spec section 3.2."
        )

    isolated, fragments = parse_units(w_2d, rt_size, threshold)
    wires: list = []

    # --- isolated runs: own wire, period-1 guard band -----------------------
    for b in isolated:
        pad = b.padded_len - b.length
        wires.append(WireRecord(
            period=1,
            rows=list(b.rows) + [-1] * pad,
            cols=list(b.cols) + [-1] * pad,
            fill=[float(b.sign)] * b.padded_len,
        ))

    # --- pooled fragments: dense rt_size wires ------------------------------
    guarded = (max_period == 2 and pool_guard > 0)
    cur = WireRecord(period=2 if guarded else 0)

    def _flush():
        nonlocal cur
        if not cur.rows:
            return
        # Trailing pad: continue the alternation for a guarded wire, else repeat
        # the last stored sign (inert -- an unguarded pool has no phase to keep).
        last = cur.fill[-1] if cur.rows[-1] < 0 else _cell_sign(cur.rows[-1], cur.cols[-1])
        while len(cur.rows) < rt_size:
            last = -last if guarded else last
            cur.rows.append(-1)
            cur.cols.append(-1)
            cur.fill.append(float(last))
        wires.append(cur)
        cur = WireRecord(period=2 if guarded else 0)

    signs_rows = torch.where(w_2d > 0, 1, -1).cpu().tolist()

    def _cell_sign(r, c):
        return signs_rows[r][c]

    def _prev_sign():
        return (cur.fill[-1] if cur.rows[-1] < 0
                else _cell_sign(cur.rows[-1], cur.cols[-1]))

    for frag in fragments:
        # Phase-aware guard: needed ONLY when the next fragment would land on the
        # wrong parity, i.e. its first sign equals the previous slot's. Whether
        # that happens depends on the parity of how many isolated runs separate
        # the two fragments, so it MUST be decided per junction -- a fixed
        # always-one-guard rule breaks the ~34% of junctions already aligned.
        need_guard = bool(guarded and cur.rows and frag.signs[0] == _prev_sign())
        if len(cur.rows) + len(frag.cells) + int(need_guard) > rt_size:
            _flush()
            need_guard = False
        if need_guard:
            prev = _prev_sign()
            cur.rows.append(-1)
            cur.cols.append(-1)
            cur.fill.append(float(-prev))
        for (r, c), s in zip(frag.cells, frag.signs):
            if len(cur.rows) >= rt_size:
                _flush()
            cur.rows.append(r)
            cur.cols.append(c)
            cur.fill.append(float(s))
    _flush()

    return wires


def build_unit_buckets(
    w_2d: "torch.Tensor",
    rt_size: int,
    *,
    threshold: int,
    max_period: int,
    pool_guard: int,
) -> dict:
    """Group wires by physical length ``P`` into dense per-bucket arrays.

    Unlike :func:`~netdrift.faults.layout.build_block_buckets`, real cells are
    NOT a contiguous prefix here: a guarded pooled wire interleaves guard slots
    among real cells, so ``length`` is a count of real cells, not a boundary
    index -- do not slice ``weight_grid[i, :length[i]]``. Select real cells
    with the ``scatter_cols >= 0`` mask instead.
    """
    import numpy as np

    wires = build_unit_wires(w_2d, rt_size, threshold=threshold,
                             max_period=max_period, pool_guard=pool_guard)
    w_host = w_2d.detach().cpu().float().numpy()

    by_p: dict[int, list] = {}
    for x in wires:
        by_p.setdefault(len(x.rows), []).append(x)

    buckets: dict[int, BlockBucket] = {}
    for p, wlist in by_p.items():
        n = len(wlist)
        grid = np.zeros((n, p), dtype=np.float32)
        srows = np.full((n, p), -1, dtype=np.int64)
        scols = np.full((n, p), -1, dtype=np.int64)
        lengths = np.zeros((n,), dtype=np.int32)
        for i, x in enumerate(wlist):
            n_real = 0
            for j, (r, c) in enumerate(zip(x.rows, x.cols)):
                if r >= 0:
                    grid[i, j] = w_host[r, c]
                    srows[i, j] = r
                    scols[i, j] = c
                    n_real += 1
                else:
                    grid[i, j] = x.fill[j]
            lengths[i] = n_real
        buckets[p] = BlockBucket(grid, srows, scols, lengths)
    return buckets
