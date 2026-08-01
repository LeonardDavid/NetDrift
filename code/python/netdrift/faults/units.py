"""Parse a layer's base-2D view into layout units (numba-free).

A *unit* is either an **isolated run** — a maximal same-sign run of length
>= ``threshold``, which gets its own guard-banded racetrack — or a **pooled
fragment**, a maximal group of consecutive shorter runs that will be packed
densely alongside other fragments.

Segmentation matches :func:`netdrift.faults.layout.extract_blocks` exactly:
runs are maximal same-sign stretches inside an ``rt_size``-wide segment of the
base ROW/COL view, they never cross a segment boundary, the ragged tail segment
is parsed, and the sign convention is ``w > 0 -> +1`` so an exact zero maps to
-1. Diverging from any of those would make the simulation disagree with the
metrics and with already-collected BLOCK artifacts.

At ``threshold == 2`` every pooled run has length 1, so a fragment's signs
strictly alternate and the fragment is period-2. At ``threshold > 2`` the pool
also holds longer runs and is **not** period-2 -- see spec section 3.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from netdrift.faults.layout import BlockRecord, next_pow2


@dataclass
class Fragment:
    """A maximal group of consecutive pooled runs, in layout order.

    ``cells`` are ``(row, col)`` coordinates in the base-2D view, ordered along
    the original sequence; ``signs`` are the corresponding stored signs.
    """
    cells: list = field(default_factory=list)
    signs: list = field(default_factory=list)


def parse_units(
    w_2d: "torch.Tensor", rt_size: int, threshold: int
) -> tuple[list, list]:
    """Split ``w_2d`` into isolated runs (L >= threshold) and pooled fragments.

    Returns ``(isolated, fragments)``. ``isolated`` entries are
    :class:`~netdrift.faults.layout.BlockRecord`\\ s so they can be packed by the
    same code path BLOCK uses. Runs longer than 64 are chunked exactly as
    ``extract_blocks`` does (inert while ``rt_size <= 64``).
    """
    if threshold < 1:
        raise ValueError(f"threshold must be >= 1, got {threshold}")

    signs_rows = torch.where(w_2d > 0, 1, -1).cpu().tolist()
    n_cols = len(signs_rows[0]) if signs_rows else 0
    isolated: list = []
    fragments: list = []

    def _emit_isolated(sign, r, run_cols):
        for start in range(0, len(run_cols), 64):
            chunk = run_cols[start:start + 64]
            isolated.append(BlockRecord(
                sign=sign,
                length=len(chunk),
                padded_len=next_pow2(len(chunk)),
                rows=[r] * len(chunk),
                cols=list(chunk),
            ))

    for r, row in enumerate(signs_rows):
        for seg_start in range(0, n_cols, rt_size):
            seg = row[seg_start:seg_start + rt_size]
            if not seg:
                continue
            # 1) run-length encode the segment
            runs: list[tuple[int, list[int]]] = []
            cur = seg[0]
            run_cols = [seg_start]
            for k in range(1, len(seg)):
                if seg[k] == cur:
                    run_cols.append(seg_start + k)
                else:
                    runs.append((cur, run_cols))
                    cur = seg[k]
                    run_cols = [seg_start + k]
            runs.append((cur, run_cols))

            # 2) isolate long runs; accumulate consecutive short ones into a
            #    fragment that is flushed whenever an isolated run splits it.
            open_frag = Fragment()
            for sign, cols in runs:
                if len(cols) >= threshold:
                    if open_frag.cells:
                        fragments.append(open_frag)
                        open_frag = Fragment()
                    _emit_isolated(sign, r, cols)
                else:
                    for c in cols:
                        open_frag.cells.append((r, c))
                        open_frag.signs.append(sign)
            if open_frag.cells:
                fragments.append(open_frag)

    return isolated, fragments
