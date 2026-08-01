#!/usr/bin/env python
"""Measure the racetrack-layout design space for a checkpoint (CPU-only).

Gating measurement for the ROW/COL <-> BLOCK tradeoff study. Answers, from one
pass over the weights, how much range any layout knob actually has:

  * run-length (sign-block) histogram  -- is there anything long to isolate?
  * alternating-sequence histogram     -- maximal groups of consecutive len-1
                                          runs (period-2 regions)
  * adjacent run-length pair histogram -- lets period-p regions (p=2,3,...) be
                                          derived offline without a re-run
  * wire / cell counts + an area model for every candidate design point

Segmentation matches ``netdrift.faults.layout.extract_blocks`` exactly: runs are
maximal same-sign stretches inside an ``rt_size``-wide segment of the base
ROW/COL view, never crossing a segment boundary, and the ragged tail segment is
included.

TWO CONVENTION NOTES (both are real mismatches in the current tree):

  * Sign of zero. ``extract_blocks`` uses ``w > 0 -> +1`` (so exactly-zero maps
    to -1), while ``count_sign_transitions.py`` uses ``w >= 0 -> +1``. This
    script defaults to the ``extract_blocks`` convention because it is modelling
    the BLOCK layout; ``--zero-sign pos`` switches. Inert for binarized weights
    (values are +-scale), but it would silently diverge on an exact zero.
  * Ragged tail. ``count_sign_transitions.py`` floors to full racetracks and
    drops the tail; ``extract_blocks`` keeps it. This script keeps it, so wire
    counts here are directly comparable to a BLOCK run.

SELF-CHECK: for VGG7-CIFAR10 the printed BLOCK wire count should reproduce the
measured 6,527,245 (base_layout=row) / 5,610,139 (base_layout=col). If it does
not, the segmentation here has drifted from the simulator and every derived
number below is suspect.

Usage:
    python scripts-python/analyze_layout_design_space.py \\
        --model vgg7_cifar10 \\
        --checkpoint models/w1a1/vgg7_cifar10/model_best.pt \\
        --layouts row col --rt-size 64 \\
        --json runs/layout_design_space.json

    # restrict to the layers a sweep leaves unprotected
    python scripts-python/analyze_layout_design_space.py ... --unprotected 2 3 4 5 6 7

    # fidelity gate: cross-check the units design points against the real packer
    python scripts-python/analyze_layout_design_space.py ... --verify-packer

CPU-only; no CUDA, no fault model, no dataset. ``--verify-packer`` additionally
imports ``netdrift.faults.packing`` (numba-transitive via the package's
``__init__``, same as the ``netdrift.faults.layout`` import this script already
does unconditionally) but still runs no CUDA kernel and no fault model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

# Make the netdrift package importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "code" / "python"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch

from netdrift.faults.layout import _layout_weight_for_racetrack, next_pow2
from netdrift.models import build_model, replace_with_quantized
from netdrift.models.checkpoint import load_checkpoint
from netdrift.quant.binary import BinaryScheme
from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

PAIR_CAP = 8       # adjacent-pair histogram is capped at this run length
PERIODS = (1, 2, 3, 4)  # periods measured by _period_region_hist


# ---------------------------------------------------------------- primitives

def _segment_matrices(signs_2d: np.ndarray, rt_size: int) -> list[np.ndarray]:
    """Split the base-2D sign view into independent ``rt_size``-wide segments.

    Returns a list of (N, S) matrices whose rows are each one segment, matching
    ``extract_blocks``' iteration (per row, then per segment, ragged tail kept).
    """
    rows, cols = signs_2d.shape
    n_full = cols // rt_size
    tail = cols - n_full * rt_size
    mats: list[np.ndarray] = []
    if n_full:
        full = signs_2d[:, : n_full * rt_size].reshape(rows, n_full, rt_size)
        mats.append(np.ascontiguousarray(full.reshape(rows * n_full, rt_size)))
    if tail:
        mats.append(np.ascontiguousarray(signs_2d[:, n_full * rt_size :]))
    return mats


def _runs_of_matrix(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run-length-encode every row of ``mat`` independently.

    Returns ``(lengths, row_idx)``. A boundary is forced at every row start so
    runs never span two segments.
    """
    n_rows, seg = mat.shape
    if n_rows == 0 or seg == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    flat = mat.reshape(-1)
    n = flat.size
    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    np.not_equal(flat[1:], flat[:-1], out=is_start[1:])
    is_start[::seg] = True  # force a boundary at every row (= segment) start
    starts = np.flatnonzero(is_start)
    ends = np.append(starts[1:], n)
    return (ends - starts).astype(np.int64), (starts // seg).astype(np.int64)


def _alt_group_sizes(lengths: np.ndarray, row_idx: np.ndarray) -> np.ndarray:
    """Sizes of maximal groups of consecutive length-1 runs, never crossing a row.

    Group size is counted in *runs*, which for length-1 runs equals bits. Matches
    the existing ``alternating_seq_histogram`` definition, except that this
    returns groups of every size (including 1) so the >=2 / >=3 threshold can be
    chosen offline.
    """
    if lengths.size == 0:
        return np.empty(0, np.int64)
    is1 = lengths == 1
    cut = np.flatnonzero(row_idx[1:] != row_idx[:-1]) + 1
    aug = np.insert(is1, cut, False) if cut.size else is1
    padded = np.concatenate(([False], aug, [False]))
    d = np.diff(padded.astype(np.int8))
    return (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).astype(np.int64)


def _period_region_hist(mat: np.ndarray, p: int) -> Counter:
    """Lengths of maximal period-``p`` regions in every row of ``mat``.

    A region has period ``p`` iff ``s[i] == s[i+p]`` throughout, so it reads
    correctly under any misalignment that is a multiple of ``p``. A maximal run
    of ``m`` satisfied positions corresponds to a region of ``m + p`` bits.
    Regions never cross a row (= segment) boundary.

    Note the nesting: a period-1 region (a same-sign block) also satisfies
    period 2, 3, ... So these counts are cumulative-by-divisor, not disjoint;
    compare p against its divisors to see what p buys on its own.
    """
    n_rows, seg = mat.shape
    if seg <= p:
        return Counter()
    eq = mat[:, :-p] == mat[:, p:]                     # (n_rows, seg - p)
    pad = np.zeros((n_rows, 1), dtype=bool)
    d = np.diff(np.hstack([pad, eq, pad]).astype(np.int8), axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    # np.argwhere scans row-major, so starts[i] and ends[i] pair up per row.
    lengths = ends[:, 1] - starts[:, 1] + p
    return Counter(lengths.tolist())


def _pair_hist(lengths: np.ndarray, row_idx: np.ndarray, cap: int) -> np.ndarray:
    """Capped 2D histogram of adjacent (run, next run) lengths within a segment."""
    hist = np.zeros((cap + 1, cap + 1), dtype=np.int64)
    if lengths.size < 2:
        return hist
    same = row_idx[1:] == row_idx[:-1]
    if not same.any():
        return hist
    a = np.minimum(lengths[:-1][same], cap)
    b = np.minimum(lengths[1:][same], cap)
    np.add.at(hist, (a, b), 1)
    return hist


# ------------------------------------------------------------- per-layer pass

def analyze_layer(
    qw: torch.Tensor, *, rt_size: int, layout: str, kernel_mapping: str, zero_sign: str
) -> dict:
    """Collect every raw distribution for one layer under one base layout."""
    km = kernel_mapping.upper() if qw.dim() == 4 else None
    w_2d, _undo = _layout_weight_for_racetrack(
        qw, rt_mapping=layout.upper(), kernel_mapping=km
    )
    w = w_2d.detach().cpu().numpy()
    # extract_blocks uses ``w > 0 -> +1``; sign(0) therefore lands on -1.
    signs = (np.where(w > 0, 1, -1) if zero_sign == "neg"
             else np.where(w >= 0, 1, -1)).astype(np.int8)

    rows, cols = signs.shape
    run_hist: Counter = Counter()
    alt_hist: Counter = Counter()
    pair_hist = np.zeros((PAIR_CAP + 1, PAIR_CAP + 1), dtype=np.int64)
    period_hist: dict[int, Counter] = {p: Counter() for p in PERIODS}
    n_segments = 0

    for mat in _segment_matrices(signs, rt_size):
        lengths, row_idx = _runs_of_matrix(mat)
        n_segments += mat.shape[0]
        run_hist.update(Counter(lengths.tolist()))
        alt_hist.update(Counter(_alt_group_sizes(lengths, row_idx).tolist()))
        pair_hist += _pair_hist(lengths, row_idx, PAIR_CAP)
        for p in PERIODS:
            period_hist[p].update(_period_region_hist(mat, p))

    return {
        "rows": int(rows),
        "cols": int(cols),
        "bits": int(rows * cols),
        "n_segments": int(n_segments),
        "dense_wires": int(rows * math.ceil(cols / rt_size)),
        "run_hist": {int(k): int(v) for k, v in sorted(run_hist.items())},
        "alt_hist": {int(k): int(v) for k, v in sorted(alt_hist.items())},
        "pair_hist": pair_hist.tolist(),
        "period_region_hist": {
            int(p): {int(k): int(v) for k, v in sorted(period_hist[p].items())}
            for p in PERIODS
        },
    }


# ------------------------------------------------------- derived design points

def _sum_pow2_cells(run_hist: dict[int, int]) -> int:
    return sum(next_pow2(L) * c for L, c in run_hist.items())


def _table_bits(
    iso_hist: dict[int, int], *, bits: int, contiguous: bool, kind_bits: int
) -> tuple[int, int]:
    """Structure-table cost for a layout, as (entropy_bound, fixed_width).

    Any non-affine layout needs a table saying which wire and offset holds a
    given logical weight; dense ROW/COL and the rt_size sweep are affine and
    need none. The table must describe the *isolated* units only -- pooled bits
    are a plain dense sequence laid down in order, so their run structure never
    has to be stored. That is why coarse thresholds are cheap and BLOCK is not.

    Two bounds are reported because the paper can defend either end:

    * ``entropy_bound`` -- ideal entropy coding. Each unit costs
      ``-log2 p(length)``; when the units do not tile the layer (a threshold
      design) each also costs ``log2(bits / n_units)`` to code the gap to the
      next one. ``kind_bits`` adds a per-unit period tag when more than one
      period is allowed, since a length alone no longer identifies the kind.
    * ``fixed_width`` -- a naive record of (absolute position, length).

    For BLOCK on a near-random sign pattern the entropy bound lands at roughly
    one bit per weight bit, because the run lengths determine the signs up to a
    single bit. That is the point, not an artifact.
    """
    n_iso = sum(iso_hist.values())
    if n_iso == 0:
        return 0, 0
    ent = 0.0
    for L, c in iso_hist.items():
        if c:
            ent += c * -math.log2(c / n_iso)
    if not contiguous:
        ent += n_iso * math.log2(max(bits / n_iso, 2.0))
    ent += n_iso * kind_bits
    fixed = n_iso * (kind_bits + 6 + math.ceil(math.log2(max(bits, 2))))
    return int(ent), int(fixed)


def derive_designs(
    agg: dict, *, rt_size: int, rows_cols: list[tuple[int, int]]
) -> list[dict]:
    """Wire/cell counts for every candidate design, from the aggregated histograms.

    All of these are closed-form in the histograms, which is why the script dumps
    the histograms: new design points can be evaluated offline with no re-run.
    """
    run_hist = {int(k): int(v) for k, v in agg["run_hist"].items()}
    alt_hist = {int(k): int(v) for k, v in agg["alt_hist"].items()}
    bits = agg["bits"]
    n_runs = sum(run_hist.values())
    block_cells = _sum_pow2_cells(run_hist)

    out: list[dict] = []

    def add(name, wires, cells, iso_hist, *, contiguous, kind_bits=0, note=""):
        t_ent, t_fix = _table_bits(iso_hist, bits=bits, contiguous=contiguous,
                                   kind_bits=kind_bits)
        out.append({"design": name, "wires": int(wires), "cells": int(cells),
                    "iso_units": int(sum(iso_hist.values())),
                    "table_bits_entropy": t_ent, "table_bits_fixed": t_fix,
                    "note": note})

    # -- the two measured endpoints -------------------------------------------
    # Dense layouts are affine: a weight's wire and offset are computed, so they
    # carry no structure table. rt_size is always included so the report's
    # normalisation baseline exists.
    for rt in sorted({64, 32, 16, 8, 4, 2, rt_size}, reverse=True):
        wires = sum(r * math.ceil(c / rt) for r, c in rows_cols)
        add(f"dense rt_size={rt}", wires, wires * rt, {}, contiguous=True,
            note="iso-cost control, table-free" if rt == 2 else "table-free")
    # BLOCK isolates every run, so the units tile the layer in order and the
    # table degenerates to the run-length sequence.
    add("BLOCK (own wire per run)", n_runs, block_cells, run_hist,
        contiguous=True, note="current block layout")

    # -- alternating-sequence merging (threshold t, in runs) ------------------
    for t in (2, 3, 4):
        merged_bits = sum(k * c for k, c in alt_hist.items() if k >= t)
        n_units = sum(c for k, c in alt_hist.items() if k >= t)
        unmerged_cells = block_cells - merged_bits  # merged runs were 1 cell each
        # Runs left un-merged: every L >= 2 run, plus the length-1 runs that sat
        # in groups smaller than t.
        unmerged = dict(run_hist)
        unmerged[1] = unmerged.get(1, 0) - merged_bits
        # own wire: alt units are isolated too, so units still tile the layer.
        # A length alone no longer says whether a unit is period-1 or period-2,
        # hence one kind bit per unit.
        iso_own = dict(unmerged)
        for k, c in alt_hist.items():
            if k >= t:
                iso_own[k] = iso_own.get(k, 0) + c
        add(f"alt-merge t={t}, own wire",
            n_runs - merged_bits + n_units,
            unmerged_cells + sum(next_pow2(k) * c
                                 for k, c in alt_hist.items() if k >= t),
            iso_own, contiguous=True, kind_bits=1)
        # alt-seq units pooled densely instead of each taking a wire. G guard
        # cells per unit; G=0 is the optimistic bound (no boundary protection).
        for g in (0, 1, 2):
            pooled = math.ceil((merged_bits + g * n_units) / rt_size)
            add(f"alt-merge t={t}, pooled G={g}",
                n_runs - merged_bits + pooled,
                unmerged_cells + pooled * rt_size,
                unmerged, contiguous=False)

    # -- isolate L>=2 + phase-aware guarded pooling ---------------------------
    # The pooled bits at T=2 are exactly the length-1 runs, so each pooled
    # fragment alternates internally and the pool can be made period-2. A
    # junction needs a guard cell only when the next fragment would land on the
    # wrong parity, which happens iff an ODD number of isolated runs separates
    # the two fragments:  P = 1 / (1 + P(run is isolated)).  A fixed
    # one-guard-per-junction rule is wrong -- it breaks the already-aligned
    # junctions. The guard decision depends on the pooled signs, which the
    # decoder cannot know in advance, so it costs one table bit per junction.
    n_frag = sum(alt_hist.values())
    if n_frag:
        len1 = run_hist.get(1, 0)
        p_iso = 1.0 - (len1 / n_runs) if n_runs else 0.0
        guards = int(round(n_frag / (1.0 + p_iso)))
        iso2 = {L: c for L, c in run_hist.items() if L >= 2}
        iso2_cells = sum(next_pow2(L) * c for L, c in iso2.items())
        pooled = math.ceil((len1 + guards) / rt_size)
        t_ent, t_fix = _table_bits(iso2, bits=bits, contiguous=False, kind_bits=0)
        out.append({
            "design": "isolate L>=2 + phase-aware guarded pooling",
            "wires": int(sum(iso2.values()) + pooled),
            "cells": int(iso2_cells + pooled * rt_size),
            "iso_units": int(sum(iso2.values())),
            # +1 bit per junction records whether a guard cell was inserted
            "table_bits_entropy": int(t_ent + n_frag),
            "table_bits_fixed": int(t_fix + n_frag),
            "note": f"period-2 pool; {guards:,}/{n_frag:,} junctions guarded",
        })

    # -- isolate long runs (L >= T), pool the rest ---------------------------
    for T in (2, 3, 4, 8, 16, 32):
        iso_hist = {L: c for L, c in run_hist.items() if L >= T}
        iso_wires = sum(iso_hist.values())
        iso_cells = sum(next_pow2(L) * c for L, c in iso_hist.items())
        pooled_bits = sum(L * c for L, c in run_hist.items() if L < T)
        pooled_units = sum(c for L, c in run_hist.items() if L < T)
        for g in (0, 1, 2):
            pooled = math.ceil((pooled_bits + g * pooled_units) / rt_size)
            add(f"isolate L>={T}, pool rest G={g}",
                iso_wires + pooled, iso_cells + pooled * rt_size,
                iso_hist, contiguous=False)

    for d in out:
        d["cells_per_bit"] = round(d["cells"] / bits, 4)
        d["bits_per_wire"] = round(bits / d["wires"], 3) if d["wires"] else 0.0
        d["table_bits_per_weight_bit"] = round(d["table_bits_entropy"] / bits, 3)
    return out


# Design points this script's analytic formulas claim to model exactly, paired
# with the real packer config that implements them. Each analytic count is a
# GLOBAL division (e.g. ``ceil((pooled_bits + g) / rt_size)``), but
# ``netdrift.faults.packing.build_unit_wires`` never splits a fragment across
# two wires -- it flushes and starts a new wire whenever the next whole
# fragment would not fit -- so the real count is provably >= the analytic one
# (spec section 6's "Cell count can rise while wires fall" is the same
# whole-unit-packing effect). Extend this list if more design points gain a
# real packer config; the two below are what section 6's arm matrix actually
# implements today (units at T=2 with guarded pooling, and T>=3 unguarded).
VERIFY_PACKER_CHECKS: list[tuple[int, int, int, str]] = [
    # (threshold, max_period, pool_guard, matching `design` name in `out`)
    (2, 2, 1, "isolate L>=2 + phase-aware guarded pooling"),
    (4, 1, 0, "isolate L>=4, pool rest G=0"),
]


def verify_packer_fidelity(
    designs: list[dict],
    layers: list[tuple[str, int, "torch.Tensor"]],
    *,
    rt_size: int,
    layout: str,
    kernel_mapping: str,
    ratio_threshold: float,
) -> bool:
    """Fidelity gate: analytic wire count (this script) vs the REAL packer's.

    This is the ``--verify-packer`` implementation. It is the same kind of
    check as spec section 7 gate 1 ("the analysis script reproduces
    5,610,139/6,527,245 BLOCK wires exactly, which is its fidelity check
    against the simulator") but for the units design points that section 6's
    arm matrix actually runs -- BLOCK's own gate does not cover units, and
    without this one there is no gate proving the paper's cost-axis numbers
    (this script's ``wires``/``cells`` columns) match what
    ``netdrift.faults.rtm_misalignment._run_units_path`` actually built and
    fault-injected.

    For each ``(threshold, max_period, pool_guard)`` in VERIFY_PACKER_CHECKS,
    calls the real ``build_unit_wires`` on every layer's actual (already
    quantized) weight -- the same per-layer 2D view ``analyze_layer`` derives
    via ``_layout_weight_for_racetrack`` -- and sums the real wire count
    across layers. Compares against this script's analytic count for the
    matching design point (looked up by name in ``designs``).

    The packer's whole-fragment-flush invariant means actual wires are always
    >= analytic; a ratio below 1.0 would mean this check itself disagrees with
    that invariant (a bug in the check, not evidence the packer is cheaper
    than modelled) and is flagged just as loudly as excess divergence.

    NOTE a second, smaller source of the same-direction gap that is NOT a
    packer bug: the analytic side's ``ceil((pooled_bits + guards) / rt_size)``
    in ``derive_designs`` is applied ONCE to bits pooled across every layer,
    while the real packer flushes per layer (a fragment never crosses a layer
    boundary any more than it crosses an ``rt_size`` segment). That is up to
    ``n_layers - 1`` extra real wires from independent per-layer ceiling
    rounding alone, on top of the whole-fragment-flush waste this check is
    meant to catch. Negligible next to the wire counts involved, but do not
    misread a small, layer-count-sized excess as fragment-packing divergence.

    Returns True iff every checked design point is within ``ratio_threshold``.
    """
    from netdrift.faults.packing import build_unit_wires  # lazy: only this path needs it

    by_name = {d["design"]: d for d in designs}
    print("\nPacker fidelity check (--verify-packer): analytic (this script) vs "
          "the real netdrift.faults.packing.build_unit_wires")
    print(f"  {'design':44s} {'analytic':>12} {'actual':>12} {'ratio':>8}  flag")
    all_ok = True
    for threshold, max_period, pool_guard, name in VERIFY_PACKER_CHECKS:
        design = by_name.get(name)
        if design is None:
            print(f"  {name!r} not found among derived designs for this layout "
                  f"-- skipped (analytic side unavailable)")
            all_ok = False
            continue
        actual = 0
        for _layer_name, _layer_id, qw in layers:
            km = kernel_mapping.upper() if qw.dim() == 4 else None
            w_2d, _undo = _layout_weight_for_racetrack(
                qw, rt_mapping=layout.upper(), kernel_mapping=km,
            )
            wires = build_unit_wires(
                w_2d, rt_size, threshold=threshold, max_period=max_period,
                pool_guard=pool_guard,
            )
            actual += len(wires)
        analytic = design["wires"]
        ratio = actual / analytic if analytic else float("inf")
        divergent = ratio > ratio_threshold or ratio < 1.0
        all_ok = all_ok and not divergent
        flag = "<<< DIVERGENT" if divergent else "ok"
        print(f"  {name[:44]:44s} {analytic:>12,} {actual:>12,} {ratio:8.4f}  {flag}")
        if ratio < 1.0:
            print("    NOTE: actual < analytic is impossible under the packer's "
                  "whole-fragment-flush invariant -- this points at a bug in this "
                  "check (e.g. a config/name mismatch), not the packer being "
                  "cheaper than modelled.")
    if not all_ok:
        print("  WARNING: divergence beyond the tolerance -- this script's cost "
              "numbers for the flagged design point(s) do NOT match what the "
              "simulator actually built. Do not report those numbers in the "
              "paper until reconciled.")
    return all_ok


def period_summary(pair_hist: np.ndarray) -> dict:
    """Frequency of adjacent run-length pairs that indicate short-period regions."""
    total = int(pair_hist.sum())
    if not total:
        return {}
    return {
        "total_adjacent_pairs": total,
        "period2_(1,1)_frac": round(float(pair_hist[1, 1]) / total, 5),
        "period3_(2,1)+(1,2)_frac": round(
            float(pair_hist[2, 1] + pair_hist[1, 2]) / total, 5),
        "period4_(3,1)+(1,3)_frac": round(
            float(pair_hist[3, 1] + pair_hist[1, 3]) / total, 5),
        "period4_(2,2)_frac": round(float(pair_hist[2, 2]) / total, 5),
    }


# ------------------------------------------------------------------- reporting

def _fmt(n: int) -> str:
    return f"{n:,}"


def report(layout: str, per_layer: list[dict], agg: dict, designs: list[dict],
           rt_size: int) -> None:
    bits = agg["bits"]
    n_runs = sum(int(v) for v in agg["run_hist"].values())
    run_hist = {int(k): int(v) for k, v in agg["run_hist"].items()}
    alt_hist = {int(k): int(v) for k, v in agg["alt_hist"].items()}

    print()
    print("=" * 78)
    print(f"base_layout={layout}   rt_size={rt_size}   bits={_fmt(bits)}")
    print("=" * 78)

    print(f"\nruns={_fmt(n_runs)}   mean run length={bits / n_runs:.4f}   "
          f"transitions={_fmt(n_runs - agg['n_segments'])}")
    print("  (BLOCK wire count == runs; compare against the measured "
          "6,527,245 row / 5,610,139 col)")

    print("\nRun-length histogram (share of runs / share of bits):")
    for L in sorted(run_hist):
        c = run_hist[L]
        if c == 0:
            continue
        print(f"  L={L:3d}  {_fmt(c):>14}  {100.0 * c / n_runs:6.2f}% of runs  "
              f"{100.0 * L * c / bits:6.2f}% of bits")

    print("\nAlternating-sequence groups (consecutive length-1 runs; size in runs = bits):")
    n_groups = sum(alt_hist.values())
    len1 = run_hist.get(1, 0)
    for k in sorted(alt_hist):
        c = alt_hist[k]
        print(f"  k={k:3d}  {_fmt(c):>14}  {100.0 * c / max(n_groups, 1):6.2f}% of groups  "
              f"{100.0 * k * c / max(len1, 1):6.2f}% of length-1 runs")

    print("\nShort-period region frequency (from the adjacent-pair histogram):")
    for k, v in period_summary(np.array(agg["pair_hist"])).items():
        print(f"  {k:28s} {v}")

    print("\nMaximal period-p regions (immune at offsets = 0 mod p). NOTE these")
    print("nest: a period-1 block also satisfies p=2,3,4 — compare p to its")
    print("divisors to see what p buys on its own. 'wires' = one per region.")
    print("'% of bits' is an upper bound: adjacent spans may overlap by < p.")
    print(f"  {'p':>2} {'min len':>8} {'regions':>12} {'% of bits':>10} {'wires/dense':>12}")
    for p, hist in sorted(agg.get("period_region_hist", {}).items()):
        hist = {int(k): int(v) for k, v in hist.items()}
        for lmin in (2, 4, 8, 16):
            regions = sum(c for L, c in hist.items() if L >= lmin)
            cov = sum(L * c for L, c in hist.items() if L >= lmin)
            if not regions:
                continue
            print(f"  {p:>2} {lmin:>8} {_fmt(regions):>12} "
                  f"{100.0 * cov / bits:9.2f}% {regions / agg['dense_wires']:11.2f}x")

    print("\nPer-layer summary:")
    print(f"  {'layer':32s} {'bits':>12} {'runs':>12} {'mean L':>7} {'dense':>10}")
    for r in per_layer:
        print(f"  {r['layer_name'][:32]:32s} {_fmt(r['bits']):>12} "
              f"{_fmt(sum(int(v) for v in r['run_hist'].values())):>12} "
              f"{r['bits'] / max(sum(int(v) for v in r['run_hist'].values()), 1):7.3f} "
              f"{_fmt(r['dense_wires']):>10}")

    base = next(d for d in designs if d["design"] == f"dense rt_size={rt_size}")
    print(f"\nDesign points, normalised to '{base['design']}'.")
    print("  A@r   = wires*r + cells        (area, A_port/A_cell = r)")
    print("  A@r+T = wires*r + cells + table_bits   (structure table included)")
    print("  tbl/b = structure-table bits per weight bit; 0 = affine, table-free")
    print(f"  {'design':34s} {'wires':>12} {'cells':>12} {'b/wire':>7} {'tbl/b':>6} "
          f"{'A@16':>7} {'A@16+T':>7} {'A@64':>7} {'A@64+T':>7}")
    for d in designs:
        row = (f"  {d['design'][:34]:34s} {_fmt(d['wires']):>12} "
               f"{_fmt(d['cells']):>12} {d['bits_per_wire']:7.2f} "
               f"{d['table_bits_per_weight_bit']:6.2f}")
        for ratio in (16, 64):
            a0 = base["wires"] * ratio + base["cells"]
            a = d["wires"] * ratio + d["cells"]
            a_t = a + d["table_bits_entropy"]
            row += f" {a / a0:7.2f} {a_t / a0:7.2f}"
        print(row + (f"   <- {d['note']}" if d["note"] else ""))


# ------------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="registry name, e.g. vgg7_cifar10")
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    p.add_argument("--rt-size", type=int, default=64)
    p.add_argument("--layouts", nargs="+", default=["row", "col"],
                   help="base layouts to analyze (row col)")
    p.add_argument("--kernel-mapping", default="row", help="row | col | clw | acw")
    p.add_argument("--unprotected", type=int, nargs="*", default=None,
                   help="1-based layer ids to include (default: all quantized layers)")
    p.add_argument("--zero-sign", choices=("neg", "pos"), default="neg",
                   help="sign of an exactly-zero weight; 'neg' matches extract_blocks")
    p.add_argument("--json", default=None, help="write the full raw dump here")
    p.add_argument("--verify-packer", action="store_true",
                   help="Fidelity gate proving the paper's cost axis matches the "
                        "simulator: for the design points that correspond to "
                        "implemented units configs (currently threshold=2/"
                        "max_period=2/pool_guard=1 and threshold=4/max_period=1/"
                        "pool_guard=0 -- see VERIFY_PACKER_CHECKS), calls the REAL "
                        "netdrift.faults.packing.build_unit_wires on each layer's "
                        "actual weight and reports the analytic wire count (this "
                        "script's closed-form histogram estimate) vs the packer's "
                        "actual count and their ratio, flagging any divergence "
                        "beyond --verify-packer-threshold. This script's wire counts "
                        "are derived analytically (e.g. a GLOBAL "
                        "ceil((pooled_bits+guards)/rt_size) for pooled wires), but "
                        "the packer flushes a wire whenever the next whole fragment "
                        "would not fit -- it never splits a fragment across two "
                        "wires -- so the real count is provably >= the analytic one "
                        "and the two must be reconciled before a design point's cost "
                        "number is reported in the paper. Off by default: importing "
                        "netdrift.faults.packing requires numba (transitively, via "
                        "netdrift.faults.__init__), so this flag is opt-in and the "
                        "normal analytic-only path is unaffected when it is off.")
    p.add_argument("--verify-packer-threshold", type=float, default=1.02,
                   help="Max tolerated (actual/analytic) wire-count ratio before "
                        "--verify-packer flags a design point as divergent. "
                        "Default: 1.02 (2%% slack) is an UNCALIBRATED starting "
                        "point, not a measured bound -- at threshold=2 pooled "
                        "wires are a small share of the total so 2%% is generous, "
                        "but at threshold=4 (unguarded, longer fragments, larger "
                        "pooled share) the whole-fragment-flush overhead may "
                        "plausibly approach or exceed it on real VGG7 layers; "
                        "re-set this from the first real --verify-packer run's "
                        "reported ratios rather than trusting the default. A "
                        "ratio < 1.0 is always flagged regardless of this value, "
                        "since the packer cannot pack fewer wires than the "
                        "analytic estimate by construction.")
    args = p.parse_args(argv)

    if args.rt_size > 64:
        p.error("rt_size > 64 would trigger extract_blocks' >64 run split, which "
                "this script does not model; keep rt_size <= 64.")

    model = build_model(args.model)
    scheme = BinaryScheme()
    replace_with_quantized(model, scheme)
    rep = load_checkpoint(
        model, args.checkpoint, mode="strict", scheme=scheme,
        scale_init="max_abs", map_location="cpu",
    )
    print(rep.summary())

    layers = []
    for _, m in model.named_modules():
        if not isinstance(m, (QuantizedConv2d, QuantizedLinear)):
            continue
        if args.unprotected is not None and m.layer_id not in args.unprotected:
            continue
        qw = scheme.quantize(m.weight.data, getattr(m, "scale_per_channel", None)).values
        layers.append((m.layer_name, m.layer_id, qw))
    if not layers:
        p.error("no quantized layers selected")

    dump: dict = {"model": args.model, "checkpoint": args.checkpoint,
                  "rt_size": args.rt_size, "kernel_mapping": args.kernel_mapping,
                  "zero_sign": args.zero_sign,
                  "layer_ids": [lid for _, lid, _ in layers], "layouts": {}}

    packer_fidelity_ok = True
    for layout in args.layouts:
        per_layer, rows_cols = [], []
        agg = {"bits": 0, "n_segments": 0, "run_hist": Counter(),
               "alt_hist": Counter(),
               "pair_hist": np.zeros((PAIR_CAP + 1, PAIR_CAP + 1), dtype=np.int64),
               "period_region_hist": {p: Counter() for p in PERIODS}}
        for name, lid, qw in layers:
            r = analyze_layer(qw, rt_size=args.rt_size, layout=layout,
                              kernel_mapping=args.kernel_mapping,
                              zero_sign=args.zero_sign)
            r["layer_name"], r["layer_id"] = name, lid
            per_layer.append(r)
            rows_cols.append((r["rows"], r["cols"]))
            agg["bits"] += r["bits"]
            agg["n_segments"] += r["n_segments"]
            agg["dense_wires"] = agg.get("dense_wires", 0) + r["dense_wires"]
            agg["run_hist"].update(r["run_hist"])
            agg["alt_hist"].update(r["alt_hist"])
            agg["pair_hist"] += np.array(r["pair_hist"], dtype=np.int64)
            for p in PERIODS:
                agg["period_region_hist"][p].update(r["period_region_hist"][p])

        agg["run_hist"] = {int(k): int(v) for k, v in sorted(agg["run_hist"].items())}
        agg["alt_hist"] = {int(k): int(v) for k, v in sorted(agg["alt_hist"].items())}
        agg["pair_hist"] = agg["pair_hist"].tolist()
        agg["period_region_hist"] = {
            int(p): {int(k): int(v) for k, v in sorted(agg["period_region_hist"][p].items())}
            for p in PERIODS
        }

        designs = derive_designs(agg, rt_size=args.rt_size, rows_cols=rows_cols)
        report(layout, per_layer, agg, designs, args.rt_size)
        dump["layouts"][layout] = {"aggregate": agg, "per_layer": per_layer,
                                   "designs": designs,
                                   "periods": period_summary(np.array(agg["pair_hist"]))}
        if args.verify_packer:
            ok = verify_packer_fidelity(
                designs, layers, rt_size=args.rt_size, layout=layout,
                kernel_mapping=args.kernel_mapping,
                ratio_threshold=args.verify_packer_threshold,
            )
            packer_fidelity_ok = packer_fidelity_ok and ok

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dump, indent=2))
        print(f"\nraw dump -> {out}")
    if args.verify_packer and not packer_fidelity_ok:
        print("\n--verify-packer: at least one design point diverged beyond "
              "tolerance -- see WARNING lines above. Exiting non-zero.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
