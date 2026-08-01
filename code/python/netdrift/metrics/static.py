"""Static-weight metrics computed on the laid-out racetrack view.

Pure PyTorch/NumPy — no numba/CUDA — so it runs on CPU (tests) and GPU alike.
Every layout-dependent metric reshapes the weight via
``netdrift.faults.layout._layout_weight_for_racetrack`` first, so the metrics
see exactly what the RTM fault simulation sees (honouring rt_size, ROW/COL
mapping, and conv kernel mapping).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from netdrift.faults.layout import (
    _layout_weight_for_racetrack,
    compute_index_offset_shape,
    next_pow2,
)


def _count_alternating_sequences(seg_signs: list[int]) -> int:
    """Number of alternating sequences in one segment (list of +1/-1 ints).

    An alternating sequence is a maximal group of >=2 consecutive length-1
    sign-runs. Mirrors the streak logic in :func:`_block_runlength_for_rows`;
    used to build the per-segment raw array under ``want_raw``.
    """
    if not seg_signs:
        return 0
    count = 0
    run_len = 1
    cur = seg_signs[0]
    alt_streak = 0
    for s in seg_signs[1:]:
        if s == cur:
            run_len += 1
        else:
            if run_len == 1:
                alt_streak += 1
            else:
                if alt_streak >= 2:
                    count += 1
                alt_streak = 0
            cur = s
            run_len = 1
    if run_len == 1:
        alt_streak += 1
    elif alt_streak >= 2:
        count += 1
        alt_streak = 0
    if alt_streak >= 2:
        count += 1
    return count


def _block_runlength_for_rows(
    rows: torch.Tensor, rt_size: int
) -> tuple[int, int, int, dict[int, int], dict[int, int]]:
    """Count sign blocks / runs along each physical racetrack segment.

    ``rows`` is the laid-out 2D view (each row is one racetrack lane). Each row
    is split into contiguous ``rt_size``-wide segments (the physical racetracks);
    runs are counted within a segment and never span the boundary.

    Returns ``(pos_blocks, neg_blocks, sign_transitions, run_length_histogram,
    alternating_seq_histogram)`` aggregated over all segments. An alternating
    sequence is a maximal group of >=2 consecutive length-1 sign-runs; its
    length is the number of those runs (overlay: length-1 runs are still counted
    as blocks/runs too). Sign uses ``w > 0 -> +1`` so exactly-zero weights map to
    -1, matching ``BinaryScheme`` (``_binarize_pure``).
    """
    pos_blocks = 0
    neg_blocks = 0
    transitions = 0
    runlen = Counter()
    alt_hist = Counter()  # alternating-sequence length -> count

    # Match the binary quantizer EXACTLY: BinaryScheme uses ``w > 0 -> +1`` so
    # exactly-zero weights binarize to -1. "Metrics see what the sim sees."
    # Move to host as nested Python lists in ONE transfer, then loop over ints.
    # Calling ``.item()`` per element on a CUDA tensor would synchronize on every
    # access (~millions for a large layer) and dominate runtime; ``.tolist()``
    # pays a single device→host copy and the rest is pure Python.
    signs_rows = torch.where(rows > 0, 1, -1).cpu().tolist()  # list[list[int]]
    n_cols = len(signs_rows[0]) if signs_rows else 0
    for start in range(0, n_cols, rt_size):
        for row in signs_rows:
            seg = row[start:start + rt_size]
            if not seg:
                continue
            run_len = 1
            cur = seg[0]
            # Streak of consecutive length-1 runs. A run of length 1 extends it;
            # any run of length >=2 ends it (flushing a >=2 streak to alt_hist).
            alt_streak = 0
            for s in seg[1:]:
                if s == cur:
                    run_len += 1
                else:
                    transitions += 1
                    runlen[run_len] += 1
                    if cur > 0:
                        pos_blocks += 1
                    else:
                        neg_blocks += 1
                    # A run just closed: update the alternating streak.
                    if run_len == 1:
                        alt_streak += 1
                    else:
                        if alt_streak >= 2:
                            alt_hist[alt_streak] += 1
                        alt_streak = 0
                    cur = s
                    run_len = 1
            # close the final run of this segment
            runlen[run_len] += 1
            if cur > 0:
                pos_blocks += 1
            else:
                neg_blocks += 1
            # Account the final run in the streak, then flush at segment end.
            if run_len == 1:
                alt_streak += 1
            else:
                if alt_streak >= 2:
                    alt_hist[alt_streak] += 1
                alt_streak = 0
            if alt_streak >= 2:
                alt_hist[alt_streak] += 1

    return pos_blocks, neg_blocks, transitions, dict(runlen), dict(alt_hist)


def _histogram(values: torch.Tensor, n_bins: int, hist_max: float) -> dict[int, int]:
    """Fixed-edge histogram over ``[0, hist_max]`` with ``n_bins`` bins.

    Values >= ``hist_max`` land in the final bin. Returns ``{bin_index: count}``
    with stable integer keys so histograms are comparable across runs.
    """
    if values.numel() == 0:
        return {}
    # Build edges on the values' device so bucketize doesn't hit a CPU/GPU mix.
    edges = torch.linspace(0.0, hist_max, n_bins + 1, device=values.device)
    idx = torch.bucketize(values.flatten(), edges[1:-1].contiguous())
    counts = torch.bincount(idx, minlength=n_bins)
    return {int(i): int(c) for i, c in enumerate(counts.tolist())}


def block_size_histogram_from_runs(run_length_histogram: dict) -> dict:
    """Map a run-length histogram to a padded-racetrack-size histogram.

    Each run of length L becomes a racetrack of padded length next_pow2(L)
    (cap 64). Runs of length > 64 split into ceil(L/64) racetracks; approximate
    that here as: a run L>64 contributes ceil(L/64) racetracks of size 64 for
    all-but-last plus next_pow2(L - 64*(chunks-1)) for the last. Since blocks are
    found within rt_size<=64 segments, L<=64 in practice and this is a straight
    next_pow2 map.
    """
    out: dict[int, int] = {}
    for length, count in run_length_histogram.items():
        L = int(length)
        if L <= 64:
            p = next_pow2(L)
            out[p] = out.get(p, 0) + count
        else:
            chunks = (L + 63) // 64
            for _ in range(chunks - 1):
                out[64] = out.get(64, 0) + count
            last = L - 64 * (chunks - 1)
            p = next_pow2(last)
            out[p] = out.get(p, 0) + count
    return out


def _magnitude_stats(
    weight: torch.Tensor,
    per_channel_scale=None,
    *,
    n_bins: int = 32,
    hist_max: float = 2.0,
) -> dict:
    """Distance-to-threshold (== |w| against the effective threshold) stats.

    ``per_channel_scale`` (shape ``(out_channels,)``) divides ``|w|`` to the
    effective decision boundary. ``None`` (binary scheme) leaves ``|w|`` as is.
    Returns ``{"mean", "std", "histogram"}``.
    """
    w = weight.detach().float()
    if per_channel_scale is not None:
        scale = per_channel_scale.detach().float().to(w.device)
        # Broadcast over the output-channel axis (dim 0).
        view = [scale.shape[0]] + [1] * (w.dim() - 1)
        w = w / scale.reshape(view)
    dist = w.abs()
    return {
        "mean": float(dist.mean().item()),
        "std": float(dist.std(unbiased=False).item()),
        "histogram": _histogram(dist, n_bins, hist_max),
    }


@dataclass
class StaticLayerMetrics:
    """Static (rt_error-independent) metrics for one layer at one snapshot."""

    block_count: dict[str, int]
    sign_transitions: int
    run_length_histogram: dict[int, int]
    alternating_seq_histogram: dict[int, int]
    weight_magnitude: dict
    dist_to_threshold: dict
    # (n_racetracks, 1) for BLOCK/UNITS (data-dependent block/wire counts);
    # a real rectangular grid otherwise. UNITS requires ``units_params`` to be
    # passed to compute_static_metrics — see that function's docstring.
    n_racetracks: tuple[int, int]
    # Optional raw per-racetrack arrays (only when want_raw=True), for .npz.
    # Typed as Any to avoid importing numpy at module scope (kept lazy in the
    # one code path that builds it); it is an ``np.ndarray`` when present.
    raw_sign_transitions: Optional[Any] = None
    raw_alternating_lengths: Optional[Any] = None
    block_racetrack_count: int = 0
    block_size_histogram: dict = field(default_factory=dict)
    block_padding_overhead: int = 0


def compute_static_metrics(
    weight: torch.Tensor,
    rt_mapping: str,
    kernel_mapping: Optional[str],
    rt_size: int,
    *,
    base_layout: Optional[str] = None,
    units_params: Optional[tuple[int, int, int]] = None,
    per_channel_scale=None,
    want_raw: bool = False,
) -> StaticLayerMetrics:
    """Compute static metrics for one layer on its laid-out racetrack view.

    Magnitude/threshold stats use the RAW weight (latent FP values), since
    binarization discards magnitude. Block/run/transition metrics use the
    laid-out, binarized racetrack view.

    For ``rt_mapping in ("BLOCK", "UNITS")`` the metrics are computed on that
    mapping's underlying base segmentation (``base_layout``, ROW or COL) —
    BLOCK's blocks / UNITS's wires are extracted from that view, so the
    block/run/size metrics must match it.

    ``units_params`` is ``(threshold, max_period, pool_guard)`` and is required
    (non-``None``) when ``rt_mapping == "UNITS"``, to pack wires via
    ``build_unit_buckets`` for the racetrack count. It has no effect for any
    other mapping. See ``n_racetracks`` below for what happens when it is
    missing.
    """
    # BLOCK/UNITS have no rectangular layout of their own; segment on the base
    # layout instead (both are data-dependent packings on top of ROW/COL).
    layout_mapping = rt_mapping
    if rt_mapping in ("BLOCK", "UNITS"):
        layout_mapping = (base_layout or "ROW").upper()
    w_2d, _ = _layout_weight_for_racetrack(weight, layout_mapping, kernel_mapping)
    pos, neg, transitions, runlen, alt_hist = _block_runlength_for_rows(w_2d, rt_size)

    # Block size histogram and padding overhead from run-length histogram
    size_hist = block_size_histogram_from_runs(runlen)
    block_rt_count = sum(size_hist.values())
    # padding overhead: sum over racetracks of (padded_len - actual run length)
    padding_overhead = 0
    for length, count in runlen.items():
        L = int(length)
        padding_overhead += count * (next_pow2(L) - L) if L <= 64 else 0

    raw_st = None
    raw_alt = None
    if want_raw:
        import numpy as np
        # Per-segment transition counts as a 2D array (n_rows, n_segments).
        signs = torch.where(w_2d > 0, 1, -1)  # match BinaryScheme: sign(0) = -1
        n_cols = signs.shape[1]
        n_seg = (n_cols + rt_size - 1) // rt_size
        st = np.zeros((signs.shape[0], n_seg), dtype=np.int32)
        alt = np.zeros((signs.shape[0], n_seg), dtype=np.int32)
        signs_rows = signs.cpu().tolist()
        for si, start in enumerate(range(0, n_cols, rt_size)):
            seg = signs[:, start:start + rt_size]
            if seg.shape[1] > 1:
                st[:, si] = (seg[:, 1:] != seg[:, :-1]).sum(dim=1).cpu().numpy()
            # Per-row count of alternating sequences in this segment.
            for ri, row in enumerate(signs_rows):
                sub = row[start:start + rt_size]
                alt[ri, si] = _count_alternating_sequences(sub)
        raw_st = st
        raw_alt = alt

    # n_racetracks is the RACETRACK GRID (compute_index_offset_shape), NOT the
    # laid-out matrix shape — this is what the spec §4 schema, the runner's
    # _metrics_meta builder, and the RTM fault state all mean by "racetracks".
    # kernel_size=None lets it derive from the 4D shape for conv (ignored for linear).
    # BLOCK has no rectangular grid; its racetrack count is the number of
    # sign-blocks, reported as (n_blocks, 1) to keep the 2-tuple contract.
    if rt_mapping == "BLOCK":
        n_rt = (block_rt_count, 1)
    elif rt_mapping == "UNITS":
        # UNITS has no rectangular grid either; its racetrack count is the
        # number of packed wires (data-dependent — a wire can pool several
        # short runs), reported as (n_wires, 1) to keep the 2-tuple contract.
        # Mirrors runner/run.py::_metrics_meta's UNITS branch exactly.
        if units_params is None:
            # Do NOT fall back to compute_index_offset_shape here: that would
            # silently report a dense ROW/COL racetrack count for a units
            # layer, which is wrong data, not missing data. Fail loudly
            # instead so the caller notices and supplies units_params (or
            # attaches a fault model first — see snapshots.py::capture_snapshot).
            raise ValueError(
                "compute_static_metrics: rt_mapping='UNITS' requires "
                "units_params=(threshold, max_period, pool_guard) to count "
                "racetracks (packed wires), but none were supplied. Pass "
                "units_params explicitly, or ensure the layer's fault_model "
                "(RTMConfig) is attached first so the caller can read "
                "mod.fault_model.cfg.units_threshold/units_max_period/"
                "units_pool_guard."
            )
        from netdrift.faults.packing import build_unit_buckets
        threshold, max_period, pool_guard = units_params
        buckets = build_unit_buckets(
            w_2d, rt_size, threshold=threshold, max_period=max_period,
            pool_guard=pool_guard,
        )
        n_wires = int(sum(b.weight_grid.shape[0] for b in buckets.values()))
        n_rt = (n_wires, 1)
    else:
        n_rt = compute_index_offset_shape(
            tuple(weight.shape), rt_size=rt_size, rt_mapping=rt_mapping, kernel_size=None
        )
    return StaticLayerMetrics(
        block_count={"pos": pos, "neg": neg, "total": pos + neg},
        sign_transitions=transitions,
        run_length_histogram=runlen,
        alternating_seq_histogram=alt_hist,
        weight_magnitude=_magnitude_stats(weight, None),  # |w| latent magnitude
        dist_to_threshold=_magnitude_stats(weight, per_channel_scale),
        n_racetracks=n_rt,
        raw_sign_transitions=raw_st,
        raw_alternating_lengths=raw_alt,
        block_racetrack_count=block_rt_count,
        block_size_histogram=size_hist,
        block_padding_overhead=padding_overhead,
    )
