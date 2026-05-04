"""Binomial reversion mitigations for RTM index offsets.

Two strategies, both adapted from the legacy ``metrics/binomial_revert``:

* :class:`BinomialRevertMid`   — zero out 50% of mid-magnitude offsets per
  side. Theoretical model of partial ECC: corrects the *most common* (small)
  offsets, leaving rarer larger offsets untouched.
* :class:`BinomialRevertEdges` — zero out 50% of unique edge values per side.
  Models corrupting the largest offsets, leaving the densely-populated middle
  intact.

Both report the absolute amount of shift removed as additional misalignment
events (the ECC has to shift the wire to undo the offset).
"""

from __future__ import annotations

import numpy as np

from netdrift.faults.mitigations.base import MitigationStep, register_mitigation


def _revert_mid(arr: np.ndarray) -> np.ndarray:
    """Zero out 50% of negative and positive elements closest to zero."""
    flat = arr.flatten()
    negative = flat[flat < 0]
    positive = flat[flat > 0]

    n_neg = int(len(negative) * 0.5)
    n_pos = int(len(positive) * 0.5)

    # Indices into the negative/positive subarrays of the elements to revert.
    # ``argsort`` ascending → smallest-magnitude negatives are at the end (closer to 0);
    # smallest-magnitude positives are at the beginning.
    neg_idx = np.argsort(negative)[-n_neg:] if n_neg > 0 else np.array([], dtype=int)
    pos_idx = np.argsort(positive)[:n_pos] if n_pos > 0 else np.array([], dtype=int)

    # Walk the flat array and zero one occurrence per chosen value (preserves
    # legacy semantics: each value selected zeroes its first matching slot).
    for idx in neg_idx:
        target = negative[idx]
        match = np.where(flat == target)[0]
        if match.size > 0:
            flat[match[0]] = 0
    for idx in pos_idx:
        target = positive[idx]
        match = np.where(flat == target)[0]
        if match.size > 0:
            flat[match[0]] = 0

    return flat.reshape(arr.shape)


def _revert_edges(arr: np.ndarray) -> np.ndarray:
    """Zero out 50% of unique negative and positive *bins* furthest from zero."""
    flat = arr.copy().flatten()
    neg_uniques = np.unique(flat[flat < 0])
    pos_uniques = np.unique(flat[flat > 0])
    n_neg = int(len(neg_uniques) * 0.5)
    n_pos = int(len(pos_uniques) * 0.5)

    targets = np.concatenate(
        [
            neg_uniques[:n_neg] if n_neg > 0 else np.array([]),
            pos_uniques[-n_pos:] if n_pos > 0 else np.array([]),
        ]
    )
    mask = np.isin(flat, targets)
    flat[mask] = 0
    return flat.reshape(arr.shape)


def _track_revert(
    before: np.ndarray,
    after: np.ndarray,
    misalign_faults: np.ndarray,
) -> np.ndarray:
    """Add the absolute shift removed to the per-racetrack fault counter."""
    if misalign_faults.shape == before.shape:
        diff = before != after
        misalign_faults[diff] = misalign_faults[diff] + np.abs(before[diff])
    return misalign_faults


@register_mitigation("bin_revert_mid")
class BinomialRevertMid(MitigationStep):
    """Zero out 50% of small-magnitude offsets per sign (closest-to-zero)."""

    def __init__(self, every_nrun: int = 1) -> None:
        self.every_nrun = every_nrun

    def apply(
        self,
        index_offset: np.ndarray,
        misalign_faults: np.ndarray,
        nr_run: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if nr_run % self.every_nrun != 0:
            return index_offset, misalign_faults
        before = index_offset.copy()
        after = _revert_mid(index_offset)
        misalign_faults = _track_revert(before, after, misalign_faults)
        return after, misalign_faults


@register_mitigation("bin_revert_edges")
class BinomialRevertEdges(MitigationStep):
    """Zero out 50% of unique large-magnitude offsets per sign (edge bins)."""

    def __init__(self, every_nrun: int = 1) -> None:
        self.every_nrun = every_nrun

    def apply(
        self,
        index_offset: np.ndarray,
        misalign_faults: np.ndarray,
        nr_run: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if nr_run % self.every_nrun != 0:
            return index_offset, misalign_faults
        before = index_offset.copy()
        after = _revert_edges(index_offset)
        misalign_faults = _track_revert(before, after, misalign_faults)
        return after, misalign_faults
