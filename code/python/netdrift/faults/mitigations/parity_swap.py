"""Parity-swap mitigations: convert offsets between odd and even values.

These strategies model deliberate alternating-pattern repairs: shifting an
odd offset by ±1 produces an even offset (and vice versa). The shift itself
counts as one extra misalignment event per modified racetrack.

Direction matters because the wire is bounded — *decrease* moves toward zero,
*increase* moves away.
"""

from __future__ import annotations

import numpy as np

from netdrift.faults.mitigations.base import MitigationStep, register_mitigation


def _track_parity(
    before: np.ndarray,
    after: np.ndarray,
    misalign_faults: np.ndarray,
    *,
    only_nonzero_before: bool,
) -> np.ndarray:
    """Add 1 fault per modified racetrack, optionally skipping originally-zero entries."""
    if misalign_faults.shape == before.shape:
        if only_nonzero_before:
            diff = (before != after) & (before != 0)
        else:
            diff = before != after
        misalign_faults[diff] = misalign_faults[diff] + 1
    return misalign_faults


class _ParityBase(MitigationStep):
    """Common machinery for the four parity-swap variants."""

    direction: int                  # +1 (increase) or -1 (decrease)
    parity_keep: int                # 0 → modify odd; 1 → modify even
    only_nonzero_before: bool       # whether to ignore originally-zero entries

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
        # Build the per-element selector: parity_keep=0 selects odd entries
        # (which we modify); parity_keep=1 selects nonzero even entries.
        if self.parity_keep == 0:  # odd → even
            mask = (index_offset % 2) != 0
        else:                       # even → odd (skip zeros)
            mask = (index_offset != 0) & ((index_offset % 2) == 0)

        # Decrease moves toward zero; increase moves away.
        sign = np.sign(index_offset)
        delta = self.direction * sign  # decrease → -sign(x); increase → +sign(x)
        after = np.where(mask, index_offset + delta, index_offset)

        misalign_faults = _track_parity(
            before, after, misalign_faults,
            only_nonzero_before=self.only_nonzero_before,
        )
        return after, misalign_faults


@register_mitigation("odd2even_dec")
class Odd2EvenDecrease(_ParityBase):
    """Convert odd offsets to even by moving one step toward zero."""

    direction = -1
    parity_keep = 0
    only_nonzero_before = True  # legacy: skip originally-zero entries


@register_mitigation("odd2even_inc")
class Odd2EvenIncrease(_ParityBase):
    """Convert odd offsets to even by moving one step away from zero."""

    direction = +1
    parity_keep = 0
    only_nonzero_before = True


@register_mitigation("even2odd_dec")
class Even2OddDecrease(_ParityBase):
    """Convert nonzero even offsets to odd by moving one step toward zero."""

    direction = -1
    parity_keep = 1
    only_nonzero_before = False


@register_mitigation("even2odd_inc")
class Even2OddIncrease(_ParityBase):
    """Convert nonzero even offsets to odd by moving one step away from zero."""

    direction = +1
    parity_keep = 1
    only_nonzero_before = False
