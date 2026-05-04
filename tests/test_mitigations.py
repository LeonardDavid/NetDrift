"""Mitigation steps: registry lookup + per-step semantics.

CPU-safe — mitigations operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from netdrift.faults.mitigations import (
    BinomialRevertEdges,
    BinomialRevertMid,
    Even2OddDecrease,
    Even2OddIncrease,
    Odd2EvenDecrease,
    Odd2EvenIncrease,
    get_mitigation,
)


def test_registry_lookup_returns_correct_class() -> None:
    assert isinstance(get_mitigation("bin_revert_mid"), BinomialRevertMid)
    assert isinstance(get_mitigation("bin_revert_edges"), BinomialRevertEdges)
    assert isinstance(get_mitigation("odd2even_dec"), Odd2EvenDecrease)
    assert isinstance(get_mitigation("odd2even_inc"), Odd2EvenIncrease)
    assert isinstance(get_mitigation("even2odd_dec"), Even2OddDecrease)
    assert isinstance(get_mitigation("even2odd_inc"), Even2OddIncrease)


def test_unknown_mitigation_raises() -> None:
    with pytest.raises(KeyError, match="Unknown mitigation"):
        get_mitigation("not_a_thing")


def test_odd2even_dec_brings_odd_offsets_toward_zero() -> None:
    step = Odd2EvenDecrease()
    offset = np.array([[3, -3, 4, 0]], dtype=np.int32)
    misalign = np.zeros_like(offset)
    new_off, new_mis = step.apply(offset, misalign, nr_run=1)
    # 3 → 2 (decrease toward zero); -3 → -2; 4 stays; 0 stays.
    assert new_off.tolist() == [[2, -2, 4, 0]]
    # +1 fault per modified racetrack with originally-nonzero entry.
    assert new_mis.tolist() == [[1, 1, 0, 0]]


def test_even2odd_inc_pushes_even_offsets_outward() -> None:
    step = Even2OddIncrease()
    offset = np.array([[2, -2, 3, 0]], dtype=np.int32)
    misalign = np.zeros_like(offset)
    new_off, new_mis = step.apply(offset, misalign, nr_run=1)
    # 2 → 3 (increase away from zero); -2 → -3; 3 stays (odd); 0 stays.
    assert new_off.tolist() == [[3, -3, 3, 0]]
    assert new_mis.tolist() == [[1, 1, 0, 0]]


def test_bin_revert_mid_zeros_some_small_offsets() -> None:
    step = BinomialRevertMid()
    # 4 negatives + 4 positives — mid revert should zero ~50% of each sign.
    offset = np.array([[-1, -2, -3, -4, 1, 2, 3, 4]], dtype=np.int32)
    misalign = np.zeros_like(offset)
    new_off, _ = step.apply(offset, misalign, nr_run=1)
    n_zero = int((new_off == 0).sum())
    assert n_zero >= 4, "expected at least half the offsets zeroed"


def test_every_nrun_skip_when_not_due() -> None:
    """``every_nrun=2`` skips on odd-numbered runs."""
    step = BinomialRevertMid(every_nrun=2)
    offset = np.array([[1, -1, 2, -2]], dtype=np.int32)
    misalign = np.zeros_like(offset)
    new_off, new_mis = step.apply(offset, misalign, nr_run=1)
    assert np.array_equal(new_off, offset)
    assert np.array_equal(new_mis, misalign)
    # On nr_run=2 the step fires
    new_off, _ = step.apply(offset, misalign, nr_run=2)
    assert (new_off == 0).any()
