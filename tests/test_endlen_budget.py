"""Budgeted endlen: candidate data contract, host-side selection, scopes.

CPU-safe — no GPU, no torch required. These tests pin the pure-Python budget
selector that decides which endlen merges survive the configured budgets, and
prove the three selection policies and three local scopes behave distinctly.
"""

from __future__ import annotations

import pytest

from netdrift.faults.weight_encoders.budget import BudgetConfig, select_merges
from netdrift.faults.weight_encoders.candidates import MergeCandidate


# ---------------------------------------------------------------------------
# Task 1: MergeCandidate data contract
# ---------------------------------------------------------------------------

def test_merge_candidate_fields():
    c = MergeCandidate(
        layer_idx=0, unit_id_layer=0, unit_id_racetrack=2, unit_id_channel=1,
        start_idx=5, n_flips=3, endlen_gain=9, min_latent_magnitude=0.25,
    )
    assert c.n_flips == 3
    assert c.endlen_gain == 9
    assert c.value_per_flip == 3.0  # endlen_gain / n_flips


# ---------------------------------------------------------------------------
# Task 2: boundary semantics (budget 0 and 1)
# ---------------------------------------------------------------------------

def _candidates_simple():
    # Three non-overlapping merges in one layer (layer_idx=0), one racetrack
    # (rt 0), one channel (ch 0).
    return [
        MergeCandidate(0, 0, 0, 0, start_idx=0,  n_flips=2, endlen_gain=10, min_latent_magnitude=0.9),
        MergeCandidate(0, 0, 0, 0, start_idx=10, n_flips=1, endlen_gain=6,  min_latent_magnitude=0.1),
        MergeCandidate(0, 0, 0, 0, start_idx=20, n_flips=4, endlen_gain=12, min_latent_magnitude=0.5),
    ]


_SIMPLE_TOTALS = dict(
    layer_totals={0: 100},
    racetrack_totals={(0, 0): 64},
    channel_totals={(0, 0): 100},
)


def test_budget_zero_flips_nothing():
    cfg = BudgetConfig(global_budget=0.0, local_budget=1.0, scope="layer", selection="greedy")
    chosen = select_merges(_candidates_simple(), cfg, **_SIMPLE_TOTALS)
    assert chosen == []


def test_budget_one_keeps_all_nonoverlapping():
    cfg = BudgetConfig(global_budget=1.0, local_budget=1.0, scope="layer", selection="greedy")
    chosen = select_merges(_candidates_simple(), cfg, **_SIMPLE_TOTALS)
    assert {c.start_idx for c in chosen} == {0, 10, 20}


# ---------------------------------------------------------------------------
# Task 3: selection ranking modes diverge
# ---------------------------------------------------------------------------

@pytest.fixture
def disagreeing_candidates():
    # Budget admits ~2 flips. Each policy prefers a DIFFERENT merge:
    #   greedy          → max endlen_gain  → A (gain 12, 2 flips)
    #   value_per_flip  → max gain/flip    → B (gain 9,  1 flip → 9.0/flip)
    #   magnitude_aware → min latent mag   → C (mag 0.05, gain 8, 2 flips)
    A = MergeCandidate(0, 0, 0, 0, start_idx=0,  n_flips=2, endlen_gain=12, min_latent_magnitude=0.80)
    B = MergeCandidate(0, 0, 1, 1, start_idx=10, n_flips=1, endlen_gain=9,  min_latent_magnitude=0.50)
    C = MergeCandidate(0, 0, 2, 2, start_idx=20, n_flips=2, endlen_gain=8,  min_latent_magnitude=0.05)
    return [A, B, C]


_DISAGREE_TOTALS = dict(
    layer_totals={0: 30},
    racetrack_totals={(0, 0): 64, (0, 1): 64, (0, 2): 64},
    channel_totals={(0, 0): 10, (0, 1): 10, (0, 2): 10},
)


def test_selection_greedy_keeps_longest_run(disagreeing_candidates):
    # global cap = floor(0.07 * 30) = 2 flips → A (2 flips) fills it exactly.
    cfg = BudgetConfig(global_budget=0.07, local_budget=1.0, scope="layer", selection="greedy")
    chosen = select_merges(disagreeing_candidates, cfg, **_DISAGREE_TOTALS)
    assert [c.start_idx for c in chosen] == [0]  # A


def test_selection_value_per_flip(disagreeing_candidates):
    # B has 9.0/flip > A's 6.0 > C's 4.0. B is 1 flip; cap=2 leaves 1, but the
    # next-best (A) needs 2 → doesn't fit. Only B survives.
    cfg = BudgetConfig(global_budget=0.07, local_budget=1.0, scope="layer", selection="value_per_flip")
    chosen = select_merges(disagreeing_candidates, cfg, **_DISAGREE_TOTALS)
    assert [c.start_idx for c in chosen] == [10]  # B


def test_selection_magnitude_aware(disagreeing_candidates):
    # C has smallest latent magnitude (0.05); 2 flips fills the cap.
    cfg = BudgetConfig(global_budget=0.07, local_budget=1.0, scope="layer", selection="magnitude_aware")
    chosen = select_merges(disagreeing_candidates, cfg, **_DISAGREE_TOTALS)
    assert [c.start_idx for c in chosen] == [20]  # C


# ---------------------------------------------------------------------------
# Task 4: local_budget_scope denominators (incl. partial racetrack)
# ---------------------------------------------------------------------------

def test_scope_racetrack_caps_per_track():
    # Two 1-flip merges in different racetracks + one 2-flip merge. local=0.03,
    # full track=64 → cap floor(0.03*64)=1 per track. The two 1-flip merges fit
    # (different units); the 2-flip merge exceeds its track cap.
    c0 = MergeCandidate(0, 0, 0, 0, start_idx=0,  n_flips=1, endlen_gain=5, min_latent_magnitude=0.5)
    c1 = MergeCandidate(0, 0, 1, 0, start_idx=64, n_flips=1, endlen_gain=5, min_latent_magnitude=0.5)
    c1big = MergeCandidate(0, 0, 1, 0, start_idx=70, n_flips=2, endlen_gain=9, min_latent_magnitude=0.5)
    cfg = BudgetConfig(global_budget=1.0, local_budget=0.03, scope="racetrack", selection="greedy")
    chosen = select_merges(
        [c0, c1, c1big], cfg,
        layer_totals={0: 200},
        racetrack_totals={(0, 0): 64, (0, 1): 64},
        channel_totals={(0, 0): 200},
    )
    assert {c.start_idx for c in chosen} == {0, 64}


def test_scope_racetrack_partial_track_denominator():
    # Trailing partial racetrack has only 20 bits. local=0.5 → cap 10.
    # An 11-flip merge must be rejected.
    c = MergeCandidate(0, 0, 5, 0, start_idx=320, n_flips=11, endlen_gain=20, min_latent_magnitude=0.5)
    cfg = BudgetConfig(global_budget=1.0, local_budget=0.5, scope="racetrack", selection="greedy")
    chosen = select_merges(
        [c], cfg,
        layer_totals={0: 340},
        racetrack_totals={(0, 5): 20},   # partial: 340 % 64 = 20
        channel_totals={(0, 0): 340},
    )
    assert chosen == []  # 11 > floor(0.5*20)=10


def test_scope_channel_denominator():
    # Per-channel cap. channel total=10, local=0.3 → cap floor(0.3*10)=3.
    # A 4-flip merge in channel 0 is rejected; a 3-flip one is accepted.
    big = MergeCandidate(0, 0, 0, 0, start_idx=0, n_flips=4, endlen_gain=8, min_latent_magnitude=0.5)
    ok = MergeCandidate(0, 0, 1, 1, start_idx=64, n_flips=3, endlen_gain=6, min_latent_magnitude=0.5)
    cfg = BudgetConfig(global_budget=1.0, local_budget=0.3, scope="channel", selection="greedy")
    chosen = select_merges(
        [big, ok], cfg,
        layer_totals={0: 100},
        racetrack_totals={(0, 0): 64, (0, 1): 64},
        channel_totals={(0, 0): 10, (0, 1): 10},
    )
    assert {c.start_idx for c in chosen} == {64}


# ---------------------------------------------------------------------------
# Task 5: global ∧ local interaction (whichever binds first)
# ---------------------------------------------------------------------------

def test_local_binds_before_global():
    # Channel 0 has 5 merges (2 flips each); local cap stops it while global
    # still has room, so channel 1's merge still gets applied.
    ch0 = [MergeCandidate(0, 0, 0, 0, start_idx=i * 4, n_flips=2, endlen_gain=10, min_latent_magnitude=0.5) for i in range(5)]
    ch1 = MergeCandidate(0, 0, 9, 1, start_idx=400, n_flips=2, endlen_gain=3, min_latent_magnitude=0.5)
    cfg = BudgetConfig(global_budget=1.0, local_budget=0.5, scope="channel", selection="greedy")
    chosen = select_merges(
        ch0 + [ch1], cfg,
        layer_totals={0: 500},
        racetrack_totals={(0, 0): 64, (0, 9): 64},
        channel_totals={(0, 0): 8, (0, 1): 8},  # ch0 cap floor(0.5*8)=4 → ≤2 merges
    )
    ch0_flips = sum(c.n_flips for c in chosen if c.unit_id_channel == 0)
    assert ch0_flips <= 4
    assert any(c.unit_id_channel == 1 for c in chosen)


def test_global_binds_before_local():
    # Generous local (1.0), tiny global → flipping stops model-wide early.
    cands = [MergeCandidate(0, 0, i, 0, start_idx=i * 64, n_flips=2, endlen_gain=10, min_latent_magnitude=0.5) for i in range(10)]
    cfg = BudgetConfig(global_budget=0.01, local_budget=1.0, scope="racetrack", selection="greedy")
    chosen = select_merges(
        cands, cfg,
        layer_totals={0: 640},
        racetrack_totals={(0, i): 64 for i in range(10)},
        channel_totals={(0, 0): 640},
    )
    # global cap = floor(0.01*640)=6 flips → at most 3 merges of 2 flips.
    assert sum(c.n_flips for c in chosen) <= 6


def test_global_only_workaround():
    # local=1.0 → single model-wide cap behaves like one budget.
    cands = [MergeCandidate(0, 0, i, 0, start_idx=i * 64, n_flips=2, endlen_gain=10, min_latent_magnitude=0.5) for i in range(10)]
    cfg = BudgetConfig(global_budget=0.02, local_budget=1.0, scope="layer", selection="greedy")
    chosen = select_merges(
        cands, cfg,
        layer_totals={0: 640},
        racetrack_totals={(0, i): 64 for i in range(10)},
        channel_totals={(0, 0): 640},
    )
    assert sum(c.n_flips for c in chosen) <= 12  # floor(0.02*640)=12


def test_local_only_workaround():
    # global=1.0 → per-unit caps only, no ceiling.
    cands = [MergeCandidate(0, 0, i, i, start_idx=i * 64, n_flips=2, endlen_gain=10, min_latent_magnitude=0.5) for i in range(4)]
    cfg = BudgetConfig(global_budget=1.0, local_budget=1.0, scope="racetrack", selection="greedy")
    chosen = select_merges(
        cands, cfg,
        layer_totals={0: 256},
        racetrack_totals={(0, i): 64 for i in range(4)},
        channel_totals={(0, i): 64 for i in range(4)},
    )
    # No ceiling, each in its own track under cap → all applied.
    assert len(chosen) == 4
