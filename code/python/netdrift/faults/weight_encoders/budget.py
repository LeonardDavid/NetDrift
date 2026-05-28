"""Host-side, budget-bounded selection over endlen merge candidates.

Pure Python — no GPU, no torch. Given the candidates emitted across all layers
plus the per-unit totals, rank them by the configured policy and greedily apply
the highest-ranked merges that keep every binding budget under its cap.

Budget semantics: a fraction in ``[0, 1]``. ``0.0`` means flip nothing; ``1.0``
means unbounded (cap == total). A merge is applied only if it keeps BOTH the
local unit under ``local_budget`` AND the model under ``global_budget`` —
whichever binds first. To run global-only set ``local_budget=1.0``; to run
local-only set ``global_budget=1.0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

from netdrift.faults.weight_encoders.candidates import MergeCandidate

Scope = Literal["layer", "racetrack", "channel"]
Selection = Literal["greedy", "value_per_flip", "magnitude_aware"]


@dataclass
class BudgetConfig:
    """Knobs controlling how many flips endlen keeps and which ones.

    Attributes:
        global_budget: Max fraction of ALL model weights that may be flipped.
        local_budget:  Max fraction within each local unit (see ``scope``).
        scope:         Which unit ``local_budget`` is measured against.
        selection:     Ranking policy deciding which candidates survive the cap.
    """

    global_budget: float = 1.0
    local_budget: float = 1.0
    scope: Scope = "layer"
    selection: Selection = "greedy"


def _unit_key(c: MergeCandidate, scope: Scope) -> tuple:
    if scope == "layer":
        return (c.layer_idx,)
    if scope == "racetrack":
        return (c.layer_idx, c.unit_id_racetrack)
    if scope == "channel":
        return (c.layer_idx, c.unit_id_channel)
    raise ValueError(f"unknown scope {scope!r}")


def _unit_total(
    c: MergeCandidate,
    scope: Scope,
    layer_totals: dict,
    racetrack_totals: dict,
    channel_totals: dict,
) -> int:
    if scope == "layer":
        return layer_totals[c.layer_idx]
    if scope == "racetrack":
        return racetrack_totals[(c.layer_idx, c.unit_id_racetrack)]
    if scope == "channel":
        return channel_totals[(c.layer_idx, c.unit_id_channel)]
    raise ValueError(f"unknown scope {scope!r}")


def _rank_key(selection: Selection) -> Callable[[MergeCandidate], tuple]:
    if selection == "greedy":
        # Highest endlen_gain first; tie-break by fewer flips.
        return lambda c: (-c.endlen_gain, c.n_flips)
    if selection == "value_per_flip":
        # Most run-length bought per flip first.
        return lambda c: (-c.value_per_flip, c.n_flips)
    if selection == "magnitude_aware":
        # Flip least-confident signs first (smallest latent magnitude);
        # break ties toward the bigger run-length gain.
        return lambda c: (c.min_latent_magnitude, -c.endlen_gain)
    raise ValueError(f"unknown selection {selection!r}")


def select_merges(
    candidates: list[MergeCandidate],
    cfg: BudgetConfig,
    *,
    layer_totals: dict[int, int],
    racetrack_totals: dict[tuple[int, int], int],
    channel_totals: dict[tuple[int, int], int],
) -> list[MergeCandidate]:
    """Return the budget-bounded, ranked subset of merges to apply.

    Args:
        candidates:       All emitted merges, across every layer.
        cfg:              Budget + scope + selection policy.
        layer_totals:     ``{layer_idx: weight_count}``.
        racetrack_totals: ``{(layer_idx, racetrack_id): bits}`` (partial tracks
                          carry their remainder length).
        channel_totals:   ``{(layer_idx, channel_id): weight_count}``.

    Returns:
        The chosen candidates, in the order they were applied (rank order).
    """
    total_weights = sum(layer_totals.values())
    global_cap = math.floor(cfg.global_budget * total_weights)

    ranked = sorted(candidates, key=_rank_key(cfg.selection))

    global_flipped = 0
    local_flipped: dict[tuple, int] = {}
    chosen: list[MergeCandidate] = []
    for c in ranked:
        if global_flipped + c.n_flips > global_cap:
            continue
        ukey = _unit_key(c, cfg.scope)
        unit_total = _unit_total(
            c, cfg.scope, layer_totals, racetrack_totals, channel_totals
        )
        local_cap = math.floor(cfg.local_budget * unit_total)
        if local_flipped.get(ukey, 0) + c.n_flips > local_cap:
            continue
        chosen.append(c)
        global_flipped += c.n_flips
        local_flipped[ukey] = local_flipped.get(ukey, 0) + c.n_flips
    return chosen
