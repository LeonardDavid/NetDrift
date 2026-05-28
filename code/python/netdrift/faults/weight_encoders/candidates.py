"""Candidate merge produced by the endlen *emit* pass.

The emit pass records, for each block-merge endlen would consider, where it is,
how many signs it would flip, how much run-length it buys, and the smallest
latent-FP magnitude in its span. The host-side budget selector then ranks and
applies a budget-bounded subset of these. Splitting emit from apply lets the
selector enforce a cross-layer global budget and magnitude-aware ranking that a
per-racetrack CUDA thread cannot coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MergeCandidate:
    """One candidate block-merge (flip a contiguous span of signs).

    Coordinates are in the 2D racetrack-aligned view of one layer's weights.
    The three ``unit_id_*`` fields identify which local-budget unit this merge
    belongs to under each ``local_budget_scope``; the selector reads only the
    one matching the configured scope.

    Attributes:
        layer_idx:            Index of the layer this merge belongs to.
        unit_id_layer:        Always 0 within a layer (the layer itself is the
                              unit; the layer is identified by ``layer_idx``).
        unit_id_racetrack:    Row-block index of the racetrack in the 2D view.
        unit_id_channel:      Output-channel (dim-0) index in the original tensor.
        start_idx:            Flat index of the span start in the 2D view
                              (C-order: ``row * cols + col``).
        n_flips:              Number of signs the merge flips.
        endlen_gain:          Combined run length if merged (the endlen score).
        min_latent_magnitude: Smallest ``|latent FP weight|`` over the span.
    """

    layer_idx: int
    unit_id_layer: int
    unit_id_racetrack: int
    unit_id_channel: int
    start_idx: int
    n_flips: int
    endlen_gain: int
    min_latent_magnitude: float

    @property
    def value_per_flip(self) -> float:
        """Run-length bought per sign flipped — the ``value_per_flip`` ranking key."""
        return self.endlen_gain / self.n_flips if self.n_flips else 0.0
