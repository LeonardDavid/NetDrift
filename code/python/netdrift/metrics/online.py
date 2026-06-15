"""Per-loop non-scalar metric accumulation for the online metrics phase.

Scalar fault metrics (bitflips, misalign_faults, affected_units, wrong_bits_read)
are summed per loop by the existing ``_loop_metric_delta`` in the runner; this
collector records those per-loop totals as TIME SERIES (one value per loop) and,
at the end, reads the raw per-racetrack arrays off each layer's fault state for
the ``.npz`` dump. It never retains per-loop arrays — memory stays flat.

SEMANTICS (important for the analyst): the per-loop SCALAR ``wrong_bits_read``
is summed over every forward pass (test batch) within that loop, because offset
accumulates across passes. The ``.npz`` ``wrong_read_mask`` is the per-racetrack
count from the FINAL forward pass only (it's read off ``fault_state`` at the
end). So the scalar time-series and the npz mask are NOT the same quantity —
the series is a per-loop sum, the mask is a final-pass snapshot. Don't compare
them elementwise.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

# Which scalar metrics get a per-loop time series in the artifact.
_SERIES_KEYS = ("bitflips", "wrong_bits_read", "misalign_faults", "affected_units")


class OnlineCollector:
    def __init__(self) -> None:
        self._total: dict[str, list] = defaultdict(list)
        self._per_layer: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    def record_loop(self, *, loop_idx: int, totals: dict, per_layer: dict) -> None:
        """Append this loop's per-metric totals to the running time series."""
        for key in _SERIES_KEYS:
            if key in totals:
                self._total[key].append(totals[key])
        for lname, metrics in per_layer.items():
            for key in _SERIES_KEYS:
                if key in metrics:
                    self._per_layer[key][lname].append(metrics[key])

    def as_per_loop(self) -> dict:
        """Return ``{metric: {"total": [...], "per_layer": {layer: [...]}}}``."""
        out: dict[str, dict] = {}
        for key in _SERIES_KEYS:
            if key in self._total:
                out[key] = {
                    "total": list(self._total[key]),
                    "per_layer": {l: list(v) for l, v in self._per_layer[key].items()},
                }
        return out

    def final_raw_arrays(self, layers: dict) -> dict:
        """Read final per-racetrack arrays off layer fault state for the .npz.

        ``layers`` maps layer name -> module (with ``.fault_state``). Returns
        ``{npz_key: ndarray}`` for index_offset and wrong-read masks where present.
        """
        arrays: dict[str, np.ndarray] = {}
        for lname, mod in layers.items():
            state = getattr(mod, "fault_state", None)
            if state is None:
                continue
            offset = getattr(state, "index_offset", None)
            if offset is not None:
                arrays[f"online__final__{lname}__index_offset"] = np.asarray(offset)
            wrong = getattr(state, "last_wrong_read", None)
            if wrong is not None:
                arrays[f"online__final__{lname}__wrong_read_mask"] = np.asarray(wrong)
        return arrays
