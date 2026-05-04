"""Per-layer metric storage.

A :class:`LayerMetrics` object lives on every quantized layer (``layer.metrics``)
and accumulates one entry per forward pass. Aggregation is the job of the
Phase 2 collector — this module just provides the bucket.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class LayerMetrics:
    """A typed bag of per-call lists, keyed by metric name.

    Standard keys (used by :class:`~netdrift.faults.RTMMisalignmentFault`):

    * ``bitflips``         — per-call int.
    * ``misalign_faults``  — per-call int.
    * ``affected_units``   — per-call int.

    Additional keys can be appended freely under ``extra``. The dict is a
    :class:`defaultdict(list)` so consumers can simply ``.append`` without
    pre-declaring the key.
    """

    def __init__(self) -> None:
        self.data: defaultdict[str, list[Any]] = defaultdict(list)

    def record_stats(self, stats) -> None:  # type: ignore[no-untyped-def]
        """Append the populated fields of a :class:`FaultStats` to the bag."""
        if stats.bitflips is not None:
            self.data["bitflips"].append(stats.bitflips)
        if stats.misalign_faults is not None:
            self.data["misalign_faults"].append(stats.misalign_faults)
        if stats.affected_units is not None:
            self.data["affected_units"].append(stats.affected_units)
        for k, v in stats.extra.items():
            self.data[k].append(v)

    def reset(self) -> None:
        """Clear all accumulated metrics. Useful between experiment iterations."""
        self.data.clear()

    def __getitem__(self, key: str) -> list[Any]:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def keys(self):
        return self.data.keys()
