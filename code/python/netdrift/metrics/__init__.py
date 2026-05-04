"""Metrics: per-layer collection and pluggable sinks.

Phase 1 ships only the in-layer storage primitive; Phase 2 adds the
``collector``, ``sinks``, and ``plot`` modules. Layers append per-call stats
into a :class:`LayerMetrics` instance attached as ``layer.metrics``. Phase 2's
collector walks ``model.named_modules()`` to harvest these.
"""

from netdrift.metrics.layer import LayerMetrics

__all__ = ["LayerMetrics"]
