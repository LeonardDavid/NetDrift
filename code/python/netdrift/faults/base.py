"""FaultModel ABC and supporting state/context dataclasses.

The contract is a single ``inject`` call per forward pass per layer:

.. code-block:: python

    faulty_w, new_state, stats = fault_model.inject(weight, state, ctx)

* ``weight`` is the already-quantized weight tensor (a :class:`QuantizedTensor`
  in Phase 3, but for Phase 1 we accept a raw tensor with metadata in ``ctx``).
* ``state`` carries per-layer simulation state that must persist across calls
  (e.g. RTM index offsets). It is opaque to the layer; only the fault model
  reads/writes it.
* ``ctx`` carries call-specific metadata: layer id, run number, training flag,
  scheme info. The fault model uses ``ctx.training`` to choose deterministic
  vs. stochastic behaviour where applicable.

The returned ``stats`` is a flat dict suitable for direct ingestion by the
metrics collector — each fault model documents its own keys.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class FaultCtx:
    """Per-call context handed from layer to fault model."""

    layer_id: int
    """Auto-assigned layer index from module-tree iteration order (1-based)."""

    layer_name: str
    """Dotted attribute path, e.g. ``"layer1.0.conv1"``."""

    nr_run: int
    """Inference iteration counter, 1-based. Increments every forward pass."""

    training: bool
    """``True`` during fault-aware training, ``False`` during test-time inference."""

    bits: int = 1
    """Bits per quantized value (binary=1). Used by storage layouts in Phase 3."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Free-form bag for fault-model-specific extras (e.g. RT mappings)."""


class FaultState:
    """Marker base class for per-layer fault state. Subclassed by each fault model."""


@dataclass
class FaultStats:
    """Per-call statistics returned by ``inject``.

    Fault models populate any subset of the standard fields and may add
    custom entries via ``extra``. The metrics collector records whichever
    keys the user has enabled.
    """

    bitflips: Optional[int] = None
    """Number of weight bits whose value changed due to the fault."""

    misalign_faults: Optional[int] = None
    """Total misalignment events (RTM-specific)."""

    affected_units: Optional[int] = None
    """Number of independent storage units (racetracks) with non-zero offset."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Fault-model-specific extras."""


class FaultModel(ABC):
    """Abstract base class for all fault models."""

    name: str
    """Short identifier used in configs and metrics keys."""

    @abstractmethod
    def init_state(self, weight_shape: tuple[int, ...], ctx: FaultCtx) -> FaultState:
        """Allocate persistent state for a layer with the given weight shape.

        Called once per layer at construction time (or on the first forward
        pass), before any :meth:`inject` calls. The layer holds the returned
        state and passes it back on every call.
        """

    @abstractmethod
    def inject(
        self,
        weight: torch.Tensor,
        state: FaultState,
        ctx: FaultCtx,
    ) -> tuple[torch.Tensor, FaultState, FaultStats]:
        """Inject faults into ``weight``.

        Args:
            weight: Already-quantized weight tensor. Shape and dtype identical to ``self.weight``.
            state:  Persistent state from a previous call (or :meth:`init_state`).
            ctx:    Per-call context.

        Returns:
            Tuple ``(faulty_weight, new_state, stats)``.
            ``faulty_weight`` has the same shape/dtype as ``weight``.
        """
