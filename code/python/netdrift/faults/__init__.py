"""Fault models.

A :class:`FaultModel` encapsulates a single physical fault mechanism — RTM
domain-wall misalignment, skyrmion collapse, stuck-at, random bit-flip, and so
on. Layers consume the model through a uniform ``inject(weight, state, ctx)``
interface, so swapping one mechanism for another is purely a config change.

CUDA toolchain conventions:

* **Numba** (``faults/kernels/numba_*``) for legacy / non-autograd paths. Used
  by the existing :class:`RTMMisalignmentFault`, where weights are detached
  and gradients flow via a straight-through estimator at the layer level.
* **PyTorch C++ extensions** (``code/cuda/*``) for any new fault model that
  must participate in autograd directly. The interface stays the same; only
  the kernel choice differs.
"""

from netdrift.faults.base import FaultCtx, FaultModel, FaultState, FaultStats
from netdrift.faults.rtm_misalignment import (
    RTMConfig,
    RTMMisalignmentFault,
    RTMState,
)

__all__ = [
    "FaultCtx",
    "FaultModel",
    "FaultState",
    "FaultStats",
    "RTMConfig",
    "RTMMisalignmentFault",
    "RTMState",
]
