"""CUDA kernels backing the fault models.

Two toolchains coexist:

* **Numba CUDA** (this directory) — fast to iterate, runs as JIT-compiled
  device code, used by the legacy RTM simulator. Weights are detached before
  injection, so gradients flow through a straight-through estimator at the
  layer level (no autograd participation in the kernel).
* **PyTorch C++ extensions** (``code/cuda/``) — compiled ahead of time,
  integrate with ``torch.autograd.Function``. Used when a fault model must
  participate directly in autograd (e.g., differentiable fault-aware training
  paths planned for Phase 2).

The choice is documented per fault model. For RTM misalignment we keep Numba
because the legacy reference uses it and we want bit-for-bit equivalence.
"""

from netdrift.faults.kernels.rtm_numba import (
    calc_index_offset_kernel,
    simulate_racetrack_kernel,
)

__all__ = [
    "calc_index_offset_kernel",
    "simulate_racetrack_kernel",
]
