"""NetDrift: Binary/Quantized Neural Networks on unreliable Racetrack Memory.

Top-level package. Submodules:

* ``netdrift.quant``     — QuantScheme abstractions (binary, ternary, int_uniform, ...).
* ``netdrift.faults``    — FaultModel abstractions (RTM misalignment, skyrmion, ...).
* ``netdrift.storage``   — WeightLayout abstractions (row/col/mix, ECC, replicated, ...).
* ``netdrift.models``    — model topologies, registry, and in-place quantization.
* ``netdrift.training``  — training/test loops with optional fault-aware modes.
* ``netdrift.metrics``   — per-layer metric collection and JSONL/W&B sinks.
* ``netdrift.data``      — dataset wrappers.
* ``netdrift.config``    — YAML schema and loader.
* ``netdrift.runner``    — orchestration entry point (``python -m netdrift.runner.run``).

Importing this package eagerly imports ``numba.cuda`` so that Numba acquires
its CUDA driver handle before PyTorch initializes the primary context. On
some systems the reverse import order causes a segfault inside Numba's
``cuda.get_current_device()``. Doing it once here is cheap and avoids
ordering-sensitive import chains in user code.
"""

# Force Numba to use the NVIDIA Python bindings (``cuda-python``) instead of
# its built-in toolkit probing. Without this, Numba can pick up a different
# CUDA runtime than the one PyTorch was built against (e.g. conda's 12.4
# alongside PyTorch's bundled 12.1) and segfault on the first kernel call.
# Setting ``setdefault`` lets users override via the env var if needed.
import os as _os

_os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

# Eager Numba-CUDA import: ensures Numba initializes its driver before
# PyTorch claims the primary CUDA context. The plain ``import`` only loads
# the Python wrappers; we don't try to eagerly *acquire* a CUDA context here
# because that would force every import of netdrift (CPU-only test runs
# included) to talk to the driver.
try:
    import numba.cuda as _numba_cuda  # noqa: F401  (side-effect import)
except ImportError:
    pass

# Suppress NumbaPerformanceWarning ("low occupancy", "host array used in CUDA
# kernel", etc). These fire on every launch of the RTM fault kernels because
# small layers (e.g. fc2 with 10 outputs) inherently produce small grids.
# They're informational, not actionable for our use-case, and they clobber the
# tqdm progress bars during inference.
import warnings as _warnings

try:
    from numba.core.errors import NumbaPerformanceWarning as _NumbaPerfWarning
    _warnings.filterwarnings("ignore", category=_NumbaPerfWarning)
except ImportError:
    pass

__all__ = [
    "quant",
    "faults",
    "storage",
    "models",
    "training",
    "metrics",
    "data",
    "config",
    "runner",
]
