"""pytest configuration for NetDrift tests.

Inserts ``code/python/`` onto ``sys.path`` so the ``netdrift`` package is
importable when tests are run from the repo root with ``pytest tests/``.
GPU-required tests are marked with ``@pytest.mark.cuda``; selecting via
``-m cuda`` or ``-m "not cuda"`` lets you run only the CPU-safe subset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "code" / "python"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Force Numba to use the NVIDIA Python bindings (``cuda-python``) instead of
# its built-in toolkit probing. This avoids segfaults from CUDA-runtime ABI
# mismatches when the conda env has a different toolkit installed than the
# one PyTorch was built against (we hit this with conda 12.4 vs torch 12.1).
import os  # noqa: E402

os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

# Eager netdrift import so ``numba.cuda`` initializes before PyTorch claims
# the primary CUDA context. See netdrift/__init__.py for the rationale.
import netdrift  # noqa: E402, F401


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``cuda`` marker so unmarked-marker warnings don't fire."""
    config.addinivalue_line(
        "markers",
        "cuda: GPU-required test (skipped automatically when CUDA is unavailable)",
    )


def _numba_cuda_works() -> bool:
    """Probe whether Numba's CUDA driver can allocate without segfaulting.

    Some environments (driver/library mismatch, Numba+PyTorch primary-context
    conflicts) cause hard segfaults inside ``create_xoroshiro128p_states`` or
    similar low-level Numba calls — even when ``cuda.is_available()`` reports
    True. We probe in a subprocess so a failure doesn't take down the whole
    test session.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent("""
        from numba import cuda
        from numba.cuda.random import create_xoroshiro128p_states
        cuda.select_device(0)
        rng = create_xoroshiro128p_states(64, seed=1)
        print('OK')
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, timeout=30, text=True,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and "OK" in result.stdout


_NUMBA_CUDA_OK = None  # cached probe result


@pytest.fixture(autouse=True)
def _skip_cuda_when_absent(request: pytest.FixtureRequest) -> None:
    """Auto-skip ``@pytest.mark.cuda`` tests on machines without a working GPU."""
    if request.node.get_closest_marker("cuda"):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable — GPU-required test")

        global _NUMBA_CUDA_OK
        if _NUMBA_CUDA_OK is None:
            _NUMBA_CUDA_OK = _numba_cuda_works()
        if not _NUMBA_CUDA_OK:
            pytest.skip(
                "Numba+CUDA segfaults in this environment "
                "(affects legacy code too); fix the driver/library mismatch first"
            )
