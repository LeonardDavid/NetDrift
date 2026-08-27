"""Whether the runner binds the fault model, per training mode.

An attached RTM fault model returns a fresh tensor built from NumPy, which
severs the autograd graph between the layer output and the latent weight (see
``tests/test_faultaware_train.py``). Only the fault-aware loops repair that
path, via ``fault_grad_passthrough``. So binding the fault model during
``mode=train`` with ``fault_aware=none`` cannot inject anything useful — it can
only freeze every binary weight while BatchNorm and Scale keep training.

CPU-safe — no CUDA needed.
"""

from __future__ import annotations

from netdrift.config import ExperimentConfig, TrainCfg
from netdrift.runner.run import _fault_model_for_mode


_SENTINEL = object()  # stands in for a built RTMMisalignmentFault


def _cfg(mode: str, fault_aware: str) -> ExperimentConfig:
    return ExperimentConfig(training=TrainCfg(mode=mode, fault_aware=fault_aware))


def test_plain_training_does_not_bind_the_fault_model() -> None:
    """mode=train + fault_aware=none: binding it would only kill weight gradients."""
    assert _fault_model_for_mode(_cfg("train", "none"), _SENTINEL) is None


def test_fault_aware_training_binds_the_fault_model() -> None:
    """The fault-aware loops need it bound; they restore the gradient path themselves."""
    for fault_aware in ("ste_inject", "regularization", "kd"):
        cfg = _cfg("train", fault_aware)
        assert _fault_model_for_mode(cfg, _SENTINEL) is _SENTINEL, fault_aware


def test_test_mode_binds_the_fault_model() -> None:
    """The rt_error sweep is the whole point of test mode — never unbind there."""
    assert _fault_model_for_mode(_cfg("test", "none"), _SENTINEL) is _SENTINEL


def test_none_passes_through_untouched() -> None:
    """Full-precision runs build no fault model; the guard must not invent one."""
    assert _fault_model_for_mode(_cfg("train", "none"), None) is None
    assert _fault_model_for_mode(_cfg("test", "none"), None) is None
