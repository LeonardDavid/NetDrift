"""Runner-side helpers that feed wandb: protection resolution + per-loop deltas.

CPU-safe (no fault injection, no CUDA) but needs ``torch`` for the real
``QuantizedLinear`` modules the helpers walk via ``isinstance``. Skipped cleanly
when torch is unavailable so the file is collectible everywhere.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from netdrift.quant.layers import QuantizedLinear  # noqa: E402
from netdrift.runner.run import (  # noqa: E402
    _layer_metric_lengths,
    _loop_metric_delta,
    _resolved_protection,
)


def _toy_model() -> nn.Module:
    model = nn.Module()
    model.a = QuantizedLinear(4, 3)
    model.b = QuantizedLinear(3, 2)
    return model


def test_resolved_protection_splits_by_flag() -> None:
    model = _toy_model()
    model.a.protected = True
    model.b.protected = False
    protected, unprotected = _resolved_protection(model)
    assert protected == ["a"]
    assert unprotected == ["b"]


def test_resolved_protection_ignores_plain_modules() -> None:
    model = nn.Module()
    model.lin = nn.Linear(2, 2)  # not a QuantizedLinear → ignored
    protected, unprotected = _resolved_protection(model)
    assert protected == []
    assert unprotected == []


def test_loop_metric_delta_slices_since_snapshot() -> None:
    model = _toy_model()
    # Simulate two prior forward passes already recorded on layer "a".
    model.a.metrics.data["bitflips"].extend([5, 7])
    before = _layer_metric_lengths(model)
    assert before["a"]["bitflips"] == 2

    # A new "loop": two batches on a, one on b.
    model.a.metrics.data["bitflips"].extend([3, 4])
    model.b.metrics.data["bitflips"].append(10)

    totals, per_layer = _loop_metric_delta(model, before, online=["bitflips"])
    # Only the post-snapshot slice counts: a -> 3+4=7, b -> 10.
    assert per_layer["a"]["bitflips"] == 7
    assert per_layer["b"]["bitflips"] == 10
    assert totals["bitflips"] == 17


def test_loop_metric_delta_respects_online_filter() -> None:
    model = _toy_model()
    before = _layer_metric_lengths(model)
    model.a.metrics.data["bitflips"].append(9)
    model.a.metrics.data["misalign_faults"].append(2)

    # Only "bitflips" is online → misalign_faults is excluded.
    totals, per_layer = _loop_metric_delta(model, before, online=["bitflips"])
    assert totals == {"bitflips": 9}
    assert "misalign_faults" not in per_layer.get("a", {})
