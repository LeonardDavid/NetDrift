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
    # bitflips is a STOCK metric → end-of-loop SNAPSHOT (last value in the
    # post-snapshot slice), NOT the batch-sum. a's slice [3,4] -> 4; b -> 10.
    assert per_layer["a"]["bitflips"] == 4
    assert per_layer["b"]["bitflips"] == 10
    assert totals["bitflips"] == 14  # sum of per-layer snapshots, not 3+4+10


def test_loop_metric_delta_flow_metric_is_summed() -> None:
    model = _toy_model()
    before = _layer_metric_lengths(model)
    # misalign_faults is a FLOW metric → summed over the loop's batches.
    model.a.metrics.data["misalign_faults"].extend([2, 3, 5])
    totals, per_layer = _loop_metric_delta(model, before, online=["misalign_faults"])
    assert per_layer["a"]["misalign_faults"] == 10  # 2+3+5, summed
    assert totals["misalign_faults"] == 10


def test_loop_metric_delta_stock_vs_flow_together() -> None:
    model = _toy_model()
    before = _layer_metric_lengths(model)
    # Same loop, both metric kinds: stock snapshots, flow sums.
    model.a.metrics.data["affected_units"].extend([100, 150, 175])  # stock -> 175
    model.a.metrics.data["misalign_faults"].extend([4, 4, 4])       # flow  -> 12
    totals, per_layer = _loop_metric_delta(
        model, before, online=["affected_units", "misalign_faults"]
    )
    assert per_layer["a"]["affected_units"] == 175  # last (snapshot)
    assert per_layer["a"]["misalign_faults"] == 12  # sum


def test_summarize_layer_metrics_by_rt_error_slices_per_boundary() -> None:
    from netdrift.runner.run import (
        _layer_metric_lengths,
        _summarize_layer_metrics_by_rt_error,
    )

    model = _toy_model()
    # rt_error #1 boundary: lengths are 0 here.
    b1 = _layer_metric_lengths(model)
    model.a.metrics.data["bitflips"].extend([1, 2, 3])  # 3 passes for rt #1
    # rt_error #2 boundary: capture lengths after rt #1's passes.
    b2 = _layer_metric_lengths(model)
    model.a.metrics.data["bitflips"].extend([10, 20])    # 2 passes for rt #2

    bounds = [(1e-07, b1), (4.55e-07, b2)]
    out = _summarize_layer_metrics_by_rt_error(model, bounds)

    assert [seg["rt_error"] for seg in out] == [1e-07, 4.55e-07]
    # rt #1 gets the first slice, rt #2 the remainder — NOT concatenated.
    assert out[0]["layer_metrics"]["a"]["bitflips"] == [1, 2, 3]
    assert out[1]["layer_metrics"]["a"]["bitflips"] == [10, 20]


def test_loop_metric_delta_respects_online_filter() -> None:
    model = _toy_model()
    before = _layer_metric_lengths(model)
    model.a.metrics.data["bitflips"].append(9)
    model.a.metrics.data["misalign_faults"].append(2)

    # Only "bitflips" is online → misalign_faults is excluded.
    totals, per_layer = _loop_metric_delta(model, before, online=["bitflips"])
    assert totals == {"bitflips": 9}
    assert "misalign_faults" not in per_layer.get("a", {})


def test_metrics_track_flags_levels():
    from netdrift.runner.run import _metrics_track_flags

    assert _metrics_track_flags("none") == set()
    assert _metrics_track_flags("offline") == set()
    online = _metrics_track_flags("online")
    assert {"bitflips", "misalign_faults", "affected_units", "wrong_bits_read"} <= online
    assert _metrics_track_flags("all") == online


def test_metrics_meta_has_geometry_and_context():
    from netdrift.config.schema import ExperimentConfig
    from netdrift.quant.binary import BinaryScheme
    from netdrift.runner.run import _metrics_meta

    lin = QuantizedLinear(8, 4, bias=False)
    lin.attach_scheme(BinaryScheme())
    lin.rt_mapping = "ROW"
    lin.layer_id = 1
    lin.layer_name = "lin"
    model = nn.Sequential()
    model.add_module("lin", lin)

    cfg = ExperimentConfig()
    cfg.model.name = "vgg3_fmnist"
    cfg.storage.rt_size = 16
    meta = _metrics_meta(cfg, model, category="cat1_baseline", subcategory=None)
    assert meta["model"] == "vgg3_fmnist"
    assert meta["category"] == "cat1_baseline"
    assert meta["storage"]["rt_size"] == 16
    assert any(l["name"] == "lin" for l in meta["layers"])
    geom = next(l for l in meta["layers"] if l["name"] == "lin")
    assert geom["rt_mapping"] == "ROW"
    assert geom["total_weights"] == 32
    assert geom["protected"] is False  # unprotected by default
    # weights block: total = protected + unprotected, summed over quantized layers.
    w = meta["weights"]
    assert w["total"] == 32 and w["unprotected"] == 32 and w["protected"] == 0
    assert w["total"] == w["protected"] + w["unprotected"]


def test_metrics_meta_weights_split_by_protection():
    import torch.nn as nn
    from netdrift.config.schema import ExperimentConfig
    from netdrift.quant.binary import BinaryScheme
    from netdrift.runner.run import _metrics_meta

    # Two layers, one protected → counts must split correctly.
    a = QuantizedLinear(8, 4, bias=False)   # 32 weights, protected
    b = QuantizedLinear(4, 2, bias=False)   # 8 weights, unprotected
    for layer, name, lid in ((a, "a", 1), (b, "b", 2)):
        layer.attach_scheme(BinaryScheme())
        layer.rt_mapping = "ROW"
        layer.layer_id = lid
        layer.layer_name = name
    a.protected = True
    model = nn.Sequential()
    model.add_module("a", a)
    model.add_module("b", b)

    cfg = ExperimentConfig()
    cfg.storage.rt_size = 16
    w = _metrics_meta(cfg, model, category="cat1_baseline", subcategory=None)["weights"]
    assert w["protected"] == 32
    assert w["unprotected"] == 8
    assert w["total"] == 40


def test_setup_run_dir_inserts_category_subcategory(tmp_path):
    from netdrift.config.schema import ExperimentConfig
    from netdrift.runner.run import _setup_run_dir

    cfg = ExperimentConfig()
    cfg.experiment.name = "vgg7_cifar10_w1a1_rtm"
    cfg.experiment.output_dir = str(tmp_path)

    # category + subcategory -> both become path segments before the timestamp.
    # No W&B project is involved here — layout keys off the labels alone.
    run_dir, ts = _setup_run_dir(cfg, "cat5_regularizer", "cat5_regularizer_lam0p01_crit-h128p0")
    assert run_dir.name == ts  # timestamp is the leaf
    parts = run_dir.relative_to(tmp_path).parts
    assert parts == (
        "vgg7_cifar10_w1a1_rtm",
        "cat5_regularizer",
        "cat5_regularizer_lam0p01_crit-h128p0",
        ts,
    )
    assert run_dir.is_dir()


def test_setup_run_dir_category_only(tmp_path):
    from netdrift.config.schema import ExperimentConfig
    from netdrift.runner.run import _setup_run_dir

    cfg = ExperimentConfig()
    cfg.experiment.name = "exp"
    cfg.experiment.output_dir = str(tmp_path)

    run_dir, ts = _setup_run_dir(cfg, "cat1_baseline", None)
    assert run_dir.relative_to(tmp_path).parts == ("exp", "cat1_baseline", ts)


def test_setup_run_dir_no_category_falls_back(tmp_path):
    from netdrift.config.schema import ExperimentConfig
    from netdrift.runner.run import _setup_run_dir

    cfg = ExperimentConfig()
    cfg.experiment.name = "exp"
    cfg.experiment.output_dir = str(tmp_path)

    # No category (e.g. ad-hoc run, no --wandb-category): original layout, no
    # empty segments inserted.
    run_dir, ts = _setup_run_dir(cfg, None, None)
    assert run_dir.relative_to(tmp_path).parts == ("exp", ts)


def test_sanitize_path_segment_replaces_unsafe_chars():
    from netdrift.runner.run import _sanitize_path_segment

    assert _sanitize_path_segment("cat5/reg lam0.01") == "cat5_reg_lam0.01"
    assert _sanitize_path_segment("  cat1_baseline  ") == "cat1_baseline"
