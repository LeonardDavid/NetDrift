"""JSON/.npz artifact round-trip and schema."""
from __future__ import annotations

import json

import numpy as np


def test_static_artifact_roundtrip(tmp_path):
    from netdrift.metrics.artifacts import SCHEMA_VERSION, write_static_artifact

    meta = {"model": "vgg3_fmnist", "category": "cat2_vanilla_endlen"}
    snapshots = [{"label": "before_encoder", "total": {}, "per_layer": {}}]
    deltas = {"before_encoder->after_encoder": {"total": {"bitflips": 7}}}
    path = write_static_artifact(
        tmp_path, model="vgg3_fmnist", category="cat2_vanilla_endlen",
        meta=meta, snapshots=snapshots, deltas=deltas, recal=None, npz_arrays=None,
    )
    assert path.name == "vgg3_fmnist__cat2_vanilla_endlen__static.json"
    data = json.loads(path.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["deltas"]["before_encoder->after_encoder"]["total"]["bitflips"] == 7
    # No .npz written when npz_arrays is None.
    assert not (tmp_path / "vgg3_fmnist__cat2_vanilla_endlen__static.npz").exists()


def test_rt_error_artifact_writes_json_and_npz(tmp_path):
    from netdrift.metrics.artifacts import write_rt_error_artifact

    meta = {"model": "vgg3_fmnist", "category": "cat1_baseline"}
    json_path, npz_path = write_rt_error_artifact(
        tmp_path, model="vgg3_fmnist", category="cat1_baseline", rt_error=1e-05,
        meta=meta,
        outcome={"per_loop_accuracy": [90.0, 89.0]},
        fault_incidence={"total": {"bitflips": 3}},
        npz_arrays={"online__final__lin__index_offset": np.zeros((4, 2), dtype=np.int32)},
    )
    assert json_path.name == "vgg3_fmnist__cat1_baseline__rt1e-05.json"
    assert npz_path is not None and npz_path.exists()
    loaded = np.load(npz_path)
    assert "online__final__lin__index_offset" in loaded
    data = json.loads(json_path.read_text())
    assert data["meta"]["npz_ref"] == npz_path.name


def test_distribution_stats():
    from netdrift.metrics.artifacts import distribution_stats

    d = distribution_stats([90.0, 80.0, 100.0, 70.0])
    assert d["min"] == 70.0
    assert d["max"] == 100.0
    assert abs(d["mean"] - 85.0) < 1e-9
    assert d["p50"] == 85.0  # median of [70,80,90,100]
    # empty list is safe
    assert distribution_stats([]) == {}
