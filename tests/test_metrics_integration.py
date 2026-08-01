"""End-to-end: --metrics levels produce the expected artifact file set."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.cuda
def test_metrics_all_writes_artifacts(tmp_path):
    from netdrift.runner.run import main

    # Use a high rt_error so faults (and thus wrong_bits_read) actually occur on
    # a single-loop run — at 1e-05 a tiny model may see zero faults and the
    # wrong-bits assertion below would be vacuous.
    cfg_path = "configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml"
    rc = main([
        "--config", cfg_path,
        "--metrics", "all",
        "--wandb-category", "cat1_baseline",
        "--override", "fault.rt_error=[0.1]",
        "--override", "training.loops=1",
        "--override", f"experiment.output_dir={tmp_path}",
    ])
    assert rc == 0

    # run.py writes metrics artifacts under "metrics_artifacts" (renamed from
    # "metrics" so mutagen sync can target run artifacts separately from the
    # code/python/netdrift/metrics/ source package). This is a fresh tmp_path
    # run against the current runner, so only the new name is expected here.
    metrics_dirs = list(Path(tmp_path).rglob("metrics_artifacts"))
    assert metrics_dirs, "no metrics_artifacts/ dir produced"
    md = metrics_dirs[0]
    static = list(md.glob("*__static.json"))
    rt = list(md.glob("*__rt0.1.json"))
    assert static, "missing static.json"
    assert rt, "missing per-rt_error json"

    data = json.loads(rt[0].read_text())
    assert data["schema_version"] == 1
    assert "outcome" in data and "fault_incidence" in data
    assert data["meta"]["static_ref"] == static[0].name
    # weights block present and self-consistent (total = protected + unprotected).
    w = data["meta"]["weights"]
    assert w["total"] == w["protected"] + w["unprotected"]
    assert w["protected"] > 0 and w["unprotected"] > 0  # vgg7 protects conv1/fc2
    # BER denominator recorded in fault_incidence must match meta's unprotected count.
    assert (data["fault_incidence"]["last_loop"]["ber_denominator_unprotected_weights"]
            == w["unprotected"])
    # Guard against the track-flag wiring silently dropping wrong_bits_read
    # (Task 10): the key must be present and, at rt_error=0.1, non-zero.
    fi_last = data["fault_incidence"]["last_loop"]
    assert "wrong_bits_read" in fi_last, "wrong_bits_read missing — track flag not wired"
    assert fi_last["wrong_bits_read"] > 0

    # Physical-plausibility guards on the STOCK metrics (regression against the
    # batch-sum artifact that produced affected_units > #racetracks and BER > 1):
    #   - affected_units (standing nonzero racetracks) must not exceed the total
    #     number of racetracks across all layers.
    #   - BER (final-loop bitflips / unprotected weights) must lie in [0, 1].
    total_racetracks = sum(l["n_racetracks"][0] * l["n_racetracks"][1]
                           for l in data["meta"]["layers"])
    assert 0 <= fi_last["affected_units"] <= total_racetracks, (
        f"affected_units {fi_last['affected_units']} exceeds total racetracks "
        f"{total_racetracks} — batch-sum artifact regressed"
    )
    assert 0.0 <= fi_last["ber"] <= 1.0, f"BER {fi_last['ber']} outside [0,1]"

    # sum_over_loops must exist ONLY for flow metrics (misalign_faults), not stocks.
    sol = data["fault_incidence"]["sum_over_loops"]
    assert "misalign_faults" in sol
    assert "bitflips" not in sol and "affected_units" not in sol

    # The .npz sidecar must exist at level 'all' and carry the wrong-read mask.
    npz = list(md.glob("*__rt0.1.npz"))
    assert npz, "missing .npz sidecar at --metrics all"


@pytest.mark.cuda
def test_metrics_none_writes_no_artifacts(tmp_path):
    from netdrift.runner.run import main

    cfg_path = "configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml"
    rc = main([
        "--config", cfg_path,
        "--metrics", "none",
        "--override", "fault.rt_error=[1e-05]",
        "--override", "training.loops=1",
        "--override", f"experiment.output_dir={tmp_path}",
    ])
    assert rc == 0
    assert not list(Path(tmp_path).rglob("metrics_artifacts")), (
        "metrics_artifacts dir created at level none"
    )
