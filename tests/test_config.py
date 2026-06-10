"""YAML config loader: schema validation, includes, and CLI overrides.

CPU-safe — no GPU work here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from netdrift.config import (
    ExperimentConfig,
    load,
    parse_overrides,
)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def test_minimal_config_loads_with_defaults(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, "exp.yaml", "experiment:\n  name: hello\n")
    cfg = load(cfg_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.experiment.name == "hello"
    # defaults from the dataclass tree
    assert cfg.quant.scheme == "binary"
    assert cfg.fault.model == "rtm_misalignment"
    assert cfg.training.mode == "test"


def test_full_config_round_trip(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, "full.yaml", """
experiment:
  name: rtm_sweep
  seed: 707

model:
  name: resnet18_cifar10
  checkpoint: /tmp/ckpt.pt
  checkpoint_mode: fp32_warmstart

data:
  name: cifar10
  batch_size: 128

quant:
  scheme: binary
  scale_init: max_abs

storage:
  layout: row
  rt_size: 64
  kernel_mapping: clw

fault:
  model: rtm_misalignment
  rt_error: [0.001, 0.01, 0.05]
  mitigations:
    - bin_revert_mid
  protection:
    policy: custom
    layers: [1, 2]

training:
  mode: train
  fault_aware: ste_inject
  epochs: 5
  lr: 0.001

metrics:
  online: [bitflips, misalign_faults]
  sinks:
    - type: jsonl
    - type: stdout
""")
    cfg = load(cfg_path)
    assert cfg.experiment.seed == 707
    assert cfg.model.checkpoint_mode == "fp32_warmstart"
    assert cfg.fault.rt_error == [0.001, 0.01, 0.05]
    assert cfg.fault.mitigations == ["bin_revert_mid"]
    assert cfg.fault.protection.policy == "custom"
    assert cfg.fault.protection.layers == [1, 2]
    assert cfg.training.fault_aware == "ste_inject"
    assert cfg.metrics.online == ["bitflips", "misalign_faults"]
    assert [s.type for s in cfg.metrics.sinks] == ["jsonl", "stdout"]


def test_defaults_include_resolves_relative(tmp_path: Path) -> None:
    defaults_dir = tmp_path / "_defaults"
    defaults_dir.mkdir()
    _write(
        defaults_dir, "data_cifar10.yaml",
        "data:\n  name: cifar10\n  batch_size: 64\n",
    )
    cfg_path = _write(tmp_path, "exp.yaml", """
defaults:
  - _defaults/data_cifar10.yaml

experiment:
  name: from_defaults
""")
    cfg = load(cfg_path)
    assert cfg.experiment.name == "from_defaults"
    assert cfg.data.name == "cifar10"
    assert cfg.data.batch_size == 64


def test_cli_override_applies_after_yaml(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path, "exp.yaml",
        "fault:\n  rt_error: 0.0\n  protection:\n    policy: all\n",
    )
    cfg = load(
        cfg_path,
        overrides=parse_overrides([
            "fault.rt_error=0.05",
            "fault.protection.policy=custom",
            "fault.protection.layers=[1,2]",
        ]),
    )
    assert cfg.fault.rt_error == 0.05
    assert cfg.fault.protection.policy == "custom"
    assert cfg.fault.protection.layers == [1, 2]


def test_override_string_value_passes_through(tmp_path: Path) -> None:
    """Strings that don't parse as JSON should reach the schema verbatim."""
    cfg_path = _write(tmp_path, "exp.yaml", "experiment:\n  name: orig\n")
    cfg = load(cfg_path, overrides=["experiment.name=experiment-foo"])
    assert cfg.experiment.name == "experiment-foo"


def test_malformed_override_rejected() -> None:
    with pytest.raises(ValueError, match="key=value"):
        parse_overrides(["fault.rt_error"])


def test_weight_encoder_fields_round_trip(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, "exp.yaml", """
experiment:
  name: enc_test

fault:
  weight_encoder: endlen
  weight_encoder_mode: per_forward
  encoded_checkpoint_save: /tmp/foo_endlen.pt
""")
    cfg = load(cfg_path)
    assert cfg.fault.weight_encoder == "endlen"
    assert cfg.fault.weight_encoder_mode == "per_forward"
    assert cfg.fault.encoded_checkpoint_save == "/tmp/foo_endlen.pt"


def test_weight_encoder_defaults(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, "exp.yaml", "experiment:\n  name: defaults\n")
    cfg = load(cfg_path)
    assert cfg.fault.weight_encoder is None
    assert cfg.fault.weight_encoder_mode == "once"
    assert cfg.fault.encoded_checkpoint_save is None


def test_unknown_dataclass_field_silently_ignored(tmp_path: Path) -> None:
    """Extra YAML keys are dropped (forward-compatible) rather than raising."""
    cfg_path = _write(
        tmp_path, "exp.yaml",
        "experiment:\n  name: ok\n  unknown_field: 42\n",
    )
    cfg = load(cfg_path)
    assert cfg.experiment.name == "ok"
    # No attribute leak.
    assert not hasattr(cfg.experiment, "unknown_field")


def test_budget_params_parse(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path, "exp.yaml",
        "fault:\n"
        "  global_bitflip_budget: 0.3\n"
        "  local_bitflip_budget: 0.5\n"
        "  local_budget_scope: racetrack\n"
        "  budget_selection: magnitude_aware\n",
    )
    cfg = load(cfg_path)
    assert cfg.fault.global_bitflip_budget == 0.3
    assert cfg.fault.local_bitflip_budget == 0.5
    assert cfg.fault.local_budget_scope == "racetrack"
    assert cfg.fault.budget_selection == "magnitude_aware"


def test_budget_defaults_are_unbounded(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, "exp.yaml", "experiment:\n  name: ok\n")
    cfg = load(cfg_path)
    assert cfg.fault.global_bitflip_budget == 1.0
    assert cfg.fault.local_bitflip_budget == 1.0
    assert cfg.fault.local_budget_scope == "layer"
    assert cfg.fault.budget_selection == "greedy"


def test_budget_out_of_range_rejected(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, "exp.yaml", "fault:\n  global_bitflip_budget: 1.5\n")
    with pytest.raises(ValueError, match="budget"):
        load(cfg_path)


def test_bad_scope_rejected(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, "exp.yaml", "fault:\n  local_budget_scope: nonsense\n")
    with pytest.raises(ValueError, match="local_budget_scope"):
        load(cfg_path)


def test_bad_selection_rejected(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, "exp.yaml", "fault:\n  budget_selection: nonsense\n")
    with pytest.raises(ValueError, match="budget_selection"):
        load(cfg_path)


def test_recalibrate_and_reg_defaults():
    from netdrift.config.schema import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.training.recalibrate.enabled is False
    assert cfg.training.recalibrate.bn_stats is True
    assert cfg.training.recalibrate.tune_affine is True
    assert cfg.training.recalibrate.on == "endlen"
    assert cfg.training.reg.lambda_ == 0.0
    assert cfg.training.reg.beta == 4.0
    assert cfg.training.reg.inject_faults is False
    assert cfg.training.fault_state_mode == "fresh"


def test_recalibrate_parses_from_yaml_dict():
    from netdrift.config.loader import _from_dict
    from netdrift.config.schema import ExperimentConfig

    raw = {
        "training": {
            "mode": "test",
            "fault_aware": "regularization",
            "fault_state_mode": "accumulate",
            "recalibrate": {"enabled": True, "epochs": 3, "on": "always"},
            "reg": {"lambda": 0.02, "beta": 8.0, "inject_faults": True},
        }
    }
    cfg = _from_dict(ExperimentConfig, raw)
    assert cfg.training.recalibrate.enabled is True
    assert cfg.training.recalibrate.epochs == 3
    assert cfg.training.recalibrate.on == "always"
    assert cfg.training.reg.lambda_ == 0.02  # YAML key 'lambda' maps to lambda_
    assert cfg.training.reg.inject_faults is True
    assert cfg.training.fault_state_mode == "accumulate"


def test_invalid_fault_state_mode_raises():
    import pytest
    from netdrift.config.schema import TrainCfg

    with pytest.raises(ValueError):
        TrainCfg(fault_state_mode="sometimes")


def test_invalid_recalibrate_on_raises():
    import pytest
    from netdrift.config.schema import RecalibrateCfg

    with pytest.raises(ValueError):
        RecalibrateCfg(on="whenever")


def test_criterion_defaults_are_hinge_128():
    from netdrift.config.schema import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.training.criterion == "hinge"
    assert cfg.training.hinge_b == 128.0
    assert cfg.training.fault_aware_criterion == "hinge"
    assert cfg.training.fault_aware_hinge_b == 128.0


def test_criterion_parses_from_yaml_dict():
    from netdrift.config.loader import _from_dict
    from netdrift.config.schema import ExperimentConfig

    raw = {
        "training": {
            "criterion": "cross_entropy",
            "hinge_b": 64.0,
            "fault_aware_criterion": "cross_entropy",
            "fault_aware_hinge_b": 32.0,
        }
    }
    cfg = _from_dict(ExperimentConfig, raw)
    assert cfg.training.criterion == "cross_entropy"
    assert cfg.training.hinge_b == 64.0
    assert cfg.training.fault_aware_criterion == "cross_entropy"
    assert cfg.training.fault_aware_hinge_b == 32.0


def test_invalid_criterion_raises():
    import pytest
    from netdrift.config.schema import TrainCfg

    with pytest.raises(ValueError):
        TrainCfg(criterion="focal")
    with pytest.raises(ValueError):
        TrainCfg(fault_aware_criterion="focal")


def test_invalid_hinge_b_raises():
    import pytest
    from netdrift.config.schema import TrainCfg

    with pytest.raises(ValueError):
        TrainCfg(hinge_b=0.0)
    with pytest.raises(ValueError):
        TrainCfg(fault_aware_hinge_b=-1.0)
