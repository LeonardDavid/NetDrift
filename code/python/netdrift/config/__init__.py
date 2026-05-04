"""YAML experiment configs.

A single :class:`ExperimentConfig` dataclass tree captures everything needed
to run an experiment: model, data, quantization, storage layout, fault model,
training/test, metrics. The YAML on disk maps 1:1 to this tree, so config
diffs are readable and overrides have predictable shapes.
"""

from netdrift.config.loader import load, parse_overrides
from netdrift.config.schema import (
    DataCfg,
    ExperimentConfig,
    FaultCfg,
    MetricsCfg,
    ModelCfg,
    ProtectionCfg,
    QuantCfg,
    SinkCfg,
    StorageCfg,
    TrainCfg,
)

__all__ = [
    "load",
    "parse_overrides",
    "ExperimentConfig",
    "ModelCfg",
    "DataCfg",
    "QuantCfg",
    "StorageCfg",
    "FaultCfg",
    "TrainCfg",
    "MetricsCfg",
    "ProtectionCfg",
    "SinkCfg",
]
