"""Typed config dataclasses backing the YAML schema.

Each top-level section corresponds to a YAML key. Defaults match the most
common BNN/RTM use-case so that minimal configs remain valid; everything
listed here is overridable from YAML or via CLI ``--override key=value`` flags.

The runner consumes :class:`ExperimentConfig` directly — no further
normalization is needed once :func:`netdrift.config.load` has produced one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExperimentMeta:
    """Top-level experiment identification."""

    name: str = "experiment"
    output_dir: str = "runs/"
    seed: int = 1


@dataclass
class ModelCfg:
    """Model topology + checkpoint loading.

    Attributes:
        name:            Registry key, e.g. ``"vgg7_cifar10"`` or ``"resnet18_cifar100"``.
        checkpoint:      Path to a checkpoint (``.pth``/``.pt``) or ``None``.
        checkpoint_mode: ``"strict"`` (BNN→BNN), ``"fp32_warmstart"``
                         (FP32→BNN), or ``"scheme_transfer"`` (BNN→QNN with
                         different scheme).
        kernel_size:     Conv kernel side length used by VGG variants. Ignored
                         by torchvision-based models (which fix their own).
        skip_first_quant: If ``True``, leave the first conv/linear in FP32
                         (a common BNN recipe).
        skip_last_quant:  If ``True``, leave the last conv/linear in FP32.
    """

    name: str = "vgg7_cifar10"
    checkpoint: Optional[str] = None
    checkpoint_mode: str = "strict"
    kernel_size: int = 3
    skip_first_quant: bool = False
    skip_last_quant: bool = False


@dataclass
class DataCfg:
    """Dataset selection + DataLoader hyperparameters."""

    name: str = "cifar10"
    batch_size: int = 256
    test_batch_size: int = 256
    num_workers: int = 1
    data_dir: str = "data"


@dataclass
class QuantCfg:
    """Quantization scheme and (Phase 3+) per-channel parameters.

    Attributes:
        scheme:           ``"binary"`` (Phase 1) or one of ``"ternary"``,
                          ``"int_uniform"``, ``"mixed_precision"`` (Phase 3+).
        bits:             Used by ``int_uniform``. Ignored otherwise.
        bits_per_channel: Used by ``mixed_precision``. Ignored otherwise.
        scale_init:       Per-channel scale initialization policy (FP32 warm-start).
    """

    scheme: str = "binary"
    bits: int = 1
    bits_per_channel: Optional[list[int]] = None
    scale_init: str = "max_abs"


@dataclass
class StorageCfg:
    """Racetrack storage layout (Phase 1: row/col/mix only).

    Attributes:
        layout:         Phase 1 supports ``row`` / ``col`` / ``mix``. Phase 3
                        adds ``interleaved``, ``gray``, ``importance_sorted``,
                        ``ecc``, ``replicated``.
        rt_size:        Bits per racetrack.
        kernel_mapping: Conv kernel layout: ``row`` / ``col`` / ``clw`` / ``acw``.
                        Ignored for linear layers.
    """

    layout: str = "row"
    rt_size: int = 64
    kernel_mapping: str = "row"


@dataclass
class ProtectionCfg:
    """Layer-protection policy.

    Attributes:
        policy:      ``"all"``, ``"custom"``, or ``"indiv"``.
        layers:      For ``custom``: list of 1-based layer indices to leave unprotected.
        indiv_layer: For ``indiv``: the single 1-based index to leave unprotected.
    """

    policy: str = "all"
    layers: Optional[list[int]] = None
    indiv_layer: Optional[int] = None


@dataclass
class FaultCfg:
    """Fault model configuration.

    Attributes:
        model:                 Fault-model registry key (``"rtm_misalignment"``,
                               ``"stuck_at"``, ``"bitflip"``, ``"skyrmion_collapse"``).
                               Phase 1 only ships ``rtm_misalignment``.
        rt_error:              Per-read fault probability or list of probabilities.
                               A list triggers a sweep at test time.
        global_bitflip_budget: Reserved (used by Phase 2+ budget enforcement).
        local_bitflip_budget:  Reserved.
        mitigations:           Ordered list of mitigation step names. Default
                               is ``[]`` — a single step max is the
                               recommended convention.
        protection:            Layer-protection policy.
    """

    model: str = "rtm_misalignment"
    rt_error: Any = 0.0  # float | list[float]
    global_bitflip_budget: float = 0.0
    local_bitflip_budget: float = 0.0
    mitigations: list[str] = field(default_factory=list)
    protection: ProtectionCfg = field(default_factory=ProtectionCfg)


@dataclass
class TrainCfg:
    """Training/test loop configuration.

    Attributes:
        mode:        ``"train"`` or ``"test"``.
        fault_aware: ``"none"`` (current behaviour) or — Phase 2 — ``"ste_inject"`` /
                     ``"kd"`` / ``"regularization"``.
        epochs:      Number of training epochs.
        loops:       Number of inference iterations under ``mode="test"``.
        lr:          Initial learning rate.
        gamma:       StepLR gamma.
        step_size:   StepLR step size in epochs.
    """

    mode: str = "test"
    fault_aware: str = "none"
    epochs: int = 10
    loops: int = 1
    lr: float = 1.0
    gamma: float = 0.1
    step_size: int = 5


@dataclass
class SinkCfg:
    """A single metrics sink. ``type`` selects the implementation."""

    type: str = "stdout"
    project: Optional[str] = None  # used by wandb sink


@dataclass
class MetricsCfg:
    """Metrics tracked online + offline + sink configuration."""

    online: list[str] = field(default_factory=lambda: ["bitflips", "misalign_faults", "affected_units"])
    offline: list[str] = field(default_factory=list)
    sinks: list[SinkCfg] = field(default_factory=lambda: [SinkCfg(type="jsonl"), SinkCfg(type="stdout")])


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    quant: QuantCfg = field(default_factory=QuantCfg)
    storage: StorageCfg = field(default_factory=StorageCfg)
    fault: FaultCfg = field(default_factory=FaultCfg)
    training: TrainCfg = field(default_factory=TrainCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)

    # The optional gpu-num runtime knob; null means torch picks.
    gpu_num: int = 0
