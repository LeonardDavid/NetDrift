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
    seed: int = 707


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
        scheme:             ``"binary"`` (Phase 1), ``"none"`` (FP), or one of
                            ``"ternary"`` / ``"int_uniform"`` / ``"mixed_precision"`` (Phase 3+).
                            Selects the **weight** quantization scheme.
        bits:               Used by ``int_uniform`` (weights). Ignored otherwise.
        bits_per_channel:   Used by ``mixed_precision`` (weights). Ignored otherwise.
        scale_init:         Per-channel scale initialization policy (FP32 warm-start).
        activation_scheme:  ``"none"`` (default, identity) or ``"int_uniform"``
                            (symmetric uniform on ``[-1, 1]``). Drives the
                            ``QuantizedActivation`` modules in the topology.
        activation_bits:    Bits per activation level when
                            ``activation_scheme="int_uniform"``. Common: 1, 2, 4, 8.
    """

    scheme: str = "binary"
    bits: int = 1
    bits_per_channel: Optional[list[int]] = None
    scale_init: str = "max_abs"
    activation_scheme: str = "none"
    activation_bits: int = 1


@dataclass
class StorageCfg:
    """Racetrack storage layout (Phase 1: row/col/mix only).

    Attributes:
        layout:         Phase 1 supports ``row`` / ``col`` / ``mix``. Phase 3
                        adds ``interleaved``, ``gray``, ``importance_sorted``,
                        ``ecc``, ``replicated``. ``block`` is the BLOCK
                        weight-storage mapping, segmented per ``base_layout``.
        rt_size:        Bits per racetrack.
        kernel_mapping: Conv kernel layout: ``row`` / ``col`` / ``clw`` / ``acw``.
                        Ignored for linear layers.
        base_layout:    For ``layout=="block"``: the ROW/COL base segmentation
                        used underneath the BLOCK mapping. Ignored otherwise.
    """

    layout: str = "row"
    rt_size: int = 64
    kernel_mapping: str = "row"
    base_layout: str = "row"

    def __post_init__(self) -> None:
        if self.base_layout.lower() not in ("row", "col"):
            raise ValueError(
                f"storage.base_layout must be row|col; got {self.base_layout!r}"
            )


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

    def __post_init__(self) -> None:
        if self.policy not in ("all", "custom", "indiv"):
            raise ValueError(
                f"fault.protection.policy must be all|custom|indiv; got {self.policy!r}"
            )
        if self.policy == "custom":
            if not self.layers:
                raise ValueError(
                    "fault.protection.policy='custom' requires a non-empty "
                    "fault.protection.layers list (1-based unprotected layer ids)"
                )
            if any(int(x) < 1 for x in self.layers):
                raise ValueError(
                    "fault.protection.layers must be 1-based (all ids >= 1); "
                    f"got {self.layers}"
                )
        if self.policy == "indiv":
            if self.indiv_layer is None:
                raise ValueError(
                    "fault.protection.policy='indiv' requires "
                    "fault.protection.indiv_layer=N (1-based)"
                )
            if int(self.indiv_layer) < 1:
                raise ValueError(
                    f"fault.protection.indiv_layer must be >= 1; got {self.indiv_layer}"
                )


@dataclass
class FaultCfg:
    """Fault model configuration.

    Attributes:
        model:                 Fault-model registry key (``"rtm_misalignment"``,
                               ``"stuck_at"``, ``"bitflip"``, ``"skyrmion_collapse"``).
                               Phase 1 only ships ``rtm_misalignment``.
        rt_error:              Per-read fault probability or list of probabilities.
                               A list triggers a sweep at test time.
        global_bitflip_budget: Max fraction (0..1) of ALL model weights the
                               endlen encoder may flip. ``1.0`` = unbounded,
                               ``0.0`` = flip nothing. mode=once only.
        local_bitflip_budget:  Max fraction (0..1) within each local unit (see
                               ``local_budget_scope``). To run global-only leave
                               this at ``1.0``; for local-only set
                               ``global_bitflip_budget`` to ``1.0``.
        local_budget_scope:    ``layer`` | ``racetrack`` | ``channel`` — the unit
                               ``local_bitflip_budget`` is measured against.
        budget_selection:      ``greedy`` | ``value_per_flip`` |
                               ``magnitude_aware`` — which merges survive the cap.
        mitigations:           Ordered list of mitigation step names. Default
                               is ``[]`` — a single step max is the
                               recommended convention.
        protection:            Layer-protection policy.
        edge_mode:             ``saturate`` (fixed access port, no random reads)
                               or ``random`` (legacy out-of-bounds ±1).
        ap_position:           Fixed access-port index for ``edge_mode=saturate``;
                               ``None`` -> ``rt_size//2 - 1``.
    """

    model: str = "rtm_misalignment"
    rt_error: Any = 0.0  # float | list[float]
    global_bitflip_budget: float = 1.0
    local_bitflip_budget: float = 1.0
    local_budget_scope: str = "layer"      # layer | racetrack | channel
    budget_selection: str = "greedy"       # greedy | value_per_flip | magnitude_aware
    mitigations: list[str] = field(default_factory=list)
    protection: ProtectionCfg = field(default_factory=ProtectionCfg)
    weight_encoder: Optional[str] = None
    """Registry name of a write-time weight encoder (``"endlen"``) or ``None``."""
    weight_encoder_mode: str = "once"
    """``"once"`` (apply before sweep, save encoded model) or ``"per_forward"``
    (re-apply inside every fault injection). Ignored when ``weight_encoder``
    is ``None``."""
    encoded_checkpoint_save: Optional[str] = None
    """Override path for the post-encoder checkpoint (mode=once). Default:
    ``<run_dir>/model_endlen.pt``."""
    edge_mode: str = "saturate"
    """Racetrack edge model. ``"saturate"`` (default): the access port is fixed
    at ``ap_position`` and every read saturates to the nearest real cell — no
    random values reach the network, and a partially-filled racetrack replicates
    its edge value into the unfilled tail. ``"random"``: legacy behaviour where a
    read shifted off the racetrack returns a random ±1 (kept for A/B comparison).
    Note ``block`` layout under ``"saturate"`` is fault-immune (same-sign blocks)
    — use ``"random"`` to obtain a non-trivial BLOCK robustness curve."""
    ap_position: Optional[int] = None
    """Fixed access-port index in ``[0, rt_size-1]`` for ``edge_mode="saturate"``.
    ``None`` (default) resolves to ``rt_size//2 - 1`` (the first middle position).
    Must be left unset for ``storage.layout=="block"``."""

    def __post_init__(self) -> None:
        for n in ("global_bitflip_budget", "local_bitflip_budget"):
            v = float(getattr(self, n))
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{n} must be a fraction in [0, 1]; got {v}")
        if self.edge_mode not in ("saturate", "random"):
            raise ValueError(
                f"fault.edge_mode must be saturate|random; got {self.edge_mode!r}"
            )
        if self.ap_position is not None and int(self.ap_position) < 0:
            raise ValueError(
                f"fault.ap_position must be >= 0; got {self.ap_position}"
            )
        if self.local_budget_scope not in ("layer", "racetrack", "channel"):
            raise ValueError(
                "local_budget_scope must be layer|racetrack|channel; "
                f"got {self.local_budget_scope!r}"
            )
        if self.budget_selection not in (
            "greedy", "value_per_flip", "magnitude_aware"
        ):
            raise ValueError(
                "budget_selection must be greedy|value_per_flip|magnitude_aware; "
                f"got {self.budget_selection!r}"
            )


@dataclass
class RecalibrateCfg:
    """Pattern-preserving recalibration (BN running stats + output Scale).

    Runs in ``mode="test"`` after the endlen encoder and before the rt_error
    sweep. Binary weight signs are NEVER changed (endlen pattern preserved);
    only BatchNorm stats/affine params and the output Scale are updated. No
    fault injection during recalibration — this recovers the clean-accuracy
    gap endlen's bit-flips introduce.

    Attributes:
        enabled:     Master switch. Default off.
        bn_stats:    Sub-step A — re-estimate BN running mean/var via forward
                     passes in train() mode (no backward).
        tune_affine: Sub-step B — short backprop fine-tune of BN gamma/beta +
                     the output Scale only (latent weights frozen).
        epochs:      Sub-step B epochs. 0 ⇒ run sub-step A only.
        lr:          Adam learning rate for sub-step B.
        num_batches: Cap on batches for sub-step A (null ⇒ full epoch).
        on:          ``endlen`` (only recalibrate when an encoder ran) or
                     ``always`` (also recalibrate a plain model).
    """

    enabled: bool = False
    bn_stats: bool = True
    tune_affine: bool = True
    epochs: int = 2
    lr: float = 0.001
    num_batches: Optional[int] = None
    on: str = "endlen"

    def __post_init__(self) -> None:
        if self.on not in ("endlen", "always"):
            raise ValueError(f"recalibrate.on must be endlen|always; got {self.on!r}")
        if self.epochs < 0:
            raise ValueError(f"recalibrate.epochs must be >= 0; got {self.epochs}")


@dataclass
class RegCfg:
    """Run-length regularizer (fault-aware training toward long same-sign runs).

    Active when ``training.fault_aware == "regularization"``. Adds
    ``lambda_ * run_length_penalty`` to the task loss. The penalty is the
    adjacent sign-agreement surrogate over the racetrack-aligned weight view of
    unprotected layers.

    Attributes:
        lambda_:       Regularizer weight (YAML key ``lambda``). 0 ⇒ plain
                       training (no run-length term).
        beta:          tanh sharpness for the sign surrogate. Larger ⇒ sharper.
        inject_faults: If True, also inject RTM faults in the forward pass via
                       the STE residual (task loss on faulted weights). Fault
                       persistence chosen by ``training.fault_state_mode``.
    """

    lambda_: float = 0.0
    beta: float = 4.0
    inject_faults: bool = False

    def __post_init__(self) -> None:
        if self.lambda_ < 0:
            raise ValueError(f"reg.lambda must be >= 0; got {self.lambda_}")
        if self.beta <= 0:
            raise ValueError(f"reg.beta must be > 0; got {self.beta}")


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
        fault_state_mode: Used when faults are injected during training.
                     ``fresh`` re-samples a NEW fault realization every batch
                     (resets per-layer fault_state) — augments over the fault
                     distribution; generalizes across realizations; RECOMMENDED.
                     ``accumulate`` keeps faults across batches (stuck-stays-
                     stuck), matching the eval-sweep semantics — faithful to one
                     deployment scenario but risks overfitting to a single
                     realization.
        recalibrate: Pattern-preserving BN+Scale recalibration config.
        reg:         Run-length regularizer config.
        criterion:   Loss for BNN baseline training AND recalibration sub-step B
                     (cat4b/cat6 tune_affine): ``hinge`` (modified hinge loss,
                     MHL per Yayla et al.) or ``cross_entropy``. Default
                     ``hinge``. Note: full-precision (scheme=None) training
                     always uses CrossEntropyLoss regardless of this field.
        hinge_b:     ``b`` parameter of the hinge loss for the baseline/recal
                     criterion. Ignored when criterion=cross_entropy.
        fault_aware_criterion: Loss for fault-aware training (cat5 regularization
                     + cat8 ste_inject): ``hinge`` | ``cross_entropy``. Default
                     ``hinge``. Lets the RTM-optimizing training use a different
                     criterion than the baseline.
        fault_aware_hinge_b: ``b`` parameter for the fault-aware hinge loss.
                     Ignored when fault_aware_criterion=cross_entropy.
    """

    mode: str = "test"
    fault_aware: str = "none"
    epochs: int = 10
    loops: int = 1
    lr: float = 1.0
    gamma: float = 0.1
    step_size: int = 5
    save_dir: Optional[str] = None
    fault_state_mode: str = "fresh"  # fresh | accumulate (when faults injected in training)
    recalibrate: RecalibrateCfg = field(default_factory=RecalibrateCfg)
    reg: RegCfg = field(default_factory=RegCfg)
    # Loss criterion (BNN paths only; FP32 always uses CrossEntropyLoss).
    criterion: str = "hinge"               # hinge | cross_entropy (baseline + recal)
    hinge_b: float = 128.0
    fault_aware_criterion: str = "hinge"   # hinge | cross_entropy (cat5 + cat8)
    fault_aware_hinge_b: float = 128.0

    def __post_init__(self) -> None:
        if self.fault_aware not in ("none", "ste_inject", "kd", "regularization"):
            raise ValueError(
                "training.fault_aware must be none|ste_inject|kd|regularization; "
                f"got {self.fault_aware!r}"
            )
        if self.fault_state_mode not in ("fresh", "accumulate"):
            raise ValueError(
                "training.fault_state_mode must be fresh|accumulate; "
                f"got {self.fault_state_mode!r}"
            )
        for fld in ("criterion", "fault_aware_criterion"):
            v = getattr(self, fld)
            if v not in ("hinge", "cross_entropy"):
                raise ValueError(
                    f"training.{fld} must be hinge|cross_entropy; got {v!r}"
                )
        for fld in ("hinge_b", "fault_aware_hinge_b"):
            if float(getattr(self, fld)) <= 0:
                raise ValueError(f"training.{fld} must be > 0; got {getattr(self, fld)}")


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
