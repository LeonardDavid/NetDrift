"""Checkpoint loading with strict / FP32-warmstart / scheme-transfer modes.

Three flows are supported (see plan section "Three checkpoint-loading flows"):

* ``strict``           — exact match. Used for existing NetDrift / HuggingFace
                         BNN checkpoints.
* ``fp32_warmstart``   — load an FP32 checkpoint into a quantized model.
                         Drops unexpected ``*.bias`` keys, applies per-arch
                         key remaps (e.g. torchvision ``fc`` → NetDrift
                         ``linear`` for ResNet), initializes any
                         ``scale_per_channel`` parameters from FP32 weight
                         statistics. After warm-start training, the
                         resulting checkpoint round-trips through ``strict``.
* ``scheme_transfer``  — load a BNN checkpoint into a different quantization
                         scheme of the same architecture (rare; future use).

Each call returns an :class:`AdapterReport` listing every key dropped,
renamed, or freshly initialized — and the user-facing log line summarises it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import torch
import torch.nn as nn

from netdrift.quant.base import QuantScheme, ScaleInit
from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear


CheckpointMode = Literal["strict", "fp32_warmstart", "scheme_transfer"]


@dataclass
class AdapterReport:
    """Summary of what the adapter did. Serialized into run logs for traceability."""

    mode: CheckpointMode
    dropped_keys: list[str] = field(default_factory=list)
    renamed_keys: list[tuple[str, str]] = field(default_factory=list)
    initialized_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"checkpoint mode={self.mode}"]
        if self.dropped_keys:
            lines.append(f"  dropped ({len(self.dropped_keys)}): {self.dropped_keys[:5]}{'...' if len(self.dropped_keys) > 5 else ''}")
        if self.renamed_keys:
            lines.append(f"  renamed ({len(self.renamed_keys)}): {self.renamed_keys[:5]}{'...' if len(self.renamed_keys) > 5 else ''}")
        if self.initialized_keys:
            lines.append(f"  initialized ({len(self.initialized_keys)}): {self.initialized_keys[:5]}{'...' if len(self.initialized_keys) > 5 else ''}")
        if self.missing_keys:
            lines.append(f"  missing ({len(self.missing_keys)}): {self.missing_keys[:5]}{'...' if len(self.missing_keys) > 5 else ''}")
        if self.unexpected_keys:
            lines.append(f"  unexpected ({len(self.unexpected_keys)}): {self.unexpected_keys[:5]}{'...' if len(self.unexpected_keys) > 5 else ''}")
        return "\n".join(lines)


# Per-architecture rename rules. The detector looks at the *target* model: if
# the target has a child named ``linear`` (legacy NetDrift custom ResNet), we
# remap source keys with ``fc.*`` (torchvision-style) onto ``linear.*``. The
# common case (torchvision FP32 → torchvision quantized, both ``fc.*``) needs
# no renames.
def _detect_arch_rules(model: nn.Module) -> list[tuple[str, str]]:
    """Sniff which rename rules apply based on the target module hierarchy."""
    target_children = {name for name, _ in model.named_children()}
    rules: list[tuple[str, str]] = []
    if "linear" in target_children and "fc" not in target_children:
        # Source uses torchvision-style ``fc.*``; target uses ``linear.*``.
        rules.append(("fc.", "linear."))
    return rules


def _apply_renames(
    state_dict: dict[str, torch.Tensor],
    rules: list[tuple[str, str]],
) -> tuple[dict[str, torch.Tensor], list[tuple[str, str]]]:
    if not rules:
        return state_dict, []
    new_sd: dict[str, torch.Tensor] = {}
    renames: list[tuple[str, str]] = []
    for k, v in state_dict.items():
        new_k = k
        for src, dst in rules:
            if src in new_k:
                new_k = new_k.replace(src, dst)
        if new_k != k:
            renames.append((k, new_k))
        new_sd[new_k] = v
    return new_sd, renames


def _drop_unexpected_biases(
    state_dict: dict[str, torch.Tensor],
    target_keys: set[str],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Drop ``*.bias`` keys present in the source but absent from the target."""
    dropped: list[str] = []
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.endswith(".bias") and k not in target_keys:
            dropped.append(k)
        else:
            out[k] = v
    return out, dropped


def _init_per_channel_scales(
    model: nn.Module,
    scheme: QuantScheme,
    policy: ScaleInit,
) -> list[str]:
    """Initialize ``scale_per_channel`` parameters from current FP32 weights."""
    initialized: list[str] = []
    if not scheme.needs_per_channel_scale:
        return initialized
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            if not hasattr(module, "scale_per_channel"):
                continue
            scale = scheme.init_scale(module.weight.data, policy=policy)
            if scale is None:
                continue
            with torch.no_grad():
                module.scale_per_channel.copy_(scale.to(module.scale_per_channel.dtype))
            initialized.append(f"{name}.scale_per_channel")
    return initialized


def load_checkpoint(
    model: nn.Module,
    path: str,
    *,
    mode: CheckpointMode = "strict",
    scheme: Optional[QuantScheme] = None,
    scale_init: ScaleInit = "max_abs",
    map_location: Optional[str] = None,
) -> AdapterReport:
    """Load a checkpoint into ``model`` according to the requested mode.

    Args:
        model:        Target model (already built and replaced; ``self.weight``
                      tensors at the right shapes).
        path:         Filesystem path or URL accepted by :func:`torch.load`.
        mode:         ``"strict"``, ``"fp32_warmstart"``, or ``"scheme_transfer"``.
        scheme:       Required for ``fp32_warmstart`` and ``scheme_transfer``;
                      ignored for ``strict``.
        scale_init:   Scale-init policy used by warm-start.
        map_location: Forwarded to :func:`torch.load`.

    Returns:
        An :class:`AdapterReport` describing what the adapter did.
    """
    raw = torch.load(path, map_location=map_location)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        sd = raw["model_state_dict"]  # legacy training-state checkpoint
    elif isinstance(raw, dict):
        sd = raw  # plain state_dict
    else:
        raise ValueError(f"unexpected checkpoint object: {type(raw)}")

    report = AdapterReport(mode=mode)
    target_keys = set(model.state_dict().keys())

    if mode == "strict":
        missing, unexpected = model.load_state_dict(sd, strict=True)  # type: ignore[arg-type]
        # strict=True raises on mismatch; we never reach this if it fails.
        report.missing_keys = list(missing) if missing else []
        report.unexpected_keys = list(unexpected) if unexpected else []
        return report

    if mode in ("fp32_warmstart", "scheme_transfer"):
        if scheme is None:
            raise ValueError(f"mode={mode!r} requires a scheme")

        # 1) Per-architecture key renames (e.g. fc → linear for ResNet)
        rules = _detect_arch_rules(model)
        sd, renames = _apply_renames(sd, rules)
        report.renamed_keys = renames

        # 2) Drop biases that the target doesn't expect
        sd, dropped = _drop_unexpected_biases(sd, target_keys)
        report.dropped_keys = dropped

        # 3) Non-strict load to deposit weights/BN stats into the model
        result = model.load_state_dict(sd, strict=False)
        report.missing_keys = list(result.missing_keys)
        report.unexpected_keys = list(result.unexpected_keys)

        # 4) Initialize per-channel scales from the now-loaded FP32 weights
        report.initialized_keys = _init_per_channel_scales(model, scheme, scale_init)

        # Anything still missing that is *not* a freshly-initialized scale
        # parameter is a real concern — surface it.
        truly_missing = [k for k in report.missing_keys if k not in set(report.initialized_keys)]
        if truly_missing:
            warnings.warn(
                f"checkpoint adapter: {len(truly_missing)} target keys remain unfilled after "
                f"warm-start: {truly_missing[:5]}{'...' if len(truly_missing) > 5 else ''}",
                stacklevel=2,
            )

        return report

    raise ValueError(f"unknown checkpoint mode: {mode}")
