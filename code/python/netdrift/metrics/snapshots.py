"""Capture static-metric snapshots at weight-mutating pipeline boundaries.

A :class:`Snapshot` is the static metrics of every quantized layer at one
labelled point in the pipeline (e.g. ``"before_encoder"``, ``"after_encoder"``,
``"after_recal"``, or ``"trained"`` for single-snapshot runs). Deltas between
two snapshots quantify what a weight-rewriting step (e.g. endlen) did.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn

from netdrift.metrics.static import StaticLayerMetrics, compute_static_metrics


@dataclass
class Snapshot:
    label: str
    per_layer: dict[str, StaticLayerMetrics]
    totals: dict
    # Captured binarized signs per layer, for cross-snapshot bitflip counting.
    _signs: dict[str, torch.Tensor]
    # Captured per-weight signed distance-to-threshold (sign(w)*|w/scale|), for
    # cross-snapshot per-weight threshold-movement deltas. A sign flip moves a
    # weight from +d to -d (change of 2d), which |w|-only stats cannot see.
    _signed_dist: dict[str, torch.Tensor]


def _quant_layers(model: torch.nn.Module):
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
    for name, mod in model.named_modules():
        if isinstance(mod, (QuantizedConv2d, QuantizedLinear)):
            yield name, mod


def capture_snapshot(
    model: torch.nn.Module, label: str, *, rt_size: int, want_raw: bool = False
) -> Snapshot:
    per_layer: dict[str, StaticLayerMetrics] = {}
    signs: dict[str, torch.Tensor] = {}
    signed_dist: dict[str, torch.Tensor] = {}
    tot_pos = tot_neg = tot_transitions = 0
    tot_runlen: Counter = Counter()
    tot_alt: Counter = Counter()
    for name, mod in _quant_layers(model):
        scale = getattr(mod, "scale_per_channel", None)
        # units_params (threshold, max_period, pool_guard) lives on the
        # layer's fault model's RTMConfig, not on the layer itself — the
        # layer-attribute path deliberately avoids the (mount-absent)
        # models/ package. The fault model may be absent entirely (e.g. a
        # clean model with nothing attached yet), so degrade to None rather
        # than raising here; compute_static_metrics decides what None means
        # for the current rt_mapping.
        fault_model = getattr(mod, "fault_model", None)
        fm_cfg = getattr(fault_model, "cfg", None) if fault_model is not None else None
        units_params = None
        polarity_params = None
        if fm_cfg is not None:
            units_params = (
                int(fm_cfg.units_threshold),
                int(fm_cfg.units_max_period),
                int(fm_cfg.units_pool_guard),
            )
            # Same rationale as units_params: (window, pad) lives on the
            # RTMConfig. getattr-with-default keeps a pre-PPM pickled/stubbed
            # cfg from breaking the snapshot path.
            polarity_params = (
                int(getattr(fm_cfg, "polarity_window", 0)),
                bool(getattr(fm_cfg, "polarity_pad", True)),
            )
        m = compute_static_metrics(
            mod.weight, rt_mapping=mod.rt_mapping or "ROW",
            kernel_mapping=mod.kernel_mapping, rt_size=rt_size,
            base_layout=getattr(mod, "base_layout", None),
            units_params=units_params,
            polarity_params=polarity_params,
            per_channel_scale=scale, want_raw=want_raw,
        )
        per_layer[name] = m
        w = mod.weight.detach()
        signs[name] = torch.where(w > 0, 1, -1)  # match BinaryScheme: sign(0) = -1
        # Per-weight signed distance to threshold-0 (effective scale applied).
        # CLONE: ``w.float()`` aliases ``mod.weight`` when it's already float, so
        # a later in-place weight mutation (e.g. endlen sign-flip) would corrupt
        # this snapshot's stored values and zero out the before/after delta.
        d = w.float().clone()
        if scale is not None:
            view = [scale.shape[0]] + [1] * (d.dim() - 1)
            d = d / scale.detach().float().reshape(view)
        signed_dist[name] = d
        tot_pos += m.block_count["pos"]
        tot_neg += m.block_count["neg"]
        tot_transitions += m.sign_transitions
        for k, v in m.run_length_histogram.items():
            tot_runlen[k] += v
        for k, v in m.alternating_seq_histogram.items():
            tot_alt[k] += v
    totals = {
        "block_count": {"pos": tot_pos, "neg": tot_neg, "total": tot_pos + tot_neg},
        "sign_transitions": tot_transitions,
        "run_length_histogram": dict(tot_runlen),
        "alternating_seq_histogram": dict(tot_alt),
        "total_alternating_sequences": int(sum(tot_alt.values())),
    }
    return Snapshot(label=label, per_layer=per_layer, totals=totals,
                    _signs=signs, _signed_dist=signed_dist)


def compute_deltas(before: Snapshot, after: Snapshot) -> dict:
    """Per-layer and total deltas between two snapshots.

    * ``bitflips`` — weights whose binarized sign changed.
    * ``block_count_change`` — after-minus-before block-count difference.
    * ``abs_dist_to_threshold_change_mean`` — mean over weights of the
      per-weight ``|signed_dist_after - signed_dist_before|``. A sign flip moves
      a weight from ``+d`` to ``-d`` (a change of ``2d``), so this captures how
      far the flipped weights sat from the threshold — unlike a diff of mean
      ``|w|``, which is sign-flip-invariant and ~0 for endlen.
    """
    per_layer: dict[str, dict] = {}
    tot_bitflips = tot_block_change = tot_alt_change = 0
    tot_abs_change_sum = 0.0
    tot_count = 0
    for name, m_after in after.per_layer.items():
        m_before = before.per_layer.get(name)
        if m_before is None:
            continue
        bitflips = int((before._signs[name] != after._signs[name]).sum().item())
        block_change = m_after.block_count["total"] - m_before.block_count["total"]
        before_alt = sum(m_before.alternating_seq_histogram.values())
        after_alt = sum(m_after.alternating_seq_histogram.values())
        alt_change = after_alt - before_alt
        abs_change = (after._signed_dist[name] - before._signed_dist[name]).abs()
        mean_abs_change = float(abs_change.mean().item())
        per_layer[name] = {
            "bitflips": bitflips,
            "block_count_change": block_change,
            "alternating_seq_count_change": alt_change,
            "abs_dist_to_threshold_change_mean": mean_abs_change,
        }
        tot_bitflips += bitflips
        tot_block_change += block_change
        tot_alt_change += alt_change
        tot_abs_change_sum += float(abs_change.sum().item())
        tot_count += abs_change.numel()
    return {
        "total": {
            "bitflips": tot_bitflips,
            "block_count_change": tot_block_change,
            "alternating_seq_count_change": tot_alt_change,
            "abs_dist_to_threshold_change_mean": (
                tot_abs_change_sum / tot_count if tot_count else 0.0
            ),
        },
        "per_layer": per_layer,
    }


def capture_recal_params(model: torch.nn.Module) -> dict:
    """Snapshot the parameters recalibration moves.

    Returns ``{"bn": {param_name: tensor}, "scale": {param_name: tensor}}`` with
    detached clones. Covers everything ``recalibrate`` touches: BN affine
    gamma/beta (``_set_recal_trainable``) AND BN running_mean/running_var
    (``_reestimate_bn_stats``) — for a binary net the running stats shift the
    EFFECTIVE decision boundary as much as gamma/beta, so they're part of the
    "recal moves the threshold" story. Plus the model-level ``scale.scale``.
    """
    bn: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if mod.weight is not None:
                bn[f"{name}.weight".lstrip(".")] = mod.weight.detach().clone()
            if mod.bias is not None:
                bn[f"{name}.bias".lstrip(".")] = mod.bias.detach().clone()
            if mod.running_mean is not None:
                bn[f"{name}.running_mean".lstrip(".")] = mod.running_mean.detach().clone()
            if mod.running_var is not None:
                bn[f"{name}.running_var".lstrip(".")] = mod.running_var.detach().clone()
    scale: dict[str, torch.Tensor] = {}
    scale_mod = getattr(model, "scale", None)
    if scale_mod is not None and hasattr(scale_mod, "scale"):
        scale["scale.scale"] = scale_mod.scale.detach().clone()
    return {"bn": bn, "scale": scale}


def recal_deltas(before: dict, after: dict) -> dict:
    """Per-parameter mean absolute change between two recal-param snapshots."""
    out: dict[str, dict] = {"bn": {}, "scale": {}}
    for group in ("bn", "scale"):
        for pname, a in after[group].items():
            b = before[group].get(pname)
            if b is None or b.shape != a.shape:
                continue
            out[group][pname] = {
                "abs_mean_change": float((a - b).abs().mean().item()),
            }
    return out
