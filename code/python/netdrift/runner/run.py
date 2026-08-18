"""Experiment orchestrator.

A thin glue layer (~150 lines) replacing ``run.py`` + ``run.sh`` +
``run_all.sh``. Drives the full experiment lifecycle from one YAML config:

1. Load + parse config.
2. Set up device / RNG.
3. Build train / test loaders.
4. Build the FP32 model from the registry.
5. ``replace_with_quantized`` to swap layers in place.
6. Load checkpoint via the appropriate adapter mode.
7. Build the fault model from config; attach to layers.
8. Train or test according to ``training.mode``.
9. Emit metrics (Phase 1: stdout summary; Phase 2: JSONL + W&B).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from netdrift.config import ExperimentConfig, load as load_config, parse_overrides
from netdrift.data import build_datasets
from netdrift.faults.mitigations import get_mitigation
from netdrift.faults.rtm_misalignment import RTMConfig, RTMMisalignmentFault
from netdrift.faults.weight_encoders import (
    apply_weight_encoder_to_model,
    get_encoder,
    is_encoded_checkpoint_path,
    with_endlen_marker,
)
from netdrift.models import (
    apply_protection_policy,
    attach_activation_scheme,
    attach_fault_model,
    build_model,
    replace_with_quantized,
)
from netdrift.models.checkpoint import load_checkpoint
from netdrift.quant.binary import BinaryScheme
from netdrift.runner.wandb_logger import init_wandb_run
from netdrift.quant.uniform import IntUniformActScheme
from netdrift.training import (
    build_criterion,
    Clippy,
    evaluate_clean,
    evaluate_with_faults,
    train_one_epoch,
)


def _build_scheme(cfg: ExperimentConfig):
    if cfg.quant.scheme == "binary":
        return BinaryScheme()
    if cfg.quant.scheme == "none":
        return None
    raise NotImplementedError(
        f"quant scheme {cfg.quant.scheme!r} not yet implemented (Phase 3+)"
    )


def _build_activation_scheme(cfg: ExperimentConfig):
    name = cfg.quant.activation_scheme
    if name in ("none", None):
        return None
    if name == "int_uniform":
        return IntUniformActScheme(bits=cfg.quant.activation_bits)
    raise NotImplementedError(f"activation scheme {name!r} not yet implemented")


def _reset_fault_state(model: torch.nn.Module) -> int:
    """Clear cached RTM fault state on every quantized layer.

    Called between rt_error sweep iterations so each rt_error starts from a
    fresh nanowire baseline. Within a single rt_error, the ``loops`` iterations
    still accumulate state (stuck nanowires stay stuck — legacy semantics).
    """
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
    n = 0
    for _, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            module.fault_state = None
            module.nr_run = 0
            n += 1
    return n


def _rt_mapping_fn_for_layout(layout: str):
    """Return a ``(layer)->str`` rt_mapping callable for ``storage.layout``.

    ``row``/``col``/``block``/``units`` are wired into the fault model.
    ``mix``/``interleaved`` are declared in the schema but not implemented for
    faults yet — raise rather than silently fall back to ROW (which is the
    latent bug this fixes: previously the runner attached the fault model
    with no rt_mapping_fn, so every layer defaulted to ROW regardless of
    ``storage.layout``).
    """
    norm = (layout or "row").lower()
    if norm == "row":
        return lambda _layer: "ROW"
    if norm == "col":
        return lambda _layer: "COL"
    if norm == "block":
        return lambda _layer: "BLOCK"
    if norm == "units":
        return lambda _layer: "UNITS"
    if norm == "polarity":
        return lambda _layer: "POLARITY"
    raise NotImplementedError(
        f"storage.layout={layout!r} is not wired into the fault model yet; "
        f"supported: row, col, block, units, polarity. "
        f"(mix/interleaved are schema placeholders.)"
    )


def _validate_block_layout_combo(cfg: ExperimentConfig) -> None:
    """Fail fast on unsupported BLOCK/UNITS-mapping combinations.

    Neither BLOCK nor UNITS has a rectangular racetrack view, so the
    run-length regularizer and the endlen weight-encoder — both of which lay
    the weight out with ``_layout_weight_for_racetrack`` on the raw mapping
    string — cannot operate on them. Raise a clear message at config time
    instead of a cryptic ``invalid rt_mapping: BLOCK``/``UNITS`` deep inside
    training / encoding.
    """
    layout = (cfg.storage.layout or "").lower()
    if layout not in ("block", "units", "polarity"):
        return
    if cfg.fault.weight_encoder is not None:
        raise ValueError(
            f"storage.layout={layout!r} is not supported with a weight encoder "
            f"(fault.weight_encoder={cfg.fault.weight_encoder!r}); the endlen "
            "encoder has no BLOCK/UNITS/POLARITY layout. Set fault.weight_encoder: null."
        )
    if (cfg.training.fault_aware or "none") != "none":
        # No fault-aware training mode is designed/tested for BLOCK/UNITS. The
        # regularizer has no BLOCK/UNITS/POLARITY layout; ste_inject/kd mutate weight
        # signs across batches while the block/units/polarity paths cache the
        # block/unit structure + guard bands from the first forward (in
        # fault_state_mode='accumulate' the cache is never rebuilt), so the
        # simulation would go silently stale — the same staleness RTMConfig
        # rejects for the per_forward encoder. Fail fast with a clear message
        # instead.
        raise ValueError(
            f"storage.layout={layout!r} is not supported with fault-aware "
            f"training (training.fault_aware={cfg.training.fault_aware!r}); "
            "BLOCK/UNITS caches its structure from the first forward and "
            "cannot track sign-mutating training. Use fault_aware: none."
        )
    if layout == "units":
        # Re-validate here so a hand-edited config fails at load time rather
        # than deep inside the first forward.
        cfg.storage.units.__post_init__()


def _warn_checkpoint_mismatch(cfg: ExperimentConfig) -> None:
    """Warn (don't error) when the checkpoint path conflicts with activation config.

    Heuristic only — parses ``w1a<N>`` or ``bnn`` from the path string. The YAML
    config remains the source of truth for ``activation_bits``; this is purely
    defensive against pointing a w1a4 cfg at a w1a8 checkpoint and vice versa.
    """
    path = (cfg.model.checkpoint or "").lower()
    if not path:
        return
    cfg_scheme = getattr(cfg.quant, "activation_scheme", "none")
    cfg_abits = int(getattr(cfg.quant, "activation_bits", 0) or 0)
    m = re.search(r"w1a(\d+)", path)
    if m:
        ckpt_abits = int(m.group(1))
        if ckpt_abits == 1:
            if cfg_scheme not in ("none", None):
                warnings.warn(
                    f"checkpoint {path!r} looks like W1A1/BNN but "
                    f"quant.activation_scheme={cfg_scheme!r}",
                    stacklevel=2,
                )
        elif cfg_scheme in ("none", None) or cfg_abits != ckpt_abits:
            warnings.warn(
                f"checkpoint {path!r} encodes activation_bits={ckpt_abits} "
                f"but cfg has scheme={cfg_scheme!r}, bits={cfg_abits}",
                stacklevel=2,
            )
        return
    if "bnn" in path and cfg_scheme not in ("none", None):
        warnings.warn(
            f"checkpoint {path!r} looks like BNN but "
            f"quant.activation_scheme={cfg_scheme!r}",
            stacklevel=2,
        )


def _build_fault_model(
    cfg: ExperimentConfig,
    metrics_online: list[str],
    *,
    weight_encoder=None,
    weight_encoder_mode: str = "once",
):
    if cfg.fault.model == "rtm_misalignment":
        # rt_error may be a single float or a list; the runner handles the
        # sweep below by iterating over a normalized list at test time.
        single_rt_error = (
            cfg.fault.rt_error[0] if isinstance(cfg.fault.rt_error, list) else cfg.fault.rt_error
        )
        rtm_cfg = RTMConfig(
            rt_size=cfg.storage.rt_size,
            rt_error=float(single_rt_error),
            mitigations=[get_mitigation(name) for name in cfg.fault.mitigations],
            track_misalign_faults="misalign_faults" in metrics_online,
            track_bitflips="bitflips" in metrics_online,
            track_affected_units="affected_units" in metrics_online,
            track_wrong_reads="wrong_bits_read" in metrics_online,
            weight_encoder=weight_encoder,
            weight_encoder_mode=weight_encoder_mode,
            block_mapping=(cfg.storage.layout == "block"),
            polarity_mapping=(cfg.storage.layout == "polarity"),
            polarity_window=int(cfg.storage.partition.window),
            polarity_pad=bool(cfg.storage.partition.pad),
            units_mapping=(cfg.storage.layout == "units"),
            units_threshold=cfg.storage.units.threshold,
            units_max_period=cfg.storage.units.max_period,
            units_pool_guard=cfg.storage.units.pool_guard,
            edge_mode=cfg.fault.edge_mode,
            ap_position=cfg.fault.ap_position,
        )
        return RTMMisalignmentFault(rtm_cfg)
    raise NotImplementedError(f"fault model {cfg.fault.model!r} not yet implemented")


def _sanitize_path_segment(s: str) -> str:
    """Make a label safe to use as a single path segment.

    Keeps alphanumerics, dot, underscore and hyphen; replaces every other
    character (including path separators and whitespace) with an underscore.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", s.strip())


def _setup_run_dir(
    cfg: ExperimentConfig,
    category: str | None = None,
    subcategory: str | None = None,
) -> tuple[Path, str]:
    """Create and return the run directory + timestamp.

    Layout is ``<output_dir>/<experiment.name>/[<category>/[<subcategory>/]]<timestamp>``.
    The ``category``/``subcategory`` segments are inserted only when provided
    (from ``--wandb-category``/``--wandb-subcategory``) so distinct experiment
    modes that share an ``experiment.name`` no longer collide under
    indistinguishable timestamp dirs. These labels are independent of W&B
    being enabled — the directory layout keys off the labels, not the project.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
    if category:
        run_dir = run_dir / _sanitize_path_segment(category)
        if subcategory:
            run_dir = run_dir / _sanitize_path_segment(subcategory)
    run_dir = run_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ts


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_config_summary(cfg: ExperimentConfig, run_dir: Path) -> None:
    """Print a scannable summary of the resolved config to stdout.

    Mirrors the YAML structure but skips reserved / unused fields. Goes out
    before any heavy lifting so the user can sanity-check what's about to run.
    """
    def fmt(v):
        if isinstance(v, list):
            return "[" + ", ".join(str(x) for x in v) + "]"
        return str(v)

    lines = []
    lines.append("=" * 64)
    lines.append(f"NetDrift run: {cfg.experiment.name}")
    lines.append("=" * 64)
    lines.append(f"experiment : name={cfg.experiment.name}  seed={cfg.experiment.seed}  output={run_dir}")
    lines.append(
        f"model      : {cfg.model.name}  kernel_size={cfg.model.kernel_size}  "
        f"skip_first={cfg.model.skip_first_quant}  skip_last={cfg.model.skip_last_quant}"
    )
    lines.append(
        f"             checkpoint={cfg.model.checkpoint}  mode={cfg.model.checkpoint_mode}"
    )
    lines.append(
        f"data       : {cfg.data.name}  batch={cfg.data.batch_size}/{cfg.data.test_batch_size}  "
        f"workers={cfg.data.num_workers}  dir={cfg.data.data_dir}"
    )
    quant_line = (
        f"quant      : weights={cfg.quant.scheme}  scale_init={cfg.quant.scale_init}  "
        f"activation={cfg.quant.activation_scheme}"
    )
    if cfg.quant.activation_scheme not in ("none", None):
        quant_line += f"({cfg.quant.activation_bits} bits)"
    lines.append(quant_line)
    lines.append(
        f"storage    : layout={cfg.storage.layout}  rt_size={cfg.storage.rt_size}  "
        f"kernel_mapping={cfg.storage.kernel_mapping}"
    )
    fault_line = (
        f"fault      : model={cfg.fault.model}  rt_error={fmt(cfg.fault.rt_error)}  "
        f"mitigations={fmt(cfg.fault.mitigations)}"
    )
    lines.append(fault_line)
    prot = cfg.fault.protection
    prot_line = f"             protection.policy={prot.policy}"
    if prot.policy == "custom" and prot.layers is not None:
        prot_line += f"  layers(unprotected)={fmt(prot.layers)}"
    elif prot.policy == "indiv" and prot.indiv_layer is not None:
        prot_line += f"  indiv_layer={prot.indiv_layer}"
    lines.append(prot_line)
    train_line = (
        f"training   : mode={cfg.training.mode}  fault_aware={cfg.training.fault_aware}"
    )
    if cfg.training.mode == "train":
        train_line += (
            f"  epochs={cfg.training.epochs}  lr={cfg.training.lr}  "
            f"step_size={cfg.training.step_size}  gamma={cfg.training.gamma}"
        )
        if cfg.training.save_dir:
            train_line += f"  save_dir={cfg.training.save_dir}"
    else:
        train_line += f"  loops={cfg.training.loops}"
    lines.append(train_line)
    sink_types = [s.type for s in cfg.metrics.sinks]
    lines.append(
        f"metrics    : online={fmt(cfg.metrics.online)}  sinks={fmt(sink_types)}"
    )
    lines.append(f"gpu_num    : {cfg.gpu_num}")
    lines.append("=" * 64)
    print("\n".join(lines))


def _summarize_layer_metrics(model: torch.nn.Module) -> dict:
    """Walk ``model.named_modules()`` and extract per-layer metric lists."""
    out: dict[str, dict] = {}
    for name, mod in model.named_modules():
        layer_metrics = getattr(mod, "metrics", None)
        if layer_metrics is None:
            continue
        if not getattr(layer_metrics, "data", None):
            continue
        out[name] = {k: list(v) for k, v in layer_metrics.data.items()}
    return out


def _summarize_layer_metrics_by_rt_error(
    model: torch.nn.Module,
    bounds: list[tuple[float, dict[str, dict[str, int]]]],
) -> list[dict]:
    """Nest the raw per-forward metric dump by rt_error.

    ``bounds`` is ``[(rt_error, start_lengths), ...]`` captured at each rt_error
    boundary (start_lengths = per-layer per-key list lengths just before that
    rt_error's passes). Each rt_error's slice runs from its own start to the
    next rt_error's start (or end of list). This makes the dump self-explanatory:
    the per-forward arrays are grouped per rt_error rather than concatenated into
    one array whose stock-metric value "resets" at each (re-seeded) boundary.

    Returns ``[{"rt_error": x, "layer_metrics": {layer: {key: [...]}}}, ...]``.
    """
    raw = _summarize_layer_metrics(model)  # {layer: {key: full_list}}
    out: list[dict] = []
    for i, (rt_error, start) in enumerate(bounds):
        end = bounds[i + 1][1] if i + 1 < len(bounds) else None
        seg: dict[str, dict] = {}
        for layer, keymap in raw.items():
            lstart = start.get(layer, {})
            lend = end.get(layer, {}) if end is not None else None
            seg[layer] = {
                k: v[lstart.get(k, 0):(lend.get(k, len(v)) if lend is not None else len(v))]
                for k, v in keymap.items()
            }
        out.append({"rt_error": rt_error, "layer_metrics": seg})
    return out


def _maybe_warn_per_forward_budget(
    *, mode: str, global_budget: float, local_budget: float
) -> None:
    """Warn that bitflip budgets are ignored in ``per_forward`` mode.

    Budgets require the latent FP weight (magnitude-aware ranking) and a single
    cross-layer selection pass, both of which only exist for ``mode=once``. In
    ``per_forward`` the encoder fires inside every inject with no budget.
    """
    if mode == "per_forward" and (global_budget < 1.0 or local_budget < 1.0):
        warnings.warn(
            "weight_encoder_mode='per_forward' ignores bitflip budgets "
            f"(global={global_budget}, local={local_budget}); budgets apply to "
            "mode='once' only. Set both budgets to 1.0 to silence this.",
            stacklevel=2,
        )


def _resolved_protection(model: torch.nn.Module) -> tuple[list[str], list[str]]:
    """Return (protected, unprotected) quantized-layer *names* after policy.

    Walks ``named_modules()`` and reads the ``protected`` flag set by
    :func:`apply_protection_policy`. Returns resolved name strings rather than
    the 1-based YAML indices so wandb config reflects the actual topology.
    """
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
    protected: list[str] = []
    unprotected: list[str] = []
    for name, mod in model.named_modules():
        if isinstance(mod, (QuantizedConv2d, QuantizedLinear)):
            (protected if getattr(mod, "protected", False) else unprotected).append(name)
    return protected, unprotected


def _layer_metric_lengths(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    """Snapshot the current length of each layer's metric lists.

    Used to slice out the entries produced by a single inference loop: take a
    snapshot before the loop, sum ``data[k][snapshot:]`` after.
    """
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
    out: dict[str, dict[str, int]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (QuantizedConv2d, QuantizedLinear)):
            lm = getattr(mod, "metrics", None)
            if lm is not None and getattr(lm, "data", None) is not None:
                out[name] = {k: len(v) for k, v in lm.data.items()}
    return out


# Per-loop reduction depends on whether a metric is a FLOW or a STOCK.
#
# FLOW metrics count NEW events that occur during a forward pass (the kernel
# increments them only when a fresh fault fires). They are additive: summing a
# flow over the loop's batches — and across loops — is physically meaningful.
# Only ``misalign_faults`` is a flow.
#
# STOCK metrics are pure functions of the STANDING per-racetrack offset state
# (bitflips = pre!=post readout, wrong_bits_read = q_in!=q_out, affected_units =
# count_nonzero(offset)). Each batch RE-MEASURES the same standing corruption,
# so summing over the loop's ~N batches multi-counts the same stuck positions
# (this produced affected_units > total racetracks and BER > 1 in early runs).
# The correct per-loop summary is the END-OF-LOOP SNAPSHOT: the value from the
# loop's LAST batch — "device corruption state after k inference passes". The
# offset state still PERSISTS across loops (stuck-stays-stuck physics), so the
# across-loop trajectory of these snapshots is the cumulative-degradation curve.
_FLOW_METRICS = frozenset({"misalign_faults"})


def _loop_metric_delta(
    model: torch.nn.Module,
    before: dict[str, dict[str, int]],
    online: list[str],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Reduce per-loop fault metrics from the slice produced since ``before``.

    Returns ``(totals, per_layer)``. Each online metric is reduced per the
    flow/stock distinction (see ``_FLOW_METRICS``):

    * FLOW (``misalign_faults``) → SUM over the loop's batches (new events).
    * STOCK (``bitflips``, ``wrong_bits_read``, ``affected_units``) → the
      LAST value in the slice (end-of-loop snapshot of standing corruption).

    ``totals`` aggregates across layers with the SAME reduction (sum of flows /
    sum of per-layer snapshots), so e.g. ``affected_units`` total is the count
    of currently-nonzero racetracks across all layers — bounded by the racetrack
    count, never multi-counted. Only keys in ``online`` are considered.
    """
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
    totals: dict[str, int] = {}
    per_layer: dict[str, dict[str, int]] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, (QuantizedConv2d, QuantizedLinear)):
            continue
        lm = getattr(mod, "metrics", None)
        if lm is None or getattr(lm, "data", None) is None:
            continue
        start = before.get(name, {})
        for key in online:
            values = lm.data.get(key)
            if not values:
                continue
            loop_slice = values[start.get(key, 0):]
            if not loop_slice:
                continue
            if key in _FLOW_METRICS:
                reduced = int(sum(loop_slice))           # additive event count
            else:
                reduced = int(loop_slice[-1])            # end-of-loop snapshot
            totals[key] = totals.get(key, 0) + reduced
            per_layer.setdefault(name, {})[key] = reduced
    return totals, per_layer


def _wandb_config(
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    n_racetracks: int | None = None,
) -> dict:
    """Assemble the wandb ``config`` dict: full resolved cfg + flat conveniences.

    The flattened keys (``model``, ``dataset``, ``rt_size`` ...) make the W&B
    runs table directly filterable/sortable without digging into the nested
    ``config`` blob. ``rt_error`` is intentionally omitted here — the caller
    overrides it per run since each rt_error is its own run.
    """
    protected, unprotected = _resolved_protection(model)
    out = {
        "config": _dataclass_to_dict(cfg),
        "model": cfg.model.name,
        "dataset": cfg.data.name,
        "quant_scheme": cfg.quant.scheme,
        "activation_scheme": cfg.quant.activation_scheme,
        "activation_bits": cfg.quant.activation_bits,
        "rt_size": cfg.storage.rt_size,
        "layout": cfg.storage.layout,
        "kernel_mapping": cfg.storage.kernel_mapping,
        "base_layout": cfg.storage.base_layout,
        "seed": cfg.experiment.seed,
        "loops": cfg.training.loops,
        "mitigations": list(cfg.fault.mitigations),
        "weight_encoder": cfg.fault.weight_encoder,
        "weight_encoder_mode": cfg.fault.weight_encoder_mode,
        "global_bitflip_budget": cfg.fault.global_bitflip_budget,
        "local_bitflip_budget": cfg.fault.local_bitflip_budget,
        "local_budget_scope": cfg.fault.local_budget_scope,
        "budget_selection": cfg.fault.budget_selection,
        "protection_policy": cfg.fault.protection.policy,
        "protected_layers": protected,
        "unprotected_layers": unprotected,
        # Loss criterion — flat keys so the W&B runs table is filterable by the
        # baseline-training loss and the fault-aware-training loss independently.
        "criterion": cfg.training.criterion,
        "hinge_b": cfg.training.hinge_b,
        "fault_aware_criterion": cfg.training.fault_aware_criterion,
        "fault_aware_hinge_b": cfg.training.fault_aware_hinge_b,
        # Edge model — always logged (see FaultCfg docstring): "saturate" (fixed
        # access port, no random reads) vs legacy "random". ``ap_position``
        # stays ``None`` (auto = rt_size//2 - 1) on every arm of the design-space
        # sweep, so logging it makes the resolved value visible rather than
        # implicit.
        "edge_mode": cfg.fault.edge_mode,
        "ap_position": cfg.fault.ap_position,
    }
    # units_* describe storage.units, which is ignored by the schema unless
    # storage.layout == "units" (see StorageCfg docstring). Logging them
    # unconditionally would put schema defaults (e.g. units_threshold=4) on
    # every dense run's config, which is actively misleading in the runs
    # table — the same reasoning that keeps category/subcategory omitted
    # below when unset.
    if cfg.storage.layout == "units":
        out["units_threshold"] = cfg.storage.units.threshold
        out["units_max_period"] = cfg.storage.units.max_period
        out["units_pool_guard"] = cfg.storage.units.pool_guard
    # n_racetracks is the design-space sweep's cost x-axis (total racetracks
    # over all quantized layers — see _total_racetracks). It is expensive to
    # compute for BLOCK/UNITS, so callers compute it once in main() and pass
    # it in; omit rather than log a stale/wrong 0 when it wasn't supplied.
    if n_racetracks is not None:
        out["n_racetracks"] = n_racetracks
    return out


def _wandb_config_with_category(
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    category: str | None,
    subcategory: str | None = None,
    n_racetracks: int | None = None,
) -> dict:
    """``_wandb_config`` plus ``category``/``subcategory`` keys when set.

    Lets the W&B UI group/filter runs by comparison-DB category (coarse, by
    mode) or subcategory (fine, by exact setting combination) natively without a
    post-hoc backfill. Keys are omitted entirely when unset so ad-hoc runs stay
    clean.
    """
    base = _wandb_config(cfg, model, n_racetracks=n_racetracks)
    if category:
        base["category"] = category
    if subcategory:
        base["subcategory"] = subcategory
    return base


def _total_racetracks(cfg: ExperimentConfig, model: torch.nn.Module) -> int:
    """Total racetrack (wire) count over every quantized layer.

    This is the design-space sweep's cost x-axis (see
    ``docs/superpowers/specs/2026-07-31-design-space-weekend-sweep.md`` §1):
    "you pay for every racetrack whether or not its layer is exposed" — the
    count must be identical for ``prot-2to7`` and ``prot-1to8`` under the same
    layout, i.e. protection-invariant.

    That is exactly why the mapping is derived from ``cfg.storage`` here
    instead of reading ``mod.rt_mapping`` / ``mod.base_layout`` /
    ``mod.kernel_mapping`` off the layer, unlike ``_metrics_meta`` below (whose
    per-layer geometry listing is purely descriptive, not a cost total).
    ``_QuantizedMixin._init_quant`` (quant/layers.py) defaults all three of
    those attributes to ``None``, and the top-level ``attach_fault_model`` is
    not guaranteed to overwrite them on a *protected* layer. ``_metrics_meta``
    papers over exactly this with ``mod.rt_mapping or "ROW"`` /
    ``mod.base_layout or "ROW"`` — fine for a display-only field, but silently
    wrong here: a protected layer would count under the ROW/ROW formula
    regardless of the run's actual layout, so two runs that differ only in
    which layers are protected would report two different totals for the same
    layout. Reading straight from ``cfg`` sidesteps protection entirely, so the
    total is correct by construction no matter which layers end up protected.
    """
    from netdrift.faults.layout import compute_index_offset_shape
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    mapping = _rt_mapping_fn_for_layout(cfg.storage.layout)(None)
    base_mapping = cfg.storage.base_layout.upper()
    km = cfg.storage.kernel_mapping.upper() if cfg.storage.kernel_mapping else "ROW"

    total = 0
    for _name, mod in model.named_modules():
        if not isinstance(mod, (QuantizedConv2d, QuantizedLinear)):
            continue
        if mapping == "BLOCK":
            from netdrift.faults.layout import _layout_weight_for_racetrack, build_block_buckets
            w_2d, _ = _layout_weight_for_racetrack(
                mod.weight, rt_mapping=base_mapping, kernel_mapping=km,
            )
            buckets = build_block_buckets(w_2d, cfg.storage.rt_size)
            total += int(sum(b.weight_grid.shape[0] for b in buckets.values()))
        elif mapping == "UNITS":
            from netdrift.faults.layout import _layout_weight_for_racetrack
            from netdrift.faults.packing import build_unit_buckets
            w_2d, _ = _layout_weight_for_racetrack(
                mod.weight, rt_mapping=base_mapping, kernel_mapping=km,
            )
            buckets = build_unit_buckets(
                w_2d, cfg.storage.rt_size,
                threshold=cfg.storage.units.threshold,
                max_period=cfg.storage.units.max_period,
                pool_guard=cfg.storage.units.pool_guard,
            )
            total += int(sum(b.weight_grid.shape[0] for b in buckets.values()))
        elif mapping == "POLARITY":
            # PPM's grid is rectangular (every wire is rt_size) but the wire
            # COUNT is data-dependent: padding each sign group to a wire
            # boundary adds at most one wire per window. Count via the plan so
            # ragged rows and sign-pure windows are exact, not estimated.
            from netdrift.faults.layout import _layout_weight_for_racetrack
            from netdrift.faults.partitioning import count_polarity_racetracks
            w_2d, _ = _layout_weight_for_racetrack(
                mod.weight, rt_mapping=base_mapping, kernel_mapping=km,
            )
            total += count_polarity_racetracks(
                w_2d, cfg.storage.rt_size,
                window=cfg.storage.partition.window,
                pad=cfg.storage.partition.pad,
            )
        else:
            ks = mod._kernel_size_for_state()
            n_rt = compute_index_offset_shape(
                tuple(mod.weight.shape), rt_size=cfg.storage.rt_size,
                rt_mapping=mapping, kernel_size=ks,
            )
            # compute_index_offset_shape returns a 2-tuple grid shape, e.g.
            # (out_dim, ceil(in_dim/rt_size)) for ROW — the racetrack COUNT is
            # the grid's cell count, i.e. the product of both elements (one
            # entry per racetrack: out_dim independent lanes, each split into
            # ceil(in_dim/rt_size) racetracks of length rt_size). Confirmed
            # against analyze_layout_design_space.py's
            # ``dense_wires = rows * ceil(cols / rt_size)`` and
            # test_run_units_wiring.py's ``dense_total = dense_shape[0] *
            # dense_shape[1]``. This is a wire count, matching what the
            # BLOCK/UNITS branches above sum — multiplying by rt_size on top
            # would give a cell count instead, which is a different (larger)
            # quantity this x-axis does not want.
            total += int(n_rt[0]) * int(n_rt[1])
    return total


_METRICS_ONLINE_KEYS = ("bitflips", "misalign_faults", "affected_units", "wrong_bits_read")


def _metrics_track_flags(level: str) -> set[str]:
    """Which fault-model track-keys the metrics level requires."""
    if level in ("online", "all"):
        return set(_METRICS_ONLINE_KEYS)
    return set()


def _metrics_meta(cfg, model, *, category, subcategory) -> dict:
    """Build the self-describing ``meta`` block for metrics artifacts.

    Reuses ``_wandb_config`` for the flat context keys and adds a per-layer
    geometry list (shape, rt_mapping, racetrack count, total weights).
    """
    from netdrift.faults.layout import compute_index_offset_shape
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    base = _wandb_config_with_category(cfg, model, category, subcategory)
    layers = []
    protected_weights = 0
    unprotected_weights = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, (QuantizedConv2d, QuantizedLinear)):
            continue
        ks = mod._kernel_size_for_state()
        shape = tuple(mod.weight.shape)
        mapping = mod.rt_mapping or "ROW"
        if mapping == "BLOCK":
            # BLOCK has no single rectangular racetrack shape; the racetrack
            # count is the number of sign-blocks (data-dependent). Report it as
            # (n_blocks, 1) so the meta geometry stays a 2-tuple like ROW/COL.
            from netdrift.faults.layout import build_block_buckets, _layout_weight_for_racetrack
            base_mapping = (mod.base_layout or "ROW")
            w_2d, _ = _layout_weight_for_racetrack(
                mod.weight, rt_mapping=base_mapping, kernel_mapping=mod.kernel_mapping,
            )
            buckets = build_block_buckets(w_2d, cfg.storage.rt_size)
            n_blocks = int(sum(b.weight_grid.shape[0] for b in buckets.values()))
            n_rt = (n_blocks, 1)
        elif mapping == "UNITS":
            # Like BLOCK, units has no single rectangular racetrack shape: the
            # count is the number of packed wires (data-dependent). Report it as
            # (n_wires, 1) so the meta geometry stays a 2-tuple like ROW/COL.
            from netdrift.faults.layout import _layout_weight_for_racetrack
            from netdrift.faults.packing import build_unit_buckets
            base_mapping = (mod.base_layout or "ROW")
            w_2d, _ = _layout_weight_for_racetrack(
                mod.weight, rt_mapping=base_mapping, kernel_mapping=mod.kernel_mapping,
            )
            buckets = build_unit_buckets(
                w_2d, cfg.storage.rt_size,
                threshold=cfg.storage.units.threshold,
                max_period=cfg.storage.units.max_period,
                pool_guard=cfg.storage.units.pool_guard,
            )
            n_wires = int(sum(b.weight_grid.shape[0] for b in buckets.values()))
            n_rt = (n_wires, 1)
        elif mapping == "POLARITY":
            # Like BLOCK/UNITS the count is data-dependent (padding per window),
            # so report it as (n_wires, 1) to keep the meta geometry a 2-tuple.
            from netdrift.faults.layout import _layout_weight_for_racetrack
            from netdrift.faults.partitioning import count_polarity_racetracks
            base_mapping = (mod.base_layout or "ROW")
            w_2d, _ = _layout_weight_for_racetrack(
                mod.weight, rt_mapping=base_mapping, kernel_mapping=mod.kernel_mapping,
            )
            n_rt = (count_polarity_racetracks(
                w_2d, cfg.storage.rt_size,
                window=cfg.storage.partition.window,
                pad=cfg.storage.partition.pad,
            ), 1)
        else:
            n_rt = compute_index_offset_shape(
                shape, rt_size=cfg.storage.rt_size,
                rt_mapping=mapping, kernel_size=ks,
            )
        nweights = int(mod.weight.numel())
        is_protected = bool(getattr(mod, "protected", False))
        if is_protected:
            protected_weights += nweights
        else:
            unprotected_weights += nweights
        layers.append({
            "id": mod.layer_id,
            "name": name,
            "weight_shape": list(shape),
            "rt_mapping": mod.rt_mapping or "ROW",
            "n_racetracks": list(n_rt),
            "total_weights": nweights,
            "protected": is_protected,
        })
    return {
        "model": base["model"],
        "dataset": base["dataset"],
        "category": category,
        "subcategory": subcategory,
        "quant": {"scheme": base["quant_scheme"], "bits": cfg.quant.bits},
        "storage": {"rt_size": base["rt_size"], "layout": base["layout"],
                    "kernel_mapping": base["kernel_mapping"]},
        "seed": base["seed"], "loops": base["loops"],
        "weight_encoder": base["weight_encoder"],
        "weight_encoder_mode": base["weight_encoder_mode"],
        "protection": {"protected": base["protected_layers"],
                       "unprotected": base["unprotected_layers"]},
        # Quantized-layer weight counts. ``unprotected`` is the BER denominator
        # (only unprotected layers can flip); ``total`` includes protected ones.
        # Counts cover quantized layers only (Conv/Linear) — not BN/Scale params.
        "weights": {
            "total": protected_weights + unprotected_weights,
            "protected": protected_weights,
            "unprotected": unprotected_weights,
        },
        "criterion": base["criterion"],
        "layers": layers,
        "config": base["config"],
    }


_WANDB_MAX_TAG_LEN = 64  # W&B rejects tags longer than this (HTTP 400).


def _wandb_tags(category: str | None, subcategory: str | None = None) -> list[str]:
    """Run tags from the category + subcategory labels (empty when unset).

    Both are always logged as config fields by :func:`_wandb_config_with_category`
    (no length limit there); here we only emit them as TAGS, which W&B caps at
    ``_WANDB_MAX_TAG_LEN`` chars. Long subcategories (e.g. the cat3
    scope×selection×budget combos) exceed that, so they are skipped as tags —
    Group-by still works via ``config.subcategory``.
    """
    return [
        t for t in (category, subcategory)
        if t and len(t) <= _WANDB_MAX_TAG_LEN
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NetDrift experiment runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config field, e.g. --override fault.rt_error=0.05 (repeatable)",
    )
    parser.add_argument(
        "--wandb-project", default=None, metavar="NAME",
        help="Enable Weights & Biases tracking and log to this project. "
             "Off when omitted.",
    )
    parser.add_argument(
        "--wandb-entity", default=None, metavar="ORG",
        help="Optional W&B entity (team/org). Only used with --wandb-project.",
    )
    parser.add_argument(
        "--wandb-category", default=None, metavar="LABEL",
        help="Comparison-DB category label (e.g. 'cat4_endlen_recal'). Logged "
             "to W&B as both config.category and a run tag so runs group "
             "natively by mode. Sweep drivers set this; safe to omit for "
             "ad-hoc runs.",
    )
    parser.add_argument(
        "--wandb-subcategory", default=None, metavar="LABEL",
        help="Finer setting-combination label within a category (e.g. "
             "'cat3_sc-channel_sel-greedy_gl1p0_lo0p1', 'cat5_lam0p05_inj-fresh'). "
             "Logged as config.subcategory + a tag so runs group by exact "
             "configuration. Sweep drivers set this per cell.",
    )
    parser.add_argument(
        "--metrics", default="none", choices=["none", "offline", "online", "all"],
        help="Metrics-tracking level. 'offline' = static-weight snapshots only; "
             "'online' = per-iteration fault/accuracy artifacts; 'all' = both. "
             "Default 'none' (existing behaviour, no JSON artifacts).",
    )
    args = parser.parse_args(argv)

    overrides = parse_overrides(args.override)
    cfg = load_config(args.config, overrides=overrides)
    _validate_block_layout_combo(cfg)

    # Union the requested metrics level into the online metric list so the
    # fault model enables the matching track_* flags. Computed once and passed
    # everywhere the online list is consumed.
    metrics_online = sorted(set(cfg.metrics.online) | _metrics_track_flags(args.metrics))

    _seed_everything(cfg.experiment.seed)
    run_dir, run_ts = _setup_run_dir(cfg, args.wandb_category, args.wandb_subcategory)
    wandb_group = f"{cfg.experiment.name}-{run_ts}"
    # Snapshot the resolved config for traceability.
    with open(run_dir / "config.json", "w") as f:
        json.dump(_dataclass_to_dict(cfg), f, indent=2, default=str)

    _print_config_summary(cfg, run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_num)

    # 1) Datasets
    train_ds, test_ds, num_classes = build_datasets(cfg.data.name, cfg.data.data_dir)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.data.test_batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # 2) Model — built FP32, then quantized in place (skipped when scheme is None)
    scheme = _build_scheme(cfg)
    activation_scheme = _build_activation_scheme(cfg)
    model = build_model(cfg.model.name)
    if scheme is not None:
        replace_with_quantized(
            model, scheme,
            skip_first=cfg.model.skip_first_quant,
            skip_last=cfg.model.skip_last_quant,
        )
    # Bind activation scheme to every QuantizedActivation in the topology.
    # With activation_scheme=None this is a no-op (qact modules remain identity).
    attach_activation_scheme(model, activation_scheme)

    # 3) Checkpoint
    if cfg.model.checkpoint:
        _warn_checkpoint_mismatch(cfg)
        report = load_checkpoint(
            model,
            cfg.model.checkpoint,
            mode=cfg.model.checkpoint_mode,  # type: ignore[arg-type]
            scheme=scheme,
            scale_init=cfg.quant.scale_init,  # type: ignore[arg-type]
            map_location=str(device),
        )
        print(report.summary())

    model.to(device)
    print(model)

    # 4) Resolve optional weight encoder; auto-disable if the loaded
    # checkpoint already looks pre-encoded (filename convention).
    encoder = None
    encoder_mode = cfg.fault.weight_encoder_mode
    encoder_auto_disabled = False
    if cfg.fault.weight_encoder is not None:
        if is_encoded_checkpoint_path(cfg.model.checkpoint):
            encoder_auto_disabled = True
            banner = "─" * 64
            print()
            print(banner)
            print("  pre-encoded checkpoint detected — auto-disabling encoder")
            print(f"  checkpoint            : {cfg.model.checkpoint}")
            print(f"  configured encoder    : {cfg.fault.weight_encoder} "
                  f"(mode={cfg.fault.weight_encoder_mode})")
            print("  To force re-encoding, point model.checkpoint at a path")
            print("  whose basename does not contain the 'endlen' marker.")
            print(banner)
        else:
            encoder = get_encoder(cfg.fault.weight_encoder)
    elif is_encoded_checkpoint_path(cfg.model.checkpoint):
        # Encoder not requested, but the user appears to be loading an
        # already-encoded model — a one-line reminder, no banner.
        print(
            f"note: checkpoint {cfg.model.checkpoint!r} looks pre-encoded "
            f"and no weight_encoder is configured; proceeding as a normal load."
        )

    # 5) Fault model (skipped for full-precision runs)
    # Encoder is forwarded into RTMConfig so per_forward mode fires inside
    # inject; for mode=once it's a no-op at the fault-model level.
    fault_model = None
    if scheme is not None:
        fault_model = _build_fault_model(
            cfg, metrics_online,
            weight_encoder=encoder,
            weight_encoder_mode=encoder_mode,
        )
        apply_protection_policy(
            model, cfg.fault.protection.policy,  # type: ignore[arg-type]
            layers=cfg.fault.protection.layers,
            indiv_layer=cfg.fault.protection.indiv_layer,
        )
        attach_fault_model(
            model, fault_model,
            rt_mapping_fn=_rt_mapping_fn_for_layout(cfg.storage.layout),
            kernel_mapping=cfg.storage.kernel_mapping.upper() if cfg.storage.kernel_mapping else "ROW",
            base_layout=(cfg.storage.base_layout.upper()
                         if cfg.storage.layout in ("block", "units", "polarity") else None),
        )

    # n_racetracks (the design-space sweep's cost x-axis) is computed ONCE
    # here rather than inside _wandb_config_with_category, because for
    # BLOCK/UNITS it runs the real packer over every layer's actual weight
    # signs — real CPU work (measured in the millions of runs at low
    # threshold), not something to redo per W&B run. The test-mode
    # base_wandb_config build below is already hoisted outside the rt_error
    # loop, so this single value covers every W&B run of a sweep (train
    # mode's one run plus every rt_error run in test mode). Guarded so a
    # packer bug/regression can never take down an experiment over a
    # convenience metric — worst case the key is omitted.
    n_racetracks: int | None = None
    try:
        n_racetracks = _total_racetracks(cfg, model)
    except Exception as exc:  # noqa: BLE001 - cost metric must never be fatal
        warnings.warn(
            f"_total_racetracks failed ({exc!r}); omitting n_racetracks from "
            "the W&B config for this run",
            stacklevel=2,
        )

    # 6) Train or test
    if cfg.training.mode == "train":
        if scheme is None:
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=cfg.training.lr,
                momentum=0.9, weight_decay=1e-4,
            )
        else:
            loss_fn = build_criterion(
                cfg.training.criterion, cfg.training.hinge_b
            )
            optimizer = Clippy(model.parameters(), lr=cfg.training.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma,
        )
        run = init_wandb_run(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=wandb_group,
            name=f"{cfg.experiment.name}-train",
            config=_wandb_config_with_category(
                cfg, model, args.wandb_category, args.wandb_subcategory,
                n_racetracks=n_racetracks,
            ),
            tags=_wandb_tags(args.wandb_category, args.wandb_subcategory),
        )
        # Fault-aware dispatch. With fault_aware != none on a quantized model,
        # route through the fault-aware loop, which handles the gradient path
        # (the attached fault model detaches gradients otherwise) and the
        # run-length regularizer / fault-state mode. fault_aware == none keeps
        # the legacy plain loop (unchanged behaviour).
        from netdrift.training import train_one_epoch_fault_aware
        fault_aware = cfg.training.fault_aware
        best_acc = 0.0
        for epoch in range(1, cfg.training.epochs + 1):
            if scheme is not None and fault_aware != "none":
                train_loss = train_one_epoch_fault_aware(
                    model, train_loader, optimizer, device,
                    rt_size=cfg.storage.rt_size,
                    layout=cfg.storage.layout,
                    kernel_mapping=cfg.storage.kernel_mapping or "row",
                    cfg=cfg.training,
                    epoch=epoch,
                )
            else:
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, loss_fn, device, epoch
                )
            acc = evaluate_clean(model, test_loader, device)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), run_dir / "model_best.pt")
            run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_accuracy": acc,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
            scheduler.step()
        run.set_summary({"best_accuracy": best_acc, "epochs": cfg.training.epochs})
        run.finish()
        torch.save(model.state_dict(), run_dir / "model.pt")
        with open(run_dir / "train_summary.json", "w") as f:
            json.dump({"best_accuracy": best_acc, "epochs": cfg.training.epochs}, f, indent=2)
        if cfg.training.save_dir:
            import shutil
            sd = Path(cfg.training.save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / "model.pt", sd / "model.pt")
            if (run_dir / "model_best.pt").exists():
                shutil.copy2(run_dir / "model_best.pt", sd / "model_best.pt")
    elif cfg.training.mode == "test":
        if fault_model is None:
            acc = evaluate_clean(model, test_loader, device)
            with open(run_dir / "summary.json", "w") as f:
                json.dump({"accuracy": acc}, f, indent=2)
        else:
            # Baseline 0: clean accuracy, no fault injection, no encoder.
            # We temporarily detach the fault model so layers run as pure
            # quantized matmuls — this is the pre-encode reference.
            print()
            print("Baseline 0: clean accuracy, no faults, no encoder")
            kernel_mapping_str = (
                cfg.storage.kernel_mapping.upper()
                if cfg.storage.kernel_mapping else "ROW"
            )
            attach_fault_model(model, None)
            baseline_clean_acc = evaluate_clean(model, test_loader, device)
            attach_fault_model(
                model, fault_model,
                rt_mapping_fn=_rt_mapping_fn_for_layout(cfg.storage.layout),
                kernel_mapping=kernel_mapping_str,
                base_layout=(cfg.storage.base_layout.upper()
                             if cfg.storage.layout in ("block", "units", "polarity") else None),
            )
            print(f"  ⇒ baseline_clean_accuracy = {baseline_clean_acc:.2f}%")

            # ---- metrics: meta + offline snapshots ----
            # NOTE: must come AFTER the re-attach above so each layer's
            # rt_mapping is set; otherwise snapshots use the wrong (ROW) layout.
            _metrics_level = args.metrics
            _do_offline = _metrics_level in ("offline", "all")
            _do_online = _metrics_level in ("online", "all")
            _cat_tok = args.wandb_category or "uncategorized"
            _recal_deltas = None
            _snap_objs: list = []
            _meta = None
            if _do_offline or _do_online:
                # Imports + meta build only when metrics are actually requested,
                # so --metrics none stays zero-overhead.
                from netdrift.metrics import snapshots as _snap
                from netdrift.metrics.artifacts import (
                    distribution_stats, write_rt_error_artifact, write_static_artifact,
                )
                from netdrift.metrics.online import OnlineCollector
                # Named distinctly from the "metrics" source package
                # (code/python/netdrift/metrics/) so mutagen sync rules can
                # target run artifacts and the source package independently —
                # both were called "metrics" before, which made them
                # impossible to tell apart by bare directory name.
                _metrics_dir = run_dir / "metrics_artifacts"
                _meta = _metrics_meta(cfg, model, category=args.wandb_category,
                                      subcategory=args.wandb_subcategory)
            if _do_offline:
                lbl = "before_encoder" if (encoder is not None and encoder_mode == "once") else "trained"
                s0 = _snap.capture_snapshot(model, label=lbl, rt_size=cfg.storage.rt_size,
                                            want_raw=(_metrics_level == "all"))
                _snap_objs.append(s0)

            # Optional encoder application.
            baseline_endlen_acc: float | None = None
            encoded_checkpoint_path: str | None = None
            # wandb summary scalars for the encoder (mode=once). Empty when no
            # encoder ran; populated below from the per-layer flip report.
            encoder_summary: dict[str, float] = {}
            if encoder is not None and encoder_mode == "once":
                print()
                print(
                    f"Applying weight encoder {cfg.fault.weight_encoder!r} "
                    f"(mode=once) to model weights..."
                )
                from netdrift.faults.weight_encoders.budget import BudgetConfig
                _maybe_warn_per_forward_budget(
                    mode=encoder_mode,
                    global_budget=cfg.fault.global_bitflip_budget,
                    local_budget=cfg.fault.local_bitflip_budget,
                )
                report = apply_weight_encoder_to_model(
                    model, encoder,
                    rt_size=cfg.storage.rt_size,
                    rt_mapping=cfg.storage.layout.upper(),
                    kernel_mapping_default=(
                        cfg.storage.kernel_mapping.upper()
                        if cfg.storage.kernel_mapping else "ROW"
                    ),
                    budget=BudgetConfig(
                        global_budget=cfg.fault.global_bitflip_budget,
                        local_budget=cfg.fault.local_bitflip_budget,
                        scope=cfg.fault.local_budget_scope,  # type: ignore[arg-type]
                        selection=cfg.fault.budget_selection,  # type: ignore[arg-type]
                    ),
                )
                from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
                layer_totals = {
                    n: m.weight.numel()
                    for n, m in model.named_modules()
                    if isinstance(m, (QuantizedConv2d, QuantizedLinear))
                }
                protected_layers = [
                    n for n, m in model.named_modules()
                    if isinstance(m, (QuantizedConv2d, QuantizedLinear))
                    and getattr(m, "protected", False)
                ]
                # report: {name: {"flipped", "rejected", "fraction"}}
                total_changed = sum(r["flipped"] for r in report.values())
                total_rejected = sum(r["rejected"] for r in report.values())
                total_weights = sum(layer_totals[n] for n in report)
                overall_pct = (
                    100.0 * total_changed / total_weights if total_weights else 0.0
                )
                print(
                    f"  encoded {len(report)} unprotected layers; "
                    f"{total_changed}/{total_weights} weight entries changed "
                    f"({overall_pct:.2f}%); {total_rejected} rejected by budget"
                )
                encoder_summary["encoder/total_weights_flipped"] = float(total_changed)
                encoder_summary["encoder/total_weights_rejected"] = float(total_rejected)
                encoder_summary["encoder/total_weights"] = float(total_weights)
                encoder_summary["encoder/overall_pct"] = float(overall_pct)
                for layer_name, r in report.items():
                    changed = r["flipped"]
                    rejected = r["rejected"]
                    layer_total = layer_totals[layer_name]
                    pct = 100.0 * changed / layer_total if layer_total else 0.0
                    encoder_summary[f"encoder/flipped/{layer_name}"] = float(changed)
                    encoder_summary[f"encoder/flipped_pct/{layer_name}"] = float(pct)
                    encoder_summary[f"encoder/rejected/{layer_name}"] = float(rejected)
                    print(
                        f"    {layer_name:32s} "
                        f"{changed:8d} / {layer_total:8d} bits flipped "
                        f"({pct:6.2f}%)  rejected={rejected}"
                    )
                if protected_layers:
                    print(
                        f"  skipped {len(protected_layers)} protected layer(s): "
                        f"{', '.join(protected_layers)}"
                    )

                # Save the encoded model. Default: alongside the source
                # checkpoint with ``_endlen`` injected into the basename
                # (e.g. ``models/.../model_best.pt`` →
                # ``models/.../model_best_endlen.pt``). Falls back to
                # ``<run_dir>/model_endlen.pt`` when no source checkpoint
                # was configured. ``encoded_checkpoint_save`` overrides both.
                if cfg.fault.encoded_checkpoint_save:
                    save_path_raw = cfg.fault.encoded_checkpoint_save
                elif cfg.model.checkpoint:
                    save_path_raw = cfg.model.checkpoint
                else:
                    save_path_raw = str(run_dir / "model_endlen.pt")
                save_path = with_endlen_marker(save_path_raw)
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                encoded_checkpoint_path = save_path
                print(f"  encoded checkpoint saved to: {save_path}")

                # Baseline 1: clean accuracy on the encoded weights, no faults.
                print()
                print("Baseline 1: clean accuracy, after encoder (mode=once)")
                attach_fault_model(model, None)
                baseline_endlen_acc = evaluate_clean(model, test_loader, device)
                attach_fault_model(
                    model, fault_model,
                    rt_mapping_fn=_rt_mapping_fn_for_layout(cfg.storage.layout),
                    kernel_mapping=kernel_mapping_str,
                    base_layout=(cfg.storage.base_layout.upper()
                                 if cfg.storage.layout in ("block", "units", "polarity") else None),
                )
                print(f"  ⇒ baseline_endlen_accuracy = {baseline_endlen_acc:.2f}%")
                print(
                    f"  Δ vs clean baseline       = "
                    f"{baseline_endlen_acc - baseline_clean_acc:+.2f}%"
                )

                # rt_mapping restored by the re-attach above — safe to snapshot.
                if _do_offline:
                    s_after = _snap.capture_snapshot(
                        model, label="after_encoder", rt_size=cfg.storage.rt_size,
                        want_raw=(_metrics_level == "all"),
                    )
                    _snap_objs.append(s_after)

            elif encoder is not None and encoder_mode == "per_forward":
                # Baseline 1 for per_forward: temporarily set rt_error=0 so
                # the offset kernel produces no shifts; the encoder still
                # fires inside inject, so the read-out reflects encoded
                # weights without any fault.
                print()
                print(
                    "Baseline 1: clean accuracy with encoder, "
                    "mode=per_forward, rt_error=0"
                )
                _reset_fault_state(model)
                _seed_everything(cfg.experiment.seed)
                saved_rt_error = fault_model.cfg.rt_error
                fault_model.cfg.rt_error = 0.0
                accs = evaluate_with_faults(
                    model, test_loader, device, loops=1,
                    desc_prefix="baseline_endlen (per_forward)",
                )
                fault_model.cfg.rt_error = saved_rt_error
                baseline_endlen_acc = accs[0] if accs else 0.0
                print(
                    f"  ⇒ baseline_endlen_accuracy = "
                    f"{baseline_endlen_acc:.2f}%"
                )
                print(
                    f"  Δ vs clean baseline       = "
                    f"{baseline_endlen_acc - baseline_clean_acc:+.2f}%"
                )

            # Capability #1: pattern-preserving recalibration (BN+Scale), no faults.
            # Runs after the encoder produced the endlen'd weights and before the
            # rt_error sweep. The fault model is detached for the recalibration
            # forward passes (no faults), then restored; binary weight signs stay
            # frozen so the endlen pattern is preserved.
            baseline_endlen_recal_acc: float | None = None
            recal_cfg = cfg.training.recalibrate
            do_recal = recal_cfg.enabled and (
                recal_cfg.on == "always" or (recal_cfg.on == "endlen" and encoder is not None)
            )
            if do_recal:
                from netdrift.training import recalibrate
                print()
                print("Recalibration: BN running stats + output Scale (no faults)")
                # Detach faults for the recalibration forward passes; restore in
                # a finally so a failure mid-recal doesn't leave the model with
                # no fault model attached for the sweep.
                _recal_before = _snap.capture_recal_params(model) if _do_offline else None
                attach_fault_model(model, None)
                try:
                    recalibrate(
                        model, train_loader, device, recal_cfg,
                        criterion=cfg.training.criterion,
                        hinge_b=cfg.training.hinge_b,
                    )
                    baseline_endlen_recal_acc = evaluate_clean(model, test_loader, device)
                finally:
                    attach_fault_model(
                        model, fault_model,
                        rt_mapping_fn=_rt_mapping_fn_for_layout(cfg.storage.layout),
                        kernel_mapping=kernel_mapping_str,
                        base_layout=(cfg.storage.base_layout.upper()
                                     if cfg.storage.layout in ("block", "units", "polarity") else None),
                    )
                print(
                    f"  ⇒ baseline_endlen_recal_accuracy = "
                    f"{baseline_endlen_recal_acc:.2f}%"
                )
                # rt_mapping is restored here (post-finally) — safe for COL configs.
                if _do_offline:
                    _recal_deltas = _snap.recal_deltas(
                        _recal_before, _snap.capture_recal_params(model)
                    )
                    s_recal = _snap.capture_snapshot(
                        model, label="after_recal", rt_size=cfg.storage.rt_size,
                        want_raw=(_metrics_level == "all"),
                    )
                    _snap_objs.append(s_recal)
                # Save the recalibrated model with a _recal marker. Next to the
                # encoded checkpoint when one exists (parallel to the _endlen
                # convention); otherwise (e.g. on=always with no encoder) fall
                # back to the run dir so the recalibrated weights are never lost.
                if encoded_checkpoint_path:
                    rp = Path(encoded_checkpoint_path)
                    recal_path = str(rp.with_name(f"{rp.stem}_recal{rp.suffix}"))
                else:
                    recal_path = str(run_dir / "model_recal.pt")
                torch.save(model.state_dict(), recal_path)
                print(f"  recalibrated checkpoint saved to: {recal_path}")

            rt_errors = (
                cfg.fault.rt_error if isinstance(cfg.fault.rt_error, list) else [cfg.fault.rt_error]
            )
            base_wandb_config = _wandb_config_with_category(
                cfg, model, args.wandb_category, args.wandb_subcategory,
                n_racetracks=n_racetracks,
            )
            online = metrics_online
            all_results = []
            # Track where each rt_error's per-forward records begin in the raw
            # LayerMetrics lists, so summary.json can nest the dump by rt_error
            # instead of flattening all sweeps into one array (whose value
            # "resets" at each rt_error boundary, which looks like a bug).
            rt_metric_bounds: list[tuple[float, dict[str, dict[str, int]]]] = []
            for rt_error in rt_errors:
                rt_metric_bounds.append((float(rt_error), _layer_metric_lengths(model)))
                n_reset = _reset_fault_state(model)
                # Re-seed RNGs so each rt_error realization is independent of
                # which sweep values preceded it. The fault kernel re-seeds via
                # random.randint(1, 1000) per inject; consuming the same number
                # of draws from a fresh RNG state makes the sweep reproducible.
                _seed_everything(cfg.experiment.seed)
                fault_model.cfg.rt_error = float(rt_error)
                bar = "─" * 64
                print()
                print(bar)
                print(f"  rt_error = {rt_error}    (reset fault state on {n_reset} layers)")
                print(bar)

                # One wandb run per rt_error; the whole sweep shares a group.
                run = init_wandb_run(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    group=wandb_group,
                    name=f"{cfg.experiment.name}-rt{rt_error}",
                    config={**base_wandb_config, "rt_error": float(rt_error)},
                    tags=_wandb_tags(args.wandb_category, args.wandb_subcategory),
                )
                # Baselines + encoder report as one-shot summary scalars so they
                # are available alongside the per-iteration curves.
                run.set_summary({
                    "baseline_clean_accuracy": baseline_clean_acc,
                    **(
                        {"baseline_endlen_accuracy": baseline_endlen_acc}
                        if baseline_endlen_acc is not None else {}
                    ),
                    **(
                        {"baseline_endlen_recal_accuracy": baseline_endlen_recal_acc}
                        if baseline_endlen_recal_acc is not None else {}
                    ),
                    **encoder_summary,
                })

                t0 = time.perf_counter()
                accs: list[float] = []
                _online = OnlineCollector() if _do_online else None
                # Drive the loops here (loops=1 per call) so each inference
                # iteration can be logged with its own fault-metric slice. State
                # still accumulates across iterations — we do NOT reset between
                # them, matching the legacy loops semantics.
                for loop_idx in range(1, cfg.training.loops + 1):
                    before = _layer_metric_lengths(model)
                    acc = evaluate_with_faults(
                        model, test_loader, device,
                        loops=1,
                        desc_prefix=f"rt_error={rt_error} iter {loop_idx}/{cfg.training.loops}",
                    )[0]
                    accs.append(acc)
                    totals, per_layer = _loop_metric_delta(model, before, online)
                    if _online is not None:
                        _online.record_loop(loop_idx=loop_idx, totals=totals, per_layer=per_layer)
                    log_data: dict[str, float] = {
                        "loop_idx": loop_idx,
                        "accuracy": acc,
                        # Logged every iteration → flat reference lines on charts.
                        "baseline_clean_accuracy": baseline_clean_acc,
                        "accuracy_drop_vs_clean": baseline_clean_acc - acc,
                    }
                    if baseline_endlen_acc is not None:
                        log_data["baseline_endlen_accuracy"] = baseline_endlen_acc
                        log_data["accuracy_drop_vs_endlen"] = baseline_endlen_acc - acc
                    for k, v in totals.items():
                        log_data[k] = v
                    for lname, lmetrics in per_layer.items():
                        for k, v in lmetrics.items():
                            log_data[f"layer/{lname}/{k}"] = v
                    run.log(log_data, step=loop_idx)

                elapsed = time.perf_counter() - t0
                mean_acc = sum(accs) / len(accs) if accs else 0.0
                print(
                    f"  ⇒ rt_error={rt_error}  mean_acc={mean_acc:.2f}%  "
                    f"min={min(accs):.2f}%  max={max(accs):.2f}%  total_elapsed={elapsed:.2f}s"
                )
                run.set_summary({
                    "mean_accuracy": mean_acc,
                    "min_accuracy": min(accs) if accs else 0.0,
                    "max_accuracy": max(accs) if accs else 0.0,
                    "elapsed_s": round(elapsed, 2),
                })
                run.finish()
                all_results.append({
                    "rt_error": float(rt_error),
                    "accuracies": accs,
                    "elapsed_s": round(elapsed, 2),
                })

                if _do_online:
                    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
                    _layers = {n: m for n, m in model.named_modules()
                               if isinstance(m, (QuantizedConv2d, QuantizedLinear))}
                    _npz = _online.final_raw_arrays(_layers) if _metrics_level == "all" else None
                    per_loop = _online.as_per_loop()
                    # BER denominator = UNPROTECTED weights only. Protected
                    # layers (e.g. conv1, fc2) can never flip — faults are only
                    # injected into unprotected layers — so dividing by all
                    # weights would dilute the rate with fault-immune params.
                    # CAVEAT: this makes BER comparable across protection
                    # policies only when the unprotected SET is the same; it is
                    # the fraction of EXPOSED weights read wrong, not of all.
                    # Reuse the count from _meta["weights"] so the BER denominator
                    # and the meta block can never disagree.
                    unprotected_weights = _meta["weights"]["unprotected"]
                    # last_loop: per-metric END-OF-LOOP value from the FINAL
                    # inference loop. For STOCK metrics (bitflips, wrong_bits_read,
                    # affected_units) this is the standing-corruption snapshot
                    # after all loops; for the FLOW metric (misalign_faults) it is
                    # that final loop's new-event count. (_loop_metric_delta already
                    # applied the per-loop flow/stock reduction; here we just take
                    # the last loop's value.)
                    last_loop = {k: (v["total"][-1] if v["total"] else 0)
                                 for k, v in per_loop.items()}
                    # sum_over_loops only for FLOW metrics — summing stock
                    # snapshots across loops is physically meaningless (it would
                    # re-sum the same standing corruption). See _FLOW_METRICS.
                    sum_over_loops = {
                        k: int(sum(v["total"]))
                        for k, v in per_loop.items() if k in _FLOW_METRICS
                    }
                    # BER from the final-loop bitflips snapshot: fraction of
                    # exposed weights standing wrong after all loops. Bounded [0,1].
                    ber = (
                        last_loop.get("bitflips", 0) / unprotected_weights
                        if unprotected_weights else 0.0
                    )
                    outcome = {
                        "baselines": {
                            "clean": baseline_clean_acc,
                            "endlen": baseline_endlen_acc,
                            "endlen_recal": baseline_endlen_recal_acc,
                        },
                        "per_loop_accuracy": accs,
                        "accuracy": distribution_stats(accs),
                        "accuracy_drop_vs_clean": distribution_stats(
                            [baseline_clean_acc - a for a in accs]
                        ),
                    }
                    fault_incidence = {
                        "last_loop": {**last_loop, "ber": ber,
                                      "ber_denominator_unprotected_weights": unprotected_weights},
                        "sum_over_loops": sum_over_loops,
                        "per_loop": per_loop,
                    }
                    write_rt_error_artifact(
                        _metrics_dir, model=cfg.model.name, category=_cat_tok,
                        rt_error=float(rt_error), meta=dict(_meta),
                        outcome=outcome, fault_incidence=fault_incidence,
                        npz_arrays=_npz,
                        static_ref=(f"{cfg.model.name}__{_cat_tok}__static.json"
                                    if _do_offline else None),
                    )
            print()
            with open(run_dir / "summary.json", "w") as f:
                json.dump({
                    "baseline_clean_accuracy": baseline_clean_acc,
                    "baseline_endlen_accuracy": baseline_endlen_acc,
                    "baseline_endlen_recal_accuracy": baseline_endlen_recal_acc,
                    "weight_encoder": (
                        cfg.fault.weight_encoder if encoder is not None else None
                    ),
                    "weight_encoder_mode": (
                        encoder_mode if encoder is not None else None
                    ),
                    "encoded_checkpoint": encoded_checkpoint_path,
                    "encoder_auto_disabled": encoder_auto_disabled,
                    # Total racetracks over ALL quantized layers under
                    # cfg.storage.layout — the design-space cost x-axis, and
                    # protection-invariant by construction (see
                    # _total_racetracks). Persisted here, not just logged to
                    # W&B, so an offline harvester can validate the analytic
                    # cost table against the simulator without rebuilding the
                    # model: scripts/sweep_design_space.py's preflight compares
                    # this against its ARMS wire counts. None if the
                    # computation was skipped or failed (never fatal).
                    "n_racetracks": n_racetracks,
                    "rt_error_sweep": all_results,
                    # Raw per-forward metric dump, NESTED PER rt_error (each entry
                    # is one forward pass / batch). Within an rt_error the stock
                    # metrics accumulate as the offset drifts; the fault state is
                    # reset + re-seeded between rt_errors, so grouping by rt_error
                    # makes the boundary explicit instead of one flat array that
                    # appears to "reset" mid-stream. For reduced/analysis-ready
                    # numbers use the metrics_artifacts/*.json artifacts, not this dump.
                    "layer_metrics_by_rt_error": _summarize_layer_metrics_by_rt_error(
                        model, rt_metric_bounds
                    ),
                }, f, indent=2)

            if _do_offline and _snap_objs:
                snapshots_json = [
                    {"label": s.label, "total": s.totals,
                     "per_layer": {n: {
                         "block_count": m.block_count,
                         "sign_transitions": m.sign_transitions,
                         "run_length_histogram": m.run_length_histogram,
                         "alternating_seq_histogram": m.alternating_seq_histogram,
                         "weight_magnitude": m.weight_magnitude,
                         "dist_to_threshold": m.dist_to_threshold,
                         "n_racetracks": list(m.n_racetracks),
                     } for n, m in s.per_layer.items()}}
                    for s in _snap_objs
                ]
                # Consecutive pairwise deltas so each boundary's effect is
                # isolated. For cat4 (before_encoder, after_encoder, after_recal)
                # this yields the pure-endlen delta AND the recal delta
                # separately, instead of one combined first->last delta.
                deltas = {}
                for b, a in zip(_snap_objs, _snap_objs[1:]):
                    deltas[f"{b.label}->{a.label}"] = _snap.compute_deltas(b, a)
                static_npz = {}
                if _metrics_level == "all":
                    import numpy as _np
                    for s in _snap_objs:
                        for n, m in s.per_layer.items():
                            if m.raw_sign_transitions is not None:
                                static_npz[f"static__{s.label}__{n}__sign_transitions"] = \
                                    _np.asarray(m.raw_sign_transitions)
                            if m.raw_alternating_lengths is not None:
                                static_npz[f"static__{s.label}__{n}__alternating_lengths"] = \
                                    _np.asarray(m.raw_alternating_lengths)
                write_static_artifact(
                    _metrics_dir, model=cfg.model.name, category=_cat_tok,
                    meta=dict(_meta), snapshots=snapshots_json,
                    deltas=deltas or None, recal=_recal_deltas,
                    npz_arrays=static_npz or None,
                )
    else:
        raise ValueError(f"unknown training mode: {cfg.training.mode}")

    print(f"Run artifacts written to {run_dir}")
    return 0


def _dataclass_to_dict(obj):  # type: ignore[no-untyped-def]
    """Recursive dataclass → dict for JSON dumping."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(obj):
        return asdict(obj)
    return obj


if __name__ == "__main__":
    sys.exit(main())
