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
from netdrift.models import (
    apply_protection_policy,
    attach_activation_scheme,
    attach_fault_model,
    build_model,
    replace_with_quantized,
)
from netdrift.models.checkpoint import load_checkpoint
from netdrift.quant.binary import BinaryScheme
from netdrift.quant.uniform import IntUniformActScheme
from netdrift.training import (
    BinaryHingeLoss,
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


def _build_fault_model(cfg: ExperimentConfig, metrics_online: list[str]):
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
        )
        return RTMMisalignmentFault(rtm_cfg)
    raise NotImplementedError(f"fault model {cfg.fault.model!r} not yet implemented")


def _setup_run_dir(cfg: ExperimentConfig) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NetDrift experiment runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config field, e.g. --override fault.rt_error=0.05 (repeatable)",
    )
    args = parser.parse_args(argv)

    overrides = parse_overrides(args.override)
    cfg = load_config(args.config, overrides=overrides)

    _seed_everything(cfg.experiment.seed)
    run_dir = _setup_run_dir(cfg)
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

    # 4) Fault model (skipped for full-precision runs)
    fault_model = None
    if scheme is not None:
        fault_model = _build_fault_model(cfg, cfg.metrics.online)
        apply_protection_policy(
            model, cfg.fault.protection.policy,  # type: ignore[arg-type]
            layers=cfg.fault.protection.layers,
            indiv_layer=cfg.fault.protection.indiv_layer,
        )
        attach_fault_model(
            model, fault_model,
            kernel_mapping=cfg.storage.kernel_mapping.upper() if cfg.storage.kernel_mapping else "ROW",
        )

    # 5) Train or test
    if cfg.training.mode == "train":
        if scheme is None:
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=cfg.training.lr,
                momentum=0.9, weight_decay=1e-4,
            )
        else:
            loss_fn = BinaryHingeLoss(b=128.0)
            optimizer = Clippy(model.parameters(), lr=cfg.training.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma,
        )
        best_acc = 0.0
        for epoch in range(1, cfg.training.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
            acc = evaluate_clean(model, test_loader, device)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), run_dir / "model_best.pt")
            scheduler.step()
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
            rt_errors = (
                cfg.fault.rt_error if isinstance(cfg.fault.rt_error, list) else [cfg.fault.rt_error]
            )
            all_results = []
            for rt_error in rt_errors:
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
                t0 = time.perf_counter()
                accs = evaluate_with_faults(
                    model, test_loader, device,
                    loops=cfg.training.loops,
                    desc_prefix=f"rt_error={rt_error}",
                )
                elapsed = time.perf_counter() - t0
                mean_acc = sum(accs) / len(accs) if accs else 0.0
                print(
                    f"  ⇒ rt_error={rt_error}  mean_acc={mean_acc:.2f}%  "
                    f"min={min(accs):.2f}%  max={max(accs):.2f}%  total_elapsed={elapsed:.2f}s"
                )
                all_results.append({
                    "rt_error": float(rt_error),
                    "accuracies": accs,
                    "elapsed_s": round(elapsed, 2),
                })
            print()
            with open(run_dir / "summary.json", "w") as f:
                json.dump({
                    "rt_error_sweep": all_results,
                    "layer_metrics": _summarize_layer_metrics(model),
                }, f, indent=2)
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
