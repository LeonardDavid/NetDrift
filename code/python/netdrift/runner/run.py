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
import sys
import time
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
    attach_fault_model,
    build_model,
    replace_with_quantized,
)
from netdrift.models.checkpoint import load_checkpoint
from netdrift.quant.binary import BinaryScheme
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
    model = build_model(cfg.model.name)
    if scheme is not None:
        replace_with_quantized(
            model, scheme,
            skip_first=cfg.model.skip_first_quant,
            skip_last=cfg.model.skip_last_quant,
        )

    # 3) Checkpoint
    if cfg.model.checkpoint:
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
                fault_model.cfg.rt_error = float(rt_error)
                print(f"--- rt_error={rt_error}")
                t0 = time.perf_counter()
                accs = evaluate_with_faults(model, test_loader, device, loops=cfg.training.loops)
                elapsed = time.perf_counter() - t0
                all_results.append({
                    "rt_error": float(rt_error),
                    "accuracies": accs,
                    "elapsed_s": round(elapsed, 2),
                })
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
