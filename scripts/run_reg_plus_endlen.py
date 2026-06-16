#!/usr/bin/env python
"""Comparison-DB category 7: regularizer-trained model + endlen on top.

The "parked third question": once fault-aware training has pushed weights toward
long same-sign racetrack runs, does applying endlen on top — at a LOW budget —
still help? The regularized model already has few sign transitions, so endlen
should find little to merge; the interesting question is whether a small budget
buys robustness the encoder-free model lacks, without endlen's usual large
clean-accuracy sacrifice.

This is NOT the Capability-#2 deliverable (that's the encoder-free model, cat 5).
It's a database cell for completeness: regularizer AND endlen, low budget.

For each regularizer-trained checkpoint (one per seed, from sweep_regularizer.py
or the reg_train mode), apply endlen at each ``--budgets`` value and evaluate
over the rt_error curve. Endlen here is local-budget (global=1.0), greedy.

Usage::

    python scripts/run_reg_plus_endlen.py \
        --config configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml \
        --checkpoints models/w1a1/vgg3_fmnist/reg_lam0p05/model.pt \
        --budgets 0.05 0.1 \
        --wandb-project netdrift-comparison-db

    # multiple seed checkpoints, with seed tags so the aggregator groups them
    python scripts/run_reg_plus_endlen.py --config <cfg> \
        --checkpoints runs/.../seed707/model.pt:707 runs/.../seed1/model.pt:1 \
        --budgets 0.1 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Shared comparison-DB contract (sibling module).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparison_common import (  # noqa: E402
    DEFAULT_LOOPS,
    DEFAULT_PROTECTION_LAYERS,
    DEFAULT_RT_ERROR_CURVE,
    base_overrides,
    fmt_num,
    harvest_summary,
    import_runner_main,
    latest_summary,
    new_sweep_out_dir,
    output_dir_from_cfg,
    run_cell,
    wandb_args,
    write_manifest,
)


def _parse_checkpoint(spec: str) -> tuple[str, str | None]:
    """Parse ``path`` or ``path:seedtag`` into ``(path, seed_tag_or_None)``.

    A trailing ``:<tag>`` lets the caller stamp a seed onto the experiment name
    so the aggregator groups multi-seed checkpoints (it strips ``_seed\\d+``).
    Only split on the LAST ':' so Windows-style paths aren't mangled (we still
    only expect POSIX paths here, but be safe).
    """
    if ":" in spec:
        head, _, tail = spec.rpartition(":")
        if tail.isdigit():
            return head, tail
    return spec, None


def _experiment_name(base_stem: str, budget: float, seed_tag: str | None) -> str:
    name = f"{base_stem}__cat7_reg-endlen_lo{fmt_num(budget)}"
    if seed_tag is not None:
        name += f"_seed{seed_tag}"
    return name


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="Base YAML config (provides model/data/storage).")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Regularizer-trained checkpoint(s), each 'path' or "
                        "'path:seedtag'. One per seed for multi-seed grouping.")
    p.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.1],
                   help="Local endlen budgets to apply on top (global=1.0). "
                        "Low values: the model is already mostly same-sign.")
    p.add_argument("--selection", default="greedy",
                   help="Endlen budget_selection (greedy|value_per_flip|"
                        "magnitude_aware). Default greedy.")
    p.add_argument("--scope", default="channel",
                   help="local_budget_scope (layer|racetrack|channel). "
                        "Default channel.")
    p.add_argument("--rt-curve", nargs="+", type=float,
                   default=DEFAULT_RT_ERROR_CURVE)
    p.add_argument("--loops", type=int, default=DEFAULT_LOOPS)
    p.add_argument("--protection-layers", nargs="+", type=int,
                   default=DEFAULT_PROTECTION_LAYERS)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 2
    base_stem = cfg_path.stem
    runner_out_dir = output_dir_from_cfg(cfg_path)
    curve = [float(x) for x in args.rt_curve]

    checkpoints = [_parse_checkpoint(c) for c in args.checkpoints]

    # One cell per (checkpoint, budget).
    cells: list[dict] = []
    for ckpt_path, seed_tag in checkpoints:
        for b in args.budgets:
            cells.append({
                "checkpoint": ckpt_path,
                "seed_tag": seed_tag,
                "budget": b,
                "experiment_name": _experiment_name(base_stem, b, seed_tag),
            })
    total = len(cells)

    print(f"cat7 reg+endlen: {len(checkpoints)} checkpoint(s) × "
          f"{len(args.budgets)} budget(s) = {total} cells")
    print(f"  endlen        : local-budget (global=1.0), scope={args.scope}, "
          f"selection={args.selection}")
    print(f"  budgets       : {args.budgets}")
    print(f"  rt_curve      : {curve}")
    print(f"  loops         : {args.loops}")
    print(f"  protection    : "
          f"{'from config' if args.protection_layers is None else f'custom layers={args.protection_layers}'}")
    print(f"  wandb         : {args.wandb_project or 'DISABLED'}")
    print()

    if args.dry_run:
        for i, c in enumerate(cells, 1):
            print(f"[{i:3d}/{total}] ckpt={c['checkpoint']} "
                  f"seed={c['seed_tag']} budget={c['budget']} "
                  f"name={c['experiment_name']}")
        return 0

    runner_main = import_runner_main()
    out_dir = new_sweep_out_dir("reg_plus_endlen")
    results: list[dict] = []
    sweep_t0 = time.perf_counter()

    for i, cell in enumerate(cells, 1):
        argv_cell = ["--config", str(cfg_path)]
        argv_cell += base_overrides(
            curve=curve, loops=args.loops,
            protection_layers=args.protection_layers,
        )
        argv_cell += [
            "--override", f"model.checkpoint={cell['checkpoint']}",
            "--override", "model.checkpoint_mode=strict",
            "--override", "training.mode=test",
            "--override", "fault.weight_encoder=endlen",
            "--override", "fault.weight_encoder_mode=once",
            "--override", "fault.global_bitflip_budget=1.0",
            "--override", f"fault.local_bitflip_budget={cell['budget']}",
            "--override", f"fault.local_budget_scope={args.scope}",
            "--override", f"fault.budget_selection={args.selection}",
            "--override", f"experiment.name={cell['experiment_name']}",
        ]
        # subcategory groups by the low endlen budget applied on top.
        argv_cell += wandb_args(
            args.wandb_project, args.wandb_entity, "cat7_reg_endlen",
            f"cat7_reg_endlen_lo{fmt_num(cell['budget'])}",
        )

        bar = "=" * 72
        print(bar)
        print(f"[cell {i}/{total}] ckpt={Path(cell['checkpoint']).name} "
              f"budget={cell['budget']} seed={cell['seed_tag']}")
        print(bar)
        t0 = time.perf_counter()
        status, err = run_cell(runner_main, argv_cell)
        if status != "ok":
            print(f"  !! cell {status}: {err}")
        elapsed = time.perf_counter() - t0

        harvested = harvest_summary(
            latest_summary(runner_out_dir, cell["experiment_name"])
        )
        if harvested["rt_curve"]:
            curve_str = "  ".join(
                f"{rt:g}:{m['mean']:.1f}"
                for rt, m in sorted(harvested["rt_curve"].items())
            )
            print(f"  ⇒ {status}  endlen_clean={harvested['baseline_endlen_accuracy']}  "
                  f"curve(mean) [{curve_str}]  ({elapsed:.1f}s)")
        else:
            print(f"  ⇒ {status}  ({elapsed:.1f}s)")

        results.append({
            **cell,
            "status": status,
            "error": err,
            "elapsed_s": round(elapsed, 1),
            "baseline_clean_accuracy": harvested["baseline_clean_accuracy"],
            "baseline_endlen_accuracy": harvested["baseline_endlen_accuracy"],
            "rt_curve": harvested["rt_curve"],
        })
        write_manifest(out_dir, {
            "config": str(cfg_path),
            "checkpoints": args.checkpoints,
            "budgets": args.budgets,
            "scope": args.scope,
            "selection": args.selection,
            "rt_curve": curve,
            "loops": args.loops,
            "protection_layers": args.protection_layers,
            "wandb_project": args.wandb_project,
            "results": results,
        })

    # CSV summary: one row per cell, rt_curve means as columns.
    csv_path = out_dir / "reg_plus_endlen.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["experiment", "checkpoint", "seed", "budget",
                  "clean", "endlen_clean"] + [f"rt_{rt:g}_mean" for rt in curve]
        w.writerow(header)
        for r in results:
            row = [
                r["experiment_name"], r["checkpoint"], r["seed_tag"], r["budget"],
                r["baseline_clean_accuracy"], r["baseline_endlen_accuracy"],
            ]
            for rt in curve:
                cell = r["rt_curve"].get(float(rt))
                row.append(f"{cell['mean']:.4f}" if cell else "")
            w.writerow(row)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print()
    print(f"cat7 done: {n_ok}/{total} ok  "
          f"({time.perf_counter() - sweep_t0:.1f}s total)")
    print(f"  → {csv_path}")
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
