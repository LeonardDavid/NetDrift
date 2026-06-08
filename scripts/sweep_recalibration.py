#!/usr/bin/env python
"""Comparison-DB driver — categories 4 and 6: endlen + recalibration.

Category 4 — endlen + recalibration (vanilla endlen, global=1.0, local=1.0):
  4a  BN-stats only (no backprop, deterministic) — 1 seed.
  4b  BN-stats + affine backprop (tune_affine=true, epochs=2) — one run per seed.

Category 6 — regularizer-trained model + recalibration (optional):
  Requires --reg-checkpoint pointing to a model.pt from the regularizer driver.
  Runs with weight_encoder=null (no endlen) and recalibrate.on=always.
  One run per seed.

Usage::

    # Category 4 only
    python scripts/sweep_recalibration.py \\
        --config configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml \\
        --seeds 707 1 42

    # Category 4 + 6
    python scripts/sweep_recalibration.py \\
        --config configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml \\
        --seeds 707 1 42 \\
        --reg-checkpoint runs/reg_trained/model.pt \\
        --wandb-project netdrift-comparison-db

    # Dry-run: list all cells without executing
    python scripts/sweep_recalibration.py \\
        --config configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

# Put the scripts/ dir on sys.path so comparison_common is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparison_common import (  # noqa: E402
    DEFAULT_LOOPS,
    DEFAULT_PROTECTION_LAYERS,
    DEFAULT_RT_ERROR_CURVE,
    DEFAULT_WANDB_PROJECT,
    REPO_ROOT,
    base_overrides,
    harvest_summary,
    import_runner_main,
    json_list,
    latest_summary,
    new_sweep_out_dir,
    output_dir_from_cfg,
    run_cell,
    wandb_args,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Experiment-name helper
# ---------------------------------------------------------------------------

def _experiment_name(base_stem: str, tag: str) -> str:
    """Return a unique, wandb-safe experiment name for one cell.

    Convention mirrors comparison_common: ``<cfg_stem>__<cell_tag>``.
    """
    return f"{base_stem}__{tag}"


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

def _build_cells(
    cfg_path: Path,
    seeds: list[int],
    reg_checkpoint: Optional[str],
    categories: Optional[set[str]] = None,
) -> list[dict]:
    """Return the ordered list of cell descriptors (no argv yet).

    ``categories`` selects which of ``{"4a", "4b", "6"}`` to build; ``None`` =
    all (legacy behaviour). Category 6 additionally requires ``reg_checkpoint``.
    """
    base_stem = cfg_path.stem
    want = categories if categories is not None else {"4a", "4b", "6"}
    cells: list[dict] = []

    # ---- Category 4a — BN-only, deterministic, 1 seed -------------------
    if "4a" in want:
        cells.append({
            "category": "4a",
            "exp_name": _experiment_name(base_stem, "cat4_recal-bn"),
            "seed": None,        # deterministic; no seed override
            "reg_ckpt": None,
        })

    # ---- Category 4b — BN + affine backprop, one run per seed -----------
    if "4b" in want:
        for seed in seeds:
            cells.append({
                "category": "4b",
                "exp_name": _experiment_name(
                    base_stem, f"cat4_recal-bn-affine_seed{seed}"
                ),
                "seed": seed,
                "reg_ckpt": None,
            })

    # ---- Category 6 — regularizer model + recalibration (optional) ------
    if "6" in want and reg_checkpoint is not None:
        for seed in seeds:
            cells.append({
                "category": "6",
                "exp_name": _experiment_name(
                    base_stem, f"cat6_reg-recal_seed{seed}"
                ),
                "seed": seed,
                "reg_ckpt": reg_checkpoint,
            })

    return cells


def _cell_argv(
    cell: dict,
    cfg_path: Path,
    curve: list[float],
    loops: int,
    protection_layers: list[int],
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
) -> list[str]:
    """Assemble the full argv list for one cell."""
    argv = ["--config", str(cfg_path)]

    # Shared overrides: rt_error curve, loops, protection policy.
    argv += base_overrides(
        curve=curve,
        loops=loops,
        protection_layers=protection_layers,
    )

    cat = cell["category"]

    if cat == "4a":
        # Vanilla endlen budgets; BN-stats only (no backprop).
        argv += [
            "--override", "fault.global_bitflip_budget=1.0",
            "--override", "fault.local_bitflip_budget=1.0",
            "--override", "training.recalibrate.enabled=true",
            "--override", "training.recalibrate.bn_stats=true",
            "--override", "training.recalibrate.tune_affine=false",
            "--override", "training.recalibrate.epochs=0",
        ]

    elif cat == "4b":
        # Vanilla endlen budgets; BN + affine backprop.
        argv += [
            "--override", "fault.global_bitflip_budget=1.0",
            "--override", "fault.local_bitflip_budget=1.0",
            "--override", "training.recalibrate.enabled=true",
            "--override", "training.recalibrate.bn_stats=true",
            "--override", "training.recalibrate.tune_affine=true",
            "--override", "training.recalibrate.epochs=2",
            "--override", "training.recalibrate.lr=0.001",
        ]

    elif cat == "6":
        # Regularizer-trained checkpoint; no endlen; always-recalibrate.
        argv += [
            "--override", f"model.checkpoint={cell['reg_ckpt']}",
            "--override", "fault.weight_encoder=null",
            "--override", "training.recalibrate.enabled=true",
            "--override", "training.recalibrate.on=always",
            "--override", "training.recalibrate.bn_stats=true",
            "--override", "training.recalibrate.tune_affine=true",
            "--override", "training.recalibrate.epochs=2",
            "--override", "training.recalibrate.lr=0.001",
        ]

    # Experiment identity overrides.
    argv += ["--override", f"experiment.name={cell['exp_name']}"]
    if cell["seed"] is not None:
        argv += ["--override", f"experiment.seed={cell['seed']}"]

    # W&B (no-op when wandb_project is None). category = coarse mode;
    # subcategory = exact recal variant so 4a/4b group separately within cat4.
    _CAT_LABEL = {
        "4a": "cat4_endlen_recal",
        "4b": "cat4_endlen_recal",
        "6": "cat6_reg_recal",
    }
    _SUBCAT_LABEL = {
        "4a": "cat4_recal-bn",            # BN-stats only
        "4b": "cat4_recal-bn-affine",     # BN + affine/Scale backprop
        "6": "cat6_reg-recal",
    }
    argv += wandb_args(
        wandb_project, wandb_entity,
        _CAT_LABEL.get(cat), _SUBCAT_LABEL.get(cat),
    )

    return argv


# ---------------------------------------------------------------------------
# Table rendering (no numpy)
# ---------------------------------------------------------------------------

def _fmt(v: object) -> str:
    """Format a numeric value or substitute an em-dash."""
    if isinstance(v, float):
        return f"{v:.2f}"
    return "—"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    return statistics.pstdev(xs) if xs else float("nan")


def _write_tables(
    out_dir: Path,
    results: list[dict],
    curve: list[float],
    cfg_path: Path,
) -> None:
    """Write the Markdown + CSV summary tables after all cells finish."""

    # Group results by category for the per-category sections.
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    md_lines: list[str] = []
    md_lines.append("# Recalibration sweep — categories 4 and 6")
    md_lines.append("")
    md_lines.append(f"- config: `{cfg_path.name}`")
    md_lines.append(f"- rt_error curve: {curve}")
    md_lines.append("")

    # ------------------------------------------------------------------ Cat 4a
    cat4a = by_cat.get("4a", [])
    md_lines.append("## Category 4a — BN-stats only (no backprop)")
    md_lines.append("")
    if cat4a:
        r = cat4a[0]
        md_lines.append(f"- status: {r['status']}")
        md_lines.append(f"- baseline_clean_accuracy: {_fmt(r['baseline_clean_accuracy'])}")
        md_lines.append(f"- baseline_endlen_accuracy: {_fmt(r['baseline_endlen_accuracy'])}")
        md_lines.append(f"- baseline_endlen_recal_accuracy: {_fmt(r['baseline_endlen_recal_accuracy'])}")
        md_lines.append("")
        # rt_curve table
        rt_curve = r.get("rt_curve") or {}
        if rt_curve:
            header = ["rt_error", "mean", "min", "max", "last"]
            md_lines.append("| " + " | ".join(header) + " |")
            md_lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for rt in curve:
                m = rt_curve.get(float(rt))
                if m:
                    md_lines.append(
                        f"| {rt:g} | {m['mean']:.2f} | {m['min']:.2f} | "
                        f"{m['max']:.2f} | {m['last']:.2f} |"
                    )
                else:
                    md_lines.append(f"| {rt:g} | — | — | — | — |")
            md_lines.append("")
    else:
        md_lines.append("_no results_")
        md_lines.append("")

    # ------------------------------------------------------------------ Cat 4b
    cat4b = by_cat.get("4b", [])
    md_lines.append("## Category 4b — BN + affine backprop (per-seed)")
    md_lines.append("")
    if cat4b:
        # Per-seed rows
        seed_header = ["seed", "status", "clean", "endlen", "endlen_recal"] + [
            f"rt={rt:g}" for rt in curve
        ]
        md_lines.append("| " + " | ".join(seed_header) + " |")
        md_lines.append("|" + "|".join(["---"] * len(seed_header)) + "|")
        for r in cat4b:
            rt_curve = r.get("rt_curve") or {}
            rt_vals = []
            for rt in curve:
                m = rt_curve.get(float(rt))
                rt_vals.append(f"{m['mean']:.2f}" if m else "—")
            row = [
                str(r["seed"]),
                r["status"],
                _fmt(r["baseline_clean_accuracy"]),
                _fmt(r["baseline_endlen_accuracy"]),
                _fmt(r["baseline_endlen_recal_accuracy"]),
            ] + rt_vals
            md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("")

        # Mean ± std across seeds for each rt_error
        md_lines.append("### Mean ± std across seeds")
        md_lines.append("")
        agg_header = ["rt_error", "mean", "std"]
        md_lines.append("| " + " | ".join(agg_header) + " |")
        md_lines.append("|" + "|".join(["---"] * len(agg_header)) + "|")
        for rt in curve:
            means = []
            for r in cat4b:
                rt_curve = r.get("rt_curve") or {}
                m = rt_curve.get(float(rt))
                if m is not None:
                    means.append(m["mean"])
            if means:
                md_lines.append(
                    f"| {rt:g} | {_mean(means):.2f} | {_std(means):.2f} |"
                )
            else:
                md_lines.append(f"| {rt:g} | — | — |")
        md_lines.append("")
    else:
        md_lines.append("_no results_")
        md_lines.append("")

    # ------------------------------------------------------------------ Cat 6
    cat6 = by_cat.get("6", [])
    md_lines.append("## Category 6 — regularizer-trained model + recalibration (per-seed)")
    md_lines.append("")
    if cat6:
        seed_header = ["seed", "status", "clean", "endlen", "endlen_recal"] + [
            f"rt={rt:g}" for rt in curve
        ]
        md_lines.append("| " + " | ".join(seed_header) + " |")
        md_lines.append("|" + "|".join(["---"] * len(seed_header)) + "|")
        for r in cat6:
            rt_curve = r.get("rt_curve") or {}
            rt_vals = []
            for rt in curve:
                m = rt_curve.get(float(rt))
                rt_vals.append(f"{m['mean']:.2f}" if m else "—")
            row = [
                str(r["seed"]),
                r["status"],
                _fmt(r["baseline_clean_accuracy"]),
                _fmt(r["baseline_endlen_accuracy"]),
                _fmt(r["baseline_endlen_recal_accuracy"]),
            ] + rt_vals
            md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("")

        md_lines.append("### Mean ± std across seeds")
        md_lines.append("")
        agg_header = ["rt_error", "mean", "std"]
        md_lines.append("| " + " | ".join(agg_header) + " |")
        md_lines.append("|" + "|".join(["---"] * len(agg_header)) + "|")
        for rt in curve:
            means = []
            for r in cat6:
                rt_curve = r.get("rt_curve") or {}
                m = rt_curve.get(float(rt))
                if m is not None:
                    means.append(m["mean"])
            if means:
                md_lines.append(
                    f"| {rt:g} | {_mean(means):.2f} | {_std(means):.2f} |"
                )
            else:
                md_lines.append(f"| {rt:g} | — | — |")
        md_lines.append("")
    else:
        md_lines.append("_Category 6 was not requested (--reg-checkpoint not provided)._")
        md_lines.append("")

    md_path = out_dir / "recalibration_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"  -> {md_path}")

    # ---------------------------------------------------------------------- CSV
    # One flat CSV with all cells for machine consumption.
    csv_path = out_dir / "recalibration_summary.csv"
    rt_col_names = [f"rt_{rt:g}_mean" for rt in curve]
    fieldnames = (
        ["category", "seed", "status", "error", "elapsed_s",
         "baseline_clean_accuracy", "baseline_endlen_accuracy",
         "baseline_endlen_recal_accuracy", "experiment_name"]
        + rt_col_names
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            rt_curve = r.get("rt_curve") or {}
            row = {
                "category": r["category"],
                "seed": r["seed"] if r["seed"] is not None else "",
                "status": r["status"],
                "error": r.get("error") or "",
                "elapsed_s": r.get("elapsed_s", ""),
                "baseline_clean_accuracy": r["baseline_clean_accuracy"],
                "baseline_endlen_accuracy": r["baseline_endlen_accuracy"],
                "baseline_endlen_recal_accuracy": r["baseline_endlen_recal_accuracy"],
                "experiment_name": r["exp_name"],
            }
            for rt in curve:
                m = rt_curve.get(float(rt))
                col = f"rt_{rt:g}_mean"
                row[col] = f"{m['mean']:.4f}" if m else ""
            w.writerow(row)
    print(f"  -> {csv_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", required=True,
        help="Base YAML config (e.g. configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml). "
             "Must have weight_encoder=endlen set.",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[707, 1, 42],
        help="Random seeds for seeded cells (cat 4b, cat 6). Default: 707 1 42",
    )
    p.add_argument(
        "--rt-curve", nargs="+", type=float, default=DEFAULT_RT_ERROR_CURVE,
        help="rt_error curve to evaluate over. Default: the canonical DB curve.",
    )
    p.add_argument(
        "--loops", type=int, default=DEFAULT_LOOPS,
        help=f"training.loops override (inference iterations per rt_error). "
             f"Default: {DEFAULT_LOOPS}",
    )
    p.add_argument(
        "--protection-layers", nargs="+", type=int,
        default=DEFAULT_PROTECTION_LAYERS,
        help="Layer indices kept UNPROTECTED (custom protection policy). "
             f"Default: {DEFAULT_PROTECTION_LAYERS}",
    )
    p.add_argument(
        "--reg-checkpoint", default=None, metavar="PATH",
        help="Path to a model.pt from the regularizer driver. Enables category 6. "
             "Omit to skip category 6.",
    )
    p.add_argument(
        "--categories", nargs="+", default=None, choices=["4a", "4b", "6"],
        metavar="CAT",
        help="Subset of {4a,4b,6} to run. Default: all. "
             "E.g. '--categories 4b' for BN+affine+Scale only (no BN-only 4a); "
             "'--categories 6' for regularizer+recal only (needs --reg-checkpoint).",
    )
    p.add_argument(
        "--wandb-project", default=None,
        help=f"W&B project name. Pass '{DEFAULT_WANDB_PROJECT}' to log to the "
             "shared comparison-DB project. Omit for local-only sweep.",
    )
    p.add_argument("--wandb-entity", default=None, help="W&B entity/team.")
    p.add_argument(
        "--dry-run", action="store_true",
        help="List every cell that would run (with argv) and exit without executing.",
    )
    args = p.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    curve: list[float] = [float(x) for x in args.rt_curve]
    seeds: list[int] = args.seeds
    reg_checkpoint: Optional[str] = args.reg_checkpoint

    # Resolve experiment output_dir (where the runner writes summary.json).
    runner_out_dir = output_dir_from_cfg(cfg_path)
    if not runner_out_dir.is_absolute():
        runner_out_dir = REPO_ROOT / runner_out_dir

    categories = set(args.categories) if args.categories else None
    cells = _build_cells(cfg_path, seeds, reg_checkpoint, categories)
    total = len(cells)

    # ---------------------------------------------------------------------- header
    print("=" * 72)
    print("Recalibration sweep — categories 4 and 6")
    print("=" * 72)
    print(f"  config             : {cfg_path}")
    print(f"  seeds              : {seeds}")
    print(f"  rt_error curve     : {curve}")
    print(f"  loops              : {args.loops}")
    print(f"  protection_layers  : {args.protection_layers}")
    print(f"  reg_checkpoint     : {reg_checkpoint or '(none — cat6 skipped)'}")
    print(f"  categories         : {sorted(categories) if categories else 'all (4a,4b,6)'}")
    print(f"  wandb_project      : {args.wandb_project or 'DISABLED'}")
    print(f"  total cells        : {total}")
    print()

    if total == 0:
        msg = "no cells to run for the requested categories"
        if categories and "6" in categories and reg_checkpoint is None:
            msg += " — category 6 needs --reg-checkpoint PATH"
        print(f"ERROR: {msg}", file=sys.stderr)
        return 2

    if reg_checkpoint is None:
        print("NOTE: --reg-checkpoint not provided; category 6 will be skipped.")
        print()

    # ------------------------------------------------------------------ dry-run
    if args.dry_run:
        print("DRY-RUN: cells that would be executed:")
        print()
        for i, cell in enumerate(cells, 1):
            argv_cell = _cell_argv(
                cell, cfg_path, curve, args.loops, args.protection_layers,
                args.wandb_project, args.wandb_entity,
            )
            seed_str = f"seed={cell['seed']}" if cell["seed"] is not None else "seed=<deterministic>"
            print(f"[{i:3d}/{total}] cat={cell['category']}  {seed_str}")
            print(f"         name={cell['exp_name']}")
            print(f"         argv={' '.join(argv_cell)}")
            print()
        return 0

    # ------------------------------------------------------------------ run
    # Import the runner only after the dry-run branch (avoids torch import on
    # dry-run, mirrors the style of the other comparison-DB drivers).
    runner_main = import_runner_main()

    out_dir = new_sweep_out_dir("recalibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_t0 = time.perf_counter()
    results: list[dict] = []

    for i, cell in enumerate(cells, 1):
        argv_cell = _cell_argv(
            cell, cfg_path, curve, args.loops, args.protection_layers,
            args.wandb_project, args.wandb_entity,
        )

        bar = "=" * 72
        print(bar)
        print(
            f"[cell {i}/{total}]  cat={cell['category']}  "
            f"seed={cell['seed']}  name={cell['exp_name']}"
        )
        print(bar)

        t0 = time.perf_counter()
        status, err = run_cell(runner_main, argv_cell)
        elapsed = time.perf_counter() - t0

        if status != "ok":
            print(f"  !! cell failed: {err}")

        # Harvest summary from the runner's output.
        summary_path = latest_summary(runner_out_dir, cell["exp_name"])
        harvested = harvest_summary(summary_path)

        rt_curve_data = harvested["rt_curve"]
        if rt_curve_data:
            curve_str = "  ".join(
                f"{rt:g}:{m['mean']:.2f}"
                for rt, m in sorted(rt_curve_data.items())
            )
            print(
                f"  => {status}  "
                f"clean={harvested['baseline_clean_accuracy']}  "
                f"endlen={harvested['baseline_endlen_accuracy']}  "
                f"endlen_recal={harvested['baseline_endlen_recal_accuracy']}  "
                f"curve(mean): [{curve_str}]  ({elapsed:.1f}s)"
            )
        else:
            print(
                f"  => {status}  "
                f"clean={harvested['baseline_clean_accuracy']}  "
                f"endlen={harvested['baseline_endlen_accuracy']}  "
                f"endlen_recal={harvested['baseline_endlen_recal_accuracy']}  "
                f"({elapsed:.1f}s)"
            )

        record = {
            "category": cell["category"],
            "exp_name": cell["exp_name"],
            "seed": cell["seed"],
            "status": status,
            "error": err,
            "elapsed_s": round(elapsed, 1),
            "summary_path": str(summary_path) if summary_path else None,
            "baseline_clean_accuracy": harvested["baseline_clean_accuracy"],
            "baseline_endlen_accuracy": harvested["baseline_endlen_accuracy"],
            "baseline_endlen_recal_accuracy": harvested["baseline_endlen_recal_accuracy"],
            "weight_encoder": harvested["weight_encoder"],
            "rt_curve": rt_curve_data,
        }
        results.append(record)

        # Persist manifest after every cell so a mid-sweep crash leaves a record.
        write_manifest(out_dir, {
            "config": str(cfg_path),
            "seeds": seeds,
            "rt_curve": curve,
            "loops": args.loops,
            "protection_layers": args.protection_layers,
            "reg_checkpoint": reg_checkpoint,
            "wandb_project": args.wandb_project,
            "results": results,
        })

    # ---------------------------------------------------------------------- tables
    print()
    print("=" * 72)
    print(f"Sweep done. Writing tables to: {out_dir}")
    print("=" * 72)
    _write_tables(out_dir, results, curve, cfg_path)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    total_elapsed = time.perf_counter() - sweep_t0
    print()
    print(f"Total: {n_ok}/{total} cells ok  ({total_elapsed:.1f}s)")
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
