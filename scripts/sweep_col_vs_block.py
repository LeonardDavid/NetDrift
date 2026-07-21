#!/usr/bin/env python
"""COL-vs-BLOCK layout comparison under RTM faults (edge_mode study).

Compares the plain **COL** racetrack layout against the **BLOCK** layout (with
both ROW and COL base segmentations) over an rt_error degradation curve, with no
mitigations. Every cell is driven from ONE vgg7 base config and overrides only
the axes under comparison, so the arms differ *only* in the intended variables.

Matrix (single seed, single edge_mode):

* layout  : col | block(base=row) | block(base=col)          — 3
* protect : all (layers 1-8) | custom (layers 2-7)           — 2
  → 3 × 2 = 6 runner invocations ("cells").

Each cell sweeps the full rt_error curve *inside one runner invocation*; the
runner logs one W&B run per rt_error. With 5 rt_error values that is
6 × 5 = 30 W&B runs (= 30 data points), 30 inference iterations each.

⚠️ edge_mode=saturate makes BLOCK fault-immune (same-sign blocks + saturating
reads → every read correct → flat/clean curve at every rt_error). That is a real
property of the model, intended here as the demonstration. COL still degrades.
Read the PER-LOOP degradation curve (logged to W&B), not just the mean over the
loops: in test mode the misalignment offset accumulates across loops, so at the
higher rates non-immune cells collapse toward chance within a handful of loops
and the loop-mean is ~chance — COL and BLOCK separate cleanly in the per-loop
curve and (for the mean) at the low rates (1e-6, 1e-7).

Usage::

    # Dry-run: list all 6 cells with their argv, no execution
    python scripts/sweep_col_vs_block.py --dry-run

    # Execute, logging to a dedicated W&B project
    python scripts/sweep_col_vs_block.py --wandb-project netdrift-col-vs-block

    # Override the curve / edge_mode / seed / loops
    python scripts/sweep_col_vs_block.py \\
        --edge-mode random --loops 100 --seed 707 \\
        --rt-curve 1e-4 4.55e-5 1e-5 1e-6 1e-7
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

# Put the scripts/ dir on sys.path so comparison_common is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparison_common import (  # noqa: E402
    REPO_ROOT,
    base_overrides,
    harvest_summary,
    import_runner_main,
    latest_summary,
    new_sweep_out_dir,
    output_dir_from_cfg,
    run_cell,
    wandb_args,
    write_manifest,
)

# Defaults specific to this study (differ from the comparison-DB canon).
DEFAULT_CONFIG = "configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml"
DEFAULT_RT_CURVE = [1e-4, 4.55e-5, 1e-5, 1e-6, 1e-7]
DEFAULT_LOOPS = 30
DEFAULT_SEED = 707
DEFAULT_WANDB_PROJECT = "netdrift-col-vs-block"

# The three layout arms. base_layout is inert for layout=col (only consumed when
# layout=block), so we leave it out of the col cell entirely.
_LAYOUTS = [
    {"key": "col",       "layout": "col",   "base_layout": None,  "tag": "lay-col"},
    {"key": "block_row", "layout": "block", "base_layout": "row", "tag": "lay-block-base-row"},
    {"key": "block_col", "layout": "block", "base_layout": "col", "tag": "lay-block-base-col"},
]

# The two protection arms. policy=all → every layer on nanowires (1-8 unprotected);
# custom [2..7] → conv1 (id 1) + fc2 (id 8) protected (the common BNN recipe).
_PROTECTIONS = [
    {"key": "prot-all",  "policy": "all",    "layers": None,               "tag": "prot-1to8"},
    {"key": "prot-2to7", "policy": "custom", "layers": [2, 3, 4, 5, 6, 7], "tag": "prot-2to7"},
]


def _build_cells() -> list[dict]:
    """Return the ordered list of cell descriptors (layout × protection)."""
    cells: list[dict] = []
    for lay in _LAYOUTS:
        for prot in _PROTECTIONS:
            cells.append({
                "layout": lay["layout"],
                "base_layout": lay["base_layout"],
                "layout_tag": lay["tag"],
                "policy": prot["policy"],
                "layers": prot["layers"],
                "prot_tag": prot["tag"],
                # subcategory groups every rt_error of this arm together in W&B.
                "subcategory": f"{lay['tag']}_{prot['tag']}",
            })
    return cells


def _cell_argv(
    cell: dict,
    cfg_path: Path,
    curve: list[float],
    loops: int,
    seed: int,
    edge_mode: str,
    base_stem: str,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
) -> list[str]:
    """Assemble the full runner argv for one cell.

    Overrides ONLY the compared axes plus the invariants that must not drift
    (no mitigation, no encoder, edge_mode, seed). Everything else comes from the
    base config so all arms share identical model/data/quant settings.
    """
    argv = ["--config", str(cfg_path)]

    # Shared: rt_error curve, loops, protection policy/layers.
    argv += base_overrides(
        curve=curve,
        loops=loops,
        protection_policy=cell["policy"],
        protection_layers=cell["layers"],
    )

    # Layout axis.
    argv += ["--override", f"storage.layout={cell['layout']}"]
    if cell["base_layout"] is not None:
        argv += ["--override", f"storage.base_layout={cell['base_layout']}"]

    # Invariants: no mitigation, no encoder, fixed edge model + seed. Set
    # explicitly (do NOT trust config defaults — some rtm configs ship endlen).
    argv += [
        "--override", "fault.weight_encoder=null",
        "--override", "fault.mitigations=[]",
        "--override", f"fault.edge_mode={edge_mode}",
        "--override", f"experiment.seed={seed}",
    ]

    # Experiment identity (unique per cell so summaries don't collide).
    exp_name = f"{base_stem}__{cell['subcategory']}"
    cell["exp_name"] = exp_name
    argv += ["--override", f"experiment.name={exp_name}"]

    # W&B: coarse category = the study, fine subcategory = this arm.
    argv += wandb_args(
        wandb_project, wandb_entity,
        "col_vs_block", cell["subcategory"],
    )
    return argv


def _write_tables(out_dir: Path, results: list[dict], curve: list[float]) -> None:
    """Write a flat CSV (one row per cell, mean accuracy per rt_error)."""
    csv_path = out_dir / "col_vs_block_summary.csv"
    rt_cols = [f"rt_{rt:g}_mean" for rt in curve]
    fieldnames = (
        ["subcategory", "layout", "base_layout", "policy", "status",
         "elapsed_s", "baseline_clean_accuracy", "experiment_name"]
        + rt_cols
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            rt_curve = r.get("rt_curve") or {}
            row = {
                "subcategory": r["subcategory"],
                "layout": r["layout"],
                "base_layout": r["base_layout"] if r["base_layout"] else "",
                "policy": r["policy"],
                "status": r["status"],
                "elapsed_s": r.get("elapsed_s", ""),
                "baseline_clean_accuracy": r.get("baseline_clean_accuracy"),
                "experiment_name": r["exp_name"],
            }
            for rt in curve:
                m = rt_curve.get(float(rt))
                row[f"rt_{rt:g}_mean"] = f"{m['mean']:.4f}" if m else ""
            w.writerow(row)
    print(f"  -> {csv_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"Base YAML (single source for all arms). Default: {DEFAULT_CONFIG}")
    p.add_argument("--rt-curve", nargs="+", type=float, default=DEFAULT_RT_CURVE,
                   help="rt_error values swept inside each cell. "
                        f"Default: {DEFAULT_RT_CURVE}")
    p.add_argument("--loops", type=int, default=DEFAULT_LOOPS,
                   help=f"Inference iterations per rt_error. Default: {DEFAULT_LOOPS}")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Single seed for all cells. Default: {DEFAULT_SEED}")
    p.add_argument("--edge-mode", default="saturate", choices=["saturate", "random"],
                   help="Racetrack edge model, applied to EVERY cell so layout is "
                        "the only varied axis. NB saturate → BLOCK is fault-immune. "
                        "Default: saturate")
    p.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT,
                   help=f"W&B project (all runs logged together). "
                        f"Default: {DEFAULT_WANDB_PROJECT}. Pass '' to disable W&B.")
    p.add_argument("--wandb-entity", default=None, help="W&B entity/team.")
    p.add_argument("--dry-run", action="store_true",
                   help="List every cell (with argv) and exit without executing.")
    p.add_argument("--print-commands", action="store_true",
                   help="Emit one standalone `netdrift_run.py ...` shell command per "
                        "cell (no execution). Use to run cells as separate processes / "
                        "across GPUs. Pipe to a file or GNU parallel. See --help notes "
                        "on single-GPU oversubscription.")
    p.add_argument("--collect-only", action="store_true",
                   help="Do NOT run anything: harvest the newest summary.json for each "
                        "of the 6 cells' experiment names and write the aggregated "
                        "col_vs_block_summary.csv. Use after running the cells as "
                        "separate processes (e.g. via --print-commands) to rebuild the "
                        "combined table the sequential driver would have written.")
    args = p.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    curve = [float(x) for x in args.rt_curve]
    base_stem = cfg_path.stem
    wandb_project = args.wandb_project or None
    cells = _build_cells()
    total = len(cells)
    n_wandb_runs = total * len(curve)

    # ---------------------------------------------------------------- header
    print("=" * 72)
    print("COL-vs-BLOCK layout comparison (edge_mode study)")
    print("=" * 72)
    print(f"  base config        : {cfg_path}")
    print(f"  layouts            : col, block(base=row), block(base=col)")
    print(f"  protection arms    : all (1-8), custom (2-7)")
    print(f"  rt_error curve     : {curve}")
    print(f"  loops              : {args.loops}")
    print(f"  seed               : {args.seed}")
    print(f"  edge_mode          : {args.edge_mode}"
          + ("  (BLOCK fault-immune!)" if args.edge_mode == "saturate" else ""))
    print(f"  mitigations        : none (weight_encoder=null, mitigations=[])")
    print(f"  wandb_project      : {wandb_project or 'DISABLED'}")
    print(f"  runner invocations : {total}  (3 layouts × 2 protection)")
    print(f"  W&B runs           : {n_wandb_runs}  ({total} cells × {len(curve)} rt_error)")
    print(f"  inference passes   : {n_wandb_runs * args.loops}  "
          f"({n_wandb_runs} runs × {args.loops} loops)")
    print()

    # ------------------------------------------------------ collect-only
    # Rebuild the aggregated CSV from summaries written by cells that were run as
    # separate processes (e.g. via --print-commands). No runner import, no GPU.
    if args.collect_only:
        runner_out_dir = output_dir_from_cfg(cfg_path)
        if not runner_out_dir.is_absolute():
            runner_out_dir = REPO_ROOT / runner_out_dir
        results: list[dict] = []
        for cell in cells:
            # Reconstruct the deterministic per-cell experiment name (matches
            # _cell_argv; no argv assembly needed).
            cell["exp_name"] = f"{base_stem}__{cell['subcategory']}"
            summary_path = latest_summary(runner_out_dir, cell["exp_name"])
            harvested = harvest_summary(summary_path)
            results.append({
                "subcategory": cell["subcategory"],
                "layout": cell["layout"],
                "base_layout": cell["base_layout"],
                "policy": cell["policy"],
                "exp_name": cell["exp_name"],
                "status": "ok" if summary_path else "missing",
                "elapsed_s": "",
                "baseline_clean_accuracy": harvested["baseline_clean_accuracy"],
                "rt_curve": harvested["rt_curve"],
            })
            found = "found" if summary_path else "MISSING summary.json"
            print(f"  {cell['subcategory']:36s} -> {found}")
        out_dir = new_sweep_out_dir("col_vs_block_collected")
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_tables(out_dir, results, curve)
        n_found = sum(1 for r in results if r["status"] == "ok")
        print(f"\nCollected {n_found}/{len(cells)} cells.")
        return 0 if n_found == len(cells) else 1

    # ------------------------------------------------------ print-commands
    # Emit each cell as a standalone `netdrift_run.py` invocation so cells can be
    # run as separate PROCESSES (the only safe way to parallelize — the in-process
    # loop shares one CUDA context; threads would collide on Numba's context).
    # On a SINGLE GPU these still contend for one device: prefer sequential, or
    # pin each to its own GPU with CUDA_VISIBLE_DEVICES when several are present.
    if args.print_commands:
        runner = REPO_ROOT / "netdrift_run.py"
        for cell in cells:
            argv_cell = _cell_argv(
                cell, cfg_path, curve, args.loops, args.seed, args.edge_mode,
                base_stem, wandb_project, args.wandb_entity,
            )
            print(f"python {runner} " + " ".join(argv_cell))
        return 0

    # ---------------------------------------------------------------- dry-run
    if args.dry_run:
        print("DRY-RUN: cells that would be executed:")
        print()
        for i, cell in enumerate(cells, 1):
            argv_cell = _cell_argv(
                cell, cfg_path, curve, args.loops, args.seed, args.edge_mode,
                base_stem, wandb_project, args.wandb_entity,
            )
            print(f"[{i}/{total}] {cell['subcategory']}")
            print(f"        layout={cell['layout']}"
                  + (f" base_layout={cell['base_layout']}" if cell['base_layout'] else "")
                  + f"  policy={cell['policy']}"
                  + (f" layers={cell['layers']}" if cell['layers'] else ""))
            print(f"        argv={' '.join(argv_cell)}")
            print()
        return 0

    # ---------------------------------------------------------------- run
    runner_main = import_runner_main()
    runner_out_dir = output_dir_from_cfg(cfg_path)
    if not runner_out_dir.is_absolute():
        runner_out_dir = REPO_ROOT / runner_out_dir

    out_dir = new_sweep_out_dir("col_vs_block")
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_t0 = time.perf_counter()
    results: list[dict] = []

    for i, cell in enumerate(cells, 1):
        argv_cell = _cell_argv(
            cell, cfg_path, curve, args.loops, args.seed, args.edge_mode,
            base_stem, wandb_project, args.wandb_entity,
        )
        bar = "=" * 72
        print(bar)
        print(f"[cell {i}/{total}]  {cell['subcategory']}  (edge_mode={args.edge_mode})")
        print(bar)

        t0 = time.perf_counter()
        status, err = run_cell(runner_main, argv_cell)
        elapsed = time.perf_counter() - t0
        if status != "ok":
            print(f"  !! cell failed: {err}")

        summary_path = latest_summary(runner_out_dir, cell["exp_name"])
        harvested = harvest_summary(summary_path)
        rt_curve_data = harvested["rt_curve"]
        if rt_curve_data:
            curve_str = "  ".join(
                f"{rt:g}:{m['mean']:.2f}" for rt, m in sorted(rt_curve_data.items())
            )
            print(f"  => {status}  clean={harvested['baseline_clean_accuracy']}  "
                  f"curve(mean): [{curve_str}]  ({elapsed:.1f}s)")
        else:
            print(f"  => {status}  clean={harvested['baseline_clean_accuracy']}  ({elapsed:.1f}s)")

        results.append({
            "subcategory": cell["subcategory"],
            "layout": cell["layout"],
            "base_layout": cell["base_layout"],
            "policy": cell["policy"],
            "exp_name": cell["exp_name"],
            "status": status,
            "error": err,
            "elapsed_s": round(elapsed, 1),
            "summary_path": str(summary_path) if summary_path else None,
            "baseline_clean_accuracy": harvested["baseline_clean_accuracy"],
            "rt_curve": rt_curve_data,
        })
        write_manifest(out_dir, {
            "study": "col_vs_block",
            "config": str(cfg_path),
            "seed": args.seed,
            "edge_mode": args.edge_mode,
            "rt_curve": curve,
            "loops": args.loops,
            "wandb_project": wandb_project,
            "results": results,
        })

    print()
    print("=" * 72)
    print(f"Sweep done. Writing table to: {out_dir}")
    print("=" * 72)
    _write_tables(out_dir, results, curve)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print()
    print(f"Total: {n_ok}/{total} cells ok  ({time.perf_counter() - sweep_t0:.1f}s)")
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
