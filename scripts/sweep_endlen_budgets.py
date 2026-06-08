#!/usr/bin/env python
"""Sweep endlen bitflip-budget parameters and log every run to W&B.

Drives the NetDrift runner across the full budget matrix

    local_budget_scope  ∈ {layer, racetrack, channel}
    budget_selection    ∈ {greedy, value_per_flip, magnitude_aware}
    global_bitflip_budget ∈ BUDGETS
    local_bitflip_budget  ∈ BUDGETS

for one or more base configs, **de-duplicating provably-identical cells**:

* When ``local == 1.0`` the local cap never binds, so ``local_budget_scope``
  has no effect — all three scopes collapse to one canonical cell.
* When ``global == 1.0 and local == 1.0`` nothing is capped, so
  ``budget_selection`` has no effect either — the three selections collapse to
  one (the unbudgeted reference).

Each surviving cell is a separate runner invocation with ``--wandb-project`` set
and a per-cell ``experiment.name`` so the runs group cleanly in the W&B UI. The
runner itself already opens one W&B run per ``rt_error`` (see the wandb-tracking
work) and logs baselines + the encoder flip/reject report, so the budget
parameters land in each run's W&B config automatically.

Usage::

    python scripts/sweep_endlen_budgets.py \
        --config configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml \
        --wandb-project endlen-budget-sweep

    # preview the cells without running anything
    python scripts/sweep_endlen_budgets.py --config <cfg> --dry-run

The runner's own ``rt_error`` sweep (from the YAML) runs inside every cell; pass
``--rt-error 1e-4`` to pin a single value for a faster exploratory pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "code" / "python"

# Shared harvest helper so the rt_error curve is parsed the same way across all
# comparison-DB drivers + the aggregator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparison_common import harvest_summary, latest_summary, output_dir_from_cfg  # noqa: E402

# --- sweep axes -------------------------------------------------------------
SCOPES = ["layer", "racetrack", "channel"]
SELECTIONS = ["greedy", "value_per_flip", "magnitude_aware"]
BUDGETS = [0.05, 0.1, 0.25, 0.5, 1.0]


def _fmt_budget(b: float) -> str:
    """Compact, filename-safe budget tag, e.g. 0.05 -> '0p05', 1.0 -> '1p0'."""
    return str(b).replace(".", "p")


def build_cells() -> list[dict]:
    """Return the de-duplicated list of sweep cells.

    Each cell is a dict with the four budget params plus a ``name_tag`` used to
    build the per-cell ``experiment.name``. Canonicalization collapses inert
    axes so no two cells produce identical results.
    """
    seen: set[tuple] = set()
    cells: list[dict] = []
    for scope in SCOPES:
        for selection in SELECTIONS:
            for gb in BUDGETS:
                for lb in BUDGETS:
                    # Canonical key: drop axes that cannot affect the result.
                    canon_scope = scope if lb < 1.0 else "-"          # scope inert when local unbounded
                    canon_sel = selection if (gb < 1.0 or lb < 1.0) else "-"  # selection inert when nothing capped
                    key = (canon_scope, canon_sel, gb, lb)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Build a readable tag from the CANONICAL params so the
                    # W&B group name reflects what actually varied.
                    parts = [f"gl{_fmt_budget(gb)}", f"lo{_fmt_budget(lb)}"]
                    if canon_scope != "-":
                        parts.append(f"sc-{canon_scope}")
                    if canon_sel != "-":
                        parts.append(f"sel-{canon_sel}")
                    # Unbudgeted (global=local=1.0) IS vanilla endlen (cat 2);
                    # everything else is a budgeted-endlen grid cell (cat 3).
                    is_vanilla = (gb >= 1.0 and lb >= 1.0)
                    cells.append({
                        "scope": scope,
                        "selection": selection,
                        "global_budget": gb,
                        "local_budget": lb,
                        "name_tag": "_".join(parts),
                        "category": (
                            "cat2_vanilla_endlen" if is_vanilla
                            else "cat3_budgeted_endlen"
                        ),
                    })
    return cells


def _overrides_for(
    cell: dict,
    base_name: str,
    rt_error: str | None,
    rt_curve: list[float] | None,
    loops: int | None,
) -> list[str]:
    ov = [
        f"fault.global_bitflip_budget={cell['global_budget']}",
        f"fault.local_bitflip_budget={cell['local_budget']}",
        f"fault.local_budget_scope={cell['scope']}",
        f"fault.budget_selection={cell['selection']}",
        f"experiment.name={base_name}__{cell['name_tag']}",
    ]
    # rt-curve (list) takes precedence over a single pinned rt-error.
    if rt_curve is not None:
        ov.append(
            "fault.rt_error=[" + ",".join(repr(float(x)) for x in rt_curve) + "]"
        )
    elif rt_error is not None:
        ov.append(f"fault.rt_error={rt_error}")
    if loops is not None:
        ov.append(f"training.loops={loops}")
    return ov


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Endlen bitflip-budget sweep")
    p.add_argument("--config", required=True, action="append",
                   help="Base YAML config (repeatable for multiple base configs)")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project. Omit for a local-only sweep (no W&B).")
    p.add_argument("--wandb-entity", default=None, help="Optional W&B entity.")
    p.add_argument("--rt-error", default=None,
                   help="Pin a single rt_error per cell (e.g. 1e-4). "
                        "Default: use the config's rt_error sweep.")
    p.add_argument("--rt-curve", nargs="+", type=float, default=None,
                   help="Evaluate each cell over this rt_error CURVE (e.g. "
                        "--rt-curve 1e-7 3e-7 1e-6 3e-6 1e-5). Harvests the "
                        "per-rt_error fault-sweep mean/min/max into the manifest "
                        "for the comparison-DB aggregator. Overrides --rt-error.")
    p.add_argument("--loops", type=int, default=None,
                   help="Override training.loops (inference iterations per "
                        "rt_error). Use 10 for the robustness curve.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the cells + overrides and exit; run nothing.")
    p.add_argument("--manifest", default=None,
                   help="Path to write the sweep manifest JSON. "
                        "Default: runs/sweeps/<timestamp>_manifest.json")
    args = p.parse_args(argv)

    cells = build_cells()
    base_configs = args.config
    total = len(cells) * len(base_configs)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest_path = Path(args.manifest) if args.manifest else (
        REPO_ROOT / "runs" / "sweeps" / f"{ts}_manifest.json"
    )

    plan = []
    for cfg in base_configs:
        base_name = Path(cfg).stem
        for cell in cells:
            plan.append({
                "config": cfg,
                "overrides": _overrides_for(
                    cell, base_name, args.rt_error, args.rt_curve, args.loops
                ),
                "cell": cell,
            })

    print(f"Sweep: {len(cells)} de-duplicated cells × {len(base_configs)} "
          f"config(s) = {total} runs")
    print(f"  budgets   : {BUDGETS}")
    print(f"  scopes    : {SCOPES}")
    print(f"  selections: {SELECTIONS}")
    print(f"  rt_error  : {args.rt_error or 'config sweep'}")
    print(f"  wandb     : {args.wandb_project or 'DISABLED'}")
    print(f"  manifest  : {manifest_path}")
    print()

    if args.dry_run:
        for i, item in enumerate(plan, 1):
            print(f"[{i:3d}/{total}] {Path(item['config']).stem}  "
                  f"{' '.join(item['overrides'])}")
        return 0

    # Import the runner once; each cell calls it with a fresh argv. The runner
    # does not cache model/data between calls (it rebuilds per invocation), but
    # one Python process avoids paying torch/numba import cost N times.
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from netdrift.runner.run import main as runner_main

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    sweep_t0 = time.perf_counter()
    for i, item in enumerate(plan, 1):
        argv_cell = ["--config", item["config"]]
        for ov in item["overrides"]:
            argv_cell += ["--override", ov]
        if args.wandb_project:
            argv_cell += ["--wandb-project", args.wandb_project]
            if args.wandb_entity:
                argv_cell += ["--wandb-entity", args.wandb_entity]
            cat = item["cell"]["category"]
            argv_cell += ["--wandb-category", cat]
            # Subcategory = exact setting combo (the cell's name_tag).
            argv_cell += [
                "--wandb-subcategory", f"{cat}_{item['cell']['name_tag']}"
            ]

        bar = "=" * 72
        print(bar)
        print(f"[cell {i}/{total}] {Path(item['config']).stem}  cell={item['cell']['name_tag']}")
        print(bar)
        t0 = time.perf_counter()
        status, error = "ok", None
        try:
            rc = runner_main(argv_cell)
            if rc != 0:
                status, error = "nonzero_exit", f"runner returned {rc}"
        except Exception as exc:  # one bad cell must not kill the sweep
            status, error = "error", repr(exc)
            print(f"  !! cell failed: {error}")
        elapsed = time.perf_counter() - t0
        # Harvest baselines + (when sweeping a curve) per-rt_error fault metrics.
        runner_out_dir = output_dir_from_cfg(Path(item["config"]).resolve())
        exp_name = f"{Path(item['config']).stem}__{item['cell']['name_tag']}"
        harvested = harvest_summary(latest_summary(runner_out_dir, exp_name))
        if harvested["rt_curve"]:
            curve_str = "  ".join(
                f"{rt:g}:{m['mean']:.1f}"
                for rt, m in sorted(harvested["rt_curve"].items())
            )
            print(f"  ⇒ cell {i}/{total} {status}  curve(mean) [{curve_str}]  ({elapsed:.1f}s)")
        else:
            print(f"  ⇒ cell {i}/{total} {status}  ({elapsed:.1f}s)")
        results.append({
            "index": i, "config": item["config"], "cell": item["cell"],
            "overrides": item["overrides"], "status": status,
            "error": error, "elapsed_s": round(elapsed, 1),
            "baseline_clean_accuracy": harvested["baseline_clean_accuracy"],
            "baseline_endlen_accuracy": harvested["baseline_endlen_accuracy"],
            "rt_curve": harvested["rt_curve"],
        })
        # Persist the manifest after every cell so a mid-sweep crash still
        # leaves a record of what ran.
        with open(manifest_path, "w") as f:
            json.dump({
                "timestamp": ts,
                "wandb_project": args.wandb_project,
                "rt_error_override": args.rt_error,
                "budgets": BUDGETS, "scopes": SCOPES, "selections": SELECTIONS,
                "total": total, "results": results,
            }, f, indent=2)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = total - n_ok
    print()
    print(f"Sweep done: {n_ok}/{total} ok, {n_fail} failed  "
          f"({time.perf_counter() - sweep_t0:.1f}s total)")
    print(f"Manifest: {manifest_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
