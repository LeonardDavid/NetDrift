#!/usr/bin/env python
"""Sweep local_budget_scope × budget_selection × local_bitflip_budget.

Holds ``global_bitflip_budget=1.0`` (so the local cap is the only one that
binds), iterates over every (scope, selection) pair and every value in
``--local-budgets``, runs the NetDrift runner per cell, then exports one CSV +
one Markdown table **per local-budget value** filled with the resulting
``baseline_endlen_accuracy`` for each (scope, selection) cell.

The runner already writes ``runs/<experiment.name>/<timestamp>/summary.json``,
which carries ``baseline_endlen_accuracy``. We override ``experiment.name`` per
cell to a unique tag so the latest-timestamp dir under that name is
unambiguously the cell's output.

Usage::

    python scripts/sweep_local_budget_scope_x_selection.py \
        --config configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml \
        --local-budgets 0.05 0.1 0.25 0.5

    # add W&B + dry-run
    python scripts/sweep_local_budget_scope_x_selection.py \
        --config <cfg> --local-budgets 0.1 0.25 --wandb-project myproj
    python scripts/sweep_local_budget_scope_x_selection.py \
        --config <cfg> --local-budgets 0.1 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "code" / "python"

# Shared harvest helper (sibling module) so curve parsing matches the other
# comparison-DB drivers + the aggregator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparison_common import harvest_summary  # noqa: E402

SCOPES = ["layer", "racetrack", "channel"]
SELECTIONS = ["greedy", "value_per_flip", "magnitude_aware"]


def _fmt_budget(b: float) -> str:
    """Filename-safe budget tag, e.g. 0.05 -> '0p05', 1.0 -> '1p0'."""
    return str(b).replace(".", "p")


def _experiment_name(base: str, lb: float, scope: str, selection: str) -> str:
    return f"{base}__lo{_fmt_budget(lb)}_sc-{scope}_sel-{selection}"


def _output_dir_from_cfg(cfg_path: Path) -> Path:
    """Best-effort read of ``experiment.output_dir`` from a YAML.

    Uses ``yaml`` when available; otherwise falls back to a minimal regex scan
    of the top-level ``experiment:`` block so the script's dry-run still works
    in environments without PyYAML installed.
    """
    try:
        import yaml  # type: ignore[import-not-found]
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
        return Path(raw.get("experiment", {}).get("output_dir", "runs/"))
    except ModuleNotFoundError:
        import re
        text = cfg_path.read_text()
        m = re.search(r"^experiment:\s*\n((?:[ \t].*\n)+)", text, re.MULTILINE)
        if m:
            block = m.group(1)
            m2 = re.search(r"^\s+output_dir:\s*(\S+)", block, re.MULTILINE)
            if m2:
                return Path(m2.group(1).strip('"\''))
        return Path("runs/")


def _latest_summary(out_dir: Path, exp_name: str) -> Path | None:
    """Return the newest ``summary.json`` under ``<out_dir>/<exp_name>/`` or None."""
    base = out_dir / exp_name
    if not base.exists():
        return None
    candidates = sorted(base.glob("*/summary.json"))
    return candidates[-1] if candidates else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="Base YAML config (passed to netdrift_run.py).")
    p.add_argument("--local-budgets", nargs="+", type=float, required=True,
                   help="Local-budget values to sweep, e.g. 0.05 0.1 0.25 0.5")
    p.add_argument("--wandb-project", default=None,
                   help="Optional W&B project. Omit for a local-only sweep.")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--rt-error", default=None,
                   help="Pin a single rt_error per cell (overrides config sweep). "
                        "Mutually exclusive with --rt-curve.")
    p.add_argument("--rt-curve", nargs="+", type=float, default=None,
                   help="Evaluate each cell over this rt_error CURVE (list of "
                        "floats), e.g. --rt-curve 1e-7 3e-7 1e-6 3e-6 1e-5. "
                        "Records per-rt_error mean/min/max/last from the fault "
                        "sweep, not just the clean baseline_endlen_accuracy. "
                        "Use this for the comparison-DB robustness tables.")
    p.add_argument("--loops", type=int, default=None,
                   help="Override training.loops (inference iterations per "
                        "rt_error). Recommended 10 for the robustness curve.")
    p.add_argument("--out-dir", default=None,
                   help="Where to write tables. Default: runs/sweeps/<ts>_scope_x_sel/")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the cells + the table layout and exit.")
    args = p.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 2

    base_name = cfg_path.stem
    runner_out_dir = _output_dir_from_cfg(cfg_path)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "runs" / "sweeps" / f"{ts}_scope_x_sel"
    )

    # Build the cell list (each (scope, selection) × each local_budget).
    cells = [
        {
            "local_budget": lb,
            "scope": scope,
            "selection": selection,
            "experiment_name": _experiment_name(base_name, lb, scope, selection),
        }
        for lb in args.local_budgets
        for scope in SCOPES
        for selection in SELECTIONS
    ]
    total = len(cells)

    print(f"Sweep: {total} cells "
          f"({len(SCOPES)} scopes × {len(SELECTIONS)} selections × "
          f"{len(args.local_budgets)} local_budgets)")
    print(f"  global_bitflip_budget : 1.0 (fixed)")
    print(f"  local_bitflip_budget  : {args.local_budgets}")
    print(f"  config                : {cfg_path}")
    print(f"  wandb                 : {args.wandb_project or 'DISABLED'}")
    print(f"  output (tables)       : {out_dir}")
    print(f"  runner output_dir     : {runner_out_dir}")
    print()

    if args.dry_run:
        for i, c in enumerate(cells, 1):
            print(f"[{i:3d}/{total}] lb={c['local_budget']} "
                  f"scope={c['scope']:9s} sel={c['selection']:17s} "
                  f"name={c['experiment_name']}")
        print()
        print(f"Would write one table per local_budget under {out_dir}/")
        return 0

    # Import the runner once; argv is built per cell.
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from netdrift.runner.run import main as runner_main

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_t0 = time.perf_counter()
    results: list[dict] = []

    for i, cell in enumerate(cells, 1):
        argv_cell = [
            "--config", str(cfg_path),
            "--override", "fault.global_bitflip_budget=1.0",
            "--override", f"fault.local_bitflip_budget={cell['local_budget']}",
            "--override", f"fault.local_budget_scope={cell['scope']}",
            "--override", f"fault.budget_selection={cell['selection']}",
            "--override", f"experiment.name={cell['experiment_name']}",
        ]
        if args.rt_curve is not None:
            curve_str = "[" + ",".join(repr(float(x)) for x in args.rt_curve) + "]"
            argv_cell += ["--override", f"fault.rt_error={curve_str}"]
        elif args.rt_error is not None:
            argv_cell += ["--override", f"fault.rt_error={args.rt_error}"]
        if args.loops is not None:
            argv_cell += ["--override", f"training.loops={args.loops}"]
        if args.wandb_project:
            argv_cell += ["--wandb-project", args.wandb_project]
            if args.wandb_entity:
                argv_cell += ["--wandb-entity", args.wandb_entity]

        bar = "=" * 72
        print(bar)
        print(f"[cell {i}/{total}] lb={cell['local_budget']} "
              f"scope={cell['scope']} sel={cell['selection']}")
        print(bar)
        t0 = time.perf_counter()
        status, err = "ok", None
        try:
            rc = runner_main(argv_cell)
            if rc != 0:
                status, err = "nonzero_exit", f"runner returned {rc}"
        except Exception as exc:
            status, err = "error", repr(exc)
            print(f"  !! cell failed: {err}")
        elapsed = time.perf_counter() - t0

        # Pull baselines + (when sweeping a curve) the per-rt_error fault metrics
        # from the runner's summary.json, via the shared harvest helper.
        summary_path = _latest_summary(runner_out_dir, cell["experiment_name"])
        harvested = harvest_summary(summary_path)
        baseline_endlen = harvested["baseline_endlen_accuracy"]
        baseline_clean = harvested["baseline_clean_accuracy"]
        rt_curve = harvested["rt_curve"]

        if rt_curve:
            curve_str = "  ".join(
                f"{rt:g}:{m['mean']:.1f}" for rt, m in sorted(rt_curve.items())
            )
            print(f"  ⇒ {status}  endlen_clean={baseline_endlen}  "
                  f"curve(mean) [{curve_str}]  ({elapsed:.1f}s)")
        else:
            print(f"  ⇒ {status}  baseline_endlen_accuracy={baseline_endlen}  "
                  f"({elapsed:.1f}s)")
        results.append({
            **cell,
            "status": status,
            "error": err,
            "elapsed_s": round(elapsed, 1),
            "summary_path": str(summary_path) if summary_path else None,
            "baseline_endlen_accuracy": baseline_endlen,
            "baseline_clean_accuracy": baseline_clean,
            "rt_curve": rt_curve,
        })

        # Persist a manifest after every cell so a mid-sweep crash leaves a record.
        with open(out_dir / "manifest.json", "w") as f:
            json.dump({
                "timestamp": ts,
                "config": str(cfg_path),
                "local_budgets": args.local_budgets,
                "scopes": SCOPES,
                "selections": SELECTIONS,
                "wandb_project": args.wandb_project,
                "rt_error_override": args.rt_error,
                "results": results,
            }, f, indent=2)

    # ------------------------------------------------------------------------
    # Build one table per local_budget: rows=scopes, cols=selections, values=
    # baseline_endlen_accuracy. Written as both CSV and Markdown for easy
    # eyeballing in the terminal/IDE.
    # ------------------------------------------------------------------------
    by_budget: dict[float, dict[tuple[str, str], float | None]] = {
        lb: {} for lb in args.local_budgets
    }
    clean_by_budget: dict[float, float | None] = {lb: None for lb in args.local_budgets}
    for r in results:
        by_budget[r["local_budget"]][(r["scope"], r["selection"])] = (
            r["baseline_endlen_accuracy"]
        )
        # baseline_clean_accuracy is the same for all cells of a given config
        # (no encoding) but we record per local_budget for completeness.
        if clean_by_budget[r["local_budget"]] is None:
            clean_by_budget[r["local_budget"]] = r["baseline_clean_accuracy"]

    def _fmt(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

    # Per-budget CSVs (one per local_budget value) for spreadsheet/pandas work,
    # plus a SINGLE consolidated Markdown file with every table.
    md_path = out_dir / "baseline_endlen_tables.md"
    md_lines: list[str] = []
    md_lines.append("# baseline_endlen_accuracy — scope × selection × local_budget")
    md_lines.append("")
    md_lines.append(f"- config: `{cfg_path.name}`")
    md_lines.append(f"- global_bitflip_budget (fixed): **1.0**")
    md_lines.append(f"- local_bitflip_budget values: {args.local_budgets}")
    if args.rt_error is not None:
        md_lines.append(f"- rt_error (override): {args.rt_error}")
    md_lines.append("")
    md_lines.append(f"One table per `local_bitflip_budget` value. Cells are "
                    f"`baseline_endlen_accuracy` (%, no faults, after encoding).")
    md_lines.append("")

    print()
    print("=" * 72)
    print(f"Tables written to: {out_dir}")
    print("=" * 72)
    for lb in args.local_budgets:
        tag = _fmt_budget(lb)
        csv_path = out_dir / f"baseline_endlen__lb{tag}.csv"

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scope \\ selection"] + SELECTIONS)
            for scope in SCOPES:
                row = [scope]
                for sel in SELECTIONS:
                    v = by_budget[lb].get((scope, sel))
                    row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
                w.writerow(row)

        clean = clean_by_budget[lb]
        section: list[str] = []
        section.append(f"## local_bitflip_budget = {lb}")
        section.append("")
        section.append(f"- baseline_clean_accuracy (no encoder): {_fmt(clean)}%")
        section.append("")
        header = ["scope \\ selection"] + SELECTIONS
        section.append("| " + " | ".join(header) + " |")
        section.append("|" + "|".join(["---"] * len(header)) + "|")
        for scope in SCOPES:
            cells_row = [scope]
            for sel in SELECTIONS:
                cells_row.append(_fmt(by_budget[lb].get((scope, sel))))
            section.append("| " + " | ".join(cells_row) + " |")
        section.append("")

        md_lines.extend(section)
        # Stdout for immediate feedback.
        print()
        print("\n".join(section))
        print(f"  → {csv_path.name}")

    md_path.write_text("\n".join(md_lines))
    print()
    print(f"  → {md_path.name}  (all tables consolidated)")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print()
    print(f"Sweep done: {n_ok}/{total} ok  "
          f"({time.perf_counter() - sweep_t0:.1f}s total)")
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
