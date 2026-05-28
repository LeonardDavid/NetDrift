#!/usr/bin/env python
"""Consolidate per-budget tables from a sweep into one Markdown file.

Reads ``manifest.json`` (written by ``sweep_local_budget_scope_x_selection.py``)
and emits ``baseline_endlen_tables.md`` next to it — same shape the updated
sweep script writes natively, for sweeps that ran with the older per-budget MD
layout.

Usage::

    python scripts/aggregate_sweep_tables.py runs/sweeps/<ts>_scope_x_sel/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep_dir", help="Directory containing manifest.json")
    p.add_argument("--out", default=None,
                   help="Output path. Default: <sweep_dir>/baseline_endlen_tables.md")
    args = p.parse_args(argv)

    sweep_dir = Path(args.sweep_dir)
    manifest_path = sweep_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    with open(manifest_path) as f:
        m = json.load(f)

    scopes = m["scopes"]
    selections = m["selections"]
    local_budgets = m["local_budgets"]
    rt_error_override = m.get("rt_error_override")
    config = m.get("config", "?")

    # Index results by (local_budget, scope, selection) and capture the
    # baseline_clean_accuracy per local_budget (it's constant across cells, but
    # logged per-cell — take the first non-null we see).
    by_budget: dict[float, dict[tuple[str, str], float | None]] = {
        lb: {} for lb in local_budgets
    }
    clean_by_budget: dict[float, float | None] = {lb: None for lb in local_budgets}
    for r in m["results"]:
        lb = r["local_budget"]
        if lb not in by_budget:
            by_budget[lb] = {}
            clean_by_budget[lb] = None
        by_budget[lb][(r["scope"], r["selection"])] = r.get("baseline_endlen_accuracy")
        if clean_by_budget[lb] is None and r.get("baseline_clean_accuracy") is not None:
            clean_by_budget[lb] = r["baseline_clean_accuracy"]

    lines: list[str] = []
    lines.append("# baseline_endlen_accuracy — scope × selection × local_budget")
    lines.append("")
    lines.append(f"- config: `{Path(config).name}`")
    lines.append(f"- global_bitflip_budget (fixed): **1.0**")
    lines.append(f"- local_bitflip_budget values: {local_budgets}")
    if rt_error_override is not None:
        lines.append(f"- rt_error (override): {rt_error_override}")
    lines.append("")
    lines.append("One table per `local_bitflip_budget` value. Cells are "
                 "`baseline_endlen_accuracy` (%, no faults, after encoding).")
    lines.append("")

    for lb in local_budgets:
        lines.append(f"## local_bitflip_budget = {lb}")
        lines.append("")
        lines.append(f"- baseline_clean_accuracy (no encoder): "
                     f"{_fmt(clean_by_budget.get(lb))}%")
        lines.append("")
        header = ["scope \\ selection"] + selections
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for scope in scopes:
            row = [scope]
            for sel in selections:
                row.append(_fmt(by_budget.get(lb, {}).get((scope, sel))))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    out_path = Path(args.out) if args.out else sweep_dir / "baseline_endlen_tables.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
