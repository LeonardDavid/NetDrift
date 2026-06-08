#!/usr/bin/env python
"""Harvest every comparison-DB run's summary.json into one master table (CSV + Markdown).

Discovers runs under ``--runs-dir`` (default ``runs/``), groups seed variants
together (mean±std), and emits a wide table with one column per rt_error point.

CSV layout: separate ``<col>_mean`` and ``<col>_std`` columns for every numeric
field (including n=1 rows, where std=0) so the table is pandas-friendly without
post-parsing string splitting.

Usage examples::

    # All runs, default rt_error columns, mean statistic:
    python scripts/aggregate_comparison_db.py

    # Only cat3 / cat4 / knee_ runs, min statistic:
    python scripts/aggregate_comparison_db.py --include cat3 cat4 knee_ --metric min

    # Dry-run: see what would be harvested:
    python scripts/aggregate_comparison_db.py --dry-run

    # Custom rt_error columns (space-separated floats):
    python scripts/aggregate_comparison_db.py --rt-errors 1e-7 1e-6 1e-5
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Bootstrap: make `scripts/` importable so comparison_common resolves cleanly
# even when the script is invoked from a different working directory.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from comparison_common import (  # noqa: E402 – after sys.path patch
    DEFAULT_RT_ERROR_CURVE,
    REPO_ROOT,
    harvest_summary,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SEED_RE = re.compile(r"_seed\d+")
_DASH = "—"  # U+2014 em-dash, used for missing cells


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _newest_summary(exp_dir: Path) -> Optional[Path]:
    """Return the newest summary.json under ``<exp_dir>/*/summary.json``.

    Sorted lexicographically by the timestamp dir name — iso-yyyymmdd-HHMMSS
    format makes this equivalent to chronological sort without touching mtime
    (mtime is unreliable on a SSHFS/NFS mount).
    """
    candidates = sorted(exp_dir.glob("*/summary.json"))
    return candidates[-1] if candidates else None


def discover_runs(
    runs_dir: Path,
    include: Optional[list[str]],
) -> list[tuple[str, Path]]:
    """Walk ``runs_dir/*/`` and return ``[(exp_name, newest_summary_path)]``.

    * Skips any subdirectory that has no ``*/summary.json`` (e.g. ``sweeps/``).
    * If ``include`` is given, only keeps dirs whose name contains at least one
      of the filter substrings (case-sensitive).
    """
    results: list[tuple[str, Path]] = []
    if not runs_dir.is_dir():
        return results

    for exp_dir in sorted(runs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary = _newest_summary(exp_dir)
        if summary is None:
            continue  # no summary.json here (e.g. sweeps/, or empty dir)
        name = exp_dir.name
        if include:
            if not any(f in name for f in include):
                continue
        results.append((name, summary))

    return results


# ---------------------------------------------------------------------------
# Seed grouping
# ---------------------------------------------------------------------------

def _group_key(name: str) -> str:
    """Strip all ``_seed<N>`` tokens from ``name`` to form the group key.

    Handles seed tokens in any position (mid-name, trailing, etc.).
    Example: ``foo_seed1_test`` → ``foo_test``, ``bar_seed42`` → ``bar``.
    """
    return _SEED_RE.sub("", name)


def group_by_seed(
    runs: list[tuple[str, Path]],
) -> dict[str, list[tuple[str, Path]]]:
    """Return ``{group_key: [(exp_name, summary_path), ...]}`` preserving insertion order."""
    groups: dict[str, list[tuple[str, Path]]] = {}
    for name, path in runs:
        key = _group_key(name)
        groups.setdefault(key, []).append((name, path))
    return groups


# ---------------------------------------------------------------------------
# Numeric helpers (no numpy)
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def _pstd(vals: list[float]) -> float:
    """Population std-dev (pstdev).  pstdev([x]) == 0; pstdev([]) handled
    before call."""
    return statistics.pstdev(vals)


def _fmt2(v: Optional[float]) -> str:
    """Format a float to 2 decimal places, or em-dash if None."""
    if v is None:
        return _DASH
    return f"{v:.2f}"


def _fmt_mean_std(mean: Optional[float], std: float) -> str:
    """Render ``mean±std`` for Markdown (2 dp each)."""
    if mean is None:
        return _DASH
    return f"{mean:.2f}±{std:.2f}"


# ---------------------------------------------------------------------------
# Harvest + aggregate one group
# ---------------------------------------------------------------------------

def aggregate_group(
    group_members: list[tuple[str, Path]],
    rt_errors: list[float],
    metric: str,
) -> dict[str, Any]:
    """Harvest all members of a seed group and return an aggregated row dict.

    Scalar baseline columns (clean / endlen / endlen_recal) and per-rt_error
    metric values are averaged over seeds; std is population std-dev.

    Returns a dict with keys:
        ``n``, ``experiments`` (list of names),
        ``clean_mean``, ``clean_std``,
        ``endlen_mean``, ``endlen_std``,
        ``endlen_recal_mean``, ``endlen_recal_std``,
        ``encoder``,
        ``<rt_col>_mean``, ``<rt_col>_std`` for each rt_error.
    """
    # Harvest all members
    harvested: list[dict[str, Any]] = []
    for _name, summary_path in group_members:
        harvested.append(harvest_summary(summary_path))

    n = len(harvested)

    # --- Scalar baselines ---
    def _collect_scalar(key: str) -> list[float]:
        return [h[key] for h in harvested if h.get(key) is not None]

    clean_vals = _collect_scalar("baseline_clean_accuracy")
    endlen_vals = _collect_scalar("baseline_endlen_accuracy")
    recal_vals = _collect_scalar("baseline_endlen_recal_accuracy")

    # --- Encoder: take the first non-None value; warn if inconsistent ---
    encoder_vals = [h["weight_encoder"] for h in harvested if h.get("weight_encoder") is not None]
    encoder = encoder_vals[0] if encoder_vals else None

    # --- Per-rt_error columns ---
    # Use exact float equality: harvest_summary calls float(rt) on the JSON
    # value, and argparse type=float on --rt-errors, so same literals parse to
    # identical IEEE-754 doubles.  Tolerance fallback catches hand-typed edge
    # cases (e.g. 1e-7 vs 1e-07 are bit-identical anyway).
    rt_data: dict[float, dict[str, Any]] = {}
    for rt in rt_errors:
        vals: list[float] = []
        for h in harvested:
            curve = h.get("rt_curve", {})
            # Exact lookup first
            cell = curve.get(rt)
            if cell is None:
                # Tolerance fallback (handles 1e-7 vs 1e-07 differences if any)
                for curve_rt, curve_cell in curve.items():
                    if abs(curve_rt - rt) <= 1e-15 * max(abs(curve_rt), abs(rt), 1.0):
                        cell = curve_cell
                        break
            if cell is not None:
                v = cell.get(metric)
                if v is not None:
                    vals.append(float(v))

        rt_col = _rt_col_name(rt)
        rt_data[rt_col] = {
            "mean": _mean(vals),
            "std": _pstd(vals) if vals else 0.0,
        }

    return {
        "n": n,
        "experiments": [name for name, _ in group_members],
        "clean_mean": _mean(clean_vals),
        "clean_std": _pstd(clean_vals) if clean_vals else 0.0,
        "endlen_mean": _mean(endlen_vals),
        "endlen_std": _pstd(endlen_vals) if endlen_vals else 0.0,
        "endlen_recal_mean": _mean(recal_vals),
        "endlen_recal_std": _pstd(recal_vals) if recal_vals else 0.0,
        "encoder": encoder,
        **{f"{col}_mean": rt_data[col]["mean"] for col in rt_data},
        **{f"{col}_std": rt_data[col]["std"] for col in rt_data},
    }


def _rt_col_name(rt: float) -> str:
    """Stable, pandas-safe column name for an rt_error float.

    Uses Python's repr so 1e-07 → ``rt_1e-07``, 3e-07 → ``rt_3e-07``.
    """
    return f"rt_{rt!r}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _md_cell(row: dict[str, Any], col: str, n: int) -> str:
    """Render one table cell for Markdown output."""
    mean_v = row.get(f"{col}_mean")
    std_v = row.get(f"{col}_std", 0.0)
    if n == 1:
        return _fmt2(mean_v)
    return _fmt_mean_std(mean_v, std_v)


def render_markdown(
    rows: list[tuple[str, dict[str, Any]]],
    rt_errors: list[float],
    metric: str,
) -> str:
    """Build the master Markdown table."""
    rt_cols = [_rt_col_name(rt) for rt in rt_errors]
    rt_headers = [repr(rt) for rt in rt_errors]  # e.g. "1e-07"

    header_parts = ["experiment", "n", "encoder", "clean", "endlen", "endlen_recal"] + rt_headers
    sep_parts = ["---"] * len(header_parts)

    lines: list[str] = []
    lines.append(f"# NetDrift Comparison DB — metric: `{metric}`")
    lines.append("")
    lines.append(f"- rt_error columns: {rt_headers}")
    lines.append(f"- metric per cell: `{metric}`")
    lines.append("- seeded rows: `mean±std` (population std); n=1 rows: value only")
    lines.append("- `—` = not present in summary")
    lines.append("")
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("|" + "|".join(sep_parts) + "|")

    for group_key, row in rows:
        n = row["n"]
        enc = row.get("encoder") or _DASH
        clean_cell = _md_cell(row, "clean", n)
        endlen_cell = _md_cell(row, "endlen", n)
        recal_cell = _md_cell(row, "endlen_recal", n)
        rt_cells = [_md_cell(row, col, n) for col in rt_cols]
        parts = [group_key, str(n), enc, clean_cell, endlen_cell, recal_cell] + rt_cells
        lines.append("| " + " | ".join(parts) + " |")

    lines.append("")
    return "\n".join(lines)


def render_csv(
    rows: list[tuple[str, dict[str, Any]]],
    rt_errors: list[float],
    metric: str,
) -> str:
    """Build the master CSV.

    Separate ``<col>_mean`` and ``<col>_std`` columns for every numeric field
    so the file is directly usable with pandas (no string splitting needed).
    """
    import io
    rt_cols = [_rt_col_name(rt) for rt in rt_errors]
    rt_headers_mean = [f"rt_{rt!r}_mean" for rt in rt_errors]
    rt_headers_std = [f"rt_{rt!r}_std" for rt in rt_errors]
    # Interleave mean/std per rt_error column
    rt_interleaved: list[str] = []
    for m_col, s_col in zip(rt_headers_mean, rt_headers_std):
        rt_interleaved.extend([m_col, s_col])

    fieldnames = (
        ["experiment", "n", "encoder",
         "clean_mean", "clean_std",
         "endlen_mean", "endlen_std",
         "endlen_recal_mean", "endlen_recal_std"]
        + rt_interleaved
        + ["metric"]
    )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()

    for group_key, row in rows:
        record: dict[str, Any] = {
            "experiment": group_key,
            "n": row["n"],
            "encoder": row.get("encoder") or "",
            "clean_mean": _fmt2(row.get("clean_mean")),
            "clean_std": f"{row.get('clean_std', 0.0):.2f}",
            "endlen_mean": _fmt2(row.get("endlen_mean")),
            "endlen_std": f"{row.get('endlen_std', 0.0):.2f}",
            "endlen_recal_mean": _fmt2(row.get("endlen_recal_mean")),
            "endlen_recal_std": f"{row.get('endlen_recal_std', 0.0):.2f}",
            "metric": metric,
        }
        for rt, m_col_name, s_col_name in zip(rt_cols, rt_headers_mean, rt_headers_std):
            mean_v = row.get(f"{rt}_mean")
            std_v = row.get(f"{rt}_std", 0.0)
            record[m_col_name] = _fmt2(mean_v)
            record[s_col_name] = f"{std_v:.2f}"
        writer.writerow(record)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--runs-dir",
        default=str(REPO_ROOT / "runs"),
        help="Root directory to scan for experiment subdirs (default: runs/)",
    )
    p.add_argument(
        "--include",
        nargs="*",
        default=None,
        metavar="SUBSTR",
        help=(
            "Optional substring filters.  If given, only experiment dirs whose "
            "name contains ANY of these strings are included."
        ),
    )
    p.add_argument(
        "--rt-errors",
        nargs="+",
        type=float,
        default=DEFAULT_RT_ERROR_CURVE,
        metavar="RT",
        help=(
            "rt_error values to use as columns (default: the canonical curve "
            f"{DEFAULT_RT_ERROR_CURVE})."
        ),
    )
    p.add_argument(
        "--metric",
        choices=["mean", "min", "max", "last"],
        default="mean",
        help="Which per-rt_error statistic to put in the cells (default: mean).",
    )
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Base path for output files (without extension).  "
            "Default: runs/sweeps/comparison_db_<timestamp>  "
            "(.md and .csv are appended)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered runs and exit without computing or writing output.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    runs_dir = Path(args.runs_dir)
    rt_errors: list[float] = args.rt_errors
    metric: str = args.metric

    # ---- Discovery ----
    runs = discover_runs(runs_dir, args.include)

    if not runs:
        print("No runs found matching criteria.", file=sys.stderr)
        return 1

    # ---- Dry run ----
    if args.dry_run:
        print(f"Discovered {len(runs)} run(s):")
        for name, summary_path in runs:
            print(f"  {name!s:55s}  {summary_path}")
        return 0

    # ---- Group by seed ----
    groups = group_by_seed(runs)

    # ---- Aggregate each group ----
    rows: list[tuple[str, dict[str, Any]]] = []
    for group_key, members in groups.items():
        row = aggregate_group(members, rt_errors, metric)
        rows.append((group_key, row))

    # Sort by group key for stable output
    rows.sort(key=lambda kv: kv[0].lower())

    # ---- Render ----
    md_text = render_markdown(rows, rt_errors, metric)
    csv_text = render_csv(rows, rt_errors, metric)

    # ---- Output paths ----
    if args.out:
        out_base = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_base = REPO_ROOT / "runs" / "sweeps" / f"comparison_db_{ts}"

    out_base.parent.mkdir(parents=True, exist_ok=True)
    md_path = out_base.with_suffix(".md")
    csv_path = out_base.with_suffix(".csv")

    md_path.write_text(md_text, encoding="utf-8")
    csv_path.write_text(csv_text, encoding="utf-8")

    # ---- Summary ----
    n_total = sum(row["n"] for _, row in rows)
    print(
        f"Harvested {n_total} run(s) → {len(rows)} group(s)  "
        f"({len(rows) - sum(1 for _, r in rows if r['n'] == 1)} seeded groups)"
    )
    print(f"  Markdown: {md_path}")
    print(f"  CSV:      {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
