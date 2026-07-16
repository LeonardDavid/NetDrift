#!/usr/bin/env python
"""Aggregate the ``runs_row-v-col/`` metrics tree into tidy + summary tables.

Outputs (default ``runs_row-v-col/aggregated/``):
  * ``tidy_long.csv``   — canonical long-format frame (one row per
    category/layout/lambda/seed/rt_error/loop/metric). Consumed by
    ``plot_row_v_col.py``; also pandas-friendly for ad-hoc analysis.
  * ``summary_wide.csv`` — one row per (category, lambda, rt_error) with the
    final-loop & mean-over-loops headline numbers for BOTH layouts side by side
    (``*_row`` / ``*_col`` columns) plus the row−col deltas.
  * ``summary.md``       — the same table rendered as Markdown for quick reading.

Usage::

    python scripts/aggregate_row_v_col.py                 # all cells
    python scripts/aggregate_row_v_col.py --include cat5 cat6
    python scripts/aggregate_row_v_col.py --dry-run       # list discovered cells
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make scripts/ importable regardless of CWD (house pattern).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pandas as pd  # noqa: E402

import rvc_common as rvc  # noqa: E402


# ---------------------------------------------------------------------------
# Headline summary
# ---------------------------------------------------------------------------
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (category, lambda, rt_error); row & col columns side by side.

    NOTE: ``lambda`` is NaN for the non-lambda categories. Pandas treats NaN
    keys as unequal under both merge and pivot, which silently drops those
    cells. We therefore stringify lambda into ``lam`` ("-" sentinel for None)
    and carry the original float separately for sorting/output.
    """
    red = rvc.reduce_over_seeds(df).copy()
    red["lam"] = red["lambda"].apply(lambda v: "-" if pd.isna(v) else f"{v:g}")

    def pick(metric: str, family: str, *, final: bool) -> pd.DataFrame:
        sub = red[(red["family"] == family) & (red["metric"] == metric)].copy()
        if final:
            # final loop = max loop within each cell (loop-indexed metric)
            sub = sub[sub["loop"].notna()]
            if sub.empty:
                return sub.assign(value=pd.Series(dtype=float))
            idx = sub.groupby(
                ["category", "layout", "lam", "rt_error"], dropna=False, observed=True
            )["loop"].idxmax()
            sub = sub.loc[idx]
        else:
            sub = sub[sub["loop"].isna()]  # cell-level scalar (loop is NaN)
        return sub.rename(columns={"mean": "value"})

    parts = {
        "acc_final": pick("accuracy", "outcome", final=True),
        "acc_mean": pick("accuracy_mean", "outcome", final=False),
        "acc_drop_mean": pick("accuracy_drop_mean", "outcome", final=False),
        "clean": pick("clean_baseline", "outcome", final=False),
        "ber_final": pick("ber", "fault_last", final=False),
        "affected_final": pick("affected_units", "fault_last", final=False),
        "bitflips_final": pick("bitflips", "fault_last", final=False),
    }

    keys = ["category", "lam", "rt_error", "layout"]
    long_parts = []
    for name, sub in parts.items():
        if sub.empty:
            continue
        s = sub[keys + ["value"]].copy()
        s["quantity"] = name
        long_parts.append(s)
    if not long_parts:
        return pd.DataFrame()
    long = pd.concat(long_parts, ignore_index=True)

    # one pivot: index=cell, columns=(quantity, layout). aggfunc=first because
    # each (cell, quantity, layout) is already unique after the picks above.
    wide = long.pivot_table(
        index=["category", "lam", "rt_error"],
        columns=["quantity", "layout"],
        values="value",
        aggfunc="first",
        observed=True,
    )
    wide.columns = [f"{q}_{layout}" for q, layout in wide.columns]
    wide = wide.reset_index()

    # row − col deltas for the headline metrics (positive = row larger)
    for metric in ("acc_final", "acc_mean", "ber_final"):
        rc, cc = f"{metric}_row", f"{metric}_col"
        if rc in wide.columns and cc in wide.columns:
            wide[f"{metric}_row_minus_col"] = wide[rc] - wide[cc]

    wide = wide.rename(columns={"lam": "lambda"})
    wide["_catorder"] = wide["category"].map(lambda c: rvc.CATEGORY_ORDER.index(c) if c in rvc.CATEGORY_ORDER else 99)
    wide = wide.sort_values(["_catorder", "lambda", "rt_error"]).drop(columns="_catorder").reset_index(drop=True)
    return wide


def _fmt(col: str, v) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, float):
        if col == "rt_error":
            return f"{v:g}"  # keep scientific notation (4.55e-07 not 0.0)
        return f"{v:g}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    cols = list(summary.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in summary.iterrows():
        lines.append("| " + " | ".join(_fmt(c, r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=rvc.DEFAULT_RUNS_DIR, help="metrics tree root")
    ap.add_argument("--out-dir", type=Path, default=None, help="output dir (default <runs-dir>/aggregated)")
    ap.add_argument("--include", nargs="*", default=None, help="only paths containing any of these tokens")
    ap.add_argument("--dry-run", action="store_true", help="list discovered cells and exit")
    args = ap.parse_args()

    runs_dir = args.runs_dir
    if not runs_dir.exists():
        print(f"ERROR: runs dir not found: {runs_dir}", file=sys.stderr)
        return 2

    rt_files = rvc.discover_rt_files(runs_dir, args.include)
    static_files = rvc.discover_static_files(runs_dir, args.include)
    print(f"Discovered {len(rt_files)} rt-files, {len(static_files)} static-files under {runs_dir}")

    if args.dry_run:
        df = rvc.load_tidy(runs_dir, args.include)
        if df.empty:
            print("(no data)")
            return 0
        cells = (
            df[["category", "layout", "lambda", "seed", "rt_error"]]
            .drop_duplicates()
            .sort_values(["category", "layout", "lambda", "rt_error"])
        )
        print(f"\nseeds present: {rvc.seeds_present(df)}")
        print(f"\n{len(cells)} (category, layout, lambda, seed, rt_error) cells:")
        for _, r in cells.iterrows():
            lam = "" if pd.isna(r["lambda"]) else f" lam={r['lambda']:g}"
            print(f"  {r['category']:22s} {r['layout']:3s}{lam:>10s} seed={r['seed']} rt={r['rt_error']:g}")
        return 0

    out_dir = args.out_dir or (runs_dir / "aggregated")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = rvc.load_tidy(runs_dir, args.include)
    if df.empty:
        print("ERROR: no metrics parsed", file=sys.stderr)
        return 1
    seeds = rvc.seeds_present(df)
    print(f"seeds present: {seeds}  (n={len(seeds)}; error bands collapse to 0 at n=1)")

    tidy_path = out_dir / "tidy_long.csv"
    df.to_csv(tidy_path, index=False)
    print(f"wrote {tidy_path}  ({len(df):,} rows)")

    summary = build_summary(df)
    if not summary.empty:
        sw = out_dir / "summary_wide.csv"
        summary.to_csv(sw, index=False)
        write_markdown(summary, out_dir / "summary.md")
        print(f"wrote {sw}  ({len(summary)} cells)")
        print(f"wrote {out_dir / 'summary.md'}")
    else:
        print("WARNING: summary empty (no outcome/fault rows matched)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
