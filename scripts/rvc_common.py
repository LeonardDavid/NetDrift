#!/usr/bin/env python
"""Shared discovery / parsing / styling for the row-vs-col metrics analysis.

This module is the single source of truth for turning the ``runs_row-v-col/``
metrics tree into tidy, plot-ready data. It is import-only (no argparse) so both
``aggregate_row_v_col.py`` and ``plot_row_v_col.py`` consume the same parsers.

Data layout (see docs/superpowers/specs/2026-06-19-row-v-col-metrics-plots-design.md)::

    runs_row-v-col/<exp>/<category>/<tag>/<timestamp>/metrics/
        <model>__<category>__rt<err>.json     # outcome + fault_incidence
        <model>__<category>__static.json      # mechanism (snapshots + deltas)
        ... (.npz siblings with raw per-racetrack arrays)

  NOTE: runner/run.py writes this directory as ``metrics_artifacts/`` as of
  2026-07-31 (renamed so mutagen sync can target run artifacts separately from
  the ``code/python/netdrift/metrics/`` source package, which shared the bare
  name ``metrics``). Older runs on disk still have ``metrics/``. The discovery
  helpers below search BOTH names so already-collected artifacts stay
  readable — do not assume only one exists.

Parsing rules that bite (load-bearing):
  * ``rt_error`` may sit at ``d["rt_error"]`` OR ``d["meta"]["rt_error"]``.
  * lambda is NOT recoverable from config (test-phase ``reg.lambda_`` is 0.0);
    parse it from the leaf-dir tag (``lam0p01`` -> 0.01).
  * static snapshot labels differ by category; treat the last as "final".
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs_row-v-col"

#: Friendly display names for the categories, in canonical (presentation) order.
CATEGORY_ORDER = [
    "cat1_baseline",
    "cat2_vanilla_endlen",
    "cat4_endlen_recal",
    "cat5_regularizer",
    "cat6_reg_recal",
    "cat8_ste_inject",
]
CATEGORY_LABELS = {
    "cat1_baseline": "cat1: baseline",
    "cat2_vanilla_endlen": "cat2: endlen",
    "cat4_endlen_recal": "cat4: endlen+recal",
    "cat5_regularizer": "cat5: regularizer",
    "cat6_reg_recal": "cat6: reg+recal",
    "cat8_ste_inject": "cat8: STE inject",
}

#: Per-loop fault metrics with a `.total` and `.per_layer` array each.
FAULT_LOOP_METRICS = [
    "bitflips",
    "wrong_bits_read",
    "misalign_faults",
    "affected_units",
]
#: Scalar fault-incidence keys reported once at the final loop.
FAULT_LAST_LOOP_KEYS = [
    "bitflips",
    "wrong_bits_read",
    "misalign_faults",
    "affected_units",
    "ber",
]

# ---------------------------------------------------------------------------
# Styling (shared so every figure is visually consistent)
# ---------------------------------------------------------------------------
#: One stable colour per category (matplotlib tab10-ish, colour-blind aware).
CATEGORY_COLORS = {
    "cat1_baseline": "#4c72b0",
    "cat2_vanilla_endlen": "#dd8452",
    "cat4_endlen_recal": "#55a868",
    "cat5_regularizer": "#c44e52",
    "cat6_reg_recal": "#8172b3",
    "cat8_ste_inject": "#937860",
}
#: Layout encoding: row = solid / filled, col = dashed / hatched.
LAYOUT_LINESTYLE = {"row": "-", "col": "--"}
LAYOUT_MARKER = {"row": "o", "col": "s"}
LAYOUT_HATCH = {"row": "", "col": "///"}
LAYOUT_LABELS = {"row": "row", "col": "col"}

_LAM_RE = re.compile(r"lam(\d+)p(\d+)")
_RT_TAG_RE = re.compile(r"__rt([0-9.eE+-]+)\.json$")


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------
def parse_lambda_from_path(path: Path) -> Optional[float]:
    """Recover the *training* lambda from a leaf-dir tag (``lam0p01`` -> 0.01).

    Returns ``None`` for categories without a lambda variant. Config is NOT a
    reliable source (test-phase ``reg.lambda_`` is 0.0).
    """
    m = _LAM_RE.search(str(path))
    if not m:
        return None
    whole, frac = m.group(1), m.group(2)
    return float(f"{whole}.{frac}")


def rt_error_of(doc: dict[str, Any]) -> Optional[float]:
    """rt_error lives at top-level OR under meta, depending on the driver."""
    val = doc.get("rt_error")
    if val is None:
        val = doc.get("meta", {}).get("rt_error")
    return None if val is None else float(val)


def lambda_label(lam: Optional[float]) -> str:
    """Render a lambda for filenames / legends (``None`` -> empty)."""
    if lam is None:
        return ""
    return f"lam{lam:g}".replace(".", "p")


def cell_key(category: str, layout: str, lam: Optional[float]) -> str:
    """Stable identifier for a (category, layout, lambda) cell."""
    parts = [category, layout]
    if lam is not None:
        parts.append(lambda_label(lam))
    return "__".join(parts)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
#: Directory names the per-run metrics artifacts may live under. Runs from
#: before the 2026-07-31 rename used "metrics"; runs after use
#: "metrics_artifacts" (see module docstring). Search both so already-
#: collected artifacts on the host stay discoverable — a reader that only
#: recognizes the new name would silently find nothing for old runs while
#: still exiting 0, which is exactly the "reports success but aggregated
#: nothing" failure mode this project has already hit once
#: (comparison_common.latest_summary, see the 2026-07-30 SDD ledger).
_METRICS_DIRNAMES = ("metrics", "metrics_artifacts")


def discover_rt_files(runs_dir: Path, include: Optional[list[str]] = None) -> list[Path]:
    """All per-rt_error outcome/fault JSONs (excludes ``__static``)."""
    files = {
        p
        for dirname in _METRICS_DIRNAMES
        for p in runs_dir.rglob(f"{dirname}/*.json")
        if "__rt" in p.name and "__static" not in p.name
    }
    if include:
        files = {p for p in files if any(tok in str(p) for tok in include)}
    return sorted(files)


def discover_static_files(runs_dir: Path, include: Optional[list[str]] = None) -> list[Path]:
    """All static (mechanism) JSONs."""
    files = {
        p
        for dirname in _METRICS_DIRNAMES
        for p in runs_dir.rglob(f"{dirname}/*__static.json")
    }
    if include:
        files = {p for p in files if any(tok in str(p) for tok in include)}
    return sorted(files)


# ---------------------------------------------------------------------------
# Outcome + fault-incidence -> tidy long frame
# ---------------------------------------------------------------------------
def _common_dims(doc: dict[str, Any], path: Path) -> dict[str, Any]:
    meta = doc["meta"]
    return {
        "category": meta["category"],
        "layout": meta["storage"]["layout"],
        "lambda": parse_lambda_from_path(path),
        "seed": meta.get("seed"),
        "rt_error": rt_error_of(doc),
    }


def tidy_from_rt_file(path: Path) -> pd.DataFrame:
    """Long-format rows for one per-rt_error JSON.

    Emits, per loop (1..N):
      * outcome.accuracy            (per_loop_accuracy)
      * fault.<metric>.total        (per_loop totals)
      * fault.<metric>.<layer>      (per_loop per-layer)
    plus per-cell scalar rows (loop=NaN):
      * outcome.clean_baseline, outcome.accuracy_{mean,std,min,max}
      * outcome.accuracy_drop_mean
      * fault_last.<key>  (final-loop bitflips/ber/affected_units/...)
    """
    doc = json.loads(path.read_text())
    dims = _common_dims(doc, path)
    rows: list[dict[str, Any]] = []

    def emit(family: str, metric: str, value: Any, loop: Optional[int] = None) -> None:
        if value is None:
            return
        rows.append({**dims, "loop": loop, "family": family, "metric": metric, "value": float(value)})

    out = doc.get("outcome", {})
    # per-loop accuracy trajectory
    for i, acc in enumerate(out.get("per_loop_accuracy", []) or [], start=1):
        emit("outcome", "accuracy", acc, loop=i)
    # cell-level outcome scalars
    base = out.get("baselines", {})
    emit("outcome", "clean_baseline", base.get("clean"))
    emit("outcome", "endlen_baseline", base.get("endlen"))
    emit("outcome", "endlen_recal_baseline", base.get("endlen_recal"))
    acc = out.get("accuracy", {})
    for stat in ("mean", "std", "min", "max", "p25", "p50", "p75"):
        emit("outcome", f"accuracy_{stat}", acc.get(stat))
    drop = out.get("accuracy_drop_vs_clean", {})
    emit("outcome", "accuracy_drop_mean", drop.get("mean"))

    fi = doc.get("fault_incidence", {})
    # per-loop fault metrics (total + per-layer)
    per_loop = fi.get("per_loop", {})
    for metric, blob in per_loop.items():
        for i, v in enumerate(blob.get("total", []) or [], start=1):
            emit("fault", f"{metric}.total", v, loop=i)
        for layer, arr in (blob.get("per_layer", {}) or {}).items():
            for i, v in enumerate(arr or [], start=1):
                emit("fault", f"{metric}.{layer}", v, loop=i)
    # final-loop scalars
    last = fi.get("last_loop", {})
    for key in FAULT_LAST_LOOP_KEYS:
        emit("fault_last", key, last.get(key))
    over = fi.get("sum_over_loops", {})
    for key, v in over.items():
        emit("fault_sum", key, v)

    out_df = pd.DataFrame(rows)
    # Pin numeric dtypes so the cross-file concat never sees an all-NA `lambda`
    # column (cat1/2/4/8 have no lambda) promoted to object -> FutureWarning.
    for col in ("lambda", "loop", "rt_error", "value", "seed"):
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce").astype("float64")
    return out_df


def load_tidy(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    include: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Concatenate the tidy long frame across all discovered rt-files."""
    frames = [f for f in (tidy_from_rt_file(p) for p in discover_rt_files(runs_dir, include)) if not f.empty]
    if not frames:
        return pd.DataFrame(
            columns=["category", "layout", "lambda", "seed", "rt_error", "loop", "family", "metric", "value"]
        )
    df = pd.concat(frames, ignore_index=True)
    # Stable categorical ordering for plotting.
    df["category"] = pd.Categorical(df["category"], categories=CATEGORY_ORDER, ordered=True)
    return df


# ---------------------------------------------------------------------------
# Mechanism (static) -> structured records
# ---------------------------------------------------------------------------
def load_static(path: Path) -> dict[str, Any]:
    """Parse one static.json into a flat record with snapshots + deltas.

    Returns dims + ``snapshots`` (list of {label, total{...}}) and ``deltas``
    (dict keyed ``a->b``). Histograms are kept as ``{int_bin: count}`` dicts.
    """
    doc = json.loads(path.read_text())
    meta = doc["meta"]
    rec: dict[str, Any] = {
        "category": meta["category"],
        "layout": meta["storage"]["layout"],
        "lambda": parse_lambda_from_path(path),
        "seed": meta.get("seed"),
        "path": str(path),
        "snapshots": [],
        "deltas": doc.get("deltas", {}) or {},
    }
    for snap in doc.get("snapshots", []):
        total = snap.get("total", {})
        rec["snapshots"].append(
            {
                "label": snap["label"],
                "block_count": total.get("block_count", {}),
                "sign_transitions": total.get("sign_transitions"),
                "run_length_histogram": _intkey(total.get("run_length_histogram", {})),
                "alternating_seq_histogram": _intkey(total.get("alternating_seq_histogram", {})),
                "total_alternating_sequences": total.get("total_alternating_sequences"),
            }
        )
    return rec


def load_all_static(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    include: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    return [load_static(p) for p in discover_static_files(runs_dir, include)]


def _intkey(hist: dict[str, Any]) -> dict[int, int]:
    """Histogram keys arrive as strings; sort numerically as ints."""
    return {int(k): int(v) for k, v in hist.items()}


def hist_to_xy(hist: dict[int, int], cumulative: bool = False, normalize: bool = False):
    """Sorted (xs, ys) from an int-keyed histogram, for line/bar plotting."""
    xs = sorted(hist)
    ys = [hist[x] for x in xs]
    if normalize:
        tot = sum(ys) or 1
        ys = [y / tot for y in ys]
    if cumulative:
        run = 0.0
        cum = []
        for y in ys:
            run += y
            cum.append(run)
        ys = cum
    return xs, ys


# ---------------------------------------------------------------------------
# Seed-aware reduction (collapses to n=1 cleanly)
# ---------------------------------------------------------------------------
def reduce_over_seeds(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Group by every dim except seed; return mean/std/n.

    With one seed, ``std`` is NaN from pandas -> filled to 0.0; ``n`` exposes
    the (currently 1) seed count so the caller can warn / draw bands.
    """
    keys = ["category", "layout", "lambda", "rt_error", "loop", "family", "metric"]
    keys = [k for k in keys if k in df.columns]
    g = df.groupby(keys, dropna=False, observed=True)[value_col]
    out = g.agg(["mean", "std", "count"]).reset_index().rename(columns={"count": "n"})
    out["std"] = out["std"].fillna(0.0)
    return out


def seeds_present(df: pd.DataFrame) -> list[int]:
    if "seed" not in df.columns or df.empty:
        return []
    return sorted({int(s) for s in df["seed"].dropna().unique()})
