"""Metrics artifact writers: per-(model, category, rt_error) JSON + .npz sidecars.

Pure serialization — no dependency on config or fault code. The ``meta`` block
is assembled by the caller (the runner, from ``_wandb_config``). Raw per-racetrack
arrays go to ``.npz`` and are referenced from the JSON by filename, never inlined.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

SCHEMA_VERSION = 1


def _fmt_rt_error(rt_error: float) -> str:
    """Stable filename token for an rt_error value, e.g. 1e-05 -> 'rt1e-05'."""
    return "rt" + format(float(rt_error), "g")


def _basename(model: str, category: str, suffix: str) -> str:
    return f"{model}__{category}__{suffix}"


def write_static_artifact(
    out_dir: Path, *, model: str, category: str, meta: dict,
    snapshots: list, deltas: Optional[dict], recal: Optional[dict],
    npz_arrays: Optional[dict] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_ref = None
    if npz_arrays:
        npz_path = out_dir / _basename(model, category, "static.npz")
        np.savez_compressed(npz_path, **npz_arrays)
        npz_ref = npz_path.name
    meta = {**meta, "npz_ref": npz_ref}
    doc = {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "snapshots": snapshots,
        "deltas": deltas or {},
        "recal": recal or {},
    }
    json_path = out_dir / _basename(model, category, "static.json")
    json_path.write_text(json.dumps(doc, indent=2, default=str))
    return json_path


def write_rt_error_artifact(
    out_dir: Path, *, model: str, category: str, rt_error: float, meta: dict,
    outcome: dict, fault_incidence: dict, npz_arrays: Optional[dict] = None,
    static_ref: Optional[str] = None,
) -> tuple[Path, Optional[Path]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{_fmt_rt_error(rt_error)}.json"
    json_path = out_dir / _basename(model, category, suffix)
    npz_path: Optional[Path] = None
    npz_ref = None
    if npz_arrays:
        npz_path = out_dir / _basename(model, category, f"{_fmt_rt_error(rt_error)}.npz")
        np.savez_compressed(npz_path, **npz_arrays)
        npz_ref = npz_path.name
    meta = {**meta, "rt_error": float(rt_error), "npz_ref": npz_ref,
            "static_ref": static_ref}
    doc = {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "outcome": outcome,
        "fault_incidence": fault_incidence,
    }
    json_path.write_text(json.dumps(doc, indent=2, default=str))
    return json_path, npz_path


def distribution_stats(values: list) -> dict:
    """Mean/std/min/max + p25/p50/p75 over a list of floats. ``{}`` if empty."""
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }
