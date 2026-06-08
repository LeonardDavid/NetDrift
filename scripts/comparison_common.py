"""Shared helpers for the NetDrift comparison-database driver scripts.

The comparison DB is a set of model/training configurations, each evaluated over
the SAME rt_error degradation curve (×N loops). Every driver in ``scripts/``
that contributes to the DB uses these helpers so naming, summary harvesting, and
the per-cell record schema stay consistent — which is what lets
``aggregate_comparison_db.py`` pull everything into one master table.

Design notes baked in from the robustness study:
* Always evaluate over an rt_error CURVE, never a single point (saturation
  hides the signal at high rt_error).
* Trained configs are seed-dependent; drivers loop seeds and the record carries
  the seed so the aggregator can compute mean±std.
* The runner writes ``runs/<experiment.name>/<timestamp>/summary.json`` carrying
  ``rt_error_sweep`` + baselines; we harvest the newest one per experiment name.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "code" / "python"

# The canonical evaluation curve for the whole comparison DB. Drivers pass this
# as a list-valued fault.rt_error override so the runner sweeps it in one run.
DEFAULT_RT_ERROR_CURVE = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5]
DEFAULT_LOOPS = 10
# Protection: first+last layer protected (VGG3: conv1+fc2). The comparison DB
# fixes this per the experiment design; drivers can override if needed.
DEFAULT_PROTECTION_LAYERS = [2, 3]
# One wandb project for the whole DB so runs are queryable together.
DEFAULT_WANDB_PROJECT = "netdrift-comparison-db"


def fmt_num(v: float) -> str:
    """Filename/tag-safe number, e.g. 0.05 -> '0p05', 1e-06 -> '1e-06'."""
    return str(v).replace(".", "p").replace("+", "")


def rt_error_list_override(curve: list[float]) -> str:
    """Render an rt_error curve as the JSON-list string the override parser wants.

    The CLI override parser runs ``json.loads`` on the value, so a Python list
    literal with no spaces (``[1e-07,3e-07,...]``) parses cleanly.
    """
    return "[" + ",".join(repr(float(x)) for x in curve) + "]"


def output_dir_from_cfg(cfg_path: Path) -> Path:
    """Best-effort read of ``experiment.output_dir`` from a YAML (PyYAML or regex)."""
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
            m2 = re.search(r"^\s+output_dir:\s*(\S+)", m.group(1), re.MULTILINE)
            if m2:
                return Path(m2.group(1).strip("\"'"))
        return Path("runs/")


def latest_summary(out_dir: Path, exp_name: str) -> Optional[Path]:
    """Newest ``summary.json`` under ``<out_dir>/<exp_name>/`` or None."""
    base = out_dir / exp_name
    if not base.exists():
        return None
    candidates = sorted(base.glob("*/summary.json"))
    return candidates[-1] if candidates else None


def harvest_summary(summary_path: Optional[Path]) -> dict[str, Any]:
    """Parse a runner summary.json into the DB's per-cell metric schema.

    Returns a dict with the baselines and, for each rt_error in the sweep, the
    per-loop accuracy list plus mean/min/max/last. Missing/absent values are
    None so the aggregator can render '—'. The ``rt_curve`` key maps
    ``rt_error -> {mean,min,max,last,accuracies}``.
    """
    out: dict[str, Any] = {
        "baseline_clean_accuracy": None,
        "baseline_endlen_accuracy": None,
        "baseline_endlen_recal_accuracy": None,
        "weight_encoder": None,
        "rt_curve": {},
    }
    if summary_path is None or not summary_path.exists():
        return out
    try:
        with open(summary_path) as f:
            s = json.load(f)
    except Exception:
        return out
    for k in (
        "baseline_clean_accuracy",
        "baseline_endlen_accuracy",
        "baseline_endlen_recal_accuracy",
        "weight_encoder",
    ):
        out[k] = s.get(k)
    for entry in s.get("rt_error_sweep", []) or []:
        rt = entry.get("rt_error")
        accs = entry.get("accuracies") or []
        if rt is None or not accs:
            continue
        out["rt_curve"][float(rt)] = {
            "mean": sum(accs) / len(accs),
            "min": min(accs),
            "max": max(accs),
            "last": accs[-1],
            "accuracies": accs,
        }
    return out


def base_overrides(
    *,
    curve: list[float],
    loops: int,
    protection_layers: list[int],
) -> list[str]:
    """The override args every comparison-DB cell shares (curve, loops, protection)."""
    return [
        "--override", f"fault.rt_error={rt_error_list_override(curve)}",
        "--override", f"training.loops={loops}",
        "--override", "fault.protection.policy=custom",
        "--override", f"fault.protection.layers={json_list(protection_layers)}",
    ]


def json_list(xs: list[int]) -> str:
    return "[" + ",".join(str(int(x)) for x in xs) + "]"


def wandb_args(
    project: Optional[str],
    entity: Optional[str],
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
) -> list[str]:
    """Runner W&B flags for one cell.

    ``category`` (coarse, by mode) and ``subcategory`` (fine, by exact setting
    combination) are each logged as a config field + a run tag so runs group
    natively in the UI.

    Category labels MUST match scripts/wandb_tag_categories.py so live-tagged and
    backfilled runs share the same label set: cat1_baseline, cat2_vanilla_endlen,
    cat3_budgeted_endlen, cat4_endlen_recal, cat5_regularizer, cat6_reg_recal,
    cat7_reg_endlen, cat8_ste_inject. Subcategory convention: ``<category>_<combo>``
    e.g. cat3_sc-channel_sel-greedy_gl1p0_lo0p1, cat4_recal-bn-affine,
    cat5_lam0p05_inj-fresh.
    """
    if not project:
        return []
    args = ["--wandb-project", project]
    if entity:
        args += ["--wandb-entity", entity]
    if category:
        args += ["--wandb-category", category]
    if subcategory:
        args += ["--wandb-subcategory", subcategory]
    return args


def run_cell(runner_main, argv_cell: list[str]) -> tuple[str, Optional[str]]:
    """Invoke the runner for one cell; return ``(status, error)``.

    ``status`` is 'ok' | 'nonzero_exit' | 'error'. Never raises — a failed cell
    is recorded so the sweep continues.
    """
    try:
        rc = runner_main(argv_cell)
        if rc != 0:
            return "nonzero_exit", f"runner returned {rc}"
        return "ok", None
    except Exception as exc:  # noqa: BLE001 — record and continue the sweep
        return "error", repr(exc)


def import_runner_main():
    """Import ``netdrift.runner.run.main`` after putting code/python on sys.path."""
    import sys
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from netdrift.runner.run import main as runner_main
    return runner_main


def new_sweep_out_dir(tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "runs" / "sweeps" / f"{ts}_{tag}"


def write_manifest(out_dir: Path, payload: dict[str, Any]) -> None:
    """Persist a manifest after each cell so a mid-sweep crash leaves a record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
