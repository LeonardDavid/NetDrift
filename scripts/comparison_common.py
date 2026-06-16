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
# Protection is NOT defaulted by the drivers. ``None`` means "use whatever the
# config specifies" — drivers only override fault.protection.* when the user
# passes it explicitly. (A hardcoded default here used to be the VGG3 value
# [2,3], which silently clobbered VGG7's config [2,3,4,5,6,7].)
DEFAULT_PROTECTION_LAYERS = None
# One wandb project for the whole DB so runs are queryable together.
DEFAULT_WANDB_PROJECT = "netdrift-comparison-db"


def fmt_num(v: float) -> str:
    """Filename/tag-safe number, e.g. 0.05 -> '0p05', 1e-06 -> '1e-06'."""
    return str(v).replace(".", "p").replace("+", "")


def crit_token(criterion: str, hinge_b: float = 128.0) -> str:
    """Compact, parser-safe criterion tag for run/dir names.

    ``hinge`` → ``crit-h<b>`` (e.g. ``crit-h128p0``); ``cross_entropy`` →
    ``crit-ce``. Used as a NAME PREFIX segment and a save_dir PATH-LEVEL segment
    (never infix, never trailing-after-seed) so the existing $-anchored
    config_key/seed parsers stay valid. A driver invocation = one criterion, so
    embedding it disambiguates artifacts/runs when criteria share a save-root +
    W&B project.
    """
    if criterion == "cross_entropy":
        return "crit-ce"
    if criterion == "hinge":
        return f"crit-h{fmt_num(float(hinge_b))}"
    raise ValueError(f"unknown criterion {criterion!r}")


def parse_crit_token(tok: Optional[str]) -> tuple[str, float]:
    """Inverse of :func:`crit_token`: ``crit-…`` → ``(criterion, hinge_b)``.

    ``crit-ce`` → ``("cross_entropy", 128.0)``; ``crit-h128p0`` →
    ``("hinge", 128.0)``. ``None`` / unrecognized → ``("hinge", 128.0)`` (the
    default), so callers can treat a missing token as plain MHL.
    """
    if not tok or not tok.startswith("crit-"):
        return "hinge", 128.0
    body = tok[len("crit-"):]
    if body == "ce":
        return "cross_entropy", 128.0
    if body.startswith("h"):
        b = float(body[1:].replace("p", "."))
        return "hinge", b
    return "hinge", 128.0


def layout_token(layout: str) -> str:
    """Compact, parser-safe racetrack-layout tag for run/dir names.

    ``row`` → ``lay-row``; ``col`` → ``lay-col``. Used exactly like
    :func:`crit_token` — as a NAME PREFIX segment, a save_dir PATH-LEVEL
    segment, and a wandb-subcategory suffix — so ROW and COL artifacts never
    collide when a sweep is re-run with the only difference being the layout.

    Case-insensitive (a ``COL`` config and a ``col`` config map to one tree;
    the runner upper-cases ``storage.layout`` internally). Only the two
    fault-wired layouts are accepted; ``mix``/``interleaved`` are schema
    placeholders and raise rather than mint a token that would silently mix
    incomparable runs.
    """
    norm = (layout or "row").lower()
    if norm in ("row", "col"):
        return f"lay-{norm}"
    raise ValueError(
        f"layout_token: unsupported layout {layout!r}; expected 'row' or 'col' "
        f"(mix/interleaved are not wired into the fault model)"
    )


def layout_from_cfg(cfg_path: Path) -> str:
    """Best-effort read of ``storage.layout`` from a YAML; default ``row``.

    Mirrors :func:`output_dir_from_cfg`. The comparison drivers derive the
    layout token from whichever ``--config`` they are given (e.g. a
    ``*_rtm_col.yaml`` variant), so the token always matches the layout the run
    actually uses — no separate ``--layout`` flag to keep in sync.
    """
    try:
        import yaml  # type: ignore[import-not-found]
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
        return str(raw.get("storage", {}).get("layout", "row"))
    except ModuleNotFoundError:
        import re
        text = cfg_path.read_text()
        m = re.search(r"^storage:\s*\n((?:[ \t].*\n)+)", text, re.MULTILINE)
        if m:
            m2 = re.search(r"^\s+layout:\s*(\S+)", m.group(1), re.MULTILINE)
            if m2:
                return m2.group(1).strip("\"'")
        return "row"


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
    protection_policy: Optional[str] = None,
    protection_layers: Optional[list[int]] = None,
) -> list[str]:
    """The override args every comparison-DB cell shares (curve, loops, protection).

    Protection is emitted ONLY when explicitly requested. When both
    ``protection_policy`` and ``protection_layers`` are ``None``, no
    ``fault.protection.*`` override is added and the runner uses the config's
    own protection block (which the schema validates / halts on if a ``custom``
    policy is missing layers). This prevents a driver default from silently
    clobbering a model's config (e.g. VGG3's ``[2,3]`` overriding VGG7's
    ``[2,3,4,5,6,7]``).

    Passing ``protection_layers`` alone implies ``policy=custom`` (the common
    escape-hatch case). Passing ``protection_policy`` alone (e.g. ``all``) emits
    just the policy.
    """
    out = [
        "--override", f"fault.rt_error={rt_error_list_override(curve)}",
        "--override", f"training.loops={loops}",
    ]
    if protection_policy is not None or protection_layers is not None:
        policy = protection_policy or "custom"
        out += ["--override", f"fault.protection.policy={policy}"]
        if protection_layers is not None:
            out += ["--override", f"fault.protection.layers={json_list(protection_layers)}"]
    return out


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
