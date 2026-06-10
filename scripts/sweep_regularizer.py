#!/usr/bin/env python
"""Comparison-DB driver for run-length regularizer (cat 5) and ste_inject (cat 8).

Each cell is a TWO-PHASE experiment:

    Phase 1 — TRAIN with fault_aware training (regularization or ste_inject),
               producing a checkpoint at a known path.
    Phase 2 — TEST the trained checkpoint WITHOUT any weight encoder, evaluated
               over the full rt_error CURVE.  The rt_curve from this test phase
               is the primary metric.

The multi-seed loop is the whole point: single-seed results for fault-aware
training were shown to be misleading.  This driver runs every configuration
across --seeds (default 3 seeds) and the output aggregates mean±std across
seeds so the comparison DB reflects real variance.

Category 5 — run-length regularizer grid
-----------------------------------------
For each seed × for each (lambda, inject_faults, fault_state_mode):
  - lambda ∈ --lambdas (default [0.0, 0.01, 0.05, 0.1]), inject=False, state=fresh
    → core lambda sweep; lambda=0.0 is the CONTROL (fine-tuning only, no reg)
  - (inject=True, state=fresh)  at lambda=0.05   — faults-in-the-loop variant A
  - (inject=True, state=accumulate) at lambda=0.05 — faults-in-the-loop variant B
Default: 4 (lambda, no-inject) + 2 (inject variants) = 6 configs × N seeds.

Category 8 — ste_inject (only with --include-ste)
--------------------------------------------------
fault_aware=ste_inject × N seeds.  Flagged as UNVALIDATED in the output because
the ste_inject path has not been experimentally verified against endlen baselines.

Training LR note
----------------
The rtm config's training.lr defaults to 1.0 (Clippy for BNNs) which DIVERGES
for fault-aware fine-tuning.  --train-lr (default 0.001) is ALWAYS applied as
training.lr on the train phase.  Do not remove it.

Usage::

    python scripts/sweep_regularizer.py \\
        --config configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml \\
        --seeds 707 1 42

    # quick smoke-test
    python scripts/sweep_regularizer.py \\
        --config configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml \\
        --seeds 707 --lambdas 0.0 0.05 --dry-run

    # include ste_inject category and W&B
    python scripts/sweep_regularizer.py \\
        --config configs/vgg3_fmnist/vgg3_fmnist_w1a1_rtm.yaml \\
        --include-ste --wandb-project netdrift-comparison-db
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

# Put the scripts/ directory on sys.path so comparison_common is importable
# without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparison_common import (  # noqa: E402
    DEFAULT_LOOPS,
    DEFAULT_PROTECTION_LAYERS,
    DEFAULT_RT_ERROR_CURVE,
    base_overrides,
    crit_token,
    fmt_num,
    harvest_summary,
    import_runner_main,
    json_list,
    latest_summary,
    new_sweep_out_dir,
    output_dir_from_cfg,
    run_cell,
    wandb_args,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------

def _make_tag(
    *,
    category: int,
    lam: float | None,
    inject: bool,
    state: str,
    seed: int,
) -> str:
    """Unique per-cell tag for experiment.name and save_dir.

    Category 5: ``cat5_lam{L}[_inj-{state}]_seed{seed}``
    Category 8: ``cat8_ste_seed{seed}``
    """
    if category == 8:
        return f"cat8_ste_seed{seed}"
    # category 5
    parts = [f"cat5_lam{fmt_num(lam)}"]
    if inject:
        parts.append(f"inj-{state}")
    parts.append(f"seed{seed}")
    return "_".join(parts)


def _config_key(
    *,
    category: int,
    lam: float | None,
    inject: bool,
    state: str,
) -> str:
    """Seed-independent key used to group seeds together for mean±std.

    Category 5: ``lam{L}`` or ``lam{L}_inj-{state}``
    Category 8: ``ste``
    """
    if category == 8:
        return "ste"
    parts = [f"lam{fmt_num(lam)}"]
    if inject:
        parts.append(f"inj-{state}")
    return "_".join(parts)


def build_cells(
    *,
    lambdas: list[float],
    seeds: list[int],
    include_ste: bool,
    inject_lambdas: list[float] | None = None,
) -> list[dict]:
    """Return the full ordered list of cells.

    ``lambdas``        — λ values for the core (no-injection) regularizer sweep.
    ``inject_lambdas`` — λ values for the faults-in-the-loop variant (fresh
                         fault_state_mode only). Defaults to ``[0.05]``.

    Each dict has keys:
        category, tag, config_key, seed, lam (None for cat8),
        inject, state (cat5 only; cat8 has fault_aware=ste_inject)
    """
    if inject_lambdas is None:
        inject_lambdas = [0.05]
    cells: list[dict] = []

    # Category 5
    for seed in seeds:
        # Core lambda sweep: no fault injection, state=fresh (irrelevant)
        for lam in lambdas:
            tag = _make_tag(category=5, lam=lam, inject=False, state="fresh", seed=seed)
            cells.append({
                "category": 5,
                "tag": tag,
                "config_key": _config_key(category=5, lam=lam, inject=False, state="fresh"),
                "seed": seed,
                "lam": lam,
                "inject": False,
                "state": "fresh",
                "fault_aware": "regularization",
            })
        # Faults-in-the-loop: one cell per inject lambda. Only the 'fresh'
        # fault_state_mode is run (accumulate dropped — fresh is the recommended
        # augmentation mode; accumulate overfits a single realization).
        for inj_lam in inject_lambdas:
            tag = _make_tag(
                category=5, lam=inj_lam, inject=True, state="fresh", seed=seed
            )
            cells.append({
                "category": 5,
                "tag": tag,
                "config_key": _config_key(
                    category=5, lam=inj_lam, inject=True, state="fresh"
                ),
                "seed": seed,
                "lam": inj_lam,
                "inject": True,
                "state": "fresh",
                "fault_aware": "regularization",
            })

    # Category 8 — ste_inject (UNVALIDATED)
    if include_ste:
        for seed in seeds:
            tag = _make_tag(category=8, lam=None, inject=False, state="fresh", seed=seed)
            cells.append({
                "category": 8,
                "tag": tag,
                "config_key": _config_key(category=8, lam=None, inject=False, state="fresh"),
                "seed": seed,
                "lam": None,
                "inject": True,   # ste_inject always injects
                "state": "fresh",
                "fault_aware": "ste_inject",
            })

    return cells


# ---------------------------------------------------------------------------
# Per-cell argv builders
# ---------------------------------------------------------------------------

def _train_argv(
    *,
    cfg_path: Path,
    cell: dict,
    base_stem: str,
    save_root: Path,
    epochs: int,
    train_lr: float,
    beta: float,
    protection_layers: list[int],
    wdb_args: list[str],
    crit_tok: str,
    fault_aware_criterion: str,
    fault_aware_hinge_b: float,
    base_checkpoint: str | None = None,
) -> list[str]:
    """Build the argv list for the TRAIN phase of one cell.

    We do NOT include base_overrides (rt_error curve + loops) here — those
    belong to the test/eval phase.  Protection is added explicitly. The
    fault-aware criterion is embedded as a NAME PREFIX (``<crit_tok>__``) and a
    save_dir PATH-LEVEL segment (``save_root/<crit_tok>/<tag>``) so the leaf tag
    — which cat6 parses — stays criterion-free, while runs/checkpoints don't
    collide across criteria that share a --save-root.

    ``base_checkpoint`` overrides the pretrained BNN that fault-aware training
    warm-starts from (e.g. a CEL-trained baseline). ``None`` → the config's
    ``model.checkpoint`` (the default MHL baseline).
    """
    tag = cell["tag"]
    exp_name = f"{base_stem}__{crit_tok}__{tag}_train"
    save_dir = str(save_root / crit_tok / tag)

    argv = [
        "--config", str(cfg_path),
        # mode + fault-aware training
        "--override", "training.mode=train",
        "--override", f"training.fault_aware={cell['fault_aware']}",
    ]
    # Optional warm-start base override (e.g. CEL-trained baseline). Mode stays
    # strict — both MHL and CEL baselines are full BNN state_dicts.
    if base_checkpoint is not None:
        argv += [
            "--override", f"model.checkpoint={base_checkpoint}",
            "--override", "model.checkpoint_mode=strict",
        ]
    argv += [
        # fault-aware loss criterion (cat5/cat8).
        "--override", f"training.fault_aware_criterion={fault_aware_criterion}",
        "--override", f"training.fault_aware_hinge_b={fault_aware_hinge_b}",
        # regularizer parameters (ignored by ste_inject but harmless).
        # NOTE: reg is a sub-section of training in the YAML, so the override
        # path is training.reg.* not reg.* — the loader traverses the raw dict.
        "--override", f"training.reg.lambda={cell['lam'] if cell['lam'] is not None else 0.0}",
        "--override", f"training.reg.beta={beta}",
        "--override", f"training.reg.inject_faults={str(cell['inject']).lower()}",
        "--override", f"training.fault_state_mode={cell['state']}",
        # CRITICAL: override lr to prevent divergence (default 1.0 in rtm configs)
        "--override", f"training.lr={train_lr}",
        "--override", f"training.epochs={epochs}",
        # save_dir: criterion path segment + unique tag → no collision across
        # seeds/lambdas/criteria.
        "--override", f"training.save_dir={save_dir}",
        # experiment identity
        "--override", f"experiment.seed={cell['seed']}",
        "--override", f"experiment.name={exp_name}",
        # protection (must match the test phase)
        "--override", "fault.protection.policy=custom",
        "--override", f"fault.protection.layers={json_list(protection_layers)}",
    ]
    argv += wdb_args
    return argv


def _test_argv(
    *,
    cfg_path: Path,
    cell: dict,
    base_stem: str,
    save_root: Path,
    curve: list[float],
    loops: int,
    protection_layers: list[int],
    wdb_args: list[str],
    crit_tok: str,
) -> list[str]:
    """Build the argv list for the TEST phase of one cell.

    Mirrors the train phase's criterion encoding: name prefix ``<crit_tok>__``
    and the checkpoint loaded from the criterion path segment
    ``save_root/<crit_tok>/<tag>/model.pt``.
    """
    tag = cell["tag"]
    exp_name = f"{base_stem}__{crit_tok}__{tag}_test"
    checkpoint = str(save_root / crit_tok / tag / "model.pt")

    argv = [
        "--config", str(cfg_path),
    ]
    # base_overrides: rt_error curve + loops + protection
    argv += base_overrides(
        curve=curve,
        loops=loops,
        protection_layers=protection_layers,
    )
    argv += [
        # Load the trained checkpoint
        "--override", f"model.checkpoint={checkpoint}",
        # Evaluate WITHOUT encoder — regularized model replaces endlen
        "--override", "fault.weight_encoder=null",
        # experiment identity
        "--override", f"experiment.seed={cell['seed']}",
        "--override", f"experiment.name={exp_name}",
    ]
    argv += wdb_args
    return argv


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _mean_std(values: list[float]) -> tuple[float, float]:
    """Population mean and sample std-dev (ddof=1). Returns (mean, 0.0) for n=1."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    variance = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(variance)


def _fmt_val(v: float | None, width: int = 6) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"


def _write_outputs(
    out_dir: Path,
    results: list[dict],
    curve: list[float],
    cfg_path: Path,
    args_ns,
) -> None:
    """Write Markdown + CSV tables to out_dir.

    Table 1 (per-config): columns = rt_error values + clean_acc; rows = config_key.
    Table 2 (per-seed/per-config): raw per-seed rows for debugging variance.
    """
    # Gather unique config keys in insertion order
    seen_keys: list[str] = []
    key_cells: dict[str, list[dict]] = {}
    for r in results:
        ck = r["config_key"]
        if ck not in seen_keys:
            seen_keys.append(ck)
            key_cells[ck] = []
        key_cells[ck].append(r)

    # ------------------------------------------------------------------
    # Aggregate: mean±std per config_key per rt_error, across seeds
    # ------------------------------------------------------------------
    agg: dict[str, dict] = {}
    for ck, rows in key_cells.items():
        clean_accs = [
            r["baseline_clean_accuracy"]
            for r in rows
            if r["baseline_clean_accuracy"] is not None
        ]
        agg[ck] = {
            "clean_mean": sum(clean_accs) / len(clean_accs) if clean_accs else None,
            "rt": {},
        }
        for rt in curve:
            means = [
                r["rt_curve"][rt]["mean"]
                for r in rows
                if rt in r.get("rt_curve", {})
            ]
            agg[ck]["rt"][rt] = _mean_std(means) if means else (None, None)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    md_path = out_dir / "regularizer_summary.md"
    md_lines: list[str] = []
    md_lines.append("# Run-length regularizer comparison (categories 5 & 8)")
    md_lines.append("")
    md_lines.append(f"- config: `{cfg_path.name}`")
    md_lines.append(f"- seeds: {args_ns.seeds}")
    md_lines.append(f"- lambdas: {args_ns.lambdas}")
    md_lines.append(f"- epochs: {args_ns.epochs}")
    md_lines.append(f"- train_lr: {args_ns.train_lr}  ← divergence guard (default cfg lr=1.0)")
    md_lines.append(f"- rt_curve: {args_ns.rt_curve}")
    md_lines.append(f"- loops: {args_ns.loops}")
    md_lines.append(f"- protection_layers: {args_ns.protection_layers}")
    if args_ns.include_ste:
        md_lines.append("- **cat8 (ste_inject): UNVALIDATED** — included via --include-ste")
    md_lines.append("")

    # Table: mean±std across seeds per rt_error
    md_lines.append("## Mean±std accuracy (across seeds) vs rt_error  [%]")
    md_lines.append("")
    md_lines.append("Columns: clean (no fault), then mean accuracy at each rt_error.")
    md_lines.append("`lambda=0.0` row is the CONTROL (fine-tuning only, no regularizer).")
    md_lines.append("")

    rt_cols = sorted(curve)
    header_parts = ["config_key", "clean_acc", "n_seeds"] + [f"rt={r:g}" for r in rt_cols]
    md_lines.append("| " + " | ".join(header_parts) + " |")
    md_lines.append("|" + "|".join(["---"] * len(header_parts)) + "|")
    for ck in seen_keys:
        a = agg[ck]
        n = len(key_cells[ck])
        clean_s = _fmt_val(a["clean_mean"])
        row_parts = [ck, clean_s, str(n)]
        for rt in rt_cols:
            m, s = a["rt"].get(rt, (None, None))
            if m is None:
                row_parts.append("—")
            else:
                row_parts.append(f"{m:.2f}±{s:.2f}")
        md_lines.append("| " + " | ".join(row_parts) + " |")
    md_lines.append("")

    # Per-seed raw table
    md_lines.append("## Per-seed raw accuracy [%]")
    md_lines.append("")
    raw_header = ["config_key", "seed", "status", "clean_acc"] + [f"rt={r:g}" for r in rt_cols]
    md_lines.append("| " + " | ".join(raw_header) + " |")
    md_lines.append("|" + "|".join(["---"] * len(raw_header)) + "|")
    for r in results:
        row_parts = [
            r["config_key"],
            str(r["seed"]),
            r.get("test_status", "—"),
            _fmt_val(r.get("baseline_clean_accuracy")),
        ]
        for rt in rt_cols:
            m = r.get("rt_curve", {}).get(rt, {}).get("mean")
            row_parts.append(_fmt_val(m))
        md_lines.append("| " + " | ".join(row_parts) + " |")
    md_lines.append("")

    md_path.write_text("\n".join(md_lines))

    # ------------------------------------------------------------------
    # CSV (per-seed rows for further analysis)
    # ------------------------------------------------------------------
    csv_path = out_dir / "regularizer_per_seed.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_key", "seed", "category", "lambda", "inject", "state",
                    "test_status", "baseline_clean_accuracy"]
                   + [f"rt_{r:g}_mean" for r in rt_cols]
                   + [f"rt_{r:g}_min" for r in rt_cols]
                   + [f"rt_{r:g}_max" for r in rt_cols])
        for r in results:
            row = [
                r["config_key"],
                r["seed"],
                r["category"],
                r.get("lam", ""),
                r.get("inject", ""),
                r.get("state", ""),
                r.get("test_status", ""),
                r.get("baseline_clean_accuracy", ""),
            ]
            for rt in rt_cols:
                row.append(r.get("rt_curve", {}).get(rt, {}).get("mean", ""))
            for rt in rt_cols:
                row.append(r.get("rt_curve", {}).get(rt, {}).get("min", ""))
            for rt in rt_cols:
                row.append(r.get("rt_curve", {}).get(rt, {}).get("max", ""))
            w.writerow(row)

    # ------------------------------------------------------------------
    # Aggregated CSV (mean±std per config_key)
    # ------------------------------------------------------------------
    agg_csv_path = out_dir / "regularizer_aggregated.csv"
    with open(agg_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_key", "n_seeds", "clean_acc_mean"]
                   + [f"rt_{r:g}_mean" for r in rt_cols]
                   + [f"rt_{r:g}_std" for r in rt_cols])
        for ck in seen_keys:
            a = agg[ck]
            n = len(key_cells[ck])
            row = [ck, n, a["clean_mean"] if a["clean_mean"] is not None else ""]
            for rt in rt_cols:
                m, _ = a["rt"].get(rt, (None, None))
                row.append(m if m is not None else "")
            for rt in rt_cols:
                _, s = a["rt"].get(rt, (None, None))
                row.append(s if s is not None else "")
            w.writerow(row)

    print(f"  Markdown : {md_path}")
    print(f"  CSV (raw): {csv_path}")
    print(f"  CSV (agg): {agg_csv_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Base YAML config passed to the NetDrift runner.")
    p.add_argument("--seeds", nargs="+", type=int, default=[707, 1, 42],
                   help="RNG seeds. Multiple seeds are the whole point of this driver. "
                        "Default: 707 1 42")
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[0.0, 0.01, 0.05, 0.1],
                   help="Regularizer lambda values for the core cat5 sweep "
                        "(no fault injection). lambda=0.0 is the CONTROL. "
                        "Default: 0.0 0.01 0.05 0.1")
    p.add_argument("--inject-lambdas", nargs="+", type=float, dest="inject_lambdas",
                   default=[0.05],
                   help="Lambda values for the faults-in-the-loop cat5 variant "
                        "(inject_faults=true, fault_state_mode=fresh). One cell "
                        "per value. Default: 0.05")
    p.add_argument("--beta", type=float, default=4.0,
                   help="Regularizer beta (tanh sharpness). Default: 4.0")
    p.add_argument("--fault-aware-criterion", dest="fault_aware_criterion",
                   default="hinge", choices=["hinge", "cross_entropy"],
                   help="Loss criterion for fault-aware training (cat5+cat8). "
                        "Embedded as a name prefix + save_dir path segment so "
                        "runs/checkpoints don't collide across criteria sharing "
                        "a --save-root. Default: hinge")
    p.add_argument("--fault-aware-hinge-b", dest="fault_aware_hinge_b",
                   type=float, default=128.0,
                   help="b parameter for the fault-aware hinge loss. "
                        "Ignored when --fault-aware-criterion=cross_entropy. "
                        "Default: 128.0")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs per cell. Default: 10")
    p.add_argument(
        "--train-lr", type=float, default=0.001, dest="train_lr",
        help="Learning rate for the train phase. MUST be << 1.0 — the RTM "
             "config's default lr=1.0 causes divergence in fault-aware training. "
             "Default: 0.001",
    )
    p.add_argument("--rt-curve", nargs="+", type=float, dest="rt_curve",
                   default=DEFAULT_RT_ERROR_CURVE,
                   help="rt_error values for the test-phase sweep curve. "
                        f"Default: {DEFAULT_RT_ERROR_CURVE}")
    p.add_argument("--loops", type=int, default=DEFAULT_LOOPS,
                   help=f"Inference iterations per rt_error in test. Default: {DEFAULT_LOOPS}")
    p.add_argument("--protection-layers", nargs="+", type=int,
                   dest="protection_layers",
                   default=DEFAULT_PROTECTION_LAYERS,
                   help="Unprotected layer indices (1-based, custom policy). "
                        f"Default: {DEFAULT_PROTECTION_LAYERS}")
    p.add_argument("--include-ste", action="store_true",
                   help="Also run category 8 (ste_inject). "
                        "UNVALIDATED — treat output as exploratory.")
    p.add_argument(
        "--save-root", default="runs/sweeps/reg_checkpoints",
        help="Parent dir for per-cell save_dir (where model.pt lands). "
             "save_dir = <save-root>/<tag>/ — unique per config_stem+seed+lambda. "
             "Default: runs/sweeps/reg_checkpoints",
    )
    p.add_argument(
        "--base-checkpoint", dest="base_checkpoint", default=None, metavar="PATH",
        help="Pretrained BNN that fault-aware training (cat5/cat8) warm-starts "
             "from. Default: the config's model.checkpoint (the MHL baseline). "
             "Pass a CEL-trained baseline (e.g. models/w1a1_cel/<model>/"
             "model_best.pt) for an all-CEL pipeline.",
    )
    p.add_argument("--wandb-project", default=None,
                   help="W&B project. Omit for a local-only sweep.")
    p.add_argument("--wandb-entity", default=None,
                   help="Optional W&B entity (team/org).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print every train+test cell for all seeds and exit; run nothing.")
    args = p.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    base_stem = cfg_path.stem
    # save_root: the --save-root arg is relative to cwd; we resolve it so the
    # runner (which may have a different cwd) always gets an absolute path.
    # We embed base_stem so two simultaneous sweeps on different configs share
    # --save-root without colliding.
    save_root = (Path(args.save_root) / base_stem).resolve()

    runner_out_dir = output_dir_from_cfg(cfg_path)

    # Per-cell W&B args carry the comparison-DB category (coarse mode) and a
    # subcategory (exact setting combo) so runs group by both. cat5 = run-length
    # regularizer, cat8 = ste_inject. Both train+test runs of a cell share the
    # labels (the *_test runs are the comparison cells). subcategory uses the
    # cell's config_key (e.g. lam0p05, lam0p05_inj-fresh, ste).
    _REG_CAT_LABEL = {5: "cat5_regularizer", 8: "cat8_ste_inject"}

    # Fault-aware criterion token (uniform across this invocation's cells).
    crit_tok = crit_token(args.fault_aware_criterion, args.fault_aware_hinge_b)

    def _wdb_for(cell: dict) -> list[str]:
        cat = _REG_CAT_LABEL.get(cell["category"])
        # subcategory includes the criterion token so CE vs hinge runs group
        # separately within a category in the W&B UI.
        sub = f"{cat}_{cell['config_key']}_{crit_tok}" if cat else None
        return wandb_args(args.wandb_project, args.wandb_entity, cat, sub)

    cells = build_cells(
        lambdas=args.lambdas,
        seeds=args.seeds,
        include_ste=args.include_ste,
        inject_lambdas=args.inject_lambdas,
    )
    total = len(cells)
    # Each cell is 2 phases (train + test).
    total_phases = total * 2

    n_cat5 = sum(1 for c in cells if c["category"] == 5)
    n_cat8 = sum(1 for c in cells if c["category"] == 8)

    print("=" * 72)
    print("NetDrift regularizer sweep — categories 5" + (" & 8" if args.include_ste else ""))
    print("=" * 72)
    print(f"  config          : {cfg_path}")
    print(f"  seeds           : {args.seeds}")
    print(f"  lambdas (cat5)  : {args.lambdas}")
    print(f"  inject lambda   : 0.05 (fixed)")
    print(f"  beta            : {args.beta}")
    print(f"  epochs          : {args.epochs}")
    print(f"  train_lr        : {args.train_lr}  (overrides rtm default 1.0)")
    print(f"  rt_curve        : {args.rt_curve}")
    print(f"  loops           : {args.loops}")
    print(f"  protection      : custom, layers={args.protection_layers}")
    print(f"  save_root       : {save_root}")
    print(f"  wandb           : {args.wandb_project or 'DISABLED'}")
    print(f"  cat5 cells      : {n_cat5}  ({len(args.lambdas)} lambdas + 2 inject) × {len(args.seeds)} seeds")
    if args.include_ste:
        print(f"  cat8 cells      : {n_cat8}  (ste_inject, UNVALIDATED) × {len(args.seeds)} seeds")
    print(f"  total cells     : {total}  ({total_phases} phases = {total} train + {total} test)")
    print()

    if args.dry_run:
        for i, cell in enumerate(cells, 1):
            tag = cell["tag"]
            save_dir = str(save_root / crit_tok / tag)
            checkpoint = str(save_root / crit_tok / tag / "model.pt")
            train_argv = _train_argv(
                cfg_path=cfg_path, cell=cell, base_stem=base_stem,
                save_root=save_root, epochs=args.epochs, train_lr=args.train_lr,
                beta=args.beta, protection_layers=args.protection_layers,
                wdb_args=_wdb_for(cell), crit_tok=crit_tok,
                fault_aware_criterion=args.fault_aware_criterion,
                fault_aware_hinge_b=args.fault_aware_hinge_b,
                base_checkpoint=args.base_checkpoint,
            )
            test_argv = _test_argv(
                cfg_path=cfg_path, cell=cell, base_stem=base_stem,
                save_root=save_root, curve=args.rt_curve, loops=args.loops,
                protection_layers=args.protection_layers, wdb_args=_wdb_for(cell),
                crit_tok=crit_tok,
            )
            cat_label = f"cat{cell['category']}"
            unvalidated = "  [UNVALIDATED]" if cell["category"] == 8 else ""
            print(f"[{i:3d}/{total}] {cat_label} seed={cell['seed']:5d} "
                  f"config_key={cell['config_key']}{unvalidated}")
            print(f"         TRAIN  {' '.join(train_argv)}")
            print(f"         TEST   {' '.join(test_argv)}")
        print()
        print(f"Would write tables to: runs/sweeps/<ts>_regularizer/")
        return 0

    # -------------------------------------------------------------------------
    # Live run
    # -------------------------------------------------------------------------
    runner_main = import_runner_main()
    out_dir = new_sweep_out_dir("regularizer")
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_t0 = time.perf_counter()
    results: list[dict] = []

    for cell_idx, cell in enumerate(cells, 1):
        tag = cell["tag"]
        cat_label = f"cat{cell['category']}"
        unvalidated_note = "  [UNVALIDATED]" if cell["category"] == 8 else ""
        bar = "=" * 72
        print()
        print(bar)
        print(f"[cell {cell_idx}/{total}] {cat_label}  seed={cell['seed']}  "
              f"config_key={cell['config_key']}{unvalidated_note}")
        print(bar)

        # ---- Phase 1: TRAIN ----
        train_argv = _train_argv(
            cfg_path=cfg_path, cell=cell, base_stem=base_stem,
            save_root=save_root, epochs=args.epochs, train_lr=args.train_lr,
            beta=args.beta, protection_layers=args.protection_layers,
            wdb_args=_wdb_for(cell), crit_tok=crit_tok,
            fault_aware_criterion=args.fault_aware_criterion,
            fault_aware_hinge_b=args.fault_aware_hinge_b,
            base_checkpoint=args.base_checkpoint,
        )
        print(f"  Phase 1 (train): fault_aware={cell['fault_aware']}  "
              f"criterion={args.fault_aware_criterion}  lr={args.train_lr}  "
              f"epochs={args.epochs}  save_dir={save_root / crit_tok / tag}")
        t_train = time.perf_counter()
        train_status, train_err = run_cell(runner_main, train_argv)
        train_elapsed = time.perf_counter() - t_train
        if train_err:
            print(f"  !! TRAIN failed: {train_err}")
        print(f"  => train {train_status}  ({train_elapsed:.1f}s)")

        # ---- Phase 2: TEST ----
        test_argv = _test_argv(
            cfg_path=cfg_path, cell=cell, base_stem=base_stem,
            save_root=save_root, curve=args.rt_curve, loops=args.loops,
            protection_layers=args.protection_layers, wdb_args=_wdb_for(cell),
            crit_tok=crit_tok,
        )
        test_exp_name = f"{base_stem}__{crit_tok}__{tag}_test"
        print(f"  Phase 2 (test):  encoder=null  rt_curve={args.rt_curve}  "
              f"exp_name={test_exp_name}")
        t_test = time.perf_counter()
        test_status, test_err = run_cell(runner_main, test_argv)
        test_elapsed = time.perf_counter() - t_test
        if test_err:
            print(f"  !! TEST failed: {test_err}")

        # Harvest metrics from the test-phase summary
        summary_path = latest_summary(runner_out_dir, test_exp_name)
        harvested = harvest_summary(summary_path)
        baseline_clean = harvested["baseline_clean_accuracy"]
        rt_curve_data = harvested["rt_curve"]

        if rt_curve_data:
            curve_str = "  ".join(
                f"{rt:g}:{v['mean']:.1f}" for rt, v in sorted(rt_curve_data.items())
            )
            print(f"  => test {test_status}  clean={baseline_clean}  "
                  f"rt_curve(mean) [{curve_str}]  ({test_elapsed:.1f}s)")
        else:
            print(f"  => test {test_status}  clean={baseline_clean}  "
                  f"(no rt_curve harvested)  ({test_elapsed:.1f}s)")

        record = {
            "cell_idx": cell_idx,
            "category": cell["category"],
            "tag": tag,
            "config_key": cell["config_key"],
            "seed": cell["seed"],
            "lam": cell.get("lam"),
            "inject": cell["inject"],
            "state": cell["state"],
            "fault_aware": cell["fault_aware"],
            "train_status": train_status,
            "train_error": train_err,
            "train_elapsed_s": round(train_elapsed, 1),
            "test_status": test_status,
            "test_error": test_err,
            "test_elapsed_s": round(test_elapsed, 1),
            "summary_path": str(summary_path) if summary_path else None,
            "baseline_clean_accuracy": baseline_clean,
            "baseline_endlen_accuracy": harvested.get("baseline_endlen_accuracy"),
            "rt_curve": rt_curve_data,
        }
        results.append(record)

        # Persist manifest after every cell so a crash leaves a partial record.
        write_manifest(out_dir, {
            "config": str(cfg_path),
            "seeds": args.seeds,
            "lambdas": args.lambdas,
            "beta": args.beta,
            "epochs": args.epochs,
            "train_lr": args.train_lr,
            "rt_curve": args.rt_curve,
            "loops": args.loops,
            "protection_layers": args.protection_layers,
            "include_ste": args.include_ste,
            "save_root": str(save_root),
            "wandb_project": args.wandb_project,
            "results": results,
        })

    # -------------------------------------------------------------------------
    # Write output tables
    # -------------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Writing output tables ...")
    _write_outputs(out_dir, results, args.rt_curve, cfg_path, args)
    print()

    n_ok = sum(
        1 for r in results
        if r["train_status"] == "ok" and r["test_status"] == "ok"
    )
    n_fail = total - n_ok
    elapsed_total = time.perf_counter() - sweep_t0
    print(f"Sweep done: {n_ok}/{total} cells fully ok, {n_fail} with errors  "
          f"({elapsed_total:.1f}s total)")
    print(f"Output dir: {out_dir}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
