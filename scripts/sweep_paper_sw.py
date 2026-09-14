#!/usr/bin/env python
"""Paper software-category sweep: one command per robustness category.

The sibling of ``sweep_paper_runs.py``. That driver holds the software technique
fixed and sweeps the racetrack LAYOUT; this one holds the layout fixed (COL) and
sweeps the SOFTWARE CATEGORY. Each category is a separate invocation so the six
can run concurrently, one per tmux pane::

    python scripts/sweep_paper_sw.py --category baseline      --gpu-num 0 ...
    python scripts/sweep_paper_sw.py --category endlen-recal  --gpu-num 1 ...
    python scripts/sweep_paper_sw.py --category reg           --gpu-num 2 ...
    python scripts/sweep_paper_sw.py --category reg-recal     --gpu-num 3 ...
    python scripts/sweep_paper_sw.py --category ste           --gpu-num 4 ...
    python scripts/sweep_paper_sw.py --category reg-ste-recal --gpu-num 5 ...

x2 models = 12 panes.

Everything here is ``training.mode=test``. The fault-aware checkpoints are
INPUTS, produced beforehand by the six prep fine-tunes in
[`paper_sw_prep.md`](../docs/paper_sw_prep.md) — nothing in this driver trains
weights. The two recalibrating categories DO run a short BN/Scale re-fit per
cell, which is a downstream re-fit, not weight training: binary signs never move.

The categories
--------------
========================  ====================  ====================================
``--category``            W&B category          what it is
========================  ====================  ====================================
``baseline``              cat1_baseline         plain w1a1 BNN, no technique
``endlen-recal``          cat4_endlen_recal     endlen encoder + BN/Scale re-fit
``reg``                   cat5_regularizer      run-length regularizer fine-tune
``reg-recal``             cat6_reg_recal        cat5 weights + BN/Scale re-fit
``ste``                   cat8_ste_inject       STE fault-injection fine-tune
``reg-ste-recal``         cat68_reg_ste_recal   regularizer AND STE injection,
                                                + BN/Scale re-fit
========================  ====================  ====================================

``reg-ste-recal`` is the new arm. Its checkpoint is trained with
``fault_aware=regularization`` AND ``reg.inject_faults=true`` — the one flag
combination that gets BOTH the run-length penalty and faults in the forward.
(``fault_aware=ste_inject`` with a lambda does NOT: ``use_reg`` in
``training/faultaware.py`` requires the mode to literally be ``regularization``,
so the penalty is silently dropped.) The ``-recal`` half is this driver's
per-cell BN/Scale re-fit, exactly as ``reg-recal`` gets it.

Axes per category
-----------------
* seed   707 | 808 | 909 — FAULT RNG over ONE checkpoint per category, not three
                           separately trained models. Same convention as
                           ``sweep_paper_runs.py``.

rt_error is NOT a cell axis: the whole curve is swept inside one runner
invocation, which logs one W&B run per rt_error. So a category is 3 cells.

Held fixed (this is the software comparison, so everything structural is pinned)
-------------------------------------------------------------------------------
``storage.layout=col`` + ``base_layout=col``, ``rt_size=64``,
``kernel_mapping=row``, ``edge_mode=saturate``, ``fault.protection.policy=custom``
(all but the stem conv and the classifier exposed — the apples-to-apples recipe
from ``docs/category_walkthrough.md``, and the protection the fault-aware
checkpoints were trained under).

COL is the deployment view every fault-aware checkpoint here was trained for:
``run_length_penalty`` and the STE-injected forward both read the
racetrack-aligned view named by ``storage.layout``, so ROW and COL are different
adjacencies and a ROW-trained cat5 would be a layout mismatch, not a technique.
``--ckpt-*`` defaults therefore resolve ``cat5_col`` / ``cat8_col`` / ``cat68_col``.

Output tree
-----------
``runs/paper-runs/paper-sw_<model>/<category>/<cell>/<timestamp>/`` — the
category is the W&B category and the cell is the W&B subcategory, and
``run.py::_setup_run_dir`` mirrors both into the directory path.

Preflight (do this before burning GPU hours)
--------------------------------------------
    python scripts/sweep_paper_sw.py --category reg-ste-recal --print-config | less
        # resolved dataclass per cell — proves every override actually landed,
        # which --dry-run (argv only) does not.
    python scripts/sweep_paper_sw.py --category reg-ste-recal --dry-run
    python scripts/sweep_paper_sw.py --category reg-ste-recal --limit 1   # smoke run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Put the scripts/ dir on sys.path so the sibling modules are importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparison_common import (  # noqa: E402
    REPO_ROOT,
    harvest_summary,
    import_runner_main,
    json_list,
    new_sweep_out_dir,
    prefer_best_checkpoint,
    rt_error_list_override,
    run_cell,
    wandb_args,
    write_manifest,
)

# The topology table is single-sourced in sweep_paper_runs: adding a model there
# (base YAML, protection layer ids, legal kernel mappings, checkpoint root) makes
# it available to BOTH paper sweeps. Only the experiment NAME differs, and this
# driver derives that from the model key rather than reusing the layout sweep's.
from sweep_paper_runs import MODELS, prot_token  # noqa: E402

DEFAULT_RT_CURVE = [1e-04, 4.55e-05, 1e-05]
DEFAULT_SEEDS = [707, 808, 909]
DEFAULT_LOOPS = 100
DEFAULT_AP_POSITION = 0
DEFAULT_OUTPUT_DIR = "runs/paper-runs/"
DEFAULT_WANDB_PROJECT = "netdrift-paper-runs-sw"
DEFAULT_LAYOUT = "col"
DEFAULT_MODEL = "vgg7_cifar10"

# category -> what makes it that category.
#
#   token      W&B category + directory segment. Matches the vocabulary in
#              scripts/wandb_tag_categories.py so live-tagged and driver-tagged
#              runs land in the same groups.
#   ckpt       which --ckpt-* flag supplies the weights.
#   recal      None, or the `training.recalibrate.on` trigger. `endlen` fires
#              because an encoder ran; `always` is needed where there is no
#              encoder to trigger it.
#   overrides  category-specific runner overrides, applied before passthrough.
CATEGORIES: dict[str, dict[str, Any]] = {
    "baseline": {
        "token": "cat1_baseline",
        "ckpt": "base",
        "recal": None,
        "overrides": ["fault.weight_encoder=null"],
    },
    "endlen-recal": {
        "token": "cat4_endlen_recal",
        "ckpt": "base",
        "recal": "endlen",
        "overrides": [
            "fault.weight_encoder=endlen",
            "fault.weight_encoder_mode=once",
            # 1.0/1.0 = vanilla (unbudgeted) endlen; budgeted endlen is cat3.
            "fault.global_bitflip_budget=1.0",
            "fault.local_bitflip_budget=1.0",
        ],
    },
    "reg": {
        "token": "cat5_regularizer",
        "ckpt": "cat5",
        "recal": None,
        "overrides": ["fault.weight_encoder=null"],
    },
    "reg-recal": {
        "token": "cat6_reg_recal",
        "ckpt": "cat5",
        "recal": "always",
        "overrides": ["fault.weight_encoder=null"],
    },
    "ste": {
        "token": "cat8_ste_inject",
        "ckpt": "cat8",
        "recal": None,
        "overrides": ["fault.weight_encoder=null"],
    },
    "reg-ste-recal": {
        "token": "cat68_reg_ste_recal",
        "ckpt": "cat68",
        "recal": "always",
        "overrides": ["fault.weight_encoder=null"],
    },
}

# Which --ckpt-* flags exist, and the conventional path each defaults to. Every
# fault-aware checkpoint is per deployment VIEW (see the module docstring), so
# each carries a {layout} placeholder; `base` alone is view-independent.
CKPT_DEFAULTS = {
    "base": "{root}/model_best.pt",
    "cat5": "{root}/cat5_{layout}/model.pt",
    "cat8": "{root}/cat8_{layout}/model.pt",
    "cat68": "{root}/cat68_{layout}/model.pt",
}


def experiment_name(model: str) -> str:
    """``paper-sw_<model>`` — the model keys already read ``<arch>_<dataset>``."""
    return f"paper-sw_{model}"


def cell_checkpoint(cell: dict[str, Any], args: argparse.Namespace) -> str:
    """The checkpoint file this cell will actually load.

    The single funnel for that question, so ``cell_overrides`` (what the runner
    is told), ``check_checkpoints`` (the pre-launch guard) and ``collect`` (the
    recorded provenance) can never disagree.

    ``prefer_best_checkpoint`` upgrades a resolved ``model.pt`` to the sibling
    ``model_best.pt`` when one exists. That is disk-dependent, which is exactly
    why ``collect`` records the answer per row rather than trusting the
    pre-launch check to still describe the run.
    """
    spec = CATEGORIES[cell["category"]]
    template = getattr(args, f"ckpt_{spec['ckpt']}")
    resolved = template.format(root=MODELS[args.model]["ckpt_root"],
                               layout=args.layout, seed=cell["seed"])
    return prefer_best_checkpoint(resolved)


def build_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Enumerate every cell of one category. Pure — no side effects, no runner."""
    return [
        {
            "category": args.category,
            "subcategory": f"{prot_token('custom', args.model)}_seed{seed}",
            "seed": seed,
        }
        for seed in args.seeds
    ]


def cell_overrides(cell: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """The full ``--override`` list for one cell (config-order, deterministic)."""
    spec = CATEGORIES[cell["category"]]
    model_spec = MODELS[args.model]

    ov = [
        f"experiment.name={experiment_name(args.model)}",
        f"experiment.output_dir={args.output_dir}",
        f"experiment.seed={cell['seed']}",
        f"model.checkpoint={cell_checkpoint(cell, args)}",
        "model.checkpoint_mode=strict",
        f"storage.layout={args.layout}",
        f"storage.base_layout={args.layout}",
        f"storage.rt_size={args.rt_size}",
        f"storage.kernel_mapping={args.kernel_mapping}",
        f"fault.rt_error={rt_error_list_override(args.rt_curve)}",
        "fault.mitigations=[]",
        f"fault.edge_mode={args.edge_mode}",
        # Protection is pinned to `custom` for every category: all but the stem
        # conv and the classifier exposed. That is the apples-to-apples recipe,
        # and it is the protection the fault-aware checkpoints were trained
        # under — evaluating them under a different one would compare the
        # technique against its own train/test mismatch.
        "fault.protection.policy=custom",
        f"fault.protection.layers={json_list(model_spec['unprotected_custom'])}",
        "training.mode=test",
        "training.fault_aware=none",
        f"training.loops={args.loops}",
    ]
    ov += spec["overrides"]
    if args.ap_position is not None:
        # COL is a dense layout, so an absolute access-port index is accepted
        # (RTMConfig rejects one only for block/units/polarity, which resolve the
        # port per bucket). Matching sweep_paper_runs' default keeps the two
        # paper sweeps comparable on this axis.
        ov.append(f"fault.ap_position={args.ap_position}")
    if spec["recal"] is not None:
        ov += [
            "training.recalibrate.enabled=true",
            f"training.recalibrate.on={spec['recal']}",
            "training.recalibrate.bn_stats=true",
            "training.recalibrate.tune_affine=true",
            f"training.recalibrate.epochs={args.recal_epochs}",
            f"training.recalibrate.lr={args.recal_lr}",
            f"training.criterion={args.criterion}",
            f"training.hinge_b={args.hinge_b}",
        ]
    if args.gpu_num is not None:
        # Must live here, not be appended at launch: --print-commands and
        # --print-config both go through cell_overrides, and a printed command
        # missing its device pin is exactly the one that gets pasted into tmux.
        ov.append(f"gpu_num={args.gpu_num}")
    # Passthrough LAST so it can override anything above (e.g. data.num_workers,
    # data.test_batch_size — throughput levers that differ per dataset).
    ov += list(args.override)
    return ov


def cell_argv(cell: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Runner argv for one cell."""
    argv = ["--config", args.config]
    for o in cell_overrides(cell, args):
        argv += ["--override", o]
    argv += ["--metrics", args.metrics]
    argv += wandb_args(args.wandb_project or None, args.wandb_entity,
                       category=CATEGORIES[cell["category"]]["token"],
                       subcategory=cell["subcategory"])
    return argv


def shell_command(cell: dict[str, Any], args: argparse.Namespace) -> str:
    """A standalone shell command for one cell (for GNU parallel / manual reruns)."""
    import shlex
    return " ".join(["python", "netdrift_run.py"]
                    + [shlex.quote(a) for a in cell_argv(cell, args)])


def cell_run_dir(cell: dict[str, Any], args: argparse.Namespace) -> Path:
    """``<output_dir>/<name>/<category token>/<cell>/`` — parent of the timestamps."""
    return (REPO_ROOT / args.output_dir / experiment_name(args.model)
            / CATEGORIES[cell["category"]]["token"] / cell["subcategory"])


def latest_cell_summary(cell: dict[str, Any], args: argparse.Namespace) -> Optional[Path]:
    """Newest ``summary.json`` under this cell's own subtree.

    Deliberately NOT ``comparison_common.latest_summary``: every cell of a model
    shares one ``experiment.name`` (that is what keeps the output tree readable),
    so keying the harvest on the name alone would return one arbitrary cell's
    summary for all of them.
    """
    base = cell_run_dir(cell, args)
    if not base.exists():
        return None
    candidates = list(base.glob("**/summary.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def check_checkpoints(cells: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    """Return the sorted list of missing checkpoint paths (empty == all present)."""
    missing: set[str] = set()
    for cell in cells:
        p = cell_checkpoint(cell, args)
        if not (REPO_ROOT / p).exists() and not Path(p).exists():
            missing.add(p)
    return sorted(missing)


def collect(cells: list[dict[str, Any]], args: argparse.Namespace, out_dir: Path) -> Path:
    """Harvest each cell's newest summary.json into one CSV for this category."""
    rt_cols = [f"rt_{rt}" for rt in args.rt_curve]
    token = CATEGORIES[args.category]["token"]
    csv_path = out_dir / f"paper_sw_{args.model}_{token}_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        # `checkpoint` is appended LAST so every other column keeps its index for
        # readers that go by position.
        w.writerow(["model", "category", "subcategory", "seed", "status",
                    "clean_accuracy", "endlen_accuracy", "endlen_recal_accuracy"]
                   + [f"{c}_mean" for c in rt_cols]
                   + [f"{c}_last" for c in rt_cols]
                   + ["checkpoint"])
        for cell in cells:
            s = latest_cell_summary(cell, args)
            h = harvest_summary(s)
            status = "ok" if s is not None and h["rt_curve"] else "missing"
            row = [args.model, token, cell["subcategory"], cell["seed"], status,
                   h["baseline_clean_accuracy"],
                   h["baseline_endlen_accuracy"],
                   h["baseline_endlen_recal_accuracy"]]
            for rt in args.rt_curve:
                e = h["rt_curve"].get(float(rt))
                row.append(round(e["mean"], 4) if e else "")
            for rt in args.rt_curve:
                e = h["rt_curve"].get(float(rt))
                row.append(round(e["last"], 4) if e else "")
            row.append(cell_checkpoint(cell, args))
            w.writerow(row)
    return csv_path


def print_resolved_configs(cells: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Load each cell's config THROUGH the real loader and dump the result.

    This is the only preflight that proves an override landed on the field you
    meant: ``--dry-run`` shows the argv you typed, not the config the runner
    builds from it. Import-light (no torch), so it runs anywhere.
    """
    sys.path.insert(0, str(REPO_ROOT / "code" / "python"))
    from netdrift.config.loader import load, parse_overrides  # noqa: E402

    for cell in cells:
        cfg = load(args.config, overrides=parse_overrides(cell_overrides(cell, args)))
        print("=" * 78)
        print(f"{CATEGORIES[cell['category']]['token']} / {cell['subcategory']}")
        print("-" * 78)
        print(json.dumps({
            "checkpoint": cfg.model.checkpoint,
            "seed": cfg.experiment.seed,
            "output_dir": cfg.experiment.output_dir,
            "storage": {"layout": cfg.storage.layout,
                        "base_layout": cfg.storage.base_layout,
                        "rt_size": cfg.storage.rt_size,
                        "kernel_mapping": cfg.storage.kernel_mapping},
            "fault": {"rt_error": cfg.fault.rt_error,
                      "edge_mode": cfg.fault.edge_mode,
                      "ap_position": cfg.fault.ap_position,
                      "weight_encoder": cfg.fault.weight_encoder,
                      "weight_encoder_mode": cfg.fault.weight_encoder_mode,
                      "global_bitflip_budget": cfg.fault.global_bitflip_budget,
                      "protection": {"policy": cfg.fault.protection.policy,
                                     "layers": cfg.fault.protection.layers}},
            "training": {"mode": cfg.training.mode,
                         "fault_aware": cfg.training.fault_aware,
                         "loops": cfg.training.loops,
                         "recalibrate": {"enabled": cfg.training.recalibrate.enabled,
                                         "on": cfg.training.recalibrate.on,
                                         "bn_stats": cfg.training.recalibrate.bn_stats,
                                         "tune_affine": cfg.training.recalibrate.tune_affine,
                                         "epochs": cfg.training.recalibrate.epochs},
                         "criterion": cfg.training.criterion},
        }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Split out of ``main`` so tests can parse an argv without running
    the sweep — the alternative is stubbing internals, which tests the stub."""
    p = argparse.ArgumentParser(
        description="Paper software-category sweep (one category per invocation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--category", required=True, choices=list(CATEGORIES),
                   help="Which software category to run. One per tmux pane.")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS),
                   help="Topology/dataset. Selects the base YAML, the protection "
                        "layer ids, the legal kernel mappings and the checkpoint "
                        f"root. Default: {DEFAULT_MODEL}")
    p.add_argument("--config", default=None,
                   help="Base YAML for every cell. Default: the --model entry's.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"experiment.output_dir. Default: {DEFAULT_OUTPUT_DIR}")
    p.add_argument("--rt-curve", nargs="+", type=float, default=DEFAULT_RT_CURVE,
                   help=f"rt_error values swept INSIDE each cell. Default: {DEFAULT_RT_CURVE}")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="Fault-RNG seeds over the ONE checkpoint per category. "
                        f"Default: {DEFAULT_SEEDS}")
    p.add_argument("--loops", type=int, default=DEFAULT_LOOPS,
                   help=f"Inference iterations per rt_error. Default: {DEFAULT_LOOPS}")
    p.add_argument("--layout", default=DEFAULT_LAYOUT, choices=["row", "col"],
                   help="storage.layout (and base_layout) for every cell, and the "
                        "{layout} the fault-aware checkpoint paths resolve to. "
                        f"Default: {DEFAULT_LAYOUT}")
    p.add_argument("--ap-position", type=int, default=DEFAULT_AP_POSITION,
                   help="Fixed access-port index for edge_mode=saturate. Accepted "
                        "on the dense row/col layouts this driver uses. "
                        f"Default: {DEFAULT_AP_POSITION} (low edge). Pass -1 to "
                        "leave it unset (null => mid-wire rt_size//2 - 1).")
    p.add_argument("--edge-mode", default="saturate", choices=["saturate", "random"],
                   help="Racetrack edge model, applied to every cell. Default: saturate")
    p.add_argument("--rt-size", type=int, default=64, help="storage.rt_size. Default: 64")
    p.add_argument("--kernel-mapping", default="row", choices=["row", "col", "clw", "acw"],
                   help="storage.kernel_mapping. Validated against the model's "
                        "legal set (topologies with non-3x3 convs are ROW-only). "
                        "Default: row")
    p.add_argument("--metrics", default="all", choices=["none", "offline", "online", "all"],
                   help="Runner --metrics level. Default: all")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="Extra runner override applied to EVERY cell, after all "
                        "driver-managed ones (so it wins). Repeatable.")
    p.add_argument("--gpu-num", type=int, default=None,
                   help="Pin this category to one CUDA device (sets gpu_num). Give "
                        "each tmux pane its own device.")

    ck = p.add_argument_group(
        "checkpoints (accept {root}, {layout} and {seed} placeholders)")
    for key, default in CKPT_DEFAULTS.items():
        ck.add_argument(f"--ckpt-{key}", default=None,
                        help=f"Checkpoint for the categories that use it. "
                             f"Default: {default}")
    ck.add_argument("--no-check-checkpoints", dest="check_checkpoints",
                    action="store_false",
                    help="Skip the pre-launch existence check on every resolved "
                         "checkpoint path.")
    p.set_defaults(check_checkpoints=True)

    rc = p.add_argument_group("recalibration (endlen-recal / reg-recal / reg-ste-recal)")
    rc.add_argument("--recal-epochs", type=int, default=2,
                    help="training.recalibrate.epochs. Default: 2")
    rc.add_argument("--recal-lr", type=float, default=0.001,
                    help="training.recalibrate.lr. Default: 0.001")
    rc.add_argument("--criterion", default="hinge", choices=["hinge", "cross_entropy"],
                    help="Criterion for the recalibration tune step. Default: hinge")
    rc.add_argument("--hinge-b", type=float, default=128.0,
                    help="b for the hinge criterion. Default: 128.0")

    wb = p.add_argument_group("weights & biases")
    wb.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT,
                    help=f"W&B project. Default: {DEFAULT_WANDB_PROJECT}. "
                         "Pass '' to disable.")
    wb.add_argument("--wandb-entity", default=None, help="W&B entity/team.")

    mo = p.add_argument_group("modes")
    mo.add_argument("--dry-run", action="store_true",
                    help="List every cell with its argv and exit.")
    mo.add_argument("--print-config", action="store_true",
                    help="Load each cell through the real config loader and dump "
                         "the RESOLVED settings. Proves the overrides landed.")
    mo.add_argument("--print-commands", action="store_true",
                    help="Emit one standalone `netdrift_run.py ...` per cell.")
    mo.add_argument("--collect-only", action="store_true",
                    help="Run nothing; harvest each cell's newest summary.json "
                         "into a CSV.")
    mo.add_argument("--limit", type=int, default=None,
                    help="Run only the first N cells (smoke test).")

    return p


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse + finalize. Every default that depends on another flag is resolved
    here, so a Namespace from this function is the same one ``main`` runs on."""
    p = build_parser()
    args = p.parse_args(argv)
    spec = MODELS[args.model]
    if args.config is None:
        args.config = spec["config"]
    if args.kernel_mapping not in spec["kernel_mappings"]:
        p.error(f"--kernel-mapping {args.kernel_mapping!r} is not legal for "
                f"{args.model!r} (legal: {spec['kernel_mappings']}). Non-ROW "
                "mappings permute a 3x3 kernel index list and raise on any "
                "other kernel size.")
    for key, default in CKPT_DEFAULTS.items():
        if getattr(args, f"ckpt_{key}") is None:
            setattr(args, f"ckpt_{key}", default)
    if args.ap_position is not None and args.ap_position < 0:
        args.ap_position = None  # sentinel: leave fault.ap_position unset
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cells = build_cells(args)
    if args.limit is not None:
        cells = cells[: args.limit]

    token = CATEGORIES[args.category]["token"]
    print(f"[{args.model} / {token}] {len(cells)} cells "
          f"(seeds {args.seeds}), {len(args.rt_curve)} rt_error x "
          f"{args.loops} loops each, layout={args.layout}")

    if args.print_config:
        print_resolved_configs(cells, args)
        return 0
    if args.print_commands:
        for cell in cells:
            print(shell_command(cell, args))
        return 0
    if args.dry_run:
        for cell in cells:
            print(f"  {cell['subcategory']}")
            print(f"    {' '.join(cell_argv(cell, args))}")
        return 0

    out_dir = new_sweep_out_dir(f"paper_sw_{args.model}_{token}")
    if args.collect_only:
        csv_path = collect(cells, args, out_dir)
        print(f"wrote {csv_path}")
        return 0

    if args.check_checkpoints:
        missing = check_checkpoints(cells, args)
        if missing:
            print("\nABORT — these checkpoints do not exist:", file=sys.stderr)
            for m in missing:
                print(f"  {m}", file=sys.stderr)
            print("\nFix the --ckpt-* templates (they accept {root}, {layout} and "
                  "{seed}) or pass --no-check-checkpoints.", file=sys.stderr)
            return 2

    runner_main = import_runner_main()
    records: list[dict[str, Any]] = []
    t0 = time.time()
    for i, cell in enumerate(cells, 1):
        argv_cell = cell_argv(cell, args)
        print(f"\n[{args.model} / {token}] cell {i}/{len(cells)}: {cell['subcategory']}")
        t = time.time()
        status, err = run_cell(runner_main, argv_cell)
        rec = {**cell, "status": status, "error": err,
               "checkpoint": cell_checkpoint(cell, args),
               "elapsed_s": round(time.time() - t, 1)}
        records.append(rec)
        write_manifest(out_dir, {"model": args.model, "category": args.category,
                                 "args": vars(args), "cells": records})
        if status != "ok":
            print(f"  !! {status}: {err}", file=sys.stderr)

    csv_path = collect(cells, args, out_dir)
    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\n[{args.model} / {token}] {ok}/{len(records)} cells ok in "
          f"{(time.time() - t0) / 3600:.2f} h — {csv_path}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
