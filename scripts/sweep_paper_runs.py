#!/usr/bin/env python
"""Paper-runs sweep: one command per racetrack-layout arm, run in parallel.

Evaluates three model bases (base w1a1 / cat6 / cat8) over a fixed rt_error
curve under five layout arms. Each arm is a SEPARATE invocation of this script
so the five can run concurrently in five tmux sessions:

    python scripts/sweep_paper_runs.py --layout row           --gpu-num 0 ...
    python scripts/sweep_paper_runs.py --layout col           --gpu-num 1 ...
    python scripts/sweep_paper_runs.py --layout block         --gpu-num 2 ...
    python scripts/sweep_paper_runs.py --layout polarity      --gpu-num 3 ...
    python scripts/sweep_paper_runs.py --layout polarity-reg  --gpu-num 4 ...

Everything here is ``training.mode=test``: the cat6 / cat8 / ppm-regularized
checkpoints are INPUTS, produced beforehand. That is forced, not a choice —
``runner/run.py::_validate_block_layout_combo`` rejects fault-aware training
under block/units/polarity outright, so no training can happen inside the
block/polarity arms anyway.

Axes per arm
------------
* variant      base | cat6 | cat8              (the ``polarity-reg`` arm is
                                                base-only: the ppm_count-
                                                regularized fine-tune OF base)
* seed         707 | 808 | 909                 (see "immune cells" below)
* base_layout  row | col                       (block / polarity arms only;
                                                the row/col arms inherit their
                                                own layout — base_layout is
                                                provably inert there, the dense
                                                fault path never reads it)
* protection   all (layers 1-8) | custom (2-7)
* pad          true | false                    (polarity arms only)

rt_error is NOT a cell axis: the whole curve is swept inside one runner
invocation, which logs one W&B run per rt_error.

Access port: ``--ap-position`` defaults to 0 (low edge) and lands on the row/col
arms only. ``RTMConfig`` REJECTS an absolute index for block/units/polarity —
those resolve the port per bucket as ``P//2 - 1`` — so the block/polarity arms
sit at mid-wire and the driver prints a note saying so. Immaterial for the cells
that are immune anyway; it does bite the ``pad=false`` polarity ablation, which
is therefore not AP-matched to row/col.

Immune cells => one seed
------------------------
Under ``edge_mode=saturate``, BLOCK (every wire a same-sign block + guard band)
and POLARITY with ``pad=true`` (every wire sign-pure by construction) are
FAULT-IMMUNE: the accuracy curve is flat at the clean accuracy for every
rt_error, every seed. Those cells therefore run at a SINGLE seed — the
structural metrics still get collected per combination, but three seeds of an
identical flat line buy nothing. Pass ``--immune-seeds`` to change, or
``--edge-mode random`` (no cell is immune then, so all cells get all seeds).

Cell counts at the defaults::

    row            3 var x 3 seed x 2 prot                    = 18
    col                                                       = 18
    block          3 var x 1 seed x 2 base x 2 prot           = 12
    polarity       3 x 1 x 2 x 2 (pad=t) + 3 x 3 x 2 x 2 (f)  = 48
    polarity-reg   1 x 1 x 2 x 2 (pad=t) + 1 x 3 x 2 x 2 (f)  = 16
                                                        total = 112

Output tree
-----------
``runs/paper-runs/<experiment.name>/<arm>/<cell>/<timestamp>/`` — the arm is the
W&B category and the cell is the W&B subcategory, and ``run.py::_setup_run_dir``
mirrors both into the directory path. BOTH polarity arms log
``storage.layout='polarity'``, so ``category`` (``lay-polarity`` vs
``lay-polarity-regularized``) is the only field that separates them — group and
sort on it, not on ``layout``.

Preflight (do this before burning GPU hours)
--------------------------------------------
    python scripts/sweep_paper_runs.py --layout polarity --print-config | less
        # resolved dataclass per cell — proves every override actually landed,
        # which --dry-run (argv only) does not.
    python scripts/sweep_paper_runs.py --layout polarity --dry-run
    python scripts/sweep_paper_runs.py --layout polarity --limit 1   # smoke run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Put the scripts/ dir on sys.path so comparison_common is importable.
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

DEFAULT_RT_CURVE = [4.55e-05, 1e-05, 1e-06]
DEFAULT_SEEDS = [707, 808, 909]
DEFAULT_IMMUNE_SEEDS = [707]
DEFAULT_LOOPS = 100
DEFAULT_AP_POSITION = 0
DEFAULT_OUTPUT_DIR = "runs/paper-runs/"
DEFAULT_WANDB_PROJECT = "netdrift-paper-runs"

# Per-model constants. Everything model-specific lives HERE so adding a topology
# is one entry, not a scatter of overrides.
#
#   unprotected_custom  1-based layer ids left exposed under policy=custom. The
#                       recipe is always "all but the stem conv and the
#                       classifier" — but the ids differ per topology because
#                       replace_with_quantized enumerates the root's own
#                       children BEFORE descending (so ResNet's classifier is
#                       id 2, not id 21).
#   kernel_mappings     which storage.kernel_mapping values this topology can
#                       take. Non-ROW mappings permute a 3x3 kernel index list
#                       (faults/layout.py::_rearrange_kernel) and raise
#                       NotImplementedError on any other kernel size — so a
#                       topology with 1x1 convs is ROW-only.
#   ckpt_root           <root>/model_best.pt, <root>/cat5_{base_layout}/model.pt,
#                       <root>/cat8_{base_layout}/model.pt,
#                       <root>/ppmreg_{base_layout}/model.pt
MODELS: dict[str, dict[str, Any]] = {
    "vgg7_cifar10": {
        "config": "configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml",
        "experiment_name": "paper_vgg7_w1a1",
        "n_layers": 8,
        "unprotected_custom": [2, 3, 4, 5, 6, 7],
        "kernel_mappings": ["row", "col", "clw", "acw"],
        "ckpt_root": "models/w1a1/vgg7_cifar10",
    },
    "resnet18_imagenette": {
        "config": "configs/resnet18_imagenette/resnet18_imagenette_w1a1_rtm.yaml",
        "experiment_name": "paper_resnet18_imagenette_w1a1",
        "n_layers": 21,
        # 1 = conv1 (stem), 2 = linear (classifier) stay protected; 3..21 are
        # the stage convs + the three 1x1 shortcut convs.
        "unprotected_custom": list(range(3, 22)),
        # ROW ONLY: the three 1x1 shortcut convs make every non-ROW kernel
        # mapping raise. (The topology's own RTM config says "row|col only" —
        # that comment is wrong, COL is a 3x3 permutation too.)
        "kernel_mappings": ["row"],
        "ckpt_root": "models/w1a1/resnet18_imagenette",
    },
}
DEFAULT_MODEL = "vgg7_cifar10"


def ckpt_defaults(model: str) -> dict[str, str]:
    """Conventional checkpoint paths for one model.

    cat6's INPUT is the cat5 (run-length regularizer) checkpoint — cat6 = those
    weights + the BN/Scale re-fit this driver performs per cell.

    EVERY fault-aware checkpoint is per deployment view, so all three carry a
    ``{base_layout}`` placeholder:

    * ``ppmreg`` — ``reg.ppm_base_layout`` is baked into the weights.
    * ``cat5`` / ``cat8`` — ``run_length_penalty`` and the STE-injected forward
      both read the racetrack-aligned view named by ``storage.layout``, and ROW
      and COL are different adjacencies. A ROW-trained cat5 optimised neighbours
      COL does not have, so reusing it under the col arm would confound the
      software technique with a layout mismatch.

    Only ``base`` is view-independent (no fault-aware training touched it).
    """
    root = MODELS[model]["ckpt_root"]
    return {
        "base": f"{root}/model_best.pt",
        "cat6": root + "/cat5_{base_layout}/model.pt",
        "cat8": root + "/cat8_{base_layout}/model.pt",
        "ppmreg": root + "/ppmreg_{base_layout}/model.pt",
    }

# arm -> (storage.layout, permutes base_layout?, permutes pad?, variants)
ARMS: dict[str, dict[str, Any]] = {
    "row":          {"layout": "row",      "base_layouts": ["row"],
                     "pads": [None], "variants": ["base", "cat6", "cat8"]},
    "col":          {"layout": "col",      "base_layouts": ["col"],
                     "pads": [None], "variants": ["base", "cat6", "cat8"]},
    "block":        {"layout": "block",    "base_layouts": ["row", "col"],
                     "pads": [None], "variants": ["base", "cat6", "cat8"]},
    "polarity":     {"layout": "polarity", "base_layouts": ["row", "col"],
                     "pads": [True, False], "variants": ["base", "cat6", "cat8"]},
    "polarity-reg": {"layout": "polarity", "base_layouts": ["row", "col"],
                     "pads": [True, False], "variants": ["ppmreg"]},
}

# W&B category (and directory segment) per arm. Spelled out rather than
# abbreviated: this is the ONLY field that separates the two polarity arms —
# both log storage.layout='polarity' — so grouping in W&B must key on
# `category`, and a cryptic token would make that grouping unreadable.
ARM_TOKEN = {
    "row": "lay-row",
    "col": "lay-col",
    "block": "lay-block",
    "polarity": "lay-polarity",
    "polarity-reg": "lay-polarity-regularized",
}


def prot_token(policy: str, model: str) -> str:
    """Protection tag naming the exposed layer range, e.g. ``prot-1to8``.

    Model-dependent because the layer count and the custom range both are:
    VGG7 -> prot-1to8 / prot-2to7, ResNet18 -> prot-1to21 / prot-3to21.
    """
    spec = MODELS[model]
    if policy == "all":
        return f"prot-1to{spec['n_layers']}"
    layers = spec["unprotected_custom"]
    return f"prot-{layers[0]}to{layers[-1]}"


def is_immune(arm: str, pad: Optional[bool], edge_mode: str) -> bool:
    """True when this cell's accuracy curve is flat by construction.

    Immunity is a property of (saturating reads) x (every wire sign-pure).
    BLOCK gets that from isolating each maximal sign-run onto its own guard-band
    padded wire; POLARITY gets it from sorting each window by sign and rounding
    each sign group up to a wire boundary — which is exactly what ``pad=true``
    does, and exactly what ``pad=false`` gives up. It does NOT depend on the
    checkpoint, the protection policy, or the base layout, so it holds for every
    variant/base/protection combination of those arms.
    """
    if edge_mode != "saturate":
        return False
    if arm == "block":
        return True
    return arm in ("polarity", "polarity-reg") and bool(pad)


def ap_position_supported(layout: str) -> bool:
    """Whether the fault model accepts an absolute ``fault.ap_position``.

    ``RTMConfig.__post_init__`` rejects it for block/units: those paths resolve
    the access port per bucket as ``P//2 - 1``, and one absolute index is
    meaningless across heterogeneous padded lengths.

    POLARITY accepts it as of 2026-09-14. ``build_polarity_buckets`` returns a
    SINGLE bucket whose wires are all exactly ``rt_size`` (the ragged tail is
    sign-filled, not shortened), and both kernels treat that full window as real
    data, so an absolute index means the same thing there as on dense ROW/COL.
    Without this the polarity arms sat at mid-wire while row/col sat at ap0, and
    any polarity-vs-dense gap carried an access-port term on top of the layout.
    """
    return layout in ("row", "col", "polarity")


def cell_loops(cell: dict[str, Any], args: argparse.Namespace) -> int:
    """Inference iterations for one cell.

    Immune cells default to the same budget as everything else; ``--immune-loops``
    exists because their accuracy curve is flat by construction, so the loop
    budget there buys only fault-incidence statistics, not accuracy resolution.
    """
    if cell["immune"] and args.immune_loops is not None:
        return args.immune_loops
    return args.loops


def resolve_ckpt(template: Optional[str], *, variant: str, seed: int,
                 base_layout: str, prefer_best: bool = True) -> str:
    """Fill ``{seed}`` / ``{base_layout}`` in a checkpoint path template.

    A template with no placeholder is used verbatim for every cell — fine when
    one checkpoint is shared across seeds, wrong when it silently hides that the
    per-seed checkpoints were never produced. ``--check-checkpoints`` (on by
    default for real runs) is the guard.

    With ``prefer_best`` (the default, ``--checkpoint-select best``) a resolved
    ``.../model.pt`` is upgraded to the sibling ``model_best.pt`` when one
    exists: a fault-aware fine-tune's final epoch is not reliably its best.

    ``--checkpoint-select final`` turns that off. It exists because the choice is
    NOT neutral for ``ppmreg``: ``model_best.pt`` is selected on clean accuracy
    ALONE, and a best-accuracy epoch is not necessarily a conforming one
    (``runner/run.py``). The ppm objective numbers in ``train_summary.json``
    describe the FINAL weights, so ``final`` is what reproduces a published
    conformance / wire-ratio figure. For base/cat6/cat8 there is no such trade —
    ``best`` is simply the better model.

    Every variant funnels through here, so ``cell_overrides``,
    ``check_checkpoints`` and ``collect`` all see the same resolved file.
    """
    if not template:
        raise SystemExit(
            f"variant {variant!r} needs a checkpoint: pass --ckpt-{variant} "
            f"(supports {{seed}} and {{base_layout}} placeholders)"
        )
    resolved = template.format(seed=seed, base_layout=base_layout)
    return prefer_best_checkpoint(resolved) if prefer_best else resolved


def build_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Enumerate every cell of one arm. Pure — no side effects, no runner calls."""
    spec = ARMS[args.layout]
    variants = args.variants or spec["variants"]
    policies = args.protection or ["all", "custom"]
    base_layouts = args.base_layouts or spec["base_layouts"]
    pads = spec["pads"] if args.pads is None else args.pads
    cells: list[dict[str, Any]] = []

    for variant in variants:
        for base_layout in base_layouts:
            for pad in pads:
                seeds = (args.immune_seeds
                         if is_immune(args.layout, pad, args.edge_mode)
                         else args.seeds)
                for policy in policies:
                    for seed in seeds:
                        parts = [f"var-{variant}", f"base-{base_layout}",
                                 prot_token(policy, args.model)]
                        if pad is not None:
                            parts.append("pad-t" if pad else "pad-f")
                        parts.append(f"seed{seed}")
                        cells.append({
                            "arm": args.layout,
                            "subcategory": "_".join(parts),
                            "variant": variant,
                            "base_layout": base_layout,
                            "pad": pad,
                            "policy": policy,
                            "seed": seed,
                            "immune": is_immune(args.layout, pad, args.edge_mode),
                        })
    return cells


def cell_checkpoint(cell: dict[str, Any], args: argparse.Namespace) -> str:
    """The checkpoint file this cell will actually load.

    The single funnel for that question: ``cell_overrides`` (what the runner is
    told), ``check_checkpoints`` (the pre-launch guard) and ``collect`` (the
    recorded provenance) must never disagree about it.

    Resolution is disk-dependent — ``resolve_ckpt`` upgrades ``model.pt`` to a
    sibling ``model_best.pt`` when one exists — so a sweep launched while its
    checkpoints are still training can resolve differently from cell to cell.
    That is why ``collect`` records the answer per row instead of assuming the
    pre-launch check still describes the run.
    """
    template = {
        "base": args.ckpt_base,
        "cat6": args.ckpt_cat6,
        "cat8": args.ckpt_cat8,
        "ppmreg": args.ckpt_ppmreg,
    }[cell["variant"]]
    return resolve_ckpt(template, variant=cell["variant"],
                        seed=cell["seed"], base_layout=cell["base_layout"],
                        prefer_best=getattr(args, "checkpoint_select", "best") == "best")


def cell_overrides(cell: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """The full ``--override`` list for one cell (config-order, deterministic)."""
    spec = ARMS[cell["arm"]]
    ckpt = cell_checkpoint(cell, args)

    ov = [
        f"experiment.name={args.experiment_name}",
        f"experiment.output_dir={args.output_dir}",
        f"experiment.seed={cell['seed']}",
        f"model.checkpoint={ckpt}",
        f"storage.layout={spec['layout']}",
        f"storage.base_layout={cell['base_layout']}",
        f"storage.rt_size={args.rt_size}",
        f"storage.kernel_mapping={args.kernel_mapping}",
        f"fault.rt_error={rt_error_list_override(args.rt_curve)}",
        "fault.weight_encoder=null",
        "fault.mitigations=[]",
        f"fault.edge_mode={args.edge_mode}",
        f"fault.protection.policy={cell['policy']}",
        "training.mode=test",
        "training.fault_aware=none",
        f"training.loops={cell_loops(cell, args)}",
    ]
    if args.ap_position is not None and ap_position_supported(spec["layout"]):
        ov.append(f"fault.ap_position={args.ap_position}")
    if args.gpu_num is not None:
        # Must live here, not be appended at launch: --print-commands and
        # --print-config both go through cell_overrides, and a printed command
        # missing its device pin is exactly the one that gets pasted into tmux.
        ov.append(f"gpu_num={args.gpu_num}")
    # Passthrough LAST so it can override anything above (e.g. data.num_workers,
    # data.test_batch_size — throughput levers that differ per dataset).
    ov += list(args.override)
    if cell["policy"] == "custom":
        ov.append(
            f"fault.protection.layers="
            f"{json_list(MODELS[args.model]['unprotected_custom'])}")
    if cell["pad"] is not None:
        ov.append(f"storage.partition.pad={'true' if cell['pad'] else 'false'}")
        ov.append(f"storage.partition.window={args.window}")
    # ap_position stays unset everywhere: block/polarity REJECT it (resolved per
    # bucket instead), so pinning it on row/col would make the arms
    # non-comparable on a second axis. null => rt_size//2 - 1 for row/col.
    if cell["variant"] == "cat6" and args.cat6_recalibrate:
        # cat6 = a cat5 (run-length regularizer) checkpoint + BN/Scale re-fit.
        # `on=always` because there is no encoder here to trigger `on=endlen`.
        ov += [
            "training.recalibrate.enabled=true",
            "training.recalibrate.on=always",
            "training.recalibrate.bn_stats=true",
            "training.recalibrate.tune_affine=true",
            f"training.recalibrate.epochs={args.recal_epochs}",
            f"training.recalibrate.lr={args.recal_lr}",
            f"training.criterion={args.criterion}",
            f"training.hinge_b={args.hinge_b}",
        ]
    return ov


def cell_argv(cell: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Runner argv for one cell."""
    argv = ["--config", args.config]
    for o in cell_overrides(cell, args):
        argv += ["--override", o]
    argv += ["--metrics", args.metrics]
    argv += wandb_args(args.wandb_project or None, args.wandb_entity,
                       category=ARM_TOKEN[cell["arm"]],
                       subcategory=cell["subcategory"])
    return argv


def shell_command(cell: dict[str, Any], args: argparse.Namespace) -> str:
    """A standalone shell command for one cell (for GNU parallel / manual reruns)."""
    import shlex
    return " ".join(["python", "netdrift_run.py"]
                    + [shlex.quote(a) for a in cell_argv(cell, args)])


def cell_run_dir(cell: dict[str, Any], args: argparse.Namespace) -> Path:
    """``<output_dir>/<name>/<arm>/<cell>/`` — the parent of the timestamp dirs."""
    return (REPO_ROOT / args.output_dir / args.experiment_name
            / ARM_TOKEN[cell["arm"]] / cell["subcategory"])


def latest_cell_summary(cell: dict[str, Any], args: argparse.Namespace) -> Optional[Path]:
    """Newest ``summary.json`` under this cell's own subtree.

    Deliberately NOT ``comparison_common.latest_summary``: every cell here
    shares one ``experiment.name`` (that is what keeps the output tree readable),
    so keying the harvest on the name alone would return one arbitrary cell's
    summary for all 112 of them.
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


def curve_spread(h: dict[str, Any]) -> Optional[float]:
    """Widest gap between clean accuracy and any per-loop accuracy in the sweep.

    The seed reduction on "immune" cells rests on a STRUCTURAL prediction (see
    :func:`is_immune`) that nothing in the pipeline verifies. This is the audit:
    a genuinely immune cell reads every bit correctly at every rt_error and
    every loop, so its spread is 0.0. Anything above ``FLAT_TOL`` means the
    prediction was wrong for that combination and it is sitting at n=1 when it
    should have three seeds.
    """
    if not h["rt_curve"]:
        return None
    accs = [a for e in h["rt_curve"].values() for a in e["accuracies"]]
    clean = h.get("baseline_clean_accuracy")
    if clean is not None:
        accs.append(float(clean))
    return round(max(accs) - min(accs), 4)


FLAT_TOL = 0.01  # percentage points; immunity is exact, so this is float slack


def collect(cells: list[dict[str, Any]], args: argparse.Namespace, out_dir: Path) -> Path:
    """Harvest each cell's newest summary.json into one CSV for this arm."""
    rt_cols = [f"rt_{rt}" for rt in args.rt_curve]
    csv_path = out_dir / f"paper_runs_{ARM_TOKEN[args.layout]}_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    violations: list[tuple[str, float]] = []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        # ``checkpoint`` is appended LAST so every existing column keeps its
        # index for downstream readers that go by position.
        w.writerow(["arm", "subcategory", "variant", "base_layout", "pad",
                    "policy", "seed", "immune", "curve_spread", "flat", "status",
                    "clean_accuracy"]
                   + [f"{c}_mean" for c in rt_cols]
                   + [f"{c}_last" for c in rt_cols]
                   + ["checkpoint"])
        for cell in cells:
            s = latest_cell_summary(cell, args)
            h = harvest_summary(s)
            status = "ok" if s is not None and h["rt_curve"] else "missing"
            spread = curve_spread(h)
            flat = "" if spread is None else (spread <= FLAT_TOL)
            if cell["immune"] and spread is not None and spread > FLAT_TOL:
                violations.append((cell["subcategory"], spread))
            row = [cell["arm"], cell["subcategory"], cell["variant"],
                   cell["base_layout"], cell["pad"], cell["policy"],
                   cell["seed"], cell["immune"],
                   "" if spread is None else spread, flat, status,
                   h["baseline_clean_accuracy"]]
            for rt in args.rt_curve:
                e = h["rt_curve"].get(float(rt))
                row.append(round(e["mean"], 4) if e else "")
            for rt in args.rt_curve:
                e = h["rt_curve"].get(float(rt))
                row.append(round(e["last"], 4) if e else "")
            row.append(cell_checkpoint(cell, args))
            w.writerow(row)
    if violations:
        print(f"\n!! {len(violations)} cell(s) were predicted fault-immune (so ran "
              f"at a single seed) but their curve is NOT flat — re-run these with "
              f"the full --seeds set:", file=sys.stderr)
        for sub, spread in violations:
            print(f"     {sub}  spread={spread} pp", file=sys.stderr)
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
        print(f"{ARM_TOKEN[cell['arm']]} / {cell['subcategory']}"
              f"{'   [immune]' if cell['immune'] else ''}")
        print("-" * 78)
        print(json.dumps({
            "checkpoint": cfg.model.checkpoint,
            "seed": cfg.experiment.seed,
            "output_dir": cfg.experiment.output_dir,
            "storage": {"layout": cfg.storage.layout,
                        "base_layout": cfg.storage.base_layout,
                        "rt_size": cfg.storage.rt_size,
                        "kernel_mapping": cfg.storage.kernel_mapping,
                        "partition": {"window": cfg.storage.partition.window,
                                      "pad": cfg.storage.partition.pad}},
            "fault": {"rt_error": cfg.fault.rt_error,
                      "edge_mode": cfg.fault.edge_mode,
                      "ap_position": cfg.fault.ap_position,
                      "weight_encoder": cfg.fault.weight_encoder,
                      "protection": {"policy": cfg.fault.protection.policy,
                                     "layers": cfg.fault.protection.layers}},
            "training": {"mode": cfg.training.mode,
                         "fault_aware": cfg.training.fault_aware,
                         "loops": cfg.training.loops,
                         "recalibrate": {"enabled": cfg.training.recalibrate.enabled,
                                         "on": cfg.training.recalibrate.on,
                                         "epochs": cfg.training.recalibrate.epochs},
                         "criterion": cfg.training.criterion},
        }, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Paper-runs layout sweep (one arm per invocation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--layout", required=True, choices=sorted(ARMS),
                   help="Which layout arm to run. One arm per tmux session.")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS),
                   help="Topology/dataset. Selects the base YAML, the "
                        "experiment name, the protection layer ids, the legal "
                        "kernel mappings and the checkpoint root. "
                        f"Default: {DEFAULT_MODEL}")
    p.add_argument("--config", default=None,
                   help="Base YAML for every cell. Default: the --model entry's.")
    p.add_argument("--experiment-name", default=None,
                   help="experiment.name (shared by all arms of one model; the "
                        "arm and cell become path segments under it). Default: "
                        "the --model entry's.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"experiment.output_dir. Default: {DEFAULT_OUTPUT_DIR}")
    p.add_argument("--rt-curve", nargs="+", type=float, default=DEFAULT_RT_CURVE,
                   help=f"rt_error values swept INSIDE each cell. Default: {DEFAULT_RT_CURVE}")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help=f"Seeds for non-immune cells. Default: {DEFAULT_SEEDS}")
    p.add_argument("--immune-seeds", nargs="+", type=int, default=DEFAULT_IMMUNE_SEEDS,
                   help="Seeds for cells whose curve is flat by construction "
                        "(block, and polarity with pad=true, under saturate). "
                        f"Default: {DEFAULT_IMMUNE_SEEDS}")
    p.add_argument("--loops", type=int, default=DEFAULT_LOOPS,
                   help=f"Inference iterations per rt_error. Default: {DEFAULT_LOOPS}")
    p.add_argument("--immune-loops", type=int, default=None,
                   help="Separate loop budget for the immune-by-construction cells "
                        "(their accuracy curve is flat, so loops there buy only "
                        "fault-incidence stats). Default: same as --loops.")
    p.add_argument("--ap-position", type=int, default=DEFAULT_AP_POSITION,
                   help="Fixed access-port index for edge_mode=saturate. Applied "
                        "to the row/col arms only — block/polarity reject an "
                        "absolute index and resolve the port per bucket. "
                        f"Default: {DEFAULT_AP_POSITION} (low edge). Pass -1 to "
                        "leave it unset (null => mid-wire rt_size//2 - 1).")
    p.add_argument("--variants", nargs="+", default=None,
                   choices=["base", "cat6", "cat8", "ppmreg"],
                   help="Subset of the arm's model bases. Default: all of them.")
    p.add_argument("--pads", nargs="+", default=None, choices=["true", "false"],
                   help="Subset of storage.partition.pad values, polarity arms "
                        "only. Default: both. NB pad=true is the fault-immune "
                        "half (every wire sign-pure by construction under "
                        "edge_mode=saturate); dropping it leaves the arm as the "
                        "pad=false ablation alone, which is NOT access-port "
                        "matched to row/col — block/polarity reject an absolute "
                        "fault.ap_position and resolve it per bucket.")
    p.add_argument("--base-layouts", nargs="+", default=None, choices=["row", "col"],
                   help="Subset of base_layout values. Default: the arm's own set "
                        "(row/col arms: just themselves — base_layout is inert there).")
    p.add_argument("--protection", nargs="+", default=None, choices=["all", "custom"],
                   help="Protection policies. all = every quantized layer "
                        "unprotected; custom = the --model entry's "
                        "unprotected_custom (all but stem conv + classifier). "
                        "Default: both.")
    p.add_argument("--edge-mode", default="saturate", choices=["saturate", "random"],
                   help="Racetrack edge model, applied to every cell. NB saturate "
                        "makes block / polarity(pad=true) fault-immune — which is "
                        "what --immune-seeds keys off. Default: saturate")
    p.add_argument("--window", type=int, default=0,
                   help="storage.partition.window for the polarity arms "
                        "(0 = channel-aligned). Default: 0")
    p.add_argument("--rt-size", type=int, default=64, help="storage.rt_size. Default: 64")
    p.add_argument("--kernel-mapping", default="row", choices=["row", "col", "clw", "acw"],
                   help="storage.kernel_mapping. Validated against the model's "
                        "legal set (topologies with non-3x3 convs are ROW-only). "
                        "Default: row")
    p.add_argument("--metrics", default="all", choices=["none", "offline", "online", "all"],
                   help="Runner --metrics level. Default: all")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="Extra runner override applied to EVERY cell, after all "
                        "driver-managed ones (so it wins). Repeatable. E.g. "
                        "--override data.num_workers=8")
    p.add_argument("--gpu-num", type=int, default=None,
                   help="Pin this arm to one CUDA device (sets gpu_num). Give each "
                        "tmux session its own device; five arms on one GPU thrash.")

    ck = p.add_argument_group("checkpoints (accept {seed} and {base_layout} placeholders)")
    ck.add_argument("--ckpt-base", default=None,
                    help="base w1a1 checkpoint. Default: <ckpt_root>/model_best.pt")
    ck.add_argument("--ckpt-cat6", default=None,
                    help="cat5 (run-length regularizer) checkpoint that cat6 "
                         "recalibrates. Default: "
                         "<ckpt_root>/cat5_{base_layout}/model.pt")
    ck.add_argument("--ckpt-cat8", default=None,
                    help="STE-injection-trained checkpoint. "
                         "Default: <ckpt_root>/cat8_{base_layout}/model.pt")
    ck.add_argument("--ckpt-ppmreg", default=None,
                    help="ppm_count-regularized fine-tune OF the base model, for "
                         "--layout polarity-reg. Its ppm_base_layout is baked into "
                         "the weights, so one per view. Default: "
                         "<ckpt_root>/ppmreg_{base_layout}/model.pt")
    ck.add_argument("--checkpoint-select", default="best", choices=["best", "final"],
                    dest="checkpoint_select",
                    help="Which file inside a checkpoint dir to load. 'best' "
                         "(default) prefers model_best.pt over model.pt; 'final' "
                         "takes model.pt as written. NOT neutral for ppmreg: "
                         "model_best.pt is selected on clean accuracy alone and a "
                         "best-accuracy epoch need not be a conforming one, while "
                         "train_summary.json's ppm objective numbers describe the "
                         "FINAL weights. Use 'final' to reproduce a published "
                         "conformance / wire-ratio figure.")
    ck.add_argument("--no-check-checkpoints", dest="check_checkpoints",
                    action="store_false",
                    help="Skip the pre-launch existence check on every resolved "
                         "checkpoint path.")
    p.set_defaults(check_checkpoints=True)

    rc = p.add_argument_group("cat6 recalibration")
    rc.add_argument("--no-cat6-recalibrate", dest="cat6_recalibrate",
                    action="store_false",
                    help="Load the cat6 checkpoint as-is instead of running the "
                         "BN/Scale re-fit in each cell (use when your checkpoint is "
                         "already recalibrated).")
    rc.add_argument("--recal-epochs", type=int, default=2,
                    help="training.recalibrate.epochs. Default: 2")
    rc.add_argument("--recal-lr", type=float, default=0.001,
                    help="training.recalibrate.lr. Default: 0.001")
    rc.add_argument("--criterion", default="hinge", choices=["hinge", "cross_entropy"],
                    help="Criterion for the recalibration tune step. Default: hinge")
    rc.add_argument("--hinge-b", type=float, default=128.0,
                    help="b for the hinge criterion. Default: 128.0")
    p.set_defaults(cat6_recalibrate=True)

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

    args = p.parse_args(argv)
    spec = MODELS[args.model]
    if args.config is None:
        args.config = spec["config"]
    if args.experiment_name is None:
        args.experiment_name = spec["experiment_name"]
    if args.kernel_mapping not in spec["kernel_mappings"]:
        p.error(f"--kernel-mapping {args.kernel_mapping!r} is not legal for "
                f"{args.model!r} (legal: {spec['kernel_mappings']}). Non-ROW "
                "mappings permute a 3x3 kernel index list and raise on any "
                "other kernel size.")
    _ck = ckpt_defaults(args.model)
    for variant, default in _ck.items():
        flag = f"ckpt_{variant}"
        if getattr(args, flag) is None:
            setattr(args, flag, default)
    if args.ap_position is not None and args.ap_position < 0:
        args.ap_position = None  # sentinel: leave fault.ap_position unset
    if args.pads is not None:
        if ARMS[args.layout]["pads"] == [None]:
            p.error(f"--pads is not meaningful for --layout {args.layout!r}: that "
                    "arm does not partition, so storage.partition.pad is never "
                    "set. Drop the flag (it applies to the polarity arms only).")
        args.pads = [v == "true" for v in args.pads]
    cells = build_cells(args)
    if args.limit is not None:
        cells = cells[: args.limit]

    arm_tok = ARM_TOKEN[args.layout]
    n_immune = sum(1 for c in cells if c["immune"])
    loops_note = (f"{args.loops} loops" if args.immune_loops is None
                  else f"{args.loops} loops ({args.immune_loops} on immune cells)")
    print(f"[{arm_tok}] {len(cells)} cells "
          f"({n_immune} immune-by-construction at {args.immune_seeds}), "
          f"{len(args.rt_curve)} rt_error x {loops_note} each")
    if args.ap_position is not None and not ap_position_supported(ARMS[args.layout]["layout"]):
        print(f"  note: --ap-position {args.ap_position} NOT applied — the "
              f"{ARMS[args.layout]['layout']} fault path rejects an absolute "
              f"access-port index and resolves it per bucket (P//2 - 1).")

    if args.print_config:
        print_resolved_configs(cells, args)
        return 0
    if args.print_commands:
        for cell in cells:
            print(shell_command(cell, args))
        return 0
    if args.dry_run:
        for cell in cells:
            print(f"  {cell['subcategory']}"
                  f"{'  [immune]' if cell['immune'] else ''}")
            print(f"    {' '.join(cell_argv(cell, args))}")
        return 0

    out_dir = new_sweep_out_dir(f"paper_runs_{arm_tok}")
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
            print("\nFix the --ckpt-* templates (they accept {seed} and "
                  "{base_layout}) or pass --no-check-checkpoints.", file=sys.stderr)
            return 2

    runner_main = import_runner_main()
    records: list[dict[str, Any]] = []
    t0 = time.time()
    for i, cell in enumerate(cells, 1):
        argv_cell = cell_argv(cell, args)
        if args.gpu_num is not None:
            argv_cell += ["--override", f"gpu_num={args.gpu_num}"]
        print(f"\n[{arm_tok}] cell {i}/{len(cells)}: {cell['subcategory']}")
        t = time.time()
        status, err = run_cell(runner_main, argv_cell)
        rec = {**cell, "status": status, "error": err,
               "elapsed_s": round(time.time() - t, 1)}
        records.append(rec)
        write_manifest(out_dir, {"arm": args.layout, "args": vars(args),
                                 "cells": records})
        if status != "ok":
            print(f"  !! {status}: {err}", file=sys.stderr)

    csv_path = collect(cells, args, out_dir)
    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\n[{arm_tok}] {ok}/{len(records)} cells ok in "
          f"{(time.time() - t0) / 3600:.2f} h — {csv_path}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
