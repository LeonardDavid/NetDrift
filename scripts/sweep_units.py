#!/usr/bin/env python
"""DENSE-vs-BLOCK-vs-UNITS racetrack-layout comparison sweep.

Compares the plain **COL** ("dense") racetrack layout at several rt_size
values against the **BLOCK** layout and the new **UNITS** layout (racetrack-
memory-style isolated runs with optional guarded period-2 pooling), crossed
with a mitigation axis (none / odd2even parity-swap) and a seed axis. Every
cell is driven from ONE vgg7 base config and overrides only the axes under
comparison, so the arms differ *only* in the intended variables (mirrors
``scripts/sweep_col_vs_block.py``).

Matrix (one runner invocation = one "cell"; each cell sweeps the full
rt_error curve internally, one W&B run per rt_error):

* arm         : dense-rt64 | dense-rt8 | dense-rt4 | dense-rt2 | block |
                units-t8 | units-t4 | units-t3 | units-t2-g1 | units-t2-g0 |
                units-t1                                            — 11
* mitigation  : nomit | odd2even                                    —  2
* seed        : subset of {707, 808, 909} per the seed-coverage policy below

No per-layer allocation arm is included: the brief's ARMS matrix (and the
plan/spec it is drawn from) does not define one, so none was invented here
per "do not invent a new pattern" — this is a deliberate omission, not an
oversight.

BASE_LAYOUT DEFAULT IS "col", NOT "row": the five dense/control arms
(dense-rt64/8/4/2, and the iso-cost control dense-rt2) are declared with
``layout: col`` above and are never affected by ``--base-layout`` at all.
``docs/superpowers/specs/2026-07-30-unit-based-racetrack-layout-design.md``
section 2's entire cost table -- wires, A@16, A@16+T, bits-immune, i.e. the
Pareto x-axis this sweep's numbers get joined against -- is computed for
``base_layout=col`` (the section 2 table header says "design (col)"; BLOCK's
own row-vs-col counts differ materially: 5,610,139 row vs 6,527,245 col). If
``--base-layout`` defaulted to ``row``, every block/units arm would be
ROW-segmented while its dense controls stayed COL-segmented -- two variables
would differ between an arm and its control instead of one, and NO units arm
would be joinable to the spec's cost table (the join key is the design
label, which carries the threshold, not the base layout). Defaulting to
``col`` makes the primary sweep's block/units arms match their controls and
the published cost axis on the base_layout variable, so only the layout
mechanism itself varies. ``--base-layout row`` remains selectable and is a
deliberate SECONDARY sweep (e.g. to reproduce the row-side of the spec's
BLOCK figures) -- it is not the sweep whose numbers should be reported
against section 2's table.

SEED-COVERAGE POLICY (see ``_arm_needs_all_seeds`` for the implementation):
BLOCK is provably fault-immune under ``edge_mode=saturate``
(``tests/test_ap_saturate.py::test_block_saturate_is_fault_immune``) — a real
property of the model, not a bug. ``units-t1`` (threshold=1, max_period=1) is
structurally identical to BLOCK (every wire is one same-sign run plus a
same-sign guard band), so the SAME immunity argument applies to it, and to
nothing else.
  * edge_mode=saturate: ONLY ``block`` and ``units-t1`` are immune/deterministic
    -> ONE seed suffices for those two arms. EVERY OTHER ARM gets ALL seeds,
    including dense-* : dense wires are neither immune nor deterministic
    under saturate -- they random-walk and either land in-bounds (correct
    read) or drift out (chance-level read), and this branch's own recovered
    ``col_vs_block`` data proves it (dense COL: 88.19% clean -> ~10.4%
    (chance) at rt_error=1e-4). An earlier version of this policy also
    collapsed dense-* to one seed on the mistaken claim that they were
    "immune or deterministic"; spec section 6 exempts ONLY BLOCK, and
    ``dense-rt2`` is the iso-cost control carrying the study's headline
    comparison against BLOCK, so it must never ship with n=1. Units arms with
    threshold >= 2 (units-t8/t4/t3/t2-g1/t2-g0) also get ALL seeds: they
    contain pooled or period-2 wires whose per-wire error is all-or-nothing,
    so a single seed cannot capture the variance.
  * edge_mode=random: BLOCK is neither immune nor deterministic there either
    -> EVERY arm needs ALL seeds, no exceptions (not even block/units-t1).
Seeds skipped under this policy are logged, never silently dropped -- see the
"[seed-policy]" lines and the applied-policy summary the run header prints
(module-level; ``main()`` prints per-arm seed counts and the resulting total
cell count every invocation, so the reduction from the full seed x arm x
mitigation grid is never silent).

METRICS: every cell runs with ``--metrics offline`` by default -- JSON
snapshots (block/run/wire counts, magnitude stats, n_racetracks per layer),
no ``.npz`` raw arrays, so disk stays modest across the full arm x mitigation
x seed matrix. Units-layout metrics are supported:
``compute_static_metrics`` (``netdrift/metrics/static.py``) has a dedicated
UNITS branch that packs wires via ``build_unit_buckets``, the same way this
driver's units arms do. Artifacts land under ``<run_dir>/metrics_artifacts/``,
which is excluded from this Mac mount's sync (host-only -- harvest there, not
here). ``scripts/rvc_common.py``'s discovery helpers read both ``metrics/``
(pre-rename runs) and ``metrics_artifacts/``, so downstream aggregation sees
runs either way. Pass ``--metrics none`` to disable, or ``online``/``all`` for
the heavier per-iteration / raw-array levels.

MITIGATION OVERRIDE: ``fault.mitigations`` is ``list[str]``, and the runner's
override parser does ``json.loads(value)`` on the raw string, falling back to
a bare Python *string* (not a list) if that fails
(``code/python/netdrift/config/loader.py:_parse_override_value``).
``comparison_common.json_list`` renders bare ints (fine for e.g.
``fault.protection.layers``) and does not quote — wrong for mitigation names.
This was empirically verified against the real loader
(``netdrift.config.loader.load``): an unquoted
``--override fault.mitigations=[odd2even_dec]`` resolves to the STRING
``"[odd2even_dec]"``, not a list, silently corrupting the mitigation config.
``mitigation_override`` below JSON-quotes each name so it round-trips as a
real ``list[str]``.

Usage::

    # Dry-run: list every cell with its argv, no execution
    python scripts/sweep_units.py --dry-run

    # Execute, logging to a dedicated W&B project
    python scripts/sweep_units.py --wandb-project netdrift-units

    # Non-immune edge model: every arm gets every seed
    python scripts/sweep_units.py --edge-mode random --dry-run
"""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
import time
from pathlib import Path
from typing import Optional

# Put the scripts/ dir on sys.path so comparison_common is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparison_common import (  # noqa: E402
    REPO_ROOT,
    base_overrides,
    harvest_summary,
    import_runner_main,
    latest_summary,
    layout_token,
    new_sweep_out_dir,
    output_dir_from_cfg,
    run_cell,
    wandb_args,
    write_manifest,
)

# Defaults specific to this study (differ from the comparison-DB canon).
DEFAULT_CONFIG = "configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml"
DEFAULT_RT_CURVE = [1e-4, 4.55e-5, 1e-5, 1e-6, 1e-7]
DEFAULT_LOOPS = 30
DEFAULT_SEEDS = [707, 808, 909]
DEFAULT_BASE_LAYOUT = "col"
DEFAULT_WANDB_PROJECT = "netdrift-units"

# (label, layout, rt_size, units{threshold,max_period,pool_guard})
# units=None => storage.units.* is left untouched (dense-*/block arms).
# max_period=2 is only legal at threshold=2 (config layer rejects other
# pairings; see code/python/netdrift/config/schema.py:UnitsCfg.__post_init__),
# so no other (threshold, 2, *) pairing is generated here.
ARMS: list[tuple[str, str, int, Optional[tuple[int, int, int]]]] = [
    ("dense-rt64",   "col",   64, None),                 # baseline
    ("dense-rt8",    "col",    8, None),                 # table-free control
    ("dense-rt4",    "col",    4, None),                 # table-free control
    ("dense-rt2",    "col",    2, None),                 # iso-cost control vs BLOCK
    ("block",        "block", 64, None),                 # upper bound
    ("units-t8",     "units", 64, (8, 1, 0)),
    ("units-t4",     "units", 64, (4, 1, 0)),
    ("units-t3",     "units", 64, (3, 1, 0)),
    ("units-t2-g1",  "units", 64, (2, 2, 1)),             # period-2 pooling
    ("units-t2-g0",  "units", 64, (2, 1, 0)),             # ablation: guards off
    ("units-t1",     "units", 64, (1, 1, 0)),             # == block, sanity arm
]

# (mit_key, mitigation names). "nomit" must stay [] (renders to the valid JSON
# literal "[]"); odd2even is expected to matter only for units-t2-g1 (the only
# arm with pooled period-2 wires + guards enabled) — that is a result to
# observe, not a filter applied by the driver (every arm still gets both).
MITIGATIONS: list[tuple[str, list[str]]] = [
    ("nomit", []),
    ("odd2even", ["odd2even_dec"]),
]


def mitigation_override(names: list[str]) -> str:
    """Render a mitigation list for --override as valid JSON (a list of strings).

    ``comparison_common.json_list`` renders bare ints and does NOT quote
    elements — wrong here. ``fault.mitigations`` is ``list[str]``, and the
    override parser does a plain ``json.loads`` on the value
    (code/python/netdrift/config/loader.py:_parse_override_value), falling
    back to a bare *string* on a JSON parse failure. An unquoted
    ``[odd2even_dec]`` is NOT valid JSON (bare identifiers aren't JSON
    tokens), so it would silently resolve to the string ``"[odd2even_dec]"``
    instead of the list ``["odd2even_dec"]`` — verified empirically against
    ``netdrift.config.loader.load``. Quoting each name fixes this: `[]` when
    empty, `["odd2even_dec"]` otherwise.
    """
    return "[" + ",".join(f'"{n}"' for n in names) + "]"


def _arm_needs_all_seeds(
    layout: str, units: Optional[tuple[int, int, int]], edge_mode: str
) -> bool:
    """Seed-coverage policy (see module docstring for the property behind it).

    * edge_mode == "random": BLOCK is neither immune nor deterministic there
      -> every arm needs every seed, no exceptions (not even block/units-t1).
    * edge_mode == "saturate": ONLY ``block`` and ``units-t1`` (threshold=1,
      max_period=1 -- structurally identical to block) are provably
      immune/deterministic
      (tests/test_ap_saturate.py::test_block_saturate_is_fault_immune) ->
      one seed suffices for those two arms alone.
      EVERY OTHER ARM needs every seed under saturate too, including
      dense-* : dense wires random-walk and are neither immune nor
      deterministic there (this branch's own recovered col_vs_block data:
      dense COL 88.19% clean -> ~10.4% (chance) at rt_error=1e-4). An earlier
      version of this function collapsed dense-* to one seed alongside
      block/units-t1 on the false premise that they shared BLOCK's immunity;
      spec section 6 exempts ONLY BLOCK. Units arms with threshold >= 2 have
      pooled/period-2 wires whose per-wire error is all-or-nothing (one seed
      could land on a lucky or unlucky phase) -> those also need every seed.
    """
    if edge_mode == "random":
        return True
    if layout == "block":
        return False
    if layout == "units" and units is not None and units[0] == 1:
        return False
    return True


def _build_cells(
    seeds: list[int],
    mitigations: list[tuple[str, list[str]]],
    edge_mode: str,
) -> tuple[list[dict], list[str]]:
    """Return (cell descriptors, seed-policy drop log) for arm x mitigation x seed.

    Unlike sweep_col_vs_block._build_cells (no args — its 3x2 matrix is fully
    static), this driver's seed axis is edge_mode/arm-dependent (see
    _arm_needs_all_seeds), so seeds/mitigations/edge_mode are threaded through
    as parameters instead of being module-level constants. Dropped seeds are
    returned as log lines rather than printed here, so callers that build
    cells more than once (dry-run, print-commands, run) don't print the
    policy log redundantly.
    """
    cells: list[dict] = []
    dropped: list[str] = []
    for label, layout, rt_size, units in ARMS:
        needs_all = _arm_needs_all_seeds(layout, units, edge_mode)
        arm_seeds = seeds if needs_all else seeds[:1]
        if not needs_all and len(seeds) > 1:
            dropped.append(
                f"{label}: seeds {seeds[1:]} skipped under edge_mode={edge_mode} "
                f"(deterministic/immune arm; kept seed={seeds[0]})"
            )
        for mit_key, mit_list in mitigations:
            for seed in arm_seeds:
                cells.append({
                    "label": label,
                    "layout": layout,
                    "rt_size": rt_size,
                    "units": units,
                    "mit_key": mit_key,
                    "mit_list": mit_list,
                    "seed": seed,
                })
    return cells, dropped


def _exp_name(base_stem: str, cell: dict, base_layout: str) -> str:
    """Deterministic per-cell experiment name — the SINGLE source used by both
    the live-run path (_cell_argv) and --collect-only (which must reconstruct
    it WITHOUT calling _cell_argv). Keeping this in one place means the two
    paths cannot drift apart and have --collect-only report false MISSING
    summaries.

    For block/units cells, folds in the base_layout tag. Without this, two
    invocations differing only in --base-layout write to the SAME experiment
    directory and --collect-only cannot tell which run produced which
    summary.json -- silent mislabeling of experimental data, not just a
    missed harvest. The fixed 11-arm ARMS matrix always uses one base_layout
    per invocation so this never fires today, but --base-layout is a real CLI
    flag, so the collision is reachable.
    """
    name = f"{base_stem}__{cell['label']}__{cell['mit_key']}__seed{cell['seed']}"
    if cell["layout"] in ("block", "units"):
        name = f"{name}__{layout_token(base_layout)}"
    return name


def _cell_argv(
    cell: dict,
    cfg_path: Path,
    curve: list[float],
    loops: int,
    edge_mode: str,
    base_layout: str,
    base_stem: str,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    metrics: str,
) -> list[str]:
    """Assemble the full runner argv for one cell.

    Overrides ONLY the compared axes (layout, rt_size, units.*, mitigation)
    plus the invariants that must not drift (no encoder, edge_mode, seed,
    metrics level). Everything else comes from the base config so all arms
    share identical model/data/quant settings. ``--metrics`` is passed
    uniformly on every cell (module default: offline; see module docstring)
    so metrics coverage never varies across the compared axes either.
    """
    argv = ["--config", str(cfg_path)]

    # Shared: rt_error curve + loops. Protection is left at the config's own
    # default (comparison_common.base_overrides only emits fault.protection.*
    # when explicitly asked, matching sweep_col_vs_block's convention).
    argv += base_overrides(curve=curve, loops=loops)

    # Layout axis.
    argv += ["--override", f"storage.layout={cell['layout']}"]
    argv += ["--override", f"storage.rt_size={cell['rt_size']}"]
    if cell["layout"] in ("block", "units"):
        argv += ["--override", f"storage.base_layout={base_layout}"]
    if cell["units"] is not None:
        t, mp, pg = cell["units"]
        argv += ["--override", f"storage.units.threshold={t}"]
        argv += ["--override", f"storage.units.max_period={mp}"]
        argv += ["--override", f"storage.units.pool_guard={pg}"]

    # Invariants: no encoder, mitigation list, fixed edge model + seed. Set
    # explicitly (do NOT trust config defaults — some rtm configs ship endlen).
    argv += [
        "--override", "fault.weight_encoder=null",
        "--override", f"fault.mitigations={mitigation_override(cell['mit_list'])}",
        "--override", f"fault.edge_mode={edge_mode}",
        "--override", f"experiment.seed={cell['seed']}",
    ]

    # Metrics level: a direct runner CLI flag (not a config override), passed
    # uniformly on every cell so metrics coverage is never itself a varying
    # axis. See module docstring for what 'offline' writes and where.
    argv += ["--metrics", metrics]

    # Experiment identity (unique per cell so summaries don't collide; for
    # block/units cells this includes the base_layout tag -- see _exp_name).
    exp_name = _exp_name(base_stem, cell, base_layout)
    argv += ["--override", f"experiment.name={exp_name}"]

    # W&B: coarse category = the study, fine subcategory = this arm + mitigation
    # + seed. For block/units arms, also fold in the base_layout tag (mirrors
    # _exp_name) so runs group correctly when base_layout is swept across
    # invocations.
    subcategory = f"{cell['label']}_{cell['mit_key']}_seed{cell['seed']}"
    if cell["layout"] in ("block", "units"):
        subcategory = f"{subcategory}_{layout_token(base_layout)}"
    argv += wandb_args(wandb_project, wandb_entity, "units_layout_sweep", subcategory)
    return argv


def _fmt_argv(argv: list[str]) -> str:
    """Shell-quote an argv list for display/emission.

    sweep_col_vs_block gets away with a bare ' '.join(argv) because its only
    list-valued override is '[]' or bare floats. Here
    ``--override fault.mitigations=["odd2even_dec"]`` contains double quotes
    that a bare join would lose if pasted into a shell (or if piped straight
    to a shell via --print-commands), silently reproducing the broken
    unquoted-string form this driver exists to avoid. shlex.quote preserves
    the literal value through a shell round-trip.
    """
    return " ".join(shlex.quote(a) for a in argv)


def _write_tables(out_dir: Path, results: list[dict], curve: list[float]) -> None:
    """Write a flat CSV (one row per cell, mean accuracy per rt_error)."""
    csv_path = out_dir / "units_sweep_summary.csv"
    rt_cols = [f"rt_{rt:g}_mean" for rt in curve]
    fieldnames = (
        ["label", "layout", "rt_size", "base_layout", "threshold", "max_period",
         "pool_guard", "mitigation", "seed", "status", "elapsed_s",
         "baseline_clean_accuracy", "experiment_name"]
        + rt_cols
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            rt_curve = r.get("rt_curve") or {}
            units = r.get("units")
            row = {
                "label": r["label"],
                "layout": r["layout"],
                "rt_size": r["rt_size"],
                "base_layout": r["base_layout"] if r["layout"] in ("block", "units") else "",
                "threshold": units[0] if units else "",
                "max_period": units[1] if units else "",
                "pool_guard": units[2] if units else "",
                "mitigation": r["mit_key"],
                "seed": r["seed"],
                "status": r["status"],
                "elapsed_s": r.get("elapsed_s", ""),
                "baseline_clean_accuracy": r.get("baseline_clean_accuracy"),
                "experiment_name": r["exp_name"],
            }
            for rt in curve:
                m = rt_curve.get(float(rt))
                row[f"rt_{rt:g}_mean"] = f"{m['mean']:.4f}" if m else ""
            w.writerow(row)
    print(f"  -> {csv_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"Base YAML (single source for all arms). Default: {DEFAULT_CONFIG}")
    p.add_argument("--rt-curve", nargs="+", type=float, default=DEFAULT_RT_CURVE,
                   help="rt_error values swept inside each cell. "
                        f"Default: {DEFAULT_RT_CURVE}")
    p.add_argument("--loops", type=int, default=DEFAULT_LOOPS,
                   help=f"Inference iterations per rt_error. Default: {DEFAULT_LOOPS}")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="Candidate seed pool. The seed-coverage policy (see module "
                        "docstring) decides how many of these are actually used per "
                        f"arm. Default: {DEFAULT_SEEDS}")
    p.add_argument("--mitigations", nargs="+", choices=[k for k, _ in MITIGATIONS],
                   default=[k for k, _ in MITIGATIONS],
                   help="Which mitigation arms to run. Default: both (nomit odd2even).")
    p.add_argument("--edge-mode", default="saturate", choices=["saturate", "random"],
                   help="Racetrack edge model, applied to EVERY cell so layout is "
                        "the only varied axis. NB saturate -> BLOCK is fault-immune "
                        "and drives the seed-coverage policy. Default: saturate")
    p.add_argument("--base-layout", default=DEFAULT_BASE_LAYOUT, choices=["row", "col"],
                   help="storage.base_layout for block/units arms (inert for dense-* "
                        "arms, which stay on layout=col). Default is 'col' because "
                        "the dense/iso-cost control arms are COL and spec section 2's "
                        "entire cost table (wires, A@16, A@16+T, bits-immune) is "
                        "computed for base_layout=col -- 'col' is the ONLY choice "
                        "under which a block/units arm differs from its control by "
                        "layout mechanism alone and is joinable to that table. "
                        "'row' remains selectable but is a deliberate SECONDARY "
                        f"sweep, not the primary one. Default: {DEFAULT_BASE_LAYOUT}")
    p.add_argument("--metrics", default="offline", choices=["none", "offline", "online", "all"],
                   help="Runner --metrics level, applied to every cell uniformly. "
                        "'offline' (default) writes JSON static-metrics snapshots "
                        "(no .npz raw arrays) to <run_dir>/metrics_artifacts/ -- "
                        "keeps disk modest across the full sweep. Dial down to "
                        "'none' to disable, or up to 'online'/'all' for heavier "
                        "per-iteration / raw-array levels. See module docstring.")
    p.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT,
                   help=f"W&B project (all runs logged together). "
                        f"Default: {DEFAULT_WANDB_PROJECT}. Pass '' to disable W&B.")
    p.add_argument("--wandb-entity", default=None, help="W&B entity/team.")
    p.add_argument("--dry-run", action="store_true",
                   help="List every cell (with argv) and exit without executing.")
    p.add_argument("--print-commands", action="store_true",
                   help="Emit one standalone `netdrift_run.py ...` shell command per "
                        "cell (no execution). Use to run cells as separate processes / "
                        "across GPUs. Pipe to a file or GNU parallel.")
    p.add_argument("--collect-only", action="store_true",
                   help="Do NOT run anything: harvest the newest summary.json for each "
                        "cell's experiment name and write the aggregated "
                        "units_sweep_summary.csv. Use after running cells as separate "
                        "processes (e.g. via --print-commands).")
    args = p.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    curve = [float(x) for x in args.rt_curve]
    base_stem = cfg_path.stem
    wandb_project = args.wandb_project or None
    mitigations = [(k, v) for k, v in MITIGATIONS if k in args.mitigations]
    cells, dropped = _build_cells(args.seeds, mitigations, args.edge_mode)
    total = len(cells)
    n_wandb_runs = total * len(curve)

    # ---------------------------------------------------------------- header
    print("=" * 72)
    print("DENSE-vs-BLOCK-vs-UNITS layout comparison")
    print("=" * 72)
    print(f"  base config        : {cfg_path}")
    print(f"  arms                : {len(ARMS)}  ({', '.join(a[0] for a in ARMS)})")
    print(f"  mitigations         : {[k for k, _ in mitigations]}")
    print(f"  seed pool           : {args.seeds}")
    print(f"  edge_mode           : {args.edge_mode}"
          + ("  (drives seed-coverage policy; ONLY block+units-t1 are fault-"
             "immune/deterministic)" if args.edge_mode == "saturate" else
             "  (every arm needs every seed, no exceptions)"))
    print(f"  base_layout         : {args.base_layout}  (block/units arms only)")
    print(f"  rt_error curve      : {curve}")
    print(f"  loops               : {args.loops}")
    print(f"  metrics             : {args.metrics}"
          + ("  (JSON snapshots -> <run_dir>/metrics_artifacts/, no .npz)"
             if args.metrics != "none" else "  (no metrics artifacts written)"))
    print(f"  wandb_project       : {wandb_project or 'DISABLED'}")
    # Seed-policy summary: printed on EVERY invocation (not just when something
    # was dropped) so the applied policy and the cell count it produced are
    # never silent -- this is what would have caught dense-rt2 (the iso-cost
    # control) shipping with n=1 under the previous (incorrect) policy.
    n_collapsed = len(dropped)
    n_full = len(ARMS) - n_collapsed
    print(f"  [seed-policy] {n_collapsed} arm(s) collapsed to 1 seed, "
          f"{n_full} arm(s) at the full {len(args.seeds)}-seed pool "
          f"-> {total} total cell(s) (arms x mitigations x seeds actually run)")
    if dropped:
        print("  [seed-policy] collapsed arms (logged, not silent):")
        for line in dropped:
            print(f"      - {line}")
    else:
        print("  [seed-policy] no seeds dropped — every arm uses the full seed pool")
    print(f"  runner invocations  : {total}  (cells, after seed-coverage filtering)")
    print(f"  W&B runs            : {n_wandb_runs}  ({total} cells x {len(curve)} rt_error)")
    print(f"  inference passes    : {n_wandb_runs * args.loops}  "
          f"({n_wandb_runs} runs x {args.loops} loops)")
    print()

    # ------------------------------------------------------ collect-only
    # Rebuild the aggregated CSV from summaries written by cells that were run
    # as separate processes (e.g. via --print-commands). No runner import, no
    # GPU. exp_name is reconstructed via the SAME _exp_name() helper _cell_argv
    # uses, so this can never drift from the names cells were actually run
    # under.
    if args.collect_only:
        runner_out_dir = output_dir_from_cfg(cfg_path)
        if not runner_out_dir.is_absolute():
            runner_out_dir = REPO_ROOT / runner_out_dir
        results: list[dict] = []
        for cell in cells:
            exp_name = _exp_name(base_stem, cell, args.base_layout)
            summary_path = latest_summary(runner_out_dir, exp_name)
            harvested = harvest_summary(summary_path)
            results.append({
                "label": cell["label"],
                "layout": cell["layout"],
                "rt_size": cell["rt_size"],
                "units": cell["units"],
                "base_layout": args.base_layout,
                "mit_key": cell["mit_key"],
                "seed": cell["seed"],
                "exp_name": exp_name,
                "status": "ok" if summary_path else "missing",
                "elapsed_s": "",
                "baseline_clean_accuracy": harvested["baseline_clean_accuracy"],
                "rt_curve": harvested["rt_curve"],
            })
            # Print the resolved experiment.name AND the harvested summary.json
            # path for every cell. _exp_name doesn't encode --loops/--rt-curve,
            # so a --collect-only run after changing either can harvest a
            # STALE summary from an earlier invocation without any other
            # signal that it happened; printing the path is what makes that
            # visible instead of silent.
            if summary_path:
                print(f"  {exp_name:60s} -> {summary_path}")
            else:
                print(f"  {exp_name:60s} -> MISSING summary.json")
        out_dir = new_sweep_out_dir("units_collected")
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_tables(out_dir, results, curve)
        n_found = sum(1 for r in results if r["status"] == "ok")
        print(f"\nCollected {n_found}/{len(cells)} cells.")
        return 0 if n_found == len(cells) else 1

    # ------------------------------------------------------ print-commands
    # Emit each cell as a standalone `netdrift_run.py` invocation so cells can
    # be run as separate PROCESSES (the only safe way to parallelize — the
    # in-process loop shares one CUDA context; threads would collide on
    # Numba's context). Shell-quoted (see _fmt_argv) so the JSON-quoted
    # mitigation override survives a shell round-trip.
    if args.print_commands:
        runner = REPO_ROOT / "netdrift_run.py"
        for cell in cells:
            argv_cell = _cell_argv(
                cell, cfg_path, curve, args.loops, args.edge_mode, args.base_layout,
                base_stem, wandb_project, args.wandb_entity, args.metrics,
            )
            print(f"python {runner} " + _fmt_argv(argv_cell))
        return 0

    # ---------------------------------------------------------------- dry-run
    if args.dry_run:
        print("DRY-RUN: cells that would be executed:")
        print()
        for i, cell in enumerate(cells, 1):
            argv_cell = _cell_argv(
                cell, cfg_path, curve, args.loops, args.edge_mode, args.base_layout,
                base_stem, wandb_project, args.wandb_entity, args.metrics,
            )
            print(f"[{i}/{total}] {cell['label']}  mit={cell['mit_key']}  seed={cell['seed']}")
            print(f"        layout={cell['layout']} rt_size={cell['rt_size']}"
                  + (f" base_layout={args.base_layout}" if cell['layout'] in ("block", "units") else "")
                  + (f" units={cell['units']}" if cell['units'] is not None else ""))
            print(f"        argv={_fmt_argv(argv_cell)}")
            print()
        return 0

    # ---------------------------------------------------------------- run
    runner_main = import_runner_main()
    runner_out_dir = output_dir_from_cfg(cfg_path)
    if not runner_out_dir.is_absolute():
        runner_out_dir = REPO_ROOT / runner_out_dir

    out_dir = new_sweep_out_dir("units")
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_t0 = time.perf_counter()
    results: list[dict] = []

    for i, cell in enumerate(cells, 1):
        argv_cell = _cell_argv(
            cell, cfg_path, curve, args.loops, args.edge_mode, args.base_layout,
            base_stem, wandb_project, args.wandb_entity, args.metrics,
        )
        exp_name = _exp_name(base_stem, cell, args.base_layout)
        bar = "=" * 72
        print(bar)
        print(f"[cell {i}/{total}]  {cell['label']}  mit={cell['mit_key']}  "
              f"seed={cell['seed']}  (edge_mode={args.edge_mode})")
        print(bar)

        t0 = time.perf_counter()
        status, err = run_cell(runner_main, argv_cell)
        elapsed = time.perf_counter() - t0
        if status != "ok":
            print(f"  !! cell failed: {err}")

        summary_path = latest_summary(runner_out_dir, exp_name)
        harvested = harvest_summary(summary_path)
        rt_curve_data = harvested["rt_curve"]
        if rt_curve_data:
            curve_str = "  ".join(
                f"{rt:g}:{m['mean']:.2f}" for rt, m in sorted(rt_curve_data.items())
            )
            print(f"  => {status}  clean={harvested['baseline_clean_accuracy']}  "
                  f"curve(mean): [{curve_str}]  ({elapsed:.1f}s)")
        else:
            print(f"  => {status}  clean={harvested['baseline_clean_accuracy']}  ({elapsed:.1f}s)")

        results.append({
            "label": cell["label"],
            "layout": cell["layout"],
            "rt_size": cell["rt_size"],
            "units": cell["units"],
            "base_layout": args.base_layout,
            "mit_key": cell["mit_key"],
            "seed": cell["seed"],
            "exp_name": exp_name,
            "status": status,
            "error": err,
            "elapsed_s": round(elapsed, 1),
            "summary_path": str(summary_path) if summary_path else None,
            "baseline_clean_accuracy": harvested["baseline_clean_accuracy"],
            "rt_curve": rt_curve_data,
        })
        write_manifest(out_dir, {
            "study": "units_layout_sweep",
            "config": str(cfg_path),
            "seed_pool": args.seeds,
            "seed_policy_dropped": dropped,
            "mitigations": [k for k, _ in mitigations],
            "edge_mode": args.edge_mode,
            "base_layout": args.base_layout,
            "rt_curve": curve,
            "loops": args.loops,
            "metrics": args.metrics,
            "wandb_project": wandb_project,
            "results": results,
        })

    print()
    print("=" * 72)
    print(f"Sweep done. Writing table to: {out_dir}")
    print("=" * 72)
    _write_tables(out_dir, results, curve)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print()
    print(f"Total: {n_ok}/{total} cells ok  ({time.perf_counter() - sweep_t0:.1f}s)")
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
