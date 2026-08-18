#!/usr/bin/env python
"""Design-space weekend sweep: every racetrack layout on the cost-vs-robustness plane.

Implements ``docs/superpowers/specs/2026-07-31-design-space-weekend-sweep.md``.
Places dense COL, BLOCK, and the UNITS threshold ladder on ONE cost-vs-
robustness plane, joinable to the existing ``netdrift-col-vs-block`` W&B
project. Cost (x) is total racetracks over ALL 8 quantized layers,
protection-invariant by construction (spec section 1) -- you pay for a wire
whether or not its layer is exposed, so the same layout has the same x-value
under both protection policies and the plot never silently mixes two cost
definitions. Robustness (y) is mean accuracy over 30 inference loops, per
rt_error.

ARMS TABLE PROVENANCE. ``wires``/``cells``/``bits_immune_pct``/
``table_bits_per_weight_bit`` for the 13 named-ladder arms + BLOCK are copied
verbatim from the spec's section 2 markdown table, which is itself sourced
from ``runs/layout_design_space.json`` -> ``layouts.col.designs`` (base_layout
=col, all 8 layers, 12,973,440 weight bits -- the SAME 12.97M/14.79M cell
counts the spec's calibrate-stage prose cites for dense-rt2/BLOCK are the
exact ``cells`` fields of that JSON's ``"dense rt_size=2"``/``"BLOCK (own wire
per run)"`` entries, which is what establishes the JSON as the table's own
source rather than an independent re-derivation here). Two provenance notes:

* ``units-t6``: the JSON's ``designs`` array only has grid points at
  threshold in {2,3,4,8,16,32} (``isolate L>=N, pool rest G=0``) -- there is
  no ``isolate L>=6`` entry. Its ``cells`` (13,399,048) and
  ``table_bits_per_weight_bit`` (0.128) were therefore recomputed directly
  from the JSON's ``aggregate.run_hist`` using the same closed forms the
  analysis script itself uses -- ``cells = sum(next_pow2(L)*c for L>=T) +
  ceil(pooled_bits/rt_size)*rt_size`` and ``analyze_layout_design_space.py::
  _table_bits`` with ``contiguous=False, kind_bits=0``. So this row is
  measured, not interpolated, like every other row -- the closed form is exact
  and cheap, so there was no reason to carry an estimate into a table the
  paper's cost axis reads from.
* ``units-t2-g1`` vs ``units-t2-g0``: the spec's ladder table has ONE
  generic "units-t2" row (wires=3,267,147, tbl/b=0.996) which matches the
  JSON's ``"isolate L>=2, pool rest G=0"`` entry EXACTLY -- that row is
  therefore g0's data, not a shared g0/g1 value. g1 (period-2 guarded
  pooling) has its own JSON entry, ``"isolate L>=2 + phase-aware guarded
  pooling"`` (wires=3,284,906, cells=15,925,546,
  table_bits_per_weight_bit=1.126), used here instead of inventing a number.
  ``bits_immune_pct`` is 74.5% for BOTH: guarding a junction adds guard
  *cells* to the pooled wire's structure, it does not reclassify any pooled
  bit as an isolated/immune bit, so the isolated-vs-pooled bit partition (and
  therefore the immune fraction) is identical for g0 and g1 at the same
  threshold.

Dense arms carry ``bits_immune_pct=0.0`` / ``table_bits_per_weight_bit=0.0``:
they are table-free by construction (JSON note: "table-free") and, per
``sweep_units.py``'s seed-coverage-policy docstring and this branch's own
col_vs_block data, dense wires random-walk under saturate and are never
provably immune at any rt_size -- 0.0 is the correct value, not a placeholder.

MAIN-MATRIX PROTECTION AXIS -- new vs sweep_units.py. sweep_units.py has no
protection axis (it always uses the config's own default). Here the spec adds
one: ``prot-2to7`` (``policy=custom, layers=[2,3,4,5,6,7]``, the common BNN
recipe) x ``prot-1to8`` (``policy=all``). Built via
``comparison_common.base_overrides(protection_policy=, protection_layers=)``:
passing ``protection_layers`` alone implies ``policy=custom`` (its docstring),
so prot-2to7 passes both; prot-1to8 passes ``policy="all"`` alone so no
``fault.protection.layers`` override is emitted (policy=all does not take a
layer list). The token ``prot-2to7``/``prot-1to8`` is folded into both
``experiment.name`` and the W&B subcategory for every cell, matching the exact
strings already used on disk (``runs/vgg7_cifar10_w1a1_rtm__lay-col_prot-*``)
so this sweep's runs share a protection vocabulary with prior sweeps for
aggregation -- NOT by reusing those existing experiment.name/dirs themselves:
those hold already-completed loops=30 runs for dense-rt64/block at seed 707
(spec section 3, "already done, joinable, not re-run") that this driver does
not touch. Reusing their directory names here would make ``latest_summary``'s
mtime-based newest-wins harvest ambiguous between the old data and a fresh
re-run. Every exp_name in this driver is namespaced under a per-stage tag
(``dspace-<stage>``, see ``_exp_name``) that never collides with those names
OR across this driver's own preflight/calibrate/main cells for the same arm
(see NAMESPACING below).

BLOCK COLLAPSES TO EXACTLY 1 CELL -- across protection AND seed AND
mitigation, always, regardless of how many seeds/mitigations are configured
on the CLI. Under ``edge_mode=saturate`` BLOCK is fault-immune
(``tests/test_ap_saturate.py::test_block_saturate_is_fault_immune``): every
wire is one same-sign run plus a same-sign guard band, so no read is ever
wrong, for ANY protection policy, ANY seed, or ANY mitigation (a mitigation
that operates on a misalignment offset that is always zero is a no-op). This
is why the spec's cell count is ``13 arms x 2 prot x 3 seeds (+ mitigations)
+ 1 (block)``, not ``14 arms x ... + block's own share``. The collapse is
logged explicitly on every invocation (never silent), mirroring
``sweep_units.py``'s "[seed-policy]" logging convention -- see
``_build_main_cells``'s ``collapse_log`` return value and the header printer.

NAMESPACING (``_exp_name``). Every cell descriptor carries a ``stage_tag`` in
{"preflight-gate", "preflight-eq", "calibrate", "main"}, folded into
``experiment.name`` as ``dspace-<stage_tag>``. This is load-bearing, not
cosmetic: without it, a calibrate cell for "dense-rt64" (rt_error=[1e-5],
loops=1) and a preflight cell for "dense-rt64" (rt_error=[0.0], loops=1) and
a main cell for "dense-rt64" (rt_error=<curve>, loops=30) would all write to
the SAME ``<out_dir>/<exp_name>/`` tree, and ``latest_summary``'s mtime-based
"newest wins" selection could harvest a 1-loop calibration summary into the
main-matrix CSV with no signal that it happened. ``_exp_name`` is the SINGLE
implementation used by every stage's live-run path AND by ``--collect-only``
(which must reconstruct names without calling ``_cell_argv``) -- kept in one
place so the two can never drift apart, exactly per ``sweep_units.py::_exp_name``'s
own docstring rationale.

PREFLIGHT'S EXACT-EQUALITY INVARIANT -- why it is safe here specifically.
At ``fault.rt_error=0.0`` the misalignment kernel's per-cell branch
(``if rand < rt_error``) is never taken (rand is drawn from [0,1)), so the
gather/scatter is a proven numerical identity
(``tests/test_rtm_fault.py::test_zero_error_is_identity_2d``). But
``baseline_clean_accuracy`` is computed via a genuinely SEPARATE code path
(fault model fully detached, ``attach_fault_model(model, None)``) than the
rt_error=0.0 sweep point (fault model attached, full inject() path run with a
no-op kernel) -- these are only guaranteed to agree when NO weight_encoder is
attached in ``per_forward`` mode (which fires unconditionally inside
``inject()``, even at rt_error=0 -- the runner computes a SEPARATE
``baseline_endlen_accuracy`` for that case precisely because the two diverge
there) and when ``fault.mitigations`` is empty (mitigations act on
``index_offset``, which is populated regardless of rt_error). This is why
EVERY preflight cell hardcodes ``fault.weight_encoder=null`` and
``fault.mitigations=[]`` and ignores whatever ``--mitigations`` was passed for
the main matrix -- the preflight gate is a fixed, uncontaminated correctness
probe, not a cell in the mitigation study.

SHARDING -- gate mechanism (spec section 5). ``--print-commands --shards N``
emits exactly N lines, one per GPU, each shaped ``gate && ( cell1 || ec=1 ;
cell2 || ec=1 ; ... ; exit $ec )``: HARD ``&&`` gate on preflight, SOFT
continuation through the body -- every cell still runs even if an earlier one
fails (a bare ``&&`` chain through the whole body would let cell 3's crash
silently skip cells 4..N) -- while the trailing ``exit $ec`` still makes the
body's OWN exit status reflect whether ANY cell failed, not just the last
one (a bare ``;``-joined list reports only the last command's status, which
would silently mask cell 1 of 18 failing while cells 2-18 all pass -- caught
empirically while verifying the ``tee`` change below, see TERMINAL OUTPUT).
The spec's own worked example shows the gate as a bare
``preflight-ish`` command, but running the REAL 14-arm-plus-equivalence
preflight once per shard would burn N x (several minutes of GPU inference)
just to re-derive the SAME pass/fail this driver already computed once --
wasteful and, worse, a second preflight invocation racing the runner's own
timestamped run directories against the first could confuse
``latest_summary``'s mtime-based harvest. Instead: shard 0's line alone runs
``python sweep_design_space.py --stage preflight`` and, via a wrapping
subshell that preserves the real exit code
(``( cmd; rc=$?; touch OK-or-FAIL; exit $rc )`` -- needed because a bare
``cmd && touch OK || touch FAIL`` would have the WHOLE expression's exit
status become ``touch``'s (always 0) on the failure branch, silently turning
a failed gate into a passing one), writes ``PREFLIGHT_OK``/``PREFLIGHT_FAIL``
into the shard out-dir. Note ``rc=$?`` already covers a CRASHING preflight
(an uncaught Python traceback), not just a clean nonzero return: the
interpreter's default excepthook still exits the process with a nonzero
status, which the subshell captures exactly like any other failure -- no
extra handling needed there.

Shards 1..N-1 gate on a BOUNDED poll loop, not an unbounded one -- an
earlier version of this driver polled forever
(``while [ ! -e OK ] && [ ! -e FAIL ]; do sleep 15; done``), which is a real
hang risk for an unattended multi-day run: if shard 0's PROCESS itself dies
before it can write either sentinel (OOM kill, node reboot, ^C, or `python`
failing to even start) -- the one failure mode ``rc=$?`` cannot help with,
since there is no live process left to run the ``touch`` -- every other
shard would poll indefinitely, turning "one shard failed" into "every other
shard silently burns the whole weekend doing nothing." ``SHARD_GATE_TIMEOUT_S``
(2 hours; see its own comment) bounds this: after
``SHARD_GATE_MAX_POLLS`` iterations of ``sleep SHARD_GATE_POLL_INTERVAL_S``
with neither sentinel present, the poll loop prints a
``PREFLIGHT-TIMEOUT:`` line to the shard's own log and exits nonzero,
which -- combined via ``&&`` with the body -- means that shard's cells never
run, exactly like an explicit ``PREFLIGHT_FAIL``. All N lines launch
back-to-back (each ends in ``&``), so shards 1..N-1 start polling
immediately while shard 0's preflight runs; nobody re-executes the gate.

Quoting: every line is built as one PAYLOAD string (gate expression + ``&&``
+ parenthesized, ``;``-joined body, where each cell is itself pre-quoted via
``_fmt_argv``/``shlex.quote`` exactly like ``sweep_units.py``), and the ENTIRE
payload is then wrapped with a single outer ``shlex.quote`` call before being
handed to ``sh -c``. This matters because ``shlex.quote`` single-quotes any
token containing shell metacharacters (``[``, ``]``, etc. -- true of
EVERY cell here, since ``fault.rt_error=[...]`` and even the default
``fault.mitigations=[]`` contain brackets) and hand-writing an outer
``sh -c '...'`` around text that already contains single-quoted substrings
would terminate the outer quoting early and corrupt the line. Composing two
``shlex.quote`` calls (inner per-token, outer on the whole payload) is safe by
construction -- ``shlex.quote`` escapes any single quote in its input via the
POSIX ``'"'"'`` idiom, so it nests to any depth -- and is verified in this
driver's own dry-run/print-commands checks (see module tests below).

TERMINAL OUTPUT, NOT JUST A LOG FILE: each shard's ``sh -c`` payload streams
through ``tee <logfile>`` (rather than a bare ``> logfile 2>&1`` redirect) so
progress is visible live on whatever terminal launched the shard lines, not
only recoverable after the fact from the log. This reintroduces the classic
"pipe breaks the exit code" problem: POSIX ``sh`` has no ``PIPESTATUS``, so a
bare ``cmd | tee logfile`` would report ``tee``'s exit status (always 0 on a
normal write) instead of ``cmd``'s, silently turning a failed shard into a
"successful" one -- undoing the whole point of propagating a real nonzero
code through the gate/body chain above. Fix: wrap ``cmd`` in a brace group
that writes its REAL ``$?`` to a per-shard ``.rc`` file as a side effect
independent of the pipe (``echo $? > rc_file`` does not write to stdout, so
it never reaches ``tee``), pipe the group's stdout+stderr through ``tee``,
and only AFTER that pipeline finishes, ``exit "$(cat rc_file)"`` re-asserts
the real code as the ``sh -c`` invocation's own exit status. Verified
empirically (see module tests below): a stubbed failing cell still yields a
nonzero exit code from the full ``CUDA_VISIBLE_DEVICES=... sh -c ...``
line despite the ``tee``.

DEPLOYMENT TARGETS -- ``--print-commands`` vs ``--shard-index``. Two
different sharding consumers for two different places this sweep runs:

* ``--print-commands --shards N`` -- USE THIS on a single host with N GPUs.
  Emits N standalone shell lines, each ``CUDA_VISIBLE_DEVICES=k sh -c ...``,
  meant to be launched together (e.g. pasted into one shell, all ending in
  ``&``) on ONE machine where device k really is the k-th GPU.
* ``--shard-index K`` (with ``--shards N``) -- USE THIS on a cluster/scheduler
  where each submitted job gets its OWN allocation and its GPU is always
  visible as device 0 within that job (so ``CUDA_VISIBLE_DEVICES=1``,
  ``=2``, ... would point at a device that job doesn't have). Runs (or,
  with ``--dry-run``, lists) ONLY shard K's cells, IN-PROCESS, via the
  existing ``_main_live`` -- no shell quoting, no device juggling, no
  cross-job coordination, and progress goes to stdout where the job
  scheduler's own log capture already handles it. Submit ``--shards N``
  separate jobs, each with a different ``--shard-index 0..N-1``.
  ``_cells_for_shard``/``_plan_shards`` are the SINGLE partitioning
  implementation both consumers share (never a second one) -- the same
  ``--shards``/``--calibration`` inputs produce IDENTICAL partitions either
  way (verified in this module's own tests). Because preflight has typically
  already been run interactively before submitting per-shard cluster jobs,
  pair this with ``--skip-preflight`` (see below) to avoid N redundant
  14-arm gates burning cluster GPU-hours for a result already known.
  ``new_sweep_out_dir``'s timestamp alone (1-second resolution) can collide
  when a scheduler starts several jobs in the same second, so the run
  directory's TAG (not just the timestamp) folds in ``shard{K}of{N}`` --
  two concurrent jobs never contend for one ``manifest.json``.

``--skip-preflight`` bypasses the in-process gate a LIVE ``--stage main`` run
otherwise always executes first. NOT the default and NOT implied by
``--shard-index`` -- the caller must ask for both independently, and doing so
prints a loud, un-suppressible WARNING (never silent) naming what was
skipped and stating that the caller is asserting preflight already passed.
Exists for exactly the cluster scenario above: preflight is a single-GPU,
single-process check with no sharding story of its own, so it is meant to be
run ONCE (interactively, ``--stage preflight``) before submitting N
``--shard-index`` jobs, not redundantly inside each one.

Weight source for the greedy longest-processing-time-first bin-packer: this
driver shards by WHOLE ARMS (spec: "cost is homogeneous within an arm" --
every cell within an arm differs only by protection/seed/mitigation, none of
which change the kernel's launch structure). Per-cell cost = (from
``--calibration <calibration.json>`` when the arm is present there: ``fixed +
len(curve) * per_loop * loops`` at the ACTUAL main-matrix curve/loops -- see
``run_calibrate``'s docstring for why the two-point fixed/per_loop fit, not a
single loops=1 timing, is needed; else the documented fallback proxy
``arm.cells + arm.wires``), x that arm's cell COUNT in the actual matrix
being sharded (1 for block, else ``len(seeds) * len(PROTECTIONS) *
len(mitigations)``) -- NOT the per-cell weight alone, since the goal is to
balance each shard's TOTAL wall-clock, and an arm with more cells contributes
proportionally more of it. Every ``--print-commands`` or ``--shard-index``
invocation prints, prominently and un-suppressibly, which of the two weight
sources was used -- this driver never invents a per-arm SECONDS table when
no ``--calibration`` file is given; the fallback is only ever the documented
cells+wires proxy. A
``calibration.json`` from before the two-point fit (a bare
``{label: seconds}`` scalar) is rejected with a clear error rather than
silently reinterpreted as either term -- see ``_arm_weight``.

FIXED, NOT FLAGS: ``edge_mode=saturate``, ``storage.base_layout=col``,
``--metrics offline`` are module-level constants, not CLI overrides. Unlike
``sweep_units.py`` (whose ``--base-layout``/``--edge-mode`` flags select a
deliberate SECONDARY sweep alongside its primary one), this driver's entire
79-cell matrix and its ARMS cost table are defined for exactly one
edge_mode/base_layout/metrics combination (spec sections 2-3) -- exposing
these as flags would let an invocation silently produce numbers that are not
joinable to the spec's own cost table, i.e. invent an axis the spec does not
define. If a secondary edge_mode=random or base_layout=row study is ever
wanted, it is a new script (or a deliberate follow-up), not a flag here.

Stages (``--stage {preflight,calibrate,main}``, default ``main``):

* ``preflight`` -- correctness gate (spec section 4). Every one of the 14
  main-matrix arms at ``fault.rt_error=[0.0]``, ``training.loops=1``, ONE
  fixed protection (prot-2to7) and ONE fixed seed (707): faulted accuracy
  must EXACTLY equal ``baseline_clean_accuracy``. Plus a 15th, matrix-only-
  during-preflight pseudo-arm, ``units-t1`` ((1,1,0)), compared against
  ``block`` at ``rt_error=[1e-5]``, ``loops=1``, the same fixed
  protection/seed -- proving threshold=1 reproduces BLOCK. EACH of the 14
  gate cells ALSO validates the cost axis (no extra GPU work -- same cell,
  same ``summary.json``): its top-level ``n_racetracks`` (added by the
  runner alongside ``baseline_clean_accuracy``) against this driver's own
  ``ARMS.wires`` -- see ``_check_cost_axis``. EXACT match required for
  dense/block (both sides compute the identical quantity); ratio-bounded to
  ``[1.0, UNITS_PACKER_RATIO_CEILING]`` for units (``build_unit_wires`` never
  splits a fragment, so the real packer's count is provably ``>=`` the
  analytic ``ARMS`` value); ``n_racetracks`` missing/``None`` always FAILS,
  never silently skips. This subsumes ``analyze_layout_design_space.py``'s
  separate ``--verify-packer`` fidelity gate for the units designs this
  sweep actually runs, inside a run that already has to happen. Prints ONE
  PASS/FAIL table covering both the accuracy and cost-axis checks; returns 1
  if anything fails. A LIVE ``--stage main`` run executes this gate
  in-process FIRST (spec: "nothing in the matrix runs until preflight
  passes") and aborts before touching the 79-cell matrix if it fails;
  ``--dry-run``/``--print-commands``/``--collect-only`` never invoke it
  (keeps those three modes torch/numba-free).
* ``calibrate`` -- timing probe (spec section 4), not a correctness check.
  TWO cells per arm at ``rt_error=[1e-5]`` (``loops=1`` and ``loops=3``),
  timed with ``time.perf_counter()`` and fit into a linear
  ``fixed + per_loop * loops`` model (see ``run_calibrate``'s docstring for
  why a single loops=1 point -- the original design -- is biased for the
  arms that dominate the budget); writes ``{arm_label: {"fixed": ...,
  "per_loop": ...}}`` to ``calibration.json`` under a fresh
  ``runs/sweeps/<ts>_design_space_calibration/`` directory. Prints the fit
  and a projected main-matrix wall-clock that is explicitly NOT called a
  lower bound -- it is a two-point fit with one small, documented blind spot
  (per-rt_error overhead is not separately identifiable from these two
  points and is under-counted by a bounded factor).
* ``main`` -- the 79-cell matrix (13 arms x 2 protections x 3 seeds x 1
  mitigation (nomit, default) + 1 block cell = 79; scales with
  ``--seeds``/``--mitigations``).

Usage::

    # Gate first -- must PASS before anything else is trustworthy.
    python scripts/sweep_design_space.py --stage preflight

    # Timing probe, writes calibration.json.
    python scripts/sweep_design_space.py --stage calibrate

    # List the 79 main-matrix cells with full argv, no execution.
    python scripts/sweep_design_space.py --dry-run

    # Emit 4 shell lines (one per GPU), balanced by measured calibration.
    python scripts/sweep_design_space.py --print-commands --shards 4 \\
        --calibration runs/sweeps/<ts>_design_space_calibration/calibration.json

    # After running the emitted commands by hand: harvest into one CSV.
    python scripts/sweep_design_space.py --collect-only

    # Opt into the mitigation study axis too (doubles non-block cells).
    python scripts/sweep_design_space.py --mitigations nomit odd2even --dry-run

    # Cluster path: 4 separate job submissions, one per --shard-index, each
    # its own GPU allocation (visible as device 0 within that job).
    # Preflight already passed interactively above, so skip re-running it.
    python scripts/sweep_design_space.py --shards 4 --shard-index 0 \\
        --calibration runs/sweeps/<ts>_design_space_calibration/calibration.json \\
        --skip-preflight
    # ... --shard-index 1, 2, 3 as three more separate job submissions.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
from pathlib import Path
from typing import NamedTuple, Optional

# Put the scripts/ dir on sys.path so comparison_common is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparison_common import (  # noqa: E402
    REPO_ROOT,
    base_overrides,
    harvest_summary,
    import_runner_main,
    latest_summary,
    new_sweep_out_dir,
    output_dir_from_cfg,
    run_cell,
    wandb_args,
    write_manifest,
)

THIS_SCRIPT = Path(__file__).resolve()

# --------------------------------------------------------------------------
# Fixed parameters (spec sections 2-3; see module docstring "FIXED, NOT
# FLAGS" for why these are constants, not CLI overrides).
# --------------------------------------------------------------------------
DEFAULT_CONFIG = "configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml"
DEFAULT_RT_CURVE = [1e-4, 4.55e-5, 1e-5, 1e-6, 1e-7]
DEFAULT_LOOPS = 30
DEFAULT_SEEDS = [707, 808, 909]
DEFAULT_BASE_LAYOUT = "col"
DEFAULT_EDGE_MODE = "saturate"
DEFAULT_METRICS = "offline"
DEFAULT_WANDB_PROJECT = "netdrift-col-vs-block"
WANDB_CATEGORY = "design_space_sweep"

# Preflight: rt_error=0.0 correctness gate + the units-t1-vs-block
# equivalence check (spec section 4).
PREFLIGHT_GATE_CURVE = [0.0]
PREFLIGHT_GATE_LOOPS = 1
PREFLIGHT_EQUIV_RT = 1e-5
PREFLIGHT_EQUIV_CURVE = [PREFLIGHT_EQUIV_RT]
PREFLIGHT_EQUIV_LOOPS = 1

# Cost-axis check ceiling (see _check_cost_axis / run_preflight). Only the
# UNITS branch is ratio-bounded -- dense/block require EXACT n_racetracks ==
# arm.wires, since both sides compute the identical quantity and any gap is
# a real ROW/COL-convention bug, not packing slack. For UNITS,
# build_unit_wires never splits a fragment across two wires -- it flushes
# and starts a new wire whenever the next whole fragment would not fit -- so
# the real packer's wire count is provably >= the analytic ARMS value
# (actual/expected >= 1.0 is required; < 1.0 is impossible by construction
# and therefore always a bug, regardless of this ceiling). 1.20 (20% slack)
# is deliberately NOT analyze_layout_design_space.py's --verify-packer-
# threshold default of 1.02 -- that script's own help text already flags
# 1.02 as an uncalibrated guess likely too tight once threshold grows past 2
# (longer, more numerous pooled fragments -> more flush-boundary waste). 20%
# is picked to almost certainly survive the first real preflight run without
# a false FAIL (this gate must not cry wolf on day one), while still catching
# an actual bug-sized gap (e.g. 2x from a mis-derived cell-vs-wire quantity).
# It is NOT a measured bound -- re-tighten it from the ratios this gate
# itself prints on the first real GPU run, the same way that script's own
# docstring recommends recalibrating its threshold.
UNITS_PACKER_RATIO_CEILING = 1.20

# Calibrate: TWO timing points per arm (loops=1 and loops=3, same single
# rt_error point), fit into a fixed/per_loop decomposition -- see
# run_calibrate's docstring for why a single point at loops=1 (the original
# design) is biased for the heaviest arms.
CALIBRATE_RT_CURVE = [1e-5]
CALIBRATE_LOOPS = 1
CALIBRATE_LOOPS_HI = 3

# Shard-gate poll bound (see module docstring "SHARDING"). The 16-cell
# preflight itself is normally minutes (loops=1 everywhere), but it includes
# `block` and `units-t1` -- 6.5M wires apiece, and (per the calibrate-stage
# discussion) BLOCK's per-bucket kernel launch structure is real overhead
# beyond what wire count alone predicts -- so "minutes" is a typical case,
# not a guarantee. 2 hours is picked to sit generously above the worst
# plausible HEALTHY preflight (so a slow-but-alive shard 0 is never falsely
# killed) while still bounding an unattended weekend run: a shard 0 that
# dies outright (OOM, node reboot, ^C, `python` failing to start) releases
# the other shards within this window instead of hanging until someone
# notices on Monday.
SHARD_GATE_POLL_INTERVAL_S = 15
SHARD_GATE_TIMEOUT_S = 2 * 60 * 60  # 2 hours
SHARD_GATE_MAX_POLLS = SHARD_GATE_TIMEOUT_S // SHARD_GATE_POLL_INTERVAL_S  # 480


class Arm(NamedTuple):
    """One row of the cost-vs-robustness ladder (spec section 2).

    ``units`` is ``(threshold, max_period, pool_guard)`` or ``None`` for
    dense/block arms (``storage.units.*`` left untouched). ``wires``/
    ``cells``/``bits_immune_pct``/``table_bits_per_weight_bit`` are carried
    as DATA (see module docstring "ARMS TABLE PROVENANCE") -- used as the
    shard-planner fallback weight and printed in the header so the cost axis
    is visible without a separate lookup into
    ``runs/layout_design_space.json``.
    """

    label: str
    layout: str  # "col" | "units" | "block"
    rt_size: int
    units: Optional[tuple[int, int, int]]
    wires: int
    cells: int
    bits_immune_pct: float
    table_bits_per_weight_bit: float


# 14 main-matrix arms, spec section 2's iso-cost ladder + BLOCK. Values are
# spec-table-exact except units-t6 (cells/table_bits_per_weight_bit computed
# directly from the JSON's aggregate.run_hist via the analysis script's own
# closed forms -- measured, not interpolated; see module docstring) and
# units-t2-g1 (its own JSON entry, not the shared "units-t2" row -- see
# module docstring). max_period=2 is only legal at threshold=2
# (code/python/netdrift/config/schema.py:UnitsCfg.__post_init__), which is
# why no other (threshold, 2, *) tuple appears below.
ARMS: list[Arm] = [
    Arm("dense-rt64",  "col",   64, None,        203_574, 13_028_736,  0.0,   0.0),
    Arm("dense-rt32",  "col",   32, None,        406_124, 12_995_968,  0.0,   0.0),
    Arm("dense-rt16",  "col",   16, None,        811_224, 12_979_584,  0.0,   0.0),
    Arm("dense-rt8",   "col",    8, None,      1_622_448, 12_979_584,  0.0,   0.0),
    Arm("dense-rt4",   "col",    4, None,      3_243_872, 12_975_488,  0.0,   0.0),
    Arm("dense-rt2",   "col",    2, None,      6_486_720, 12_973_440,  0.0,   0.0),  # iso-cost control vs BLOCK
    Arm("units-t16",   "units", 64, (16, 1, 0),   203_043, 12_977_104,  0.1,   0.001),
    Arm("units-t8",    "units", 64, (8, 1, 0),    249_994, 13_147_408,  3.9,   0.043),
    Arm("units-t6",    "units", 64, (6, 1, 0),    386_379, 13_399_048, 11.3,   0.128),
    Arm("units-t4",    "units", 64, (4, 1, 0),    939_639, 13_993_748, 31.1,   0.374),
    Arm("units-t3",    "units", 64, (3, 1, 0),  1_697_612, 14_788_960, 49.5,   0.621),
    Arm("units-t2-g1", "units", 64, (2, 2, 1),  3_284_906, 15_925_546, 74.5,   1.126),  # period-2 guarded pooling
    Arm("units-t2-g0", "units", 64, (2, 1, 0),  3_267_147, 14_788_970, 74.5,   0.996),  # ablation: guards off
    Arm("block",       "block", 64, None,       6_527_245, 14_788_944, 100.0,  1.000),  # immune anchor, 1 cell
]
ARMS_BY_LABEL: dict[str, Arm] = {a.label: a for a in ARMS}

# Protection arms (new vs sweep_units.py; see module docstring). Order
# matches the spec's own listing (prot-2to7 first).
PROTECTIONS: list[dict] = [
    {"key": "prot-2to7", "policy": "custom", "layers": [2, 3, 4, 5, 6, 7], "tag": "prot-2to7"},
    {"key": "prot-1to8", "policy": "all",    "layers": None,               "tag": "prot-1to8"},
]

# Mitigation axis (spec section 3): nomit by default, odd2even opt-in via
# --mitigations. "nomit" must stay [] (renders to the valid JSON literal
# "[]"). Mirrors sweep_units.py::MITIGATIONS exactly.
MITIGATIONS: list[tuple[str, list[str]]] = [
    ("nomit", []),
    ("odd2even", ["odd2even_dec"]),
]
DEFAULT_MITIGATIONS = ["nomit"]


def mitigation_override(names: list[str]) -> str:
    """Render a mitigation list for --override as valid JSON (list[str]).

    Duplicated from ``scripts/sweep_units.py::mitigation_override`` rather
    than imported -- sibling sweep drivers are standalone over
    ``comparison_common``, which is the only shared module (task
    constraint: do not modify ``sweep_units.py``). The reasoning is
    identical and empirically verified there: the override parser does a
    plain ``json.loads`` on the value
    (code/python/netdrift/config/loader.py:_parse_override_value), falling
    back to a bare *string* on a JSON parse failure, so an UNQUOTED
    ``[odd2even_dec]`` silently resolves to the string
    ``"[odd2even_dec]"`` rather than the list ``["odd2even_dec"]``.
    """
    return "[" + ",".join(f'"{n}"' for n in names) + "]"


def _cell_ap(arm: Arm, ap_position: Optional[int]) -> Optional[int]:
    """Resolve the access-port override for one arm, or ``None`` for "don't pin".

    ``None`` (no ``--ap-position``) leaves every arm on the config default, so
    names and behaviour are byte-identical to a pre-flag invocation.

    Returns ``None`` for units/block arms even when ``ap_position`` is set:
    those layouts pack heterogeneous per-bucket racetrack lengths ``P`` and
    resolve the port per bucket as ``P//2 - 1``, so a single absolute index is
    meaningless. The fault model rejects it outright
    (code/python/netdrift/faults/rtm_misalignment.py: "ap_position is not
    supported with BLOCK or units mapping"), so pinning them would kill 8 of
    the 14 arms at config time. They stay on the per-bucket default and, being
    unpinned, carry NO token -- each cell's name states exactly what it ran.
    """
    if ap_position is None or arm.layout in ("units", "block"):
        return None
    return int(ap_position)


def ap_token(ap: Optional[int]) -> str:
    """Name segment for a pinned access port: ``""`` when unpinned, else ``_ap<N>``.

    Deliberately EMPTY by default. A token is only emitted for cells that
    actually received a ``fault.ap_position`` override, so:

    * default runs keep their canonical names and stay joinable with the
      artifacts already on disk (``--collect-only`` reconstructs the same
      strings -- see ``_exp_name``);
    * a name never asserts an AP that the run did not pin. A default-valued
      token (e.g. ``_apmid``) would go false the moment the resolution rule
      changes underneath it, and a wrong token is worse than none: absence
      makes a reader check, a confident-but-stale token stops them checking.

    ⚠️ MERGE HAZARD, deliberately NOT papered over here. On this branch an
    unset ``fault.ap_position`` resolves mid-wire (``rt_size//2 - 1``); on the
    ap0 branch it resolves to ``0`` (low edge), which roughly halves the rate
    at which racetracks leave the aligned state. Default-named artifacts from
    the two branches are therefore NOT comparable despite identical names, and
    ``comparison_common.latest_summary()`` picks by mtime. Before mixing eras,
    re-tag or re-run -- do not rely on the name. The AP change is a measured
    no-op for ``block`` (immune under saturate) but NOT for the units arms,
    whose pooled mixed-sign wires are fully exposed to port position.
    """
    return "" if ap is None else f"_ap{ap}"


def validate_ap_position(ap_position: Optional[int]) -> Optional[str]:
    """Return an error string if ``ap_position`` cannot apply to the dense ladder.

    The port index must lie in ``[0, rt_size-1]`` for EVERY dense arm, because
    the main matrix always runs all of them (there is no arm filter). The fault
    model would otherwise silently clamp (``if ap > rt_size - 1: ap = rt_size - 1``),
    so ``--ap-position 31`` would run at 31 on dense-rt64 but at 1 on dense-rt4
    while every cell carried an identical ``_ap31`` token -- a name asserting an
    AP the run never used. Reject instead of clamping.
    """
    if ap_position is None:
        return None
    ap = int(ap_position)
    if ap < 0:
        return f"--ap-position must be >= 0; got {ap}"
    offenders = [a for a in ARMS if a.layout not in ("units", "block") and ap > a.rt_size - 1]
    if offenders:
        worst = min(a.rt_size for a in offenders)
        return (
            f"--ap-position {ap} exceeds the racetrack length of "
            f"{len(offenders)} dense arm(s): "
            + ", ".join(f"{a.label} (max {a.rt_size - 1})" for a in offenders)
            + f".\n  The main matrix runs the whole dense ladder, so a pinned index must be "
              f"valid at the SHORTEST wire (rt_size={worst} => max index {worst - 1}). "
              f"The fault model would clamp silently, making the _ap{ap} token lie on the "
              f"short arms.\n  This is why the spec leaves ap_position unset: an absolute "
              f"index is not meaningful across a ladder whose rt_size varies 64->2 (the "
              f"default tracks mid-wire per arm). Use --ap-position 0 (the low edge, "
              f"well-defined at every rt_size) or drop the flag."
        )
    return None


def _fmt_argv(argv: list[str]) -> str:
    """Shell-quote an argv list for display/emission (see sweep_units.py::_fmt_argv).

    Needed because ``fault.mitigations=["odd2even_dec"]`` and
    ``fault.rt_error=[...]`` contain characters (``"``, ``[``, ``]``) that a
    bare ``' '.join(argv)`` would lose across a shell round-trip.
    """
    return " ".join(shlex.quote(a) for a in argv)


# --------------------------------------------------------------------------
# Cell construction
# --------------------------------------------------------------------------

def _exp_name(base_stem: str, cell: dict) -> str:
    """Deterministic per-cell experiment name -- the SINGLE source used by
    every stage's live-run path AND by --collect-only (which reconstructs it
    without calling _cell_argv). See module docstring "NAMESPACING" for why
    the stage_tag segment is load-bearing, not cosmetic.
    """
    arm: Arm = cell["arm"]
    return (
        f"{base_stem}__dspace-{cell['stage_tag']}__{arm.label}"
        f"__{cell['mit_key']}__{cell['prot_tag']}__seed{cell['seed']}"
        f"{ap_token(cell.get('ap'))}"
    )


def _subcategory(cell: dict) -> str:
    """W&B subcategory: fine-grained combo, mirrors sweep_units.py's convention."""
    arm: Arm = cell["arm"]
    return (
        f"{arm.label}_{cell['mit_key']}_{cell['prot_tag']}_seed{cell['seed']}"
        f"{ap_token(cell.get('ap'))}"
    )


def _cell_argv(
    cell: dict,
    cfg_path: Path,
    base_stem: str,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
) -> list[str]:
    """Assemble the full runner argv for one cell (any stage).

    Overrides ONLY the compared axes (layout, rt_size, units.*, protection,
    mitigation) plus the invariants that must not drift (no encoder,
    edge_mode, seed, metrics level) -- mirrors sweep_units.py::_cell_argv.
    """
    arm: Arm = cell["arm"]
    argv = ["--config", str(cfg_path)]

    argv += base_overrides(
        curve=cell["curve"],
        loops=cell["loops"],
        protection_policy=cell["policy"],
        protection_layers=cell["layers"],
    )

    # Layout axis.
    argv += ["--override", f"storage.layout={arm.layout}"]
    argv += ["--override", f"storage.rt_size={arm.rt_size}"]
    if arm.layout in ("block", "units"):
        argv += ["--override", f"storage.base_layout={DEFAULT_BASE_LAYOUT}"]
    if arm.units is not None:
        t, mp, pg = arm.units
        argv += ["--override", f"storage.units.threshold={t}"]
        argv += ["--override", f"storage.units.max_period={mp}"]
        argv += ["--override", f"storage.units.pool_guard={pg}"]

    # Invariants: no encoder, mitigation list, fixed edge model + seed.
    argv += [
        "--override", "fault.weight_encoder=null",
        "--override", f"fault.mitigations={mitigation_override(cell['mit_list'])}",
        "--override", f"fault.edge_mode={DEFAULT_EDGE_MODE}",
        "--override", f"experiment.seed={cell['seed']}",
    ]

    # Access port: only ever emitted for cells _cell_ap pinned (dense arms
    # under --ap-position). Unpinned cells omit the override entirely, so the
    # config default applies and no token is added -- see ap_token().
    if cell.get("ap") is not None:
        argv += ["--override", f"fault.ap_position={cell['ap']}"]

    # Metrics level: a direct runner CLI flag, fixed (see module docstring).
    argv += ["--metrics", DEFAULT_METRICS]

    exp_name = _exp_name(base_stem, cell)
    argv += ["--override", f"experiment.name={exp_name}"]

    argv += wandb_args(wandb_project, wandb_entity, WANDB_CATEGORY, _subcategory(cell))
    return argv


def _build_main_cells(
    seeds: list[int],
    mitigations: list[tuple[str, list[str]]],
    curve: list[float],
    loops: int,
    ap_position: Optional[int] = None,
) -> tuple[list[dict], list[str]]:
    """Return (79-at-defaults cell descriptors, block-collapse log lines).

    See module docstring "BLOCK COLLAPSES TO EXACTLY 1 CELL" -- block always
    gets exactly one cell (first protection, first seed, first/nomit
    mitigation), regardless of how many seeds/protections/mitigations this
    invocation configures for every other arm.
    """
    cells: list[dict] = []
    collapse_log: list[str] = []
    for arm in ARMS:
        if arm.label == "block":
            prot = PROTECTIONS[0]
            mit_key, mit_list = mitigations[0] if mitigations else MITIGATIONS[0]
            seed = seeds[0]
            cells.append({
                "arm": arm, "prot_tag": prot["tag"], "policy": prot["policy"],
                "layers": prot["layers"], "mit_key": mit_key, "mit_list": mit_list,
                "seed": seed, "curve": curve, "loops": loops, "stage_tag": "main",
                "ap": _cell_ap(arm, ap_position),
            })
            n_would_be = len(PROTECTIONS) * len(seeds) * len(mitigations)
            collapse_log.append(
                f"block: collapsed {n_would_be} -> 1 cell "
                f"(kept prot={prot['tag']} seed={seed} mit={mit_key}); immune to "
                f"protection/seed/mitigation under edge_mode={DEFAULT_EDGE_MODE} "
                f"(tests/test_ap_saturate.py::test_block_saturate_is_fault_immune)"
            )
            continue
        for prot in PROTECTIONS:
            for seed in seeds:
                for mit_key, mit_list in mitigations:
                    cells.append({
                        "arm": arm, "prot_tag": prot["tag"], "policy": prot["policy"],
                        "layers": prot["layers"], "mit_key": mit_key, "mit_list": mit_list,
                        "seed": seed, "curve": curve, "loops": loops, "stage_tag": "main",
                        "ap": _cell_ap(arm, ap_position),
                    })
    return cells, collapse_log


def _preflight_gate_cells(ap_position: Optional[int] = None) -> list[dict]:
    """14 cells: every main-matrix arm at rt_error=[0.0], loops=1, ONE fixed
    protection+seed, weight_encoder=null + mitigations=[] forced (see module
    docstring "PREFLIGHT'S EXACT-EQUALITY INVARIANT").
    """
    prot = PROTECTIONS[0]
    seed = DEFAULT_SEEDS[0]
    cells = []
    for arm in ARMS:
        cells.append({
            "arm": arm, "prot_tag": prot["tag"], "policy": prot["policy"],
            "layers": prot["layers"], "mit_key": "nomit", "mit_list": [],
            "seed": seed, "curve": PREFLIGHT_GATE_CURVE, "loops": PREFLIGHT_GATE_LOOPS,
            "stage_tag": "preflight-gate",
            "ap": _cell_ap(arm, ap_position),
        })
    return cells


def _preflight_equiv_cells(ap_position: Optional[int] = None) -> tuple[dict, dict]:
    """units-t1 vs block at rt_error=1e-5, loops=1, same fixed protection+seed.

    units-t1 ((1,1,0)) is NOT a main-matrix arm (spec section 2) -- it exists
    only to prove threshold=1 reproduces BLOCK, so it is built here directly
    rather than added to ARMS.
    """
    prot = PROTECTIONS[0]
    seed = DEFAULT_SEEDS[0]
    units_t1 = Arm("units-t1", "units", 64, (1, 1, 0), 0, 0, 100.0, 1.0)
    common = {
        "prot_tag": prot["tag"], "policy": prot["policy"], "layers": prot["layers"],
        "mit_key": "nomit", "mit_list": [], "seed": seed,
        "curve": PREFLIGHT_EQUIV_CURVE, "loops": PREFLIGHT_EQUIV_LOOPS,
        "stage_tag": "preflight-eq",
    }
    # Both arms are units/block, so _cell_ap pins neither (per-bucket port) --
    # the equivalence check is unaffected by --ap-position by construction.
    block_arm = ARMS_BY_LABEL["block"]
    return (
        {**common, "arm": units_t1, "ap": _cell_ap(units_t1, ap_position)},
        {**common, "arm": block_arm, "ap": _cell_ap(block_arm, ap_position)},
    )


def _calibrate_cells(loops: int, tag_suffix: str,
                     ap_position: Optional[int] = None) -> list[dict]:
    """14 cells: one per arm at rt_error=[1e-5], ONE fixed protection+seed, at
    the given ``loops`` count. Called twice (loops=1 and loops=3) so
    ``run_calibrate`` can separate fixed setup cost from marginal per-loop
    cost -- see that function's docstring. ``tag_suffix`` (``"l1"``/``"l3"``)
    keeps the two loop-count variants' experiment names distinct (module
    docstring "NAMESPACING") even though neither is ever harvested by
    --collect-only (calibrate has no CSV story) -- cheap to keep consistent
    with the rest of the driver's naming invariant.
    """
    prot = PROTECTIONS[0]
    seed = DEFAULT_SEEDS[0]
    cells = []
    for arm in ARMS:
        cells.append({
            "arm": arm, "prot_tag": prot["tag"], "policy": prot["policy"],
            "layers": prot["layers"], "mit_key": "nomit", "mit_list": [],
            "seed": seed, "curve": CALIBRATE_RT_CURVE, "loops": loops,
            "stage_tag": f"calibrate-{tag_suffix}",
            "ap": _cell_ap(arm, ap_position),
        })
    return cells


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

def _print_header(
    args: argparse.Namespace,
    cfg_path: Path,
    curve: list[float],
    loops: int,
    seeds: list[int],
    mitigations: list[tuple[str, list[str]]],
) -> None:
    main_cells, collapse_log = _build_main_cells(
        seeds, mitigations, curve, loops, getattr(args, "ap_position", None)
    )
    n_main = len(main_cells)
    n_wandb_main = n_main * len(curve)
    n_passes_main = n_wandb_main * loops

    print("=" * 78)
    print("DESIGN-SPACE WEEKEND SWEEP -- every racetrack layout, cost vs robustness")
    print("=" * 78)
    print(f"  base config          : {cfg_path}")
    print(f"  stage                : {args.stage}")
    print(f"  edge_mode            : {DEFAULT_EDGE_MODE}  (fixed constant, not a flag)")
    print(f"  base_layout          : {DEFAULT_BASE_LAYOUT}  (fixed constant, not a flag; block/units arms only)")
    print(f"  metrics              : {DEFAULT_METRICS}  (fixed constant, not a flag)")
    if getattr(args, "ap_position", None) is None:
        print(f"  ap_position          : unset  (config default per arm; NO name token)")
    else:
        print(f"  ap_position          : {args.ap_position}  -> token '{ap_token(args.ap_position)}' "
              f"on DENSE arms only (units/block resolve per bucket, unpinned + untokened)")
    print(f"  wandb_project        : {args.wandb_project or 'DISABLED'}   category={WANDB_CATEGORY}")
    print()
    print(f"  {'arm':14s} {'layout':7s} {'units(t,mp,g)':14s} {'wires':>10s} {'cells':>11s} "
          f"{'immune%':>8s} {'tbl/bit':>8s}")
    for arm in ARMS:
        u = str(arm.units) if arm.units else "-"
        print(f"  {arm.label:14s} {arm.layout:7s} {u:14s} {arm.wires:>10,d} {arm.cells:>11,d} "
              f"{arm.bits_immune_pct:>7.1f}% {arm.table_bits_per_weight_bit:>8.3f}")
    print()
    print(f"  protections          : {[p['tag'] for p in PROTECTIONS]}")
    print(f"  seed pool            : {seeds}")
    print(f"  mitigations          : {[k for k, _ in mitigations]}")
    print(f"  rt_error curve (main): {curve}   loops={loops}")
    print()
    print(f"  MAIN MATRIX          : {n_main} cells  (79 at defaults: 13 arms x "
          f"{len(PROTECTIONS)} prot x {len(seeds)} seeds x {len(mitigations)} mit + 1 block)")
    for line in collapse_log:
        print(f"    [block-collapse] {line}")
    print(f"  W&B runs (main)      : {n_wandb_main}  ({n_main} cells x {len(curve)} rt_error)")
    print(f"  inference passes     : {n_passes_main}  ({n_wandb_main} runs x {loops} loops)")
    if getattr(args, "skip_preflight", False):
        print(f"  NOTE: --skip-preflight is set -- a LIVE '--stage main' run will SKIP the "
              f"preflight gate (spec section 4 default: nothing in the matrix runs until "
              f"preflight passes). The caller is asserting it already passed elsewhere.")
    else:
        print(f"  NOTE: a LIVE '--stage main' run FIRST executes the full preflight gate in-process "
              f"(spec section 4: nothing in the matrix runs until preflight passes) unless "
              f"--skip-preflight is given; --dry-run / --print-commands / --collect-only never "
              f"invoke it (stay torch/numba-free).")
    if getattr(args, "shard_index", None) is not None:
        print(f"  THIS INVOCATION targets shard {args.shard_index}/{args.shards} only "
              f"(--shard-index) -- see the SHARD line below for its arms/cells/share.")
    print()


# --------------------------------------------------------------------------
# Stage: preflight
# --------------------------------------------------------------------------

def _read_n_racetracks(summary_path: Optional[Path]) -> Optional[int]:
    """Read the TOP-LEVEL ``n_racetracks`` key straight from a cell's
    ``summary.json`` (a plain ``json.load``, deliberately NOT routed through
    ``comparison_common.harvest_summary`` -- shared with five other drivers,
    and out of scope to extend for one cost-axis check).

    ``run.py`` writes this at the same top level as ``baseline_clean_accuracy``
    / ``rt_error_sweep`` (test-mode write site, ~run.py:1512), computed once
    per process via ``_total_racetracks`` and guarded so a packer failure can
    never take the run down -- it is ``None`` when that computation was
    skipped or raised. Do NOT confuse this with the unrelated, differently-
    shaped ``"n_racetracks"`` list nested inside each snapshot's ``per_layer``
    block under ``layer_metrics_by_rt_error`` (a per-layer static-metrics
    array from a completely different computation) -- this function reads
    ONLY the scalar cost-axis total.

    Returns ``None`` (never raises) on a missing file, unparseable JSON, or
    an absent/null key -- every one of those is treated as FAIL by
    ``_check_cost_axis``, not silently skipped (see its docstring).
    """
    if summary_path is None or not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            data = json.load(f)
    except Exception:
        return None
    return data.get("n_racetracks")


def _check_cost_axis(arm: Arm, n_racetracks: Optional[int]) -> tuple[bool, str, str]:
    """Compare a cell's measured ``n_racetracks`` against ``arm.wires``.

    Pure predicate (no I/O) so it is directly unit-testable without a GPU --
    feed it a synthetic ``(arm, n_racetracks)`` pair. Returns
    ``(passed, kind_label, detail)``.

    Rule (see ``UNITS_PACKER_RATIO_CEILING``'s comment for the ratio bound's
    rationale):

    * ``n_racetracks is None`` (key missing OR explicitly null in
      ``summary.json``) -> always FAIL, regardless of layout. A cost-axis
      check that silently PASSES when the value is simply absent is worse
      than no check at all -- that is the exact vacuous-pass shape this gate
      exists to prevent (the same principle behind the rt_error=0.0 gate
      failing closed on a missing ``baseline_clean_accuracy``).
    * dense (``layout=="col"``) / ``block`` -> EXACT equality. Both sides
      compute the identical quantity (this driver's ``ARMS`` table vs
      ``_total_racetracks``'s live computation over the real model), so any
      gap means ``compute_index_offset_shape``'s ROW/COL convention disagrees
      with ``analyze_layout_design_space.py``'s ``rows_cols`` orientation --
      a real bug to catch before 11,850 passes, not slack to tolerate.
    * ``units`` -> ratio-bounded, not exact. ``build_unit_wires`` never
      splits a fragment across two wires -- it flushes and starts a new one
      whenever the next whole fragment would not fit -- so the real packer's
      wire count is provably ``>=`` the analytic ``ARMS`` value. Ratio
      ``< 1.0`` is impossible by construction and therefore always a bug
      regardless of the ceiling; ratio ``> UNITS_PACKER_RATIO_CEILING`` fails
      too, on the (unproven, deliberately generous) assumption that packing
      overhead this large signals something other than ordinary flush waste.
    """
    if n_racetracks is None:
        return False, "cost-axis: n_racetracks present", "n_racetracks missing/None in summary.json"
    if arm.layout in ("col", "block"):
        passed = (n_racetracks == arm.wires)
        detail = "" if passed else f"n_racetracks={n_racetracks:,} != arm.wires={arm.wires:,}"
        return passed, "cost-axis: n_racetracks==wires (exact)", detail
    # units
    ratio = n_racetracks / arm.wires
    passed = 1.0 <= ratio <= UNITS_PACKER_RATIO_CEILING
    detail = (f"ratio={ratio:.4f} (n_racetracks={n_racetracks:,}, arm.wires={arm.wires:,}, "
              f"ceiling={UNITS_PACKER_RATIO_CEILING})")
    if not passed:
        reason = ("ratio<1.0, impossible by construction -> real bug" if ratio < 1.0
                  else f"ratio>{UNITS_PACKER_RATIO_CEILING} ceiling")
        detail += f"  FAIL: {reason}"
    return passed, "cost-axis: ratio bounded (units)", detail


def run_preflight(
    cfg_path: Path,
    base_stem: str,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    dry_run: bool,
    ap_position: Optional[int] = None,
) -> int:
    gate_cells = _preflight_gate_cells(ap_position)
    eq_t1, eq_block = _preflight_equiv_cells(ap_position)
    all_cells = gate_cells + [eq_t1, eq_block]

    print("-" * 78)
    print(f"PREFLIGHT: {len(gate_cells)} correctness-gate cells "
          f"(rt_error={PREFLIGHT_GATE_CURVE}, loops={PREFLIGHT_GATE_LOOPS}) + "
          f"2 equivalence cells (units-t1 vs block @ rt_error={PREFLIGHT_EQUIV_RT}, "
          f"loops={PREFLIGHT_EQUIV_LOOPS})")
    print("  weight_encoder=null, mitigations=[] forced on every preflight cell regardless of "
          "--mitigations (exact-equality only holds without a per_forward encoder or "
          "mitigations -- see module docstring).")
    print("  Each of the 14 gate cells ALSO validates the cost axis: summary.json's top-level "
          "n_racetracks vs this driver's own ARMS.wires (exact for dense/block, ratio-bounded "
          f"[1.0, {UNITS_PACKER_RATIO_CEILING}] for units -- see _check_cost_axis). This runs "
          "inside cells already executed for the accuracy gate, at no extra GPU cost, and "
          "subsumes analyze_layout_design_space.py's separate --verify-packer fidelity check "
          "for the units designs this sweep actually exercises.")
    print("-" * 78)

    if dry_run:
        for i, cell in enumerate(all_cells, 1):
            argv_cell = _cell_argv(cell, cfg_path, base_stem, wandb_project, wandb_entity)
            print(f"[{i}/{len(all_cells)}] {cell['arm'].label}  ({cell['stage_tag']})  "
                  f"rt_error={cell['curve']}  loops={cell['loops']}")
            print(f"        argv={_fmt_argv(argv_cell)}")
        return 0

    runner_main = import_runner_main()
    runner_out_dir = output_dir_from_cfg(cfg_path)
    if not runner_out_dir.is_absolute():
        runner_out_dir = REPO_ROOT / runner_out_dir

    def _run_one(cell: dict):
        argv_cell = _cell_argv(cell, cfg_path, base_stem, wandb_project, wandb_entity)
        exp_name = _exp_name(base_stem, cell)
        status, err = run_cell(runner_main, argv_cell)
        summary_path = latest_summary(runner_out_dir, exp_name)
        harvested = harvest_summary(summary_path)
        return status, err, harvested, summary_path

    # rows: (label, kind, expected, actual, passed, detail). expected/actual
    # are floats for the accuracy checks, ints for the cost-axis check --
    # both render fine through the same generic print loop below.
    rows: list[tuple[str, str, object, object, bool, str]] = []
    all_pass = True

    for cell in gate_cells:
        status, err, harvested, summary_path = _run_one(cell)
        clean = harvested["baseline_clean_accuracy"]
        rt0 = harvested["rt_curve"].get(0.0)
        if status != "ok" or clean is None or rt0 is None or len(rt0["accuracies"]) != 1:
            passed = False
            actual = None
            detail = err or "missing/malformed rt_curve[0.0] or baseline_clean_accuracy"
        else:
            actual = rt0["accuracies"][0]
            passed = (actual == clean)
            detail = "" if passed else f"clean={clean} faulted={actual}"
        all_pass = all_pass and passed
        rows.append((cell["arm"].label, "gate: faulted(rt=0.0)==clean", clean, actual, passed, detail))

        # Cost-axis check (coordinator follow-up): piggybacks on this same
        # cell/summary.json, no extra GPU work. See _check_cost_axis for the
        # exact-vs-ratio-bounded rule and _read_n_racetracks for why this is
        # a plain json.load rather than a harvest_summary() extension.
        n_rt = _read_n_racetracks(summary_path) if status == "ok" else None
        cost_passed, cost_kind, cost_detail = _check_cost_axis(cell["arm"], n_rt)
        all_pass = all_pass and cost_passed
        rows.append((cell["arm"].label, cost_kind, cell["arm"].wires, n_rt, cost_passed, cost_detail))

    status_t1, err_t1, harv_t1, _sp_t1 = _run_one(eq_t1)
    status_b, err_b, harv_b, _sp_b = _run_one(eq_block)
    rt_t1 = harv_t1["rt_curve"].get(PREFLIGHT_EQUIV_RT)
    rt_b = harv_b["rt_curve"].get(PREFLIGHT_EQUIV_RT)
    if (status_t1 != "ok" or status_b != "ok" or rt_t1 is None or rt_b is None
            or len(rt_t1["accuracies"]) != 1 or len(rt_b["accuracies"]) != 1):
        eq_pass = False
        a_t1 = a_b = None
        eq_detail = f"units-t1_status={status_t1} err={err_t1}; block_status={status_b} err={err_b}"
    else:
        a_t1 = rt_t1["accuracies"][0]
        a_b = rt_b["accuracies"][0]
        eq_pass = (a_t1 == a_b)
        eq_detail = "" if eq_pass else f"units-t1={a_t1} block={a_b}"
    all_pass = all_pass and eq_pass
    rows.append(("units-t1 vs block", f"equiv: units-t1==block @ rt={PREFLIGHT_EQUIV_RT}",
                 a_b, a_t1, eq_pass, eq_detail))

    def _fmt_cell(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.6f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    print()
    print(f"  {'check':22s} {'kind':32s} {'expected':>14s} {'actual':>14s}  result  detail")
    for label, kind, expected, actual, passed, detail in rows:
        print(f"  {label:22s} {kind:32s} {_fmt_cell(expected):>14s} {_fmt_cell(actual):>14s}  "
              f"{'PASS' if passed else 'FAIL'}    {detail}")

    n_fail = sum(1 for r in rows if not r[4])
    print()
    if all_pass:
        print(f"PREFLIGHT: PASS ({len(rows)}/{len(rows)} checks)")
        return 0
    print(f"PREFLIGHT: FAIL ({len(rows) - n_fail}/{len(rows)} checks passed, {n_fail} failed)")
    return 1


# --------------------------------------------------------------------------
# Stage: calibrate
# --------------------------------------------------------------------------

def run_calibrate(
    cfg_path: Path,
    base_stem: str,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    dry_run: bool,
    main_seeds: list[int],
    main_mitigations: list[tuple[str, list[str]]],
    ap_position: Optional[int] = None,
) -> int:
    """Two-point timing probe: fixed setup cost vs marginal per-loop cost.

    A SINGLE measurement at loops=1 (the original design) cannot separate
    these two terms, and for the arms that dominate the main-matrix budget
    that is not a rounding error: model/dataset/checkpoint construction PLUS
    ``_total_racetracks``'s packer pass over the real weights (BLOCK/UNITS
    branches -- ``run.py:875-971``, strictly outside the rt_error loop that
    starts at ``run.py:~1297``) run exactly ONCE PER PROCESS, i.e. once per
    CELL, never once per loop or per rt_error point. For ``block``/
    ``units-t2`` (3.3-6.5M wires) that fixed pass is large. Naively scaling a
    loops=1 measurement by the main matrix's loops (e.g. x30) multiplies that
    ENTIRE fixed cost by 30 too, wildly over-estimating exactly the arms this
    projection matters most for.

    Fix: time each arm at loops=1 (``t1``) AND loops=3 (``t3``), same single
    rt_error point, and solve the 2x2 system this implies for a linear
    cost-per-loop model: ``per_loop = (t3-t1)/2``, ``fixed = t1 - per_loop``.
    ``per_loop`` is clamped to ``>= 0`` (a negative value is measurement
    noise, not a real per-loop refund -- most likely on a cheap arm where t1
    and t3 are both small and dominated by run-to-run process-startup jitter,
    NOT the packer's real fixed cost); when the clamp fires, ``fixed``
    collapses to plain ``t1`` and the arm is flagged (``*``) in the printed
    table rather than silently reporting ``per_loop=0`` (which would make
    that arm look free to scale to loops=30).

    What this DOES capture: the fixed/per_loop split at ONE rt_error point.
    What it does NOT: the main-matrix curve sweeps
    ``len(DEFAULT_RT_CURVE)=5`` rt_error points per cell, each paying its own
    (small) reset/reseed/W&B-run-creation overhead -- two calibration points
    at a fixed rt_error-count of 1 cannot separately identify that per-
    rt_error overhead, so it is folded into ``fixed`` at 1x and under-counted
    by ``(n_rt_error-1)x`` in the projection below. That is a much smaller
    bias than the one being removed here (per-rt_error overhead is reset/
    reseed/logging, not a multi-million-wire packer pass), so this is not
    labelled a "lower bound" -- it is a two-point linear fit with one known,
    small, and explained blind spot, not a bound in either direction.
    """
    cells_lo = _calibrate_cells(CALIBRATE_LOOPS, "l1", ap_position)
    cells_hi = _calibrate_cells(CALIBRATE_LOOPS_HI, "l3", ap_position)
    print("-" * 78)
    print(f"CALIBRATE: {len(cells_lo)} arms x 2 timing points each "
          f"(loops={CALIBRATE_LOOPS} and loops={CALIBRATE_LOOPS_HI}, rt_error="
          f"{CALIBRATE_RT_CURVE}) = {len(cells_lo) + len(cells_hi)} cells. "
          f"Timing probe, not a correctness check.")
    print("-" * 78)

    if dry_run:
        for i, (c_lo, c_hi) in enumerate(zip(cells_lo, cells_hi), 1):
            argv_lo = _cell_argv(c_lo, cfg_path, base_stem, wandb_project, wandb_entity)
            argv_hi = _cell_argv(c_hi, cfg_path, base_stem, wandb_project, wandb_entity)
            print(f"[{i}/{len(cells_lo)}] {c_lo['arm'].label}")
            print(f"        loops={CALIBRATE_LOOPS}  argv={_fmt_argv(argv_lo)}")
            print(f"        loops={CALIBRATE_LOOPS_HI}  argv={_fmt_argv(argv_hi)}")
        return 0

    runner_main = import_runner_main()
    t1_by_arm: dict[str, float] = {}
    t3_by_arm: dict[str, float] = {}

    print(f"-- pass 1/2: loops={CALIBRATE_LOOPS} --")
    for i, cell in enumerate(cells_lo, 1):
        argv_cell = _cell_argv(cell, cfg_path, base_stem, wandb_project, wandb_entity)
        t0 = time.perf_counter()
        status, err = run_cell(runner_main, argv_cell)
        elapsed = time.perf_counter() - t0
        t1_by_arm[cell["arm"].label] = elapsed
        print(f"[{i}/{len(cells_lo)}] {cell['arm'].label:14s} {status:12s} {elapsed:8.1f}s"
              + (f"  !! {err}" if status != "ok" else ""))

    print(f"-- pass 2/2: loops={CALIBRATE_LOOPS_HI} --")
    for i, cell in enumerate(cells_hi, 1):
        argv_cell = _cell_argv(cell, cfg_path, base_stem, wandb_project, wandb_entity)
        t0 = time.perf_counter()
        status, err = run_cell(runner_main, argv_cell)
        elapsed = time.perf_counter() - t0
        t3_by_arm[cell["arm"].label] = elapsed
        print(f"[{i}/{len(cells_hi)}] {cell['arm'].label:14s} {status:12s} {elapsed:8.1f}s"
              + (f"  !! {err}" if status != "ok" else ""))

    calibration: dict[str, dict[str, float]] = {}
    clamped: list[str] = []
    for arm in ARMS:
        t1 = t1_by_arm.get(arm.label, 0.0)
        t3 = t3_by_arm.get(arm.label, 0.0)
        n_loops_delta = CALIBRATE_LOOPS_HI - CALIBRATE_LOOPS
        per_loop_raw = (t3 - t1) / n_loops_delta
        per_loop = max(0.0, per_loop_raw)
        if per_loop != per_loop_raw:
            clamped.append(arm.label)
        fixed = t1 - per_loop
        calibration[arm.label] = {"fixed": fixed, "per_loop": per_loop}

    out_dir = new_sweep_out_dir("design_space_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_path = out_dir / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\n  -> {cal_path}")

    print()
    print("Calibration (fixed + per_loop, seconds), sorted by fixed cost ascending:")
    print(f"  {'arm':14s} {'t1':>8s} {'t3':>8s} {'fixed':>10s} {'per_loop':>10s}")
    for label, entry in sorted(calibration.items(), key=lambda kv: kv[1]["fixed"]):
        flag = " *" if label in clamped else ""
        print(f"  {label:14s} {t1_by_arm.get(label, 0.0):8.1f} {t3_by_arm.get(label, 0.0):8.1f} "
              f"{entry['fixed']:10.1f} {entry['per_loop']:10.2f}{flag}")
    if clamped:
        print(f"  * = t3 <= t1 (per_loop clamped to 0; fixed=t1 as-measured) -- likely "
              f"run-to-run noise dominating a small marginal cost on a cheap arm, not a "
              f"real negative per-loop cost: {clamped}")

    n_prot = len(PROTECTIONS)
    n_seeds = len(main_seeds)
    n_mit = len(main_mitigations)
    n_rt = len(DEFAULT_RT_CURVE)
    total = 0.0
    for arm in ARMS:
        entry = calibration[arm.label]
        per_cell_full = entry["fixed"] + n_rt * entry["per_loop"] * DEFAULT_LOOPS
        n_cells_arm = 1 if arm.label == "block" else n_prot * n_seeds * n_mit
        total += per_cell_full * n_cells_arm

    print()
    print(f"Projected main-matrix wall-clock: {total:.0f}s  ({total / 3600:.2f}h) sequential.")
    print(f"  = sum over arms of [fixed + {n_rt} rt_error x per_loop x {DEFAULT_LOOPS} loops] "
          f"x that arm's main-matrix cell count ({n_prot} prot x {n_seeds} seeds x {n_mit} mit, "
          f"block=1). 'fixed' is paid once per CELL (not per rt_error point, per the packer-cost "
          f"reasoning above) and is NOT multiplied by {n_rt}; 'per_loop' IS, since every one of "
          f"the {n_rt} rt_error points independently runs {DEFAULT_LOOPS} loops. Known blind spot: "
          f"per-rt_error overhead (reset/reseed/W&B-run-creation) is not separable from these two "
          f"points and is under-counted by ({n_rt}-1)x -- small next to the fixed-cost bias this "
          f"replaces, not zero. Not a lower bound in either direction; a two-point linear fit with "
          f"one documented, small blind spot.")
    return 0


# --------------------------------------------------------------------------
# Stage: main -- dry-run / collect-only / print-commands / live
# --------------------------------------------------------------------------

def _write_main_table(out_dir: Path, results: list[dict], curve: list[float]) -> None:
    """Write the aggregated CSV. Columns include the cost axis (wires/cells/
    bits_immune_pct, from the ARMS table) alongside the per-cell schema.
    """
    csv_path = out_dir / "design_space_summary.csv"
    rt_cols = [f"rt_{rt:g}_mean" for rt in curve]
    fieldnames = (
        ["arm", "layout", "rt_size", "base_layout", "threshold", "max_period", "pool_guard",
         "protection", "mitigation", "seed", "ap", "status", "elapsed_s",
         "baseline_clean_accuracy", "experiment_name", "wires", "cells", "bits_immune_pct"]
        + rt_cols
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            arm: Arm = r["arm"]
            rt_curve = r.get("rt_curve") or {}
            row = {
                "arm": arm.label,
                "layout": arm.layout,
                "rt_size": arm.rt_size,
                "base_layout": DEFAULT_BASE_LAYOUT if arm.layout in ("block", "units") else "",
                "threshold": arm.units[0] if arm.units else "",
                "max_period": arm.units[1] if arm.units else "",
                "pool_guard": arm.units[2] if arm.units else "",
                "protection": r["prot_tag"],
                "mitigation": r["mit_key"],
                "seed": r["seed"],
                # "" (not 0) when unpinned -- 0 is a REAL port index here.
                "ap": "" if r.get("ap") is None else r["ap"],
                "status": r["status"],
                "elapsed_s": r.get("elapsed_s", ""),
                "baseline_clean_accuracy": r.get("baseline_clean_accuracy"),
                "experiment_name": r["exp_name"],
                "wires": arm.wires,
                "cells": arm.cells,
                "bits_immune_pct": arm.bits_immune_pct,
            }
            for rt in curve:
                m = rt_curve.get(float(rt))
                row[f"rt_{rt:g}_mean"] = f"{m['mean']:.4f}" if m else ""
            w.writerow(row)
    print(f"  -> {csv_path}")


def _main_dry_run(
    cfg_path: Path, base_stem: str, wandb_project: Optional[str], wandb_entity: Optional[str],
    seeds: list[int], mitigations: list[tuple[str, list[str]]], curve: list[float], loops: int,
    shards: int = 1, shard_index: Optional[int] = None,
    calibration_path: Optional[Path] = None,
    ap_position: Optional[int] = None,
) -> int:
    cells, collapse_log = _build_main_cells(seeds, mitigations, curve, loops, ap_position)
    if shard_index is not None:
        calibration, missing = _load_calibration(calibration_path)
        if missing:
            print(f"WARNING: --calibration is missing arms {missing}; falling back to the "
                  f"cells+wires proxy for those specifically.")
        shard_arms, shard_load, _ = _plan_shards(shards, seeds, mitigations, calibration, curve, loops)
        cells = _cells_for_shard(cells, shard_arms, shard_index)
        weight = shard_load[shard_index]
        weight_str = (f"{weight:,.0f}s (~{weight / 3600:.2f}h) of projected wall-clock "
                       f"(measured, from {calibration_path})" if calibration else
                       f"{weight:,.0f} cells+wires proxy units (no --calibration -- NOT seconds)")
        print(f"DRY-RUN: shard {shard_index}/{shards} ONLY -- "
              f"arms={[a.label for a in shard_arms[shard_index]]}  "
              f"cells={len(cells)}  share={weight_str}")
    print("DRY-RUN: main-matrix cells that would be executed:")
    for line in collapse_log:
        print(f"  [block-collapse] {line}")
    print()
    for i, cell in enumerate(cells, 1):
        arm: Arm = cell["arm"]
        argv_cell = _cell_argv(cell, cfg_path, base_stem, wandb_project, wandb_entity)
        print(f"[{i}/{len(cells)}] {arm.label}  prot={cell['prot_tag']}  "
              f"mit={cell['mit_key']}  seed={cell['seed']}")
        print(f"        layout={arm.layout} rt_size={arm.rt_size}"
              + (f" base_layout={DEFAULT_BASE_LAYOUT}" if arm.layout in ("block", "units") else "")
              + (f" units={arm.units}" if arm.units else "")
              + f"  wires={arm.wires:,} cells={arm.cells:,} immune={arm.bits_immune_pct}%")
        print(f"        argv={_fmt_argv(argv_cell)}")
        print()
    return 0


def _main_collect_only(
    cfg_path: Path, base_stem: str, seeds: list[int],
    mitigations: list[tuple[str, list[str]]], curve: list[float], loops: int,
    ap_position: Optional[int] = None,
) -> int:
    cells, _ = _build_main_cells(seeds, mitigations, curve, loops, ap_position)
    runner_out_dir = output_dir_from_cfg(cfg_path)
    if not runner_out_dir.is_absolute():
        runner_out_dir = REPO_ROOT / runner_out_dir

    results: list[dict] = []
    for cell in cells:
        exp_name = _exp_name(base_stem, cell)
        summary_path = latest_summary(runner_out_dir, exp_name)
        harvested = harvest_summary(summary_path)
        results.append({
            "arm": cell["arm"], "prot_tag": cell["prot_tag"], "mit_key": cell["mit_key"],
            "seed": cell["seed"], "ap": cell.get("ap"), "exp_name": exp_name,
            "status": "ok" if summary_path else "missing",
            "elapsed_s": "",
            "baseline_clean_accuracy": harvested["baseline_clean_accuracy"],
            "rt_curve": harvested["rt_curve"],
        })
        if summary_path:
            print(f"  {exp_name:70s} -> {summary_path}")
        else:
            print(f"  {exp_name:70s} -> MISSING summary.json")

    out_dir = new_sweep_out_dir("design_space_collected")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_main_table(out_dir, results, curve)
    n_found = sum(1 for r in results if r["status"] == "ok")
    print(f"\nCollected {n_found}/{len(cells)} cells.")
    return 0 if n_found == len(cells) else 1


def _main_live(
    cfg_path: Path, base_stem: str, wandb_project: Optional[str], wandb_entity: Optional[str],
    seeds: list[int], mitigations: list[tuple[str, list[str]]], curve: list[float], loops: int,
    shards: int = 1, shard_index: Optional[int] = None,
    calibration_path: Optional[Path] = None,
    ap_position: Optional[int] = None,
) -> int:
    cells, collapse_log = _build_main_cells(seeds, mitigations, curve, loops, ap_position)
    out_dir_tag = "design_space"
    shard_arms_i: Optional[list[Arm]] = None
    if shard_index is not None:
        calibration, missing = _load_calibration(calibration_path)
        if missing:
            print(f"WARNING: --calibration is missing arms {missing}; falling back to the "
                  f"cells+wires proxy for those specifically.")
        shard_arms, shard_load, _ = _plan_shards(shards, seeds, mitigations, calibration, curve, loops)
        cells = _cells_for_shard(cells, shard_arms, shard_index)
        shard_arms_i = shard_arms[shard_index]
        weight = shard_load[shard_index]
        weight_str = (f"{weight:,.0f}s (~{weight / 3600:.2f}h) of projected wall-clock "
                       f"(measured, from {calibration_path})" if calibration else
                       f"{weight:,.0f} cells+wires proxy units (no --calibration -- NOT seconds)")
        print(f"SHARD {shard_index}/{shards}: arms={[a.label for a in shard_arms_i]}  "
              f"cells={len(cells)}  share={weight_str}")
        # Fold the shard index into the out-dir TAG (not just relying on the
        # timestamp): new_sweep_out_dir() timestamps at 1-second resolution,
        # and cluster-scheduled jobs routinely start within the same second
        # of each other -- two concurrent shards writing ONE manifest.json
        # would corrupt both. A distinct tag per shard makes the directory
        # name unique even when the timestamp collides.
        out_dir_tag = f"design_space_shard{shard_index}of{shards}"

    total = len(cells)
    print(f"MAIN: running {total} cells.")
    for line in collapse_log:
        print(f"  [block-collapse] {line}")

    runner_main = import_runner_main()
    runner_out_dir = output_dir_from_cfg(cfg_path)
    if not runner_out_dir.is_absolute():
        runner_out_dir = REPO_ROOT / runner_out_dir

    out_dir = new_sweep_out_dir(out_dir_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_t0 = time.perf_counter()
    results: list[dict] = []

    for i, cell in enumerate(cells, 1):
        arm: Arm = cell["arm"]
        argv_cell = _cell_argv(cell, cfg_path, base_stem, wandb_project, wandb_entity)
        exp_name = _exp_name(base_stem, cell)
        bar = "=" * 72
        print(bar)
        print(f"[cell {i}/{total}]  {arm.label}  prot={cell['prot_tag']}  "
              f"mit={cell['mit_key']}  seed={cell['seed']}")
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
            "arm": arm, "prot_tag": cell["prot_tag"], "mit_key": cell["mit_key"],
            "seed": cell["seed"], "ap": cell.get("ap"), "exp_name": exp_name, "status": status, "error": err,
            "elapsed_s": round(elapsed, 1),
            "summary_path": str(summary_path) if summary_path else None,
            "baseline_clean_accuracy": harvested["baseline_clean_accuracy"],
            "rt_curve": rt_curve_data,
        })
        write_manifest(out_dir, {
            "study": "design_space_sweep",
            "config": str(cfg_path),
            "seed_pool": seeds,
            "mitigations": [k for k, _ in mitigations],
            "edge_mode": DEFAULT_EDGE_MODE,
            "base_layout": DEFAULT_BASE_LAYOUT,
            "rt_curve": curve,
            "loops": loops,
            "metrics": DEFAULT_METRICS,
            "wandb_project": wandb_project,
            "shard_index": shard_index,
            "shards": shards if shard_index is not None else None,
            "shard_arms": [a.label for a in shard_arms_i] if shard_arms_i is not None else None,
            "results": results,
        })

    print()
    print("=" * 72)
    print(f"Sweep done. Writing table to: {out_dir}")
    print("=" * 72)
    _write_main_table(out_dir, results, curve)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print()
    print(f"Total: {n_ok}/{total} cells ok  ({time.perf_counter() - sweep_t0:.1f}s)")
    return 0 if n_ok == total else 1


# --------------------------------------------------------------------------
# Sharding / command emission (spec section 5; see module docstring
# "SHARDING -- gate mechanism").
# --------------------------------------------------------------------------

def _arm_main_cell_count(arm: Arm, n_seeds: int, n_prot: int, n_mit: int) -> int:
    return 1 if arm.label == "block" else n_seeds * n_prot * n_mit


def _arm_weight(
    arm: Arm, n_cells: int, calibration: Optional[dict], curve: list[float], loops: int,
) -> tuple[float, str]:
    """Per-arm TOTAL weight (per-cell cost x that arm's cell count) + the
    source used ("measured" or "fallback"). Never invents a per-arm seconds
    table (task constraint) -- the only two sources are a real calibration
    file or the documented cells+wires proxy.

    ``calibration[arm.label]`` is ``{"fixed": ..., "per_loop": ...}`` (the
    two-point fit from ``run_calibrate`` -- see its docstring). Per-cell cost
    at the ACTUAL main-matrix curve/loops is ``fixed + len(curve) * per_loop
    * loops``: ``fixed`` (model/checkpoint/packer construction) is paid once
    per cell regardless of how many rt_error points that cell sweeps, while
    ``per_loop`` scales with the total loop-iterations across the whole cell
    (every rt_error point independently runs ``loops`` iterations). This is
    a real accuracy improvement over reusing the calibrate-stage's raw
    loops=1 timing as a stand-in for a loops=30 cell, which is the same bias
    ``run_calibrate``'s two-point fit exists to remove.

    A legacy SCALAR calibration.json (pre-two-point-fit format) is rejected
    with a clear error rather than silently reinterpreted as either term --
    guessing ``fixed=0`` would UNDER-weight exactly the packer-dominated arms
    this fix targets, silently reintroducing the bug being fixed.
    """
    if calibration and arm.label in calibration:
        entry = calibration[arm.label]
        if not isinstance(entry, dict) or "fixed" not in entry or "per_loop" not in entry:
            raise ValueError(
                f"--calibration entry for {arm.label!r} is {entry!r}, not a "
                f"{{'fixed': ..., 'per_loop': ...}} dict. This looks like a calibration.json "
                f"from before the two-point timing fit -- re-run 'sweep_design_space.py "
                f"--stage calibrate' to regenerate it in the current format; a stale scalar "
                f"cannot be safely reinterpreted as either term (guessing fixed=0 would "
                f"under-weight exactly the packer-dominated arms this format change fixes)."
            )
        per_cell = float(entry["fixed"]) + len(curve) * float(entry["per_loop"]) * loops
        return per_cell * n_cells, "measured"
    per_cell = float(arm.cells + arm.wires)
    return per_cell * n_cells, "fallback(cells+wires)"


def _load_calibration(calibration_path: Optional[Path]) -> tuple[Optional[dict], list[str]]:
    """Load a calibration.json and report which ARMS labels it's missing (those
    fall back to the cells+wires proxy in ``_arm_weight``).

    Shared by ``--print-commands`` and ``--shard-index`` (both live and
    ``--dry-run``) so every caller that needs a shard partition loads
    ``--calibration`` the SAME way -- no duplicated parsing/validation, and
    the two paths can never disagree about what "missing" means. Callers
    format their own warning message (``#``-prefixed for --print-commands's
    emitted-script output, plain elsewhere) since the right prefix depends on
    where the message is printed, not on how the file was loaded.
    """
    if calibration_path is None:
        return None, []
    with open(calibration_path) as f:
        calibration = json.load(f)
    missing = [a.label for a in ARMS if a.label not in calibration]
    return calibration, missing


def _cells_for_shard(
    all_cells: list[dict], shard_arms: list[list[Arm]], shard_index: int,
) -> list[dict]:
    """Filter ``all_cells`` (from ``_build_main_cells``) down to one shard's
    arms, preserving ``ARMS`` order.

    The SINGLE cell-selection logic reused by ``--print-commands`` (building
    every shard's body), ``--shard-index`` (live, one shard), and
    ``--dry-run --shard-index`` (listing one shard) -- so none of the three
    can ever partition the 79 cells differently for the same
    inputs+``--calibration`` file (task constraint: do not invent a second
    partitioning path).
    """
    cells_by_arm: dict[str, list[dict]] = {}
    for c in all_cells:
        cells_by_arm.setdefault(c["arm"].label, []).append(c)
    cells: list[dict] = []
    for arm in shard_arms[shard_index]:
        cells += cells_by_arm.get(arm.label, [])
    return cells


def _plan_shards(
    n_shards: int, seeds: list[int], mitigations: list[tuple[str, list[str]]],
    calibration: Optional[dict], curve: list[float], loops: int,
) -> tuple[list[list[Arm]], list[float], list[str]]:
    """Greedy longest-processing-time-first bin-packing, by WHOLE ARMS."""
    n_prot = len(PROTECTIONS)
    n_seeds = len(seeds)
    n_mit = len(mitigations)
    weighted = []
    sources = []
    for arm in ARMS:
        n_cells_arm = _arm_main_cell_count(arm, n_seeds, n_prot, n_mit)
        w, source = _arm_weight(arm, n_cells_arm, calibration, curve, loops)
        weighted.append((w, arm))
        sources.append(source)
    weighted.sort(key=lambda x: -x[0])

    shard_load = [0.0] * n_shards
    shard_arms: list[list[Arm]] = [[] for _ in range(n_shards)]
    for w, arm in weighted:
        idx = min(range(n_shards), key=lambda i: shard_load[i])
        shard_load[idx] += w
        shard_arms[idx].append(arm)

    order = {a.label: i for i, a in enumerate(ARMS)}
    for lst in shard_arms:
        lst.sort(key=lambda a: order[a.label])

    return shard_arms, shard_load, sources


def _emit_shard_commands(
    cfg_path: Path, base_stem: str, wandb_project: Optional[str], wandb_entity: Optional[str],
    curve: list[float], loops: int, seeds: list[int], mitigations: list[tuple[str, list[str]]],
    n_shards: int, calibration_path: Optional[Path],
    ap_position: Optional[int] = None,
) -> None:
    calibration, missing = _load_calibration(calibration_path)
    if missing:
        print(f"# WARNING: --calibration is missing arms {missing}; falling back to the "
              f"cells+wires proxy for those specifically.")

    shard_arms, shard_load, sources = _plan_shards(
        n_shards, seeds, mitigations, calibration, curve, loops,
    )
    weight_source = ("measured seconds from " + str(calibration_path)) if calibration_path else \
        "fallback proxy (cells+wires per arm x that arm's cell count) -- no --calibration given"

    all_cells, _ = _build_main_cells(seeds, mitigations, curve, loops, ap_position)

    out_dir = new_sweep_out_dir("design_space_shards")
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_file = out_dir / "PREFLIGHT_OK"
    fail_file = out_dir / "PREFLIGHT_FAIL"

    print(f"# design-space main-matrix sharding: {n_shards} shard(s)  out_dir={out_dir}")
    print(f"# NOTE: --print-commands is the SINGLE multi-GPU HOST path (CUDA_VISIBLE_DEVICES=0.."
          f"{n_shards - 1} on one machine). On a cluster where each job gets its own allocation "
          f"(GPU always visible as device 0 within that job), use --shard-index instead -- see "
          f"module docstring.")
    print(f"# WARNING: shard balance is ESTIMATED, not measured wall-clock truth. "
          f"Weight source: {weight_source}.")
    print(f"# Gate mechanism (see module docstring 'SHARDING'): shard 0 alone runs "
          f"'sweep_design_space.py --stage preflight' ONCE and writes "
          f"{ok_file.name}/{fail_file.name} into {out_dir}; shards 1..{n_shards - 1} poll for "
          f"one of those two sentinel files instead of redundantly re-running the 14-arm gate.")
    print(f"# Poll bound: {SHARD_GATE_MAX_POLLS} x {SHARD_GATE_POLL_INTERVAL_S}s "
          f"(~{SHARD_GATE_TIMEOUT_S / 3600:.0f}h) -- if neither sentinel appears within that "
          f"window (shard 0 died without writing one), the waiting shard logs "
          f"'PREFLIGHT-TIMEOUT:' and aborts non-zero rather than polling forever.")
    print(f"# Each shard's output streams to BOTH the terminal (so progress is visible live) "
          f"AND its own log file, via 'tee' -- exit status is re-asserted after the pipe (see "
          f"comment below) so a failed shard still reports non-zero despite the pipe.")
    for i, (arms_i, load_i) in enumerate(zip(shard_arms, shard_load)):
        print(f"#   shard {i}: weight={load_i:,.0f}  arms={[a.label for a in arms_i]}")

    runner = REPO_ROOT / "netdrift_run.py"
    preflight_argv = [
        "python", str(THIS_SCRIPT), "--config", str(cfg_path), "--stage", "preflight",
    ]
    if wandb_project:
        preflight_argv += ["--wandb-project", wandb_project]
    if wandb_entity:
        preflight_argv += ["--wandb-entity", wandb_entity]
    preflight_cmd = _fmt_argv(preflight_argv)

    for i in range(n_shards):
        cells_i = _cells_for_shard(all_cells, shard_arms, i)
        cell_cmds = [
            "python " + str(runner) + " " + _fmt_argv(
                _cell_argv(cell, cfg_path, base_stem, wandb_project, wandb_entity)
            )
            for cell in cells_i
        ]
        if cell_cmds:
            # `;`-joined commands report only the LAST one's exit status --
            # cell1 failing with cell2..N passing would make the WHOLE body
            # (and therefore this shard's line) report success, silently
            # masking an earlier failure. `CMD || ec=1` after each cell
            # preserves the SOFT semantics (every cell still runs regardless
            # of prior failures -- `||` only short-circuits the `ec=1`
            # assignment, never the next cell in the `;`-joined list) while
            # making the body's own exit status "1 if ANY cell failed, else
            # 0" via the trailing `exit $ec`. Found and fixed while verifying
            # the tee change below -- an independent, pre-existing gap in
            # the body's aggregate exit status, not something tee introduced.
            guarded = " ; ".join(f"{c} || ec=1" for c in cell_cmds)
            body = f"( ec=0; {guarded}; exit $ec )"
        else:
            body = "( : )"

        if i == 0:
            # Runs the real gate ONCE; the wrapping subshell re-asserts the
            # preflight exit code via `exit $rc` because a bare
            # `cmd && touch OK || touch FAIL` would make the WHOLE
            # expression's exit status be touch's (always 0) on the failure
            # branch, silently turning a failed gate into a passing one.
            # This already covers a CRASHING preflight (uncaught Python
            # traceback), not just a clean nonzero return -- the interpreter
            # exits the process with a nonzero status either way, and rc=$?
            # captures that regardless of cause. It does NOT cover shard 0's
            # process being killed outright before this line finishes
            # running (OOM/reboot/^C) -- nothing here can help with that,
            # which is exactly what the bounded poll below (shards 1..N-1)
            # exists to bound.
            gate = (
                f"( {preflight_cmd}; rc=$?; "
                f"if [ $rc -eq 0 ]; then touch {shlex.quote(str(ok_file))}; "
                f"else touch {shlex.quote(str(fail_file))}; fi; exit $rc )"
            )
        else:
            # Bounded poll (see module docstring "SHARDING" and the
            # SHARD_GATE_* constants' comment): an unbounded
            # `while ... ; do sleep 15; done` would hang forever if shard 0's
            # PROCESS dies before writing either sentinel -- the one failure
            # mode the shard-0 gate above cannot self-report, since a dead
            # process cannot run its own `touch`. POSIX sh (no bashisms:
            # `$((...))` arithmetic expansion, not `((...))` or `let`) so
            # this runs correctly under `sh -c`, not just bash.
            gate = (
                f"( n=0; "
                f"while [ ! -e {shlex.quote(str(ok_file))} ] && "
                f"[ ! -e {shlex.quote(str(fail_file))} ]; do "
                f"n=$((n+1)); "
                f"if [ $n -ge {SHARD_GATE_MAX_POLLS} ]; then "
                f'echo "PREFLIGHT-TIMEOUT: sentinel never appeared after $n polls '
                f"(~{SHARD_GATE_TIMEOUT_S}s); aborting shard without running its cells "
                f'(shard 0 may have died before writing a sentinel file)." >&2; '
                f"exit 1; "
                f"fi; "
                f"sleep {SHARD_GATE_POLL_INTERVAL_S}; "
                f"done; "
                f"[ -e {shlex.quote(str(ok_file))} ] )"
            )

        cmd = f"{gate} && {body}"
        log_path = out_dir / f"shard{i}.log"
        rc_path = out_dir / f"shard{i}.rc"
        # Stream to BOTH the terminal and a log file via `tee` (the user
        # asked to see progress live, not just in a file) -- but a bare
        # `cmd | tee logfile` breaks exit-status propagation: POSIX sh has no
        # PIPESTATUS, so the PIPELINE's exit status becomes tee's (always 0
        # on a normal write), silently turning a failed shard into a
        # "successful" one. Fix: the brace-group writes cmd's REAL $? to
        # rc_path as a side effect independent of the pipe (`echo ... > file`
        # does not write to stdout, so it never reaches tee), then AFTER the
        # pipe finishes, `exit "$(cat rc_path)"` re-asserts that real code as
        # this sh -c invocation's own exit status. rc_path is unique per
        # shard (under this invocation's own out_dir) so concurrent shards on
        # the same host never collide on it.
        payload = (
            f"{{ {cmd}; echo $? > {shlex.quote(str(rc_path))}; }} 2>&1 | "
            f"tee {shlex.quote(str(log_path))}; exit \"$(cat {shlex.quote(str(rc_path))})\""
        )
        # The ENTIRE payload is quoted with ONE outer shlex.quote call, not a
        # hand-written `sh -c '...'`: individual cell argvs already contain
        # shlex.quote-produced single-quoted substrings (e.g. around
        # `fault.rt_error=[...]`), and shlex.quote composes safely to any
        # nesting depth (it escapes embedded single quotes via the POSIX
        # '"'"' idiom) while a hand-written outer '...' would terminate early
        # on the first inner quote and corrupt the line.
        line = f"CUDA_VISIBLE_DEVICES={i} sh -c {shlex.quote(payload)} &"
        print(line)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG,
                    help=f"Base YAML (single source for all arms). Default: {DEFAULT_CONFIG}")
    p.add_argument("--stage", choices=["preflight", "calibrate", "main"], default="main",
                    help="Which stage to run. preflight=correctness gate (spec section 4), "
                         "calibrate=timing probe, main=the 79-cell matrix. Default: main.")
    p.add_argument("--rt-curve", nargs="+", type=float, default=DEFAULT_RT_CURVE,
                    help="rt_error values swept inside each MAIN-stage cell (preflight/"
                         f"calibrate hardcode their own curve per spec). Default: {DEFAULT_RT_CURVE}")
    p.add_argument("--loops", type=int, default=DEFAULT_LOOPS,
                    help=f"Inference iterations per rt_error, MAIN stage only. Default: {DEFAULT_LOOPS}")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help=f"Seed pool for the MAIN matrix. Default: {DEFAULT_SEEDS}")
    p.add_argument("--mitigations", nargs="+", choices=[k for k, _ in MITIGATIONS],
                    default=list(DEFAULT_MITIGATIONS),
                    help="Mitigation arms for the MAIN matrix. Default: nomit only "
                         "(spec section 3 -- odd2even is opt-in, a secondary study).")
    p.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT,
                    help=f"W&B project. Default: {DEFAULT_WANDB_PROJECT} (joins existing "
                         f"col-vs-block runs). Pass '' to disable W&B.")
    p.add_argument("--wandb-entity", default=None, help="W&B entity/team.")
    p.add_argument("--dry-run", action="store_true",
                    help="List every cell for the selected --stage (with argv) and exit "
                         "without executing. Torch/numba-free.")
    p.add_argument("--print-commands", action="store_true",
                    help="MAIN stage only. Emit --shards standalone shell lines (one per "
                         "GPU) instead of executing anything. Torch/numba-free.")
    p.add_argument("--shards", type=int, default=1,
                    help="Number of partitions the shard planner computes. Consumed by "
                         "--print-commands (emits this many shell lines) and --shard-index "
                         "(selects one of them to run in-process). Default: 1.")
    p.add_argument("--shard-index", type=int, default=None, metavar="K",
                    help="0-based index into --shards partitions. Runs (live) or lists "
                         "(--dry-run) ONLY that shard's cells IN-PROCESS via _main_live -- no "
                         "shell quoting, no CUDA_VISIBLE_DEVICES, no cross-job coordination. "
                         "THE CLUSTER PATH: submit --shards N separate jobs, each with "
                         "--shard-index 0..N-1 -- every job gets its own allocation where the "
                         "GPU is visible as device 0, so CUDA_VISIBLE_DEVICES juggling (what "
                         "--print-commands does) is wrong there. Reuses the SAME _plan_shards "
                         "partition as --print-commands for the same inputs + --calibration "
                         "(never a second partitioning path). Requires --shards > 1. Pair with "
                         "--skip-preflight once preflight has already passed interactively -- "
                         "see that flag.")
    p.add_argument("--calibration", type=Path, default=None,
                    help="Path to a calibration.json (from --stage calibrate) used as the "
                         "shard-planner's per-arm weight source (consumed by --print-commands "
                         "and --shard-index alike). Without it, falls back to the documented "
                         "cells+wires proxy (see module docstring).")
    p.add_argument("--collect-only", action="store_true",
                    help="MAIN stage only. Do NOT run anything: harvest summaries for the "
                         "79-cell matrix's experiment names and write the aggregated CSV. "
                         "Torch/numba-free.")
    p.add_argument("--skip-preflight", action="store_true",
                    help="Bypass the in-process preflight gate before a LIVE '--stage main' "
                         "run (dry-run/print-commands/collect-only never ran it anyway). For "
                         "when preflight has ALREADY passed in a prior interactive run (e.g. "
                         "once on a shared/login node before submitting per-shard cluster jobs "
                         "via --shard-index) and re-running the 14-arm gate on every job would "
                         "waste GPU time. NOT the default, and NOT implied by --shard-index -- "
                         "must be requested explicitly. Prints a loud WARNING naming what was "
                         "skipped; never silent.")
    p.add_argument("--ap-position", type=int, default=None, metavar="N",
                    help="Pin the racetrack access port to absolute index N on the DENSE arms "
                         "and append an '_ap<N>' token to their experiment.name and W&B "
                         "subcategory. Omitted by default: every arm keeps the config's own "
                         "resolution and NO token is added, so names stay identical to existing "
                         "artifacts and --collect-only still finds them. units/block arms are "
                         "never pinned (their port is resolved per bucket as P//2-1 and the "
                         "fault model rejects an absolute index) -- they run unpinned and "
                         "untokened even when this flag is set. N must be valid at the SHORTEST "
                         "dense wire (rt_size=2 => N<=1), since the model would otherwise clamp "
                         "silently and the token would lie; in practice use 0.")
    args = p.parse_args(argv)

    ap_err = validate_ap_position(args.ap_position)
    if ap_err:
        print(f"ERROR: {ap_err}", file=sys.stderr)
        return 2

    if args.shards < 1:
        print("ERROR: --shards must be >= 1", file=sys.stderr)
        return 2
    if (args.print_commands or args.collect_only) and args.stage != "main":
        print(f"ERROR: --print-commands/--collect-only only apply to --stage main "
              f"(got --stage {args.stage}); preflight/calibrate are single-process, "
              f"single-GPU probes with no sharding or harvesting story.", file=sys.stderr)
        return 2
    if args.calibration and not (args.print_commands or args.shard_index is not None):
        print("WARNING: --calibration has no effect without --print-commands or "
              "--shard-index; ignoring.", file=sys.stderr)
    if args.shard_index is not None:
        if args.stage != "main":
            print(f"ERROR: --shard-index only applies to --stage main (got --stage "
                  f"{args.stage}).", file=sys.stderr)
            return 2
        if args.print_commands or args.collect_only:
            print("ERROR: --shard-index cannot be combined with --print-commands/"
                  "--collect-only -- --print-commands already emits every shard, and "
                  "--collect-only harvests the full 79-cell matrix regardless of sharding.",
                  file=sys.stderr)
            return 2
        if args.shards <= 1:
            print("ERROR: --shard-index requires --shards > 1.", file=sys.stderr)
            return 2
        if not (0 <= args.shard_index < args.shards):
            print(f"ERROR: --shard-index must satisfy 0 <= shard_index < shards "
                  f"(got shard_index={args.shard_index}, shards={args.shards}).",
                  file=sys.stderr)
            return 2
    if args.skip_preflight and args.stage != "main":
        print(f"ERROR: --skip-preflight only applies to --stage main (got --stage "
              f"{args.stage}).", file=sys.stderr)
        return 2

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    curve = [float(x) for x in args.rt_curve]
    base_stem = cfg_path.stem
    wandb_project = args.wandb_project or None
    mitigations = [(k, v) for k, v in MITIGATIONS if k in args.mitigations]

    _print_header(args, cfg_path, curve, args.loops, args.seeds, mitigations)

    if args.stage == "preflight":
        return run_preflight(cfg_path, base_stem, wandb_project, args.wandb_entity, args.dry_run,
                             ap_position=args.ap_position)

    if args.stage == "calibrate":
        return run_calibrate(cfg_path, base_stem, wandb_project, args.wandb_entity,
                              args.dry_run, args.seeds, mitigations,
                              ap_position=args.ap_position)

    # stage == "main"
    if args.collect_only:
        return _main_collect_only(cfg_path, base_stem, args.seeds, mitigations, curve, args.loops,
                                  ap_position=args.ap_position)

    if args.print_commands:
        try:
            _emit_shard_commands(cfg_path, base_stem, wandb_project, args.wandb_entity, curve,
                                  args.loops, args.seeds, mitigations, args.shards, args.calibration,
                                  ap_position=args.ap_position)
        except ValueError as exc:
            # Raised by _arm_weight on a stale (pre-two-point-fit) scalar
            # calibration.json -- surface as a clean error, not a traceback.
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.dry_run:
        return _main_dry_run(cfg_path, base_stem, wandb_project, args.wandb_entity,
                              args.seeds, mitigations, curve, args.loops,
                              shards=args.shards, shard_index=args.shard_index,
                              calibration_path=args.calibration,
                              ap_position=args.ap_position)

    # Live execution: gate FIRST (spec section 4), in-process -- see module
    # docstring -- UNLESS --skip-preflight explicitly asserts it already
    # passed (loud, not silent: see the flag's help text). This check is
    # shared by the plain and --shard-index paths alike -- --shard-index does
    # NOT imply --skip-preflight; the user must ask for both separately.
    # Only this branch (and _main_live below) imports the runner (torch/numba).
    if args.skip_preflight:
        print("\nWARNING: --skip-preflight set -- SKIPPING the correctness gate. The caller "
              "asserts preflight has ALREADY passed (e.g. a prior interactive "
              "'--stage preflight' run). If it has not, every number this invocation produces "
              "may be garbage.", file=sys.stderr)
    else:
        gate_rc = run_preflight(cfg_path, base_stem, wandb_project, args.wandb_entity, dry_run=False,
                                ap_position=args.ap_position)
        if gate_rc != 0:
            print("\nABORT: preflight gate failed; the main matrix would produce garbage "
                  "(see PREFLIGHT FAIL rows above). Not running the 79-cell matrix.",
                  file=sys.stderr)
            return gate_rc
    return _main_live(cfg_path, base_stem, wandb_project, args.wandb_entity,
                       args.seeds, mitigations, curve, args.loops,
                       shards=args.shards, shard_index=args.shard_index,
                       calibration_path=args.calibration,
                       ap_position=args.ap_position)


if __name__ == "__main__":
    sys.exit(main())
