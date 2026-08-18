#!/usr/bin/env bash
# PPM (polarity-partitioned mapping) — experiment driver.
#
# PPM is NOT yet an arm in sweep_design_space.py (no ARMS row: its wires/cells
# must be MEASURED on the real model first, which is what STAGE 1 does). So
# these are direct runner invocations, not a sweep-driver call.
#
# ── Cell-count logic (read before adding cells) ────────────────────────────
# PADDED PPM IS FAULT-IMMUNE, so protection policy, seed and mitigation cannot
# change its result — exactly like BLOCK under saturate. It therefore collapses
# to ONE cell per window. Running the full protection x seed matrix on it would
# repeat the ~32 GPU-hours already burned on immune BLOCK cells in the Aug 4-5
# col_vs_block sweeps. The UNPADDED ablation is the arm that actually varies,
# so that is where the matrix goes.
#
# ── base_layout is pinned to col, deliberately ─────────────────────────────
# PPM's ap_reads is the bucket length (rt_size); dense ROW is rt_size^2. At
# base_layout=row PPM would get ~64x fewer read events — an advantage with
# nothing to do with sign purity. Any PPM-vs-dense claim must use col.
#
# Usage:  bash scripts/run_ppm_experiments.sh [stage]
#         stage = gate | immunity | ablation | baseline | all   (default: gate)
set -uo pipefail
cd "$(dirname "$0")/.."

CFG=configs/modes/vgg7_cifar10_w1a1_polarity.yaml
DENSE_CFG=configs/modes/vgg7_cifar10_w1a1_baseline_col.yaml
# Stages 2-4 curve. A 3-point SUBSET of sweep_design_space.py's 5-point
# DEFAULT_RT_CURVE ([1e-4, 4.55e-5, 1e-5, 1e-6, 1e-7]), dropping the two ends,
# and matching the curve the Aug 4-5 col_vs_block runs used. Still joinable with
# the existing design-space data on these three shared points: run.py calls
# _reset_fault_state() + _seed_everything(seed) at the top of every rt_error
# iteration, so each point is independent of its predecessors and a 3-point run
# is directly comparable to the same 3 slices of a 5-point sweep.
# NB stage 1's gate curves are separate literals ([0.0] and [1e-05]) and are
# deliberately NOT driven by this variable.
CURVE='[4.55e-05,1e-05,1e-06]'
LOOPS=30
SEEDS=(707 808 909)
WINDOWS=(0 2 8 32)          # 0 = channel-aligned

# Protection policies, matching sweep_design_space.py::PROTECTIONS exactly so
# PPM joins that data on the same axis:
#   prot-2to7 = policy custom, layers 2..7 UNPROTECTED => first+last protected
#   prot-1to8 = policy all,    every layer UNPROTECTED => nothing protected
# (the tag names the UNPROTECTED range, not the protected one).
PROTS=(prot-2to7 prot-1to8)
prot_args() {  # prot_args <tag> -> override flags on stdout (word-split on use)
  case "$1" in
    prot-2to7) echo "--override fault.protection.policy=custom --override fault.protection.layers=[2,3,4,5,6,7]" ;;
    prot-1to8) echo "--override fault.protection.policy=all" ;;
    *) echo "unknown protection tag: $1" >&2; return 1 ;;
  esac
}
# W&B project. Defaults to the project the Aug 4-5 col_vs_block runs actually
# used (per their manifests) so PPM joins that data rather than landing in a
# separate project. NB sweep_design_space.py's DEFAULT_WANDB_PROJECT is
# "netdrift-col-vs-block" — a different project; do not assume they match.
PROJ="${WANDB_PROJECT:-netdrift-design_space}"
LOGDIR=runs/logs/ppm; mkdir -p "$LOGDIR"

run() {  # run <logname> <args...>
  local name="$1"; shift
  echo "=== $name"
  python netdrift_run.py "$@" 2>&1 | tee "$LOGDIR/$name.log" | tail -4
}

# ───────────────────────────────────────────────────────────────────────────
# STAGE 1 — GATE + COST.  Seconds per cell. Nothing else runs until this passes.
#
# rt_error=0 with loops=1 must reproduce clean accuracy EXACTLY. This is a
# sharper gate for PPM than for other layouts: PPM changes no weight VALUE at
# all, so any mismatch means the permutation lost or duplicated a weight, not
# that the fault model misbehaved.
#
# It also MEASURES n_racetracks per window — the cost-axis x-value, and the
# number an ARMS row needs. Read it from the summary.json / W&B for each run.
# ───────────────────────────────────────────────────────────────────────────
stage_gate() {
  for w in "${WINDOWS[@]}"; do
    # Logged under a SEPARATE category (ppm_gate, not ppm) so these rt_error=0
    # correctness runs stay filterable out of the curve plots, while still
    # putting n_racetracks — the cost axis — into W&B as a flat config key.
    run "gate_w${w}" --config "$CFG" --metrics offline \
      --wandb-project "$PROJ" --wandb-category ppm_gate \
      --wandb-subcategory "ppm_gate_w${w}" \
      --override "storage.partition.window=$w" \
      --override "storage.partition.pad=true" \
      --override "fault.rt_error=[0.0]" \
      --override "training.loops=1" \
      --override "experiment.name=vgg7_ppm_gate_w${w}"

    # Second half of the gate, mirroring sweep_design_space.py's preflight
    # (PREFLIGHT_GATE_CURVE=[0.0] AND PREFLIGHT_EQUIV_CURVE=[1e-5]): rt_error=0
    # only proves the permutation is LOSSLESS. It says nothing about immunity,
    # which is the actual claim. This spot-check costs one loop and fails fast
    # before stage 2 commits to the full 5-point curve.
    run "gatefault_w${w}" --config "$CFG" --metrics offline \
      --wandb-project "$PROJ" --wandb-category ppm_gate \
      --wandb-subcategory "ppm_gatefault_w${w}" \
      --override "storage.partition.window=$w" \
      --override "storage.partition.pad=true" \
      --override "fault.rt_error=[1e-05]" \
      --override "training.loops=1" \
      --override "experiment.name=vgg7_ppm_gatefault_w${w}"
  done
  echo
  echo "CHECK: every run's faulted accuracy == baseline_clean_accuracy (88.19) EXACTLY."
  echo "       gate_*      = rt_error 0    -> permutation is lossless"
  echo "       gatefault_* = rt_error 1e-5 -> padded PPM is immune"
  echo
  echo "COST AXIS — n_racetracks per window (send these to add the ARMS rows):"
  python3 - <<'PYEOF'
import json, pathlib
# n_racetracks is the TOP-LEVEL scalar cost-axis total that run.py writes
# alongside baseline_clean_accuracy — the same key sweep_design_space.py's
# _read_n_racetracks reads. Deliberately NOT the differently-shaped per-layer
# "n_racetracks" array nested under layer_metrics_by_rt_error, which is a
# separate static-metrics computation.
for tag in ("gate", "gatefault"):
  for w in (0, 2, 8, 32):
    hits = list(pathlib.Path("runs").glob(f"vgg7_ppm_{tag}_w{w}/**/summary.json"))
    if not hits:
        print(f"  {tag:<9} window={w:<3} (no summary.json found)"); continue
    j = json.load(open(max(hits, key=lambda p: p.stat().st_mtime)))
    n_rt = j.get("n_racetracks")
    clean = j.get("baseline_clean_accuracy")
    sweep = j.get("rt_error_sweep", [])
    accs = sweep[0].get("accuracies") if sweep else None
    acc = (sum(accs) / len(accs)) if accs else None
    ok = ("EXACT" if (acc is not None and clean is not None and abs(acc - clean) < 1e-9)
          else "*** MISMATCH — STOP ***")
    n_txt = f"{n_rt:,}" if isinstance(n_rt, int) else f"{n_rt!r} (key absent => cost axis unavailable)"
    print(f"  {tag:<9} window={w:<3} n_racetracks={n_txt:>22}  clean={clean}  faulted={acc}  [{ok}]")
PYEOF
}

# ───────────────────────────────────────────────────────────────────────────
# STAGE 2 — IMMUNITY.  One cell per window (immune => seed/protection inert).
# Expect a FLAT 88.19 across the whole rt_error curve. A non-flat row means
# padding is not delivering sign purity on the real model.
# ───────────────────────────────────────────────────────────────────────────
stage_immunity() {
  # BOTH protection policies. Immunity makes protection provably inert, so this
  # is not a matrix -- it is a one-off invariance CHECK on the real model
  # (the GPU immunity tests use synthetic weights). Seed stays fixed at 707 for
  # the same reason: an immune arm cannot vary with it.
  for w in "${WINDOWS[@]}"; do
    for p in "${PROTS[@]}"; do
      run "immune_w${w}_${p}" --config "$CFG" --metrics offline \
        --wandb-project "$PROJ" --wandb-category ppm \
        --wandb-subcategory "ppm_pad_w${w}_${p}_seed707" \
        --override "storage.partition.window=$w" \
        --override "storage.partition.pad=true" \
        --override "fault.rt_error=$CURVE" \
        --override "training.loops=$LOOPS" \
        $(prot_args "$p") \
        --override "experiment.seed=707" \
        --override "experiment.name=vgg7_ppm_pad_w${w}_${p}"
    done
  done
  echo
  echo "CHECK: all ${#WINDOWS[@]} x ${#PROTS[@]} rows FLAT at 88.19 and IDENTICAL"
  echo "       across prot-2to7 / prot-1to8 -- protection is inert under immunity."
}

# ───────────────────────────────────────────────────────────────────────────
# STAGE 3 — PADDING ABLATION.  The scientifically interesting arm.
#
# pad=false leaves 1/K of wires mixed-sign and fully exposed — structurally the
# units-t2-g0 failure mode (74.5% immune bits, still scored 28.27%). Prediction:
# accuracy DEGRADES as the window widens (wider window => the one mixed wire
# covers more weights)... or stays poor throughout, if the worst-wire effect
# dominates as it did for units. Either way this is what makes the padding
# claim falsifiable rather than decorative.
#
# This arm is NOT immune, so it gets the full seed treatment.
# ───────────────────────────────────────────────────────────────────────────
stage_ablation() {
  for w in "${WINDOWS[@]}"; do
    for p in "${PROTS[@]}"; do
      for sd in "${SEEDS[@]}"; do
        run "nopad_w${w}_${p}_s${sd}" --config "$CFG" --metrics offline \
          --wandb-project "$PROJ" --wandb-category ppm \
          --wandb-subcategory "ppm_nopad_w${w}_${p}_seed${sd}" \
          --override "storage.partition.window=$w" \
          --override "storage.partition.pad=false" \
          --override "fault.rt_error=$CURVE" \
          --override "training.loops=$LOOPS" \
          $(prot_args "$p") \
          --override "experiment.seed=$sd" \
          --override "experiment.name=vgg7_ppm_nopad_w${w}_${p}_seed${sd}"
      done
    done
  done
}

# ───────────────────────────────────────────────────────────────────────────
# STAGE 4 — DENSE BASELINE at matched settings (the y-axis reference).
#
# dense-rt64 col already exists from the design-space sweep at seed 707, but
# re-running it here buys one self-contained comparison from a single code
# revision — and it is cheap. BLOCK is NOT re-run: it is immune and already
# measured flat at 88.19 (see the ap0 memory: ~32 GPU-h spent proving that).
# ───────────────────────────────────────────────────────────────────────────
stage_baseline() {
  for p in "${PROTS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      run "dense_rt64_${p}_s${sd}" --config "$DENSE_CFG" --metrics offline \
        --wandb-project "$PROJ" --wandb-category ppm \
        --wandb-subcategory "dense-rt64_${p}_seed${sd}" \
        --override "storage.rt_size=64" \
        --override "fault.edge_mode=saturate" \
        --override "fault.weight_encoder=null" \
        --override "fault.rt_error=$CURVE" \
        --override "training.loops=$LOOPS" \
        $(prot_args "$p") \
        --override "experiment.seed=$sd" \
        --override "experiment.name=vgg7_dense_rt64_ppmref_${p}_seed${sd}"
    done
  done
}

case "${1:-gate}" in
  gate)      stage_gate ;;
  immunity)  stage_immunity ;;
  ablation)  stage_ablation ;;
  baseline)  stage_baseline ;;
  all)       stage_gate && stage_immunity && stage_ablation && stage_baseline ;;
  *) echo "usage: $0 [gate|immunity|ablation|baseline|all]"; exit 2 ;;
esac
