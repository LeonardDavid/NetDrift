#!/usr/bin/env bash
# Re-run the six col-vs-block cells at rt_error=1e-06 with the access port at the
# LOW EDGE of the wire (fault.ap_position=0).
#
# Produces these six W&B runs (name = <experiment.name>-rt<rt_error>):
#   vgg7_cifar10_w1a1_rtm__lay-col_prot-1to8_ap0-rt1e-06
#   vgg7_cifar10_w1a1_rtm__lay-col_prot-2to7_ap0-rt1e-06
#   vgg7_cifar10_w1a1_rtm__lay-block-base-row_prot-1to8_ap0-rt1e-06
#   vgg7_cifar10_w1a1_rtm__lay-block-base-row_prot-2to7_ap0-rt1e-06
#   vgg7_cifar10_w1a1_rtm__lay-block-base-col_prot-1to8_ap0-rt1e-06
#   vgg7_cifar10_w1a1_rtm__lay-block-base-col_prot-2to7_ap0-rt1e-06
#
# ⚠️ ONLY THE TWO lay-col CELLS CAN PRODUCE NEW INFORMATION. Verified from the
# 2026-07-20 artifacts under runs/*/col_vs_block/: all four lay-block-base-* cells
# sit at exactly 88.19% (= baseline_clean_accuracy) at every one of the 5 rt_error
# values. That is edge_mode=saturate BLOCK fault-immunity (same-sign blocks + reads
# clamping inside the block). ap_position only sets the offset WINDOW; the read
# still clamps into the same-sign block, so ap=0 changes nothing there. Those four
# are re-run purely as confirmation. Drop them with --layouts col if you'd rather
# not spend the GPU time (or just read the two col rows from the output CSV).
#
# HOW TO SEPARATE THESE FROM THE OLD RUNS IN W&B:
#   Use the `subcategory` / `experiment.name` token `_ap0`. Do NOT group by the flat
#   `ap_position` config key to compare against the pre-2026-08-03 runs: that key
#   did not exist when they were logged, so you would get 0 vs. undefined rather
#   than 0 vs. 31. (The old runs' nested config.fault.ap_position is `null` — the
#   effective mid-wire value was resolved at kernel-launch time and never
#   persisted.) The flat `ap_position` key IS correct for grouping runs logged from
#   now on. To build a real ap_position axis spanning both eras you'd have to
#   backfill `ap_position: 31` onto the old col_vs_block runs' W&B config.
#
# SETTINGS BELOW ARE PINNED TO THE ORIGINAL RUNS (verified, not assumed):
#   loops=30      — 30 entries in each rt_error_sweep[].accuracies array
#   edge_mode=saturate — proven by the flat BLOCK curves above
#   seed=707, config vgg7_cifar10_w1a1_rtm.yaml — driver defaults, unchanged
#
# WHY A SINGLE-POINT CURVE IS COMPARABLE TO THE 1e-6 SLICE OF THE ORIGINAL
# 5-POINT SWEEP: the runner calls _reset_fault_state() and _seed_everything(seed)
# at the top of every rt_error iteration (runner/run.py), so each point's
# realization is independent of which values preceded it.
set -euo pipefail

cd "$(dirname "$0")/.."

LOOPS="${LOOPS:-30}"
SEED="${SEED:-707}"
EDGE_MODE="${EDGE_MODE:-saturate}"
WANDB_PROJECT="${WANDB_PROJECT:-netdrift-col-vs-block}"

COMMON=(
    --config configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml
    --rt-curve 1e-6
    --ap-position 0
    --loops "$LOOPS"
    --seed "$SEED"
    --edge-mode "$EDGE_MODE"
    --wandb-project "$WANDB_PROJECT"
)

# Sanity-check the matrix and the exact argv before burning GPU time.
python scripts/sweep_col_vs_block.py "${COMMON[@]}" --dry-run

# Execute all six cells sequentially (one runner invocation per cell).
python scripts/sweep_col_vs_block.py "${COMMON[@]}"
