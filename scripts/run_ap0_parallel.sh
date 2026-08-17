#!/usr/bin/env bash
# Three experiment groups in PARALLEL, each sequential internally, all at ap_position=0.
#
#   exp1  lay-col             prot-2to7 then prot-1to8
#   exp2  lay-block-base-col  prot-1to8 then prot-2to7
#   exp3  lay-block-base-row  prot-1to8 then prot-2to7
#
# Each cell sweeps rt_error {1e-5, 4.55e-5, 1e-6} inside one runner invocation,
# and the seed loop is here (one invocation per seed):
#   3 groups x 2 protection arms x 3 seeds = 18 invocations
#   18 x 3 rt_error = 54 W&B runs in project netdrift-design_space
#
# All three groups share ONE GPU (user-confirmed they fit). They run as separate
# PROCESSES, which is the only safe way to overlap them -- the runner holds a
# single CUDA context per process and Numba's context is not thread-safe. Note
# they still contend for the device, so wall-clock per run will be longer than a
# solo run; total throughput is what improves.
#
# ⚠️ SEEDS: sweep_col_vs_block.py takes a SINGLE --seed, so each seed is its own
# invocation with --name-suffix seed<N>. Without that suffix all three seeds would
# share one experiment.name and overwrite each other's summary.json.
#
# ⚠️ exp2/exp3 (BLOCK) are expected FLAT at baseline (88.19%) under
# edge_mode=saturate: same-sign blocks + reads clamping inside the block make them
# fault-immune, and ap_position only moves the offset window -- verified from the
# 2026-07-20 artifacts at every rt_error. Included because you asked for them;
# expect confirmation, not variation. Use EDGE_MODE=random for a live BLOCK curve.
set -euo pipefail
# Each group is a pipeline (group | tee | sed), so `wait` sees the LAST command's
# status. pipefail makes the pipeline inherit the group's failure instead of sed's
# success, so a crashed runner is still reported below.
cd "$(dirname "$0")/.."

# Unbuffered Python: the runner's stdout is a pipe here (tee), and CPython
# block-buffers non-tty stdout, so without this you would see nothing for minutes
# and then a flood. Applies to every child process.
export PYTHONUNBUFFERED=1

RT_CURVE="${RT_CURVE:-1e-5 4.55e-5 1e-6}"
SEEDS="${SEEDS:-707 708 709}"
LOOPS="${LOOPS:-100}"
EDGE_MODE="${EDGE_MODE:-saturate}"
WANDB_PROJECT="${WANDB_PROJECT:-netdrift-design_space}"
CONFIG="${CONFIG:-configs/vgg7_cifar10/vgg7_cifar10_w1a1_rtm.yaml}"
LOG_DIR="${LOG_DIR:-runs/logs/ap0_parallel_$(date +%Y%m%d-%H%M%S)}"
PYTHON="${PYTHON:-python}"

mkdir -p "$LOG_DIR"

# Line-buffer the label filter if stdbuf exists (GNU coreutils); otherwise fall
# back to plain sed, which only costs some output latency.
if command -v stdbuf >/dev/null 2>&1; then
    label() { stdbuf -oL sed "s/^/$1/"; }
else
    label() { sed "s/^/$1/"; }
fi

# One group: its layout's two protection arms, sequentially, each over all seeds.
run_group() {
    local tag="$1" layout="$2"; shift 2
    local prots=("$@")
    for prot in "${prots[@]}"; do
        for seed in $SEEDS; do
            echo "=== [$tag] layout=$layout prot=$prot seed=$seed ==="
            "$PYTHON" scripts/sweep_col_vs_block.py \
                --config "$CONFIG" \
                --layouts "$layout" \
                --protections "$prot" \
                --rt-curve $RT_CURVE \
                --ap-position 0 \
                --seed "$seed" \
                --name-suffix "seed${seed}" \
                --loops "$LOOPS" \
                --edge-mode "$EDGE_MODE" \
                --wandb-project "$WANDB_PROJECT"
        done
    done
    echo "[$tag] GROUP DONE"
}

echo "rt_curve = $RT_CURVE"
echo "seeds    = $SEEDS"
echo "loops    = $LOOPS   edge_mode = $EDGE_MODE   ap_position = 0"
echo "project  = $WANDB_PROJECT"
echo "logs     = $LOG_DIR"
echo "18 runner invocations -> 54 W&B runs, 3 groups sharing one GPU"
echo

# Each group streams to the terminal AND to its log file. The sed prefix labels
# every line with its group, since three streams interleave on one console; the
# raw (unprefixed) text still goes to the log. stdbuf keeps the pipe line-buffered
# so progress appears live instead of in delayed 4-KiB blocks.
run_group exp1 col       prot-2to7 prot-all 2>&1 \
    | tee "$LOG_DIR/exp1_col.log"       | label '[exp1 col      ] ' &
P1=$!
run_group exp2 block_col prot-all  prot-2to7 2>&1 \
    | tee "$LOG_DIR/exp2_block_col.log" | label '[exp2 block-col] ' &
P2=$!
run_group exp3 block_row prot-all  prot-2to7 2>&1 \
    | tee "$LOG_DIR/exp3_block_row.log" | label '[exp3 block-row] ' &
P3=$!

echo "launched  exp1=$P1  exp2=$P2  exp3=$P3"
echo "output is live below (also saved to $LOG_DIR/); per-group log:"
echo "  tail -f $LOG_DIR/exp1_col.log"
echo

rc=0
wait $P1 || { echo "!! exp1 FAILED (see $LOG_DIR/exp1_col.log)"; rc=1; }
wait $P2 || { echo "!! exp2 FAILED (see $LOG_DIR/exp2_block_col.log)"; rc=1; }
wait $P3 || { echo "!! exp3 FAILED (see $LOG_DIR/exp3_block_row.log)"; rc=1; }

echo
echo "all groups finished (rc=$rc). Logs: $LOG_DIR"
echo
echo "Collect the per-cell tables (one call per group/seed combination):"
echo "  for s in $SEEDS; do"
echo "    for L in col block_col block_row; do"
echo "      $PYTHON scripts/sweep_col_vs_block.py --layouts \$L --ap-position 0 \\"
echo "          --name-suffix seed\$s --rt-curve $RT_CURVE --collect-only"
echo "    done"
echo "  done"
exit $rc
