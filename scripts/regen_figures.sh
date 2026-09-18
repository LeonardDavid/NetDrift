#!/usr/bin/env bash
# Regenerate every TikZ figure directory under docs/figures/ from the run trees.
#
#   bash scripts/regen_figures.sh
#
# Order matters: the compact bodies are restyled FROM the roomy ones, and the
# combined grids are built FROM the compact ones, so stage 1 must run first.
# Everything is stdlib-only Python -- no GPU, no torch, runs on the Mac mount.
set -euo pipefail
cd "$(dirname "$0")/.."

GEN=scripts/plot_paper_sw_tikz.py
CMP=scripts/plot_compact_tikz.py
FIG=docs/figures

# ---------------------------------------------------------------- knobs -----
# The paper-sw figure you tuned by hand: four arms, 50 iterations, 4-up legend.
# Drop --only to get all six categories back; drop --prune to keep the .dat
# files of the categories you excluded (they are unused, just untidy).
PAPERSW_ARGS=(--max-iter 50 --legend-columns 4 --prune
              --only cat1_baseline cat5_regularizer cat6_reg_recal cat8_ste_inject)

# Training variant plotted in the layout figures, and the BLOCK reference level.
LAYOUT_ARGS=(--rt-error 4.55e-05 --variant cat8 --block auto)

# The tradeoff figures are OFF by default: they come from plot_tradeoff_tikz.py and
# their bodies carry a large block of hand-tunable knobs (\ShowMirror, \DropSign,
# \RightMax, per-layout labels...). Regenerating resets all of that. Opt in with
#     REGEN_TRADEOFF=1 bash scripts/regen_figures.sh
REGEN_TRADEOFF="${REGEN_TRADEOFF:-0}"

# Panels combined in the two-up grids (the four-up ones use all four trees).
PAIR_1x2=(vgg-fl r18-fl)     # side by side
PAIR_2x1=(vgg-nop r18-nop)   # stacked
# -----------------------------------------------------------------------------

TREES=(r18-fl r18-nop vgg-fl vgg-nop)
lay_dir() { echo "$FIG/$1_rt4.55e-05_var-cat8"; }

echo "== 1/4  source figures: layout trees =========================="
for t in "${TREES[@]}"; do
  python3 "$GEN" --run-dir "runs/paper-runs/plots/$t" "${LAYOUT_ARGS[@]}"
done

echo "== 2/4  source figures: paper-sw =============================="
python3 "$GEN" --rt-error 1e-05 "${PAPERSW_ARGS[@]}"
python3 "$GEN" --rt-error 4.55e-05

echo "== 3/4  compact restyles ======================================"
COMPACT_DIRS=()
for t in "${TREES[@]}"; do COMPACT_DIRS+=("$(lay_dir "$t")"); done
COMPACT_DIRS+=("$FIG/paper-sw_resnet18_imagenette_rt1e-05"
               "$FIG/paper-sw_resnet18_imagenette_rt4.55e-05")
python3 "$CMP" --squish "${COMPACT_DIRS[@]}"

echo "== 4/4  combined grids ========================================"
ALL4=()
for t in "${TREES[@]}"; do ALL4+=("$(lay_dir "$t")"); done
for g in 2x2 1x4 4x1; do
  python3 "$CMP" --squish --grid "$g" "${ALL4[@]}"
done
python3 "$CMP" --squish --grid 1x2 "$(lay_dir "${PAIR_1x2[0]}")" "$(lay_dir "${PAIR_1x2[1]}")"
python3 "$CMP" --squish --grid 2x1 "$(lay_dir "${PAIR_2x1[0]}")" "$(lay_dir "${PAIR_2x1[1]}")"

if [ "$REGEN_TRADEOFF" = "1" ]; then
  echo "== 5/5  tradeoff figures (REGEN_TRADEOFF=1) ==================="
  python3 scripts/plot_tradeoff_tikz.py --style pareto
  python3 scripts/plot_tradeoff_tikz.py --style combo
else
  echo "-- skipping tradeoff-r18{,-combo} (REGEN_TRADEOFF=1 to include; resets their knobs)"
fi

echo
echo "done -- $(find "$FIG" -name '*_body.tex' | wc -l | tr -d ' ') figure bodies under $FIG/"
