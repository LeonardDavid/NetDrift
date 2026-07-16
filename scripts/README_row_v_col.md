# Row-vs-Col metrics analysis

Aggregate + plot the `runs_row-v-col/` metrics tree (6 categories × 2 layouts ×
N seeds × 2 rt_errors). See the design spec:
`docs/superpowers/specs/2026-06-19-row-v-col-metrics-plots-design.md`.

## Scripts (`scripts/`)

| file | role |
|------|------|
| `rvc_common.py` | discovery, defensive parsers, tidy long-frame builder, shared style maps, seed-aware reduction. Import-only. |
| `aggregate_row_v_col.py` | CLI → `tidy_long.csv` (canonical), `summary_wide.csv`, `summary.md`. |
| `plot_row_v_col.py` | CLI → PNG/PDF figure catalog. |

## Quick start (on the GPU host)

```bash
conda activate netdrift           # provides numpy / pandas / matplotlib
python scripts/aggregate_row_v_col.py            # writes runs_row-v-col/aggregated/
python scripts/plot_row_v_col.py --format png pdf  # writes runs_row-v-col/figures/
```

Useful flags:
```bash
python scripts/aggregate_row_v_col.py --dry-run            # list discovered cells
python scripts/plot_row_v_col.py --figures acc_bars ber_bars
python scripts/plot_row_v_col.py --tidy runs_row-v-col/aggregated/tidy_long.csv
python scripts/aggregate_row_v_col.py --include cat5 cat6  # subset by path token
```

## Figure catalog

**Row-vs-col (layout = headline contrast)**
- `traj_rt<err>` — accuracy-vs-loop trajectories, col vs row, per category.
- `fault_loop_<metric>` — affected_units / bitflips vs loop, row vs col.
- `layer_heatmap_<cell>_rt<err>` — per-layer bitflips (layer × loop); titles
  carry each panel's peak + the row/col ratio (panels use per-panel colour
  scales so internal structure stays readable).

**Category-ranking (technique = headline)**
- `acc_bars` — grouped final accuracy (clean / each rt_error), row vs col.
- `ber_bars` — final-loop BER ranking across categories, row vs col.
- `mech_runlength`, `mech_altseq` — weight-structure histograms, row vs col.
- `pipeline_deltas` — cat2/cat4/cat6 encoder/recal stage deltas.

## Reading notes (load-bearing)

- **n=1 today** (seed 707). The aggregator is seed-aware: with >1 seed it draws
  mean±std bands; at n=1 the bands collapse to 0. The seed count is printed on
  every run — never assume the set is complete.
- `affected_units` is a pure function of fault geometry (rt_error × layout): it
  is **constant across techniques**. `bitflips` is **not** — the endlen encoder
  (cat2/cat4) roughly halves bitflips. Same fault exposure, different damage,
  because of weight structure. BER = bitflips / unprotected-weight-count, so BER
  tracks bitflips, not affected_units.
- **Recal stages are structurally inert**: `after_encoder→after_recal` and
  `trained→after_recal` deltas are ~0 — recal tunes BN/affine, not quantized
  weights (the "dead-gradient" effect). The pipeline figure shows this.
- cat2 (endlen) has the *fewest* bitflips yet still *collapses* in accuracy
  (~9%). Fewer flipped bits ≠ better accuracy here — interpret with care.
- `lambda` is parsed from the directory tag (`lam0p01`→0.01), NOT config
  (test-phase `reg.lambda_` is 0.0). cat5/cat6 each have two λ variants.
