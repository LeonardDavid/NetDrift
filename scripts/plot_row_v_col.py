#!/usr/bin/env python
"""Generate the row-vs-col / category-ranking figure catalog.

Reads the metrics tree directly (or a pre-built ``tidy_long.csv``) and writes a
catalog of PNG/PDF figures into ``runs_row-v-col/figures/``.

Two figure families:
  ROW-vs-COL (layout = headline contrast, paired)
    traj          accuracy-vs-loop trajectories, col vs row, per category
    fault_loop    BER & affected_units vs loop, col vs row
    layer_heatmap per-layer bitflips heatmap (layer x loop), per cell
  CATEGORY-RANKING (technique = headline)
    acc_bars      grouped final-accuracy bars (clean / each rt_error), row vs col
    ber_bars      final-BER ranking bars across categories, row vs col
    mech_hist     run-length & alternating-seq histograms, row vs col
    pipeline      cat2/cat4/cat6 before->after deltas

Usage::

    python scripts/plot_row_v_col.py                      # all figures, png
    python scripts/plot_row_v_col.py --figures traj acc_bars --format png pdf
    python scripts/plot_row_v_col.py --tidy runs_row-v-col/aggregated/tidy_long.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless host
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rvc_common as rvc  # noqa: E402

ALL_FIGURES = [
    "traj",
    "fault_loop",
    "layer_heatmap",
    "acc_bars",
    "ber_bars",
    "mech_hist",
    "pipeline",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cells(df: pd.DataFrame, *, lam_only_headline: bool = True) -> list[tuple]:
    """Distinct (category, lambda) groups in canonical order.

    For cat5/cat6 (which have two lambdas) keep both; ``lam_only_headline``
    restricts to the higher lambda where a headline-only view is wanted.
    """
    sub = df[["category", "lambda"]].drop_duplicates()
    out = []
    for cat in rvc.CATEGORY_ORDER:
        lams = sorted(sub[sub["category"] == cat]["lambda"].dropna().unique())
        has_none = sub[(sub["category"] == cat) & (sub["lambda"].isna())].shape[0] > 0
        if has_none:
            out.append((cat, None))
        if lams:
            chosen = [max(lams)] if lam_only_headline else lams
            out.extend((cat, l) for l in chosen)
    return out


def _label(cat: str, lam) -> str:
    base = rvc.CATEGORY_LABELS.get(cat, cat)
    if lam is not None and not pd.isna(lam):
        base += f" (λ={lam:g})"
    return base


def _select(df: pd.DataFrame, cat: str, lam, layout: str = None) -> pd.DataFrame:
    mask = df["category"] == cat
    mask &= df["lambda"].isna() if (lam is None or pd.isna(lam)) else (df["lambda"] == lam)
    if layout is not None:
        mask &= df["layout"] == layout
    return df[mask]


def _rt_values(df: pd.DataFrame) -> list[float]:
    return sorted(df["rt_error"].dropna().unique())


def _save(fig, out_dir: Path, name: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.{{{','.join(formats)}}}")


def _line_with_band(ax, x, mean, std, *, color, ls, marker, label, n):
    ax.plot(x, mean, color=color, linestyle=ls, marker=marker, markersize=3,
            markevery=max(1, len(x) // 12), label=label, linewidth=1.6)
    if n > 1:
        ax.fill_between(x, np.array(mean) - np.array(std), np.array(mean) + np.array(std),
                        color=color, alpha=0.15, linewidth=0)


# ---------------------------------------------------------------------------
# ROW-vs-COL figures
# ---------------------------------------------------------------------------
def fig_traj(df, red, out_dir, formats):
    """Accuracy-vs-loop trajectories, col vs row overlaid, one panel per cell,
    a separate figure per rt_error."""
    cells = _cells(df, lam_only_headline=False)
    for rt in _rt_values(df):
        n = len(cells)
        ncol = 3
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
        for ax, (cat, lam) in zip(axes.flat, cells):
            color = rvc.CATEGORY_COLORS[cat]
            clean = None
            for layout in ("row", "col"):
                d = red[(red["category"] == cat) & (red["rt_error"] == rt) & (red["layout"] == layout)
                        & (red["family"] == "outcome") & (red["metric"] == "accuracy") & red["loop"].notna()]
                if lam is None or pd.isna(lam):
                    d = d[d["lambda"].isna()]
                else:
                    d = d[d["lambda"] == lam]
                d = d.sort_values("loop")
                if d.empty:
                    continue
                _line_with_band(ax, d["loop"].to_numpy(), d["mean"].to_numpy(), d["std"].to_numpy(),
                                color=color, ls=rvc.LAYOUT_LINESTYLE[layout], marker=rvc.LAYOUT_MARKER[layout],
                                label=f"{layout}", n=int(d["n"].max()))
                cb = _select(df, cat, lam, layout)
                cbv = cb[(cb["family"] == "outcome") & (cb["metric"] == "clean_baseline")]["value"]
                if not cbv.empty:
                    clean = cbv.iloc[0]
            if clean is not None:
                ax.axhline(clean, color="gray", ls=":", lw=1, label="clean")
            ax.set_title(_label(cat, lam), fontsize=9)
            ax.set_xlabel("loop")
            ax.set_ylabel("accuracy (%)")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, loc="upper right")
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.suptitle(f"Accuracy vs loop — row vs col  (rt_error={rt:g}, seeds={rvc.seeds_present(df)})",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        _save(fig, out_dir, f"traj_rt{rt:g}", formats)


def fig_fault_loop(df, red, out_dir, formats):
    """BER and affected_units vs loop, col vs row, per category — one figure per metric."""
    specs = [("ber", "fault_last", "BER (final-loop only)"),  # ber is scalar; fall back to bitflips trajectory
             ("affected_units.total", "fault", "affected units"),
             ("bitflips.total", "fault", "bitflips")]
    cells = _cells(df, lam_only_headline=False)
    for metric, family, ylabel in specs:
        has_loop = red[(red["family"] == family) & (red["metric"] == metric) & red["loop"].notna()]
        if has_loop.empty:
            continue
        n = len(cells)
        ncol = 3
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
        rts = _rt_values(df)
        for ax, (cat, lam) in zip(axes.flat, cells):
            for rt in rts:
                for layout in ("row", "col"):
                    d = red[(red["category"] == cat) & (red["rt_error"] == rt) & (red["layout"] == layout)
                            & (red["family"] == family) & (red["metric"] == metric) & red["loop"].notna()]
                    d = d[d["lambda"].isna()] if (lam is None or pd.isna(lam)) else d[d["lambda"] == lam]
                    d = d.sort_values("loop")
                    if d.empty:
                        continue
                    color = plt.cm.viridis(rts.index(rt) / max(1, len(rts) - 1))
                    ax.plot(d["loop"], d["mean"], color=color, ls=rvc.LAYOUT_LINESTYLE[layout],
                            lw=1.4, label=f"rt={rt:g} {layout}")
            ax.set_title(_label(cat, lam), fontsize=9)
            ax.set_xlabel("loop")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6, loc="best")
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.suptitle(f"{ylabel} vs loop — row (solid) vs col (dashed)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        _save(fig, out_dir, f"fault_loop_{metric.replace('.', '_')}", formats)


def fig_layer_heatmap(df, red, out_dir, formats):
    """Per-layer bitflips heatmap (layer x loop) — one panel per layout, per cell+rt."""
    cells = _cells(df, lam_only_headline=True)
    rts = _rt_values(df)
    for rt in rts:
        for cat, lam in cells:
            layers = sorted({m.split(".", 1)[1]
                             for m in red[(red["category"] == cat) & (red["family"] == "fault")
                                          & red["metric"].str.startswith("bitflips.")]["metric"].unique()
                             if not m.endswith(".total")})
            if not layers:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11, 0.4 * len(layers) + 2), squeeze=False)
            panel_max = {}
            # Per-panel colour scale (keeps each panel's internal layer structure
            # readable); the row/col magnitude gap is annotated as text instead of
            # a shared vmax that would black out the smaller-magnitude panel.
            for j, layout in enumerate(("row", "col")):
                ax = axes[0, j]
                mat = []
                ylabels = []
                for layer in layers:
                    d = red[(red["category"] == cat) & (red["rt_error"] == rt) & (red["layout"] == layout)
                            & (red["family"] == "fault") & (red["metric"] == f"bitflips.{layer}")
                            & red["loop"].notna()]
                    d = d[d["lambda"].isna()] if (lam is None or pd.isna(lam)) else d[d["lambda"] == lam]
                    d = d.sort_values("loop")
                    if d.empty:
                        continue
                    mat.append(d["mean"].to_numpy())
                    ylabels.append(layer)
                if not mat:
                    ax.axis("off")
                    continue
                mat = np.array(mat)
                panel_max[layout] = float(mat.max())
                im = ax.imshow(mat, aspect="auto", cmap="magma", origin="lower")
                ax.set_yticks(range(len(ylabels)))
                ax.set_yticklabels(ylabels, fontsize=8)
                ax.set_xlabel("loop")
                ax.set_title(f"{layout}  (peak {panel_max[layout]:,.0f} bitflips)", fontsize=10)
                fig.colorbar(im, ax=ax, fraction=0.046, label="bitflips")
            ratio = ""
            if panel_max.get("col") and panel_max.get("row"):
                ratio = f"   row peak / col peak = {panel_max['row'] / panel_max['col']:.1f}×"
            fig.suptitle(f"Per-layer bitflips — {_label(cat, lam)}  (rt={rt:g}){ratio}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            _save(fig, out_dir, f"layer_heatmap_{rvc.cell_key(cat, 'x', lam).replace('__x', '')}_rt{rt:g}", formats)


# ---------------------------------------------------------------------------
# CATEGORY-RANKING figures
# ---------------------------------------------------------------------------
def _final_accuracy(red, cat, lam, rt, layout):
    d = red[(red["category"] == cat) & (red["rt_error"] == rt) & (red["layout"] == layout)
            & (red["family"] == "outcome") & (red["metric"] == "accuracy") & red["loop"].notna()]
    d = d[d["lambda"].isna()] if (lam is None or pd.isna(lam)) else d[d["lambda"] == lam]
    if d.empty:
        return np.nan, 0.0
    row = d.loc[d["loop"].idxmax()]
    return row["mean"], row["std"]


def _scalar(red, cat, lam, rt, layout, family, metric):
    d = red[(red["category"] == cat) & (red["rt_error"] == rt) & (red["layout"] == layout)
            & (red["family"] == family) & (red["metric"] == metric)]
    d = d[d["lambda"].isna()] if (lam is None or pd.isna(lam)) else d[d["lambda"] == lam]
    if d.empty:
        return np.nan
    return d["mean"].iloc[0]


def fig_acc_bars(df, red, out_dir, formats):
    """Grouped final-accuracy bars per category, row vs col, panels: clean + each rt_error."""
    cells = _cells(df, lam_only_headline=False)
    rts = _rt_values(df)
    panels = ["clean"] + [f"rt={rt:g}" for rt in rts]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.2), squeeze=False, sharey=True)
    labels = [_label(c, l) for c, l in cells]
    x = np.arange(len(cells))
    w = 0.38
    for pi, panel in enumerate(panels):
        ax = axes[0, pi]
        for k, layout in enumerate(("row", "col")):
            vals, errs = [], []
            for cat, lam in cells:
                if panel == "clean":
                    v = _scalar(red, cat, lam, rts[0], layout, "outcome", "clean_baseline")
                    e = 0.0
                else:
                    rt = float(panel.split("=")[1])
                    v, e = _final_accuracy(red, cat, lam, rt, layout)
                vals.append(v)
                errs.append(e)
            ax.bar(x + (k - 0.5) * w, vals, w, yerr=errs, capsize=2,
                   label=layout, hatch=rvc.LAYOUT_HATCH[layout],
                   color=[rvc.CATEGORY_COLORS[c] for c, _ in cells],
                   edgecolor="black", alpha=0.85 if layout == "row" else 0.65)
        ax.set_title(panel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        if pi == 0:
            ax.set_ylabel("accuracy (%)")
        ax.legend(fontsize=8, title="layout")
    fig.suptitle(f"Final accuracy by category — row vs col  (seeds={rvc.seeds_present(df)})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, "acc_bars", formats)


def fig_ber_bars(df, red, out_dir, formats):
    """Final-BER ranking bars across categories, row vs col, one panel per rt_error."""
    cells = _cells(df, lam_only_headline=False)
    rts = _rt_values(df)
    fig, axes = plt.subplots(1, len(rts), figsize=(5.5 * len(rts), 4.2), squeeze=False, sharey=True)
    labels = [_label(c, l) for c, l in cells]
    x = np.arange(len(cells))
    w = 0.38
    for pi, rt in enumerate(rts):
        ax = axes[0, pi]
        for k, layout in enumerate(("row", "col")):
            vals = [_scalar(red, cat, lam, rt, layout, "fault_last", "ber") for cat, lam in cells]
            ax.bar(x + (k - 0.5) * w, vals, w, label=layout, hatch=rvc.LAYOUT_HATCH[layout],
                   color=[rvc.CATEGORY_COLORS[c] for c, _ in cells],
                   edgecolor="black", alpha=0.85 if layout == "row" else 0.65)
        ax.set_title(f"rt={rt:g}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        if pi == 0:
            ax.set_ylabel("final-loop BER (over unprotected weights)")
        ax.legend(fontsize=8, title="layout")
    fig.suptitle("Final-loop BER by category — lower is better", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, "ber_bars", formats)


def fig_mech_hist(statics, out_dir, formats):
    """Run-length & alternating-seq histograms per category, row vs col overlaid.

    Uses the 'final delivered' snapshot (last label) of each cell.
    """
    # group statics by (category, lambda)
    groups: dict[tuple, dict[str, dict]] = {}
    for rec in statics:
        key = (rec["category"], rec["lambda"])
        groups.setdefault(key, {})[rec["layout"]] = rec
    ordered = [k for k in sorted(groups, key=lambda kv: (rvc.CATEGORY_ORDER.index(kv[0]), kv[1] or -1))]

    for hist_key, title, fname in [
        ("run_length_histogram", "Run-length histogram", "mech_runlength"),
        ("alternating_seq_histogram", "Alternating-seq histogram", "mech_altseq"),
    ]:
        n = len(ordered)
        ncol = 3
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow), squeeze=False)
        for ax, key in zip(axes.flat, ordered):
            cat, lam = key
            for layout in ("row", "col"):
                rec = groups[key].get(layout)
                if not rec or not rec["snapshots"]:
                    continue
                snap = rec["snapshots"][-1]  # final delivered
                hist = snap.get(hist_key, {})
                if not hist:
                    continue
                xs, ys = rvc.hist_to_xy(hist, normalize=True)
                ax.plot(xs, ys, ls=rvc.LAYOUT_LINESTYLE[layout], marker=rvc.LAYOUT_MARKER[layout],
                        markersize=3, color=rvc.CATEGORY_COLORS[cat], label=f"{layout} ({snap['label']})",
                        linewidth=1.5)
            ax.set_title(_label(cat, lam), fontsize=9)
            ax.set_xlabel("length")
            ax.set_ylabel("fraction")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.suptitle(f"{title} — row vs col (final weights, log-y, normalized)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        _save(fig, out_dir, fname, formats)


def fig_pipeline(statics, out_dir, formats):
    """cat2/cat4/cat6 stage deltas (before->after_encoder->after_recal)."""
    metrics = [("bitflips", "bitflips"),
               ("block_count_change", "Δ block count"),
               ("alternating_seq_count_change", "Δ alt-seq count"),
               ("abs_dist_to_threshold_change_mean", "mean |Δ dist-to-threshold|")]
    # collect cells that have deltas
    rows = []  # (cat, lam, layout, stage, metric, value)
    for rec in statics:
        for stage, blob in (rec["deltas"] or {}).items():
            tot = blob.get("total", {})
            for mk, _ in metrics:
                if mk in tot:
                    rows.append((rec["category"], rec["lambda"], rec["layout"], stage, mk, tot[mk]))
    if not rows:
        print("  (no pipeline deltas found)")
        return
    dfp = pd.DataFrame(rows, columns=["category", "lambda", "layout", "stage", "metric", "value"])
    cats_lams = sorted(dfp.groupby(["category", "lambda"], dropna=False).groups,
                       key=lambda kv: (rvc.CATEGORY_ORDER.index(kv[0]), kv[1] or -1))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(8, 1.6 * len(cats_lams)), 3.0 * len(metrics)),
                             squeeze=False)
    for mi, (mk, mlabel) in enumerate(metrics):
        ax = axes[mi, 0]
        sub = dfp[dfp["metric"] == mk]
        stages = list(dict.fromkeys(sub["stage"]))
        xlabels = [f"{_label(c, l)}" for c, l in cats_lams]
        x = np.arange(len(cats_lams))
        nbar = len(stages) * 2  # stage x layout
        w = 0.8 / max(1, nbar)
        bi = 0
        for stage in stages:
            for layout in ("row", "col"):
                vals = []
                for c, l in cats_lams:
                    m = ((sub["category"] == c) & (sub["stage"] == stage) & (sub["layout"] == layout))
                    m &= sub["lambda"].isna() if (l is None or pd.isna(l)) else (sub["lambda"] == l)
                    v = sub[m]["value"]
                    vals.append(v.iloc[0] if not v.empty else np.nan)
                ax.bar(x + (bi - nbar / 2 + 0.5) * w, vals, w,
                       label=f"{stage} [{layout}]", hatch=rvc.LAYOUT_HATCH[layout],
                       edgecolor="black", alpha=0.85)
                bi += 1
        ax.set_ylabel(mlabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=20, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.6)
        ax.grid(axis="y", alpha=0.25)
    # single shared legend below the suptitle (all panels share stage×layout keys)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Encoder/recal pipeline deltas (cat2/cat4/cat6) — row vs col\n"
                 "recal stages (after_encoder→after_recal, trained→after_recal) are ~0: "
                 "recal tunes BN/affine, not quantized weights",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_dir, "pipeline_deltas", formats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=rvc.DEFAULT_RUNS_DIR)
    ap.add_argument("--tidy", type=Path, default=None, help="pre-built tidy_long.csv (else parse runs-dir)")
    ap.add_argument("--out-dir", type=Path, default=None, help="figure output (default <runs-dir>/figures)")
    ap.add_argument("--include", nargs="*", default=None, help="only paths containing any of these tokens")
    ap.add_argument("--figures", nargs="*", default=ALL_FIGURES, choices=ALL_FIGURES, help="which figures")
    ap.add_argument("--format", nargs="*", default=["png"], help="output formats (png pdf svg)")
    args = ap.parse_args()

    if args.tidy and args.tidy.exists():
        df = pd.read_csv(args.tidy)
        df["category"] = pd.Categorical(df["category"], categories=rvc.CATEGORY_ORDER, ordered=True)
    else:
        df = rvc.load_tidy(args.runs_dir, args.include)
    if df.empty:
        print("ERROR: no metrics parsed", file=sys.stderr)
        return 1
    print(f"loaded {len(df):,} tidy rows; seeds present: {rvc.seeds_present(df)}")
    red = rvc.reduce_over_seeds(df)
    out_dir = args.out_dir or (args.runs_dir / "figures")

    need_static = any(f in args.figures for f in ("mech_hist", "pipeline"))
    statics = rvc.load_all_static(args.runs_dir, args.include) if need_static else []

    dispatch = {
        "traj": lambda: fig_traj(df, red, out_dir, args.format),
        "fault_loop": lambda: fig_fault_loop(df, red, out_dir, args.format),
        "layer_heatmap": lambda: fig_layer_heatmap(df, red, out_dir, args.format),
        "acc_bars": lambda: fig_acc_bars(df, red, out_dir, args.format),
        "ber_bars": lambda: fig_ber_bars(df, red, out_dir, args.format),
        "mech_hist": lambda: fig_mech_hist(statics, out_dir, args.format),
        "pipeline": lambda: fig_pipeline(statics, out_dir, args.format),
    }
    for fig in args.figures:
        print(f"[{fig}]")
        dispatch[fig]()
    print(f"\nfigures written under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
