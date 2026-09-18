#!/usr/bin/env python3
"""Block-length / misalignment-mechanism figure catalog for a paper-runs tree.

Answers one question in five figures: *why* does each training category resist
racetrack misalignment better or worse than the others?  The hypothesis under
test is that longer runs of same-signed weights ("blocks") make a shifted read
more likely to return the same bit, so the figures walk the causal chain

    block-length distribution  ->  sign-transition density
                               ->  bit errors per misread cell
                               ->  accuracy under fault

Everything is read straight out of ``metrics_artifacts/``:

  ``*__static.json``      block/run-length histograms, alternating-sequence
                          histograms, sign-transition counts, per layer and
                          in total, for every weight snapshot of the run.
  ``*__rt<err>.json``     per-loop accuracy plus the per-loop fault counters
                          (``bitflips``, ``wrong_bits_read``, ...), total and
                          per layer.

The tree is a MATCHED experiment: at a given seed every category is handed the
bit-identical fault realisation (same wires misaligned, same offsets, same
``wrong_bits_read``).  Only the *stored bits* differ, which is exactly the
variable the hypothesis is about -- so every between-category difference in
``bitflips`` is attributable to block structure alone.

Figures (``--figures`` picks a subset)::

    blocklen      run-length distribution: P(L), enrichment vs baseline,
                  weight-weighted survival
    alternating   alternating-sequence histogram + change vs baseline
    perlayer      mean block length per layer, and its change vs baseline
    mechanism     structure -> damage: the transition-density identity, and
                  how the advantage erodes as offsets accumulate
    payoff        damage / tolerance / cost / outcome decomposition

Usage::

    python scripts/plot_blocklen_png.py
    python scripts/plot_blocklen_png.py --figures blocklen mechanism
    python scripts/plot_blocklen_png.py --rt-error 4.55e-05 --out /tmp/figs
    python scripts/plot_blocklen_png.py --only cat1_baseline cat5_regularizer \
        cat6_reg_recal cat8_ste_inject

Data extraction below is stdlib-only and returns plain dicts/lists, so the
planned TikZ emitter can reuse it without matplotlib.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = REPO_ROOT / "runs" / "paper-runs" / "paper-sw_resnet18_imagenette"
DEFAULT_OUT = REPO_ROOT / "docs" / "figures" / "blocklen"

# --------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------
# Presentation order. cat6/cat68 share cat5's checkpoint signs (recalibration
# only touches BN statistics and the affine/scale parameters, never a weight's
# sign), so they carry the SAME structure -- they appear in the outcome figures
# but would draw an identical line in the distribution figures.
CATEGORY_ORDER = [
    "cat1_baseline",
    "cat8_ste_inject",
    "cat5_regularizer",
    "cat6_reg_recal",
    "cat68_reg_ste_recal",
    "cat4_endlen_recal",
]
PRETTY = {
    "cat1_baseline": "Baseline",
    "cat8_ste_inject": "STE inject",
    "cat5_regularizer": "RL-Reg",
    "cat6_reg_recal": "RL-Reg + recal",
    "cat68_reg_ste_recal": "RL-Reg + STE + recal",
    "cat4_endlen_recal": "Endlen + recal",
}
#: Which weight snapshot is the one actually resident in the racetracks.
STRUCTURE_SNAPSHOT = {
    "cat4_endlen_recal": "after_encoder",  # the encoder is what rewrites signs
}
DEFAULT_SNAPSHOT = "trained"

# Slots of the validated default categorical palette, one per arm, stable across
# every figure -- colour follows the entity, never its rank. Several slots sit
# below 3:1 on a white surface, so every series also carries a direct label or an
# axis label (the relief rule).
COLORS = {
    "cat1_baseline": "#2a78d6",
    "cat8_ste_inject": "#eb6834",
    "cat5_regularizer": "#1baf7a",
    "cat6_reg_recal": "#e87ba4",
    "cat68_reg_ste_recal": "#008300",
    "cat4_endlen_recal": "#eda100",
}
#: Arms sharing a structure still get their own hue: the distribution figures
#: draw only one representative per structure, while the outcome figures put
#: those arms side by side and exist to tell them apart. Aqua/magenta sit in the
#: 6-8 CVD floor band, legal because every bar carries its axis label.
MARKERS = {
    "cat1_baseline": "o",
    "cat8_ste_inject": "s",
    "cat5_regularizer": "o",
    "cat6_reg_recal": "s",
    "cat68_reg_ste_recal": "^",
    "cat4_endlen_recal": "D",
}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e3e2de"

SEEDS = ("707", "808", "909")


# --------------------------------------------------------------------------
# Data extraction (stdlib only)
# --------------------------------------------------------------------------
def find_runs(root: Path, rt_tag: str) -> dict[tuple[str, str], Path]:
    """Map (category, seed) -> the one canonical run directory.

    The tree carries re-runs (bit-identical repeats written on a later day) and
    aborted directories with no ``metrics_artifacts``.  A directory qualifies
    only if it holds BOTH a ``*__static.json`` and the requested rt_error file;
    among survivors the oldest is canonical, so a later partial re-run never
    displaces the sweep it came from.
    """
    runs: dict[tuple[str, str], Path] = {}
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        for sub in sorted(cat_dir.iterdir()):
            seed = sub.name.rsplit("seed", 1)[-1]
            cands = []
            for ts in sorted(sub.iterdir()):
                ma = ts / "metrics_artifacts"
                if not ma.is_dir():
                    continue
                names = os.listdir(ma)
                has_static = any(n.endswith("static.json") for n in names)
                has_rt = any(n.endswith(f"rt{rt_tag}.json") for n in names)
                if has_static and has_rt:
                    cands.append(ts)
            if cands:
                # Sort by DIRECTORY NAME, which is the run timestamp. On a
                # mutagen-synced mount st_mtime is the sync time, not the run
                # time, so it would order the tree by when it last replicated.
                cands.sort(key=lambda p: p.name)
                runs[(cat_dir.name, seed)] = cands[0]
    return runs


def _artifact(run_dir: Path, suffix: str) -> Path:
    ma = run_dir / "metrics_artifacts"
    hits = [p for p in ma.iterdir() if p.name.endswith(suffix)]
    if not hits:
        raise FileNotFoundError(f"no *{suffix} under {ma}")
    return hits[0]


def load_structure(run_dir: Path, category: str) -> dict:
    """Static weight structure of the snapshot that is resident in the racetracks.

    Restricted to the UNPROTECTED layers -- the protected ones (``conv1``,
    ``linear``) never see a fault, so including them would put weights in the
    histogram that cannot contribute a bit error.
    """
    doc = json.loads(_artifact(run_dir, "static.json").read_text())
    label = STRUCTURE_SNAPSHOT.get(category, DEFAULT_SNAPSHOT)
    snaps = {s["label"]: s for s in doc["snapshots"]}
    if label not in snaps:  # e.g. a category that never ran an encoder
        label = doc["snapshots"][-1]["label"]
    per_layer = snaps[label]["per_layer"]
    unprotected = [l for l in doc["meta"]["protection"]["unprotected"] if l in per_layer]

    run_hist: dict[int, int] = defaultdict(int)
    alt_hist: dict[int, int] = defaultdict(int)
    transitions = 0
    wires = 0
    layers: dict[str, dict] = {}
    for name in unprotected:
        st = per_layer[name]
        h = {int(k): v for k, v in st["run_length_histogram"].items()}
        a = {int(k): v for k, v in st["alternating_seq_histogram"].items()}
        nrt = 1
        for dim in st["n_racetracks"]:
            nrt *= dim
        bits = sum(k * v for k, v in h.items())
        for k, v in h.items():
            run_hist[k] += v
        for k, v in a.items():
            alt_hist[k] += v
        transitions += st["sign_transitions"]
        wires += nrt
        layers[name] = {
            "run_hist": h,
            "alt_hist": a,
            "bits": bits,
            "wires": nrt,
            "transitions": st["sign_transitions"],
            "mean_run_length": bits / sum(h.values()),
            # P(neighbouring cell holds the opposite sign): sign_transitions
            # counted over the (bits - wires) ADJACENT PAIRS, since the last
            # cell of a wire has no successor inside that wire.
            "transition_density": st["sign_transitions"] / (bits - nrt),
        }
    bits = sum(k * v for k, v in run_hist.items())
    return {
        "label": label,
        "layers": layers,
        "layer_order": unprotected,
        "run_hist": dict(run_hist),
        "alt_hist": dict(alt_hist),
        "bits": bits,
        "wires": wires,
        "blocks": sum(run_hist.values()),
        "transitions": transitions,
        "mean_run_length": bits / sum(run_hist.values()),
        "transition_density": transitions / (bits - wires),
    }


def load_dynamic(run_dir: Path, rt_tag: str) -> dict:
    """Per-loop accuracy and fault counters for one run at one rt_error."""
    doc = json.loads(_artifact(run_dir, f"rt{rt_tag}.json").read_text())
    outcome, inc = doc["outcome"], doc["fault_incidence"]
    return {
        "clean": outcome["baselines"]["clean"],
        "endlen": outcome["baselines"]["endlen"],
        "endlen_recal": outcome["baselines"]["endlen_recal"],
        "acc_mean": outcome["accuracy"]["mean"],
        "acc_per_loop": outcome["per_loop_accuracy"],
        "bitflips": inc["per_loop"]["bitflips"]["total"],
        "wrong_reads": inc["per_loop"]["wrong_bits_read"]["total"],
        "bitflips_layer": inc["per_loop"]["bitflips"]["per_layer"],
        "wrong_reads_layer": inc["per_loop"]["wrong_bits_read"]["per_layer"],
        "last": inc["last_loop"],
        "unprotected_weights": inc["last_loop"]["ber_denominator_unprotected_weights"],
    }


def acc_at_ber(acc: list[float], bitflips: list[int], denom: int, target: float):
    """Accuracy where the damage level first reaches ``target`` BER.

    Comparing arms at MATCHED damage separates tolerance (how much accuracy a
    given number of wrong bits costs) from damage rate (how fast wrong bits
    accumulate). Returns ``None`` for a run that never reaches ``target``.
    """
    ber = [b / denom for b in bitflips]
    for i in range(len(ber) - 1):
        if ber[i] <= target <= ber[i + 1]:
            span = ber[i + 1] - ber[i]
            w = 0.0 if span == 0 else (target - ber[i]) / span
            return acc[i] + w * (acc[i + 1] - acc[i])
    return None


def loops_above(acc: list[float], threshold: float) -> int:
    """How many inference iterations run before accuracy first drops below."""
    for i, a in enumerate(acc):
        if a < threshold:
            return i
    return len(acc)


def collect(root: Path, rt_tag: str, only: list[str] | None = None) -> dict:
    """Everything the figures need, keyed by category.

    ``only`` restricts the arm set (same spelling as ``regen_figures.sh``'s
    ``--only``). Every axis range in the figures is derived from whatever
    survives this filter, so a subset re-scales instead of leaving the plots
    stretched around an arm that is no longer drawn.
    """
    runs = find_runs(root, rt_tag)
    cats = [c for c in CATEGORY_ORDER if any(k[0] == c for k in runs)]
    if only:
        missing = [c for c in only if c not in cats]
        if missing:
            raise SystemExit(f"--only: no complete runs for {', '.join(missing)}")
        cats = [c for c in cats if c in only]
    data: dict[str, dict] = {}
    for cat in cats:
        seeds = sorted(s for (c, s) in runs if c == cat)
        struct = load_structure(runs[(cat, seeds[0])], cat)
        dyn = {s: load_dynamic(runs[(cat, s)], rt_tag) for s in seeds}
        data[cat] = {"structure": struct, "dynamic": dyn, "seeds": seeds,
                     "run_dirs": {s: runs[(cat, s)] for s in seeds}}
    return {"categories": cats, "data": data, "rt_tag": rt_tag}


def structure_arms(bundle) -> list[str]:
    """One representative per DISTINCT weight structure, in presentation order.

    Arms that differ only by a post-training step which cannot flip a sign
    (recalibration) hold byte-identical histograms and would draw exactly the
    same curve; the first of each such group stands for it.
    """
    seen: dict[tuple, str] = {}
    for cat in bundle["categories"]:
        s = bundle["data"][cat]["structure"]
        key = (s["blocks"], s["transitions"], s["bits"])
        seen.setdefault(key, cat)
    return [c for c in bundle["categories"] if c in set(seen.values())]


def shared_structure_note(bundle) -> str:
    """'cat6 draws the same curve as cat5' -- as a sentence, or '' if none do."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for cat in bundle["categories"]:
        s = bundle["data"][cat]["structure"]
        groups[(s["blocks"], s["transitions"], s["bits"])].append(cat)
    dupes = [g for g in groups.values() if len(g) > 1]
    if not dupes:
        return ""
    parts = [f"{' and '.join(PRETTY[c] for c in g[1:])} share {PRETTY[g[0]]}'s weight signs"
             for g in dupes]
    return "; ".join(parts) + " (recalibration never flips one) and draw the same curve"


# --------------------------------------------------------------------------
# Derived series
# --------------------------------------------------------------------------
def block_pmf(hist: dict[int, int], lmax: int) -> tuple[list[int], list[float]]:
    """P(block has length L) over all blocks."""
    n = sum(hist.values())
    ls = [L for L in range(1, lmax + 1)]
    return ls, [hist.get(L, 0) / n for L in ls]


def weight_survival(hist: dict[int, int], lmax: int) -> tuple[list[int], list[float]]:
    """Share of weight bits that sit inside a block of length >= L."""
    total = sum(k * v for k, v in hist.items())
    ls = list(range(1, lmax + 1))
    return ls, [sum(k * v for k, v in hist.items() if k >= L) / total for L in ls]


def conversion_series(dyn: dict[str, dict]) -> list[float]:
    """P(bitflip | cell read at a wrong index), per loop, pooled over seeds.

    ``bitflips`` and ``wrong_bits_read`` are both end-of-loop snapshots of
    standing corruption, so their ratio at loop k is the probability that the
    standing offset lands on a cell of the opposite sign -- the structural
    quantity, read directly off the hardware counters.
    """
    seeds = sorted(dyn)
    n = len(dyn[seeds[0]]["bitflips"])
    out = []
    for i in range(n):
        b = sum(dyn[s]["bitflips"][i] for s in seeds)
        w = sum(dyn[s]["wrong_reads"][i] for s in seeds)
        out.append(b / w if w else float("nan"))
    return out


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def _style(ax, *, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3, width=0.8)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SOFT)
    ax.yaxis.label.set_color(INK_SOFT)
    ax.title.set_color(INK)


def _panel_title(ax, text, sub=None):
    # The subtitle sits just above the axes, so the title has to clear it.
    ax.set_title(text, fontsize=10, fontweight="bold", loc="left", pad=20 if sub else 6)
    if sub:
        ax.text(0, 1.008, sub, transform=ax.transAxes, fontsize=8, color=INK_MUTED,
                ha="left", va="bottom")


def _place_end_labels(ax, items, *, dx=6, min_gap=11.5, size=8, ha="left"):
    """Label series at their last point, pushed apart so coincident lines stay legible.

    Several arms draw curves that lie on top of each other (cat8's structure is
    a copy of the baseline's), which would stack their labels into an unreadable
    blot. Separation is enforced in DISPLAY space, so it holds on log axes too --
    call this only once the axes limits and scales are final.
    """
    if not items:
        return
    disp = [ax.transData.transform((x, y)) for x, y, _, _ in items]
    order = sorted(range(len(items)), key=lambda i: disp[i][1])
    ys = [disp[i][1] for i in order]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + min_gap)
    inv = ax.transData.inverted()
    for k, i in enumerate(order):
        x, y, text, color = items[i]
        _, y_shifted = inv.transform((disp[i][0], ys[k]))
        ax.annotate(text, xy=(x, y_shifted), xytext=(dx, 0), textcoords="offset points",
                    color=color, fontsize=size, va="center", ha=ha,
                    fontweight="bold", annotation_clip=False)


def _titles(fig, title, sub=None, *, top=0.86):
    """Reserve headroom, then set the figure title flush left above it."""
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.text(0.006, 0.995, title, fontsize=12.5, fontweight="bold", color=INK,
             ha="left", va="top")
    if sub:
        fig.text(0.006, 0.995 - (1 - top) * 0.42, sub, fontsize=8.5, color=INK_MUTED,
                 ha="left", va="top")


def _log_ticks(ax, axis, values):
    """Plain decimal labels on a log axis (the default 6x10^-2 clutter collides)."""
    import matplotlib.ticker as mticker
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(mticker.FixedLocator(values))
    target.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in values]))
    target.set_minor_locator(mticker.NullLocator())


def _trim_noise(hist: dict[int, int], lmax: int, min_count: int) -> int:
    """Largest L worth plotting: beyond this the baseline holds too few blocks."""
    ok = [L for L in range(1, lmax + 1) if hist.get(L, 0) >= min_count]
    return max(ok) if ok else lmax


def _save(fig, out: Path, name: str, fmts, dpi: int):
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in fmts:
        path = out / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        written.append(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# Figure 1: block-length distribution
# --------------------------------------------------------------------------
def fig_blocklen(bundle, out: Path, fmts, dpi, lmax=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    data = bundle["data"]
    arms = structure_arms(bundle)
    base = data["cat1_baseline"]["structure"]["run_hist"]
    # Plot out to where the distributions still hold enough blocks to be a
    # measurement rather than a single lucky racetrack.
    if lmax is None:
        lmax = max(max((L for L, n in data[c]["structure"]["run_hist"].items() if n >= 5),
                       default=1) for c in arms)
    # Past this length the baseline holds only a handful of blocks, so a ratio
    # against it is sampling noise rather than structure.
    lmax_ratio = _trim_noise(base, lmax, min_count=200)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))
    jobs = []

    # (a) the distribution itself, against a fair-coin reference ---------------
    ax = axes[0]
    ls = list(range(1, lmax + 1))
    ax.plot(ls, [0.5 ** L for L in ls], color=INK_MUTED, linewidth=1.4,
            linestyle=(0, (4, 2)), zorder=1)
    items = [(ls[-1], 0.5 ** ls[-1], "  random signs", INK_MUTED)]
    for cat in arms:
        h = data[cat]["structure"]["run_hist"]
        x, p = block_pmf(h, lmax)
        pts = [(a, b) for a, b in zip(x, p) if b > 0]
        ax.plot([a for a, _ in pts], [b for _, b in pts], color=COLORS[cat],
                linewidth=2, marker=MARKERS[cat], markersize=3.5, zorder=3)
        items.append((pts[-1][0], pts[-1][1], f"  {PRETTY[cat]}", COLORS[cat]))
    ax.set_yscale("log")
    ax.set_xlabel("block length $L$ (cells of equal sign)")
    ax.set_ylabel("share of all blocks")
    ax.set_xlim(0.5, lmax + 4.0)
    _style(ax, grid_axis="both")
    _panel_title(ax, "a  Block-length distribution",
                 "the baseline is a fair coin"
                 + ("; the encoder obeys a different law"
                    if any(data[c]["structure"]["mean_run_length"] > 4 for c in arms)
                    else ", and no training arm leaves that law"))
    jobs.append((ax, items))

    # (b) enrichment: what the mean hides -------------------------------------
    ax = axes[1]
    nb = sum(base.values())
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
    items = []
    for cat in arms:
        if cat == "cat1_baseline":
            continue
        h = data[cat]["structure"]["run_hist"]
        n = sum(h.values())
        xs, ys = [], []
        for L in range(1, lmax_ratio + 1):
            pb = base.get(L, 0) / nb
            if pb <= 0 or h.get(L, 0) == 0:
                continue
            xs.append(L)
            ys.append((h[L] / n) / pb)
        ax.plot(xs, ys, color=COLORS[cat], linewidth=2, marker=MARKERS[cat],
                markersize=3.5, zorder=3)
        items.append((xs[-1], ys[-1], f"  {PRETTY[cat]}", COLORS[cat]))
    ratios = [y for _, y, _, _ in items]
    if ratios and max(ratios) / min(ratios) > 8:
        ax.set_yscale("log")
    ax.set_xlabel("block length $L$")
    ax.set_ylabel("blocks at $L$, relative to baseline")
    ax.set_xlim(0.5, lmax_ratio + 3.0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.text(0.02, 0.96, f"$L \\leq {lmax_ratio}$: beyond it the baseline\nhas too few blocks to divide by",
            transform=ax.transAxes, fontsize=7, color=INK_MUTED, ha="left", va="top")
    _style(ax, grid_axis="both")
    _panel_title(ax, "b  Enrichment over baseline",
                 "the regulariser grows the tail, it does not move the bulk")
    jobs.append((ax, items))

    # (c) where the weights actually sit --------------------------------------
    ax = axes[2]
    items = []
    for cat in arms:
        h = data[cat]["structure"]["run_hist"]
        x, s = weight_survival(h, lmax)
        pts = [(a, b) for a, b in zip(x, s) if b > 0]
        ax.plot([a for a, _ in pts], [100 * b for _, b in pts], color=COLORS[cat],
                linewidth=2, zorder=3)
        items.append((pts[-1][0], 100 * pts[-1][1], f"  {PRETTY[cat]}", COLORS[cat]))
    ax.set_yscale("log")
    ax.set_xlabel("block length $L$")
    ax.set_ylabel("% of weight bits in a block $\\geq L$")
    ax.set_xlim(0.5, lmax + 4.0)
    _style(ax, grid_axis="both")
    _panel_title(ax, "c  Weight mass in long blocks",
                 "the share of weights a 1-cell shift cannot corrupt")
    # The encoder's curve runs past the right edge -- name where it ends, since
    # that tail is the whole point of the axis.
    # Only worth a note if an arm's off-axis tail actually carries weight --
    # a handful of long blocks in one racetrack is not a finding.
    def _tail_mass(cat):
        h = data[cat]["structure"]["run_hist"]
        total = sum(k * v for k, v in h.items())
        return sum(k * v for k, v in h.items() if k > lmax) / total
    off_axis = [c for c in arms if _tail_mass(c) >= 0.005]
    enc = max(off_axis, key=lambda c: max(data[c]["structure"]["run_hist"])) if off_axis else None
    if enc:
        h = data[enc]["structure"]["run_hist"]
        total = sum(k * v for k, v in h.items())
        rt_size = max(h)
        ge32 = sum(k * v for k, v in h.items() if k >= 32) / total
        at_max = h.get(rt_size, 0) * rt_size / total
        # Bottom-left is the empty corner here: every curve descends left to right.
        ax.text(0.03, 0.05,
                f"{PRETTY[enc]} continues past $L={lmax}$:\n"
                f"{100 * ge32:.1f} % of weights in blocks $\\geq 32$,\n"
                f"{100 * at_max:.1f} % in a full {rt_size}-cell wire",
                transform=ax.transAxes, fontsize=7.5, color=COLORS[enc],
                ha="left", va="bottom", fontweight="bold")
    jobs.append((ax, items))

    _titles(fig,
            "Longer same-sign blocks: what each training category actually changes",
            "ResNet-18 / Imagenette, w1a1, the 19 unprotected conv layers"
            + (("  ·  " + shared_structure_note(bundle)) if shared_structure_note(bundle) else ""),
            top=0.85)
    fig.canvas.draw()
    for ax, items in jobs:
        _place_end_labels(ax, items)
    return _save(fig, out, "fig1_blocklen_distribution", fmts, dpi)


# --------------------------------------------------------------------------
# Figure 2: alternating sequences (the vulnerability dual)
# --------------------------------------------------------------------------
def fig_alternating(bundle, out: Path, fmts, dpi, lmax=14):
    import matplotlib.pyplot as plt

    data = bundle["data"]
    arms = structure_arms(bundle)
    base = data["cat1_baseline"]["structure"]["alt_hist"]

    lmax_ratio = _trim_noise(base, lmax, min_count=200)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    jobs = []

    ax = axes[0]
    items, absent = [], []
    for cat in arms:
        a = data[cat]["structure"]["alt_hist"]
        xs = [L for L in range(2, lmax + 1) if a.get(L, 0) > 0]
        if not xs:
            absent.append(cat)
            continue
        ys = [a[L] for L in xs]
        ax.plot(xs, ys, color=COLORS[cat], linewidth=2, marker=MARKERS[cat],
                markersize=3.5, zorder=3)
        items.append((xs[-1], ys[-1], f"  {PRETTY[cat]}", COLORS[cat]))
    ax.set_yscale("log")
    ax.set_xlabel("length of the alternating stretch (cells)")
    ax.set_ylabel("number of stretches")
    ax.set_xlim(1.5, max(x for x, *_ in items) + 3.2)
    _style(ax, grid_axis="both")
    _panel_title(ax, "a  Strictly alternating stretches",
                 "every cell in one of these flips when the head shifts by 1")
    # A log axis cannot draw zero, and zero is the whole point for the encoder.
    for i, cat in enumerate(absent):
        ax.text(0.97, 0.90 - 0.10 * i,
                f"{PRETTY[cat]}: not one stretch,\nat any length, in any encoded layer",
                transform=ax.transAxes, fontsize=8, color=COLORS[cat],
                ha="right", va="top", fontweight="bold")
    jobs.append((ax, items))

    ax = axes[1]
    ax.axhline(0.0, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
    items = []
    for cat in arms:
        if cat == "cat1_baseline":
            continue
        a = data[cat]["structure"]["alt_hist"]
        xs, ys = [], []
        for L in range(2, lmax_ratio + 1):
            if base.get(L, 0) == 0:
                continue
            xs.append(L)
            ys.append(100 * (a.get(L, 0) - base[L]) / base[L])
        ax.plot(xs, ys, color=COLORS[cat], linewidth=2, marker=MARKERS[cat],
                markersize=3.5, zorder=3)
        items.append((xs[-1], ys[-1], f"  {PRETTY[cat]}", COLORS[cat]))
    ax.set_xlabel("length of the alternating stretch (cells)")
    ax.set_ylabel("change vs baseline (%)")
    ax.set_xlim(1.5, lmax_ratio + 3.2)
    ch = [y for _, y, _, _ in items] or [0.0]
    span = max(ch + [0.0]) - min(ch + [0.0])
    ax.set_ylim(min(ch + [0.0]) - 0.12 * span, max(ch + [0.0]) + 0.12 * span)
    _style(ax, grid_axis="both")
    _panel_title(ax, "b  Change vs baseline",
                 "the longer the stretch, the harder the regulariser cuts it")
    jobs.append((ax, items))

    _titles(fig,
            "The dual signal: the most misalignment-vulnerable stretches go first",
            "an alternating stretch is the worst case for a 1-cell shift -- every cell in it "
            "reads back inverted",
            top=0.85)
    fig.canvas.draw()
    for ax, items in jobs:
        _place_end_labels(ax, items)
    return _save(fig, out, "fig2_alternating_sequences", fmts, dpi)


# --------------------------------------------------------------------------
# Figure 3: per-layer view
# --------------------------------------------------------------------------
def fig_perlayer(bundle, out: Path, fmts, dpi):
    import matplotlib.pyplot as plt

    data = bundle["data"]
    arms = structure_arms(bundle)
    order = data["cat1_baseline"]["structure"]["layer_order"]
    ypos = list(range(len(order)))[::-1]

    # cat1 and cat8 land on the same value in almost every layer, so a straight
    # overlay hides one of them entirely -- dodge each arm onto its own row band.
    def _dodge(cats):
        n = len(cats)
        span = 0.62
        return {c: (i - (n - 1) / 2) * span / max(n - 1, 1) for i, c in enumerate(cats)}

    def _mrl(cat):
        layers = data[cat]["structure"]["layers"]
        return [layers[l]["mean_run_length"] for l in order]

    # An arm an order of magnitude clear of the rest (the endlen encoder) would
    # crush every other arm into one column, so panel (a) breaks the axis around
    # it. With no such arm in the set, one plain axis is the honest layout.
    means = {c: sum(_mrl(c)) / len(order) for c in arms}
    near = [c for c in arms if means[c] < 3 * min(means.values())]
    far = [c for c in arms if c not in near]

    fig = plt.figure(figsize=(12.6, 5.9))
    if far:
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.30, 0.10, 1.12], wspace=0.10)
        axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        axB = fig.add_subplot(gs[3])
        panes = ((axL, near), (axR, far))
    else:
        gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.10, 1.12], wspace=0.10)
        axL = fig.add_subplot(gs[0])
        axR = None
        axB = fig.add_subplot(gs[2])
        panes = ((axL, near),)

    off = _dodge(arms)
    for ax, cats in panes:
        for cat in cats:
            ys = [y + off[cat] for y in ypos]
            ax.plot(_mrl(cat), ys, color=COLORS[cat], linewidth=0, marker=MARKERS[cat],
                    markersize=5.5, markeredgecolor="white", markeredgewidth=0.7,
                    label=PRETTY[cat], zorder=3)
        ax.set_yticks(ypos)
        ax.set_ylim(-0.9, len(order) - 0.1)
        _style(ax, grid_axis="x")

    axL.axvline(2.0, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
    axL.set_yticklabels(order, fontsize=7.5)
    near_vals = [v for c in near for v in _mrl(c)]
    pad = 0.08 * (max(near_vals) - min(near_vals))
    axL.set_xlim(min(near_vals) - pad, max(near_vals) + pad)
    axL.text(2.0 + pad * 0.12, len(order) - 0.45, "random signs", fontsize=7.5,
             color=INK_MUTED, ha="left", va="center")
    if axR is not None:
        far_vals = [v for c in far for v in _mrl(c)]
        fpad = 0.35 * (max(far_vals) - min(far_vals)) or 0.5
        axR.set_yticklabels([])
        axR.set_xlim(min(far_vals) - fpad, max(far_vals) + fpad)
        # Break marks: hide the facing spines and draw the diagonal cut.
        axL.spines["right"].set_visible(False)
        axR.spines["left"].set_visible(False)
        axR.tick_params(left=False)
        kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=7, linestyle="none",
                  color=INK_MUTED, mec=INK_MUTED, mew=1.1, clip_on=False)
        axL.plot([1, 1], [0, 1], transform=axL.transAxes, **kw)
        axR.plot([0, 0], [0, 1], transform=axR.transAxes, **kw)
        axR.text(0.5, 1.008, " / ".join(PRETTY[c] for c in far), transform=axR.transAxes,
                 fontsize=8, color=COLORS[far[0]], ha="center", va="bottom",
                 fontweight="bold")

    axL.set_xlabel("mean block length (cells per block)", x=0.75 if axR is not None else 0.5)
    _panel_title(axL, "a  Mean block length, per layer",
                 "the trained arms never leave the neighbourhood of a random sequence")
    leg = axL.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.96,
                     edgecolor=GRID, borderpad=0.6)
    for t in leg.get_texts():
        t.set_color(INK_SOFT)

    ax = axB
    base_layers = data["cat1_baseline"]["structure"]["layers"]
    # The baseline is the reference (a flat zero line), and a far-off-scale arm
    # would compress everything else into the axis.
    shown = [c for c in arms if c != "cat1_baseline" and c not in far]
    off = _dodge(shown)
    ax.axvline(0.0, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
    deltas_all = []
    for cat in shown:
        layers = data[cat]["structure"]["layers"]
        xs = [100 * (layers[l]["mean_run_length"] / base_layers[l]["mean_run_length"] - 1)
              for l in order]
        deltas_all.extend(xs)
        ys = [y + off[cat] for y in ypos]
        ax.plot(xs, ys, color=COLORS[cat], linewidth=0, marker=MARKERS[cat],
                markersize=5.5, markeredgecolor="white", markeredgewidth=0.7,
                label=PRETTY[cat], zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.set_xlabel("change in mean block length vs baseline (%)")
    ax.set_ylim(-0.9, len(order) - 0.1)
    if deltas_all:
        dpad = 0.12 * (max(deltas_all) - min(deltas_all))
        ax.set_xlim(min(deltas_all) - dpad, max(deltas_all) + dpad)
    _style(ax, grid_axis="x")
    _panel_title(ax, "b  Where the regulariser acts",
                 "deep, weight-heavy layers move; early layers barely do")
    if len(shown) > 1:
        leg = ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.96,
                        edgecolor=GRID, borderpad=0.6)
        for t in leg.get_texts():
            t.set_color(INK_SOFT)
    elif shown:
        ax.text(0.98, 0.97, PRETTY[shown[0]], transform=ax.transAxes, fontsize=8.5,
                color=COLORS[shown[0]], ha="right", va="top", fontweight="bold")
    for i, cat in enumerate(far):
        base_mrl = data["cat1_baseline"]["structure"]["mean_run_length"]
        delta = 100 * (data[cat]["structure"]["mean_run_length"] / base_mrl - 1)
        ax.text(0.99, 0.985 - 0.05 * i,
                f"{PRETTY[cat]} is off this scale: {delta:+.0f} %",
                transform=ax.transAxes, fontsize=7.5, color=COLORS[cat],
                ha="right", va="top", fontweight="bold")

    fig.text(0.006, 0.995, "Block lengths layer by layer", fontsize=12.5,
             fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.006, 0.955,
             "the regulariser's reach is uneven -- it concentrates where the weights are",
             fontsize=8.5, color=INK_MUTED, ha="left", va="top")
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.085, right=0.985)
    return _save(fig, out, "fig3_per_layer", fmts, dpi)


# --------------------------------------------------------------------------
# Figure 4: structure -> damage
# --------------------------------------------------------------------------
def fig_mechanism(bundle, out: Path, fmts, dpi):
    import matplotlib.pyplot as plt

    data = bundle["data"]
    # An all-pairs scatter caps the categorical palette at three slots, so the
    # panel takes the first three DISTINCT structures (arms whose structure
    # duplicates another's are already folded out).
    arms = structure_arms(bundle)[:3]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    jobs = []

    # (a) the identity: predicted density vs measured flips-per-misread --------
    # Collect first: the axis is log only when the arms span enough of a range
    # to need it. Restricted to a few near-random arms a log axis would hide
    # the very spread the panel exists to show.
    ax = axes[0]
    series = {}
    for cat in arms:
        struct, dyn = data[cat]["structure"], data[cat]["dynamic"]
        seeds = sorted(dyn)
        pts = []
        for lname in struct["layer_order"]:
            reads = sum(dyn[s]["wrong_reads_layer"][lname][0] for s in seeds)
            if reads == 0:
                continue
            flips = sum(dyn[s]["bitflips_layer"][lname][0] for s in seeds)
            pts.append((struct["layers"][lname]["transition_density"], flips / reads, reads))
        series[cat] = pts
    vals = [v for pts in series.values() for p in pts for v in p[:2]]
    span = max(vals) / min(vals)
    log_scale = span > 3
    if log_scale:
        lo, hi = min(vals) / 1.6, max(vals) * 1.6
    else:
        margin = 0.10 * (max(vals) - min(vals))
        lo, hi = min(vals) - margin, max(vals) + margin
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=1.3,
            linestyle=(0, (4, 2)), zorder=1)
    at = ((lo ** 0.18) * (hi ** 0.82)) if log_scale else lo + 0.84 * (hi - lo)
    ax.text(at, at, "measured = predicted ", fontsize=7.5, color=INK_MUTED,
            rotation=45, rotation_mode="anchor", ha="center", va="bottom",
            transform=ax.transData)
    items = []
    for cat in arms:
        struct = data[cat]["structure"]
        dyn = data[cat]["dynamic"]
        seeds = sorted(dyn)
        xs = [p[0] for p in series[cat]]
        ys = [p[1] for p in series[cat]]
        ss = [p[2] for p in series[cat]]
        smax = max(ss)
        ax.scatter(xs, ys, s=[14 + 120 * (v / smax) ** 0.5 for v in ss],
                   facecolor=COLORS[cat], edgecolor="white", linewidth=0.7,
                   alpha=0.9, zorder=3, marker=MARKERS[cat])
        # the whole-network point, drawn as an outlined anchor
        gx = struct["transition_density"]
        gflips = sum(dyn[s]["bitflips"][0] for s in seeds)
        greads = sum(dyn[s]["wrong_reads"][0] for s in seeds)
        gy = gflips / greads
        ax.scatter([gx], [gy], s=95, facecolor="white", edgecolor=COLORS[cat],
                   linewidth=2.2, zorder=4, marker=MARKERS[cat])
        items.append((gx, gy, f"  {PRETTY[cat]}", COLORS[cat]))
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = [t for t in (0.03, 0.05, 0.1, 0.2, 0.5) if lo <= t <= hi]
        _log_ticks(ax, "x", ticks)
        _log_ticks(ax, "y", ticks)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted: sign-transition density of the stored weights")
    ax.set_ylabel("measured: bit errors per misread cell")
    _style(ax, grid_axis="both")
    _panel_title(ax, "a  Structure predicts damage exactly",
                 "19 layers per arm (area = cells misread); hollow = whole network")
    smallest = min(p[2] for pts in series.values() for p in pts)
    ax.text(0.98, 0.03,
            f"outliers are the smallest layers:\nas few as {smallest} cells misread, so the\n"
            f"ratio is binomial noise, not model error",
            transform=ax.transAxes, fontsize=7, color=INK_MUTED, ha="right", va="bottom")
    jobs.append((ax, items))

    # (b) how the advantage erodes -------------------------------------------
    ax = axes[1]
    items = []
    for cat in structure_arms(bundle):
        curve = conversion_series(data[cat]["dynamic"])
        loops = list(range(1, len(curve) + 1))
        ax.plot(loops, curve, color=COLORS[cat], linewidth=2, zorder=3)
        items.append((loops[-1], curve[-1], f"  {PRETTY[cat]}", COLORS[cat]))
        n_loops = len(curve)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
    ax.text(2, 0.502, "fully decorrelated signs: every second misread flips a bit",
            fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")
    ax.set_xlabel("inference iteration")
    ax.set_ylabel("bit errors per misread cell")
    ax.set_xlim(0, n_loops * 1.22)
    cvals = [v for c in structure_arms(bundle) for v in conversion_series(data[c]["dynamic"])]
    cpad = 0.14 * (max(cvals + [0.5]) - min(cvals))
    ax.set_ylim(max(0.0, min(cvals) - cpad), max(cvals + [0.5]) + cpad)
    _style(ax, grid_axis="both")
    _panel_title(ax, "b  The shield wears off",
                 "long blocks stop a 1-cell shift; accumulated drift outgrows them")
    jobs.append((ax, items))

    seeds_used = sorted({s for cat in data for s in data[cat]["seeds"]})
    _titles(fig,
            "From block structure to bit errors: a shifted read only hurts when it crosses a sign boundary",
            f"rt_error = {bundle['rt_tag']}, pooled over seeds {', '.join(seeds_used)} -- "
            f"every arm receives the bit-identical fault realisation, so only the stored bits differ",
            top=0.85)
    fig.canvas.draw()
    for ax, items in jobs:
        _place_end_labels(ax, items)
    return _save(fig, out, "fig4_mechanism", fmts, dpi)


# --------------------------------------------------------------------------
# Figure 5: what it buys
# --------------------------------------------------------------------------
def fig_payoff(bundle, out: Path, fmts, dpi, ber_target=0.05, acc_threshold=70.0):
    import matplotlib.pyplot as plt

    data = bundle["data"]
    cats = [c for c in CATEGORY_ORDER if c in data]
    base = data["cat1_baseline"]
    base_seeds = sorted(base["dynamic"])

    dmg, tol, cost, endur, tol_note = [], [], [], [], []
    b_last = {s: base["dynamic"][s]["bitflips"][-1] for s in base_seeds}
    base_clean = base["dynamic"][base_seeds[0]]["clean"]
    for cat in cats:
        dyn = data[cat]["dynamic"]
        seeds = sorted(dyn)
        dmg.append(100 * (sum(dyn[s]["bitflips"][-1] for s in seeds)
                          / sum(b_last[s] for s in seeds) - 1))
        # What the deployed model scores with no faults at all. For the encoder
        # arm that is the ENCODED network, not the checkpoint it came from --
        # and its recalibration is stochastic, so average over seeds like
        # everything else rather than reporting whichever seed sorted first.
        eff = [dyn[s]["endlen_recal"] or dyn[s]["endlen"] or dyn[s]["clean"] for s in seeds]
        eff_mean = sum(eff) / len(eff)
        cost.append(eff_mean - base_clean)
        endur.append(sum(loops_above(dyn[s]["acc_per_loop"], acc_threshold)
                         for s in seeds) / len(seeds))

        # "Accuracy at matched damage" is only meaningful for a model that works
        # in the first place. An arm sitting nearer chance than the baseline has
        # nothing left for faults to take away, and plotting its huge negative
        # delta would flatten the panel's real spread to a hairline. Its cost is
        # already the headline of panel c.
        if eff_mean < base_clean / 2:
            tol.append(0.0)
            tol_note.append(f"non-functional before\nany fault ({eff_mean:.0f} %)")
            continue
        deltas = []
        for s in seeds:
            d = dyn[s]
            a = acc_at_ber(d["acc_per_loop"], d["bitflips"], d["unprotected_weights"], ber_target)
            b = base["dynamic"][s]
            a0 = acc_at_ber(b["acc_per_loop"], b["bitflips"], b["unprotected_weights"], ber_target)
            if a is not None and a0 is not None:
                deltas.append(a - a0)
        tol.append(sum(deltas) / len(deltas) if deltas else 0.0)
        tol_note.append(None if deltas else "never reaches\nthis damage level")

    labels = [PRETTY[c] for c in cats]
    colors = [COLORS[c] for c in cats]
    x = list(range(len(cats)))

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))
    panels = [
        (dmg, "a  Damage", "bit errors vs baseline (%)",
         "how much corruption accumulates", None),
        (tol, "b  Tolerance", f"accuracy at matched damage,\nvs baseline (pp)",
         f"accuracy at BER = {ber_target:g}, paired per seed", None),
        (cost, "c  Cost", "fault-free accuracy,\nvs baseline (pp)",
         f"what it costs upfront -- baseline scores {base_clean:.1f} %", None),
        (endur, "d  Outcome", f"iterations above {acc_threshold:g}% accuracy",
         "what actually reaches the user", None),
    ]
    for ax, (vals, title, ylab, sub, floor) in zip(axes, panels):
        plotted = [0.0 if (title.startswith("b") and tol_note[i]) else v
                   for i, v in enumerate(vals)]
        ax.bar(x, plotted, color=colors, width=0.66, zorder=3)
        lo = min(plotted + ([floor] if floor is not None else []))
        hi = max(plotted + ([floor] if floor is not None else []))
        pad = 0.16 * (hi - lo if hi > lo else abs(hi) or 1)
        top = hi + pad * 1.5
        bottom = min(lo - pad, 0) if lo < 0 else 0
        if floor is not None:
            ax.axhline(floor, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
            # Keep the annotation out of the leftmost bar.
            ax.set_xlim(-1.0, len(cats) - 0.4)
            ax.text(-0.95, floor + (top - bottom) * 0.015, f"chance ({floor:.0f}%)",
                    fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")
        else:
            ax.axhline(0.0, color=GRID, linewidth=1.0, zorder=2)
        for i, v in enumerate(vals):
            if title.startswith("b") and tol_note[i]:
                # No bar for an arm with no meaningful value here -- a zero-height
                # bar would read as "no effect", which is wrong.
                ax.text(i, (top - bottom) * 0.03, tol_note[i], fontsize=7,
                        color=INK_MUTED, ha="center", va="bottom", rotation=90,
                        style="italic")
                continue
            nudge = (top - bottom) * 0.025
            ax.text(i, v + (nudge if v >= 0 else -nudge),
                    f"{v:+.1f}" if title[0] in "abc" else f"{v:.1f}",
                    fontsize=8, color=INK_SOFT, ha="center",
                    va="bottom" if v >= 0 else "top", fontweight="bold")
        ax.set_ylim(bottom, top)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=32, ha="right")
        ax.set_ylabel(ylab, fontsize=8.5)
        _style(ax, grid_axis="y")
        _panel_title(ax, title, sub)

    # An arm that pays a large upfront accuracy cost is the headline when one is
    # present; with only trained arms the story is the split between the two axes.
    broken = [(c, v) for c, v in zip(cats, cost) if v < -10]
    n_seeds = len(base_seeds)
    n_loops = len(base["dynamic"][base_seeds[0]]["acc_per_loop"])
    if broken:
        headline = ("Longer blocks do cut the damage -- but only one arm moves them, "
                    "and it pays with the network")
        note = (f"the {PRETTY[broken[0][0]]} arm's deployed model is the ENCODED network, "
                f"which costs {-broken[0][1]:.0f} pp before a single fault arrives")
    else:
        # Name the arms that actually top each axis rather than assuming which
        # ones are in the set.
        best_dmg = cats[min(range(len(cats)), key=lambda i: dmg[i])]
        best_tol = cats[max(range(len(cats)), key=lambda i: tol[i])]
        headline = (f"Damage and tolerance are separate axes: {PRETTY[best_dmg]} moves the "
                    f"first, {PRETTY[best_tol]} moves the second")
        sig = lambda c: (data[c]["structure"]["blocks"], data[c]["structure"]["transitions"])
        twins = [(a, b) for a in cats for b in cats
                 if a != b and a != "cat1_baseline" and sig(a) == sig(b)
                 and cats.index(a) < cats.index(b)]
        if sig(best_tol) == sig("cat1_baseline"):
            note = (f"{PRETTY[best_tol]} changes no block lengths at all (panel a) yet gains "
                    f"the most at matched damage -- that gain is tolerance, not structure")
        elif twins:
            a, b = twins[0]
            note = (f"{PRETTY[a]} and {PRETTY[b]} store identical weight signs, so the gap "
                    f"between them in panels b-d is recalibration alone, with no structural change")
        else:
            note = "paired per seed: every arm sees the bit-identical fault realisation"
    _titles(fig,
            headline,
            f"rt_error = {bundle['rt_tag']}, {n_seeds} seeds, {n_loops} inference iterations  ·  {note}",
            top=0.83)
    return _save(fig, out, "fig5_payoff", fmts, dpi)


# --------------------------------------------------------------------------
# Figure 6: both distributions as one spectrum
# --------------------------------------------------------------------------
def fig_spectrum(bundle, out: Path, fmts, dpi):
    """Block lengths and alternating stretches on one mirrored length axis.

    The two histograms are the same statistic at two levels: ``run_length``
    counts maximal runs of equal sign (in cells), ``alternating_seq`` counts
    maximal streaks of consecutive LENGTH-1 runs (in blocks). Under a fair coin
    both are geometric with the SAME ratio 1/2 -- the alternating arm merely
    shifted down by 1/4, the probability that the streak is bounded on both
    sides by a non-singleton. So one reference curve, mirrored, describes both
    halves, structure reads as ASYMMETRY about the centre, and "the blocks got
    longer" and "the alternating stretches got rarer" become one statement.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    data = bundle["data"]
    arms = structure_arms(bundle)
    base = data["cat1_baseline"]["structure"]

    def trim(hist, floor):
        ok = [L for L, n in hist.items() if n >= 5 and L >= floor]
        return max(ok) if ok else floor
    lmax = max(trim(data[c]["structure"]["run_hist"], 1) for c in arms)
    kmax = max(trim(data[c]["structure"]["alt_hist"], 2) for c in arms)

    fig, ax = plt.subplots(figsize=(11.6, 5.4))

    # One fair-coin law across the whole spectrum, drawn as its two arms.
    n_blocks = base["blocks"]
    ls = list(range(1, lmax + 1))
    ks = list(range(2, kmax + 1))
    ref_r = [(L, n_blocks * 0.5 ** L) for L in ls if n_blocks * 0.5 ** L >= 1]
    ref_l = [(-k, n_blocks * 0.5 ** (k + 2)) for k in ks if n_blocks * 0.5 ** (k + 2) >= 1]
    ax.plot([x for x, _ in ref_r], [y for _, y in ref_r], color=INK_MUTED,
            linewidth=1.4, linestyle=(0, (4, 2)), zorder=1)
    ax.plot([x for x, _ in ref_l], [y for _, y in ref_l], color=INK_MUTED,
            linewidth=1.4, linestyle=(0, (4, 2)), zorder=1)

    right_items, left_items, absent = [], [], []
    for cat in arms:
        st = data[cat]["structure"]
        xs = [L for L in ls if st["run_hist"].get(L, 0) > 0]
        ys = [st["run_hist"][L] for L in xs]
        ax.plot(xs, ys, color=COLORS[cat], linewidth=2, marker=MARKERS[cat],
                markersize=3.5, zorder=3)
        right_items.append((xs[-1], ys[-1], f"  {PRETTY[cat]}", COLORS[cat]))

        xa = [k for k in ks if st["alt_hist"].get(k, 0) > 0]
        if not xa:
            absent.append(cat)
            continue
        ya = [st["alt_hist"][k] for k in xa]
        ax.plot([-k for k in xa], ya, color=COLORS[cat], linewidth=2,
                marker=MARKERS[cat], markersize=3.5, zorder=3)
        left_items.append((-xa[-1], ya[-1], f"{PRETTY[cat]}  ", COLORS[cat]))

    ax.set_yscale("log")
    ax.set_ylim(0.62, None)
    ax.set_xlim(-(kmax + 3.4), lmax + 4.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{abs(int(v)):g}"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.set_ylabel("number of stretches")
    ax.set_xlabel("stretch length  (cells of equal sign, right;"
                  "  consecutive sign-flipping cells, left)")
    _style(ax, grid_axis="both")

    # Centre divider plus a heading for each half, so the mirrored axis cannot
    # be misread as one signed quantity.
    ax.axvline(0, color=GRID, linewidth=1.4, zorder=2)
    ax.text(-0.4, 1.065, "ALTERNATING STRETCHES", transform=ax.get_xaxis_transform(),
            fontsize=8.5, color=INK, ha="right", va="bottom", fontweight="bold")
    ax.text(-0.4, 1.012, "every cell flips sign -- worst case for a 1-cell shift",
            transform=ax.get_xaxis_transform(), fontsize=7.5, color=INK_MUTED,
            ha="right", va="bottom")
    ax.text(0.4, 1.065, "SAME-SIGN BLOCKS", transform=ax.get_xaxis_transform(),
            fontsize=8.5, color=INK, ha="left", va="bottom", fontweight="bold")
    ax.text(0.4, 1.012, "a shift inside one of these returns the same bit",
            transform=ax.get_xaxis_transform(), fontsize=7.5, color=INK_MUTED,
            ha="left", va="bottom")

    # Which way is better, stated once on each side. Both curves decay outward,
    # so the outer TOP corners are the empty ones -- the end labels own the
    # bottom corners.
    arrow = dict(arrowstyle="-|>", color=INK_MUTED, linewidth=1.1)
    ax.annotate("", xy=(0.015, 0.93), xytext=(0.115, 0.93), xycoords="axes fraction",
                textcoords="axes fraction", arrowprops=arrow)
    ax.text(0.128, 0.93, "more vulnerable", transform=ax.transAxes, fontsize=7.5,
            color=INK_MUTED, ha="left", va="center")
    ax.annotate("", xy=(0.985, 0.93), xytext=(0.885, 0.93), xycoords="axes fraction",
                textcoords="axes fraction", arrowprops=arrow)
    ax.text(0.872, 0.93, "more robust", transform=ax.transAxes, fontsize=7.5,
            color=INK_MUTED, ha="right", va="center")

    for i, cat in enumerate(absent):
        ax.text(0.25, 0.10 + 0.07 * i,
                f"{PRETTY[cat]}: no alternating stretch at any length",
                transform=ax.transAxes, fontsize=8, color=COLORS[cat],
                ha="center", va="bottom", fontweight="bold")

    _titles(fig,
            "One spectrum: the regulariser rotates it -- blocks out, alternation in",
            "under a fair coin both halves are the SAME geometric law (ratio 1/2, dashed); "
            "structure shows up as asymmetry about the centre",
            top=0.84)
    fig.canvas.draw()
    _place_end_labels(ax, right_items)
    _place_end_labels(ax, left_items, dx=-6, ha="right")
    # One law, named once. The end of the block arm is the clear spot: every
    # series has risen above the reference by then, so the space below it is free.
    if ref_r:
        ax.annotate("fair coin", xy=ref_r[-1], xytext=(5, -7), textcoords="offset points",
                    color=INK_MUTED, fontsize=7.5, va="top", ha="left",
                    annotation_clip=False)
    return _save(fig, out, "fig6_sign_run_spectrum", fmts, dpi)


FIGURES = {
    "spectrum": fig_spectrum,
    "blocklen": fig_blocklen,
    "alternating": fig_alternating,
    "perlayer": fig_perlayer,
    "mechanism": fig_mechanism,
    "payoff": fig_payoff,
}


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS,
                    help="paper-runs tree to read (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="figure output directory (default: %(default)s)")
    ap.add_argument("--rt-error", default="1e-05",
                    help="rt_error tag as it appears in the artifact filename")
    ap.add_argument("--only", nargs="+", metavar="CATEGORY",
                    help="restrict to these categories (e.g. cat1_baseline "
                         "cat5_regularizer cat6_reg_recal cat8_ste_inject); "
                         "every axis range re-scales to what is left")
    ap.add_argument("--figures", nargs="+", choices=sorted(FIGURES) + ["all"],
                    default=["all"])
    ap.add_argument("--format", nargs="+", default=["png"],
                    help="output formats (png, pdf, svg)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # headless host
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    bundle = collect(args.runs, args.rt_error, only=args.only)
    if not bundle["categories"]:
        raise SystemExit(f"no complete runs for rt{args.rt_error} under {args.runs}")
    print(f"categories: {', '.join(bundle['categories'])}")
    for cat in bundle["categories"]:
        s = bundle["data"][cat]["structure"]
        print(f"  {cat:22s} snapshot={s['label']:15s} seeds={','.join(bundle['data'][cat]['seeds'])} "
              f"meanRL={s['mean_run_length']:.3f} transition_density={s['transition_density']:.5f}")

    names = sorted(FIGURES) if "all" in args.figures else args.figures
    for name in names:
        for path in FIGURES[name](bundle, args.out, args.format, args.dpi):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
