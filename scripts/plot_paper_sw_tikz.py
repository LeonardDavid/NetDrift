#!/usr/bin/env python3
"""Emit pgfplots/TikZ figure sources for per-loop accuracy of a paper-runs tree.

Reads ``outcome.per_loop_accuracy`` out of every
``metrics_artifacts/<model>__<category>__rt<err>.json`` under a run tree, groups
the runs by ``meta.category``, averages over the runs behind each category, and
writes:

  * one ``<category>.dat`` per category (loop, mean, lo, hi, one column per run)
  * ``accuracy_vs_iterations{,_grid}_body.tex`` -- the figure bodies (one shared
    axis / small multiples); \\input these into a paper
  * ``accuracy_vs_iterations{,_grid}.tex``      -- standalone wrappers to compile
  * ``README.md`` / ``seeds.json``              -- what actually backs each line

Works on two tree shapes:

  paper-sw_*/<training category>/<prot_seed>/<ts>/metrics_artifacts/
      -> one line per training category, averaged over seeds

  plots/<tree>/lay-<layout>/var-<variant>_..._seed<n>/<ts>/metrics_artifacts/
      -> one line per LAYOUT. Each layout holds several training variants, so
         ``--variant`` picks which one (default ``cat8``); a layout that does not
         have the requested variant but has exactly one of its own falls back to
         it (that is how lay-polarity-regularized contributes its ppmreg runs).
         ``--variant all`` averages over every variant and seed.

Both .tex bodies carry ``\\def\\MaxIter{N}`` at the top: change that one number
and recompile to plot fewer inference iterations. No regeneration needed.

Stdlib only -- runs on the Mac mount without torch/numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

# dataviz categorical slots 1-8, light mode (validated: worst adjacent CVD dE 9.1).
PALETTE = [
    ("ndBlue", "2a78d6"),
    ("ndOrange", "eb6834"),
    ("ndAqua", "1baf7a"),
    ("ndYellow", "eda100"),
    ("ndMagenta", "e87ba4"),
    ("ndGreen", "008300"),
    ("ndViolet", "4a3aa7"),
    ("ndRed", "e34948"),
]

# Secondary encoding, so identity never rests on hue alone (three slots sit under
# 3:1 on a light surface, and paper figures get photocopied).
DASHES = [
    "solid",
    "dashed",
    "dotted",
    "densely dashdotted",
    "densely dashed",
    "dashdotdotted",
    "loosely dashed",
    "densely dotted",
]
MARKS = ["*", "square*", "triangle*", "diamond*", "pentagon*", "o", "square", "triangle"]

# Fixed, caller-specified colours. Anything not listed falls back to the PALETTE
# slot for its position.
FIXED_COLORS = {
    # racetrack layouts
    "lay-row": "e87ba4",
    "lay-col": "1baf7a",
    "lay-polarity": "eb6834",
    "lay-polarity-regularized": "eda100",
    "lay-block": "2a78d6",
    # paper-sw training categories
    "cat1_baseline": "008300",
    "cat4_endlen_recal": "e87ba4",
    "cat5_regularizer": "eb6834",
    "cat6_reg_recal": "eda100",
    "cat8_ste_inject": "2a78d6",
    "cat68_reg_ste_recal": "1baf7a",
}

# Legend/plot order. Plain sorting puts cat68 before cat6 ('8' < '_') and would
# scatter the layouts; anything unlisted is appended alphabetically.
ORDER = [
    # paper-sw training categories
    "cat1_baseline",
    "cat2_endlen",
    "cat4_endlen_recal",
    "cat5_regularizer",
    "cat6_reg_recal",
    "cat8_ste_inject",
    "cat68_reg_ste_recal",
    # Racetrack layouts, in the order they should read in the legend:
    # Dense-Row, Dense-Col, Polarity, Polarity-Regularized, Individual.
    "lay-row",
    "lay-col",
    "lay-polarity",
    "lay-polarity-regularized",
    "lay-block",
]

# Human-readable legend entries; anything unlisted falls back to the raw dir name.
PRETTY = {
    "cat1_baseline": "Baseline",
    "cat2_endlen": "Endlen encoder",
    "cat4_endlen_recal": "Endlen + recal",
    "cat5_regularizer": "RL-Reg",
    "cat6_reg_recal": "RL-Reg-Recal",
    "cat8_ste_inject": "STE inject",
    "cat68_reg_ste_recal": "Reg. + STE + recal",
    "lay-row": "Dense-Row",
    "lay-col": "Dense-Col",
    "lay-polarity": "Polarity",
    "lay-polarity-regularized": "Polarity-Regularized",
    "lay-block": "Individual",
}


def tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
             .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#"))


def rt_matches(filename: str, rt_error: str) -> bool:
    """Match the rt token in a metrics filename against a requested rt_error.

    Compares numerically so ``1e-05``, ``1e-5`` and ``0.00001`` all agree.
    """
    m = re.search(r"__rt([0-9.eE+-]+)\.json$", filename)
    if not m:
        return False
    try:
        return float(m.group(1)) == float(rt_error)
    except ValueError:
        return False


def order_key(cat: str) -> tuple[int, str]:
    return (ORDER.index(cat), "") if cat in ORDER else (len(ORDER), cat)


def collect(run_dir: Path, rt_error: str) -> tuple[dict, dict]:
    """(category -> {(variant, seed): per_loop_accuracy}, category -> [baselines]).

    ``variant`` is the ``var-<x>`` token of the subcategory, or None on trees that
    do not carry one (the paper-sw shape).
    """
    out: dict[str, dict[tuple[str | None, int], list[float]]] = {}
    bases: dict[str, dict[tuple[str | None, int], dict]] = {}
    for path in sorted(run_dir.glob("*/*/*/metrics_artifacts/*.json")):
        if not rt_matches(path.name, rt_error):
            continue
        blob = json.loads(path.read_text())
        meta, outcome = blob.get("meta", {}), blob.get("outcome", {})
        series = outcome.get("per_loop_accuracy")
        seed = meta.get("seed")
        if not series or seed is None:
            continue
        category = meta.get("category") or path.parts[-4]
        sub = meta.get("subcategory") or path.parts[-3]
        m = re.search(r"var-([A-Za-z0-9]+)", sub)
        key = (m.group(1) if m else None, int(seed))
        # The same run can appear under several timestamps; last one wins.
        out.setdefault(category, {})[key] = [float(v) for v in series]
        bases.setdefault(category, {})[key] = outcome.get("baselines") or {}
    return out, bases


def select_variant(runs: dict[tuple[str | None, int], list[float]], want: str
                   ) -> dict[tuple[str | None, int], list[float]]:
    """Keep the runs of one training variant, with a single-variant fallback.

    A layout that has no run of the requested variant but exactly one variant of
    its own keeps that one -- this is what lets lay-polarity-regularized (ppmreg
    only) sit beside the other layouts.
    """
    variants = {v for v, _ in runs}
    if want == "all" or variants == {None}:
        return runs
    picked = {k: v for k, v in runs.items() if k[0] == want}
    if picked:
        return picked
    return runs if len(variants) == 1 else {}


BLOCK_CAT = "lay-block"


def block_accuracy(bases: dict, spec: str, plotted: set | None = None) -> float:
    """Clean accuracy to draw the fault-immune BLOCK layout at.

    ``auto``     -> the plain-training clean (layout-independent, unambiguous)
    ``variant``  -> the best clean among the arms actually plotted
    ``<number>`` -> that accuracy

    BLOCK is a pure storage layout and is immune at these error rates, so its
    curve is flat at the clean accuracy of the checkpoint it maps. Plain (`base`)
    training is layout-independent -- every layout in a tree reports the same
    clean for it -- which makes it the unambiguous reference; fault-aware training
    buys nothing once the layout is already immune.
    """
    if spec == "variant":
        # Internally consistent with the arms actually drawn: the best clean among
        # them. Use this when the point is "the same checkpoint, stored immune".
        sel = [b["clean"] for c, byrun in bases.items() if plotted is None or c in plotted
               for b in byrun.values() if b.get("clean") is not None]
        if not sel:
            raise SystemExit("--block variant: no clean baseline among the plotted arms")
        return max(sel)
    if spec != "auto":
        return float(spec)
    base = [b["clean"] for byrun in bases.values() for (v, _), b in byrun.items()
            if v == "base" and b.get("clean") is not None]
    if base:
        return max(base)
    any_clean = [b["clean"] for byrun in bases.values() for b in byrun.values()
                 if b.get("clean") is not None]
    if not any_clean:
        raise SystemExit("--block auto: no clean baseline found to read the BLOCK level from")
    return max(any_clean)


def col_name(key: tuple[str | None, int], multi: bool) -> str:
    variant, seed = key
    return f"{variant}_{seed}" if multi and variant else f"s{seed}"


def write_dat(out_dir: Path, category: str,
              runs: dict[tuple[str | None, int], list[float]],
              per_run_columns: bool = True) -> tuple[list[str], int]:
    """Write <category>.dat. ``per_run_columns=False`` emits mean/lo/hi only, which
    is what the synthetic BLOCK arm needs -- it has no runs to show a spread for."""
    keys = sorted(runs, key=lambda k: (k[0] or "", k[1]))
    multi = len({v for v, _ in keys}) > 1
    cols = [col_name(k, multi) for k in keys] if per_run_columns else []
    n = min(len(runs[k]) for k in keys)
    lines = [" ".join(["loop", "mean", "lo", "hi", *cols])]
    for i in range(n):
        vals = [runs[k][i] for k in keys]
        # Few runs per line, so min/max IS the honest interval; a std would imply
        # a sampling distribution this many runs cannot support.
        row = [i + 1, sum(vals) / len(vals), min(vals), max(vals), *(vals if cols else [])]
        lines.append(" ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row))
    (out_dir / f"{category}.dat").write_text("\n".join(lines) + "\n")
    return cols, n


BODY_NOTE = r"""% Generated by scripts/plot_paper_sw_tikz.py -- regenerating overwrites this file.
% Figure body only (no \documentclass). Two ways to use it:
%   1. compile the sibling standalone wrapper, or
%   2. \input{} this file inside a figure environment in the paper -- the paper
%      preamble then needs: \usepackage{pgfplots} \pgfplotsset{compat=1.18}
%      \usepgfplotslibrary{fillbetween}   % and groupplots for the grid variant
%      and, if the .dat files are not beside the main .tex,
%      \pgfplotsset{table/search path={<dir holding the .dat files>}}
"""

WRAPPER_NOTE = r"""% Generated by scripts/plot_paper_sw_tikz.py -- regenerating overwrites this file.
% Standalone wrapper; all the figure content (and the \MaxIter knob) lives in the
% _body.tex file this inputs.
"""

WRAPPER_TEMPLATE = r"""{note}\documentclass[tikz, border=2pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\usepgfplotslibrary{{{libs}}}
\begin{{document}}
\input{{{body}}}
\end{{document}}
"""


def color_macro(cat: str, used: set[str]) -> str:
    """`lay-polarity-regularized` -> `ndLayPolarityRegularized`.

    Letters and digits only, so the name is safe everywhere xcolor accepts one.
    """
    # `cat6_reg_recal` -> `ndCat6` (the number already identifies it);
    # anything else keeps its full name, e.g. `lay-polarity-regularized`.
    m = re.match(r"(cat\d+)", cat)
    parts = [m.group(1)] if m else [p for p in re.split(r"[^A-Za-z0-9]+", cat) if p]
    name = "nd" + "".join(p[:1].upper() + p[1:] for p in parts)
    base, i = name, 2
    while name in used:  # two categories can sanitise to the same name
        name, i = f"{base}{i}", i + 1
    used.add(name)
    return name


def assign_colors(cats: list[str]) -> dict[str, tuple[str, str]]:
    """category -> (colour macro name, hex). One macro per series, so changing a
    series' colour is a single edit at the top of the body file."""
    used: set[str] = set()
    return {cat: (color_macro(cat, used),
                  FIXED_COLORS.get(cat, PALETTE[i % len(PALETTE)][1]))
            for i, cat in enumerate(cats)}


def color_defs(cats: list[str], colors: dict[str, tuple[str, str]]) -> str:
    width = max(len(colors[c][0]) for c in cats)
    return "\n".join(
        f"\\definecolor{{{colors[c][0]}}}{{HTML}}{{{colors[c][1].upper()}}}"
        f"{' ' * (width - len(colors[c][0]))}  % {PRETTY.get(c, c)}"
        for c in cats)


def build_single(cats: list[str], cols_by_cat: dict[str, list[str]], max_iter: int,
                 ymin: float, ymax: float, show_seeds: int, show_band: int,
                 colors: dict[str, tuple[str, str]], ytick_step: float, yminor: int,
                 legend_cols: int) -> str:
    body = []
    for idx, cat in enumerate(cats):
        color = colors[cat][0]
        dash = DASHES[idx % len(DASHES)]
        mark = MARKS[idx % len(MARKS)]
        body.append(f"      %% ---- {cat} ----")
        # Run envelope (min-max). forget plot on every helper, or the legend multiplies.
        body.append(r"      \ifnum\ShowBand=1")
        body.append(f"        \\addplot[draw=none, forget plot, name path=lo{idx}] table[x=loop, y=lo] {{{cat}.dat}};")
        body.append(f"        \\addplot[draw=none, forget plot, name path=hi{idx}] table[x=loop, y=hi] {{{cat}.dat}};")
        body.append(f"        \\addplot[{color}, fill opacity=0.12, forget plot] fill between[of=lo{idx} and hi{idx}];")
        body.append(r"      \fi")
        # Individual runs. With only a handful these ARE the interval, so the band
        # above is the same object drawn as ink.
        body.append(r"      \ifnum\ShowSeeds=1")
        for col in cols_by_cat[cat]:
            body.append(
                f"        \\addplot[{color}, opacity=0.35, line width=0.35pt, forget plot]"
                f" table[x=loop, y={col}] {{{cat}.dat}};"
            )
        body.append(r"      \fi")
        # Mean last, so it sits on top of its own runs.
        body.append(f"      \\pgfmathtruncatemacro{{\\MarkPhase}}{{1 + mod({idx}, \\MarkRepeat)}}")
        body.append(
            f"      \\addplot[{color}, {dash}, line width=1.1pt, mark={mark}, mark size=1.6pt,"
            f" mark repeat=\\MarkRepeat, mark phase=\\MarkPhase, mark options={{solid, fill={color}}}]"
            f" table[x=loop, y=mean] {{{cat}.dat}};"
        )
        body.append(f"      \\addlegendentry{{{tex_escape(PRETTY.get(cat, cat))}}}")
    return SINGLE_TEMPLATE.format(
        note=BODY_NOTE, colors=color_defs(cats, colors), max_iter=max_iter,
        ymin=ymin, ymax=ymax, show_seeds=show_seeds, show_band=show_band,
        ytick_step=ytick_step, yminor=yminor, legend_cols=legend_cols,
        plots="\n".join(body),
    )


SINGLE_TEMPLATE = r"""{note}
%% ======================================================================
%%  THE KNOB: how many inference iterations to plot.
%%  The .dat files hold all available loops; this just crops the x-axis,
%%  so changing it needs a recompile only -- never a regeneration.
%% ======================================================================
\def\MaxIter{{{max_iter}}}

%% Figure size.
\def\FigWidth{{10cm}}
\def\FigHeight{{6cm}}

%% Accuracy window and horizontal grid: a labelled line every \YTickStep points,
%% plus \YMinorNum unlabelled line(s) between them (1 -> a line every 5 pp).
\def\YMin{{{ymin:g}}}
\def\YMax{{{ymax:g}}}
\def\YTickStep{{{ytick_step:g}}}
\def\YMinorNum{{{yminor:d}}}

%% Legend entries per row.
\def\LegendCols{{{legend_cols:d}}}

%% Presentation toggles for the run-to-run spread (1 = on, 0 = off).
%% Plain \def, not \newif, so this body survives being \input twice.
\def\ShowSeeds{{{show_seeds}}}   % thin per-run curves
\def\ShowBand{{{show_band}}}    % shaded min--max envelope
%% ======================================================================

\pgfmathtruncatemacro{{\MarkRepeat}}{{max(1, round(\MaxIter/10))}}

%% One colour per series, named after it -- recolour a series here, once.
{colors}

\begin{{tikzpicture}}
  \begin{{axis}}[
      width=\FigWidth, height=\FigHeight,
      xlabel={{Inference iteration}},
      ylabel={{Top-1 accuracy (\%)}},
      xmin=1, xmax=\MaxIter,
      ymin=\YMin, ymax=\YMax,
      enlarge x limits=false,
      enlarge y limits=false,
      clip=true,
      ytick distance=\YTickStep,
      minor y tick num=\YMinorNum,
      grid=both,
      major grid style={{gray!22, line width=0.3pt}},
      minor grid style={{gray!12, line width=0.25pt}},
      tick align=outside,
      tick style={{gray!55, line width=0.4pt}},
      axis line style={{gray!55, line width=0.4pt}},
      label style={{font=\small}},
      tick label style={{font=\footnotesize}},
      legend style={{
        font=\footnotesize,
        at={{(0.5,-0.22)}}, anchor=north,
        legend columns=\LegendCols,
        draw=gray!40, fill=white, fill opacity=0.9, text opacity=1,
        /tikz/every even column/.append style={{column sep=6pt}},
      }},
      legend cell align=left,
    ]

{plots}

  \end{{axis}}
\end{{tikzpicture}}
"""


def build_grid(cats: list[str], cols_by_cat: dict[str, list[str]], max_iter: int,
               ymin: float, ymax: float, show_seeds: int,
               colors: dict[str, tuple[str, str]], ytick_step: float, yminor: int,
               cols: int = 3) -> str:
    rows = (len(cats) + cols - 1) // cols
    body = []
    for idx, cat in enumerate(cats):
        color = colors[cat][0]
        body.append(f"    \\nextgroupplot[title={{{tex_escape(PRETTY.get(cat, cat))}}}]")
        body.append(f"      \\addplot[draw=none, forget plot, name path=glo{idx}] table[x=loop, y=lo] {{{cat}.dat}};")
        body.append(f"      \\addplot[draw=none, forget plot, name path=ghi{idx}] table[x=loop, y=hi] {{{cat}.dat}};")
        body.append(f"      \\addplot[{color}, fill opacity=0.18, forget plot] fill between[of=glo{idx} and ghi{idx}];")
        body.append(r"      \ifnum\ShowSeeds=1")
        for col in cols_by_cat[cat]:
            body.append(
                f"        \\addplot[{color}, opacity=0.40, line width=0.35pt, forget plot]"
                f" table[x=loop, y={col}] {{{cat}.dat}};"
            )
        body.append(r"      \fi")
        body.append(f"      \\addplot[{color}, line width=1.1pt, forget plot] table[x=loop, y=mean] {{{cat}.dat}};")
    return GRID_TEMPLATE.format(
        note=BODY_NOTE, colors=color_defs(cats, colors), max_iter=max_iter,
        rows=rows, cols=cols, ymin=ymin, ymax=ymax, show_seeds=show_seeds,
        ytick_step=ytick_step, yminor=yminor,
        plots="\n".join(body),
    )


GRID_TEMPLATE = r"""{note}
%% ======================================================================
%%  THE KNOB: how many inference iterations to plot (crops the x-axis).
%% ======================================================================
\def\MaxIter{{{max_iter}}}
\def\YMin{{{ymin:g}}}
\def\YMax{{{ymax:g}}}
\def\ShowSeeds{{{show_seeds}}}   % thin per-run curves under each panel's band
\def\YTickStep{{{ytick_step:g}}}
\def\YMinorNum{{{yminor:d}}}

%% Size of ONE panel (the figure is 3 panels wide).
\def\PanelWidth{{5.4cm}}
\def\PanelHeight{{4.2cm}}
%% ======================================================================

%% One colour per series, named after it -- recolour a series here, once.
{colors}

\begin{{tikzpicture}}
  \begin{{groupplot}}[
      group style={{
        group size={cols} by {rows},
        horizontal sep=1.1cm, vertical sep=1.2cm,
        xlabels at=edge bottom, ylabels at=edge left,
      }},
      width=\PanelWidth, height=\PanelHeight,
      xlabel={{Inference iteration}},
      ylabel={{Top-1 accuracy (\%)}},
      xmin=1, xmax=\MaxIter,
      ymin=\YMin, ymax=\YMax,
      enlarge x limits=false, enlarge y limits=false, clip=true,
      ytick distance=\YTickStep,
      minor y tick num=\YMinorNum,
      grid=both,
      major grid style={{gray!22, line width=0.3pt}},
      minor grid style={{gray!12, line width=0.25pt}},
      tick align=outside,
      tick style={{gray!55, line width=0.4pt}},
      axis line style={{gray!55, line width=0.4pt}},
      title style={{font=\small}},
      label style={{font=\small}},
      tick label style={{font=\footnotesize}},
    ]

{plots}

  \end{{groupplot}}
\end{{tikzpicture}}
"""


def write_readme(out_dir: Path, cats: list[str], picked: dict, bases: dict, lengths: dict[str, int],
                 rt_error: str, run_dir: Path, max_iter: int, variant: str,
                 show_seeds: int, show_band: int, ymin: float, ymax: float,
                 block_level: float | None = None, block_spec: str = "auto") -> None:
    """Record what actually backs the figure -- run coverage and collapsed arms."""
    starts = {c: sum(v[0] for v in picked[c].values()) / len(picked[c]) for c in cats}

    def mean_of(c, key):
        vals = [b[key] for b in bases.get(c, {}).values() if b.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    # A pre-fault baseline far under `clean` means the configuration was already
    # broken before any fault landed. A healthy baseline with a low iteration 1
    # is the opposite: the layout really does die that fast at this error rate.
    broken, fast = [], []
    for c in (c for c in cats if c != BLOCK_CAT):
        clean = mean_of(c, "clean")
        pre = max((v for k in ("endlen_recal", "endlen") if (v := mean_of(c, k)) is not None),
                  default=None)
        if clean is not None and pre is not None and pre < clean - 25:
            broken.append((c, pre, clean))
        elif clean is not None and starts[c] < clean - 25:
            fast.append((c, starts[c], clean))

    rows = []
    for c in cats:
        variants = sorted({v for v, _ in picked[c] if v})
        seeds = sorted({s for _, s in picked[c]})
        rows.append(
            f"| `{c}` | {PRETTY.get(c, c)} | {', '.join(variants) or '-'} "
            f"| {', '.join(str(s) for s in seeds)} | {len(picked[c])} | {lengths[c]} | {starts[c]:.2f} |")

    counts = {len(picked[c]) for c in cats if c != BLOCK_CAT}
    lines = [
        f"# Per-loop accuracy - `{run_dir.name}` @ rt\\_error = {rt_error}",
        "",
        f"Variant selection: **`--variant {variant}`**. Generated by",
        "`scripts/plot_paper_sw_tikz.py`; **regenerating overwrites every file here.**",
        "Source: `outcome.per_loop_accuracy` in each run's",
        "`metrics_artifacts/*__rt<err>.json`.",
        "",
        "## Plotting fewer iterations",
        "",
        "Edit one line in `accuracy_vs_iterations_body.tex` and recompile - the `.dat`",
        f"files always hold all {min(lengths.values())} loops, so no regeneration is needed:",
        "",
        "```latex",
        f"\\def\\MaxIter{{{max_iter}}}   % <- iterations to show",
        f"\\def\\YMin{{{ymin:g}}}  \\def\\YMax{{{ymax:g}}}",
        f"\\def\\ShowSeeds{{{show_seeds}}}  \\def\\ShowBand{{{show_band}}}",
        "```",
        "",
        "`ShowSeeds` draws one thin curve per run; `ShowBand` shades the min-max",
        "envelope. Both are plain `\\def` toggles (not `\\newif`), so a body can be",
        "`\\input` more than once in the same document.",
        "",
        "## What backs each line",
        "",
        "| Category | Legend | Variant(s) | Seeds | Runs | Loops | Mean acc. @ iter 1 |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]
    if block_level is not None:
        lines += [
            f"**BLOCK is synthetic.** `{BLOCK_CAT}` has no runs on disk: BLOCK is",
            "fault-immune at these error rates, so its curve is drawn flat at the clean",
            f"accuracy **{block_level:.2f}%** for every iteration (`--block {block_spec}`).",
            "",
            "`--block auto` uses the plain-training (`var-base`) clean accuracy: it is",
            "identical across the layouts in this tree, and fault-aware training buys",
            "nothing once the layout is already immune. `--block variant` instead uses the",
            "best clean among the arms plotted, which is the number to quote if the claim",
            "is \"the same checkpoint, stored immune\" - on VGG the two differ by ~3pp",
            "because the fault-aware checkpoints are simply better networks.",
            "`--block <accuracy>` sets it outright.",
            "",
        ]

    fallback = [c for c in cats if c != BLOCK_CAT and variant != "all" and {v for v, _ in picked[c] if v}
                and variant not in {v for v, _ in picked[c]}]
    if fallback:
        lines += [
            "**Variant fallback:** " + "; ".join(
                f"`{c}` has no `{variant}` run, so its only variant "
                f"(`{'/'.join(sorted({v for v, _ in picked[c] if v}))}`) is used instead"
                for c in fallback) + ".",
            "",
        ]
    if len(counts) == 1:
        n = counts.pop()
        lines += [
            f"**Interval = min/max over the {n} runs behind each line.** With n={n} the",
            "envelope *is* those runs, so a standard deviation would imply a sampling",
            "distribution this many runs does not support.",
            "",
        ]
    if broken:
        lines += [
            "**Broken configuration(s):** " + "; ".join(
                f"`{c}` has a pre-fault baseline of {pre:.1f}% against a clean {clean:.1f}%"
                for c, pre, clean in broken) + ".",
            "These arms are damaged before any fault is injected, so their curves say",
            "nothing about fault tolerance. Drop them with `--only` for a zoomed figure.",
            "",
        ]
    if fast:
        lines += [
            "**Already collapsed by iteration 1:** " + "; ".join(
                f"`{c}` starts at {s_:.1f}% from a healthy clean {clean:.1f}%"
                for c, s_, clean in fast) + ".",
            "The checkpoint is fine - at this error rate the layout is destroyed inside",
            "the first inference pass. That is a result, not an artifact, but it does",
            "mean the curve carries no usable dynamic range here.",
            "",
        ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path,
                    default=Path("runs/paper-runs/paper-sw_resnet18_imagenette"),
                    help="tree holding <category>/<subcategory>/<timestamp>/metrics_artifacts")
    ap.add_argument("--rt-error", default="1e-05", help="which rt_error point to plot")
    ap.add_argument("--variant", default="cat8",
                    help="training variant to plot on trees that have them: base|cat6|cat8|all "
                         "(default: cat8). Ignored on trees without var- tokens. A category with "
                         "only one variant of its own falls back to it.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: docs/figures/<run-dir name>_rt<rt_error>[_var-<variant>]")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="initial \\MaxIter (default: all available loops)")
    ap.add_argument("--ymin", type=float, default=0.0, help="initial \\YMin (default: 0)")
    ap.add_argument("--ymax", type=float, default=None,
                    help="initial \\YMax (default: rounded up from the data)")
    ap.add_argument("--ytick-step", type=float, default=10.0,
                    help="initial \\YTickStep: labelled horizontal grid line every N pp (default: 10)")
    ap.add_argument("--yminor", type=int, default=1,
                    help="initial \\YMinorNum: unlabelled lines between them (default: 1, i.e. every 5 pp)")
    ap.add_argument("--legend-columns", type=int, default=3,
                    help="initial \\LegendCols: legend entries per row (default: 3)")
    ap.add_argument("--show-seeds", type=int, default=0, choices=(0, 1),
                    help="initial \\ShowSeeds: thin per-run curves (default: 0)")
    ap.add_argument("--show-band", type=int, default=1, choices=(0, 1),
                    help="initial \\ShowBand: shaded min--max envelope (default: 1)")
    ap.add_argument("--block", default=None, metavar="auto|ACC",
                    help="add a flat, fault-immune BLOCK layout line. 'auto' uses the plain "
                         "(var-base) clean accuracy; 'variant' uses the best clean among the "
                         "arms plotted; a number sets it explicitly. Off by default.")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these category names")
    ap.add_argument("--prune", action="store_true",
                    help="delete .dat files in the output dir that this run did not write "
                         "(left over from an earlier, wider --only)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    raw, bases = collect(run_dir, args.rt_error)
    if not raw:
        raise SystemExit(f"no per_loop_accuracy found for rt_error={args.rt_error} under {run_dir}")

    picked = {c: sel for c, runs in raw.items() if (sel := select_variant(runs, args.variant))}
    if not picked:
        raise SystemExit(f"--variant {args.variant} matched no runs under {run_dir}")
    block_level = None
    if args.block is not None:
        block_level = block_accuracy(bases, args.block, plotted=set(picked))
        n_loops = min(len(v) for byrun in picked.values() for v in byrun.values())
        # BLOCK is fault-immune: one synthetic run, flat at the clean accuracy.
        picked[BLOCK_CAT] = {(None, 0): [block_level] * n_loops}
        bases[BLOCK_CAT] = {(None, 0): {"clean": block_level}}

    cats = sorted(picked, key=order_key)
    if args.only is not None:
        cats = [c for c in cats if c in args.only]
    if not cats:
        raise SystemExit("--only matched no categories")

    has_variants = any(v for c in cats for v, _ in picked[c])
    suffix = f"_var-{args.variant}" if has_variants else ""
    out_dir = args.out_dir or Path("docs/figures") / f"{run_dir.name}_rt{args.rt_error}{suffix}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cols_by_cat: dict[str, list[str]] = {}
    lengths: dict[str, int] = {}
    for cat in cats:
        cols_by_cat[cat], lengths[cat] = write_dat(out_dir, cat, picked[cat],
                                                   per_run_columns=cat != BLOCK_CAT)

    colors = assign_colors(cats)
    max_iter = args.max_iter or min(lengths.values())
    show_seeds, show_band = args.show_seeds, args.show_band
    peak = max(max(v) for c in cats for v in picked[c].values())
    ymax = args.ymax if args.ymax is not None else min(100.0, 5 * math.ceil(peak / 5))

    for stem, body, libs in (
        ("accuracy_vs_iterations",
         build_single(cats, cols_by_cat, max_iter, args.ymin, ymax, show_seeds, show_band,
                      colors, args.ytick_step, args.yminor, args.legend_columns),
         "fillbetween"),
        ("accuracy_vs_iterations_grid",
         build_grid(cats, cols_by_cat, max_iter, args.ymin, ymax, show_seeds, colors,
                    args.ytick_step, args.yminor),
         "fillbetween, groupplots"),
    ):
        (out_dir / f"{stem}_body.tex").write_text(body)
        (out_dir / f"{stem}.tex").write_text(
            WRAPPER_TEMPLATE.format(note=WRAPPER_NOTE, libs=libs, body=f"{stem}_body.tex"))
    write_readme(out_dir, cats, picked, bases, lengths, args.rt_error, run_dir, max_iter,
                 args.variant, show_seeds, show_band, args.ymin, ymax, block_level,
                 args.block or "auto")
    (out_dir / "seeds.json").write_text(json.dumps({
        "run_dir": str(run_dir), "rt_error": args.rt_error, "variant": args.variant,
        "categories": {c: {"runs": [f"{v or '-'}:{s}" for v, s in sorted(picked[c], key=lambda k: (k[0] or "", k[1]))],
                           "columns": cols_by_cat[c], "loops": lengths[c]} for c in cats},
    }, indent=2) + "\n")

    # A narrower --only leaves .dat files from a previous run behind. The .tex never
    # references them, but they make the directory look like it holds more series
    # than the figure draws, so say so (and remove them only when asked).
    orphans = sorted(f for f in out_dir.glob("*.dat") if f.stem not in cats)
    for f in orphans:
        if args.prune:
            f.unlink()
    print(f"wrote {out_dir}")
    for cat in cats:
        print(f"  {cat:26s} runs={len(cols_by_cat[cat])} cols={cols_by_cat[cat]}")
    print(f"  \\MaxIter={max_iter}  \\YMax={ymax:g}  ShowSeeds={show_seeds} ShowBand={show_band}")
    if orphans:
        verb = "removed" if args.prune else "STALE (not in this figure; --prune removes them)"
        print(f"  {verb}: {', '.join(f.name for f in orphans)}")


if __name__ == "__main__":
    main()
