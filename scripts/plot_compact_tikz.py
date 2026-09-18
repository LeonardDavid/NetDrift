#!/usr/bin/env python3
"""Restyle a generated figure directory into a compact, space-efficient figure.

Reads a directory already produced by ``scripts/plot_paper_sw_tikz.py`` -- its
``.dat`` files plus ``accuracy_vs_iterations_body.tex`` -- and writes a second,
tighter body beside it:

    compact_body.tex   the figure body (\\input this into a paper)
    compact.tex        standalone wrapper to compile it

Nothing in the source directory is modified, and no run tree is re-read, so the
two figures always plot identical numbers.

Series order, colour macros and hexes, legend labels, dash patterns and marker
shapes are parsed OUT OF the source body rather than recomputed, so hand-edits
there (a recolour, a renamed legend entry) carry over instead of being silently
reverted. Only when the source omits something does it fall back to the tables
in plot_paper_sw_tikz.py.

Where the space comes from, with the legend still outside the plot:

  * ``scale only axis`` -- \\FigWidth/\\FigHeight describe the PLOT BOX, not the
    box plus its labels, so the declared size is the real footprint
  * a frameless legend, packed into the most columns that still fit
  * ``x/ylabel near ticks`` -- drops pgfplots' reserved label gap
  * smaller tick/label fonts and sparser ticks

Stdlib only -- runs on the Mac mount without torch/numpy.
"""

from __future__ import annotations

import argparse
import math
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_paper_sw_tikz import DASHES, MARKS, PRETTY, color_macro  # noqa: E402

SOURCE_BODY = "accuracy_vs_iterations_body.tex"

# Panel titles derived from the figure-directory name. `fl` protects the first and
# last layer (conv1 + classifier); `nop` protects nothing -- verified against
# meta.protection in the runs, not guessed from the tag.
MODEL_TITLES = {
    "r18-fl": "ResNet-18 (first/last protected)",
    "r18-nop": "ResNet-18 (unprotected)",
    "vgg-fl": "VGG-7 (first/last protected)",
    "vgg-nop": "VGG-7 (unprotected)",
}

# rows x cols, as the user writes it.
GRID_LAYOUTS = {"1x2": (1, 2), "2x1": (2, 1), "2x2": (2, 2), "1x4": (1, 4), "4x1": (4, 1)}

# Legend text metrics, for deciding how many columns fit. \footnotesize is ~9pt,
# so ~4.5pt per character, plus the line sample and its gap before the label.
LEGEND_CHAR_PT = 4.5
LEGEND_SAMPLE_PT = 25.0

PANEL_DEFAULT_WIDTH = "5.2cm"   # a figure standing on its own
GRID_PANEL_WIDTH = "5.2cm"      # every panel of a --grid, whatever the layout


def title_macro(i: int) -> str:
    r"""Panel-title macro name for cell `i`.

    TeX control-sequence names are LETTERS ONLY: `\def\Title1{..}` does not define
    `\Title1`, it defines `\Title` taking a delimited `1`, and `\Title2` then dies
    with "Use of \Title doesn't match its definition". So spell the index.
    """
    letters = string.ascii_uppercase
    return "Title" + (letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26])


def dir_tag(src: Path) -> str:
    """`r18-fl_rt4.55e-05_var-cat8` -> `r18-fl`."""
    return src.name.split("_rt")[0]


def panel_title(src: Path) -> str:
    tag = dir_tag(src)
    return MODEL_TITLES.get(tag, tag.replace("_", " "))


class Series:
    """One plotted line, as recovered from the source body."""

    def __init__(self, cat: str, color: str, hexv: str, label: str, dash: str, mark: str):
        self.cat, self.color, self.hex = cat, color, hexv
        self.label, self.dash, self.mark = label, dash, mark
        self.columns: list[str] = []


def read_macro(body: str, name: str, default: str | None = None) -> str | None:
    m = re.search(r"\\def\\" + name + r"\{([^}]*)\}", body)
    return m.group(1) if m else default


def parse_source(src: Path) -> tuple[list[Series], dict[str, str]]:
    """Recover the series (in plot order) and the knob values from the source body."""
    body = (src / SOURCE_BODY).read_text()

    colors = dict(re.findall(r"\\definecolor\{(\w+)\}\{HTML\}\{(\w+)\}", body))
    labels = re.findall(r"\\addlegendentry\{(.+?)\}", body)

    # The mean \addplot of each series carries its colour, dash and mark.
    means = re.findall(
        r"\\addplot\[(nd\w+), ([^,]+), line width=[\d.]+pt, mark=([^,]+),[^]]*\]"
        r" table\[x=loop, y=mean\] \{([\w.-]+)\.dat\}",
        body,
    )
    if not means:
        raise SystemExit(f"{src / SOURCE_BODY}: found no mean plots to restyle")
    if len(labels) != len(means):
        raise SystemExit(f"{src / SOURCE_BODY}: {len(means)} series but {len(labels)} legend entries")

    series = []
    for i, ((color, dash, mark, cat), label) in enumerate(zip(means, labels)):
        hexv = colors.get(color)
        if hexv is None:  # source never defined it -- fall back to a palette slot
            color, hexv = color_macro(cat, set()), "2A78D6"
        s = Series(cat, color, hexv, label or PRETTY.get(cat, cat),
                   dash or DASHES[i % len(DASHES)], mark or MARKS[i % len(MARKS)])
        dat = src / f"{cat}.dat"
        if not dat.exists():
            raise SystemExit(f"missing data file {dat}")
        header = dat.read_text().split("\n", 1)[0].split()
        s.columns = [c for c in header if c not in ("loop", "mean", "lo", "hi")]
        series.append(s)

    knobs = {k: read_macro(body, k) for k in
             ("MaxIter", "YMin", "YMax", "ShowSeeds", "ShowBand", "YTickStep", "YMinorNum")}
    return series, knobs


BODY_NOTE = r"""% Generated by scripts/plot_compact_tikz.py -- regenerating overwrites this file.
% A compact restyle of accuracy_vs_iterations_body.tex in this same directory;
% both read the same .dat files, so they always plot identical numbers.
%
% \input this inside a figure environment. The preamble needs:
%     \usepackage{pgfplots} \pgfplotsset{compat=1.18}
%     \usepgfplotslibrary{fillbetween}
% and, if the .dat files are not beside the main .tex,
%     \pgfplotsset{table/search path={<dir holding the .dat files>}}
%
% \FigWidth/\FigHeight are the PLOT BOX (scale only axis), not the box plus its
% labels -- the total float is that plus roughly 1.1cm for ticks/labels/legend.
"""

WRAPPER = r"""% Generated by scripts/plot_compact_tikz.py -- regenerating overwrites this file.
\documentclass[tikz, border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{fillbetween}
\begin{document}
\input{compact_body.tex}
\end{document}
"""

TEMPLATE = r"""{note}
%% ======================================================================
%%  KNOBS -- all compile-time; none needs a regeneration.
%% ======================================================================
\def\MaxIter{{{max_iter}}}        % inference iterations to plot

%% Size of the PLOT BOX itself (scale only axis).
\def\FigWidth{{{fig_width}}}
\def\FigHeight{{{fig_height}}}

%% Axis window and ticks.
\def\YMin{{{ymin}}}
\def\YMax{{{ymax}}}
{ytick_block}\def\YMinorNum{{{yminor}}}       % unlabelled rules between them (0 = none)
\def\XTickStep{{{xtick}}}       % labelled vertical rule every N iterations

%% Legend: kept OUTSIDE the axis. Columns are chosen so the widest row fits
%% inside \FigWidth -- raise it only if you also shorten the labels.
\def\LegendCols{{{legend_cols}}}
\def\LegendPos{{{legend_pos}}}     % axis-height fractions below the axis. This is
                         % the one value that may need a nudge after the
                         % first compile -- lower it if the legend and the
                         % x label collide.

%% Axis label text (shorten to buy a few mm).
\def\XLabel{{Inference iteration}}
\def\YLabel{{Top-1 accuracy (\%)}}

%% Run-to-run spread (1 = on, 0 = off).
\def\ShowSeeds{{{show_seeds}}}
\def\ShowBand{{{show_band}}}
%% ======================================================================

\pgfmathtruncatemacro{{\MarkRepeat}}{{max(1, round(\MaxIter/6))}}

%% One colour per series, named after it -- recolour a series here, once.
{colors}

\begin{{tikzpicture}}
  \begin{{axis}}[
      scale only axis,
      width=\FigWidth, height=\FigHeight,
      xlabel={{\XLabel}}, ylabel={{\YLabel}},
      xlabel near ticks, ylabel near ticks,
      xmin=1, xmax=\MaxIter,
      ymin=\YMin, ymax=\YMax,
      enlarge x limits=false, enlarge y limits=false, clip=true,
      xtick distance=\XTickStep,
      {ytick_key},
      minor y tick num=\YMinorNum,
      grid=both,
      major grid style={{gray!22, line width=0.3pt}},
      minor grid style={{gray!12, line width=0.25pt}},
      tick align=outside,
      tick style={{gray!55, line width=0.4pt}},
      axis line style={{gray!55, line width=0.4pt}},
      label style={{font=\footnotesize}},
      tick label style={{font=\scriptsize}},
      legend style={{
        font=\footnotesize,
        at={{(0.5,\LegendPos)}}, anchor=north,
        legend columns=\LegendCols,
        draw=none, fill=none,
        inner sep=1pt,
        /tikz/every even column/.append style={{column sep=5pt}},
      }},
      legend cell align=left,
    ]

{plots}

  \end{{axis}}
\end{{tikzpicture}}
"""


def width_pt(spec: str) -> float | None:
    """'8cm' -> 227.6pt. None when the width is a LaTeX length we cannot measure."""
    m = re.fullmatch(r"([\d.]+)\s*(cm|mm|in|pt)", spec.strip())
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2)
    return v * {"cm": 28.45, "mm": 2.845, "in": 72.27, "pt": 1.0}[unit]


def fit_legend_columns(labels: list[str], fig_width: str) -> int:
    r"""Widest column count whose widest row still fits inside the plot box.

    An overflowing legend is neither compact nor readable, so 'all on one row'
    is only the default when the labels are actually short enough. Estimated at
    \scriptsize (~7pt, ~3.5pt per character) plus the line sample and column gap.
    """
    avail = width_pt(fig_width)
    if avail is None:
        return min(3, len(labels))
    entries = [len(l) * LEGEND_CHAR_PT + LEGEND_SAMPLE_PT for l in labels]
    for ncols in range(len(labels), 1, -1):
        rows = [entries[i:i + ncols] for i in range(0, len(entries), ncols)]
        if max(sum(r) + 8.0 * (len(r) - 1) for r in rows) <= avail * 0.95:
            return ncols
    return 1


# Ticks + x label below the axis need a roughly constant ~1.15cm regardless of how
# tall the axis is, but \LegendPos is measured in axis-height fractions -- so the
# fraction has to grow as the axis shrinks, or a squished figure puts its legend on
# top of the x label.
LEGEND_GAP_CM = 1.15


def default_legend_pos(height: str) -> float:
    pt = width_pt(height)
    if pt is None:
        return -0.30
    return -round(LEGEND_GAP_CM / (pt / 28.45), 2)


def ytick_spec(args, knobs: dict, ymax) -> tuple[str, str, float | str]:
    """Return (knob block, axis key, possibly-raised ymax) for the y rules."""
    step = args.ytick_step or (knobs.get("YTickStep") or "10")
    if args.yticks:
        ticks = [float(t) for t in args.yticks.replace(" ", "").split(",") if t]
        if max(ticks) > float(ymax):
            ymax = max(ticks)
        block = (f"\\def\\YTicks{{{args.yticks.replace(' ', '')}}}   % explicit horizontal rules\n"
                 f"%% \\def\\YTickStep{{{step}}}    % (unused while \\YTicks is set)\n")
        return block, "ytick={\\YTicks}", ymax
    block = (f"\\def\\YTickStep{{{step}}}       % labelled horizontal rule every N pp\n"
             f"%% \\def\\YTicks{{10,30,50,70,90}}  % (set this to pick rules explicitly)\n")
    return block, "ytick distance=\\YTickStep", ymax


GRID_NOTE = r"""% Generated by scripts/plot_compact_tikz.py --grid -- regenerating overwrites this file.
% Several compact figures combined into one groupplot with a SINGLE shared legend.
% The .dat files beside this one were copied from the source figure directories
% and prefixed with their tag, so this directory is self-contained.
%
% \input this inside a figure environment. The preamble needs:
%     \usepackage{pgfplots} \pgfplotsset{compat=1.18}
%     \usepgfplotslibrary{fillbetween, groupplots}
%
% THIS FIGURE NEEDS TWO LATEX PASSES: the shared legend travels through
% \label/\ref, so on a first, cold run the legend area comes out empty.
"""


def build(series: list[Series], knobs: dict, args) -> str:
    plots = []
    for idx, s in enumerate(series):
        plots.append(f"      %% ---- {s.cat} ----")
        plots.append(r"      \ifnum\ShowBand=1")
        plots.append(f"        \\addplot[draw=none, forget plot, name path=clo{idx}] table[x=loop, y=lo] {{{s.cat}.dat}};")
        plots.append(f"        \\addplot[draw=none, forget plot, name path=chi{idx}] table[x=loop, y=hi] {{{s.cat}.dat}};")
        plots.append(f"        \\addplot[{s.color}, fill opacity=0.13, forget plot] fill between[of=clo{idx} and chi{idx}];")
        plots.append(r"      \fi")
        plots.append(r"      \ifnum\ShowSeeds=1")
        for col in s.columns:
            plots.append(
                f"        \\addplot[{s.color}, opacity=0.35, line width=0.3pt, forget plot]"
                f" table[x=loop, y={col}] {{{s.cat}.dat}};")
        plots.append(r"      \fi")
        # Thinner lines and smaller marks than the roomy figure; at this size the
        # heavier strokes of the original merge into each other.
        plots.append(f"      \\pgfmathtruncatemacro{{\\MarkPhase}}{{1 + mod({idx}, \\MarkRepeat)}}")
        plots.append(
            f"      \\addplot[{s.color}, {s.dash}, line width=0.9pt, mark={s.mark}, mark size=1.3pt,"
            f" mark repeat=\\MarkRepeat, mark phase=\\MarkPhase, mark options={{solid, fill={s.color}}}]"
            f" table[x=loop, y=mean] {{{s.cat}.dat}};")
        plots.append(f"      \\addlegendentry{{{s.label}}}")

    width = max(len(s.color) for s in series)
    colors = "\n".join(
        f"\\definecolor{{{s.color}}}{{HTML}}{{{s.hex.upper()}}}"
        f"{' ' * (width - len(s.color))}  % {s.label}" for s in series)

    max_iter = args.max_iter or int(knobs.get("MaxIter") or 100)
    ymin = args.ymin if args.ymin is not None else (knobs.get("YMin") or "0")
    ymax = args.ymax if args.ymax is not None else (knobs.get("YMax") or "90")
    ytick_block, ytick_key, ymax = ytick_spec(args, knobs, ymax)
    ymax = f"{float(ymax):g}"
    xtick = args.xtick_step or max(1, 10 * round(max_iter / 5 / 10) or round(max_iter / 5))
    return TEMPLATE.format(
        note=BODY_NOTE, colors=colors, plots="\n".join(plots),
        max_iter=max_iter,
        fig_width=args.width, fig_height=args.height,
        ymin=ymin, ymax=ymax, ytick_block=ytick_block, ytick_key=ytick_key,
        yminor=args.yminor if args.yminor is not None else 0,
        xtick=xtick,
        legend_cols=args.legend_columns or fit_legend_columns([s.label for s in series], args.width),
        legend_pos=f"{args.legend_pos if args.legend_pos is not None else default_legend_pos(args.height):g}",
        show_seeds=args.show_seeds if args.show_seeds is not None else (knobs.get("ShowSeeds") or "0"),
        show_band=args.show_band if args.show_band is not None else (knobs.get("ShowBand") or "1"),
    )


GRID_TEMPLATE = r"""{note}
%% ======================================================================
%%  KNOBS -- all compile-time; none needs a regeneration.
%% ======================================================================
\def\MaxIter{{{max_iter}}}        % inference iterations to plot

%% Size of ONE PANEL's plot box (scale only axis). The whole figure is
%% {cols} x {rows} of these, plus separators, labels and the shared legend.
\def\PanelWidth{{{fig_width}}}
\def\PanelHeight{{{fig_height}}}
\def\HSep{{{hsep}}}
\def\VSep{{{vsep}}}

%% Axis window and ticks (shared by every panel).
\def\YMin{{{ymin}}}
\def\YMax{{{ymax}}}
{ytick_block}\def\YMinorNum{{{yminor}}}       % unlabelled rules between them (0 = none)
\def\XTickStep{{{xtick}}}       % labelled vertical rule every N iterations

%% ---- PANEL TITLES -- one per cell, in reading order -------------------
{titles}
%% ======================================================================

%% ONE shared legend, drawn below the whole group. It is exported from the
%% first panel with `legend to name` and recalled with \ref, so THIS FIGURE
%% NEEDS TWO LATEX PASSES -- on the first the legend space is blank.
\def\LegendCols{{{legend_cols}}}
\def\LegendGap{{{legend_gap}}}    % distance from the group to the legend

%% Run-to-run spread (1 = on, 0 = off).
\def\ShowSeeds{{{show_seeds}}}
\def\ShowBand{{{show_band}}}
%% ======================================================================

\pgfmathtruncatemacro{{\MarkRepeat}}{{max(1, round(\MaxIter/6))}}

%% One colour per series, named after it -- recolour a series here, once.
{colors}

\begin{{tikzpicture}}
  \begin{{groupplot}}[
      group style={{
        group name=ndgrp,
        group size={cols} by {rows},
        horizontal sep=\HSep, vertical sep=\VSep,
        xlabels at=edge bottom, ylabels at=edge left,
        xticklabels at=edge bottom, yticklabels at=edge left,
      }},
      scale only axis,
      width=\PanelWidth, height=\PanelHeight,
      xlabel={{{xlabel}}}, ylabel={{{ylabel}}},
      xlabel near ticks, ylabel near ticks,
      xmin=1, xmax=\MaxIter,
      ymin=\YMin, ymax=\YMax,
      enlarge x limits=false, enlarge y limits=false, clip=true,
      xtick distance=\XTickStep,
      {ytick_key},
      minor y tick num=\YMinorNum,
      grid=both,
      major grid style={{gray!22, line width=0.3pt}},
      minor grid style={{gray!12, line width=0.25pt}},
      tick align=outside,
      tick style={{gray!55, line width=0.4pt}},
      axis line style={{gray!55, line width=0.4pt}},
      title style={{font=\footnotesize, yshift=-1pt}},
      label style={{font=\footnotesize}},
      tick label style={{font=\scriptsize}},
      legend style={{
        font=\footnotesize,
        legend columns=\LegendCols,
        draw=none, fill=none, inner sep=1pt,
        /tikz/every even column/.append style={{column sep=5pt}},
      }},
      legend cell align=left,
    ]

{panels}

  \end{{groupplot}}
  %% Centre the shared legend under the whole group.
  \node[anchor=north] at ([yshift=-\LegendGap]current bounding box.south)
    {{\ref{{{legend_ref}}}}};
\end{{tikzpicture}}
"""


def build_combined(panels: list[tuple[Path, list[Series], dict]], args, rows: int, cols: int,
                   legend_ref: str, titles: list[str]) -> str:
    """One groupplot, one legend, a title per cell. Panels are in reading order."""
    ref_series = panels[0][1]
    body = []
    for pi, ((src, series, _), title) in enumerate(zip(panels, titles)):
        tag = dir_tag(src)
        opts = [f"title={{\\{title_macro(pi)}}}"]
        if pi == 0:  # only the first panel contributes legend entries
            opts.append(f"legend to name={legend_ref}")
        body.append(f"    %% ---- cell {pi + 1}: {tag} ----")
        body.append(f"    \\nextgroupplot[{', '.join(opts)}]")
        for si, sr in enumerate(series):
            dat = f"{tag}__{sr.cat}.dat"
            body.append(r"      \ifnum\ShowBand=1")
            body.append(f"        \\addplot[draw=none, forget plot, name path=g{pi}lo{si}] table[x=loop, y=lo] {{{dat}}};")
            body.append(f"        \\addplot[draw=none, forget plot, name path=g{pi}hi{si}] table[x=loop, y=hi] {{{dat}}};")
            body.append(f"        \\addplot[{sr.color}, fill opacity=0.13, forget plot] fill between[of=g{pi}lo{si} and g{pi}hi{si}];")
            body.append(r"      \fi")
            body.append(r"      \ifnum\ShowSeeds=1")
            for col in sr.columns:
                body.append(f"        \\addplot[{sr.color}, opacity=0.35, line width=0.3pt, forget plot]"
                            f" table[x=loop, y={col}] {{{dat}}};")
            body.append(r"      \fi")
            body.append(f"      \\pgfmathtruncatemacro{{\\MarkPhase}}{{1 + mod({si}, \\MarkRepeat)}}")
            entry = "" if pi == 0 else ", forget plot"
            body.append(
                f"      \\addplot[{sr.color}, {sr.dash}, line width=0.9pt, mark={sr.mark}, mark size=1.3pt,"
                f" mark repeat=\\MarkRepeat, mark phase=\\MarkPhase,"
                f" mark options={{solid, fill={sr.color}}}{entry}]"
                f" table[x=loop, y=mean] {{{dat}}};")
            if pi == 0:
                body.append(f"      \\addlegendentry{{{sr.label}}}")

    width = max(len(sr.color) for sr in ref_series)
    colors = "\n".join(
        f"\\definecolor{{{sr.color}}}{{HTML}}{{{sr.hex.upper()}}}"
        f"{' ' * (width - len(sr.color))}  % {sr.label}" for sr in ref_series)
    title_defs = "\n".join(f"\\def\\{title_macro(i)}{{{t}}}" for i, t in enumerate(titles))

    knobs = panels[0][2]
    max_iter = args.max_iter or int(knobs.get("MaxIter") or 100)
    ymin = args.ymin if args.ymin is not None else (knobs.get("YMin") or "0")
    # One shared window: panels on different scales cannot be compared by eye.
    ymax = args.ymax if args.ymax is not None else max(
        (float(k.get("YMax") or 90) for _, _, k in panels), default=90.0)
    ytick_block, ytick_key, ymax = ytick_spec(args, knobs, ymax)
    xtick = args.xtick_step or max(1, round(max_iter / 5 / 10) * 10 or round(max_iter / 5))
    return GRID_TEMPLATE.format(
        note=GRID_NOTE, colors=colors, titles=title_defs, panels="\n".join(body),
        max_iter=max_iter, fig_width=args.width, fig_height=args.height,
        hsep=args.hsep, vsep=args.vsep, rows=rows, cols=cols,
        ymin=ymin, ymax=f"{float(ymax):g}", ytick_block=ytick_block, ytick_key=ytick_key,
        yminor=args.yminor if args.yminor is not None else 0, xtick=xtick,
        xlabel="Inference iteration", ylabel=r"Top-1 accuracy (\%)",
        legend_cols=args.legend_columns or fit_legend_columns(
            [sr.label for sr in ref_series], f"{cols * (width_pt(args.width) or 227.6):g}pt"),
        legend_gap=args.legend_gap, legend_ref=legend_ref,
        show_seeds=args.show_seeds if args.show_seeds is not None else (knobs.get("ShowSeeds") or "0"),
        show_band=args.show_band if args.show_band is not None else (knobs.get("ShowBand") or "1"),
    )


def run_grid(args) -> None:
    import shutil

    rows, cols = GRID_LAYOUTS[args.grid]
    cells = rows * cols
    if len(args.figure_dir) != cells:
        raise SystemExit(f"--grid {args.grid} needs exactly {cells} directories, got {len(args.figure_dir)}")

    panels = []
    for d in args.figure_dir:
        src = d.resolve()
        if not (src / SOURCE_BODY).exists():
            raise SystemExit(f"{src}: no {SOURCE_BODY} -- run plot_paper_sw_tikz.py for it first")
        series, knobs = parse_source(src)
        panels.append((src, series, knobs))

    # One legend for all panels only makes sense if they draw the same series.
    signature = lambda ser: [(x.label, x.color, x.hex.upper()) for x in ser]
    ref = signature(panels[0][1])
    for src, ser, _ in panels[1:]:
        if signature(ser) != ref:
            raise SystemExit(
                f"{src.name} does not draw the same series as {panels[0][0].name}, so one shared "
                f"legend would mislabel it:\n  {panels[0][0].name}: {[x[0] for x in ref]}\n"
                f"  {src.name}: {[x[0] for x in signature(ser)]}")

    titles = ([t.strip() for t in args.titles.split(";")] if args.titles
              else [panel_title(src) for src, _, _ in panels])
    if len(titles) != cells:
        raise SystemExit(f"--titles needs {cells} entries, got {len(titles)}")

    # Every panel in every grid is the same width, so a 2x1 stack and a 2x2 read at
    # the same scale and can sit side by side in the same document.
    if args.width == PANEL_DEFAULT_WIDTH:
        args.width = GRID_PANEL_WIDTH

    out = (args.out or Path("docs/figures") / f"combined-{args.grid}").resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Every source dir has a lay-row.dat etc., so the names collide; copy them in
    # under a per-source prefix and keep this directory self-contained.
    for src, series, _ in panels:
        for sr in series:
            shutil.copyfile(src / f"{sr.cat}.dat", out / f"{dir_tag(src)}__{sr.cat}.dat")

    legend_ref = "ndlegend" + re.sub(r"[^A-Za-z0-9]", "", out.name)
    (out / "combined_body.tex").write_text(
        build_combined(panels, args, rows, cols, legend_ref, titles))
    (out / "combined.tex").write_text(
        WRAPPER.replace("compact_body.tex", "combined_body.tex")
               .replace("{fillbetween}", "{fillbetween, groupplots}"))

    print(f"wrote {out}/combined_body.tex")
    tot_w = cols * (width_pt(args.width) or 0) + (cols - 1) * (width_pt(args.hsep) or 0)
    tot_h = rows * (width_pt(args.height) or 0) + (rows - 1) * (width_pt(args.vsep) or 0)
    print(f"  {rows}x{cols} (rows x cols), panel box {args.width} x {args.height}"
          f"  ->  plot area {tot_w / 28.45:.1f}cm x {tot_h / 28.45:.1f}cm")
    for i, t in enumerate(titles):
        print(f"    cell r{i // cols + 1}c{i % cols + 1}: {dir_tag(panels[i][0]):10s} -> {t}")
    print(f"  one shared legend ({len(ref)} entries) -- needs two LaTeX passes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure_dir", type=Path, nargs="+",
                    help="directory written by plot_paper_sw_tikz.py (several may be given)")
    ap.add_argument("--width", default=PANEL_DEFAULT_WIDTH,
                    help=f"plot-box width per panel (default: {PANEL_DEFAULT_WIDTH} standalone, "
                         f"{GRID_PANEL_WIDTH} per panel under --grid)")
    ap.add_argument("--height", default="4.2cm", help="plot-box height (default: 4.2cm)")
    ap.add_argument("--max-iter", type=int, default=None, help="default: inherit from the source figure")
    ap.add_argument("--ymin", default=None, help="default: inherit")
    ap.add_argument("--ymax", default=None, help="default: inherit")
    ap.add_argument("--ytick-step", default=None, help="default: inherit (else 10)")
    ap.add_argument("--yticks", default=None, metavar="LIST",
                    help="explicit horizontal rules, e.g. '10,30,50,70,90'; overrides --ytick-step "
                         "and raises \\YMax if the top tick would fall outside the axis")
    ap.add_argument("--grid", choices=sorted(GRID_LAYOUTS), default=None, metavar="RxC",
                    help="combine the given figure dirs into ONE figure with a single shared "
                         "legend: 1x2, 2x1, 2x2, 1x4 or 4x1 (rows x columns). The directories "
                         "are placed in reading order, left to right then top to bottom.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory for --grid (default: docs/figures/combined-<RxC>)")
    ap.add_argument("--titles", default=None, metavar="'A;B;C'",
                    help="semicolon-separated panel titles, in the same order as the "
                         "directories (default: derived from each directory name)")
    ap.add_argument("--hsep", default="1.0cm", help="--grid: horizontal gap between panels")
    ap.add_argument("--vsep", default="1.0cm", help="--grid: vertical gap between panels")
    ap.add_argument("--legend-gap", default="4mm", help="--grid: gap from the group to the legend")
    ap.add_argument("--squish", action="store_true",
                    help="short-and-wide preset: half height and rules every 20pp "
                         "(equivalent to --height 2.6cm --yticks 10,30,50,70,90)")
    ap.add_argument("--yminor", type=int, default=None,
                    help="unlabelled rules between labelled ones (default: 0 -- they crowd at this size)")
    ap.add_argument("--xtick-step", type=int, default=None, help="default: ~5 ticks across the axis")
    ap.add_argument("--legend-columns", type=int, default=None,
                    help="default: the most columns whose widest row still fits the plot box")
    ap.add_argument("--legend-pos", type=float, default=None,
                    help="legend offset in axis-height fractions "
                         "(default: derived from --height to keep a constant ~1.15cm gap)")
    ap.add_argument("--show-seeds", type=int, default=None, choices=(0, 1), help="default: inherit")
    ap.add_argument("--show-band", type=int, default=None, choices=(0, 1), help="default: inherit")
    args = ap.parse_args()
    if args.squish:
        if args.height == ap.get_default("height"):
            args.height = "2.6cm"
        args.yticks = args.yticks or "10,30,50,70,90"

    if args.grid:
        run_grid(args)
        return

    for d in args.figure_dir:
        src = d.resolve()
        if not (src / SOURCE_BODY).exists():
            raise SystemExit(f"{src}: no {SOURCE_BODY} -- run plot_paper_sw_tikz.py for it first")
        series, knobs = parse_source(src)
        (src / "compact_body.tex").write_text(build(series, knobs, args))
        (src / "compact.tex").write_text(WRAPPER)
        print(f"wrote {src}/compact_body.tex")
        print(f"  {len(series)} series: {', '.join(s.label for s in series)}")
        ncols = args.legend_columns or fit_legend_columns([x.label for x in series], args.width)
        print(f"  plot box {args.width} x {args.height}, legend {ncols} col(s) / "
              f"{math.ceil(len(series) / ncols)} row(s)")


if __name__ == "__main__":
    main()
