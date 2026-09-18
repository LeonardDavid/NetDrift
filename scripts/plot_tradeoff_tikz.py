#!/usr/bin/env python3
"""Emit a pgfplots/TikZ Pareto scatter of accuracy vs. racetrack (nanowire) cost.

Companion to ``plot_paper_sw_tikz.py``: that one plots accuracy *over inference
iterations*; this one plots the end state of those curves against what the layout
costs in nanowires, so the accuracy/robustness/area tradeoff is one picture.

Input is a ``;``-delimited, comma-decimal CSV (the shape LibreOffice writes on a
German locale)::

    layout;Acc @ 100 iters;Acc drop (after 100 iters);immune RTs;mixed RTs;total RTs
    dense-row;8,60;1,56;0;174480;174480

``Acc drop`` is redundant -- it is ``<baseline acc> - acc`` -- so it is not given
its own visual encoding (that would draw the same shape twice). It rides along in
each point's direct label instead, and ``\\DropSign`` in the .tex flips its sign
without a regeneration.

Writes, into ``--outdir``:

  * ``accuracy_vs_wires_body.tex`` -- the figure body; \\input this into a paper
  * ``accuracy_vs_wires.tex``      -- a standalone wrapper to compile on its own
  * ``README.md``                  -- what backs each point, and the caveats

The five points are inlined as ``coordinates {}``, so unlike the sibling script's
output there are no ``.dat`` files and no ``table/search path`` to configure.

Stdlib only -- runs on the Mac mount without torch/numpy.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Presentation tables, kept byte-identical to plot_paper_sw_tikz.py so the two
# figures agree on what colour and glyph each layout wears.
# ---------------------------------------------------------------------------
STYLE = {
    # csv layout key -> (pretty name, tikz colour name, hex, pgfplots mark)
    "dense-row":    ("Dense-Row",            "ndRow",    "e87ba4", "square*"),
    "dense-col":    ("Dense-Col",            "ndCol",    "1baf7a", "triangle*"),
    "polarity":     ("Polarity",             "ndPol",    "eb6834", "*"),
    "polarity-reg": ("Polarity-Regularized", "ndPolReg", "eda100", "diamond*"),
    "individual":   ("Individual",           "ndBlock",  "2a78d6", "pentagon*"),
}
FALLBACK = ("ndX", "4a3aa7", "o")

# --- combo style -----------------------------------------------------------
# Caller-requested pairing: hue encodes MEANING, not series identity -- green
# for the good quantity (accuracy, immune wires), red for the bad one (drop,
# mixed wires). A line and a bar therefore share a hue and are told apart by
# form. Through the dataviz validator (scripts/validate_palette.js, light mode)
# 00875e/c73039 passes lightness, chroma, normal-vision separation (dE 28.8) and
# contrast, with a CVD WARN at dE 6.3 (deutan) -- red/green is the classic
# confusion pair. That sits in the 6-8 band, which is legal only alongside
# secondary encoding; here there are four such cues, none of them colour:
#   * the two bars are side by side in a fixed order within every group
#   * the lines differ in dash pattern AND marker shape
#   * \ShowBarLabels prints each bar's value (on by default)
#   * the legend swatches are bar-shaped vs line-shaped
COMBO = {
    "acc":    ("ndAcc",    "00875e", "Top-1 accuracy after 100 iters"),
    "drop":   ("ndDrop",   "c73039", "Accuracy drop vs. baseline (pp)"),
    "immune": ("ndImmune", "00875e", "Immune nanowires"),
    "mixed":  ("ndMixed",  "c73039", "Mixed nanowires"),
}


def texname(key: str) -> str:
    r"""'polarity-reg' -> 'PolarityReg', for \ShowPolarityReg / \LblPolarityReg."""
    return "".join(w.capitalize() for w in re.split(r"[^0-9A-Za-z]+", key) if w)

# Layout policy constants. Every one of these is a y- or x-fraction, so the rules
# below hold for any dataset of this shape, not just the r18 one.
BUDGET_TOL = 0.02   # x within 2% of the cheapest point => same "wire budget"
MIN_GAP = 0.12      # two points closer than 12% of the y-range collide as labels
LEADER_X = 2.6      # leader-line labels park at 2.6x the cluster's x
SLOT_GAP = 0.07     # and sit +/- 7% of the y-range from the colliding pair's mid
HEADROOM = 0.12     # a point with this much room above gets its label above it
LABEL_H = 0.11      # a 2-line \footnotesize label is ~11% of the axis height
RIGHT_THIRD = 0.66  # past this x-fraction a label must extend leftwards
NOTE_X = 0.62       # the cost note goes 62% along the log-x axis...
NOTE_Y = 0.30       # ...and 30% up the y-axis: the empty quadrant


def de(s: str) -> float:
    """Parse a comma-decimal field ('8,60' -> 8.6). No thousands separators."""
    return float(s.strip().replace(",", "."))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = []
        for r in csv.DictReader(fh, delimiter=";"):
            key = (r.get("layout") or "").strip()
            if not key:
                continue          # trailing blank line
            cols = list(r)
            rows.append({
                "key": key,
                "acc": de(r[cols[1]]),
                "drop": de(r[cols[2]]),
                "immune": int(de(r[cols[3]])),
                "mixed": int(de(r[cols[4]])),
                "total": int(de(r[cols[5]])),
            })
    return rows


def pareto(rows: list[dict]) -> list[dict]:
    """Points no other point beats on both axes (cheaper-or-equal AND better)."""
    out = []
    for p in rows:
        if not any(q["total"] <= p["total"] and q["acc"] > p["acc"] for q in rows):
            out.append(p)
    return sorted(out, key=lambda p: p["total"])


def place(rows: list[dict], ymax: float) -> None:
    """Decide each point's label placement; annotate rows in place.

    Three modes, picked by geometry rather than by hand:
      ``above``  -- there is headroom over the marker (used for the costly point)
      ``below``  -- the default: label hangs down-right of the marker
      ``leader`` -- the marker has a neighbour too close to label inline, so the
                    label parks out to the right on a thin leader line
    """
    xmin = min(p["total"] for p in rows)
    cluster = [p for p in rows if p["total"] <= xmin * (1 + BUDGET_TOL)]
    loners = [p for p in rows if p not in cluster]

    # Which way the text runs: a point near the right edge must extend leftwards
    # or it walks off the frame.
    lx, hx = math.log10(xmin), math.log10(max(p["total"] for p in rows) * 1.26)
    for p in rows:
        p["side"] = "l" if (math.log10(p["total"]) - lx) / (hx - lx) > RIGHT_THIRD else "r"

    for p in loners:
        p["mode"] = "above" if (ymax - p["acc"]) / ymax > HEADROOM else "below"

    # Walk the cluster top-down; a pair closer than MIN_GAP cannot be labelled
    # inline, so both of them get a leader out into the empty right-hand space.
    # Inside the cluster every label competes for the same sliver of x, so each
    # one hangs into whichever of its two vertical gaps is roomier.
    cluster.sort(key=lambda p: -p["acc"])
    for i, p in enumerate(cluster):
        gap_up = (ymax - p["acc"]) if i == 0 else (cluster[i-1]["acc"] - p["acc"])
        gap_dn = p["acc"] if i == len(cluster) - 1 else (p["acc"] - cluster[i+1]["acc"])
        p["mode"] = "above" if gap_up > gap_dn else "below"
    for a, b in zip(cluster, cluster[1:]):
        if (a["acc"] - b["acc"]) / ymax < MIN_GAP:
            a["mode"] = b["mode"] = "leader"

    colliding = [p for p in cluster if p["mode"] == "leader"]
    if colliding:
        # Stack the slots up from the axis floor rather than centring them on the
        # pair: centred slots put the lowest label under the frame, on top of the
        # x tick labels. Ascending order means the leader lines never cross.
        floor = (LABEL_H / 2 + 0.01) * ymax
        step = max(SLOT_GAP, 1.35 * LABEL_H) * ymax
        for i, p in enumerate(sorted(colliding, key=lambda q: q["acc"])):
            p["slot"] = (xmin * LEADER_X, floor + i * step)


def detect_baseline(rows: list[dict], override: str | None) -> str:
    """The layout the 'Acc drop' column is measured from: the row whose drop is
    zero. Falls back to --baseline, then to the best-accuracy row."""
    if override:
        return override
    zero = [r["key"] for r in rows if abs(r["drop"]) < 5e-3]
    if len(zero) == 1:
        return zero[0]
    return max(rows, key=lambda r: r["acc"])["key"]


def mirror_facts(rows: list[dict]):
    """If acc + drop is constant, the two lines are mirror images about half of
    it. Returns (mirror_y, [(x, y, left_key, right_key), ...]) or (None, [])."""
    sums = {round(r["acc"] + r["drop"], 4) for r in rows}
    if len(sums) != 1:
        return None, []
    mirror = sum(sums) / 2
    cross = []
    for i in range(len(rows) - 1):
        f1 = rows[i]["acc"] - rows[i]["drop"]
        f2 = rows[i + 1]["acc"] - rows[i + 1]["drop"]
        if f1 == 0 or f1 * f2 < 0:
            t = f1 / (f1 - f2)
            cross.append((i + 1 + t,
                          rows[i]["acc"] + t * (rows[i + 1]["acc"] - rows[i]["acc"]),
                          rows[i]["key"], rows[i + 1]["key"]))
    return mirror, cross


def nice_ceil(x: float) -> float:
    """Smallest 1/2/2.5/5 x 10^k that is >= x -- for a readable tick ladder."""
    if x <= 0:
        return 1.0
    k = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if m * k >= x:
            return m * k
    return 10 * k


def fmt_wires(n: int) -> str:
    return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.0f}k"


def tex_label(p: dict, pretty: str) -> str:
    """Two lines: who it is, then acc / immune share / drop-vs-baseline."""
    imm = 100.0 * p["immune"] / p["total"]
    immtxt = "100\\%" if imm >= 99.995 else f"{imm:.1f}\\%"
    drop = ("baseline" if abs(p["drop"]) < 5e-3
            else f"\\DropNum{{{p['drop']:.1f}}}")
    return (f"{pretty}\\\\[-1pt]"
            f"\\scriptsize {p['acc']:.1f}\\%"
            f"\\,$\\cdot$\\,{immtxt} immune"
            f"\\,$\\cdot$\\,{drop}")


def build_body(rows: list[dict], baseline: str, csv_name: str) -> str:
    ymax = 96.0
    place(rows, ymax)
    xlo = min(p["total"] for p in rows)
    xhi = max(p["total"] for p in rows)
    xmin, xmax = xlo / 1.16, xhi * 1.26

    front = pareto(rows)
    L = []
    w = L.append

    w(f"% Generated by scripts/plot_tradeoff_tikz.py from {csv_name}"
      " -- regenerating overwrites this file.")
    w("% Figure body only (no \\documentclass). Two ways to use it:")
    w("%   1. compile the sibling standalone wrapper, or")
    w("%   2. \\input{} this file inside a figure environment in the paper -- the")
    w("%      paper preamble then needs only:")
    w("%          \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}")
    w("%      No .dat files and no table/search path: the five points are inline.")
    w("%")
    w(f"% 'Acc drop' in the CSV is {baseline}'s accuracy minus the row's, so it is a")
    w("% pure affine restatement of the accuracy column -- plotting it as its own")
    w("% series would draw the same shape twice. It rides in the point labels.")
    w("%")
    w("% CAVEAT (verified against runs/paper-runs/paper_resnet18_imagenette_w1a1/):")
    w("%   the wire counts below are NOT all quoted in the same base_layout view.")
    w("%   polarity is base=col; polarity-reg and individual are base=row. The")
    w("%   col-view counterparts on disk are polarity-reg 171859/3016/174875 and")
    w("%   individual 5632790 (32.2x, not 22.8x). See README.md beside this file.")
    w("")
    w("%% ======================================================================")
    w("%%  KNOBS -- all compile-time, none of them need a regeneration.")
    w("%% ======================================================================")
    w("\\def\\FigWidth{12cm}")
    w("\\def\\FigHeight{6.6cm}")
    w("")
    w("%% Accuracy window. \\YMax keeps headroom above the top point for its label.")
    w("\\def\\YMin{0}")
    w(f"\\def\\YMax{{{ymax:.0f}}}")
    w("\\def\\YTickStep{20}")
    w("\\def\\YMinorNum{1}")
    w("")
    w("%% Sign of the drop shown in the labels. The CSV's convention is")
    w("%%   drop = baseline - acc, so an IMPROVEMENT is negative (-64.4).")
    w("%% Set to -1 to print it as a gain instead (+64.4).")
    w("\\def\\DropSign{1}")
    w("")
    w("%% Presentation toggles (1 = on, 0 = off). Plain \\def, not \\newif, so this")
    w("%% body survives being \\input twice.")
    w("\\def\\ShowChance{1}    % horizontal line at random-guess accuracy")
    w("\\def\\ShowBudget{1}    % vertical line through the shared wire budget")
    w("\\def\\ShowFrontier{1}  % dashed arrow along the Pareto frontier")
    w("\\def\\ShowNote{1}      % the boxed 'what the last points cost' note")
    w("")
    w("%% There is deliberately no legend: every point is directly labelled with")
    w("%% its own name, so a legend box would just repeat all five of them. If a")
    w("%% reviewer wants one anyway, uncomment the \\addlegendentry line that sits")
    w("%% commented under each \\addplot below -- the legend style is already set up")
    w("%% in the axis options.")
    w("")
    w("%% Random-guess accuracy: 100/#classes. Imagenette has 10 classes.")
    w("\\def\\ChanceAcc{10}")
    w("%% ======================================================================")
    w("")
    w("%% Prints a signed drop, honouring \\DropSign, with a real minus sign.")
    w("\\def\\DropNum#1{%")
    w("  \\pgfmathparse{\\DropSign*(#1)}%")  # noqa: E501
    w("  \\pgfmathprintnumber[fixed, fixed zerofill, precision=1, print sign]"
      "{\\pgfmathresult}\\,pp}")
    w("")
    w("%% One colour per layout, named after it -- recolour a layout here, once.")
    for p in rows:
        pretty, cname, hexv, _ = STYLE.get(p["key"], (p["key"],) + FALLBACK)
        w(f"\\definecolor{{{cname}}}{{HTML}}{{{hexv.upper()}}}   % {pretty}")
    w("\\colorlet{ndGuide}{black!45}")
    w("")
    w("\\begin{tikzpicture}")
    w("  \\begin{axis}[")
    w("      width=\\FigWidth, height=\\FigHeight,")
    w("      xmode=log, log basis x=10,")
    w("      xlabel={Racetracks (nanowires) for the whole network},")
    w("      ylabel={Top-1 accuracy after 100 iterations (\\%)},")
    w(f"      xmin={xmin:.0f}, xmax={xmax:.0f},")
    w("      ymin=\\YMin, ymax=\\YMax,")
    w("      enlarge x limits=false, enlarge y limits=false,")
    w("      clip=false,   % direct labels are allowed to sit just past the frame")
    w("      xtick={2e5,5e5,1e6,2e6,5e6},")
    w("      xticklabels={0.2\\,M,0.5\\,M,1\\,M,2\\,M,5\\,M},")
    w("      minor xtick={3e5,4e5,6e5,7e5,8e5,9e5,3e6,4e6},")
    w("      ytick distance=\\YTickStep,")
    w("      minor y tick num=\\YMinorNum,")
    w("      grid=both,")
    w("      major grid style={gray!22, line width=0.3pt},")
    w("      minor grid style={gray!12, line width=0.25pt},")
    w("      tick align=outside,")
    w("      tick style={gray!55, line width=0.4pt},")
    w("      axis line style={gray!55, line width=0.4pt},")
    w("      label style={font=\\small},")
    w("      tick label style={font=\\footnotesize},")
    w("      legend style={")
    w("        font=\\footnotesize,")
    w("        at={(0.5,-0.24)}, anchor=north, legend columns=3,")
    w("        draw=gray!40, fill=white, fill opacity=0.9, text opacity=1,")
    w("        /tikz/every even column/.append style={column sep=6pt},")
    w("      },")
    w("      legend cell align=left,")
    w("    ]")
    w("")

    # --- reference lines, drawn first so the markers sit on top of them -------
    w("      %% ---- random-guess floor -------------------------------------------")
    w("      \\ifnum\\ShowChance=1")
    w(f"        \\draw[ndGuide, dashed, line width=0.5pt]")
    w(f"          (axis cs:{xmin:.0f},\\ChanceAcc) -- "
      f"(axis cs:{xmax:.0f},\\ChanceAcc);")
    w(f"        \\node[anchor=south east, font=\\scriptsize, text=black!55]")
    w(f"          at (axis cs:{xmax*0.985:.0f},\\ChanceAcc) "
      "{chance $=$ 10\\% (10 classes)};")
    w("      \\fi")
    w("")
    w("      %% ---- the shared wire budget ---------------------------------------")
    w("      %% Four layouts land within 0.2\\% of each other in x, so on a log axis")
    w("      %% they are one vertical line. That is the finding, not a defect: this")
    w("      %% guide names it instead of pretending there is spread to see.")
    w("      \\ifnum\\ShowBudget=1")
    w(f"        \\draw[ndGuide, dotted, line width=0.6pt]")
    w(f"          (axis cs:{xlo:.0f},\\YMin) -- "
      f"(axis cs:{xlo:.0f},{ymax*0.93:.1f});")
    w(f"        \\node[anchor=west, align=left, font=\\scriptsize, text=black!55]")
    w(f"          at (axis cs:{xlo:.0f},{ymax*0.955:.1f}) "
      "{same wire budget\\\\[-2pt]($1\\times$ dense)};")
    w("      \\fi")
    w("")

    if len(front) >= 2:
        a, b = front[0], front[-1]
        ratio = b["total"] / a["total"]
        gain = b["acc"] - a["acc"]
        w("      %% ---- Pareto frontier ----------------------------------------------")
        w(f"      %% {STYLE.get(a['key'],(a['key'],))[0]} -> "
          f"{STYLE.get(b['key'],(b['key'],))[0]}: everything else is dominated")
        w("      %% (same or higher cost at strictly lower accuracy).")
        w("      \\ifnum\\ShowFrontier=1")
        w("        \\draw[ndGuide, dashed, line width=0.7pt, ->]")
        w(f"          (axis cs:{a['total']*1.10:.0f},{a['acc']+1.0:.2f}) -- "
          f"(axis cs:{b['total']*0.93:.0f},{b['acc']:.2f});")
        w("      \\fi")
        w("")
        notex = 10 ** (math.log10(xmin) + NOTE_X * (math.log10(xmax) - math.log10(xmin)))
        w("      %% ---- what that last step costs -------------------------------------")
        w("      \\ifnum\\ShowNote=1")
        w("        \\node[anchor=west, align=left, font=\\scriptsize,")
        w("              draw=gray!40, fill=white, rounded corners=1.5pt, "
          "inner sep=3.5pt]")
        w(f"          at (axis cs:{notex:.0f},{NOTE_Y*ymax:.1f}) {{%")
        w(f"            \\textbf{{{fmt_wires(b['total'] - a['total'])} more nanowires}}"
          "\\\\[1pt]")
        w(f"            $\\mathbf{{{ratio:.1f}\\times}}$ the area, for "
          f"$\\mathbf{{+{gain:.1f}}}$\\,pp\\\\[1pt]")
        w(f"            over {STYLE.get(a['key'],(a['key'],))[0]}%")
        w("          };")
        w("      \\fi")
        w("")

    # --- the five points -----------------------------------------------------
    for p in rows:
        pretty, cname, _hexv, mark = STYLE.get(p["key"], (p["key"],) + FALLBACK)
        imm = 100.0 * p["immune"] / p["total"]
        w(f"      %% ---- {p['key']} "
          f"(acc {p['acc']:.2f}\\%, {p['immune']}/{p['total']} immune "
          f"= {imm:.2f}\\%) ----")
        w(f"      \\addplot[only marks, mark={mark}, mark size=2.6pt,")
        w(f"               color={cname}, mark options={{fill={cname}, "
          "draw=white, line width=0.6pt}]")
        w(f"        coordinates {{({p['total']},{p['acc']:.2f})}};")
        w(f"      %\\addlegendentry{{{pretty}}}   % uncomment for a legend")
        label = tex_label(p, pretty)
        if p["mode"] == "leader":
            lx, ly = p["slot"]
            w(f"      \\draw[ndGuide, line width=0.4pt] "
              f"(axis cs:{p['total']*1.06:.0f},{p['acc']:.2f}) -- "
              f"(axis cs:{lx*0.97:.0f},{ly:.1f});")
            w(f"      \\node[anchor=west, align=left, font=\\footnotesize, "
              f"text={cname}!75!black]")
            w(f"        at (axis cs:{lx:.0f},{ly:.1f}) {{{label}}};")
        else:
            horiz, align = ("east", "right") if p["side"] == "l" else ("west", "left")
            vert, dy = ("south", 5) if p["mode"] == "above" else ("north", -3)
            dx = -5 if p["side"] == "l" else 5
            w(f"      \\node[anchor={vert} {horiz}, align={align}, "
              f"font=\\footnotesize, text={cname}!75!black, "
              f"xshift={dx}pt, yshift={dy}pt]")
            w(f"        at (axis cs:{p['total']},{p['acc']:.2f}) {{{label}}};")
        w("")

    w("  \\end{axis}")
    w("\\end{tikzpicture}")
    return "\n".join(L) + "\n"


def build_combo_body(rows: list[dict], baseline: str, csv_name: str) -> str:
    """Grouped bars (nanowires, right axis) under two lines (accuracy, left axis).

    Two overlaid ``axis`` environments: the bars are drawn first so the lines sit
    on top of them. Both share width/height/xmin/xmax and are pinned together
    with ``at=(ndbars.south west)``, so the frames coincide exactly.
    """
    n = len(rows)
    step = 20.0                       # accuracy tick spacing
    # The drop is plotted exactly as the CSV records it, so it runs negative and
    # the left axis has to reach below zero far enough to hold the deepest one.
    lo = min([0.0] + [r["drop"] for r in rows])
    ymin, ymax = math.floor(lo / step) * step, 100.0
    above, below = round(ymax / step), round(-ymin / step)
    # Give the right axis the SAME number of intervals on each side of zero, so
    # the two tick ladders land on identical rows and one grid serves both.
    rstep = nice_ceil(max(r["total"] for r in rows) / above)
    rmin, rmax = -below * rstep, above * rstep

    def ytk(v: float) -> str:
        return f"{v:.0f}"

    def rtk(v: float) -> str:
        """Negative wire counts are meaningless, so those ticks go unlabelled."""
        if v < 0:
            return ""
        if v == 0:
            return "0"
        if rmax >= 1e6:
            return f"{v/1e6:g}\\,M"
        return f"{v/1e3:g}\\,k"

    mirror, crossings = mirror_facts(rows)

    lticks = [ymin + i * step for i in range(below + above + 1)]
    rticks = [rmin + i * rstep for i in range(below + above + 1)]
    L = []
    w = L.append

    w(f"% Generated by scripts/plot_tradeoff_tikz.py --style combo from {csv_name}")
    w("% -- regenerating overwrites this file.")
    w("% Figure body only (no \\documentclass). \\input this inside a figure")
    w("% environment; the preamble needs only:")
    w("%     \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}")
    w("%")
    w("% This is a deliberate dual-axis chart: accuracy (left, %) and nanowire")
    w("% counts (right, absolute) are different quantities on one x-axis. Two")
    w("% things to know when reading it:")
    w("%   * bar HEIGHTS are only comparable to each other, never to the lines;")
    w(f"%   * {baseline} is the reference the 'change' line is measured from, so its")
    w("%     change is 0 by construction.")
    w("%")
    w("% CAVEAT: the wire counts are not all quoted in the same base_layout view")
    w("% (polarity is base=col; polarity-reg and individual are base=row).")
    w("% See README.md beside this file.")
    w("")
    w("%% ======================================================================")
    w("%%  KNOBS -- every one is compile-time; none needs a regeneration.")
    w("%% ======================================================================")
    w("\\def\\FigWidth{12.5cm}")
    w("\\def\\FigHeight{6.2cm}")
    w("\\def\\BarWidth{9pt}")
    w("\\def\\LegendCols{2}")
    w("")
    w("%% ---- 1. WHICH LAYOUTS (1 = show, 0 = hide) --------------------------")
    w("%% Hiding a layout drops its bars and its line points but KEEPS its slot")
    w("%% and tick label. Blank the matching \\Lbl... below to clear the label")
    w("%% too, or regenerate with --layouts to close the gap properly.")
    for r in rows:
        w(f"\\def\\Show{texname(r['key'])}{{1}}")
    w("")
    w("%% ---- 2. TICK LABELS -------------------------------------------------")
    w("%% Use \\\\ to break a long name over two lines.")
    for r in rows:
        pretty = STYLE.get(r["key"], (r["key"],))[0]
        w(f"\\def\\Lbl{texname(r['key'])}{{{pretty}}}")
    w("")
    w("%% ---- 3. WHICH SERIES (1 = show, 0 = hide) ---------------------------")
    w("\\def\\ShowAccLine{1}     % final accuracy after 100 iterations")
    w("\\def\\ShowDropLine{1}    % its change against the baseline")
    w("\\def\\ShowImmuneBars{1}  % immune nanowires")
    w("\\def\\ShowMixedBars{1}   % mixed nanowires")
    w("\\def\\ShowBarLabels{1}   % print each bar's exact count above it")
    w("\\def\\ShowGrid{true}     % true/false (not 1/0 -- it is a pgfplots")
    w("                        % boolean). Rules key to BOTH axes at once.")
    w("\\def\\MinorGridNum{1}    % extra unlabelled rules BETWEEN the major ones")
    if mirror is not None:
        w("\\def\\ShowMirror{0}      % mark the line the two curves mirror about")
    w("")
    w("%% ---- 4. COLOURS -- one line each ------------------------------------")
    for k in ("acc", "drop", "immune", "mixed"):
        cname, hexv, label = COMBO[k]
        w(f"\\definecolor{{{cname}}}{{HTML}}{{{hexv.upper()}}}   % {label}")
    w("")
    w("%% ---- 5. AXES ---------------------------------------------------------")
    w(f"\\def\\YMin{{{ymin:.0f}}}"
      + ("        % all drops are positive, so 0 is the floor"
         if ymin >= 0 else "      % low enough for the deepest drop"))
    w(f"\\def\\YMax{{{ymax:.0f}}}")
    w(f"\\def\\RightMin{{{rmin:.0f}}}")
    w(f"\\def\\RightMax{{{rmax:.0f}}}")
    w("%% \\RightMin/\\RightMax deliberately hold the same ratio as \\YMin/\\YMax,")
    w("%% which puts both zeros on the same line and makes the two tick ladders")
    w("%% coincide -- that is the only reason one grid can serve both axes, and")
    w("%% why the bars grow from exactly the height where accuracy reads 0.")
    w("%% If you retune one pair, retune the other to match, and update the two")
    w("%% ytick lists in the axes below.")
    w(f"%% Hiding Individual? Then \\RightMax{{{rstep*above/20:.0f}}} "
      f"\\RightMin{{{-rstep*below/20:.0f}}} zooms the remaining bars.")
    w("")
    w(f"%% The CSV's drop is <{baseline} acc> - <acc>. Plotted verbatim")
    w("%% (\\DropSign{1}), drop and accuracy sum to a constant on every layout, so")
    w("%% the drop line is an exact mirror of the accuracy line about their shared")
    w("%% half-sum -- which is the shape this figure is for. \\DropSign{-1} folds")
    w("%% it into a gain instead (both lines then rise together) and lets you")
    w("%% raise \\YMin back to 0.")
    w("\\def\\DropSign{1}")
    w("%% ======================================================================")
    w("")
    w("%% Appends to the legend without expanding the entry -- this is what lets")
    w("%% the series toggles above add and remove entries without \\addlegendentry")
    w("%% ever being scanned inside a conditional.")
    w("\\def\\ndEmpty{}")
    w("\\def\\ndAddLegend#1{%")
    w("  \\ifx\\LegendList\\ndEmpty")
    w("    \\gdef\\LegendList{#1}%")
    w("  \\else")
    w("    \\expandafter\\gdef\\expandafter\\LegendList"
      "\\expandafter{\\LegendList, #1}%")
    w("  \\fi}")
    w("")
    w("%% Drops a layout's coordinate from every series at once: an empty")
    w("%% \\pgfmathresult tells pgfplots to discard the point.")
    w("\\pgfplotsset{")
    w("  ndLayouts/.style={")
    w("    x filter/.code={%")
    for i, r in enumerate(rows):
        w(f"      \\ifnum\\coordindex={i} \\ifnum\\Show{texname(r['key'])}=0 "
          "\\def\\pgfmathresult{}\\fi\\fi")
    w("    },")
    w("  },")
    w("  ndBarLabels/.style={")
    w("    nodes near coords={\\ifnum\\ShowBarLabels=1"
      "\\pgfmathprintnumber[fixed, precision=0, "
      "1000 sep={\\,}]{\\pgfplotspointmeta}\\fi},")
    w("    every node near coord/.append style={font=\\tiny, rotate=90, "
      "anchor=west, inner sep=1.5pt, text=black!65},")
    w("  },")
    w("}")
    w("")
    w("\\begin{tikzpicture}")
    w("  %% The legend accumulates globally (it has to cross the axis group")
    w("  %% boundary), so it is reset here, inside the picture and ahead of any")
    w("  %% \\ndAddLegend, rather than at file scope.")
    w("  \\gdef\\LegendList{}")
    w("")
    w("  %% ==== axis 1: the bars, on the RIGHT-hand scale =====================")
    w("  %% Drawn first so the lines land on top of it.")
    w("  \\begin{axis}[")
    w("      name=ndbars,")
    w("      width=\\FigWidth, height=\\FigHeight, scale only axis,")
    w("      ybar, bar width=\\BarWidth,")
    w(f"      xmin=0.5, xmax={n}.5,")
    w("      ymin=\\RightMin, ymax=\\RightMax,")
    w("      axis y line*=right,")
    w("      axis x line*=bottom,")
    w("      ylabel={Nanowires (racetracks)},")
    w("      xtick={" + ",".join(str(i + 1) for i in range(n)) + "},")
    w("      xticklabels={" + ",".join("\\Lbl" + texname(r["key"]) for r in rows) + "},")
    w("      ytick={" + ",".join(f"{v:.0f}" for v in rticks) + "},")
    w("      yticklabels={" + ",".join(rtk(v) for v in rticks) + "},")
    w("      %% Without this pgfplots factors a common multiplier out of the")
    w("      %% large tick values and prints a stray '.10^6' above the axis --")
    w("      %% the labels above already carry the unit.")
    w("      scaled y ticks=false,")
    w("      enlarge x limits=false, enlarge y limits=false,")
    w("      ymajorgrids=\\ShowGrid,")
    w("      yminorgrids=\\ShowGrid,")
    w("      minor y tick num=\\MinorGridNum,")
    w("      major grid style={gray!20, line width=0.3pt},")
    w("      minor grid style={gray!11, line width=0.25pt},")
    w("      %% the minor ladder exists for the grid only -- the labelled left")
    w("      %% axis carries the matching tick marks, this side shows none.")
    w("      minor tick length=0pt,")
    w("      tick align=outside,")
    w("      tick style={gray!55, line width=0.4pt},")
    w("      axis line style={gray!55, line width=0.4pt},")
    w("      label style={font=\\small},")
    w("      tick label style={font=\\footnotesize},")
    w("      x tick label style={align=center, font=\\footnotesize},")
    w("    ]")
    w("")
    for k, col in (("immune", "immune"), ("mixed", "mixed")):
        cname = COMBO[k][0]
        toggle = "ShowImmuneBars" if k == "immune" else "ShowMixedBars"
        w(f"    %% ---- {COMBO[k][2]} ----")
        w(f"    \\ifnum\\{toggle}=1")
        w(f"      \\addplot[ndLayouts, ndBarLabels, fill={cname}, "
          f"draw={cname}!70!black, line width=0.3pt]")
        w("        coordinates {")
        for i, r in enumerate(rows):
            w(f"          ({i+1},{r[col]})   % {r['key']}")
        w("        };")
        w(f"      \\ndAddLegend{{{COMBO[k][2]}}}")
        w("    \\fi")
        w("")
    w("  \\end{axis}")
    w("")
    w("  %% ==== axis 2: the lines, on the LEFT-hand scale =====================")
    w("  %% Pinned to axis 1's frame, so the two grids coincide exactly.")
    w("  \\begin{axis}[")
    w("      at=(ndbars.south west), anchor=south west,")
    w("      width=\\FigWidth, height=\\FigHeight, scale only axis,")
    w(f"      xmin=0.5, xmax={n}.5,")
    w("      ymin=\\YMin, ymax=\\YMax,")
    w("      axis y line*=left,")
    w("      axis x line=none,")
    w("      ylabel={Top-1 accuracy (\\%) / change (pp)},")
    w("      ytick={" + ",".join(ytk(v) for v in lticks) + "},")
    w("      minor y tick num=\\MinorGridNum,   % marks matching the minor grid")
    w("      enlarge x limits=false, enlarge y limits=false,")
    w("      tick align=outside,")
    w("      tick style={gray!55, line width=0.4pt},")
    w("      axis line style={gray!55, line width=0.4pt},")
    w("      label style={font=\\small},")
    w("      tick label style={font=\\footnotesize},")
    w("      legend style={")
    w("        font=\\footnotesize,")
    w("        at={(0.5,-0.16)}, anchor=north,")
    w("        legend columns=\\LegendCols,")
    w("        draw=gray!40, fill=white, fill opacity=0.9, text opacity=1,")
    w("        /tikz/every even column/.append style={column sep=8pt},")
    w("      },")
    w("      legend cell align=left,")
    w("    ]")
    w("")
    if mirror is not None:
        w("    %% ---- axis of symmetry ---------------------------------------------")
        w(f"    %% accuracy + drop = {mirror*2:.2f} ({baseline}'s accuracy) on every")
        w(f"    %% layout, so the two lines are exact mirror images about "
          f"y = {mirror:.2f}")
        w("    %% and every crossing they have lies on it.")
        if crossings:
            for x, y, k1, k2 in crossings:
                w(f"    %%   crossing at x = {x:.3f} (between {k1} and {k2}), "
                  f"y = {y:.2f}")
        else:
            w(f"    %%   they never cross: accuracy stays on one side of "
              f"y = {mirror:.2f}.")
        w("    \\ifnum\\ShowMirror=1")
        w("      \\draw[black!40, dotted, line width=0.5pt]")
        w(f"        (axis cs:0.5,{mirror:.2f}) -- (axis cs:{n}.5,{mirror:.2f});")
        w("      \\node[anchor=south west, font=\\scriptsize, text=black!50, "
          "inner sep=1.5pt]")
        w(f"        at (axis cs:0.55,{mirror:.2f}) "
          f"{{mirror axis, $y={mirror:.2f}$}};")
        w("    \\fi")
        w("")
    w("    %% ---- legend swatches for the two bar series ------------------------")
    w("    %% They live in the other axis, so the legend needs a stand-in here.")
    w("    %% \\relax after each one keeps any trailing lookahead off the \\fi.")
    for k in ("immune", "mixed"):
        cname = COMBO[k][0]
        toggle = "ShowImmuneBars" if k == "immune" else "ShowMixedBars"
        w(f"    \\ifnum\\{toggle}=1")
        w(f"      \\addlegendimage{{ybar, ybar legend, fill={cname}, "
          f"draw={cname}!70!black}}\\relax")
        w("    \\fi")
    w("")

    w("    %% ---- " + COMBO["acc"][2] + " ----")
    w("    \\ifnum\\ShowAccLine=1")
    w("      \\addplot[ndLayouts, ndAcc, line width=1.1pt, mark=*, mark size=2.2pt,")
    w("               mark options={fill=ndAcc, draw=white, line width=0.5pt}]")
    w("        coordinates {")
    for i, r in enumerate(rows):
        w(f"          ({i+1},{r['acc']:.2f})   % {r['key']}")
    w("        };")
    w(f"      \\ndAddLegend{{{COMBO['acc'][2]}}}")
    w("    \\fi")
    w("")
    w("    %% ---- " + COMBO["drop"][2] + " ----")
    w("    %% Plotted through \\DropSign, so flipping that macro flips this line.")
    w("    \\ifnum\\ShowDropLine=1")
    w("      \\addplot[ndLayouts, ndDrop, line width=1.1pt, densely dashed,")
    w("               y filter/.expression={\\DropSign*y},")
    w("               mark=square*, mark size=2.0pt,")
    w("               mark options={solid, fill=ndDrop, draw=white, "
      "line width=0.5pt}]")
    w("        %% raw CSV values; \\DropSign above decides which way up they go")
    w("        coordinates {")
    for i, r in enumerate(rows):
        w(f"          ({i+1},{r['drop']:.2f})   % {r['key']}")
    w("        };")
    w(f"      \\ndAddLegend{{{COMBO['drop'][2]}}}")
    w("    \\fi")
    w("")
    w("    \\ifx\\LegendList\\ndEmpty\\else")
    w("      \\expandafter\\legend\\expandafter{\\LegendList}")
    w("    \\fi")
    w("")
    w("  \\end{axis}")
    w("\\end{tikzpicture}")
    return "\n".join(L) + "\n"


def build_wrapper(csv_name: str, body: str) -> str:
    return f"""% Generated by scripts/plot_tradeoff_tikz.py from {csv_name} -- \
regenerating overwrites this file.
% Standalone wrapper; all the figure content (and every knob) lives in the
% _body.tex file this inputs.
\\documentclass[tikz, border=2pt]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\begin{{document}}
\\input{{{body}}}
\\end{{document}}
"""


def build_readme(rows: list[dict], baseline: str, csv_path: Path) -> str:
    front = pareto(rows)
    lines = [
        "# Accuracy vs. nanowire cost (ResNet-18 / Imagenette, w1a1)",
        "",
        f"Generated by `scripts/plot_tradeoff_tikz.py` from `{csv_path}`.",
        "",
        "## Compile",
        "",
        "```sh",
        "cd docs/figures/tradeoff-r18 && pdflatex accuracy_vs_wires.tex",
        "```",
        "",
        "To drop it into the paper instead, `\\input{...accuracy_vs_wires_body.tex}`",
        "inside a `figure` environment. The preamble needs `pgfplots` with",
        "`compat=1.18`; there are no `.dat`",
        "files to put on `table/search path`.",
        "",
        "## The points",
        "",
        "| layout | acc @100 | drop vs. baseline | immune | mixed | total | immune % |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in rows:
        pretty = STYLE.get(p["key"], (p["key"],))[0]
        imm = 100.0 * p["immune"] / p["total"]
        drop = "baseline (ref)" if abs(p["drop"]) < 5e-3 else f"{p['drop']:+.2f} pp"
        lines.append(
            f"| {pretty} | {p['acc']:.2f}% | {drop} | "
            f"{p['immune']:,} | {p['mixed']:,} | {p['total']:,} | {imm:.2f}% |"
        )
    a, b = front[0], front[-1]
    xlo = min(p["total"] for p in rows)
    cl_acc = [p["acc"] for p in rows if p["total"] <= xlo * (1 + BUDGET_TOL)]
    lines += [
        "",
        "## What the figure says",
        "",
        f"* Four of the five layouts sit within 0.2% of each other in total wires, "
        f"so on a log x-axis they are a single vertical line. Along that one wire "
        f"budget, accuracy runs from {min(cl_acc):.1f}% to {max(cl_acc):.1f}% -- "
        f"robustness here is bought by *arrangement*, not by area.",
        f"* Dense-Row and Dense-Col are the same point in every column: identical "
        f"budget, 0% immune, and 8.6%/10.2% are both 10-class chance. They are "
        f"labelled on leader lines because no 2D projection separates them.",
        f"* The Pareto frontier is just "
        f"{STYLE.get(a['key'], (a['key'],))[0]} -> "
        f"{STYLE.get(b['key'], (b['key'],))[0]}; the other three are dominated. "
        f"That last step buys {b['acc'] - a['acc']:+.1f} pp for "
        f"{b['total'] / a['total']:.1f}x the nanowires.",
        "",
        "## Caveats",
        "",
        f"1. **`Acc drop` is not independent data.** It is "
        f"`{baseline}` accuracy minus the row's accuracy, i.e. an affine "
        f"restatement of column 2. It is therefore reported in each point's label "
        f"rather than given its own axis or series.",
        "2. **The wire counts mix two `base_layout` views.** Every triple in the "
        "CSV matches an artifact under "
        "`runs/paper-runs/paper_resnet18_imagenette_w1a1/` exactly, but:",
        "",
        "   | CSV row | artifact | `base_layout` |",
        "   |---|---|---|",
        "   | dense-row | `lay-row` var-base | row |",
        "   | dense-col | `lay-col` (1 pure wire, rounded to 0) | col |",
        "   | polarity | `lay-polarity` pad=f var-**cat8** | **col** |",
        "   | polarity-reg | `lay-polarity-regularized` pad=f var-ppmreg | **row** |",
        "   | individual | `lay-block` var-**base** | **row** |",
        "",
        "   The col-view counterparts on disk are polarity-reg "
        "`171859 / 3016 / 174875` (98.3% immune) and individual `5632790` "
        "(**32.2x**, not 22.8x). Per the project's overhead-accounting rule, "
        "ratios must be same-view or they disagree with the published tables -- "
        "so decide which view the paper quotes before this figure ships.",
        "3. **The rows are also not matched on training variant**: polarity is a "
        "`cat8` (STE-inject) run while individual is a `base` run.",
        "",
        "## Knobs",
        "",
        "All at the top of `accuracy_vs_wires_body.tex`; each is compile-time, so "
        "changing one needs a recompile, never a regeneration:",
        "`\\FigWidth`, `\\FigHeight`, `\\YMin`/`\\YMax`, `\\YTickStep`, "
        "`\\DropSign` (set `-1` to print gains as positive), `\\ChanceAcc`, and the "
        "toggles `\\ShowChance`, `\\ShowBudget`, `\\ShowFrontier`, `\\ShowNote`, "
        "`\\ShowLegend`.",
        "",
    ]
    return "\n".join(lines)


def build_combo_readme(rows: list[dict], baseline: str, csv_path: Path) -> str:
    mirror, crossings = mirror_facts(rows)
    pretty = lambda k: STYLE.get(k, (k,))[0]
    if crossings:
        where = "; ".join(
            f"between **{pretty(k1)}** and **{pretty(k2)}** "
            f"(x = {x:.2f}, y = {y:.2f})" for x, y, k1, k2 in crossings)
        cross_txt = [
            f"   They cross {'once' if len(crossings) == 1 else f'{len(crossings)} times'}, "
            f"and every crossing sits on the mirror line: {where}.",
            f"   `\\def\\ShowMirror{{1}}` draws that line if you want it marked.",
        ]
    else:
        closest = min(rows, key=lambda r: abs(r["acc"] - r["drop"]))
        cross_txt = [
            f"   They never cross: accuracy stays on one side of "
            f"y = {mirror:.2f}, closing to "
            f"{abs(closest['acc'] - closest['drop']):.2f} pp at "
            f"{pretty(closest['key'])} before diverging.",
        ]
    lines = [
        "# Accuracy and nanowire cost per layout (ResNet-18 / Imagenette, w1a1)",
        "",
        f"Generated by `scripts/plot_tradeoff_tikz.py --style combo` from "
        f"`{csv_path}`.",
        "",
        "Grouped bars (nanowire counts, right axis) with two lines over them",
        "(accuracy and its change, left axis), one group per layout.",
        "",
        "## Compile",
        "",
        "```sh",
        "cd docs/figures/tradeoff-r18-combo && pdflatex accuracy_and_wires.tex",
        "```",
        "",
        "To use it in the paper, `\\input` `accuracy_and_wires_body.tex` inside a",
        "`figure` environment; the preamble needs only `pgfplots` with",
        "`compat=1.18`. No `.dat` files.",
        "",
        "## Turning things on and off",
        "",
        "Everything is a `\\def` in the KNOBS block at the top of the body file,",
        "numbered 1-5. Change one and recompile; nothing needs regenerating.",
        "",
        "| I want to... | Edit |",
        "|---|---|",
        "| hide a layout | block 1: `\\def\\ShowPolarity{0}` |",
        "| rename / re-wrap a tick label | block 2: `\\def\\LblPolarity{Polarity}` |",
        "| hide a line or a bar series | block 3: `\\def\\ShowDropLine{0}` |",
        "| hide the per-bar counts | block 3: `\\def\\ShowBarLabels{0}` |",
        "| recolour any series | block 4: one `\\definecolor` per series |",
        "| more/fewer rules between the major ones | block 3: "
        "`\\def\\MinorGridNum{1}` |",
        "| rescale either axis | block 5: `\\YMax`, `\\RightMax` |",
        "| fold the drop into a gain instead | block 5: `\\def\\DropSign{-1}` |",
        "| mark the axis the two lines mirror about | block 3: "
        "`\\def\\ShowMirror{1}` |",
        "",
        "Hiding a layout drops its bars and line points but **keeps its slot and",
        "tick label** — blank the matching `\\Lbl...` to clear the label, or",
        "regenerate with `--layouts` to close the gap:",
        "",
        "```sh",
        "python3 scripts/plot_tradeoff_tikz.py --style combo \\",
        "    --layouts dense-col,polarity,polarity-reg,individual",
        "```",
        "",
        "## Colour",
        "",
        "Hue encodes **meaning, not series**: green for the good quantity",
        "(accuracy, immune wires), red for the bad one (drop, mixed wires). So a",
        "line and a bar deliberately share a hue and are told apart by form.",
        "",
        "Red/green is the classic confusion pair — the validator puts it at",
        "ΔE 6.3 under deuteranopia (ΔE 28.8 for normal vision, and contrast",
        "passes). That is legal in this palette's terms only because identity",
        "never rests on hue here: the two bars sit side by side in a fixed order",
        "within each group, the lines differ in both dash pattern and marker",
        "shape, `\\ShowBarLabels` prints every bar's value, and the legend",
        "swatches are bar-shaped vs line-shaped. Keep at least one of those if",
        "you restyle it.",
        "",
        "## Two things to know when reading it",
        "",
        "1. **It is a dual-axis chart by request.** Bar heights are comparable to",
        "   each other and to the right axis only — never to the lines. The two",
        "   scales are pinned so their zeros and their tick ladders coincide",
        "   (`\\RightMin/\\RightMax` holds the same ratio as `\\YMin/\\YMax`), which",
        "   is the only reason a single grid is honest here. Keep that ratio if",
        "   you retune either axis.",
        "2. **The drop line is the accuracy line, mirrored.** `Acc drop` is",
        f"   **{pretty(baseline)}**'s accuracy minus the row's, so accuracy + drop",
        f"   equals {mirror*2:.2f} on *every* layout. The two curves are therefore",
        f"   exact mirror images about **y = {mirror:.2f}**, and the drop line adds",
        "   no information the accuracy line does not already carry — it is in the",
        "   figure because it was asked for, and it is plotted verbatim from the",
        "   CSV (`\\DropSign{1}`).",
        "",
        *cross_txt,
        "",
        "## Scale warning",
        "",
        f"`individual` needs {rows[-1]['total']:,} nanowires against ~175,000 for",
        "the other four — about 22.8x. On a linear right axis that makes four of",
        "the five bar groups slivers near the floor. Three ways out, all one edit:",
        "",
        "* keep `\\ShowBarLabels{1}` (the default) so every bar prints its count;",
        "* `\\def\\ShowIndividual{0}` plus `\\RightMax{200000}` `\\RightMin{-20000}`",
        "  to zoom the remaining four;",
        "* or use the Pareto scatter in `docs/figures/tradeoff-r18/`, which puts",
        "  cost on a log axis and does not have this problem.",
        "",
        "## The points",
        "",
        "| layout | acc @100 | change vs. baseline | immune | mixed | total |",
        "|---|---|---|---|---|---|",
    ]
    for p in rows:
        pretty = STYLE.get(p["key"], (p["key"],))[0]
        d = "baseline (ref)" if abs(p["drop"]) < 5e-3 else f"{p['drop']:+.2f} pp"
        lines.append(f"| {pretty} | {p['acc']:.2f}% | {d} | {p['immune']:,} | "
                     f"{p['mixed']:,} | {p['total']:,} |")
    lines += [
        "",
        "(The drop column is exactly as the CSV records it, which is how the",
        "figure plots it: negative means better than the baseline.)",
        "",
        "## Caveat carried over from the Pareto figure",
        "",
        "The wire counts are **not all quoted in the same `base_layout` view**:",
        "`polarity` is base=col, while `polarity-reg` and `individual` are",
        "base=row. The col-view counterparts on disk are polarity-reg",
        "`171859 / 3016 / 174875` and individual `5632790`. See",
        "`docs/figures/tradeoff-r18/README.md` for the full table.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=Path("runs/paper-runs/tradeoff/r18-tradeoff.csv"))
    ap.add_argument("--style", choices=("pareto", "combo"), default="pareto",
                    help="pareto: accuracy vs. wires on a log cost axis. "
                         "combo: per-layout bars (wires) under lines (accuracy).")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="default: docs/figures/tradeoff-r18[-combo]")
    ap.add_argument("--layouts", default=None,
                    help="comma-separated CSV layout keys to keep, in this order; "
                         "default is every row, in CSV order")
    ap.add_argument("--baseline", default=None,
                    help="the layout the CSV's 'Acc drop' column is measured from; "
                         "by default the row whose drop is 0")
    args = ap.parse_args()

    rows = read_csv(args.csv)
    if not rows:
        raise SystemExit(f"no data rows in {args.csv}")
    if args.layouts:
        want = [k.strip() for k in args.layouts.split(",") if k.strip()]
        have = {r["key"]: r for r in rows}
        missing = [k for k in want if k not in have]
        if missing:
            raise SystemExit(f"--layouts: no such row(s) {missing}; "
                             f"the CSV has {sorted(have)}")
        rows = [have[k] for k in want]
    baseline = detect_baseline(rows, args.baseline)

    outdir = args.outdir or Path(
        "docs/figures/tradeoff-r18" + ("-combo" if args.style == "combo" else ""))
    outdir.mkdir(parents=True, exist_ok=True)
    name = args.csv.as_posix()

    if args.style == "combo":
        stem, body_fn, readme_fn = ("accuracy_and_wires", build_combo_body,
                                    build_combo_readme)
    else:
        stem, body_fn, readme_fn = ("accuracy_vs_wires", build_body, build_readme)

    (outdir / f"{stem}_body.tex").write_text(
        body_fn(rows, baseline, name), encoding="utf-8")
    (outdir / f"{stem}.tex").write_text(
        build_wrapper(name, f"{stem}_body.tex"), encoding="utf-8")
    (outdir / "README.md").write_text(
        readme_fn(rows, baseline, args.csv), encoding="utf-8")

    print(f"wrote 3 files to {outdir}/ from {len(rows)} rows "
          f"({args.style}, baseline={baseline})")
    for p in rows:
        extra = f"  label={p['mode']}" if "mode" in p else ""
        print(f"  {p['key']:<14} acc={p['acc']:>6.2f}  "
              f"immune={100*p['immune']/p['total']:>6.2f}%  "
              f"wires={p['total']:>9,}{extra}")


if __name__ == "__main__":
    main()
