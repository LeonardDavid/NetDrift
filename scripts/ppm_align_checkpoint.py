#!/usr/bin/env python
"""Greedy sign alignment for PPM — produce an aligned checkpoint and report the cost.

A PPM sort window costs one extra padded wire (``pad=true``), or leaves one
mixed fault-exposed wire (``pad=false``), exactly when its positive count is not
a multiple of ``rt_size``. This flips the fewest, smallest-|w| weights per window
to remove that — the provably minimal repair, and the control arm any
training-time regularizer has to beat.

Two payoffs at once, both reported below:
  * pad=true  — padding overhead -> 0, i.e. PPM costs exactly dense wires.
  * pad=false — every wire sign-pure, i.e. immunity at dense wire count.

Usage:
  # measure only (no write): what does the real checkpoint cost to align?
  python scripts/ppm_align_checkpoint.py \\
      --config configs/modes/vgg7_cifar10_w1a1_polarity.yaml --dry-run

  # write the aligned checkpoint alongside the source, with a _ppmalign marker
  python scripts/ppm_align_checkpoint.py \\
      --config configs/modes/vgg7_cifar10_w1a1_polarity.yaml

Then evaluate it exactly like any other checkpoint (the printed commands do
this): once padded to confirm the wire count fell to dense, once unpadded to see
whether immunity now holds without padding.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code" / "python"))

import torch  # noqa: E402

ALIGN_MARKER = "ppmalign"


def aligned_checkpoint_path(source: str | Path, base_layout: str, window: int) -> Path:
    """``model_best.pt`` -> ``model_best_ppmalign-col-w0.pt``.

    The layout and window are part of the name because the alignment is
    layout-specific: a checkpoint aligned for ``col``/``w0`` is NOT aligned for
    ``row`` or another window, and evaluating the wrong pairing silently reports
    the unaligned cost. Mirrors the ``_endlen`` marker convention in
    ``faults/weight_encoders/apply.py::with_endlen_marker``.
    """
    p = Path(source)
    tag = f"{ALIGN_MARKER}-{base_layout.lower()}-w{int(window)}"
    return p.with_name(f"{p.stem}_{tag}{p.suffix}")


def cheap_to_align(window_width: int, rt_size: int) -> bool:
    """Is aligning a window of this width cheap?

    A balanced window of ``W`` weights has its positive count concentrated at
    ``W/2`` (sd ``sqrt(W)/2``), so the flips needed are the distance from ``W/2``
    to the nearest multiple of ``rt_size``. That distance is 0 only when
    ``W % (2*rt_size) == 0``; otherwise the mode sits up to ``rt_size/2`` away and
    alignment costs ~``rt_size/2`` flips per window — measured at ~45% of all
    weights, versus 1-4% in the divisible case. Windows narrower than ``rt_size``
    are a separate, unfixable case (only multiple reachable is 0).
    """
    return window_width >= rt_size and window_width % (2 * rt_size) == 0


def _measure(model, cfg, rt_size, base_layout, window):
    """Per-layer + total (dense wires, padded wires, unpadded mixed wires)."""
    from netdrift.faults.layout import _layout_weight_for_racetrack
    from netdrift.faults.purity import wire_purity
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    km = (cfg.storage.kernel_mapping or "ROW").upper()
    rows = []
    tot = {"dense": 0, "padded": 0, "mixed": 0, "weights": 0}
    for name, mod in model.named_modules():
        if not isinstance(mod, (QuantizedConv2d, QuantizedLinear)):
            continue
        w_2d, _ = _layout_weight_for_racetrack(
            mod.weight.detach(), rt_mapping=base_layout.upper(),
            kernel_mapping=(km if mod.weight.dim() == 4 else None),
        )
        nr, nc = w_2d.shape
        dense = nr * math.ceil(nc / rt_size)
        padded = wire_purity(w_2d, rt_size, "POLARITY",
                             polarity_params=(window, True)).total
        mixed = wire_purity(w_2d, rt_size, "POLARITY",
                            polarity_params=(window, False)).mixed
        rows.append((name, tuple(mod.weight.shape), dense, padded, mixed,
                     bool(getattr(mod, "protected", False))))
        tot["dense"] += dense
        tot["padded"] += padded
        tot["mixed"] += mixed
        tot["weights"] += int(mod.weight.numel())
    return rows, tot


def _print_table(label, rows, tot):
    print(f"\n{label}")
    print(f"  {'layer':<10}{'shape':<22}{'dense':>9}{'padded':>9}{'overhead':>10}"
          f"{'mixed(nopad)':>14}  prot")
    for name, shape, dense, padded, mixed, prot in rows:
        print(f"  {name:<10}{str(shape):<22}{dense:>9}{padded:>9}"
              f"{padded - dense:>10}{mixed:>14}  {'y' if prot else '-'}")
    ratio = tot["padded"] / tot["dense"] if tot["dense"] else 0.0
    print(f"  {'TOTAL':<10}{'':<22}{tot['dense']:>9}{tot['padded']:>9}"
          f"{tot['padded'] - tot['dense']:>10}{tot['mixed']:>14}"
          f"   ({ratio:.4f}x dense)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="defaults to model.checkpoint from the config")
    ap.add_argument("--out", default=None,
                    help="defaults to the source path with a _ppmalign-<layout>-w<K> marker")
    ap.add_argument("--rt-size", type=int, default=None)
    ap.add_argument("--base-layout", default=None, choices=["row", "col"])
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--include-protected", action="store_true",
                    help="also align protected layers (buys their padding back "
                         "at an accuracy cost that no fault can ever justify)")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure and report, write nothing")
    args = ap.parse_args(argv)

    from netdrift.config.loader import load
    from netdrift.faults.ppm_align import align_model_for_ppm
    from netdrift.models import (
        apply_protection_policy, attach_activation_scheme, build_model,
        replace_with_quantized,
    )
    from netdrift.models.checkpoint import load_checkpoint
    from netdrift.runner.run import _build_activation_scheme, _build_scheme

    cfg = load(args.config)
    rt_size = args.rt_size or cfg.storage.rt_size
    base_layout = (args.base_layout or cfg.storage.base_layout or "col").lower()
    window = args.window if args.window is not None else cfg.storage.partition.window
    ckpt = args.checkpoint or cfg.model.checkpoint
    if not ckpt:
        print("error: no checkpoint given and none in the config", file=sys.stderr)
        return 2

    # Same build sequence as runner/run.py steps 2-3, plus the protection policy
    # (alignment skips protected layers, so the policy must be applied first).
    scheme = _build_scheme(cfg)
    model = build_model(cfg.model.name)
    if scheme is not None:
        replace_with_quantized(model, scheme, skip_first=cfg.model.skip_first_quant,
                               skip_last=cfg.model.skip_last_quant)
    attach_activation_scheme(model, _build_activation_scheme(cfg))
    print(load_checkpoint(model, ckpt, mode=cfg.model.checkpoint_mode, scheme=scheme,
                          scale_init=cfg.quant.scale_init,
                          map_location="cpu").summary())
    apply_protection_policy(model, cfg.fault.protection.policy,
                            layers=cfg.fault.protection.layers,
                            indiv_layer=cfg.fault.protection.indiv_layer)

    print(f"\nPPM alignment target: rt_size={rt_size}  base_layout={base_layout}  "
          f"window={window}  ({'channel-aligned' if window == 0 else f'{window} wires'})")
    if window % 2 == 1 and window != 0:
        print("  WARNING: odd window puts the sign-count mode R/2 away from a "
              "multiple, so alignment needs ~R/2 flips per window (~45% of all "
              "weights). Even windows and channel-aligned cost 1-4%.")

    rows_before, tot_before = _measure(model, cfg, rt_size, base_layout, window)
    _print_table("BEFORE", rows_before, tot_before)

    # Flag layers whose window width makes alignment structurally expensive
    # BEFORE any weights move — cheaper than discovering it in the flip report.
    from netdrift.faults.layout import _layout_weight_for_racetrack
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
    km = (cfg.storage.kernel_mapping or "ROW").upper()
    expensive = []
    for name, mod in model.named_modules():
        if not isinstance(mod, (QuantizedConv2d, QuantizedLinear)):
            continue
        if not args.include_protected and getattr(mod, "protected", False):
            continue
        w_2d, _ = _layout_weight_for_racetrack(
            mod.weight.detach(), rt_mapping=base_layout.upper(),
            kernel_mapping=(km if mod.weight.dim() == 4 else None))
        width = w_2d.shape[1] if window == 0 else window * rt_size
        if width < rt_size:
            expensive.append((name, width, "narrower than rt_size: unfixable, skipped"))
        elif not cheap_to_align(width, rt_size):
            expensive.append((name, width, f"not a multiple of 2*rt_size "
                                           f"({2 * rt_size}): ~{rt_size // 2} flips/window"))
    for name, width, why in expensive:
        print(f"  WARNING {name}: window width {width} is {why}")

    report = align_model_for_ppm(model, rt_size=rt_size, base_layout=base_layout,
                                 window=window,
                                 kernel_mapping=cfg.storage.kernel_mapping,
                                 include_protected=args.include_protected)
    rows_after, tot_after = _measure(model, cfg, rt_size, base_layout, window)
    _print_table("AFTER", rows_after, tot_after)

    flips = sum(r["flips"] for r in report.values())
    aligned_weights = sum(r["weights"] for r in report.values())
    print(f"\nFLIP COST  {flips} weights flipped of {aligned_weights} aligned "
          f"({100 * flips / max(aligned_weights, 1):.3f}%), "
          f"{tot_before['weights']} total in model")
    print(f"  {'layer':<10}{'flips':>9}   {'of':<10}{'%':>8}"
          f"{'|w| pctl mean/max':>20}")
    for name, r in report.items():
        print(f"  {name:<10}{r['flips']:>9} / {r['weights']:<10}"
              f"{100 * r['flip_frac']:>7.3f}%"
              f"{r['flip_abs_pctl_mean']:>11.1f} /{r['flip_abs_pctl_max']:>6.1f}")
    print("  (|w| percentile of the flipped weights within their layer: low means "
          "the flips came from the least confident weights, i.e. cheapest for the "
          "task loss)")
    if tot_after["padded"] != tot_after["dense"] or tot_after["mixed"]:
        print("\n  NOTE: residual overhead/mixed wires remain — expected only for "
              "windows narrower than rt_size (they cannot reach a multiple "
              "without going all-negative) and for skipped protected layers.")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out = Path(args.out) if args.out else aligned_checkpoint_path(ckpt, base_layout, window)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"\naligned checkpoint saved to: {out}")
    print("\nEvaluate it (the padded run should now report n_racetracks == dense, "
          "the unpadded run should be immune):")
    for pad in ("true", "false"):
        print(f"  python netdrift_run.py --config {args.config} --metrics offline \\\n"
              f"      --override model.checkpoint={out} \\\n"
              f"      --override storage.partition.pad={pad} \\\n"
              f"      --override storage.partition.window={window} \\\n"
              f"      --override experiment.name=ppmalign_{base_layout}_w{window}_pad{pad}")
    print("\nAccuracy is the number to watch: compare clean accuracy against the "
          "unaligned checkpoint, then add "
          "--override recalibrate.enabled=true --override recalibrate.on=always "
          "to see how much of any drop BN/Scale recalibration recovers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
