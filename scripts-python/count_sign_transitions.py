#!/usr/bin/env python
"""Count per-racetrack sign transitions in a checkpoint's weights.

Ground-truth measurement for the run-length regularizer (Capability #2): it
counts, for each unprotected quantized layer, how many adjacent same-racetrack
weight pairs change sign — exactly what the regularizer targets and what endlen
later tries to merge away. Unlike the endlen "weight entries changed %", this
metric is NOT capped by any bitflip budget, so it cleanly shows whether
fault-aware training pushed the weights toward longer same-sign runs.

It lays weights out with the SAME helper the fault model / regularizer use
(``_layout_weight_for_racetrack``), counts ``sign(w[:, i]) != sign(w[:, i+1])``
within each ``rt_size`` block (no cross-block pairs), and reports the total and
the fraction of adjacent pairs that are transitions.

Usage:
    python scripts-python/count_sign_transitions.py \\
        --model vgg3_fmnist --checkpoint runs/reg_lam05/model.pt \\
        --rt-size 64 --layout row --kernel-mapping row \\
        --unprotected 2 3

Compare two checkpoints (regularizer ON vs control) by running it on each: a
lower transition fraction for the higher-lambda model is the proof Capability #2
works. CPU-only; no CUDA / fault model needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the netdrift package importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "code" / "python"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from netdrift.faults.layout import _layout_weight_for_racetrack
from netdrift.models import build_model, replace_with_quantized
from netdrift.models.checkpoint import load_checkpoint
from netdrift.quant.binary import BinaryScheme
from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear


def count_transitions_in_layer(
    weight: torch.Tensor, *, rt_size: int, layout: str, kernel_mapping: str
) -> tuple[int, int]:
    """Return ``(transitions, adjacent_pairs)`` for one layer's weight.

    Lays the weight into its racetrack-aligned 2D view, then counts sign changes
    between adjacent columns WITHIN each ``rt_size`` block (floor over full
    racetracks, matching endlen's coverage). A "transition" is an adjacent pair
    whose signs differ; treating sign(0) as +1 so the count is well-defined.
    """
    km = kernel_mapping.upper() if weight.dim() == 4 else None
    w_2d, _undo = _layout_weight_for_racetrack(
        weight, rt_mapping=layout.upper(), kernel_mapping=km
    )
    rows, cols = w_2d.shape
    n_full = cols // rt_size
    if n_full == 0:
        return 0, 0
    block = w_2d[:, : n_full * rt_size].reshape(rows, n_full, rt_size)
    # sign with 0 -> +1 so a zero weight never registers a spurious transition.
    s = torch.where(block >= 0, 1, -1)
    diff = s[..., :-1] != s[..., 1:]  # (rows, n_full, rt_size-1)
    transitions = int(diff.sum().item())
    adjacent_pairs = diff.numel()
    return transitions, adjacent_pairs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="registry name, e.g. vgg3_fmnist")
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    p.add_argument("--rt-size", type=int, default=64)
    p.add_argument("--layout", default="row", help="row | col")
    p.add_argument("--kernel-mapping", default="row", help="row | col | clw | acw")
    p.add_argument(
        "--unprotected", type=int, nargs="*", default=None,
        help="1-based layer ids to count (default: all quantized layers). "
             "Pass the same ids the run leaves unprotected to match the "
             "regularizer's scope.",
    )
    args = p.parse_args(argv)

    model = build_model(args.model)
    scheme = BinaryScheme()
    replace_with_quantized(model, scheme)
    report = load_checkpoint(
        model, args.checkpoint, mode="strict", scheme=scheme,
        scale_init="max_abs", map_location="cpu",
    )
    print(report.summary())

    total_t = 0
    total_pairs = 0
    print(f"\nSign transitions per racetrack  (rt_size={args.rt_size}, "
          f"layout={args.layout}, kernel_mapping={args.kernel_mapping})")
    print("-" * 72)
    for _, m in model.named_modules():
        if not isinstance(m, (QuantizedConv2d, QuantizedLinear)):
            continue
        if args.unprotected is not None and m.layer_id not in args.unprotected:
            continue
        # Binarize the latent weight the way the fault model reads it.
        qw = scheme.quantize(m.weight.data, getattr(m, "scale_per_channel", None)).values
        t, pairs = count_transitions_in_layer(
            qw, rt_size=args.rt_size, layout=args.layout,
            kernel_mapping=args.kernel_mapping,
        )
        frac = (100.0 * t / pairs) if pairs else 0.0
        total_t += t
        total_pairs += pairs
        print(f"  {m.layer_name:32s} layer_id={m.layer_id}  "
              f"{t:10d} / {pairs:10d} transitions ({frac:6.2f}%)")
    overall = (100.0 * total_t / total_pairs) if total_pairs else 0.0
    print("-" * 72)
    print(f"  {'TOTAL':32s}            "
          f"{total_t:10d} / {total_pairs:10d} transitions ({overall:6.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
