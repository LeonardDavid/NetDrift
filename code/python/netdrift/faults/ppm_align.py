"""Greedy sign alignment for PPM — the minimum-flip baseline.

A PPM sort window of ``W`` weights holding ``p`` positives costs one extra wire
(``pad=True``) or leaves one mixed, fault-exposed wire (``pad=False``) exactly
when ``p`` is not a multiple of ``rt_size`` — measured and cross-checked against
``purity.wire_purity`` in the feasibility probe. Making every window conform
therefore does two things at once:

* ``pad=True``  — padding overhead drops to zero: PPM costs exactly dense.
* ``pad=False`` — every wire becomes sign-pure: immunity at dense wire count.

Repairing one window needs ``min(p mod R, R - p mod R)`` flips: give up the
excess positives, or recruit the missing few from the negatives, whichever side
is nearer. Since windows are independent and each flip moves the count by
exactly one, greedy is *optimal* — no search, and no regularizer can beat it on
flip count. What a regularizer can do that this cannot is let the network
*co-adapt* while the constraint is imposed, instead of having its signs rewritten
after training; this module is the control arm that measurement is against.

Flips pick the smallest ``|w|`` on the giving side, i.e. the weights nearest the
decision boundary, and negate them (preserving ``|w|``, so distance-to-threshold
statistics stay comparable and later fine-tuning can move them back cheaply).

Sign convention matches ``layout.extract_blocks``/``BinaryScheme``: ``w > 0 ->
+1``. Windows shorter than ``rt_size`` are skipped: their only reachable
multiple is 0, which would force the whole window to one sign (vgg7's ``fc2`` is
10 wide at ``rt_size=64``).

No numba/CUDA: this is a latent-weight transform, applied to a checkpoint before
training or evaluation, not a write-time encoder. (The
``faults.weight_encoders`` interface sees only the binarized ``±1`` device array
and so cannot pick the cheapest weights to flip.)
"""

from __future__ import annotations

import torch

__all__ = ["align_windows_to_multiple", "align_model_for_ppm"]


def align_windows_to_multiple(
    w_2d: torch.Tensor,
    rt_size: int,
    *,
    window: int = 0,
) -> tuple[torch.Tensor, int]:
    """Flip the fewest weights so every window's positive count divides ``rt_size``.

    Args:
        w_2d:    Racetrack-aligned 2D view of the LATENT weights (already
                 transposed for COL and kernel-permuted for convs), the same
                 convention ``wire_purity`` uses.
        rt_size: Bits per racetrack.
        window:  Racetracks per sort window; ``0`` = channel-aligned (one window
                 per row), matching ``partitioning.CHANNEL_ALIGNED``.

    Returns:
        ``(aligned, n_flips)`` — a new tensor (the input is never mutated) and
        the number of weights whose sign changed.

    Raises:
        ValueError: if ``rt_size < 1`` or ``window < 0``.
    """
    if rt_size < 1:
        raise ValueError(f"rt_size must be >= 1, got {rt_size}")
    if window < 0:
        raise ValueError(f"window must be >= 0 (0 = channel-aligned), got {window}")

    out = w_2d.detach().clone()
    if out.numel() == 0:
        return out, 0

    n_rows, n_cols = out.shape
    span = n_cols if window == 0 else window * rt_size
    flips = 0
    for r in range(n_rows):
        for start in range(0, n_cols, span):
            stop = min(start + span, n_cols)
            if stop - start < rt_size:
                # Only reachable multiple is 0 => the whole window would have to
                # go negative. Leave it; report nothing.
                continue
            seg = out[r, start:stop]
            pos = seg > 0
            p = int(pos.sum())
            excess = p % rt_size
            if excess == 0:
                continue
            deficit = rt_size - excess
            if excess <= deficit:
                # Cheaper to shed positives: flip the smallest of them.
                give, take_from = excess, pos
            else:
                # Cheaper to recruit negatives up to the next multiple.
                give, take_from = deficit, ~pos
            idx = take_from.nonzero(as_tuple=True)[0]
            chosen = idx[seg[idx].abs().argsort()[:give]]
            seg[chosen] = -seg[chosen]
            flips += int(chosen.numel())
    return out, flips


def align_model_for_ppm(
    model: torch.nn.Module,
    *,
    rt_size: int,
    base_layout: str = "COL",
    window: int = 0,
    kernel_mapping: str = "ROW",
    include_protected: bool = False,
) -> dict:
    """Align every quantized layer's weights in place; return a per-layer report.

    Protected layers are skipped by default, matching
    ``losses.run_length_penalty``: their weights can never be hit by a fault, so
    flipping them costs accuracy for no robustness gain (they do still carry
    padding overhead — pass ``include_protected=True`` to buy that back too).

    Args:
        model:          Any module tree; quantized conv/linear layers are found
                        by type.
        rt_size:        Bits per racetrack.
        base_layout:    ``ROW`` or ``COL`` — the view PPM sorts within. The
                        alignment is layout-specific: a checkpoint aligned for
                        COL is *not* aligned for ROW.
        window:         Racetracks per sort window (``0`` = channel-aligned).
        kernel_mapping: Conv kernel permutation, as in ``storage.kernel_mapping``.
        include_protected: Also align protected layers.

    Returns:
        ``{layer_name: {"flips": int, "weights": int, "flip_frac": float}}`` for
        every layer that was touched.
    """
    from netdrift.faults.layout import _layout_weight_for_racetrack
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    layout = (base_layout or "COL").upper()
    km = (kernel_mapping or "ROW").upper()
    report: dict[str, dict] = {}

    for name, module in model.named_modules():
        if not isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            continue
        if not include_protected and getattr(module, "protected", False):
            continue
        weight = module.weight
        if weight.numel() == 0:
            continue
        w_2d, undo = _layout_weight_for_racetrack(
            weight.detach(), rt_mapping=layout,
            kernel_mapping=(km if weight.dim() == 4 else None),
        )
        aligned, flips = align_windows_to_multiple(w_2d, rt_size, window=window)
        # How deep into the |w| distribution the flips cut: a weight at the 2nd
        # percentile is nearly free for a BNN to give up, one at the 40th is not.
        # The flip COUNT alone says nothing about the accuracy risk.
        flipped = (aligned > 0) != (w_2d > 0)
        pctl_mean = pctl_max = 0.0
        if bool(flipped.any()):
            mags = w_2d.abs()
            ranks = (torch.searchsorted(mags.flatten().sort().values,
                                        mags[flipped].sort().values).to(torch.float32)
                     / mags.numel() * 100.0)
            pctl_mean, pctl_max = float(ranks.mean()), float(ranks.max())
        with torch.no_grad():
            weight.copy_(undo(aligned).to(weight.dtype))
        key = getattr(module, "layer_name", None) or name
        report[key] = {
            "flips": flips,
            "weights": int(weight.numel()),
            "flip_frac": (flips / weight.numel()) if weight.numel() else 0.0,
            "flip_abs_pctl_mean": pctl_mean,
            "flip_abs_pctl_max": pctl_max,
        }
    return report
