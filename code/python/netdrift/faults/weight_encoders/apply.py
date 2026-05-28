"""Apply a WeightEncoder to every quantized layer in a model.

Used in ``mode="once"`` to rewrite the stored weights of a model on disk —
the encoder runs after checkpoint load and before any fault simulation, and
the result is treated as the new persistent state of the model (saveable to
a checkpoint and reloadable).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from numba import cuda

from netdrift.faults.rtm_misalignment import (
    _layout_weight_for_racetrack,
)
from netdrift.faults.weight_encoders.base import WeightEncoder


_ENDLEN_FILENAME_MARKER = re.compile(r"(^|[_./-])endlen(\.|_|-|$)", re.IGNORECASE)


def is_encoded_checkpoint_path(path: str | Path | None) -> bool:
    """Return True iff ``path`` looks like a post-encoder checkpoint.

    Detection is by filename convention: the basename must contain the token
    ``endlen`` bounded by a separator (``_``, ``-``, ``.``, ``/``) or string
    edge. Case-insensitive.

    Examples of matches: ``model_endlen.pt``, ``vgg7_endlen_w1a4.pt``,
    ``ENDLEN-model.pt``, ``/runs/foo/endlen.pt``.

    Non-matches: ``model.pt``, ``model_endless.pt``, ``bnn_model.pt``.
    """
    if path is None:
        return False
    name = Path(str(path)).name
    return bool(_ENDLEN_FILENAME_MARKER.search(name))


def with_endlen_marker(path: str | Path) -> str:
    """Ensure ``path``'s basename contains the ``_endlen`` marker.

    If the basename already matches :func:`is_encoded_checkpoint_path`, the
    input is returned unchanged. Otherwise ``_endlen`` is inserted before
    the final suffix (e.g. ``foo.pt`` → ``foo_endlen.pt``).
    """
    p = Path(str(path))
    if is_encoded_checkpoint_path(p):
        return str(p)
    new_name = f"{p.stem}_endlen{p.suffix}"
    return str(p.with_name(new_name))


def _is_unbudgeted(budget) -> bool:
    """True when no budget binds (fast path): both budgets >= 1.0, or None."""
    if budget is None:
        return True
    return budget.global_budget >= 1.0 and budget.local_budget >= 1.0


def _encodable_layers(model):
    """Yield ``(name, module)`` for layers the encoder should touch.

    Skips non-quantized layers, layers with no bound scheme, protected layers,
    and empty-weight layers — matching the original walk's filters.
    """
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
    for name, module in model.named_modules():
        if not isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            continue
        if getattr(module, "scheme", None) is None:
            continue
        if getattr(module, "protected", False):
            continue
        if module.weight.data.numel() == 0:
            continue
        yield name, module


def _q_pre_and_layout(module, *, rt_mapping: str, kernel_mapping_default: str):
    """Return ``(q_pre, w_2d, undo, original_shape)`` for one layer.

    ``q_pre`` is the ±1 binarization of the latent weight; ``w_2d`` is its
    racetrack-aligned 2D view; ``undo`` maps a 2D view back to ``original_shape``.
    """
    weight = module.weight.data
    qt = module.scheme.quantize(weight, getattr(module, "scale_per_channel", None))
    q_pre = qt.values.detach().clone()
    original_shape = tuple(q_pre.shape)
    kernel_mapping = kernel_mapping_default if q_pre.dim() == 4 else None
    w_2d, undo = _layout_weight_for_racetrack(
        q_pre, rt_mapping=rt_mapping, kernel_mapping=kernel_mapping
    )
    return q_pre, w_2d, undo, original_shape


def apply_weight_encoder_to_model(
    model: torch.nn.Module,
    encoder: WeightEncoder,
    *,
    rt_size: int,
    rt_mapping: str,
    kernel_mapping_default: str = "ROW",
    budget=None,
) -> dict[str, dict]:
    """Run ``encoder`` on every unprotected quantized layer's weights, in place.

    Walks ``model.named_modules()`` looking for ``QuantizedConv2d`` and
    ``QuantizedLinear`` layers. For each, applies the same reshape pipeline that
    :class:`RTMMisalignmentFault.inject` uses (kernel-mapping permute, then
    rt_mapping transpose), encodes the racetrack-aligned ±1 view, and propagates
    the sign changes back into the latent FP weight.

    When ``budget`` is ``None`` (or both its budgets are ``>= 1.0``), the fast
    path runs the encoder in place per layer — bit-identical to the original,
    unbudgeted behavior. Otherwise the budgeted path is used: emit candidate
    merges across **all** layers, select a budget-bounded subset
    (:func:`~netdrift.faults.weight_encoders.budget.select_merges`), and apply
    only the chosen spans. The global budget therefore spans the whole model.

    Args:
        model:                  The model. Already quantized (layers swapped).
        encoder:                A :class:`WeightEncoder` instance.
        rt_size:                Bits per racetrack.
        rt_mapping:             ``"ROW"`` or ``"COL"``.
        kernel_mapping_default: Kernel mapping for conv layers. Default ``"ROW"``.
        budget:                 Optional
                                :class:`~netdrift.faults.weight_encoders.budget.BudgetConfig`.

    Returns:
        ``{layer_name: {"flipped": int, "rejected": int, "fraction": float}}``.
        ``rejected`` is the number of candidate flips a binding budget dropped
        (always ``0`` on the fast path); ``fraction`` is ``flipped / numel``.
    """
    if _is_unbudgeted(budget):
        return _apply_unbudgeted(
            model, encoder, rt_size=rt_size, rt_mapping=rt_mapping,
            kernel_mapping_default=kernel_mapping_default,
        )
    return _apply_budgeted(
        model, encoder, rt_size=rt_size, rt_mapping=rt_mapping,
        kernel_mapping_default=kernel_mapping_default, budget=budget,
    )


def _apply_unbudgeted(
    model, encoder, *, rt_size, rt_mapping, kernel_mapping_default,
) -> dict[str, dict]:
    """Original in-place per-layer encode. Bit-identical to legacy behavior."""
    report: dict[str, dict] = {}
    for name, module in _encodable_layers(model):
        q_pre, w_2d, undo, original_shape = _q_pre_and_layout(
            module, rt_mapping=rt_mapping, kernel_mapping_default=kernel_mapping_default
        )
        w_np = w_2d.detach().cpu().numpy()
        w_gpu = cuda.to_device(w_np)
        encoder.apply(w_gpu, rt_size)
        cuda.synchronize()
        w_2d_new = torch.from_numpy(w_gpu.copy_to_host()).to(
            q_pre.device, dtype=q_pre.dtype
        )
        q_post = undo(w_2d_new).reshape(original_shape).contiguous()
        flips_mask = q_post != q_pre
        if flips_mask.any():
            module.weight.data[flips_mask] = -module.weight.data[flips_mask]
        flipped = int(flips_mask.sum().item())
        report[name] = {
            "flipped": flipped,
            "rejected": 0,
            "fraction": flipped / q_pre.numel() if q_pre.numel() else 0.0,
        }
    return report


def _apply_budgeted(
    model, encoder, *, rt_size, rt_mapping, kernel_mapping_default, budget,
) -> dict[str, dict]:
    """Emit candidates across all layers, select under budget, apply chosen spans.

    Uses the endlen emit kernel to record candidate merges per layer, accumulates
    the per-unit totals, runs the host-side selector once across the whole model
    (so the global budget spans layers), then applies the surviving spans by
    building a scatter mask in the 2D view and mapping it back to original
    coordinates via ``undo`` — never inverting the layout math by hand.
    """
    import numpy as np

    from netdrift.faults.weight_encoders.budget import select_merges
    from netdrift.faults.weight_encoders.endlen import emit_candidates_gpu

    # Pass 1 — emit candidates and totals for every layer; cache per-layer
    # tensors so the apply pass doesn't recompute the layout.
    layer_order: list[str] = []
    cache: dict[str, dict] = {}
    all_candidates = []
    layer_totals: dict[int, int] = {}
    racetrack_totals: dict[tuple[int, int], int] = {}
    channel_totals: dict[tuple[int, int], int] = {}
    emitted_by_layer: dict[int, int] = {}

    for layer_idx, (name, module) in enumerate(_encodable_layers(model)):
        q_pre, w_2d, undo, original_shape = _q_pre_and_layout(
            module, rt_mapping=rt_mapping, kernel_mapping_default=kernel_mapping_default
        )
        w_np = w_2d.detach().cpu().numpy().astype(np.float32, copy=False)
        # Latent FP weight laid out the same way (for min_latent_magnitude).
        lat_2d, _ = _layout_weight_for_racetrack(
            module.weight.data.detach(),
            rt_mapping=rt_mapping,
            kernel_mapping=(kernel_mapping_default if q_pre.dim() == 4 else None),
        )
        lat_np = lat_2d.detach().cpu().numpy().astype(np.float32, copy=False)

        cands = emit_candidates_gpu(w_np, lat_np, rt_size=rt_size, layer_idx=layer_idx)
        all_candidates.extend(cands)
        emitted_by_layer[layer_idx] = sum(c.n_flips for c in cands)

        rows, cols = w_2d.shape
        n_rt = cols // rt_size  # floor: matches emit kernel coverage
        layer_totals[layer_idx] = rows * cols
        for i in range(rows):
            channel_totals[(layer_idx, i)] = cols
            for j in range(n_rt):
                track_end = min((j + 1) * rt_size, cols)
                racetrack_totals[(layer_idx, i * n_rt + j)] = track_end - j * rt_size

        layer_order.append(name)
        cache[name] = dict(
            module=module, layer_idx=layer_idx, q_pre=q_pre, w_2d_shape=(rows, cols),
            undo=undo, original_shape=original_shape,
        )

    # Pass 2 — select once across the whole model.
    chosen = select_merges(
        all_candidates, budget,
        layer_totals=layer_totals,
        racetrack_totals=racetrack_totals,
        channel_totals=channel_totals,
    )
    chosen_by_layer: dict[int, list] = {}
    for c in chosen:
        chosen_by_layer.setdefault(c.layer_idx, []).append(c)

    # Pass 3 — apply chosen spans per layer via a scatter mask.
    report: dict[str, dict] = {}
    for name in layer_order:
        info = cache[name]
        module = info["module"]
        q_pre = info["q_pre"]
        rows, cols = info["w_2d_shape"]
        layer_idx = info["layer_idx"]

        applied = chosen_by_layer.get(layer_idx, [])
        emitted = emitted_by_layer.get(layer_idx, 0)
        applied_flips = sum(c.n_flips for c in applied)

        if applied:
            flat_mask = torch.zeros(rows * cols, dtype=torch.bool)
            for c in applied:
                flat_mask[c.start_idx:c.start_idx + c.n_flips] = True
            mask_2d = flat_mask.reshape(rows, cols)
            mask_orig = info["undo"](mask_2d).reshape(info["original_shape"])
            mask_orig = mask_orig.to(module.weight.device).bool()
            module.weight.data[mask_orig] = -module.weight.data[mask_orig]

        report[name] = {
            "flipped": int(applied_flips),
            "rejected": int(emitted - applied_flips),
            "fraction": applied_flips / q_pre.numel() if q_pre.numel() else 0.0,
        }
    return report
