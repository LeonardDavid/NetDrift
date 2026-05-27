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


def apply_weight_encoder_to_model(
    model: torch.nn.Module,
    encoder: WeightEncoder,
    *,
    rt_size: int,
    rt_mapping: str,
    kernel_mapping_default: str = "ROW",
) -> dict[str, int]:
    """Run ``encoder`` on every quantized layer's weight tensor, in place.

    Walks ``model.named_modules()`` looking for ``QuantizedConv2d`` and
    ``QuantizedLinear`` layers. For each, applies the same reshape pipeline
    that :class:`RTMMisalignmentFault.inject` uses (kernel-mapping permute,
    then rt_mapping transpose), runs the encoder on the resulting 2D view,
    and writes the result back into ``module.weight.data``.

    Args:
        model:                  The model. Already quantized (layers swapped).
        encoder:                A :class:`WeightEncoder` instance.
        rt_size:                Bits per racetrack.
        rt_mapping:             ``"ROW"`` or ``"COL"``.
        kernel_mapping_default: Kernel mapping for conv layers. Default ``"ROW"``.

    Returns:
        ``{layer_name: bits_changed}`` — for reporting; the bits-changed
        count is the number of weight entries that differ between
        pre-encode and post-encode.
    """
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    report: dict[str, int] = {}
    for name, module in model.named_modules():
        if not isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            continue
        if getattr(module, "scheme", None) is None:
            # Layer hasn't had a quant scheme bound — skip (e.g. an FP-only
            # baseline run where replace_with_quantized was not called).
            continue
        if getattr(module, "protected", False):
            # Protected layers don't experience fault injection, so encoding
            # their stored weights has no effect — and would inflate the
            # bit-flip stats spuriously. Skip them, matching the protection
            # policy applied to the fault model.
            continue
        weight = module.weight.data
        if weight.numel() == 0:
            continue

        # The encoder operates on the *quantized* ±1 view, not the latent
        # FP master weight. After encoding, we propagate the sign changes
        # back into the latent weight by flipping the sign of any FP entry
        # whose binarized value just got flipped — re-quantizing on the
        # next forward then reproduces the encoded bits while preserving
        # the magnitude information of the latent FP weight.
        qt = module.scheme.quantize(
            weight, getattr(module, "scale_per_channel", None)
        )
        q_pre = qt.values.detach().clone()  # ±1 binarization of the FP weight

        original_shape = tuple(q_pre.shape)
        kernel_mapping = (
            kernel_mapping_default if q_pre.dim() == 4 else None
        )
        w_2d, undo = _layout_weight_for_racetrack(
            q_pre, rt_mapping=rt_mapping, kernel_mapping=kernel_mapping
        )

        # Move to a device array, run the encoder, copy back.
        w_np = w_2d.detach().cpu().numpy()
        w_gpu = cuda.to_device(w_np)
        encoder.apply(w_gpu, rt_size)
        cuda.synchronize()
        w_2d_new = torch.from_numpy(w_gpu.copy_to_host()).to(
            q_pre.device, dtype=q_pre.dtype
        )

        q_post = undo(w_2d_new).reshape(original_shape).contiguous()

        # Flip latent FP entries where the binarized sign changed.
        flips_mask = q_post != q_pre
        if flips_mask.any():
            module.weight.data[flips_mask] = -module.weight.data[flips_mask]

        report[name] = int(flips_mask.sum().item())

    return report
