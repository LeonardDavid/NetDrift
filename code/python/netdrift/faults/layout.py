"""Racetrack storage-layout helpers (numba-free).

Single source of truth for how a layer's weight tensor maps onto nanowire
racetracks. Shared by the RTM fault model, the endlen encoder, and the
run-length regularizer so all three agree on the layout — and so a future
``interleaved`` mapping is a one-place change here.

This module deliberately imports **no** numba/CUDA so it can be used on
CPU-only machines (tests, the differentiable regularizer). ``rtm_misalignment``
re-exports these names to preserve its existing public surface.

Coordinate conventions (matching legacy):

* ``rt_mapping="ROW"``: each row of the (reshaped 2D) weight matrix is laid
  along ``rt_size`` racetracks.
* ``rt_mapping="COL"``: each column is one racetrack; the kernel sees a
  transposed view internally to keep the per-thread layout uniform.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch


def compute_index_offset_shape(
    weight_shape: Sequence[int],
    rt_size: int,
    rt_mapping: str,
    kernel_size: Optional[int] = None,
) -> tuple[int, int]:
    """Determine ``index_offset`` array shape for the given weight + mapping.

    Args:
        weight_shape: Layer weight shape. ``(out, in)`` for linear,
                      ``(out, in, k, k)`` for conv2d.
        rt_size:      Racetrack length in bits.
        rt_mapping:   ``"ROW"`` or ``"COL"``.
        kernel_size:  Kernel side for convs; ignored for linear.

    Returns:
        ``(num_rt_x, num_rt_y)`` integer pair.
    """
    if len(weight_shape) == 2:
        out_dim, in_dim = weight_shape
    elif len(weight_shape) == 4:
        out_dim, in_dim, kh, kw = weight_shape
        if kernel_size is None:
            kernel_size = kh
        if kh != kw:
            raise ValueError(f"non-square conv kernel: {kh}x{kw}")
        in_dim = in_dim * kernel_size * kernel_size
    else:
        raise ValueError(f"weight shape must be 2D or 4D, got {weight_shape}")

    if rt_mapping == "ROW":
        return out_dim, math.ceil(in_dim / rt_size)
    elif rt_mapping == "COL":
        return in_dim, math.ceil(out_dim / rt_size)
    else:
        raise ValueError(f"invalid rt_mapping: {rt_mapping}")


# Kernel-rearrangement indices (3x3 only, matching legacy). Used to interpret
# how the weights of a single 3x3 kernel are laid out along a racetrack.
_KERNEL_INDICES_3X3: dict[str, torch.Tensor] = {
    "ROW": torch.arange(9),
    "COL": torch.tensor([0, 3, 6, 1, 4, 7, 2, 5, 8]),
    "CLW": torch.tensor([0, 1, 2, 5, 8, 7, 6, 3, 4]),
    "ACW": torch.tensor([0, 3, 6, 7, 8, 5, 2, 1, 4]),
}
_REVERSE_KERNEL_INDICES_3X3: dict[str, torch.Tensor] = {
    "ROW": torch.arange(9),
    "COL": torch.tensor([0, 3, 6, 1, 4, 7, 2, 5, 8]),
    "CLW": torch.tensor([0, 1, 2, 7, 8, 3, 6, 5, 4]),
    "ACW": torch.tensor([0, 7, 6, 1, 8, 5, 2, 3, 4]),
}


def _rearrange_kernel(weight: torch.Tensor, mapping: str) -> torch.Tensor:
    """Apply the kernel-mapping permutation to a 4D conv weight tensor.

    The legacy code only supports 3×3 kernels for the non-ROW mappings; we
    preserve that constraint and pass through unchanged for ROW.
    """
    if mapping == "ROW":
        return weight
    if weight.shape[-1] != 3 or weight.shape[-2] != 3:
        raise NotImplementedError(
            f"kernel_mapping={mapping} only supported for 3x3 kernels (got {weight.shape})"
        )
    out_c, in_c, h, w = weight.shape
    order = _KERNEL_INDICES_3X3[mapping].to(weight.device)
    flat = weight.reshape(-1, h * w)
    return flat[:, order].reshape(out_c, in_c, h * w)  # 3D for the racetrack-sim view


def _restore_kernel(
    weight: torch.Tensor, mapping: str, original_shape: tuple[int, ...]
) -> torch.Tensor:
    """Undo :func:`_rearrange_kernel`, returning a tensor of the original shape."""
    if mapping == "ROW":
        return weight.reshape(original_shape)
    rev = _REVERSE_KERNEL_INDICES_3X3[mapping].to(weight.device)
    out_c, in_c = original_shape[0], original_shape[1]
    return weight.reshape(out_c, in_c, -1)[..., rev].reshape(original_shape)


def _layout_weight_for_racetrack(
    weight: torch.Tensor,
    rt_mapping: str,
    kernel_mapping: Optional[str],
) -> tuple[torch.Tensor, "callable"]:
    """Reshape ``weight`` into a 2D racetrack-aligned view, plus an undo fn.

    Handles the same chain :meth:`RTMMisalignmentFault.inject` does:

    1. For 4D conv weights, permute kernel entries by ``kernel_mapping``
       and flatten to ``(out_channels, in_channels * kh * kw)``.
    2. For 2D linear weights, reshape to ``(out, in)``.
    3. If ``rt_mapping == "COL"``, transpose so the racetrack-aligned axis
       is always row-wise from the kernels' point of view.

    Returns:
        ``(weight_2d, undo)`` where ``undo(weight_2d_new)`` reverses steps
        1–3 and returns a tensor of ``weight``'s original shape.
    """
    original_shape = tuple(weight.shape)
    is_conv = weight.dim() == 4
    km = kernel_mapping or "ROW"

    if is_conv:
        w = _rearrange_kernel(weight, km)
        w_2d = w.reshape(w.size(0), -1)
    else:
        w_2d = weight.reshape(weight.size(0), -1)

    if rt_mapping == "COL":
        w_2d = w_2d.t().contiguous()
    elif rt_mapping != "ROW":
        raise ValueError(f"invalid rt_mapping: {rt_mapping}")

    def undo(new_w_2d: torch.Tensor) -> torch.Tensor:
        if rt_mapping == "COL":
            new_w_2d = new_w_2d.t().contiguous()
        if is_conv:
            return _restore_kernel(new_w_2d, km, original_shape)
        return new_w_2d.reshape(original_shape)

    return w_2d, undo


def _ap_reads_for_mapping(rt_size: int, rt_mapping: str) -> int:
    """Number of access-port reads simulated for one full word read-out."""
    if rt_mapping == "ROW":
        return rt_size * rt_size
    if rt_mapping == "COL":
        return rt_size
    raise ValueError(f"invalid rt_mapping: {rt_mapping}")
