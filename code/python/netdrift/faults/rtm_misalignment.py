"""Racetrack-memory domain-wall misalignment fault model.

Models domain walls along nanowire racetracks shifting unpredictably under
read access. Each access-port read may misalign the wire by ±1 with
probability ``rt_error``; the cumulative offset is tracked per racetrack and
applied when the layer's weights are read back.

Wraps the legacy Numba CUDA kernels (relocated to
:mod:`netdrift.faults.kernels.rtm_numba`) and the legacy mitigation logic
(relocated to :mod:`netdrift.faults.mitigations`). The numerical behaviour at
a fixed RNG seed is identical to the pre-refactor ``racetrack_sim``.

Coordinate conventions (matching legacy):

* ``rt_mapping="ROW"``: each row of the (reshaped 2D) weight matrix is laid
  along ``rt_size`` racetracks; reading one word costs ``rt_size`` shifts.
* ``rt_mapping="COL"``: each column is one racetrack; reading one word costs
  one shift. The kernel sees a transposed view internally to keep the
  per-thread layout uniform across mappings.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from netdrift.faults.base import FaultCtx, FaultModel, FaultState, FaultStats
from netdrift.faults.kernels import (
    calc_index_offset_kernel,
    simulate_racetrack_kernel,
)
from netdrift.faults.mitigations.base import MitigationStep

# Forward type used in RTMConfig; imported lazily inside methods that need it
# to avoid a circular import (weight_encoders.apply imports from this module).
if False:  # TYPE_CHECKING-style guard without the import
    from netdrift.faults.weight_encoders.base import WeightEncoder


@dataclass
class RTMConfig:
    """Static configuration of an RTM fault model.

    A single instance is shared across all layers in a model — the per-layer
    state (the cumulative ``index_offset`` array) lives separately in
    :class:`RTMState`.

    Attributes:
        rt_size:     Bits per racetrack.
        rt_error:    Per-read misalignment probability in ``[0, 1]``.
        mitigations: Ordered list of :class:`MitigationStep`\\ s applied
                     between fault generation and read-out. Default empty.
        track_misalign_faults: If ``True``, allocate a per-racetrack fault
                     counter and accumulate misalignment events. Setting this
                     ``False`` is significantly cheaper (no extra allocation
                     or device transfer).
        track_bitflips:        If ``True``, layers compare the pre/post
                     weights to count bit changes after read-out.
        track_affected_units:  If ``True``, layers count racetracks with
                     non-zero offset after each call.
    """

    rt_size: int = 64
    rt_error: float = 0.0
    mitigations: list[MitigationStep] = field(default_factory=list)
    track_misalign_faults: bool = False
    track_bitflips: bool = False
    track_affected_units: bool = False
    weight_encoder: Optional["WeightEncoder"] = None
    """Optional write-time weight encoder (e.g. endlen). ``None`` disables it."""
    weight_encoder_mode: str = "once"
    """``"once"`` (encoder fires once in the runner) or ``"per_forward"``
    (encoder fires inside :meth:`RTMMisalignmentFault.inject` on every call).
    Ignored when ``weight_encoder is None``."""

    def __post_init__(self) -> None:
        if self.weight_encoder_mode not in ("once", "per_forward"):
            raise ValueError(
                f"weight_encoder_mode must be 'once' or 'per_forward', "
                f"got {self.weight_encoder_mode!r}"
            )
        if self.weight_encoder is not None and self.rt_size > 64:
            # The endlen kernel uses a hardcoded 64-element local buffer.
            # If we ever add an encoder without this limit, gate this check
            # on the encoder type.
            raise ValueError(
                f"weight_encoder requires rt_size <= 64 (legacy kernel "
                f"uses a fixed-size local buffer); got rt_size={self.rt_size}"
            )


@dataclass
class RTMState(FaultState):
    """Persistent per-layer RTM state.

    Holds the cumulative ``index_offset`` array. Shape is determined by the
    chosen ``rt_mapping`` and the layer's weight dimensions; once allocated
    it is reused across calls.
    """

    index_offset: np.ndarray
    """2D int32 array, shape depends on ``rt_mapping``."""

    rt_mapping: str
    """``"ROW"`` or ``"COL"``."""

    kernel_mapping: Optional[str] = None
    """For conv layers: ``"ROW"``, ``"COL"``, ``"CLW"``, or ``"ACW"``. ``None`` for linear."""


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


class RTMMisalignmentFault(FaultModel):
    """RTM domain-wall misalignment fault model.

    A single :class:`RTMConfig` is shared across all layers; per-layer state
    (``index_offset``) is allocated lazily on the first :meth:`inject` call.
    """

    name = "rtm_misalignment"

    def __init__(self, cfg: RTMConfig) -> None:
        self.cfg = cfg

    def init_state(self, weight_shape: tuple[int, ...], ctx: FaultCtx) -> RTMState:
        rt_mapping = ctx.extra.get("rt_mapping")
        if rt_mapping is None:
            raise ValueError("RTMMisalignmentFault requires ctx.extra['rt_mapping']")
        kernel_mapping = ctx.extra.get("kernel_mapping")  # may be None for linear
        kernel_size = ctx.extra.get("kernel_size")

        shape = compute_index_offset_shape(
            weight_shape=weight_shape,
            rt_size=self.cfg.rt_size,
            rt_mapping=rt_mapping,
            kernel_size=kernel_size,
        )
        return RTMState(
            index_offset=np.zeros(shape, dtype=np.int32),
            rt_mapping=rt_mapping,
            kernel_mapping=kernel_mapping,
        )

    def inject(
        self,
        weight: torch.Tensor,
        state: FaultState,
        ctx: FaultCtx,
    ) -> tuple[torch.Tensor, RTMState, FaultStats]:
        if not isinstance(state, RTMState):
            raise TypeError(f"RTM fault model expected RTMState, got {type(state)}")

        # Bookkeeping for bitflip stats: cache the pre-fault values to compare against later.
        pre_fault: Optional[torch.Tensor] = None
        if self.cfg.track_bitflips:
            pre_fault = weight.detach().clone()

        # 1+2) Reshape into the racetrack-aligned 2D view.
        w_2d, undo_layout = _layout_weight_for_racetrack(
            weight, rt_mapping=state.rt_mapping, kernel_mapping=state.kernel_mapping
        )

        # 3) Run the racetrack simulation kernels
        ap_reads = _ap_reads_for_mapping(self.cfg.rt_size, state.rt_mapping)
        new_w_2d_np, new_offset, total_misalign = self._run_rtm_kernels(
            w_2d.detach(), state.index_offset, ap_reads, ctx.nr_run,
        )
        new_w_2d = torch.from_numpy(new_w_2d_np).to(weight.device, dtype=weight.dtype)

        # 4+5) Undo COL transpose and kernel mapping.
        new_w = undo_layout(new_w_2d)

        # 6) Compute stats
        stats = FaultStats()
        if self.cfg.track_misalign_faults:
            stats.misalign_faults = int(total_misalign)
        if self.cfg.track_affected_units:
            stats.affected_units = int(np.count_nonzero(new_offset))
        if self.cfg.track_bitflips and pre_fault is not None:
            stats.bitflips = int((pre_fault != new_w).sum().item())

        new_state = RTMState(
            index_offset=new_offset,
            rt_mapping=state.rt_mapping,
            kernel_mapping=state.kernel_mapping,
        )
        return new_w, new_state, stats

    def _run_rtm_kernels(
        self,
        weight_2d: torch.Tensor,
        index_offset: np.ndarray,
        ap_reads: int,
        nr_run: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Drive the two CUDA kernels and apply mitigations between them.

        Returns ``(new_weight_2d_np, new_index_offset, total_misalign_count)``.
        """
        # Match the legacy ``racetrack_sim`` initialization sequence: select
        # device 0 (or the value of NUMBA_CUDA_DEFAULT_DEVICE if set) before
        # any kernel launch. This avoids ``cuda.get_current_device()``, which
        # can segfault when PyTorch already holds the primary context.
        cuda.select_device(int(os.environ.get("NUMBA_CUDA_DEFAULT_DEVICE", "0")))

        # Per-call misalignment counter — full shape only if tracking is on.
        if self.cfg.track_misalign_faults:
            misalign_faults = np.zeros_like(index_offset)
        else:
            misalign_faults = np.zeros((1, 1), dtype=index_offset.dtype)

        # Kernel grid: 32x32 threads/block, blocks chosen to cover index_offset.
        nr, nc = index_offset.shape
        threads = (min(nr, 32), min(nc, 32))
        blocks = (math.ceil(nr / threads[0]), math.ceil(nc / threads[1]))
        rng = create_xoroshiro128p_states(
            threads[0] * threads[1] * blocks[0] * blocks[1],
            seed=random.randint(1, 1000),
        )

        offset_gpu = cuda.to_device(index_offset)
        misalign_gpu = cuda.to_device(misalign_faults)

        calc_index_offset_kernel[blocks, threads](
            rng, offset_gpu, misalign_gpu, self.cfg.rt_size, ap_reads, self.cfg.rt_error,
        )
        cuda.synchronize()
        index_offset = offset_gpu.copy_to_host()
        misalign_faults = misalign_gpu.copy_to_host()

        # Mitigations operate on host arrays, in declared order.
        for step in self.cfg.mitigations:
            index_offset, misalign_faults = step.apply(index_offset, misalign_faults, nr_run)

        # Read out the (possibly mitigated) racetracks.
        weight_np = weight_2d.detach().cpu().numpy().astype(np.float32, copy=False)
        weight_in_gpu = cuda.to_device(weight_np)
        offset_gpu = cuda.to_device(index_offset)
        weight_out_np = np.zeros(weight_np.shape, dtype=weight_np.dtype)
        weight_out_gpu = cuda.to_device(weight_out_np)

        # Per-forward encoder: rewrite the (transposed/kernel-mapped) weight
        # view in place, *before* the racetrack read kernel sees it. Matches
        # legacy ``EXEC_ENDLEN`` ordering in ``racetrack_sim``.
        if (
            self.cfg.weight_encoder is not None
            and self.cfg.weight_encoder_mode == "per_forward"
        ):
            self.cfg.weight_encoder.apply(weight_in_gpu, self.cfg.rt_size)
            cuda.synchronize()

        simulate_racetrack_kernel[blocks, threads](
            rng, weight_in_gpu, weight_out_gpu, offset_gpu, self.cfg.rt_size,
        )
        cuda.synchronize()
        weight_out_np = weight_out_gpu.copy_to_host()

        total_misalign = int(misalign_faults.sum()) if self.cfg.track_misalign_faults else 0
        return weight_out_np, index_offset, total_misalign
