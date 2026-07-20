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
from typing import Optional

import numpy as np
import torch
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from netdrift.faults.base import FaultCtx, FaultModel, FaultState, FaultStats
from netdrift.faults.kernels import (
    calc_index_offset_kernel,
    simulate_racetrack_kernel,
)
from netdrift.faults.layout import (  # re-export for back-compat
    _ap_reads_for_mapping,
    _layout_weight_for_racetrack,
    compute_index_offset_shape,
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
    track_wrong_reads: bool = False
    """If ``True``, count per-racetrack non-identity reads (wrong bits read due
    to persistent misalignment) in the simulate kernel and surface the scalar
    total via ``FaultStats.extra['wrong_bits_read']``."""
    weight_encoder: Optional["WeightEncoder"] = None
    """Optional write-time weight encoder (e.g. endlen). ``None`` disables it."""
    weight_encoder_mode: str = "once"
    """``"once"`` (encoder fires once in the runner) or ``"per_forward"``
    (encoder fires inside :meth:`RTMMisalignmentFault.inject` on every call).
    Ignored when ``weight_encoder is None``."""
    block_mapping: bool = False
    """Set True when storage.layout == 'block'. Used only to reject the
    per_forward encoder combination at config-construction time."""
    edge_mode: str = "saturate"
    """Racetrack edge model. ``"saturate"`` (default): the access port is fixed
    at ``ap_position`` and reads saturate to the nearest real cell — no random
    values reach the network. ``"random"``: legacy behaviour, out-of-bounds
    reads return a random ±1 (kept for A/B comparison)."""
    ap_position: Optional[int] = None
    """Fixed access-port index in ``[0, rt_size-1]`` for ``edge_mode="saturate"``.
    ``None`` resolves per (effective) racetrack length to ``rt_size//2 - 1`` (the
    first middle position). Must be ``None`` for BLOCK mapping, where each bucket
    has its own padded length ``P`` and the port is resolved per bucket."""

    def __post_init__(self) -> None:
        if self.weight_encoder_mode not in ("once", "per_forward"):
            raise ValueError(
                f"weight_encoder_mode must be 'once' or 'per_forward', "
                f"got {self.weight_encoder_mode!r}"
            )
        if self.edge_mode not in ("saturate", "random"):
            raise ValueError(
                f"edge_mode must be 'saturate' or 'random', got {self.edge_mode!r}"
            )
        if self.ap_position is not None:
            if int(self.ap_position) < 0:
                raise ValueError(
                    f"ap_position must be >= 0, got {self.ap_position}"
                )
            if self.block_mapping:
                # A single absolute AP index is meaningless across heterogeneous
                # per-bucket racetrack lengths; BLOCK resolves the AP per bucket.
                raise ValueError(
                    "ap_position is not supported with BLOCK mapping (each bucket "
                    "has its own padded length P; the access port is resolved per "
                    "bucket as P//2 - 1). Leave ap_position unset for BLOCK."
                )
        if self.weight_encoder is not None and self.rt_size > 64:
            # The endlen kernel uses a hardcoded 64-element local buffer.
            # If we ever add an encoder without this limit, gate this check
            # on the encoder type.
            raise ValueError(
                f"weight_encoder requires rt_size <= 64 (legacy kernel "
                f"uses a fixed-size local buffer); got rt_size={self.rt_size}"
            )
        if self.block_mapping and self.weight_encoder_mode == "per_forward" \
                and self.weight_encoder is not None:
            raise ValueError(
                "BLOCK mapping is incompatible with weight_encoder_mode="
                "'per_forward' (block structure is derived from the "
                "post-encoder sign pattern and would go stale). Use "
                "weight_encoder_mode='once' or disable the encoder."
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

    base_mapping: Optional[str] = None
    """For BLOCK: the ROW/COL base layout used to segment before block extraction."""

    block_buckets: Optional[dict] = None
    """For BLOCK: cached {P: BlockBucket} immutable structure (positions/lengths)."""

    block_offsets: Optional[dict] = None
    """For BLOCK: {P: np.ndarray (n_P, 1) int32} persistent misalignment state."""

    last_wrong_read: Optional[np.ndarray] = None
    """Most-recent per-racetrack wrong-read counts (2D int array) or ``None``.
    Populated only when ``track_wrong_reads`` is enabled; used for the .npz dump."""


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

        if rt_mapping == "BLOCK":
            base_mapping = ctx.extra.get("base_layout")
            if base_mapping is None:
                raise ValueError("BLOCK mapping requires ctx.extra['base_layout']")
            # Structure is built lazily on first inject (needs the weight tensor).
            return RTMState(
                index_offset=np.zeros((1, 1), dtype=np.int32),  # unused for BLOCK
                rt_mapping="BLOCK",
                kernel_mapping=kernel_mapping,
                base_mapping=base_mapping.upper(),
            )

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

        if state.rt_mapping == "BLOCK":
            return self._run_block_path(weight, state, ctx, pre_fault)

        # 1+2) Reshape into the racetrack-aligned 2D view.
        w_2d, undo_layout = _layout_weight_for_racetrack(
            weight, rt_mapping=state.rt_mapping, kernel_mapping=state.kernel_mapping
        )

        # 3) Run the racetrack simulation kernels
        ap_reads = _ap_reads_for_mapping(self.cfg.rt_size, state.rt_mapping)
        new_w_2d_np, new_offset, total_misalign, wrong_read = self._run_rtm_kernels(
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
        if self.cfg.track_wrong_reads:
            stats.extra["wrong_bits_read"] = int(wrong_read.sum())

        new_state = RTMState(
            index_offset=new_offset,
            rt_mapping=state.rt_mapping,
            kernel_mapping=state.kernel_mapping,
            last_wrong_read=(wrong_read if self.cfg.track_wrong_reads else None),
        )
        return new_w, new_state, stats

    def _run_block_path(self, weight, state, ctx, pre_fault):
        """BLOCK-mapping fault injection: run kernels per bucket, scatter back.

        Segments the weight into a ROW/COL base view (``state.base_mapping``),
        splits it into contiguous same-sign blocks (padded to a power of two
        <= 64), groups blocks by padded length ``P`` into dense per-bucket
        grids, and runs the existing RTM kernels once per bucket (with
        ``rt_size=P``). Results are scattered back to their original
        ``(row, col)`` positions in the base-2D view; padding cells are
        discarded. The block structure (positions/lengths) is cached on
        ``state.block_buckets`` across calls since it is only a function of
        the weight's sign pattern (fixed after training in ``once`` encoder
        mode); only the per-bucket ``index_offset`` (``block_offsets``)
        evolves across calls.
        """
        from netdrift.faults.layout import (
            _layout_weight_for_racetrack, build_block_buckets,
        )
        # 1) base layout (ROW/COL), kernel-permuted for convs
        w_2d, undo_base = _layout_weight_for_racetrack(
            weight, rt_mapping=state.base_mapping, kernel_mapping=state.kernel_mapping,
        )
        # 2) build (or reuse) the immutable block structure
        buckets = state.block_buckets
        if buckets is None:
            buckets = build_block_buckets(w_2d, self.cfg.rt_size)
            offsets = {p: np.zeros((b.weight_grid.shape[0], 1), dtype=np.int32)
                       for p, b in buckets.items()}
        else:
            offsets = state.block_offsets

        w_out_2d = w_2d.detach().cpu().float().numpy().copy()

        total_misalign = 0
        total_wrong = 0
        affected = 0
        cuda.select_device(int(os.environ.get("NUMBA_CUDA_DEFAULT_DEVICE", "0")))

        for p in sorted(buckets):
            bucket = buckets[p]
            off = offsets[p]
            # Re-read current real-cell values from w_2d each call (values change,
            # positions do not). Padding keeps the cached block-sign guard band.
            grid = bucket.weight_grid.copy()
            rmask = bucket.scatter_cols >= 0
            # scatter/gather uses cached (row,col); safe because structure is fixed
            gr = bucket.scatter_rows[rmask]
            gc = bucket.scatter_cols[rmask]
            grid[rmask] = w_2d.detach().cpu().float().numpy()[gr, gc]

            new_grid, new_off, tm, wrong = self._run_rtm_kernels(
                torch.from_numpy(grid), off, ap_reads=p, nr_run=ctx.nr_run, rt_size=p,
            )
            offsets[p] = new_off
            total_misalign += int(tm)
            total_wrong += int(wrong.sum())
            affected += int(np.count_nonzero(new_off))

            # scatter real cells back; discard padding
            w_out_2d[gr, gc] = new_grid[rmask]

        new_w_2d = torch.from_numpy(w_out_2d).to(weight.device, dtype=weight.dtype)
        new_w = undo_base(new_w_2d)

        stats = FaultStats()
        if self.cfg.track_misalign_faults:
            stats.misalign_faults = total_misalign
        if self.cfg.track_affected_units:
            stats.affected_units = affected
        if self.cfg.track_bitflips and pre_fault is not None:
            stats.bitflips = int((pre_fault != new_w).sum().item())
        if self.cfg.track_wrong_reads:
            stats.extra["wrong_bits_read"] = total_wrong

        new_state = RTMState(
            index_offset=np.zeros((1, 1), dtype=np.int32),
            rt_mapping="BLOCK",
            kernel_mapping=state.kernel_mapping,
            base_mapping=state.base_mapping,
            block_buckets=buckets,
            block_offsets=offsets,
        )
        return new_w, new_state, stats

    def _run_rtm_kernels(
        self,
        weight_2d: torch.Tensor,
        index_offset: np.ndarray,
        ap_reads: int,
        nr_run: int,
        rt_size: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Drive the two CUDA kernels and apply mitigations between them.

        Args:
            rt_size: Racetrack length used for the two kernel launches. Defaults
                to ``self.cfg.rt_size`` (the ROW/COL case, one shared racetrack
                length for the whole layer). BLOCK callers pass the per-bucket
                padded length ``P`` instead, since each bucket is itself a batch
                of same-length racetracks that generally differ from the
                config's nominal ``rt_size``.

        Returns ``(new_weight_2d_np, new_index_offset, total_misalign_count, wrong_read)``.
        ``wrong_read`` is the per-racetrack non-identity-read count array when
        ``track_wrong_reads`` is on, else a ``(1, 1)`` zero array.
        """
        rt_size = self.cfg.rt_size if rt_size is None else rt_size

        # Resolve the edge model and fixed access-port position for this launch.
        # ``ap`` is resolved against the *effective* rt_size (which is the
        # per-bucket padded length ``P`` on the BLOCK path), so a P=1 bucket
        # yields ap=0 (lo==hi==0 -> offset frozen -> length-1 blocks are safe).
        edge_mode = 1 if self.cfg.edge_mode == "saturate" else 0
        ap = self.cfg.ap_position if self.cfg.ap_position is not None else rt_size // 2 - 1
        if ap < 0:
            ap = 0
        if ap > rt_size - 1:
            ap = rt_size - 1

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
            rng, offset_gpu, misalign_gpu, rt_size, ap_reads, self.cfg.rt_error,
            ap, edge_mode,
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

        # Per-racetrack wrong-read counter — full grid shape only when tracking
        # is on; a (1,1) dummy otherwise. The kernel gates on the explicit
        # ``track_wrong`` flag, NOT the array shape, so a genuine (1,1) grid is
        # still counted.
        track_wrong = 1 if self.cfg.track_wrong_reads else 0
        if self.cfg.track_wrong_reads:
            wrong_read = np.zeros_like(index_offset)
        else:
            wrong_read = np.zeros((1, 1), dtype=index_offset.dtype)
        wrong_gpu = cuda.to_device(wrong_read)

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
            rng, weight_in_gpu, weight_out_gpu, offset_gpu, rt_size,
            wrong_gpu, track_wrong, edge_mode,
        )
        cuda.synchronize()
        weight_out_np = weight_out_gpu.copy_to_host()
        wrong_read = wrong_gpu.copy_to_host()

        total_misalign = int(misalign_faults.sum()) if self.cfg.track_misalign_faults else 0
        return weight_out_np, index_offset, total_misalign, wrong_read
