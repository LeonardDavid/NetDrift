"""Sign purity of racetracks — how many wires can bitflip at all.

Under ``edge_mode=saturate`` a read is clamped to the wire's own data window
(``kernels/rtm_numba.py::simulate_racetrack_kernel``), so a shifted wire whose
**physical** cells all hold the same sign returns the correct value at every
offset. Purity is therefore not a proxy for robustness — it *is* the immunity
criterion, per wire:

    a wire is PURE  <=>  all of its physical cells share a sign
    mixed == 0      <=>  the layer cannot bitflip, at any rt_error

"Physical" includes padding, guard-band and phase-guard cells, because those are
materialised in the weight grid and read like any other cell. Real cells are the
only ones scattered back, so the weight-weighted counts below (``weights_pure`` /
``weights_mixed``) attribute each *weight* to the wire it sits on — the "how many
bits are exposed" number, which differs from the wire fraction because mixed
wires need not be equally full.

Sign convention matches ``layout.extract_blocks``/``BinaryScheme``: ``w > 0 ->
+1``, so an exactly-zero weight is negative.

This module imports no numba/CUDA (like ``layout.py``), so the metric is
computable on CPU-only machines. It lives beside the packers rather than inside
``layout.py`` because it needs ``packing``/``partitioning``, both of which import
from ``layout``.

Cost, measured on a (512, 512, 3, 3) layer (2.36 M weights) at ``rt_size=64``:
dense ~0 s (vectorised), POLARITY ~0.4 s (one pass over the wire plan), BLOCK
~3.9 s (it builds the bucket grids, the same work ``build_block_buckets``
already does elsewhere). Nothing here runs per loop or per ``rt_error`` — the
metric is static, so it is computed once per run and once per snapshot. Callers
that already hold buckets should pass them to :func:`wire_purity_from_buckets`
rather than paying that build twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["WirePurity", "wire_purity", "wire_purity_from_buckets"]


@dataclass(frozen=True)
class WirePurity:
    """Pure/mixed wire counts for one layer (or a sum over layers).

    Attributes:
        pure:          Wires whose physical cells all share a sign.
        mixed:         Wires holding both signs — the only ones that can bitflip.
        weights_pure:  Real weights sitting on pure wires.
        weights_mixed: Real weights sitting on mixed wires.
    """

    pure: int = 0
    mixed: int = 0
    weights_pure: int = 0
    weights_mixed: int = 0

    @property
    def total(self) -> int:
        """Physical wires — equals the layer's ``n_racetracks``."""
        return self.pure + self.mixed

    @property
    def total_weights(self) -> int:
        return self.weights_pure + self.weights_mixed

    @property
    def mixed_frac(self) -> float:
        return (self.mixed / self.total) if self.total else 0.0

    @property
    def pure_frac(self) -> float:
        return (self.pure / self.total) if self.total else 0.0

    @property
    def weights_mixed_frac(self) -> float:
        return (self.weights_mixed / self.total_weights) if self.total_weights else 0.0

    def __add__(self, other: "WirePurity") -> "WirePurity":
        if not isinstance(other, WirePurity):  # pragma: no cover - defensive
            return NotImplemented
        return WirePurity(
            pure=self.pure + other.pure,
            mixed=self.mixed + other.mixed,
            weights_pure=self.weights_pure + other.weights_pure,
            weights_mixed=self.weights_mixed + other.weights_mixed,
        )

    def as_dict(self) -> dict:
        """JSON block: absolute counts AND shares, so consumers never divide."""
        return {
            "pure": int(self.pure),
            "mixed": int(self.mixed),
            "total": int(self.total),
            "mixed_frac": round(self.mixed_frac, 6),
            "pure_frac": round(self.pure_frac, 6),
            "weights_pure": int(self.weights_pure),
            "weights_mixed": int(self.weights_mixed),
            "weights_mixed_frac": round(self.weights_mixed_frac, 6),
        }


def wire_purity_from_buckets(buckets: dict) -> WirePurity:
    """Purity of already-built ``{P: BlockBucket}`` grids (BLOCK/UNITS/POLARITY).

    Measures the materialised grid — the exact array the fault kernels read — so
    padding and guard cells participate. Real cells are selected via
    ``scatter_cols >= 0`` rather than ``length``, because a guarded UNITS wire
    interleaves guard slots among its real cells (see ``BlockBucket``).

    Callers that already hold buckets should use this instead of
    :func:`wire_purity` to avoid rebuilding them.
    """
    import numpy as np

    pure = mixed = w_pure = w_mixed = 0
    for bucket in buckets.values():
        grid = np.asarray(bucket.weight_grid)
        if grid.size == 0:
            continue
        pos = grid > 0
        is_pure = pos.all(axis=1) | (~pos).all(axis=1)
        n_real = (np.asarray(bucket.scatter_cols) >= 0).sum(axis=1)
        pure += int(is_pure.sum())
        mixed += int((~is_pure).sum())
        w_pure += int(n_real[is_pure].sum())
        w_mixed += int(n_real[~is_pure].sum())
    return WirePurity(pure=pure, mixed=mixed, weights_pure=w_pure,
                      weights_mixed=w_mixed)


def _dense_purity(w_2d: "torch.Tensor", rt_size: int) -> WirePurity:
    """ROW/COL: each wire is ``rt_size`` consecutive columns of one row.

    ``w_2d`` is already the laid-out view (the COL transpose happens in
    ``_layout_weight_for_racetrack``), so both mappings segment row-wise. The
    ragged last segment of a row is a wire of its own, judged on the real cells
    it holds — matching the kernel, which clamps to the materialised row width.
    """
    import numpy as np

    pos = (w_2d.detach() > 0).cpu().numpy()
    n_rows, n_cols = pos.shape if pos.ndim == 2 else (0, 0)
    if not n_rows or not n_cols:
        return WirePurity()

    n_full = n_cols // rt_size
    tail = n_cols - n_full * rt_size
    pure = mixed = w_pure = w_mixed = 0

    if n_full:
        full = pos[:, : n_full * rt_size].reshape(n_rows, n_full, rt_size)
        is_pure = full.all(axis=2) | (~full).all(axis=2)
        n_pure = int(is_pure.sum())
        pure += n_pure
        mixed += n_rows * n_full - n_pure
        w_pure += n_pure * rt_size
        w_mixed += (n_rows * n_full - n_pure) * rt_size
    if tail:
        t = pos[:, n_full * rt_size:]
        is_pure = t.all(axis=1) | (~t).all(axis=1)
        n_pure = int(is_pure.sum())
        pure += n_pure
        mixed += n_rows - n_pure
        w_pure += n_pure * tail
        w_mixed += (n_rows - n_pure) * tail
    return WirePurity(pure=pure, mixed=mixed, weights_pure=w_pure,
                      weights_mixed=w_mixed)


def _polarity_purity(
    w_2d: "torch.Tensor", rt_size: int, window: int, pad: bool
) -> WirePurity:
    """POLARITY: measured from the wire plan, without materialising grids.

    Padded PPM is sign-pure by construction, but this counts it rather than
    assuming it — a planner regression must show up in the reported metric, not
    only in the test suite. Filler cells carry ``wire["sign"]``, so a partially
    filled wire is pure only when that sign matches its real cells.
    """
    import numpy as np

    from netdrift.faults.partitioning import polarity_wire_plan

    pos = (w_2d.detach() > 0).cpu().numpy()
    pure = mixed = w_pure = w_mixed = 0
    for wire in polarity_wire_plan(w_2d, rt_size, window=window, pad=pad):
        cells = pos[np.asarray(wire["rows"]), np.asarray(wire["cols"])]
        all_pos, all_neg = bool(cells.all()), bool((~cells).all())
        is_pure = all_pos or all_neg
        if is_pure and wire["length"] < rt_size:
            # Filler present: it stores the wire's sign, which must agree.
            is_pure = (wire["sign"] > 0) == all_pos
        if is_pure:
            pure += 1
            w_pure += wire["length"]
        else:
            mixed += 1
            w_mixed += wire["length"]
    return WirePurity(pure=pure, mixed=mixed, weights_pure=w_pure,
                      weights_mixed=w_mixed)


def wire_purity(
    w_2d: "torch.Tensor",
    rt_size: int,
    rt_mapping: str,
    *,
    units_params: Optional[tuple] = None,
    polarity_params: Optional[tuple] = None,
) -> WirePurity:
    """Count pure vs mixed wires for one layer's laid-out view.

    Args:
        w_2d:            The racetrack-aligned 2D view. For BLOCK/UNITS/POLARITY
                         this is the **base-layout** view (ROW or COL), the same
                         convention ``compute_static_metrics`` uses.
        rt_size:         Bits per racetrack.
        rt_mapping:      ``ROW`` / ``COL`` / ``BLOCK`` / ``UNITS`` / ``POLARITY``.
        units_params:    ``(threshold, max_period, pool_guard)``; required for UNITS.
        polarity_params: ``(window, pad)``; required for POLARITY.

    Raises:
        ValueError: on an unknown mapping, ``rt_size < 1``, or missing params for
            UNITS/POLARITY. Missing params are fatal rather than defaulted for the
            same reason as in ``compute_static_metrics``: reporting a dense
            purity for a packed layout is wrong data, not missing data.
    """
    if rt_size < 1:
        raise ValueError(f"rt_size must be >= 1, got {rt_size}")

    mapping = (rt_mapping or "ROW").upper()
    if mapping in ("ROW", "COL"):
        return _dense_purity(w_2d, rt_size)
    if mapping == "BLOCK":
        from netdrift.faults.layout import build_block_buckets
        return wire_purity_from_buckets(build_block_buckets(w_2d, rt_size))
    if mapping == "UNITS":
        if units_params is None:
            raise ValueError(
                "wire_purity: rt_mapping='UNITS' requires units_params="
                "(threshold, max_period, pool_guard); refusing to report dense "
                "purity for a packed layout."
            )
        from netdrift.faults.packing import build_unit_buckets
        threshold, max_period, pool_guard = units_params
        return wire_purity_from_buckets(build_unit_buckets(
            w_2d, rt_size, threshold=int(threshold), max_period=int(max_period),
            pool_guard=int(pool_guard),
        ))
    if mapping == "POLARITY":
        if polarity_params is None:
            raise ValueError(
                "wire_purity: rt_mapping='POLARITY' requires polarity_params="
                "(window, pad); refusing to report dense purity for a packed "
                "layout."
            )
        window, pad = polarity_params
        return _polarity_purity(w_2d, rt_size, int(window), bool(pad))
    raise ValueError(f"invalid rt_mapping for wire_purity: {rt_mapping!r}")
