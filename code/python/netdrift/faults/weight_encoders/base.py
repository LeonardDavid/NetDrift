"""WeightEncoder interface and registry.

A :class:`WeightEncoder` is a write-time transformation applied to the
quantized weight tensor that lives on a racetrack. Unlike
:class:`netdrift.faults.mitigations.MitigationStep`, which adjusts the per-RT
``index_offset`` array between fault generation and read-out, a weight encoder
rewrites the *stored* values themselves. Endlen (block-hypothesis) is the
first instance.

The encoder is invoked on the 2D weight view that has already been
transposed (for ``rt_mapping="COL"``) and kernel-permuted (for non-ROW
kernel mappings), so racetracks are aligned along the row dimension. The
caller is responsible for undoing those reshapes after the encoder returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class WeightEncoder(ABC):
    """A write-time, in-place transformation of a 2D quantized weight tensor.

    Stateless; instances are safe to share across layers and call repeatedly.
    """

    name: str

    @abstractmethod
    def apply(self, weight_2d_gpu, rt_size: int) -> None:
        """Transform ``weight_2d_gpu`` in place.

        Args:
            weight_2d_gpu: Numba CUDA device array of shape ``(N, M)``,
                with ``M`` divisible into racetracks of ``rt_size`` bits
                along the row dimension. Values are ``+1`` / ``-1``.
            rt_size:       Bits per racetrack.
        """


_REGISTRY: dict[str, Callable[..., WeightEncoder]] = {}


def register_encoder(
    name: str,
) -> Callable[[type[WeightEncoder]], type[WeightEncoder]]:
    """Decorator to register a weight encoder by name."""

    def deco(cls: type[WeightEncoder]) -> type[WeightEncoder]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_encoder(name: str, **kwargs) -> WeightEncoder:
    """Look up and instantiate a weight encoder by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown weight encoder '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)
