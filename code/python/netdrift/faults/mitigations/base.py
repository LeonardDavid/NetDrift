"""Mitigation step interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class MitigationStep(ABC):
    """A single in-place adjustment of the RTM index-offset array.

    A step is invoked between fault generation and racetrack read-out. It
    returns the modified offset array along with the cumulative number of
    *additional* misalignment events caused by the correction itself (some
    correction strategies must shift the wire to reach the corrected position
    and that shift adds to the bookkeeping).
    """

    name: str

    every_nrun: int = 1
    """Apply only when ``nr_run % every_nrun == 0``. Default 1 = every call."""

    @abstractmethod
    def apply(
        self,
        index_offset: np.ndarray,
        misalign_faults: np.ndarray,
        nr_run: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adjust ``index_offset`` in place-or-by-return; update ``misalign_faults``.

        Args:
            index_offset:    2D int array of per-racetrack offsets.
            misalign_faults: 2D int array of per-racetrack fault counts (zero
                             for layers where misalign-fault tracking is off).
            nr_run:          Current inference iteration (1-based).

        Returns:
            ``(new_index_offset, new_misalign_faults)``.
        """


_REGISTRY: dict[str, Callable[..., MitigationStep]] = {}


def register_mitigation(name: str) -> Callable[[type[MitigationStep]], type[MitigationStep]]:
    """Decorator to register a mitigation step by name."""

    def deco(cls: type[MitigationStep]) -> type[MitigationStep]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_mitigation(name: str, **kwargs) -> MitigationStep:
    """Look up and instantiate a mitigation step by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown mitigation '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)
