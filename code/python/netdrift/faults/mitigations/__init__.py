"""Composable mitigation steps for the RTM fault model.

A :class:`MitigationStep` adjusts an array of per-racetrack ``index_offset``
values just before the racetrack-read kernel runs, modelling the effect of an
error-correction policy. Steps are looked up by name from a registry so that
configs can declare them as a list:

.. code-block:: yaml

    fault:
      mitigations: [bin_revert_mid]   # default: a single step

Multiple steps can be chained; they are applied in declared order. The
defaults provided in ``configs/`` use a single step per the project policy.
"""

from netdrift.faults.mitigations.base import MitigationStep, get_mitigation
from netdrift.faults.mitigations.binomial_revert import (
    BinomialRevertEdges,
    BinomialRevertMid,
)
from netdrift.faults.mitigations.parity_swap import (
    Even2OddDecrease,
    Even2OddIncrease,
    Odd2EvenDecrease,
    Odd2EvenIncrease,
)

__all__ = [
    "MitigationStep",
    "get_mitigation",
    "BinomialRevertMid",
    "BinomialRevertEdges",
    "Odd2EvenDecrease",
    "Odd2EvenIncrease",
    "Even2OddDecrease",
    "Even2OddIncrease",
]
