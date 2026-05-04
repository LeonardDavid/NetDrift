"""Optimizers used by NetDrift.

The legacy ``Clippy`` optimizer is Adam followed by a per-parameter
``data.clamp(-1, 1)``. Used to keep weights inside the binarization range
during training.
"""

from __future__ import annotations

import torch


class Clippy(torch.optim.Adam):
    """Adam with a post-step clamp of every parameter to ``[-1, 1]``.

    Equivalent to the legacy ``Traintest_Utils.Clippy``. Note: the legacy
    code calls ``p.data.clamp(-1, 1)`` *non*-inplace and discards the
    result — we keep that behaviour bit-for-bit.
    """

    def step(self, closure=None):  # type: ignore[no-untyped-def]
        loss = super().step(closure=closure)
        for group in self.param_groups:
            for p in group["params"]:
                p.data.clamp(-1, 1)
        return loss
