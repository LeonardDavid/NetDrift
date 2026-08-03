"""Layer-protection policies: which layers run with fault injection.

Replaces the legacy ``protectLayers=[0,1,1,...]`` model-wide array with a
per-module ``protected`` flag set by name or by index. Indices match the
order of ``replace_with_quantized``-assigned ``layer_id`` values (1-based,
matching the legacy convention).

Three policies, mirroring the legacy CLI flags:

* ``all``    — all quantized layers unprotected (full fault injection).
* ``custom`` — explicit list of unprotected layer indices; everything else protected.
* ``indiv``  — exactly one layer unprotected at a time; caller iterates.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import torch.nn as nn

from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear


ProtectionPolicy = Literal["all", "custom", "indiv"]


def apply_protection_policy(
    model: nn.Module,
    policy: ProtectionPolicy,
    *,
    layers: Optional[Iterable[int]] = None,
    indiv_layer: Optional[int] = None,
) -> nn.Module:
    """Set ``layer.protected`` on every quantized layer in ``model``.

    Args:
        model:       Root module (post-replacement).
        policy:      Protection policy.
        layers:      For ``policy="custom"``: 1-based indices to leave unprotected.
        indiv_layer: For ``policy="indiv"``: the single 1-based index to leave unprotected.

    Returns:
        The same model.
    """
    # The set of valid 1-based layer_ids this model actually has — used to
    # reject protection ids that point at non-existent layers (e.g. VGG3's
    # [2,3] silently applied to a model that expected [2,3,4,5,6,7], or a typo).
    valid_ids = {
        m.layer_id for m in model.modules()
        if isinstance(m, (QuantizedConv2d, QuantizedLinear))
    }

    if policy == "all":
        unprotected = None
    elif policy == "custom":
        if not layers:
            raise ValueError("policy='custom' requires a non-empty layers=[...]")
        unprotected = set(layers)
    elif policy == "indiv":
        if indiv_layer is None:
            raise ValueError("policy='indiv' requires indiv_layer=N")
        unprotected = {indiv_layer}
    else:
        raise ValueError(f"unknown protection policy: {policy}")

    if unprotected is not None:
        bad = sorted(i for i in unprotected if i not in valid_ids)
        if bad:
            raise ValueError(
                f"protection {policy!r} references layer id(s) {bad} not present "
                f"in this model; valid 1-based layer ids are {sorted(valid_ids)}"
            )

    for module in model.modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            if unprotected is None:
                module.protected = False  # all unprotected
            else:
                module.protected = module.layer_id not in unprotected

    return model
