"""Write-time weight encoders for the RTM fault model.

A :class:`WeightEncoder` rewrites a layer's stored quantized weights to
make them more robust under racetrack-shift faults. Unlike
:class:`netdrift.faults.mitigations.MitigationStep` (which adjusts the
``index_offset`` array at read-out), an encoder transforms the stored
weight tensor itself. Endlen (block-hypothesis) is the first instance.

Two operating modes are supported via ``RTMConfig.weight_encoder_mode``:

* ``"once"``: applied a single time after checkpoint load, before any fault
  simulation. The encoded weights are the new persistent state of the
  model (saveable to a checkpoint and reloadable). See
  :func:`apply_weight_encoder_to_model`.
* ``"per_forward"``: applied inside :meth:`RTMMisalignmentFault.inject` on
  every forward pass, between offset mitigations and read-out. Matches
  legacy ``EXEC_ENDLEN`` semantics exactly.
"""

from netdrift.faults.weight_encoders.apply import (
    apply_weight_encoder_to_model,
    is_encoded_checkpoint_path,
    with_endlen_marker,
)
from netdrift.faults.weight_encoders.base import (
    WeightEncoder,
    get_encoder,
    register_encoder,
)
from netdrift.faults.weight_encoders.endlen import (
    EndlenEncoder,
    _endlen_cpu_reference,
)

__all__ = [
    "WeightEncoder",
    "get_encoder",
    "register_encoder",
    "EndlenEncoder",
    "apply_weight_encoder_to_model",
    "is_encoded_checkpoint_path",
    "with_endlen_marker",
    "_endlen_cpu_reference",
]
