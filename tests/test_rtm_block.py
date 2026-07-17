"""BLOCK storage-layout fault-injection path — config-guard checks.

Only the CPU-safe config-construction guard is exercised here (no GPU in
this environment). The full BLOCK fault path (``_run_block_path`` running
the real CUDA kernels per bucket) is GPU-verified in Task 7.
"""

import pytest
from netdrift.faults.rtm_misalignment import RTMConfig


def test_block_rejects_per_forward_encoder():
    class _Enc:  # stand-in; real WeightEncoder not needed for the guard
        def apply(self, *a, **k):
            pass

    with pytest.raises(ValueError, match="per_forward"):
        RTMConfig(
            weight_encoder=_Enc(),
            weight_encoder_mode="per_forward",
            block_mapping=True,
        )
