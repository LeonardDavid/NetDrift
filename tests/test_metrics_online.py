"""OnlineCollector: per-loop time series + final raw arrays."""
from __future__ import annotations

import numpy as np


def test_records_per_loop_time_series():
    from netdrift.metrics.online import OnlineCollector

    oc = OnlineCollector()
    # loop 1
    oc.record_loop(
        loop_idx=1,
        totals={"bitflips": 10, "wrong_bits_read": 4},
        per_layer={"lin": {"bitflips": 10, "wrong_bits_read": 4}},
    )
    # loop 2
    oc.record_loop(
        loop_idx=2,
        totals={"bitflips": 12, "wrong_bits_read": 5},
        per_layer={"lin": {"bitflips": 12, "wrong_bits_read": 5}},
    )
    series = oc.as_per_loop()
    assert series["wrong_bits_read"]["total"] == [4, 5]
    assert series["wrong_bits_read"]["per_layer"]["lin"] == [4, 5]
    assert series["bitflips"]["total"] == [10, 12]


def test_final_raw_arrays_collected_from_state():
    from netdrift.metrics.online import OnlineCollector

    class _FakeState:
        index_offset = np.array([[1, 0], [0, -1]], dtype=np.int32)
        last_wrong_read = np.array([[2, 0], [0, 3]], dtype=np.int32)

    class _FakeLayer:
        fault_state = _FakeState()

    oc = OnlineCollector()
    arrays = oc.final_raw_arrays({"lin": _FakeLayer()})
    assert "online__final__lin__index_offset" in arrays
    assert "online__final__lin__wrong_read_mask" in arrays
    assert arrays["online__final__lin__index_offset"].tolist() == [[1, 0], [0, -1]]
