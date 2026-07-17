from types import SimpleNamespace

import pytest

from netdrift.config.schema import StorageCfg
from netdrift.runner.run import (
    _rt_mapping_fn_for_layout,
    _validate_block_layout_combo,
)


def test_storage_cfg_base_layout_default():
    s = StorageCfg(layout="block")
    assert s.base_layout == "row"


def test_rt_mapping_fn_block():
    fn = _rt_mapping_fn_for_layout("block")
    assert fn(object()) == "BLOCK"


def test_rt_mapping_fn_still_row_col():
    assert _rt_mapping_fn_for_layout("row")(object()) == "ROW"
    assert _rt_mapping_fn_for_layout("col")(object()) == "COL"


def _cfg(layout, weight_encoder=None, fault_aware="none"):
    # Lightweight stub carrying just the fields _validate_block_layout_combo reads.
    return SimpleNamespace(
        storage=SimpleNamespace(layout=layout),
        fault=SimpleNamespace(weight_encoder=weight_encoder),
        training=SimpleNamespace(fault_aware=fault_aware),
    )


def test_block_rejects_weight_encoder():
    with pytest.raises(ValueError, match="weight encoder"):
        _validate_block_layout_combo(_cfg("block", weight_encoder="endlen"))


def test_block_rejects_regularizer():
    with pytest.raises(ValueError, match="fault-aware"):
        _validate_block_layout_combo(_cfg("block", fault_aware="regularization"))


def test_block_rejects_ste_inject():
    # ste_inject mutates weight signs across batches; BLOCK caches its block
    # structure from the first forward, so fault-aware training must be rejected.
    with pytest.raises(ValueError, match="fault-aware"):
        _validate_block_layout_combo(_cfg("block", fault_aware="ste_inject"))


def test_block_rejects_kd():
    with pytest.raises(ValueError, match="fault-aware"):
        _validate_block_layout_combo(_cfg("block", fault_aware="kd"))


def test_block_allows_plain_test_run():
    # No encoder, fault_aware=none -> no error.
    _validate_block_layout_combo(_cfg("block"))


def test_non_block_layouts_unaffected():
    # row/col configs with an encoder or regularizer must NOT be rejected here.
    _validate_block_layout_combo(_cfg("row", weight_encoder="endlen"))
    _validate_block_layout_combo(_cfg("col", fault_aware="regularization"))
