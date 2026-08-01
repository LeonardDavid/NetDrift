import pytest

from netdrift.config.schema import StorageCfg, UnitsCfg
from netdrift.faults.rtm_misalignment import RTMConfig


def test_units_cfg_defaults():
    u = UnitsCfg()
    assert (u.threshold, u.max_period, u.pool_guard) == (4, 1, 0)


def test_storage_accepts_units_layout():
    s = StorageCfg(layout="units", base_layout="col")
    assert s.layout == "units"
    assert s.units.threshold == 4


def test_max_period_2_requires_threshold_2():
    with pytest.raises(ValueError, match="requires threshold=2"):
        UnitsCfg(threshold=4, max_period=2)
    UnitsCfg(threshold=2, max_period=2, pool_guard=1)  # must not raise


@pytest.mark.parametrize("kwargs", [
    {"threshold": 0},
    {"max_period": 3},
    {"threshold": 2, "max_period": 2, "pool_guard": 2},
])
def test_units_cfg_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        UnitsCfg(**kwargs)


def test_rtm_config_units_fields_default_off():
    c = RTMConfig(rt_size=64, rt_error=0.0)
    assert c.units_mapping is False


def test_rtm_config_rejects_ap_position_with_units():
    with pytest.raises(ValueError, match="ap_position"):
        RTMConfig(rt_size=64, rt_error=0.0, units_mapping=True, ap_position=8)


def test_rtm_config_revalidates_units_combo():
    with pytest.raises(ValueError, match="requires threshold=2"):
        RTMConfig(rt_size=64, rt_error=0.0, units_mapping=True,
                  units_threshold=4, units_max_period=2)


def test_units_rejects_rt_size_over_64():
    with pytest.raises(ValueError, match="rt_size <= 64"):
        StorageCfg(layout="units", rt_size=128)


def test_non_units_layouts_still_allow_large_rt_size():
    # The cap is units-specific; do not regress row/col/block.
    StorageCfg(layout="row", rt_size=128)
    StorageCfg(layout="block", rt_size=128)


@pytest.mark.parametrize("kwargs", [
    {"units_threshold": 0},
    {"units_max_period": 3},
    {"units_pool_guard": 2},
])
def test_rtm_config_rejects_bad_units_values(kwargs):
    with pytest.raises(ValueError):
        RTMConfig(rt_size=64, rt_error=0.0, units_mapping=True, **kwargs)


def test_rtm_config_ignores_units_values_when_units_mapping_off():
    # The guards are units-gated: a row/col/block run must not trip them.
    RTMConfig(rt_size=64, rt_error=0.0, units_threshold=0, units_pool_guard=2)
