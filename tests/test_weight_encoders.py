"""Weight encoders: registry, endlen CPU reference, filename auto-detect.

CPU-safe — these tests don't touch a GPU. GPU-vs-CPU parity for the endlen
kernel lives in ``test_rtm_fault.py`` under ``@pytest.mark.cuda``.
"""

from __future__ import annotations

import numpy as np
import pytest

from netdrift.faults.weight_encoders import (
    EndlenEncoder,
    _endlen_cpu_reference,
    get_encoder,
    is_encoded_checkpoint_path,
    with_endlen_marker,
)
from netdrift.faults.rtm_misalignment import RTMConfig


def test_registry_lookup_returns_endlen() -> None:
    enc = get_encoder("endlen")
    assert isinstance(enc, EndlenEncoder)


def test_unknown_encoder_raises() -> None:
    with pytest.raises(KeyError, match="Unknown weight encoder"):
        get_encoder("not_a_thing")


def test_cpu_reference_idempotent_on_single_run() -> None:
    """A racetrack consisting of one big bitgroup must be unchanged."""
    rt_size = 8
    weight = np.ones((1, rt_size), dtype=np.int32)
    before = weight.copy()
    _endlen_cpu_reference(weight, rt_size)
    assert np.array_equal(weight, before)


def test_cpu_reference_flips_single_isolated_bit() -> None:
    """A length-1 bitgroup wedged between two long runs should be merged.

    Racetrack: 1 1 1 -1 1 1 1 1   (length 8)
      bitgroups: [3, 1, 4]
      The single tuple emitted at the end has endlen = 3+1+4 = 8, flips=1,
      start_index_mid=3. Greedy pick flips index 3 from -1 to +1, producing
      a single bitgroup of length 8.
    """
    rt_size = 8
    weight = np.array([[1, 1, 1, -1, 1, 1, 1, 1]], dtype=np.int32)
    _endlen_cpu_reference(weight, rt_size)
    assert weight.tolist() == [[1, 1, 1, 1, 1, 1, 1, 1]]


def test_cpu_reference_picks_largest_endlen_first() -> None:
    """Two non-overlapping merge candidates; the larger endlen wins, the
    second non-overlapping one also fires.

    Racetrack: 1 1 -1 1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1  (length 16)
      bitgroups: [2, 1, 3, 4, 1, 5]
      Tuples emitted (k, endlen, flips, start_mid):
        k=0: endlen=6  flips=1 start_mid=2   (window over groups 0,1,2)
        k=1: endlen=8  flips=3 start_mid=3   (groups 1,2,3)
        k=2: endlen=8  flips=4 start_mid=6   (groups 2,3,4)
        k=3: endlen=10 flips=1 start_mid=10  (final tuple from groups 3,4,5)
      Sort by (-endlen, flips):  k=3, k=1, k=2, k=0
      Greedy picks:
        k=3 (endlen=10, flips=1) → flip idx 10 to -1;  remove k=2,3
        k=1 (endlen=8,  flips=3) → flip idxs 3,4,5 to -1;  remove k=0,1
      Output: 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
    """
    rt_size = 16
    weight = np.array(
        [[1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1]],
        dtype=np.int32,
    )
    _endlen_cpu_reference(weight, rt_size)
    expected = [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    assert weight.tolist() == [expected]


def test_cpu_reference_processes_rows_independently() -> None:
    rt_size = 4
    w = np.array(
        [
            [1, 1, 1, 1],     # one bitgroup → unchanged
            [1, -1, 1, 1],    # bitgroups [1,1,2] → final tuple flips idx 1 to +1
        ],
        dtype=np.int32,
    )
    _endlen_cpu_reference(w, rt_size)
    assert w.tolist() == [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]


def test_rt_size_over_64_rejected_in_rtm_config() -> None:
    cfg_ok = RTMConfig(rt_size=64, weight_encoder=EndlenEncoder())
    assert cfg_ok.rt_size == 64

    with pytest.raises(ValueError, match="rt_size <= 64"):
        RTMConfig(rt_size=128, weight_encoder=EndlenEncoder())


def test_invalid_weight_encoder_mode_rejected() -> None:
    with pytest.raises(ValueError, match="weight_encoder_mode"):
        RTMConfig(weight_encoder_mode="twice")


def test_is_encoded_checkpoint_path_positive_cases() -> None:
    assert is_encoded_checkpoint_path("model_endlen.pt")
    assert is_encoded_checkpoint_path("/runs/foo/vgg7_endlen_w1a4.pt")
    assert is_encoded_checkpoint_path("/path/to/foo_endlen.pth")
    assert is_encoded_checkpoint_path("ENDLEN_model.pt")
    assert is_encoded_checkpoint_path("foo-endlen.pt")
    assert is_encoded_checkpoint_path("/runs/endlen.pt")


def test_is_encoded_checkpoint_path_negative_cases() -> None:
    assert not is_encoded_checkpoint_path("model.pt")
    assert not is_encoded_checkpoint_path("model_endless.pt")
    assert not is_encoded_checkpoint_path("bnn_model.pt")
    assert not is_encoded_checkpoint_path(None)
    assert not is_encoded_checkpoint_path("")


def test_with_endlen_marker_adds_when_absent() -> None:
    assert with_endlen_marker("foo.pt").endswith("foo_endlen.pt")
    assert with_endlen_marker("/abs/path/model.pth").endswith("model_endlen.pth")


def test_with_endlen_marker_idempotent_when_present() -> None:
    assert with_endlen_marker("model_endlen.pt") == "model_endlen.pt"
    assert with_endlen_marker("/runs/foo_endlen.pt") == "/runs/foo_endlen.pt"
