"""``replace_with_quantized`` correctness:

* Layer types are swapped while attribute paths and state-dict keys are preserved.
* ``layer_id`` matches iteration order.
* ``skip_first`` / ``skip_last`` / ``skip_modules`` work.
* Constructed checkpoints round-trip via strict ``load_state_dict`` after replacement.

CPU-safe — no CUDA needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from netdrift.models import build_model, replace_with_quantized
from netdrift.quant import BinaryScheme, QuantizedConv2d, QuantizedLinear


def test_vgg7_state_dict_keys_preserved() -> None:
    """The state-dict keys of a freshly-quantized VGG7 must match the FP32 baseline."""
    fp32 = build_model("vgg7_cifar10")
    fp32_keys = set(fp32.state_dict().keys())

    qmodel = build_model("vgg7_cifar10")
    replace_with_quantized(qmodel, BinaryScheme())
    q_keys = set(qmodel.state_dict().keys())

    # BinaryScheme adds no per-channel scales, so keys must match exactly.
    assert q_keys == fp32_keys, (
        f"state_dict keys diverged after quantization\n"
        f"missing: {fp32_keys - q_keys}\n"
        f"extra:   {q_keys - fp32_keys}"
    )


def test_vgg7_strict_load_after_replace() -> None:
    """A vanilla VGG7 state_dict loads strictly into the quantized variant."""
    fp32 = build_model("vgg7_cifar10")
    sd = fp32.state_dict()

    qmodel = build_model("vgg7_cifar10")
    replace_with_quantized(qmodel, BinaryScheme())
    missing, unexpected = qmodel.load_state_dict(sd, strict=True)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"


def test_replaced_layer_types() -> None:
    """Conv2d → QuantizedConv2d, Linear → QuantizedLinear; nothing else changes."""
    qmodel = build_model("vgg3_mnist")
    replace_with_quantized(qmodel, BinaryScheme())

    assert isinstance(qmodel.conv1, QuantizedConv2d)
    assert isinstance(qmodel.conv2, QuantizedConv2d)
    assert isinstance(qmodel.fc1, QuantizedLinear)
    assert isinstance(qmodel.fc2, QuantizedLinear)
    # BatchNorm and Scale are not swapped
    assert isinstance(qmodel.bn1, nn.BatchNorm2d)
    assert isinstance(qmodel.bn2, nn.BatchNorm2d)
    assert isinstance(qmodel.bn3, nn.BatchNorm1d)


def test_layer_ids_are_iteration_order() -> None:
    """Every quantized layer gets a 1-based id matching the order it was visited."""
    qmodel = build_model("vgg3_mnist")
    replace_with_quantized(qmodel, BinaryScheme())

    expected = {"conv1": 1, "conv2": 2, "fc1": 3, "fc2": 4}
    seen = {}
    for name, mod in qmodel.named_modules():
        if isinstance(mod, (QuantizedConv2d, QuantizedLinear)):
            seen[name] = mod.layer_id
    assert seen == expected


def test_skip_first_keeps_first_layer_fp32() -> None:
    """``skip_first=True`` leaves the first conv as a vanilla nn.Conv2d."""
    qmodel = build_model("vgg7_cifar10")
    replace_with_quantized(qmodel, BinaryScheme(), skip_first=True)
    assert isinstance(qmodel.conv1, nn.Conv2d)
    assert not isinstance(qmodel.conv1, QuantizedConv2d)
    # Subsequent layers ARE quantized
    assert isinstance(qmodel.conv2, QuantizedConv2d)


def test_skip_last_keeps_last_layer_fp32() -> None:
    """``skip_last=True`` leaves the final classifier as a vanilla nn.Linear."""
    qmodel = build_model("vgg7_cifar10")
    replace_with_quantized(qmodel, BinaryScheme(), skip_last=True)
    assert isinstance(qmodel.fc2, nn.Linear)
    assert not isinstance(qmodel.fc2, QuantizedLinear)


def test_skip_modules_by_name() -> None:
    """Explicitly-named modules are not swapped."""
    qmodel = build_model("vgg7_cifar10")
    replace_with_quantized(qmodel, BinaryScheme(), skip_modules={"fc1"})
    assert not isinstance(qmodel.fc1, QuantizedLinear)
    assert isinstance(qmodel.fc1, nn.Linear)
    assert isinstance(qmodel.fc2, QuantizedLinear)


def test_resnet18_cifar_replacement_preserves_keys() -> None:
    """Torchvision-based resnet18_cifar10 keeps its state-dict shape after replacement."""
    fp32 = build_model("resnet18_cifar10")
    keys = set(fp32.state_dict().keys())

    qmodel = build_model("resnet18_cifar10")
    replace_with_quantized(qmodel, BinaryScheme())
    q_keys = set(qmodel.state_dict().keys())
    assert q_keys == keys


def test_replace_preserves_weight_values() -> None:
    """Weights copied from the FP32 module must survive the swap unchanged."""
    fp32 = build_model("vgg3_mnist")
    fp32_w = fp32.conv1.weight.detach().clone()

    replace_with_quantized(fp32, BinaryScheme())
    assert torch.equal(fp32.conv1.weight.detach(), fp32_w)
