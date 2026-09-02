"""BNN ResNet18 topology for Imagenette (64x64).

The framework's w1a1 recipe needs a topology that (a) clamps activations with
``Hardtanh`` and (b) carries ``QuantizedActivation`` sites, so the activation
half of W1A_n_ is reachable at all. The torchvision wrappers satisfy neither
(ReLU, no qact modules), which is why ``resnet18_imagenette`` is a custom
topology ported from the legacy ``Models.py`` ResNet.

Also pins the FP-twin contract: ``resnet18_imagenette_fp`` must produce the
same state-dict keys as ``resnet18_imagenette`` so ``fp32_warmstart`` is a
pure load rather than a silent from-scratch run.

CPU-safe — no CUDA needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from netdrift.models import build_model, replace_with_quantized
from netdrift.models.registry import list_models
from netdrift.quant import BinaryScheme, QuantizedConv2d, QuantizedLinear
from netdrift.quant.layers import QuantizedActivation


# resnet18 = BasicBlock x [2,2,2,2]: stem conv + 16 block convs + 3 shortcut
# convs (one per stride-2 stage transition) + the classifier.
EXPECTED_QUANT_LAYERS = 1 + 16 + 3 + 1
# One activation site after the stem, two per basic block.
EXPECTED_QACT_SITES = 1 + 2 * 8


def _quant_layers(model: nn.Module) -> list[nn.Module]:
    return [
        m for m in model.modules()
        if isinstance(m, (QuantizedConv2d, QuantizedLinear))
    ]


def test_both_variants_registered() -> None:
    """The BNN topology and its FP warm-start twin are both in the registry."""
    names = list_models()
    assert "resnet18_imagenette" in names
    assert "resnet18_imagenette_fp" in names


def test_forward_shape_at_64px() -> None:
    """Imagenette is served at 64x64; the head must reduce it to (N, 10)."""
    model = build_model("resnet18_imagenette")
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 10)


def test_bnn_variant_clamps_activations() -> None:
    """The BNN variant uses Hardtanh, never ReLU — IntUniformActScheme needs the clamp."""
    model = build_model("resnet18_imagenette")
    assert not any(isinstance(m, nn.ReLU) for m in model.modules())
    assert any(isinstance(m, nn.Hardtanh) for m in model.modules())


def test_fp_variant_uses_relu() -> None:
    """The FP warm-start twin trains with ReLU (stage 1 is full precision)."""
    model = build_model("resnet18_imagenette_fp")
    assert any(isinstance(m, nn.ReLU) for m in model.modules())
    assert not any(isinstance(m, nn.Hardtanh) for m in model.modules())


def test_activation_quant_sites_present() -> None:
    """Every post-BN activation is a qact site, so W1A_n_ is reachable."""
    model = build_model("resnet18_imagenette")
    sites = [m for m in model.modules() if isinstance(m, QuantizedActivation)]
    assert len(sites) == EXPECTED_QACT_SITES


def test_fp_and_bnn_state_dict_keys_match() -> None:
    """FP twin ↔ BNN keys must be identical, or fp32_warmstart silently no-ops."""
    fp_keys = set(build_model("resnet18_imagenette_fp").state_dict().keys())
    bnn_keys = set(build_model("resnet18_imagenette").state_dict().keys())
    assert fp_keys == bnn_keys, (
        f"key sets diverged\n"
        f"fp-only : {sorted(fp_keys - bnn_keys)}\n"
        f"bnn-only: {sorted(bnn_keys - fp_keys)}"
    )


def test_quantized_layer_count() -> None:
    """21 quantized layers, numbered 1..21 in traversal order."""
    model = build_model("resnet18_imagenette")
    replace_with_quantized(model, BinaryScheme())
    layers = _quant_layers(model)
    assert len(layers) == EXPECTED_QUANT_LAYERS
    assert sorted(m.layer_id for m in layers) == list(
        range(1, EXPECTED_QUANT_LAYERS + 1)
    )


def test_shortcut_conv_is_quantized_after_its_blocks_convs() -> None:
    """Protection configs are written against this order: conv1, conv2, shortcut.0."""
    model = build_model("resnet18_imagenette")
    replace_with_quantized(model, BinaryScheme())
    ids = {m.layer_name: m.layer_id for m in _quant_layers(model)}
    assert ids["layer2.0.conv1"] < ids["layer2.0.conv2"] < ids["layer2.0.shortcut.0"]


def test_state_dict_keys_preserved_by_replacement() -> None:
    """BinaryScheme adds no parameters, so a vanilla state_dict loads strictly."""
    sd = build_model("resnet18_imagenette").state_dict()
    qmodel = build_model("resnet18_imagenette")
    replace_with_quantized(qmodel, BinaryScheme())
    qmodel.load_state_dict(sd, strict=True)


def test_head_is_named_linear_for_checkpoint_adapter() -> None:
    """``fc.`` → ``linear.`` remap in the checkpoint adapter keys off this name."""
    model = build_model("resnet18_imagenette")
    children = dict(model.named_children())
    assert "linear" in children
    assert "fc" not in children


def test_classifier_layer_id_is_two_not_last() -> None:
    """``replace_with_quantized`` enumerates the ROOT's children before descending.

    On a nested topology that puts the classifier at layer_id 2 — not 21 —
    because ``conv1`` and ``linear`` are both direct children of the root.
    Protection lists in the configs are written against this numbering.
    """
    model = build_model("resnet18_imagenette")
    replace_with_quantized(model, BinaryScheme())
    ids = {m.layer_name: m.layer_id for m in _quant_layers(model)}
    assert ids["conv1"] == 1
    assert ids["linear"] == 2


def test_skip_last_quant_does_not_spare_the_classifier() -> None:
    """Characterizes a footgun: ``skip_last`` drops the last *traversed* layer.

    On this topology that is ``layer4.1.conv2``, so ``model.skip_last_quant``
    cannot be used to keep the classifier in full precision — use
    ``fault.protection`` instead.
    """
    model = build_model("resnet18_imagenette")
    replace_with_quantized(model, BinaryScheme(), skip_last=True)
    assert isinstance(model.linear, QuantizedLinear)
    assert not isinstance(model.layer4[1].conv2, QuantizedConv2d)
