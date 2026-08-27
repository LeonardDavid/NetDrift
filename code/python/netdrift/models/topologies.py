"""Model topologies registered with the model registry.

Two families:

1. **Custom topologies** (VGG3, VGG7, ResNet18) — ported from the legacy ``Models.py``
   with their original attribute layout (``conv1``, ``bn1``, ``qact1``, ...,
   ``scale``). Built initially as ``nn.Conv2d``/``nn.Linear`` so that
   ``replace_with_quantized`` can swap them, and the resulting state-dict
   keys match the existing HuggingFace checkpoints.

   The ResNet18 variant is sized for 64x64 Imagenette and, like the VGGs,
   carries ``Hardtanh`` + ``QuantizedActivation`` at every activation site —
   the torchvision ResNets below cannot express w1a1 for want of both.

2. **Torchvision wrappers** — ResNet/MobileNet/ViT in both ImageNet
   (224×224) and CIFAR (32×32) variants. The CIFAR variants patch the first
   conv to stride=1 with a smaller kernel and remove the initial maxpool —
   the standard CIFAR-ResNet recipe.

All builders return non-quantized models. The runner applies
:func:`replace_with_quantized` afterward.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from netdrift.models.registry import register_model
from netdrift.quant.layers import QuantizedActivation


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class Scale(nn.Module):
    """Single learnable scalar multiplier — preserved for VGG checkpoint compat."""

    def __init__(self, init_value: float = 1e-3) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _Htanh(nn.Hardtanh):
    """Subclass for clarity — same as nn.Hardtanh()."""


# ---------------------------------------------------------------------------
# VGG3 (MNIST/FMNIST) — 4 quantized layers (2 conv + 2 fc)
# ---------------------------------------------------------------------------


class VGG3(nn.Module):
    """VGG3 used for MNIST/FMNIST. State-dict matches legacy checkpoints.

    Args:
        in_channels: 1 for MNIST/FMNIST.
        num_classes: 10 for both MNIST/FMNIST.
        kernel_size: 3, 5, or 7. Determines the FC1 input width.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size == 3:
            fc1_in = 7 * 7 * 64
        elif kernel_size == 5:
            fc1_in = 5 * 5 * 64
        elif kernel_size == 7:
            fc1_in = 4 * 4 * 64
        else:
            raise ValueError(f"unsupported kernel_size: {kernel_size}")

        self.htanh = _Htanh()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # qactN are activation-quantization sites. With scheme=None (default)
        # they pass activations through unchanged, so FP and plain-BNN training
        # are unaffected. attach_activation_scheme() binds a scheme for W1A_n_.
        # Carrying no parameters/buffers, they leave state_dict keys unchanged.
        self.qact1 = QuantizedActivation()
        self.conv2 = nn.Conv2d(64, 64, kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation()
        self.fc1 = nn.Linear(fc1_in, 2048, bias=False)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation()
        self.fc2 = nn.Linear(2048, num_classes, bias=False)
        self.scale = Scale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.htanh(x)
        x = self.qact1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh(x)
        x = self.qact2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.htanh(x)
        x = self.qact3(x)
        x = self.fc2(x)
        x = self.scale(x)
        return x


@register_model("vgg3_mnist")
def _vgg3_mnist() -> nn.Module:
    return VGG3(in_channels=1, num_classes=10, kernel_size=3)


@register_model("vgg3_fmnist")
def _vgg3_fmnist() -> nn.Module:
    return VGG3(in_channels=1, num_classes=10, kernel_size=3)


# ---------------------------------------------------------------------------
# VGG7 (CIFAR-10/100) — 8 quantized layers (6 conv + 2 fc)
# ---------------------------------------------------------------------------


class VGG7(nn.Module):
    """VGG7 used for CIFAR-10/100. Attribute layout matches legacy checkpoints."""

    def __init__(self, num_classes: int = 10, kernel_size: int = 3, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.htanh = activation if activation is not None else _Htanh()
        # qactN are activation-quantization sites. With scheme=None (default)
        # they pass activations through unchanged, so FP and plain-BNN training
        # are unaffected. attach_activation_scheme() binds a scheme for W1A_n_.
        # Carrying no parameters/buffers, they leave state_dict keys unchanged.
        # block 1
        self.conv1 = nn.Conv2d(3, 128, kernel_size, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.qact1 = QuantizedActivation()
        # block 2
        self.conv2 = nn.Conv2d(128, 128, kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.qact2 = QuantizedActivation()
        # block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.qact3 = QuantizedActivation()
        # block 4
        self.conv4 = nn.Conv2d(256, 256, kernel_size, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.qact4 = QuantizedActivation()
        # block 5
        self.conv5 = nn.Conv2d(256, 512, kernel_size, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.qact5 = QuantizedActivation()
        # block 6
        self.conv6 = nn.Conv2d(512, 512, kernel_size, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.qact6 = QuantizedActivation()
        # block 7 (fc1)
        self.fc1 = nn.Linear(8192, 1024, bias=False)
        self.bn7 = nn.BatchNorm1d(1024)
        self.qact7 = QuantizedActivation()
        # block 8 (fc2)
        self.fc2 = nn.Linear(1024, num_classes, bias=False)
        self.scale = Scale(init_value=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qact1(self.htanh(self.bn1(self.conv1(x))))
        x = self.qact2(self.htanh(self.bn2(F.max_pool2d(self.conv2(x), 2))))
        x = self.qact3(self.htanh(self.bn3(self.conv3(x))))
        x = self.qact4(self.htanh(self.bn4(F.max_pool2d(self.conv4(x), 2))))
        x = self.qact5(self.htanh(self.bn5(self.conv5(x))))
        x = self.qact6(self.htanh(self.bn6(F.max_pool2d(self.conv6(x), 2))))
        x = torch.flatten(x, 1)
        x = self.qact7(self.htanh(self.bn7(self.fc1(x))))
        x = self.fc2(x)
        x = self.scale(x)
        return x


@register_model("vgg7_cifar10")
def _vgg7_cifar10() -> nn.Module:
    return VGG7(num_classes=10)


@register_model("vgg7_cifar100")
def _vgg7_cifar100() -> nn.Module:
    return VGG7(num_classes=100)


@register_model("vgg7_cifar10_fp")
def _vgg7_cifar10_fp() -> nn.Module:
    return VGG7(num_classes=10, activation=nn.ReLU())


@register_model("vgg7_cifar100_fp")
def _vgg7_cifar100_fp() -> nn.Module:
    return VGG7(num_classes=100, activation=nn.ReLU())


# ---------------------------------------------------------------------------
# ResNet18 (Imagenette, 64x64) — 21 quantized layers (20 conv + 1 linear)
# ---------------------------------------------------------------------------


class BasicBlock(nn.Module):
    """Binary ResNet basic block, ported from the legacy ``Models.py`` ResNet.

    Attribute names (``conv1``, ``bn1``, ``conv2``, ``bn2``, ``shortcut``) match
    the legacy NetDrift ResNet, so legacy checkpoints keep loading and the
    ``fc.`` → ``linear.`` remap in the checkpoint adapter still applies.

    Activation placement is what makes this a BNN block rather than a
    torchvision one: ``Hardtanh`` (not ReLU) clamps to ``[-1, 1]`` after ``bn1``
    and after the residual add, and a :class:`QuantizedActivation` site sits at
    each of those two points. The clamp is a hard requirement of
    :class:`~netdrift.quant.uniform.IntUniformActScheme` (W1A2/W1A4), and the
    qact sites are where ``attach_activation_scheme`` binds — a ReLU block has
    neither, which is why the torchvision wrappers cannot express w1a1.

    The 1x1 ``shortcut`` conv exists only where the stage changes stride or
    width; elsewhere ``shortcut`` is an empty ``Sequential`` (identity).
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        kernel_size: int = 3,
        activation: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        super().__init__()
        act = activation if activation is not None else _Htanh
        self.htanh = act()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.qact1 = QuantizedActivation()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.qact2 = QuantizedActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qact1(self.htanh(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.qact2(self.htanh(out))


class ResNet(nn.Module):
    """ResNet18 sized for 64x64 Imagenette. Attribute layout matches legacy ``Models.py``.

    Geometry (64x64 in): stem keeps resolution (3x3 stride 1, no ImageNet
    maxpool), stages 2-4 halve it, an extra ``max_pool2d(2)`` sits between
    stage 3 and stage 4, and the head pools whatever is left to 1x1:
    64 → 64 → 32 → 16 → 8 → 4 → 1, so ``linear`` sees 512 features. The legacy
    forward hard-coded ``max_pool2d(·, 4)`` for that last step, which is exactly
    ``adaptive_max_pool2d(·, 1)`` at 64px but also survives other input sizes.

    ``activation`` is a *factory* (each site needs its own module instance):
    ``_Htanh`` for the BNN, ``nn.ReLU`` for the full-precision warm-start twin.
    Activation modules carry no parameters, so both variants produce identical
    state-dict keys and ``fp32_warmstart`` is a pure load.

    Layer numbering, for ``fault.protection.layers``: ``replace_with_quantized``
    enumerates the root's own children before descending, so the ids are
    ``1 = conv1``, ``2 = linear``, then ``3..21`` walking the stages
    (``conv1``, ``conv2``, ``shortcut.0`` within each block). Note that
    ``model.skip_last_quant`` therefore skips ``layer4.1.conv2``, NOT the
    classifier — keep the first conv and the classifier out of the fault
    simulation with ``fault.protection`` instead.
    """

    def __init__(
        self,
        num_blocks: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_classes: int = 10,
        kernel_size: int = 3,
        activation: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        super().__init__()
        act = activation if activation is not None else _Htanh
        self._act = act
        self._kernel_size = kernel_size
        self.in_planes = 64

        self.htanh = act()
        self.conv1 = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation()

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes, bias=False)
        self.scale = Scale(init_value=1e-3)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(
                BasicBlock(
                    self.in_planes, planes, stride=s,
                    kernel_size=self._kernel_size, activation=self._act,
                )
            )
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qact1(self.htanh(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.max_pool2d(out, 2)
        out = self.layer4(out)
        out = F.adaptive_max_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        out = self.scale(out)
        return out


@register_model("resnet18_imagenette")
def _resnet18_imagenette() -> nn.Module:
    return ResNet(num_classes=10)


@register_model("resnet18_imagenette_fp")
def _resnet18_imagenette_fp() -> nn.Module:
    return ResNet(num_classes=10, activation=nn.ReLU)


# ---------------------------------------------------------------------------
# Torchvision wrappers (ResNet/MobileNet/ViT, ImageNet + CIFAR variants)
# ---------------------------------------------------------------------------


def _patch_resnet_for_cifar(model: nn.Module) -> nn.Module:
    """Apply the standard CIFAR-ResNet patch: stride-1 3x3 conv1, no maxpool.

    Torchvision ResNets assume 224x224 ImageNet inputs (stride-2 conv1 +
    maxpool). For 32x32 CIFAR we down-sample 7-fold less aggressively.
    """
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # type: ignore[attr-defined]
    model.maxpool = nn.Identity()  # type: ignore[attr-defined]
    return model


def _replace_classifier(model: nn.Module, num_classes: int, attr: str = "fc") -> nn.Module:
    """Swap the classifier head for one with ``num_classes`` outputs."""
    head = getattr(model, attr)
    if isinstance(head, nn.Linear):
        setattr(model, attr, nn.Linear(head.in_features, num_classes, bias=head.bias is not None))
    else:
        # MobileNetV2: classifier is a Sequential ending in Linear
        seq = head
        last = seq[-1]
        if isinstance(last, nn.Linear):
            seq[-1] = nn.Linear(last.in_features, num_classes, bias=last.bias is not None)
        else:
            raise NotImplementedError(f"don't know how to swap classifier {head!r}")
    return model


def _build_torchvision(name: str, num_classes: int, *, cifar: bool) -> nn.Module:
    """Build a torchvision model with random init, optionally CIFAR-patched."""
    import torchvision.models as tvm  # local import keeps torchvision optional

    builders = {
        "resnet18": tvm.resnet18,
        "resnet34": tvm.resnet34,
        "resnet50": tvm.resnet50,
        "mobilenetv2": tvm.mobilenet_v2,
        "vit_b_16": tvm.vit_b_16,
    }
    if name not in builders:
        raise KeyError(f"unsupported torchvision model: {name}")

    model = builders[name](weights=None)

    head_attr = "classifier" if name == "mobilenetv2" else (
        "heads" if name == "vit_b_16" else "fc"
    )

    if name == "vit_b_16":
        # ViT classifier head is heads.head; treat as the linear layer to swap.
        if num_classes != 1000:
            head = model.heads.head  # type: ignore[attr-defined]
            model.heads.head = nn.Linear(head.in_features, num_classes)  # type: ignore[attr-defined]
    else:
        if num_classes != 1000:
            _replace_classifier(model, num_classes, attr=head_attr)

    if cifar and name.startswith("resnet"):
        _patch_resnet_for_cifar(model)

    return model


def _make_tv_builder(arch: str, num_classes: int, cifar: bool):
    def _builder() -> nn.Module:
        return _build_torchvision(arch, num_classes=num_classes, cifar=cifar)
    return _builder


# ImageNet-1k variants
register_model("resnet18_imagenet")(_make_tv_builder("resnet18", 1000, cifar=False))
register_model("resnet34_imagenet")(_make_tv_builder("resnet34", 1000, cifar=False))
register_model("resnet50_imagenet")(_make_tv_builder("resnet50", 1000, cifar=False))
register_model("mobilenetv2_imagenet")(_make_tv_builder("mobilenetv2", 1000, cifar=False))
register_model("vit_b_16_imagenet")(_make_tv_builder("vit_b_16", 1000, cifar=False))

# CIFAR-10 variants
register_model("resnet18_cifar10")(_make_tv_builder("resnet18", 10, cifar=True))
register_model("resnet34_cifar10")(_make_tv_builder("resnet34", 10, cifar=True))
register_model("resnet50_cifar10")(_make_tv_builder("resnet50", 10, cifar=True))
register_model("mobilenetv2_cifar10")(_make_tv_builder("mobilenetv2", 10, cifar=False))
register_model("vit_b_16_cifar10")(_make_tv_builder("vit_b_16", 10, cifar=False))

# CIFAR-100 variants
register_model("resnet18_cifar100")(_make_tv_builder("resnet18", 100, cifar=True))
register_model("resnet34_cifar100")(_make_tv_builder("resnet34", 100, cifar=True))
register_model("resnet50_cifar100")(_make_tv_builder("resnet50", 100, cifar=True))
register_model("mobilenetv2_cifar100")(_make_tv_builder("mobilenetv2", 100, cifar=False))
register_model("vit_b_16_cifar100")(_make_tv_builder("vit_b_16", 100, cifar=False))
