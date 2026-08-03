"""Model registry, topologies, and in-place quantization utilities.

* :func:`register_model` / :func:`build_model` — name → builder dispatch.
* :mod:`topologies` — VGG3/VGG7 (custom) and torchvision wrappers
  (ResNet/MobileNet/ViT, both ImageNet and CIFAR variants).
* :func:`replace_with_quantized` — walk a model tree and swap
  ``nn.Conv2d``/``nn.Linear`` with their quantized equivalents in place,
  preserving attribute paths so existing checkpoints keep loading.
* :func:`apply_protection_policy` — flip the ``protected`` flag on each
  layer based on a config-driven policy.
"""

from netdrift.models.registry import build_model, list_models, register_model
from netdrift.models.replace import (
    DEFAULT_SKIP_TYPES,
    attach_activation_scheme,
    attach_fault_model,
    replace_with_quantized,
)
from netdrift.models.protection import apply_protection_policy
from netdrift.models import topologies  # registers names as a side effect

__all__ = [
    "build_model",
    "list_models",
    "register_model",
    "replace_with_quantized",
    "attach_activation_scheme",
    "attach_fault_model",
    "apply_protection_policy",
    "DEFAULT_SKIP_TYPES",
    "topologies",
]
