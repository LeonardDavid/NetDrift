"""Model registry: name → ``Callable[[], nn.Module]``.

A builder takes no arguments and returns a fresh, non-quantized model. The
caller (``runner.run``) then applies :func:`replace_with_quantized` to swap
in :class:`QuantizedConv2d`/:class:`QuantizedLinear` modules in place,
preserving attribute paths so checkpoints keep loading.

Builders are registered via :func:`register_model` (typically as a decorator
in ``topologies.py``).
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn


_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Register a model builder under a unique name.

    Usage::

        @register_model("vgg7_cifar10")
        def vgg7_cifar10() -> nn.Module:
            ...
    """

    def deco(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def build_model(name: str, **kwargs) -> nn.Module:
    """Look up a builder by name and instantiate it."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return all registered model names."""
    return sorted(_REGISTRY)
