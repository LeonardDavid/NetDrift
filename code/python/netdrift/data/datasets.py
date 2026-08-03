"""Dataset builders.

Each entry point returns ``(train_dataset, test_dataset, num_classes)``.
Transforms preserve the legacy NetDrift defaults (CIFAR uses ``(0.5, 0.5,
0.5)`` mean/std, Imagenette uses ImageNet stats, etc.) so accuracy numbers
are directly comparable to the legacy code.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def _mnist_like(name: str, data_dir: str, cls) -> Tuple[Dataset, Dataset, int]:
    transform = transforms.Compose([transforms.ToTensor()])
    train = cls(data_dir, train=True, download=True, transform=transform)
    test = cls(data_dir, train=False, transform=transform)
    return train, test, 10


def _cifar_like(num_classes: int, data_dir: str, cls) -> Tuple[Dataset, Dataset, int]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train = cls(data_dir, train=True, download=True, transform=transform_train)
    test = cls(data_dir, train=False, transform=transform_test)
    return train, test, num_classes


def _imagenette(data_dir: str) -> Tuple[Dataset, Dataset, int]:
    root = os.path.join(data_dir, "imagenette2")
    if not os.path.exists(root):
        raise FileNotFoundError(
            f"Imagenette not found at {root}. Download from "
            f"https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz "
            f"and extract under {data_dir}/."
        )
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train = datasets.ImageFolder(os.path.join(root, "train"), transform=transform_train)
    test = datasets.ImageFolder(os.path.join(root, "val"), transform=transform_test)
    return train, test, 10


def _imagenet(data_dir: str) -> Tuple[Dataset, Dataset, int]:
    """Standard ImageNet-1k via ``ImageFolder``. Expects ``data_dir/imagenet/{train,val}``."""
    root = os.path.join(data_dir, "imagenet")
    if not os.path.exists(os.path.join(root, "train")):
        raise FileNotFoundError(
            f"ImageNet not found at {root}/train. Place the standard "
            f"train/ and val/ folders under {root}/."
        )
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train = datasets.ImageFolder(os.path.join(root, "train"), transform=transform_train)
    test = datasets.ImageFolder(os.path.join(root, "val"), transform=transform_test)
    return train, test, 1000


_DATASETS = {
    "mnist":      lambda d: _mnist_like("mnist", d, datasets.MNIST),
    "fmnist":     lambda d: _mnist_like("fmnist", d, datasets.FashionMNIST),
    "kmnist":     lambda d: _mnist_like("kmnist", d, datasets.KMNIST),
    "svhn":       lambda d: (
        datasets.SVHN(root=os.path.join(d, "SVHN"), split="train", download=True,
                      transform=transforms.Compose([transforms.ToTensor()])),
        datasets.SVHN(root=os.path.join(d, "SVHN"), split="test", download=True,
                      transform=transforms.Compose([transforms.ToTensor()])),
        10,
    ),
    "cifar10":    lambda d: _cifar_like(10, d, datasets.CIFAR10),
    "cifar100":   lambda d: _cifar_like(100, d, datasets.CIFAR100),
    "imagenette": _imagenette,
    "imagenet":   _imagenet,
}


def build_datasets(name: str, data_dir: str = "data") -> Tuple[Dataset, Dataset, int]:
    """Return ``(train, test, num_classes)`` for ``name``."""
    os.makedirs(data_dir, exist_ok=True)
    key = name.lower()
    if key not in _DATASETS:
        raise KeyError(f"unknown dataset {name!r}. Available: {sorted(_DATASETS)}")
    return _DATASETS[key](data_dir)
