"""Dataset wrappers used by the runner.

Phase 1 ports the seven datasets supported by the legacy ``Utils.py``
(MNIST, FashionMNIST, KMNIST, SVHN, CIFAR-10, CIFAR-100, Imagenette) plus
adds ImageNet-1k (folder-based, expects standard layout). The wrappers
return ``(train_dataset, test_dataset, num_classes)``.
"""

from netdrift.data.datasets import build_datasets

__all__ = ["build_datasets"]
