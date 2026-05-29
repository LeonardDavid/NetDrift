"""Pattern-preserving recalibration: BN stats + Scale update, signs frozen."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from netdrift.config.schema import RecalibrateCfg


def _tiny_loader(n=32, in_ch=1, hw=28, classes=10):
    x = torch.randn(n, in_ch, hw, hw)
    y = torch.randint(0, classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def _build_quant_vgg3():
    from netdrift.models import build_model, replace_with_quantized
    from netdrift.quant.binary import BinaryScheme

    model = build_model("vgg3_fmnist")
    replace_with_quantized(model, BinaryScheme())
    return model


def test_bn_stats_change_signs_frozen():
    from netdrift.training.recalibrate import recalibrate

    model = _build_quant_vgg3()
    loader = _tiny_loader()
    # Snapshot binary signs of every quantized weight.
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear
    pre_signs = {
        n: torch.sign(m.weight.detach().clone())
        for n, m in model.named_modules()
        if isinstance(m, (QuantizedConv2d, QuantizedLinear))
    }
    bn = model.bn1
    pre_mean = bn.running_mean.detach().clone()

    cfg = RecalibrateCfg(enabled=True, bn_stats=True, tune_affine=False, epochs=0)
    recalibrate(model, loader, torch.device("cpu"), cfg)

    # BN running stats moved.
    assert not torch.allclose(bn.running_mean, pre_mean)
    # Binary signs unchanged (endlen pattern preserved).
    for n, m in model.named_modules():
        if isinstance(m, (QuantizedConv2d, QuantizedLinear)):
            assert torch.equal(torch.sign(m.weight.detach()), pre_signs[n]), n


def test_tune_affine_only_bn_and_scale_have_grad():
    from netdrift.training.recalibrate import _set_recal_trainable
    from netdrift.quant.layers import QuantizedConv2d, QuantizedLinear

    model = _build_quant_vgg3()
    _set_recal_trainable(model)
    # Quantized weights frozen.
    for _, m in model.named_modules():
        if isinstance(m, (QuantizedConv2d, QuantizedLinear)):
            assert m.weight.requires_grad is False
    # BN params + Scale trainable.
    assert model.bn1.weight.requires_grad is True
    assert model.bn1.bias.requires_grad is True
    assert model.scale.scale.requires_grad is True


def test_tune_affine_false_leaves_gamma_beta_fixed():
    from netdrift.training.recalibrate import recalibrate

    model = _build_quant_vgg3()
    loader = _tiny_loader()
    pre_gamma = model.bn1.weight.detach().clone()
    cfg = RecalibrateCfg(enabled=True, bn_stats=True, tune_affine=False, epochs=0)
    recalibrate(model, loader, torch.device("cpu"), cfg)
    # Affine gamma untouched when tune_affine is False.
    assert torch.allclose(model.bn1.weight, pre_gamma)
