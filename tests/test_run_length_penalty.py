"""Run-length (adjacent sign-agreement) penalty."""
from __future__ import annotations

import torch
import torch.nn as nn


def _make_linear_layer(weight: torch.Tensor):
    """Build a QuantizedLinear with a binary scheme and the given latent weight."""
    from netdrift.quant.binary import BinaryScheme
    from netdrift.quant.layers import QuantizedLinear

    out_f, in_f = weight.shape
    layer = QuantizedLinear(in_f, out_f, bias=False)
    layer.attach_scheme(BinaryScheme())
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.rt_mapping = "ROW"
    layer.kernel_mapping = None
    return layer


def test_penalty_zero_for_all_same_sign():
    from netdrift.training.losses import run_length_penalty

    # rt_size=4, one racetrack per row, all positive -> no transitions.
    w = torch.ones(2, 4)
    layer = _make_linear_layer(w)
    model = nn.Module()
    model.fc = layer
    loss = run_length_penalty(model, beta=10.0, rt_size=4, layout="ROW", kernel_mapping="ROW")
    # -mean(tanh*tanh) over same-sign pairs ~ -1.0 (perfect agreement is minimal loss).
    assert loss.item() < -0.99


def test_penalty_higher_for_alternating_than_same_sign():
    from netdrift.training.losses import run_length_penalty

    same = _make_linear_layer(torch.ones(2, 4))
    alt = _make_linear_layer(torch.tensor([[1.0, -1.0, 1.0, -1.0]] * 2))
    m_same, m_alt = nn.Module(), nn.Module()
    m_same.fc = same
    m_alt.fc = alt
    l_same = run_length_penalty(m_same, beta=10.0, rt_size=4, layout="ROW", kernel_mapping="ROW")
    l_alt = run_length_penalty(m_alt, beta=10.0, rt_size=4, layout="ROW", kernel_mapping="ROW")
    assert l_alt.item() > l_same.item()


def test_gradient_pushes_checkerboard_toward_agreement():
    from netdrift.training.losses import run_length_penalty

    w = torch.tensor([[0.5, -0.5, 0.5, -0.5]], requires_grad=True)
    layer = _make_linear_layer(w)
    # Make the leaf param require grad and be the same tensor.
    layer.weight = nn.Parameter(w.detach().clone().requires_grad_(True))
    model = nn.Module()
    model.fc = layer
    loss = run_length_penalty(model, beta=4.0, rt_size=4, layout="ROW", kernel_mapping="ROW")
    loss.backward()
    g = layer.weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert g.abs().sum() > 0  # nonzero gradient


def test_no_pairs_cross_racetrack_boundary():
    from netdrift.training.losses import run_length_penalty

    # rt_size=2, 4 columns => two racetracks per row: [a b][c d].
    # A boundary sign change (b vs c) must NOT be penalized.
    # within-track all same sign, only boundary flips:
    w = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    layer = _make_linear_layer(w)
    model = nn.Module()
    model.fc = layer
    loss = run_length_penalty(model, beta=10.0, rt_size=2, layout="ROW", kernel_mapping="ROW")
    # Both tracks internally same-sign -> minimal loss ~ -1.0 despite boundary flip.
    assert loss.item() < -0.99


def test_protected_layers_excluded():
    from netdrift.training.losses import run_length_penalty

    alt = _make_linear_layer(torch.tensor([[1.0, -1.0, 1.0, -1.0]]))
    alt.protected = True
    model = nn.Module()
    model.fc = alt
    loss = run_length_penalty(model, beta=10.0, rt_size=4, layout="ROW", kernel_mapping="ROW")
    # No unprotected layers contribute -> zero penalty by convention.
    assert loss.item() == 0.0
