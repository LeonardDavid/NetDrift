"""PPM window-count penalty: train toward sign counts that divide rt_size.

A PPM sort window costs one extra padded wire (or leaves one mixed, exposed wire
when ``pad=false``) exactly when its positive count is not a multiple of
``rt_size``. This penalty is the differentiable form of that objective, so
minimising it during training removes PPM's padding overhead — and, at zero,
makes unpadded PPM immune.

The surrogate is straight-through: the penalty's VALUE is the true integer
distance to the nearest multiple, while its GRADIENT comes from a soft count
``sum(sigmoid(beta*w))``. Because ``d sigmoid/dw`` peaks at ``w = 0``, the
pressure lands on the weights nearest the decision boundary — the cheapest ones
for a BNN to give up. Soft-count-only variants stall (measured: 25% of windows
left non-conforming versus 0%).

The greedy baseline in ``faults/ppm_align.py`` achieves the same objective
optimally on a finished checkpoint; this exists so the network can co-adapt while
the constraint is imposed instead of being rewritten afterwards.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from netdrift.training.losses import ppm_count_penalty

R = 4


def _layer(weight: torch.Tensor, mapping="ROW"):
    from netdrift.quant.binary import BinaryScheme
    from netdrift.quant.layers import QuantizedLinear

    out_f, in_f = weight.shape
    layer = QuantizedLinear(in_f, out_f, bias=False)
    layer.attach_scheme(BinaryScheme())
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.rt_mapping = mapping
    layer.kernel_mapping = None
    return layer


def _model(weight, mapping="ROW"):
    m = nn.Module()
    m.fc = _layer(weight, mapping)
    return m


def _kw(**over):
    kw = dict(beta=20.0, rt_size=R, base_layout="ROW", window=0,
              kernel_mapping="ROW")
    kw.update(over)
    return kw


# --------------------------------------------------------------------------
# Value: the penalty IS the objective
# --------------------------------------------------------------------------

def test_penalty_is_zero_when_every_window_count_divides_rt_size():
    """4 positives of 8 at rt_size=4 -> conforming -> nothing to pay."""
    w = torch.tensor([[0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8]])

    assert ppm_count_penalty(_model(w), **_kw()).item() == 0.0


def test_penalty_is_zero_for_a_uniform_window():
    """All-positive is p == W, a multiple when W is."""
    assert ppm_count_penalty(_model(torch.ones(2, 8)), **_kw()).item() == 0.0


def test_penalty_value_is_the_squared_normalised_distance_to_the_nearest_multiple():
    """5 positives of 8: distance 1 from 4, so (1/R)^2 = 0.0625 at R=4.

    Breaks if the surrogate reports the SOFT count instead of the true one —
    the failure mode that leaves windows non-conforming.
    """
    w = torch.tensor([[0.5, 0.6, 0.7, 0.8, 0.9, -0.6, -0.7, -0.8]])

    got = ppm_count_penalty(_model(w), **_kw()).item()

    assert abs(got - (1.0 / R) ** 2) < 1e-6


def test_penalty_grows_with_distance_from_a_multiple():
    near = torch.tensor([[0.5, 0.6, 0.7, 0.8, 0.9, -0.6, -0.7, -0.8]])   # 5 -> d=1
    far = torch.tensor([[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -0.7, -0.8]])     # 6 -> d=2

    assert (ppm_count_penalty(_model(far), **_kw()).item()
            > ppm_count_penalty(_model(near), **_kw()).item())


# --------------------------------------------------------------------------
# Gradient: it must move the cheapest weights, in the cheaper direction
# --------------------------------------------------------------------------

def test_gradient_pushes_the_smallest_magnitude_weight_of_the_giving_side():
    """5 positives of 8 need one to give way; the gradient must be largest on
    the smallest positive (0.05), not on the confident ones.
    """
    w = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.05, -0.6, -0.7, -0.8]])
    model = _model(w)

    ppm_count_penalty(model, **_kw()).backward()
    g = model.fc.weight.grad[0]

    assert g[4].abs() > g[:4].abs().max()
    assert g[4] > 0            # positive gradient => descent lowers it toward -


def test_gradient_is_zero_for_a_conforming_layer():
    """No pressure once the constraint is met: the task loss gets the network back."""
    w = torch.tensor([[0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8]])
    model = _model(w)

    ppm_count_penalty(model, **_kw()).backward()

    assert torch.count_nonzero(model.fc.weight.grad) == 0


def test_descent_drives_a_random_layer_to_full_conformance():
    """End to end on the surrogate: optimise the penalty alone and every window
    must end up conforming, which is what makes unpadded PPM immune.
    """
    from netdrift.faults.purity import wire_purity

    torch.manual_seed(0)
    w0 = torch.rand(8, 16) * 2 - 1
    model = _model(w0.clone())
    assert wire_purity(w0, R, "POLARITY", polarity_params=(0, False)).mixed > 0

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        ppm_count_penalty(model, **_kw()).backward()
        opt.step()

    final = model.fc.weight.detach()
    assert ppm_count_penalty(model, **_kw()).item() == 0.0
    assert wire_purity(final, R, "POLARITY", polarity_params=(0, False)).mixed == 0
    # ...and it got there by flipping few weights, all of them low-magnitude ones.
    flipped = (final > 0) != (w0 > 0)
    assert int(flipped.sum()) <= 8 * 2          # <= 2 per window of 16
    assert w0.abs()[flipped].max() < 0.5


# --------------------------------------------------------------------------
# Layout is a real parameter
# --------------------------------------------------------------------------

def test_base_layout_changes_which_windows_are_counted():
    """col counts down the transposed view. A weight matrix conforming in ROW
    need not conform in COL, so the two must disagree.

    (8, 4) with the first 5 rows positive: every ROW window holds 4 or 0
    positives (conforming at R=4), while every COL window holds 5 of 8 (not).
    """
    w = torch.cat([torch.full((5, 4), 0.5), torch.full((3, 4), -0.5)])
    model = _model(w)

    row = ppm_count_penalty(model, **_kw(base_layout="ROW")).item()
    col = ppm_count_penalty(model, **_kw(base_layout="COL")).item()

    assert row == 0.0
    assert col > 0.0


def test_window_parameter_changes_the_counting_span():
    """Two windows of 4 (K=1) versus one of 8 (channel-aligned): the same
    weights conform at K=0 and not at K=1.
    """
    w = torch.tensor([[0.5, 0.6, 0.7, -0.8, 0.9, -0.6, -0.7, -0.8]])   # p=4 overall
    model = _model(w)

    assert ppm_count_penalty(model, **_kw(window=0)).item() == 0.0
    assert ppm_count_penalty(model, **_kw(window=1)).item() > 0.0


def test_windows_narrower_than_rt_size_are_skipped():
    """A window of W < R can only conform by going all-negative (vgg7's fc2).
    Charging for it would drag the whole layer to one sign.
    """
    w = torch.tensor([[0.5, -0.6, 0.7]])       # W=3 < R=4

    assert ppm_count_penalty(_model(w), **_kw()).item() == 0.0


def test_conv_weights_are_handled():
    from netdrift.quant.binary import BinaryScheme
    from netdrift.quant.layers import QuantizedConv2d

    torch.manual_seed(1)
    conv = QuantizedConv2d(4, 8, kernel_size=3, bias=False)
    conv.attach_scheme(BinaryScheme())
    conv.rt_mapping = "COL"
    conv.kernel_mapping = "ROW"
    model = nn.Module()
    model.conv = conv

    loss = ppm_count_penalty(model, **_kw(base_layout="COL"))
    loss.backward()

    assert loss.item() > 0.0
    assert torch.count_nonzero(conv.weight.grad) > 0


# --------------------------------------------------------------------------
# Layer selection + degenerate cases
# --------------------------------------------------------------------------

def test_protected_layers_are_excluded():
    """Matches run_length_penalty: a protected layer can never be hit by a
    fault, so flipping its signs costs accuracy for no robustness gain.
    """
    torch.manual_seed(2)
    model = _model(torch.rand(4, 16) * 2 - 1)
    assert ppm_count_penalty(model, **_kw()).item() > 0.0

    model.fc.protected = True

    assert ppm_count_penalty(model, **_kw()).item() == 0.0


def test_returns_zero_tensor_when_no_layer_contributes():
    loss = ppm_count_penalty(nn.Module(), **_kw())

    assert loss.item() == 0.0
    assert loss.dim() == 0


# --------------------------------------------------------------------------
# Config + training-loop wiring
# --------------------------------------------------------------------------

def test_reg_objective_defaults_to_run_length_so_existing_configs_are_unchanged():
    from netdrift.config.schema import RegCfg

    assert RegCfg().objective == "run_length"


def test_reg_config_validates_the_new_fields():
    import pytest

    from netdrift.config.schema import RegCfg

    with pytest.raises(ValueError, match="objective"):
        RegCfg(objective="ppm")
    with pytest.raises(ValueError, match="ppm_window"):
        RegCfg(objective="ppm_count", ppm_window=-1)
    with pytest.raises(ValueError, match="ppm_base_layout"):
        RegCfg(objective="ppm_count", ppm_base_layout="diagonal")
    with pytest.raises(ValueError, match="ppm_beta"):
        RegCfg(objective="ppm_count", ppm_beta=0.0)


def test_nested_reg_overrides_reach_the_dataclass():
    """project_comparison_database records a nested override-path gotcha, and a
    sweep driver will drive this objective, so pin the path.
    """
    from netdrift.config.loader import load

    cfg = load("configs/modes/vgg7_cifar10_w1a1_baseline_col.yaml",
               overrides=["training.fault_aware=regularization",
                          "training.reg.objective=ppm_count",
                          "training.reg.lambda=0.05",
                          "training.reg.ppm_window=2",
                          "training.reg.ppm_base_layout=col",
                          "training.reg.ppm_beta=20"])

    assert cfg.training.reg.objective == "ppm_count"
    assert cfg.training.reg.lambda_ == 0.05
    assert cfg.training.reg.ppm_window == 2
    assert cfg.training.reg.ppm_base_layout == "col"
    assert cfg.training.reg.ppm_beta == 20.0


def test_training_dispatches_on_the_objective():
    """``_reg_penalty`` must pick the objective the config asked for.

    Breaks if the training loop keeps calling run_length_penalty regardless —
    which would train the wrong thing while looking configured.
    """
    from netdrift.config.schema import RegCfg
    from netdrift.training.faultaware import _reg_penalty
    from netdrift.training.losses import ppm_count_penalty, run_length_penalty

    torch.manual_seed(3)
    model = _model(torch.rand(4, 16) * 2 - 1)
    kw = dict(rt_size=R, layout="ROW", kernel_mapping="ROW")

    ppm = _reg_penalty(model, RegCfg(objective="ppm_count", ppm_base_layout="row",
                                     ppm_beta=20.0), **kw)
    rl = _reg_penalty(model, RegCfg(objective="run_length", beta=4.0), **kw)

    assert ppm.item() == ppm_count_penalty(model, beta=20.0, rt_size=R,
                                          base_layout="ROW", window=0,
                                          kernel_mapping="ROW").item()
    assert rl.item() == run_length_penalty(model, beta=4.0, rt_size=R,
                                           layout="ROW",
                                           kernel_mapping="ROW").item()
    assert ppm.item() != rl.item()


def test_a_real_training_epoch_reduces_the_objective():
    """Integration: run the actual fault-aware epoch loop with the new objective
    and confirm the PPM window-count penalty falls.

    This is the path ``netdrift_run.py --override training.fault_aware=
    regularization --override training.reg.objective=ppm_count`` takes, minus
    the dataset. Breaks if the penalty is detached from the graph, if the
    optimizer never sees its gradient, or if the dispatch silently falls back to
    run_length.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from netdrift.config.schema import RegCfg, TrainCfg
    from netdrift.quant.binary import BinaryScheme
    from netdrift.quant.layers import QuantizedLinear
    from netdrift.training.faultaware import train_one_epoch_fault_aware

    torch.manual_seed(4)
    n_classes, n_feat = 4, 16
    model = nn.Sequential()
    for i, (i_f, o_f) in enumerate([(n_feat, 16), (16, n_classes)]):
        layer = QuantizedLinear(i_f, o_f, bias=False)
        layer.attach_scheme(BinaryScheme())
        layer.rt_mapping = "ROW"
        layer.kernel_mapping = None
        layer.layer_name = f"fc{i}"
        model.add_module(f"fc{i}", layer)

    loader = DataLoader(TensorDataset(torch.randn(32, n_feat),
                                      torch.randint(0, n_classes, (32,))),
                        batch_size=8)
    cfg = TrainCfg(
        mode="train", fault_aware="regularization", criterion="hinge",
        fault_aware_criterion="hinge",
        reg=RegCfg(lambda_=5.0, objective="ppm_count", ppm_base_layout="row",
                   ppm_window=0, ppm_beta=20.0),
    )
    kw = dict(rt_size=R, layout="ROW", kernel_mapping="ROW")
    penalty_kw = _kw(base_layout="ROW")

    before = ppm_count_penalty(model, **penalty_kw).item()
    assert before > 0.0
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for epoch in range(6):
        train_one_epoch_fault_aware(model, loader, opt, torch.device("cpu"),
                                    cfg=cfg, epoch=epoch, **kw)
    after = ppm_count_penalty(model, **penalty_kw).item()

    assert after < before


def test_the_mode_config_parses_and_selects_the_ppm_objective():
    """The overlay must actually reach the dataclasses (nested-override gotcha)
    and must NOT set storage.layout=polarity, which the runner rejects for
    fault-aware training.
    """
    from netdrift.config.loader import load
    from netdrift.runner.run import _validate_block_layout_combo

    cfg = load("configs/modes/vgg7_cifar10_w1a1_ppmreg.yaml")

    assert cfg.training.fault_aware == "regularization"
    assert cfg.training.reg.objective == "ppm_count"
    assert cfg.training.reg.ppm_base_layout == "col"
    assert cfg.training.reg.ppm_window == 0
    assert cfg.training.reg.ppm_beta == 20.0
    assert cfg.training.reg.lambda_ > 0
    assert cfg.storage.layout == "col"
    _validate_block_layout_combo(cfg)      # must not raise


# --------------------------------------------------------------------------
# Per-epoch observability
# --------------------------------------------------------------------------

def test_ppm_objective_report_counts_nonconforming_windows_and_wire_cost():
    """A training run needs the objective visible per epoch, not just at the end.

    ``nonconforming`` covers the layers the penalty can actually move (protected
    ones are excluded, exactly as in the penalty); the wire counts cover the
    whole model, because that is the deployment cost.
    """
    from types import SimpleNamespace

    from netdrift.config.schema import RegCfg
    from netdrift.faults.ppm_align import align_model_for_ppm
    from netdrift.runner.run import _ppm_objective_report

    torch.manual_seed(5)
    model = _model(torch.rand(8, 16) * 2 - 1)
    cfg = SimpleNamespace(
        storage=SimpleNamespace(rt_size=R, kernel_mapping="row"),
        training=SimpleNamespace(reg=RegCfg(objective="ppm_count",
                                            ppm_base_layout="row", ppm_window=0)),
    )

    before = _ppm_objective_report(model, cfg)
    assert before["ppm_nonconforming_windows"] > 0
    assert before["ppm_padded_wires"] > before["ppm_dense_wires"]
    assert before["ppm_wire_ratio"] > 1.0

    align_model_for_ppm(model, rt_size=R, base_layout="ROW", window=0)
    after = _ppm_objective_report(model, cfg)

    assert after["ppm_nonconforming_windows"] == 0
    assert after["ppm_padded_wires"] == after["ppm_dense_wires"]
    assert after["ppm_wire_ratio"] == 1.0


def test_ppm_objective_report_excludes_protected_layers_from_the_penalised_count():
    """Protected layers cannot be moved by the penalty, so counting them as
    non-conforming would make the number look stuck.
    """
    from types import SimpleNamespace

    from netdrift.config.schema import RegCfg
    from netdrift.runner.run import _ppm_objective_report

    torch.manual_seed(6)
    model = _model(torch.rand(8, 16) * 2 - 1)
    cfg = SimpleNamespace(
        storage=SimpleNamespace(rt_size=R, kernel_mapping="row"),
        training=SimpleNamespace(reg=RegCfg(objective="ppm_count",
                                            ppm_base_layout="row", ppm_window=0)),
    )
    assert _ppm_objective_report(model, cfg)["ppm_nonconforming_windows"] > 0

    model.fc.protected = True
    report = _ppm_objective_report(model, cfg)

    assert report["ppm_nonconforming_windows"] == 0
    # ...but its wires still count toward the deployment cost.
    assert report["ppm_padded_wires"] > report["ppm_dense_wires"]


# --------------------------------------------------------------------------
# lambda warm-up
# --------------------------------------------------------------------------

def test_lambda_warmup_ramps_the_penalty_weight_over_the_first_epochs():
    """Applying full lambda at once to a converged net asks it to migrate ~2.4%
    of its signs immediately — measured to destroy vgg7 (88.19% -> 10.00%) when
    done post-hoc. A ramp lets the task loss compensate as the signs move.
    """
    from netdrift.config.schema import RegCfg
    from netdrift.training.faultaware import _effective_lambda

    reg = RegCfg(lambda_=10.0, objective="ppm_count", lambda_warmup_epochs=4)

    assert _effective_lambda(reg, epoch=1) == 2.5
    assert _effective_lambda(reg, epoch=2) == 5.0
    assert _effective_lambda(reg, epoch=4) == 10.0
    assert _effective_lambda(reg, epoch=9) == 10.0     # clamped, never overshoots


def test_no_warmup_is_the_default_and_keeps_lambda_constant():
    from netdrift.config.schema import RegCfg
    from netdrift.training.faultaware import _effective_lambda

    reg = RegCfg(lambda_=10.0)

    assert reg.lambda_warmup_epochs == 0
    assert _effective_lambda(reg, epoch=1) == 10.0


def test_warmup_epochs_must_be_non_negative():
    import pytest

    from netdrift.config.schema import RegCfg

    with pytest.raises(ValueError, match="lambda_warmup_epochs"):
        RegCfg(lambda_warmup_epochs=-1)


def test_the_training_loop_uses_the_ramped_lambda():
    """Integration: epoch 1 of a 10-epoch warm-up must apply a tenth of lambda,
    so the first epoch's total loss is closer to the plain task loss.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from netdrift.config.schema import RegCfg, TrainCfg
    from netdrift.quant.binary import BinaryScheme
    from netdrift.quant.layers import QuantizedLinear
    from netdrift.training.faultaware import train_one_epoch_fault_aware

    def _run(warmup):
        torch.manual_seed(7)
        model = nn.Sequential()
        layer = QuantizedLinear(16, 4, bias=False)
        layer.attach_scheme(BinaryScheme())
        layer.rt_mapping = "ROW"
        layer.kernel_mapping = None
        model.add_module("fc", layer)
        loader = DataLoader(TensorDataset(torch.randn(16, 16),
                                          torch.randint(0, 4, (16,))), batch_size=8)
        cfg = TrainCfg(mode="train", fault_aware="regularization",
                       criterion="hinge", fault_aware_criterion="hinge",
                       reg=RegCfg(lambda_=100.0, objective="ppm_count",
                                  ppm_base_layout="row", ppm_beta=20.0,
                                  lambda_warmup_epochs=warmup))
        opt = torch.optim.SGD(model.parameters(), lr=0.0)   # freeze: isolate the loss value
        return train_one_epoch_fault_aware(model, loader, opt, torch.device("cpu"),
                                           rt_size=R, layout="ROW",
                                           kernel_mapping="ROW", cfg=cfg, epoch=1)

    assert _run(warmup=10) < _run(warmup=0)


# --------------------------------------------------------------------------
# Checkpoint selection on the objective
# --------------------------------------------------------------------------

def test_ppm_checkpoint_selection_prefers_fewer_nonconforming_windows():
    """Accuracy-only selection cannot retrieve the objective's best epoch.

    Observed on a real run: epoch 6 held 2,275 non-conforming windows at 84.47%
    while epoch 9 held 2,881 at 86.70%. Accuracy selection keeps epoch 9, so the
    area-optimal model is lost unless it is saved on its own criterion.
    """
    from netdrift.runner.run import _ppm_checkpoint_is_better

    # fewer windows wins, whatever the accuracy
    assert _ppm_checkpoint_is_better((2275, 84.47), (2881, 86.70))
    assert not _ppm_checkpoint_is_better((2881, 86.70), (2275, 84.47))
    # ties broken by accuracy
    assert _ppm_checkpoint_is_better((2275, 85.0), (2275, 84.47))
    assert not _ppm_checkpoint_is_better((2275, 84.0), (2275, 84.47))
    # first epoch always wins against "nothing yet"
    assert _ppm_checkpoint_is_better((9999, 10.0), None)


def test_the_run_banner_shows_pad_and_window_for_polarity_runs(capsys):
    """Arms that differ ONLY in storage.partition.pad print an identical banner
    otherwise, so the padded and unpadded runs are indistinguishable in a log.

    Breaks if the storage line goes back to layout/rt_size/kernel_mapping only.
    """
    from pathlib import Path

    from netdrift.config.loader import load
    from netdrift.runner.run import _print_config_summary

    cfg = load("configs/modes/vgg7_cifar10_w1a1_polarity.yaml",
               overrides=["storage.partition.pad=false",
                          "storage.partition.window=2"])
    _print_config_summary(cfg, Path("/tmp/x"))

    line = [l for l in capsys.readouterr().out.splitlines()
            if l.startswith("storage")][0]
    assert "pad=False" in line
    assert "window=2" in line


def test_the_banner_omits_partition_for_non_polarity_runs(capsys):
    """Same omit-when-irrelevant rule as the metrics meta block and units_* keys."""
    from pathlib import Path

    from netdrift.config.loader import load
    from netdrift.runner.run import _print_config_summary

    _print_config_summary(load("configs/modes/vgg7_cifar10_w1a1_baseline_col.yaml"),
                          Path("/tmp/x"))

    line = [l for l in capsys.readouterr().out.splitlines()
            if l.startswith("storage")][0]
    assert "pad=" not in line and "window=" not in line
