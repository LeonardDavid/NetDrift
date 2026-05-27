"""wandb logger wrapper: null backend, live backend, lazy import.

CPU-safe — no GPU work, and no real wandb package is imported. The "live"
backend is exercised against a fake ``wandb`` module injected via ``sys.modules``
so the test asserts the wrapper forwards the right payloads without any network.
"""

from __future__ import annotations

import sys
import types

import pytest

from netdrift.runner.wandb_logger import WandbRun, _NullRun, init_wandb_run


def test_null_run_is_a_noop() -> None:
    run = _NullRun()
    assert run.enabled is False
    # None of these should raise or do anything observable.
    run.log({"accuracy": 1.0}, step=1)
    run.set_summary({"baseline_clean_accuracy": 90.0})
    run.update_config({"model": "vgg3"})
    run.finish()


def test_init_returns_null_run_when_project_unset() -> None:
    assert isinstance(init_wandb_run(project=None), _NullRun)
    assert isinstance(init_wandb_run(project=""), _NullRun)


def test_init_does_not_import_wandb_when_disabled(monkeypatch) -> None:
    # Make any attempt to import wandb explode, then confirm the disabled path
    # never triggers it.
    monkeypatch.setitem(sys.modules, "wandb", None)  # import wandb -> ImportError
    run = init_wandb_run(project=None)
    assert isinstance(run, _NullRun)


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int | None]] = []
        self.summary: dict = {}
        self.config = types.SimpleNamespace(
            updates=[],
            update=lambda d, allow_val_change=False: self.config.updates.append(d),
        )
        self.finished = False

    def log(self, data, step=None) -> None:
        self.logged.append((data, step))

    def finish(self) -> None:
        self.finished = True


class _FakeWandb(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.last_init_kwargs: dict | None = None
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.last_init_kwargs = kwargs
        return self.run


@pytest.fixture()
def fake_wandb(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_init_creates_live_run_with_kwargs(fake_wandb) -> None:
    run = init_wandb_run(
        project="proj", entity="org", group="grp", name="run-1",
        config={"model": "vgg3", "rt_error": 0.01},
    )
    assert isinstance(run, WandbRun)
    assert run.enabled is True
    kw = fake_wandb.last_init_kwargs
    assert kw["project"] == "proj"
    assert kw["entity"] == "org"
    assert kw["group"] == "grp"
    assert kw["name"] == "run-1"
    assert kw["config"]["rt_error"] == 0.01
    assert kw["reinit"] is True


def test_live_run_forwards_log_summary_finish(fake_wandb) -> None:
    run = init_wandb_run(project="proj", config={})
    run.log({"loop_idx": 1, "accuracy": 88.5}, step=1)
    run.log({"loop_idx": 2, "accuracy": 80.0}, step=2)
    run.set_summary({"baseline_clean_accuracy": 91.2, "mean_accuracy": 84.25})
    run.update_config({"extra": 1})
    run.finish()

    fr = fake_wandb.run
    assert fr.logged == [
        ({"loop_idx": 1, "accuracy": 88.5}, 1),
        ({"loop_idx": 2, "accuracy": 80.0}, 2),
    ]
    assert fr.summary["baseline_clean_accuracy"] == 91.2
    assert fr.summary["mean_accuracy"] == 84.25
    assert {"extra": 1} in fr.config.updates
    assert fr.finished is True
