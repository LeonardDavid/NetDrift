"""Thin Weights & Biases wrapper for the NetDrift runner.

wandb is *optional* and *off by default*. The runner only enables it when the
user passes ``--wandb-project NAME`` on the command line; without that flag the
runner uses a :class:`_NullRun` whose methods are all no-ops, so ``run.py`` never
has to branch around individual ``log`` calls and the ``wandb`` package is never
imported (and therefore not required) for ordinary runs.

Typical use::

    run = init_wandb_run(
        project=args.wandb_project, entity=args.wandb_entity,
        group=group, name=name, config=cfg_dict,
    )
    run.set_summary({"baseline_clean_accuracy": 91.2})
    for loop_idx, acc in enumerate(accs, start=1):
        run.log({"loop_idx": loop_idx, "accuracy": acc}, step=loop_idx)
    run.finish()
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol


class WandbRunLike(Protocol):
    """Structural type implemented by both backends."""

    def log(self, data: Mapping[str, Any], step: Optional[int] = None) -> None: ...
    def set_summary(self, mapping: Mapping[str, Any]) -> None: ...
    def update_config(self, mapping: Mapping[str, Any]) -> None: ...
    def finish(self) -> None: ...


class _NullRun:
    """No-op stand-in used when wandb is disabled. Every method does nothing."""

    enabled = False

    def log(self, data: Mapping[str, Any], step: Optional[int] = None) -> None:
        return None

    def set_summary(self, mapping: Mapping[str, Any]) -> None:
        return None

    def update_config(self, mapping: Mapping[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbRun:
    """Wraps a single live ``wandb.run`` with a small, stable surface."""

    enabled = True

    def __init__(self, run: Any) -> None:
        self._run = run

    def log(self, data: Mapping[str, Any], step: Optional[int] = None) -> None:
        # wandb treats ``step`` as a monotonic global counter; passing the
        # explicit per-iteration index keeps the x-axis stable and lets the UI
        # plot any logged key against it.
        self._run.log(dict(data), step=step)

    def set_summary(self, mapping: Mapping[str, Any]) -> None:
        for k, v in mapping.items():
            self._run.summary[k] = v

    def update_config(self, mapping: Mapping[str, Any]) -> None:
        self._run.config.update(dict(mapping), allow_val_change=True)

    def finish(self) -> None:
        self._run.finish()


def init_wandb_run(
    *,
    project: Optional[str],
    entity: Optional[str] = None,
    group: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    tags: Optional[list[str]] = None,
) -> WandbRunLike:
    """Create a :class:`WandbRun`, or a :class:`_NullRun` when ``project`` is unset.

    ``wandb`` is imported lazily here so the dependency is only needed when the
    user actually opted into tracking. Each call starts a fresh run with
    ``reinit=True`` so one process can open several runs in sequence (one per
    ``rt_error`` in a sweep). ``tags`` (e.g. the comparison-DB category) are
    attached so runs are filterable/groupable in the UI.
    """
    if not project:
        return _NullRun()

    import wandb  # lazy: only imported when tracking is enabled

    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        config=dict(config) if config is not None else None,
        tags=tags or None,
        reinit=True,
    )
    return WandbRun(run)
