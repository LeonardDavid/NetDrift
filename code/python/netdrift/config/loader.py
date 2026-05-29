"""YAML config loader with CLI override support.

Loads a YAML file into the typed :class:`ExperimentConfig` dataclass tree.
Overrides have the form ``key.subkey=value`` (e.g. ``fault.rt_error=0.05``);
they apply on top of the YAML values. Lists pass through verbatim if the
value parses as JSON; otherwise as strings.

YAML "include" support is intentionally minimal: a top-level ``defaults``
list at the *root* of the file is merged shallow-into the rest of the file,
giving a way to share fragments under ``configs/_defaults/``.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from netdrift.config.schema import ExperimentConfig


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``. Overlay wins on leaves."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_with_defaults(path: Path) -> dict[str, Any]:
    """Read the YAML, resolve any ``defaults: [path1, path2]`` includes."""
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    defaults = raw.pop("defaults", [])
    if not defaults:
        return raw
    merged: dict[str, Any] = {}
    base_dir = path.parent
    for inc in defaults:
        inc_path = (base_dir / inc).resolve()
        merged = _deep_merge(merged, _load_yaml_with_defaults(inc_path))
    merged = _deep_merge(merged, raw)
    return merged


def _from_dict(cls, data: dict[str, Any]):  # type: ignore[no-untyped-def]
    """Instantiate a (potentially nested) dataclass from a plain dict."""
    if not is_dataclass(cls):
        return data
    # 'lambda' is a Python keyword and can't be a dataclass field name; YAML
    # authors write 'lambda', we store it as 'lambda_'.
    if "lambda" in data and "lambda_" not in data:
        data = {**data, "lambda_": data["lambda"]}
        data.pop("lambda", None)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        # Recurse into nested dataclasses
        if is_dataclass(f.type) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(f.type, val)
        elif isinstance(f.type, type) and is_dataclass(f.type) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(f.type, val)
        else:
            # The fields whose annotation is a class object (resolved at runtime)
            # land here; check by inspecting the default factory if any.
            kwargs[f.name] = _coerce_field(f, val)
    return cls(**kwargs)


def _coerce_field(field_obj, val):  # type: ignore[no-untyped-def]
    """Best-effort coercion: if the default is a dataclass instance, recurse."""
    # ``field.type`` is a string under ``from __future__ import annotations``.
    # Use the default factory to introspect the actual type at runtime.
    if field_obj.default_factory is not None and field_obj.default_factory is not _MISSING:  # type: ignore[comparison-overlap]
        try:
            default = field_obj.default_factory()
            if is_dataclass(default) and isinstance(val, dict):
                return _from_dict(type(default), val)
            if isinstance(default, list) and isinstance(val, list):
                # If list elements are dataclasses, recurse.
                if default and is_dataclass(default[0]) and val and isinstance(val[0], dict):
                    return [_from_dict(type(default[0]), item) for item in val]
                return val
        except TypeError:
            pass
    return val


# Sentinel for absent default_factory — dataclasses uses MISSING; we re-export for clarity.
from dataclasses import MISSING as _MISSING  # noqa: E402


def load(path: str | Path, *, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Args:
        path:      Path to the YAML config.
        overrides: List of ``key.subkey=value`` strings. Applied after YAML.

    Returns:
        Validated :class:`ExperimentConfig`.
    """
    p = Path(path)
    raw = _load_yaml_with_defaults(p)
    if overrides:
        raw = _apply_overrides(raw, overrides)
    return _from_dict(ExperimentConfig, raw)


def parse_overrides(args: list[str]) -> list[str]:
    """Trivial helper: pass through ``key=value`` strings; reject malformed ones."""
    out: list[str] = []
    for a in args:
        if "=" not in a:
            raise ValueError(f"override must have form key=value, got: {a!r}")
        out.append(a)
    return out


def _apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for ov in overrides:
        key, _, val = ov.partition("=")
        path = key.split(".")
        cur = raw
        for p in path[:-1]:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                raise ValueError(f"override path {key!r} traverses non-dict")
        cur[path[-1]] = _parse_override_value(val)
    return raw


def _parse_override_value(val: str) -> Any:
    """Parse a CLI override value: try JSON first (numbers/lists/bools), fall back to string."""
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return val
