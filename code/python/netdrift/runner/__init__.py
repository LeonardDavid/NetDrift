"""Experiment runner.

Single CLI entry point::

    python -m netdrift.runner.run --config configs/<name>.yaml [--override key=value ...]

The runner orchestrates: load config → build datasets → build model →
replace with quantized → load checkpoint → attach fault model → train or
test → emit metrics. Each step is a small helper in this package.
"""
