#!/usr/bin/env python
"""Tiny launcher for the NetDrift runner.

Lets you invoke the runner from the repo root without setting ``PYTHONPATH``::

    python netdrift_run.py --config configs/vgg7_cifar10_rtm.yaml

Equivalent to ``PYTHONPATH=code/python python -m netdrift.runner.run`` but
avoids the env-var dance.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _main() -> int:
    src = Path(__file__).resolve().parent / "code" / "python"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from netdrift.runner.run import main as runner_main
    return runner_main()


if __name__ == "__main__":
    sys.exit(_main())
