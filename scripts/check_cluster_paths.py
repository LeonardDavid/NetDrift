#!/usr/bin/env python
"""Assert NetDrift's resolved paths land on the right Lamarr storage tier.

The Lamarr cluster splits storage into two tiers with very different
guarantees (cluster_setup.md):

  /home/{user}   CephFS. Shared across every node, survives job termination
                 and node failure. Slower I/O.
  /raid/{user}   Node-local SSD. Much faster, NOT shared between nodes, and
                 eventually purged after inactivity or node failure.

Putting a file on the wrong tier fails in one of two ways, and only one of
them is loud:

  * results on /raid  -> SILENT DATA LOSS. The sweep finishes, the node is
    reclaimed, and 47 GPU-hours of summaries are gone. Nothing errors.
  * hot data on /home -> just slow. CIFAR-10 is re-read every loop of every
    cell (30 loops x 5 rt_error x 79 cells), so this one is measurable.

So the rule this script enforces is: anything the sweep CANNOT REGENERATE
goes on /home; anything it can rebuild from scratch goes on /raid.

Run it before launching. It exits non-zero on any violation, so it composes
as a gate::

    python scripts/check_cluster_paths.py --config <cluster.yaml> && \
        python -u scripts/sweep_design_space.py --config <cluster.yaml> --stage main

Deliberately torch/numba-free: it imports only ``netdrift.config``, so it
runs on the login node before a GPU is ever allocated.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code" / "python"))

# torchvision's CIFAR10(root=R) reads R/cifar-10-batches-py/. These are the
# files it expects inside; checking for them is what distinguishes "the data
# is staged" from "the directory happens to exist".
CIFAR10_DIRNAME = "cifar-10-batches-py"
CIFAR10_FILES = ("test_batch", "data_batch_1", "batches.meta")

# Env vars that must redirect scratch/cache writes onto the fast local tier.
# Each entry is (var, why it matters if left unset).
TEMP_ENV_VARS = (
    ("WANDB_DIR", "W&B run cache; 395 runs' worth of staging data"),
    ("NUMBA_CACHE_DIR", "compiled CUDA kernel cache -- on CephFS this is slow, "
                        "and next to the source tree it may not be writable"),
    ("TMPDIR", "general scratch; container /tmp is small"),
)


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []
        self.failed = 0

    def add(self, ok: bool, tier: str, what: str, detail: str) -> None:
        if not ok:
            self.failed += 1
        self.rows.append(("PASS" if ok else "FAIL", tier, what, detail))

    def warn(self, tier: str, what: str, detail: str) -> None:
        self.rows.append(("WARN", tier, what, detail))

    def render(self) -> None:
        w = max(len(r[2]) for r in self.rows) if self.rows else 20
        print(f"\n  {'result':6s} {'tier':5s} {'what':{w}s}  detail")
        for res, tier, what, detail in self.rows:
            print(f"  {res:6s} {tier:5s} {what:{w}s}  {detail}")


def _under(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _writable(p: Path) -> bool:
    """Can we actually create a file here? Walks up to the nearest existing
    ancestor, because a not-yet-created output dir is fine as long as its
    parent accepts a mkdir."""
    probe = p
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return os.access(probe, os.W_OK)


def main(argv: list[str] | None = None) -> int:
    user = os.environ.get("USER", "bereholschi")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Cluster config YAML to check.")
    p.add_argument("--home-root", default=f"/home/{user}",
                   help="Persistent tier root. Default: /home/$USER")
    p.add_argument("--raid-root", default=f"/raid/{user}",
                   help="Node-local fast tier root. Default: /raid/$USER")
    p.add_argument("--skip-env", action="store_true",
                   help="Skip the temp-env-var checks (use when inspecting a "
                        "config from outside the job, where they aren't set yet).")
    args = p.parse_args(argv)

    from netdrift.config import load as load_config

    cfg = load_config(args.config)
    home, raid = Path(args.home_root), Path(args.raid_root)
    r = Report()

    print("=" * 78)
    print("NetDrift cluster path check")
    print("=" * 78)
    print(f"  config     : {args.config}")
    print(f"  home tier  : {home}   (persistent, shared, survives the job)")
    print(f"  raid tier  : {raid}   (node-local, fast, eventually purged)")

    # ---------------------------------------------------------------- /raid
    # Dataset: hot path, must be local. The root/dirname distinction is the
    # bug this check exists to catch -- data_dir is torchvision's ROOT, so the
    # actual data sits one level deeper.
    data_dir = Path(cfg.data.data_dir)
    r.add(_under(data_dir, raid), "raid", "data.data_dir",
          f"{data_dir}" + ("" if _under(data_dir, raid)
                           else f"  <- NOT under {raid}; dataset I/O is on the hot path"))

    cifar = data_dir / CIFAR10_DIRNAME
    if cifar.is_dir():
        missing = [f for f in CIFAR10_FILES if not (cifar / f).exists()]
        r.add(not missing, "raid", f"  {CIFAR10_DIRNAME}/",
              str(cifar) if not missing else f"{cifar}  <- missing {missing}")
    else:
        r.add(False, "raid", f"  {CIFAR10_DIRNAME}/",
              f"{cifar} does not exist. data_dir is torchvision's ROOT and it "
              f"appends '{CIFAR10_DIRNAME}'. rsync the data, or torchvision may "
              f"try to DOWNLOAD it -- which hangs on a node with no internet.")

    if not args.skip_env:
        for var, why in TEMP_ENV_VARS:
            val = os.environ.get(var)
            if not val:
                r.add(False, "raid", f"${var}", f"unset -- {why}")
            else:
                ok = _under(Path(val), raid)
                detail = val if ok else f"{val}  <- NOT under {raid}"
                if ok and not _writable(Path(val)):
                    ok, detail = False, f"{val}  <- not writable"
                r.add(ok, "raid", f"${var}", detail)

    # ---------------------------------------------------------------- /home
    # Everything below is un-regenerable output. On /raid it would vanish
    # silently when the node is reclaimed.
    out_dir = Path(cfg.experiment.output_dir)
    ok = _under(out_dir, home)
    detail = str(out_dir) if ok else f"{out_dir}  <- NOT under {home}; RESULTS WOULD BE PURGED"
    if ok and not _writable(out_dir):
        ok, detail = False, f"{out_dir}  <- not writable"
    r.add(ok, "home", "experiment.output_dir", detail)

    ckpt = Path(cfg.model.checkpoint)
    if not _under(ckpt, home):
        r.add(False, "home", "model.checkpoint",
              f"{ckpt}  <- NOT under {home}; the checkpoint is an input the sweep "
              f"cannot regenerate")
    else:
        r.add(ckpt.exists(), "home", "model.checkpoint",
              str(ckpt) if ckpt.exists() else f"{ckpt} does not exist")

    # Sweep aggregates (CSV + manifest.json) go to REPO_ROOT/runs/sweeps --
    # derived from THIS file's location, not from experiment.output_dir, so a
    # repo checked out on the wrong tier silently relocates them.
    sweeps = REPO_ROOT / "runs" / "sweeps"
    r.add(_under(sweeps, home), "home", "sweep CSV/manifest",
          f"{sweeps}" + ("" if _under(sweeps, home)
                         else f"  <- repo is not under {home}; sweep tables would be purged"))

    # metrics_artifacts/ is nested inside each run dir, so it inherits
    # output_dir's tier. Stated rather than checked, so the coverage is
    # explicit rather than looking like an omission.
    r.warn("home", "metrics_artifacts/",
           "nested per-run under experiment.output_dir -> inherits its tier (above)")

    r.render()

    n_fail = r.failed
    print()
    if n_fail:
        print(f"FAIL: {n_fail} path check(s) failed. Fix these before launching --")
        print("      results on /raid are lost silently when the node is reclaimed.")
    else:
        print("PASS: every path is on the correct tier.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
