#!/usr/bin/env python3
"""Verify COL-layout metrics integrity from a runner metrics dir.

The user's explicit concern: under column-wise mapping there is a DIFFERENT
number of racetracks per layer, so the metrics must (a) report the COL racetrack
geometry, not the ROW one, and (b) keep the standing-corruption count bounded by
that geometry (the `affected_units > racetracks` / `BER > 1` class of bug that
bit us before is exactly what COL stresses).

This reads the metrics artifacts written by a `--metrics all` col run and checks:

  1. Every unprotected layer's `meta.layers[].n_racetracks` is the COL grid
     (in_dim, ceil(out_dim/rt_size)) — i.e. the racetrack axis is the OUTPUT
     dim. (We just print row-vs-col so you can eyeball that they differ.)
  2. For each rt_error, total `affected_units` (last-loop snapshot) ≤ total
     racetracks over unprotected layers, and BER ≤ 1.0.

Usage:
    python scripts/verify_col_metrics.py <run_dir>/metrics_artifacts
    # <run_dir> is the timestamped dir the col cat1 run wrote under runs/.
    # Runs from before the 2026-07-31 rename wrote this as <run_dir>/metrics
    # instead — pass that path for older runs; this script takes the
    # directory explicitly so either name works unmodified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_static(metrics_dir: Path) -> dict:
    cands = sorted(metrics_dir.glob("*__static.json"))
    if not cands:
        raise SystemExit(f"no *__static.json under {metrics_dir} (run with --metrics all)")
    return json.loads(cands[-1].read_text())


def _rt_artifacts(metrics_dir: Path) -> list[Path]:
    # *__rt<...>.json, excluding the static one.
    return sorted(p for p in metrics_dir.glob("*__rt*.json"))


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    metrics_dir = Path(argv[0])
    if not metrics_dir.is_dir():
        raise SystemExit(f"not a dir: {metrics_dir}")

    static = _load_static(metrics_dir)
    meta = static["meta"]
    layout = meta.get("storage", {}).get("layout")
    rt_size = meta.get("storage", {}).get("rt_size")
    print(f"layout (from artifact meta) : {layout}")
    print(f"rt_size                     : {rt_size}")
    if layout != "col":
        print("WARNING: this artifact's layout is not 'col' — run the col config.")

    # ---- Check 1: per-layer racetrack geometry, unprotected only ----
    layers = meta["layers"]
    total_unprot_rt = 0
    print("\nper-layer racetrack geometry (unprotected layers exposed to faults):")
    print(f"  {'layer':32s} {'shape':>22s} {'rt_mapping':>10s} {'n_racetracks':>16s} {'#rt':>10s}")
    for L in layers:
        nrt = L["n_racetracks"]  # [x, y]
        n = int(nrt[0]) * int(nrt[1])
        mark = "" if L["protected"] else "  <- faulted"
        if not L["protected"]:
            total_unprot_rt += n
        print(f"  {L['name']:32s} {str(L['weight_shape']):>22s} "
              f"{L['rt_mapping']:>10s} {str(nrt):>16s} {n:>10d}{mark}")
    print(f"\n  total racetracks over UNPROTECTED layers = {total_unprot_rt}")
    print("  (compare this number against the ROW run's total — they MUST differ)")

    # ---- Check 2: affected_units bounded, BER bounded, per rt_error ----
    print("\nper-rt_error invariants:")
    ok = True
    for art in _rt_artifacts(metrics_dir):
        doc = json.loads(art.read_text())
        rt = doc["meta"].get("rt_error")
        fi = doc.get("fault_incidence", {})
        last = fi.get("last_loop", {})
        au = last.get("affected_units")
        ber = last.get("ber")
        msgs = []
        if au is not None:
            if au > total_unprot_rt:
                ok = False
                msgs.append(f"FAIL affected_units={au} > racetracks={total_unprot_rt}")
            else:
                msgs.append(f"affected_units={au} <= racetracks={total_unprot_rt} OK")
        if ber is not None:
            if ber > 1.0 + 1e-9:
                ok = False
                msgs.append(f"FAIL BER={ber:.4f} > 1.0")
            else:
                msgs.append(f"BER={ber:.4f} <= 1.0 OK")
        print(f"  rt_error={rt!s:>10}  " + "  ".join(msgs))

    print("\n" + ("ALL COL METRIC INVARIANTS HOLD ✓" if ok else "INVARIANT VIOLATED ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
