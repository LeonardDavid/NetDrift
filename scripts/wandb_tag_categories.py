#!/usr/bin/env python
"""Backfill a ``category`` tag + config field onto finished W&B runs.

The comparison-DB drivers encode the category in each run's NAME (e.g.
``..._cat4_recal-bn-affine_seed707-rt3e-07``, ``..._cat5_lam0p05_..._test-rt...``,
``..._gl1p0_lo1p0_...`` = vanilla endlen). W&B has no native "category" axis, so
this one-time script walks every run in a project, infers the category from the
name, and writes it back as BOTH:

  * a wandb tag           (``cat1_baseline`` … ``cat8_ste``) — for quick filtering
  * a config field        (``category``)                     — for native Group-by

After running it, in the W&B UI: Group by → config → ``category`` puts every
mode in its own group, across the whole project. Idempotent: re-running updates
the same fields, no duplicates.

Usage::

    # dry-run: print the inferred category for every run, change nothing
    python scripts/wandb_tag_categories.py --project netdrift-modes-vgg3 --dry-run

    # apply (writes tag + config.category to every run)
    python scripts/wandb_tag_categories.py --project netdrift-modes-vgg3
    python scripts/wandb_tag_categories.py --project netdrift-modes-vgg7

Requires ``wandb`` installed and ``wandb login`` already done (same creds the
runs were logged with). Pass --entity if your runs live under a team.
"""
from __future__ import annotations

import argparse
import re
import sys

# Ordered (label, regex) pairs. FIRST match wins, so put the specific
# cat-tagged patterns before the structural endlen patterns. Matched against the
# run name (case-insensitive). The drivers stamp explicit _catN_ tokens for
# 4/5/6/7/8; categories 1/2/3 are inferred structurally.
_RULES: list[tuple[str, str]] = [
    ("cat4_endlen_recal", r"_cat4_recal"),
    ("cat6_reg_recal", r"_cat6_reg-recal"),
    ("cat7_reg_endlen", r"_cat7_reg-endlen"),
    ("cat8_ste_inject", r"_cat8_ste"),
    ("cat5_regularizer", r"_cat5_lam"),
    ("cat1_baseline", r"baseline"),
    # Vanilla endlen = unbudgeted cell of the endlen grid (global=local=1.0).
    ("cat2_vanilla_endlen", r"__gl1p0_lo1p0(_|$|-)"),
    # Any other budgeted-endlen grid cell.
    ("cat3_budgeted_endlen", r"__gl\d|__lo\d|sc-(layer|racetrack|channel)|sel-(greedy|value_per_flip|magnitude_aware)"),
]
_COMPILED = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in _RULES]


def infer_category(run_name: str) -> str | None:
    """Return the category label for a run name, or None if nothing matches."""
    for label, rx in _COMPILED:
        if rx.search(run_name):
            return label
    return None


# Subcategory = exact setting combo, parsed from the name. Mirrors what the
# drivers now emit live via --wandb-subcategory, so backfilled (weekend) runs
# and future live-tagged runs share one subcategory axis. Patterns capture the
# combo token sequence; the W&B run-name suffix (-rt<val> / -train) is ignored.
def infer_subcategory(run_name: str, category: str) -> str | None:
    """Best-effort subcategory from a run name, given its inferred category."""
    name = run_name
    # Strip the runner's per-run suffix so the combo token is at the tail.
    name = re.sub(r"-rt[0-9.eE+-]+$", "", name)
    name = re.sub(r"-train$", "", name)

    if category == "cat3_budgeted_endlen":
        m = re.search(r"(gl\w+?_lo\w+?(?:_sc-\w+)?(?:_sel-\w+)?)$", name)
        return f"cat3_budgeted_endlen_{m.group(1)}" if m else None
    if category == "cat2_vanilla_endlen":
        return "cat2_vanilla_endlen_gl1p0_lo1p0"
    if category == "cat4_endlen_recal":
        if "recal-bn-affine" in name:
            return "cat4_recal-bn-affine"
        if "recal-bn" in name:
            return "cat4_recal-bn"
        return None
    if category == "cat5_regularizer":
        m = re.search(r"_(lam\w+?(?:_inj-\w+)?)(?:_seed\d+)?(?:_test)?$", name)
        return f"cat5_regularizer_{m.group(1)}" if m else None
    if category == "cat7_reg_endlen":
        m = re.search(r"(lo\w+?)(?:_seed\d+)?$", name)
        return f"cat7_reg_endlen_{m.group(1)}" if m else None
    # cat1/cat6/cat8 are single-variant — subcategory == category is fine.
    if category in ("cat1_baseline", "cat6_reg_recal", "cat8_ste_inject"):
        return category
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True, help="W&B project name.")
    p.add_argument("--entity", default=None, help="W&B entity/team (optional).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print inferred category per run; write nothing.")
    p.add_argument("--also-train-runs", action="store_true",
                   help="Also tag the cat5/8 *_train runs (default: leave them "
                        "untagged so only the comparable *_test runs carry a "
                        "category). Train runs have no rt_error sweep.")
    args = p.parse_args(argv)

    try:
        import wandb
    except ImportError:
        print("wandb not installed in this env", file=sys.stderr)
        return 2

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = api.runs(path)

    counts: dict[str, int] = {}
    subcounts: dict[str, int] = {}
    unmatched: list[str] = []
    skipped_train = 0
    n_written = 0

    for run in runs:
        name = run.name or ""
        # Skip cat5/8 training runs unless asked — they aren't a comparison cell.
        # The runner names train runs "<experiment.name>-train" and the driver
        # already suffixes the experiment name with "_train", so the W&B run
        # name ends in "_train-train" (and never contains "-rt"). Sweep/test
        # runs are "<...>-rt<value>". Detect train by the "-train" suffix or the
        # "_train" experiment token, AND the absence of an rt sweep suffix.
        is_train = ("-rt" not in name) and ("train" in name.lower())
        if is_train and not args.also_train_runs:
            skipped_train += 1
            continue

        cat = infer_category(name)
        if cat is None:
            unmatched.append(name)
            continue
        counts[cat] = counts.get(cat, 0) + 1
        sub = infer_subcategory(name, cat)
        if sub:
            subcounts[sub] = subcounts.get(sub, 0) + 1

        if args.dry_run:
            continue

        # Write config fields + tags. Group-by uses the CONFIG fields, which
        # have no length cap; tags are a convenience and W&B caps them at 64
        # chars, so the (often long) subcategory is config-only — only the short
        # category is added as a tag.
        run.config["category"] = cat
        if sub:
            run.config["subcategory"] = sub
        tags = set(run.tags or [])
        # Drop any stale cat* tag before re-adding the (short) category tag.
        tags = {t for t in tags if not t.startswith("cat")}
        tags.add(cat)
        run.tags = sorted(tags)
        run.update()
        n_written += 1

    print("=" * 60)
    print(f"project: {path}   ({'DRY-RUN' if args.dry_run else 'APPLIED'})")
    print("=" * 60)
    for cat in sorted(counts):
        print(f"  {cat:24s} {counts[cat]:4d} runs")
    if skipped_train:
        print(f"  (skipped {skipped_train} *_train runs; "
              f"pass --also-train-runs to include)")
    if subcounts:
        print("\n  subcategories (fine grouping within cat3/4/5/7):")
        for sub in sorted(subcounts):
            print(f"    {sub:48s} {subcounts[sub]:4d} runs")
    if unmatched:
        print(f"\n  !! {len(unmatched)} run(s) matched NO category — inspect names:")
        for nm in unmatched[:20]:
            print(f"     {nm}")
        if len(unmatched) > 20:
            print(f"     ... and {len(unmatched) - 20} more")
    if not args.dry_run:
        print(f"\n  wrote category (+subcategory) to {n_written} runs.")
        print("  In the W&B UI: Group by → config → 'category' (coarse) or "
              "'subcategory' (fine); or filter by tag.")
    return 0 if not unmatched else 1


if __name__ == "__main__":
    sys.exit(main())
