#!/usr/bin/env python3
"""Build the LSR-Synth evolution splits (20 train / 20 val) from a 90 s difficulty run.

Train = "barely unsolvable" at the evolution budget, by analogy with
splits/barely_unsolvable.txt: problems the starting bundle (709715, full opset)
is close to solving at 90 s. Priority tiers, filled in order until 20:
  1. 709715 solves 1/3 seeds
  2. 709715 solves 2/3 seeds
  3. 709715 solves 0/3 but the baseline solves >= 1/3
  4. 709715 solves 0/3, closest first (lowest median affine ID-test NMSE)
A tier that overflows is sampled at random (seeded). Train need not be balanced
across domains.
Val = 5 random problems per domain from those not in train (seeded).
"Solved" = the benchmark metric (lsr_synth.SOLVED_KEY on the ID test, lowest
train-NMSE frontier member; affine rescale for the evolved bundle, raw for the
baseline, exactly as scripts/lsrsynth_summarize.py reports).

    python scripts/lsrsynth_make_splits.py runs/lsrsynth_fullops_11-13_3seed_90s
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from lsr_synth import DOMAINS, SOLVED_KEY, dataset_name, load_manifest  # noqa: E402
from lsrsynth_summarize import pick  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--val-per-domain", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="splits/lsr_synth")
    args = ap.parse_args()

    solved = defaultdict(lambda: defaultdict(list))   # method -> key -> [bool]
    nmse = defaultdict(list)                           # key -> evolved affine ID nmse
    for f in sorted(Path(args.run_dir, "results").glob("*.json")):
        r = json.loads(f.read_text())
        if not r.get("completed"):
            continue
        key = (r["domain"], r["name"])
        s = pick(r, affine=(r["method"] == "evolve"))
        solved[r["method"]][key].append(bool(s and s["id"][SOLVED_KEY]))
        if r["method"] == "evolve" and s:
            nmse[key].append(s["id"]["nmse"])

    keys = [(m["domain"], m["name"]) for m in load_manifest()]
    missing = [k for k in keys if len(solved["evolve"][k]) < 3]
    if missing:
        print(f"WARNING: {len(missing)} problems have <3 evolved seeds: {missing[:5]}...")
    ke = {k: sum(solved["evolve"][k]) for k in keys}
    kb = {k: sum(solved["baseline"][k]) for k in keys}
    med = {k: float(np.median(nmse[k])) if nmse[k] else float("inf") for k in keys}

    tiers = [
        [k for k in keys if ke[k] == 1],
        [k for k in keys if ke[k] == 2],
        [k for k in keys if ke[k] == 0 and kb[k] >= 1],
        sorted([k for k in keys if ke[k] == 0 and kb[k] == 0], key=lambda k: med[k]),
    ]
    rng = random.Random(args.seed)
    train = []
    for i, tier in enumerate(tiers):
        room = args.n_train - len(train)
        if room <= 0:
            break
        take = tier[:room] if i == 3 else (tier if len(tier) <= room else rng.sample(tier, room))
        train += take
        print(f"tier {i + 1}: {len(tier)} candidates, took {len(take)}")
    rest = [k for k in keys if k not in train]
    val = []
    for d in DOMAINS:
        pool = sorted(k for k in rest if k[0] == d)
        val += rng.sample(pool, args.val_per_domain)

    for name, ks in (("train", train), ("val", val)):
        path = REPO / f"{args.out_prefix}_{name}.txt"
        path.write_text("".join(dataset_name(*k) + "\n" for k in ks))
        print(f"\n{path.relative_to(REPO)} ({len(ks)}):")
        for k in ks:
            print(f"  {dataset_name(*k):28s} 709715 {ke[k]}/3  baseline {kb[k]}/3  "
                  f"median NMSE {med[k]:.2e}")
    print("\nall-problem 90 s solve counts (709715):",
          {d: [ke[k] for k in keys if k[0] == d].count(3) for d in DOMAINS}, "always solved per domain")


if __name__ == "__main__":
    main()
