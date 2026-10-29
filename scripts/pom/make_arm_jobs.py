#!/usr/bin/env python3
"""Build rerun jobs for genuinely-unsolved baseline tasks.

Arms (each with --seeds fresh PySR seeds, all on the identical training split):
  mse        — plain MSE rerun (controls for "just a new seed")
  stat       — loss = statistic difference only (alpha=1)
  stat_mse   — 0.5 * statistic + 0.5 * standardized MSE
The statistic is the top-z family from select_stats.py (--top-k families).
Tasks whose frontier already contains a numerically exact equation (val R² ≥
1 - 1e-9) are skipped: they are matcher failures, not search failures.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--selection", default="stat_selection_best.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--arms", nargs="+", default=["mse", "stat", "stat_mse"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    base = Path(args.baseline)
    sel = json.loads((base / args.selection).read_text())
    jobs = []
    for ds, s in sel.items():
        r = json.loads((base / f"{ds}__mse__s0.json").read_text())
        best_val = max((f["r2_val"] or -1) for f in r["frontier"])
        if best_val >= 1 - 1e-9:
            continue
        fams = [e["family"] for e in s["ranked"][: args.top_k]]
        for seed in args.seeds:
            for arm in args.arms:
                if arm == "mse":
                    loss = {"type": "mse"}
                else:
                    loss = {"type": "stat", "families": fams,
                            "alpha": 1.0 if arm == "stat" else 0.5}
                jobs.append({"id": f"{ds}__{arm}{args.tag}__s{seed}", "dataset": ds,
                             "seed": seed, "timeout": args.timeout, "arm": arm + args.tag,
                             "loss": loss})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{len(jobs)} jobs, {len(jobs) // (len(args.seeds) * len(args.arms))} tasks")


if __name__ == "__main__":
    main()
