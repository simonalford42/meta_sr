"""Per-split (train/val/test) solve rates from srbench_full_eval results.

Usage: python scripts/split_scores_from_full_eval.py RESULTS_DIR TRAIN_SPLIT VAL_SPLIT
Test = all evaluated tasks not in train or val. Score = mean `solved` over all
(seed, noise) runs of the tasks in the subset (equal weight per task).
"""
import json, sys
from collections import defaultdict

def split_scores(results_dir, train_split, val_split):
    res = json.load(open(f"{results_dir}/srbench_full_results.json"))["results"]
    per_task = defaultdict(list)
    for r in res.values():
        per_task[r["dataset"]].append(float(bool(r.get("solved"))))
    tr = set(open(train_split).read().split()); va = set(open(val_split).read().split())
    subsets = {"train": tr, "val": va, "test": set(per_task) - tr - va}
    out = {}
    for name, s in subsets.items():
        vals = [sum(per_task[t]) / len(per_task[t]) for t in s if t in per_task]
        out[name] = (sum(vals) / len(vals), len(vals))
    return out

if __name__ == "__main__":
    for k, (v, n) in split_scores(*sys.argv[1:4]).items():
        print(f"{k}: {v:.3f} (n={n})")
