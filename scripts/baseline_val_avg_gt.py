"""Recreate the baseline val avg_gt under the fixed (per-seed, per-dataset) noise map.

The evolve loop assigns each val dataset a *single* target noise level per seed via
`_stable_target_noise(dataset, seed, [0.0, 0.001, 0.01, 0.1])`. The srbench_full_eval
baseline run (runs/398735) evaluated the default PySR operator on ALL 133 datasets x
10 seeds x 4 noise levels, so for any (dataset, seed) we can look up the gt_match_score
at exactly the noise level the fixed map would have picked.

avg_gt per seed = mean over val datasets of gt_match_score (missing -> 0.0, same fill
policy as evolve's fitness_metric="gt"). Final number = mean over the 10 seeds.
"""
import json
from pathlib import Path

from utils import load_dataset_names_from_split
from evolution_helpers import TARGET_NOISE_LEVELS, _stable_target_noise
import srbench_results_io as srio

BASELINE_RUN = "runs/398735"
VAL_SPLIT = "splits/val.txt"
SEEDS = list(range(42, 52))  # 10 seeds, matching manifest

val_datasets = load_dataset_names_from_split(VAL_SPLIT)
results = srio.load_keyed_results(BASELINE_RUN)
assert results is not None, f"no srbench_full_results.json in {BASELINE_RUN}"

per_seed_avg = []
missing = []
print(f"Val datasets: {len(val_datasets)} | seeds: {SEEDS[0]}..{SEEDS[-1]}")
print(f"Noise levels: {TARGET_NOISE_LEVELS}\n")

for seed in SEEDS:
    scores = []
    for ds in val_datasets:
        noise = _stable_target_noise(ds, seed, TARGET_NOISE_LEVELS)
        key = srio.result_key(ds, seed, noise)
        entry = results.get(key)
        if entry is None or not entry["present"] or entry["error"] is not None:
            missing.append((ds, seed, noise))
            gt = 0.0
        else:
            gt = entry["gt_match_score"]
            gt = 0.0 if gt is None else float(gt)
        scores.append(gt)
    avg = sum(scores) / len(scores)
    per_seed_avg.append(avg)
    print(f"  seed {seed}: avg_gt = {avg:.4f}")

overall = sum(per_seed_avg) / len(per_seed_avg)
print(f"\nBaseline val avg_gt (fixed noise map, mean over {len(SEEDS)} seeds): {overall:.4f}")
print(f"  min/max per-seed: {min(per_seed_avg):.4f} / {max(per_seed_avg):.4f}")
if missing:
    print(f"\n[warn] {len(missing)} missing (dataset,seed,noise) lookups (counted as gt=0):")
    for ds, seed, noise in missing[:20]:
        print(f"    {ds} | seed {seed} | noise {noise}")
