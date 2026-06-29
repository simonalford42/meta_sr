#!/usr/bin/env python3
"""Compare evolved bundle runs/121270 vs runs/538190 on splits/train.txt & val.txt.

- 121270: read its fixed-noise final eval (final_eval_summary.json) directly.
- 538190: it has NO final eval on these splits, but its full-SRBench eval lives in
  runs/737094 (manifest method_meta.source == runs/538190). We derive:
    (a) pooled solve% over all 4 noise levels (the default inspect_srbench metric)
    (b) noise=0 (clean) solve%
    (c) FIXED-noise solve%: for each task use only the runs at the SAME per-task
        noise level that 121270's final eval used (seed=42 stable noise map),
        so the two bundles can be compared apples-to-apples.

solved == gt_match_score >= 1.0 in both pipelines.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import srbench_results_io as srio
from utils import load_dataset_names_from_split
from evolution_helpers import _build_target_noise_map, TARGET_NOISE_LEVELS

ROOT = Path(__file__).resolve().parent.parent
FULL_EVAL_RUN = ROOT / "runs/737094"   # full-SRBench eval of bundle 538190
EVO_SEED = 42                          # 121270 evolution seed -> built the noise map


def solve_rate(entries):
    present = [e for e in entries if e.get("present") and e.get("error") is None]
    if not present:
        return None, 0, 0
    solved = sum(1 for e in present if e["solved"])
    return solved / len(present), solved, len(present)


def subset_entries(keyed, datasets, noise_str=None, per_task_noise=None):
    """Entries restricted to `datasets`, optionally to a single noise level
    (noise_str) or a per-task noise level (per_task_noise: {ds: noise_str})."""
    out = []
    for e in keyed.values():
        if e["dataset"] not in datasets:
            continue
        en = srio.fmt_noise(e["noise"])
        if noise_str is not None and en != noise_str:
            continue
        if per_task_noise is not None and en != per_task_noise[e["dataset"]]:
            continue
        out.append(e)
    return out


def main():
    train = load_dataset_names_from_split("splits/train.txt")
    val = load_dataset_names_from_split("splits/val.txt")

    keyed = srio.load_keyed_results(FULL_EVAL_RUN)

    # Per-task fixed noise level, exactly as 121270's final eval built it (seed 42).
    all_ds = list(dict.fromkeys(train + val))
    noise_map = _build_target_noise_map(all_ds, EVO_SEED, TARGET_NOISE_LEVELS)
    noise_map_str = {ds: srio.fmt_noise(n) for ds, n in noise_map.items()}

    print("Per-task FIXED noise level (seed=42 stable map):")
    for split_name, ds_list in (("train", train), ("val", val)):
        print(f"  [{split_name}]")
        for ds in ds_list:
            print(f"    {ds:24s} noise={noise_map[ds]}")

    print("\n" + "=" * 78)
    print(f"538190 full-SRBench eval ({FULL_EVAL_RUN.name}) -- solve% on train/val")
    print("=" * 78)
    for split_name, ds_list in (("train", set(train)), ("val", set(val))):
        pooled = solve_rate(subset_entries(keyed, ds_list))
        clean = solve_rate(subset_entries(keyed, ds_list, noise_str=srio.fmt_noise(0)))
        fixed = solve_rate(subset_entries(keyed, ds_list, per_task_noise=noise_map_str))
        print(f"\n[{split_name}]  (n_tasks={len(ds_list)})")
        print(f"  pooled (all 4 noise levels): {pooled[0]*100:5.1f}%   ({pooled[1]}/{pooled[2]})")
        print(f"  noise=0 (clean)            : {clean[0]*100:5.1f}%   ({clean[1]}/{clean[2]})")
        print(f"  FIXED per-task noise (=42) : {fixed[0]*100:5.1f}%   ({fixed[1]}/{fixed[2]})")

    # 121270 fixed-noise final eval numbers, straight from its summary.
    print("\n" + "=" * 78)
    print("121270 final eval (fixed-noise, --random-target-noise) -- from summary")
    print("=" * 78)
    summ = json.load(open(ROOT / "runs/121270/final_eval_summary.json"))
    for key, label in (("train", "train"), ("val", "val")):
        block = summ[key]
        gt = sum(block["per_run_gt_avgs"]) / len(block["per_run_gt_avgs"])
        print(f"[{label:5s}] GT(fixed) solve% = {gt*100:5.1f}%   "
              f"(split_name='{block['split_name']}', avg_r2={block['avg_r2']:.4f})")

    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE  (fixed per-task noise, seed=42; gt_match>=1.0)")
    print("=" * 78)
    print(f"{'split':6s} {'121270':>10s} {'538190':>10s}")
    for split_name, ds_list in (("train", set(train)), ("val", set(val))):
        fixed = solve_rate(subset_entries(keyed, ds_list, per_task_noise=noise_map_str))
        gt121 = sum(summ[split_name]["per_run_gt_avgs"]) / len(summ[split_name]["per_run_gt_avgs"])
        print(f"{split_name:6s} {gt121*100:9.1f}% {fixed[0]*100:9.1f}%")


if __name__ == "__main__":
    main()
