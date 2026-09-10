#!/usr/bin/env python3
"""Compare complete no-early-stop GT runs using persisted search-only timing."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def load_times(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if not manifest.get("no_early_stop") or manifest["max_evals"] != 1_000_000:
        raise ValueError(f"{run_dir}: expected no early stopping and 1M max evals")
    expected = {(dataset, seed, float(noise))
                for dataset in manifest["datasets"]
                for seed in manifest["seeds"]
                for noise in manifest["noise_levels"]}
    times = {}
    for batch in manifest["batches"]:
        directory = run_dir / batch["batch_dir"]
        tasks = json.loads((directory / "tasks.json").read_text())
        for index, task in enumerate(tasks):
            key = (task["dataset_name"], int(task["seed"]) + int(task.get("run_index", 0)),
                   float(task.get("target_noise", 0)))
            if key not in expected:
                continue
            result = json.loads((directory / "results" / f"task_{index:06d}.json").read_text())
            seconds = result.get("search_runtime_seconds")
            if (result.get("error") or result.get("timed_out") or seconds is None
                    or not math.isfinite(seconds) or seconds <= 0):
                raise ValueError(f"{run_dir}: invalid or incomplete timing for {key}")
            if key in times:
                raise ValueError(f"{run_dir}: duplicate trial {key}")
            times[key] = seconds
    if times.keys() != expected:
        raise ValueError(f"{run_dir}: missing {len(expected - times.keys())} trials")
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=Path("runs/srbench_gt_baseline_1m_noearly_1seed"))
    parser.add_argument("--evolved", type=Path,
                        default=Path("runs/709715/srbench_gt_1m_noearly_1seed"))
    args = parser.parse_args()
    baseline, evolved = load_times(args.baseline), load_times(args.evolved)
    if baseline.keys() != evolved.keys():
        raise ValueError("The two evaluations must have identical task/seed/noise grids")
    print(f"Search-only seconds per run; {len(baseline)} matched trials; 1M max evals; no early stopping")
    print("| Noise | PySR | 709715 |")
    print("|---|---:|---:|")
    for noise in [None] + sorted({key[2] for key in baseline}):
        keys = [key for key in baseline if noise is None or key[2] == noise]
        label = "All" if noise is None else f"{noise:g}"
        print(f"| {label} | {mean(baseline[k] for k in keys):.2f} | {mean(evolved[k] for k in keys):.2f} |")


if __name__ == "__main__":
    main()
