#!/usr/bin/env python3
"""Snapshot the best-so-far bundle from a running evolve_pysr run and evaluate it.

Reads runs/<run_id>/run_data.json, finds the highest-scoring bundle across all
generations (even though the run hasn't finalized a best_bundle yet), writes a
snapshot run_data.json with that bundle injected as `best_bundle`, then invokes
evaluate_new_pysr.py with 10 seeds on train/val/barely_unsolvable.

Usage:
    python evaluate_best_so_far.py <run_id_or_run_data_path> [extra args to evaluate_new_pysr.py]

Example:
    sbatch run.sh evaluate_best_so_far.py 499254
    sbatch run.sh evaluate_best_so_far.py 499255
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_best_bundle(data: dict) -> tuple[dict, int]:
    best = None
    best_gen = None
    for g in data.get("generations", []):
        for entry in g.get("population", []):
            if "operators" not in entry:
                continue
            score = entry.get("score")
            if score is None:
                continue
            if best is None or score > best["score"]:
                best = entry
                best_gen = g.get("generation")
    if best is None:
        raise ValueError("No bundle with a score found in any generation")
    return best, best_gen


def resolve_run_data(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    candidate = Path("/home/sca63/meta_sr/runs") / arg / "run_data.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not resolve run_data.json from {arg!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="Run id (e.g. 499254) or path to run_data.json")
    parser.add_argument("--splits", nargs="+",
                        default=["splits/train.txt", "splits/val.txt", "splits/barely_unsolvable.txt"])
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--partition", default="ellis")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: outputs/eval_best_so_far_<run_id>_<gen>)")
    args, passthrough = parser.parse_known_args()

    src = resolve_run_data(args.run)
    run_id = src.parent.name
    print(f"Reading snapshot of {src}")

    with open(src) as f:
        data = json.load(f)

    best, best_gen = find_best_bundle(data)
    score = best["score"]
    op_names = {k: v["name"] for k, v in best["operators"].items() if v is not None}
    print(f"Best bundle: gen={best_gen}, score={score:.4f}")
    for k, v in op_names.items():
        print(f"  {k}: {v}")

    # Build a self-contained bundle dict that matches evaluate_new_pysr.load_evolve_results
    data["best_bundle"] = {
        "operators": best["operators"],
        "score": score,
        "score_vector": best.get("score_vector"),
    }

    out_root = Path(args.output_dir or f"outputs/eval_best_so_far_{run_id}_gen{best_gen}")
    out_root.mkdir(parents=True, exist_ok=True)
    snap_path = out_root / "run_data_snapshot.json"
    with open(snap_path, "w") as f:
        json.dump(data, f)
    print(f"Wrote snapshot: {snap_path}")

    cmd = [
        sys.executable, "-u", "evaluate_new_pysr.py",
        "--evolve-results", str(snap_path),
        "--splits", *args.splits,
        "--n-runs", str(args.n_runs),
        "--seed", str(args.seed),
        "--partition", args.partition,
        "--output-dir", str(out_root),
        *passthrough,
    ]
    print("Running:", " ".join(cmd))
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
