#!/usr/bin/env python3
"""Fixed PySR autoresearch evaluation harness.

Quick evaluation uses three paired seeds on the official training split.
Confirmation uses ten fresh seeds on both official train and validation.
The selected SymbolicRegression.jl commit is run in an isolated sandbox.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SUBMODULE = ROOT / "SymbolicRegression.jl"
RESULTS_TSV = ROOT / "autoresearch_sr" / "results.tsv"
SANDBOXES = ROOT / "outputs" / "autoresearch_pysr_sandboxes"
TRAIN = "splits/barely_unsolvable.txt"
VAL = "splits/barely_unsolvable_val2.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a committed SymbolicRegression.jl autoresearch candidate"
    )
    parser.add_argument("--target", default="latest",
                        help="Commit/branch, latest, best, or expN")
    parser.add_argument("--confirm", action="store_true",
                        help="Use 10 fresh seeds and include held-out validation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the protocol seed")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Override the protocol run count")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT))
    from autoresearch_pysr import resolve_commit

    commit = resolve_commit(args.target, SUBMODULE, RESULTS_TSV)
    seed = args.seed if args.seed is not None else (192 if args.confirm else 42)
    n_runs = args.n_runs if args.n_runs is not None else (10 if args.confirm else 3)
    splits = [TRAIN, VAL] if args.confirm else [TRAIN]
    phase = "confirm" if args.confirm else "quick"
    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "autoresearch_sr" / "eval_results" / commit[:12]
        / f"{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    command = [
        sys.executable,
        str(ROOT / "evaluate_new_pysr.py"),
        "--autoresearch", commit,
        "--autoresearch-submodule", str(SUBMODULE),
        "--autoresearch-results", str(RESULTS_TSV),
        "--autoresearch-sandboxes", str(SANDBOXES),
        "--splits", *splits,
        "--n-runs", str(n_runs),
        "--seed", str(seed),
        "--max-evals", "1000000",
        "--max-samples", "1000",
        "--timeout", "500",
        "--pysr-wall-limit", "600",
        "--max-concurrent-jobs", "300",
        "--fitness-metric", "gt",
        "--random-target-noise",
        "--output-dir", str(output_dir),
    ]
    print(f"commit: {commit}", flush=True)
    print(f"phase: {phase}  seed: {seed}  n_runs: {n_runs}", flush=True)
    print("command: " + " ".join(command), flush=True)
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "disabled")
    subprocess.run(command, cwd=ROOT, env=env, check=True)

    summary_path = output_dir / "eval_summary.json"
    summary = json.loads(summary_path.read_text())
    print("\n--- autoresearch result ---")
    print(f"commit:        {commit}")
    print(f"phase:         {phase}")
    for split in splits:
        name = Path(split).stem
        values = summary[name].get("per_run_gt_avgs") or []
        score = float(np.mean(values)) if values else 0.0
        print(f"{name}: {score:.6f}")
    print(f"summary:       {summary_path}")


if __name__ == "__main__":
    main()
