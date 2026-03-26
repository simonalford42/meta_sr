#!/usr/bin/env python3
"""
Launch OpenEvolve for PySR custom mutation evolution.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenEvolve to evolve PySR custom mutations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=50, help="OpenEvolve iterations")
    parser.add_argument("--split", type=str, default="splits/train.txt", help="Dataset split file")
    parser.add_argument("--stage2-datasets", type=int, default=5, help="Datasets used in cascade stage 2")
    parser.add_argument("--fitness-metric", type=str, default="gt", choices=["r2", "gt"], help="Final-stage fitness metric")
    parser.add_argument("--n-runs", type=int, default=1, help="Runs per dataset during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--data-seed", type=int, default=42, help="Dataset subsampling seed")
    parser.add_argument("--target-noise", type=float, default=0.0, help="Target noise level")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples per dataset")
    parser.add_argument("--max-evals", type=int, default=100000, help="PySR max_evals")
    parser.add_argument("--timeout", type=int, default=300, help="PySR timeout_in_seconds")
    parser.add_argument("--partition", type=str, default="default_partition", help="SLURM partition")
    parser.add_argument("--time-limit", type=str, default="04:00:00", help="SLURM time limit")
    parser.add_argument("--mem-per-cpu", type=str, default="8G", help="SLURM memory per CPU")
    parser.add_argument("--job-timeout", type=float, default=3000.0, help="Wait timeout for SLURM jobs")
    parser.add_argument("--output-dir", type=str, default=None, help="OpenEvolve output directory")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "openevolve_pysr" / "config.yaml"), help="OpenEvolve config file")
    parser.add_argument("--api-base", type=str, default=None, help="Override API base")
    parser.add_argument("--primary-model", type=str, default=None, help="Override primary model")
    parser.add_argument("--secondary-model", type=str, default=None, help="Override secondary model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/openevolve_pysr_{timestamp}"

    output_path = REPO_ROOT / output_dir
    initial_program = REPO_ROOT / "openevolve_pysr" / "initial_program.py"
    evaluator = REPO_ROOT / "openevolve_pysr" / "evaluator.py"
    runner = REPO_ROOT / "openevolve" / "openevolve-run.py"

    env = os.environ.copy()
    env.update(
        {
            "OE_PYSR_SPLIT": args.split,
            "OE_PYSR_STAGE2_DATASETS": str(args.stage2_datasets),
            "OE_PYSR_FITNESS_METRIC": args.fitness_metric,
            "OE_PYSR_N_RUNS": str(args.n_runs),
            "OE_PYSR_SEED": str(args.seed),
            "OE_PYSR_DATA_SEED": str(args.data_seed),
            "OE_PYSR_TARGET_NOISE": str(args.target_noise),
            "OE_PYSR_MAX_SAMPLES": str(args.max_samples),
            "OE_PYSR_MAX_EVALS": str(args.max_evals),
            "OE_PYSR_TIMEOUT_IN_SECONDS": str(args.timeout),
            "OE_PYSR_PARTITION": args.partition,
            "OE_PYSR_TIME_LIMIT": args.time_limit,
            "OE_PYSR_MEM_PER_CPU": args.mem_per_cpu,
            "OE_PYSR_JOB_TIMEOUT": str(args.job_timeout),
            "OE_PYSR_RESULTS_DIR": str(output_path / "pysr_eval"),
            "OE_PYSR_USE_CACHE": "true",
        }
    )

    cmd = [
        sys.executable,
        str(runner),
        str(initial_program),
        str(evaluator),
        "--config",
        args.config,
        "--output",
        str(output_path),
        "--iterations",
        str(args.iterations),
    ]
    if args.api_base:
        cmd.extend(["--api-base", args.api_base])
    if args.primary_model:
        cmd.extend(["--primary-model", args.primary_model])
    if args.secondary_model:
        cmd.extend(["--secondary-model", args.secondary_model])

    print("Running:", " ".join(cmd))
    print(f"Output dir: {output_path}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
