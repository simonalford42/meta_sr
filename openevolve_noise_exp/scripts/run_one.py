"""Launch a single openevolve run for the noise experiment.
u

Usage:
  python scripts/run_one.py --noise 0.3 --n-trials 3 --budget 60 --seed 0 \
      --out outputs/n3_sigma0.3_seed0

The total compute "budget" is in algorithm-trial units. Openevolve iterations
= budget // n_trials, so higher n_trials → fewer iterations.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
OPENEVOLVE_CLI = "/home/sca63/meta_sr/openevolve/openevolve-run.py"


def build_config(base_cfg_path, out_path, max_iterations, seed):
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["max_iterations"] = int(max_iterations)
    cfg["random_seed"] = int(seed)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, required=True, help="per-trial Gaussian noise stddev")
    ap.add_argument("--n-trials", type=int, required=True, help="trials averaged per evaluation")
    ap.add_argument("--budget", type=int, default=None, help="total trial-units budget per run (iterations = budget//n_trials)")
    ap.add_argument("--iters", type=int, default=None, help="fixed iteration count, overrides --budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True, help="output directory")
    ap.add_argument("--base-config", type=str, default=str(REPO / "configs" / "base.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.iters is None and args.budget is None:
        ap.error("must specify --iters or --budget")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.iters is not None:
        iterations = int(args.iters)
    else:
        iterations = max(1, args.budget // args.n_trials)

    run_config = out_dir / "run_config.yaml"
    build_config(args.base_config, run_config, iterations, args.seed)

    # Record the run params for post-hoc analysis
    params = {
        "noise_std": args.noise,
        "num_trials": args.n_trials,
        "budget_trials": args.budget if args.budget is not None else iterations * args.n_trials,
        "iterations": iterations,
        "seed": args.seed,
        "timestamp": time.time(),
    }
    (out_dir / "run_params.json").write_text(json.dumps(params, indent=2))

    eval_log = out_dir / "eval_log.jsonl"

    env = os.environ.copy()
    env["NOISE_STD"] = str(args.noise)
    env["NUM_TRIALS"] = str(args.n_trials)
    env["EVAL_LOG_PATH"] = str(eval_log)
    env["NOISE_SEED"] = str(args.seed)

    cmd = [
        sys.executable,
        OPENEVOLVE_CLI,
        str(REPO / "initial_program.py"),
        str(REPO / "evaluator.py"),
        "--config", str(run_config),
        "--output", str(out_dir),
        "--iterations", str(iterations),
    ]

    print(f"[run_one] iterations={iterations} noise={args.noise} N={args.n_trials} seed={args.seed}")
    print(f"[run_one] out_dir={out_dir}")
    print(f"[run_one] cmd={' '.join(cmd)}")

    if args.dry_run:
        return 0

    log_file = out_dir / "run_stdout.log"
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    status = {
        "returncode": proc.returncode,
        "finished_at": time.time(),
        "elapsed_s": time.time() - params["timestamp"],
    }
    (out_dir / "run_status.json").write_text(json.dumps(status, indent=2))
    print(f"[run_one] done rc={proc.returncode} elapsed={status['elapsed_s']:.1f}s log={log_file}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
