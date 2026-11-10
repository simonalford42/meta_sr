#!/usr/bin/env python3
"""Fixed PySR autoresearch evaluation harness.

Every experiment is scored by a single evaluation: 10 runs per task on the
official training split (splits/barely_unsolvable.txt), with fixed seed 42 so
candidates are paired. Validation data is never evaluated by this harness.
The selected SymbolicRegression.jl commit is run in an isolated sandbox.

Run-scoped mode (used for budgeted runs): when AUTORESEARCH_RUN_DIR is set, the
SymbolicRegression.jl worktree, results.tsv, sandboxes, and eval outputs all
live under that run directory, every evaluation is appended to
<run_dir>/experiment_log.jsonl (used for wall-clock / cost accounting), and the
harness refuses to evaluate more than max_experiments distinct candidate
commits (the run's base commit, i.e. the baseline, is not counted).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
TRAIN = "splits/barely_unsolvable.txt"
SEED = 42
N_RUNS = 10

RUN_DIR = Path(os.environ["AUTORESEARCH_RUN_DIR"]).resolve() \
    if os.environ.get("AUTORESEARCH_RUN_DIR") else None
if RUN_DIR is not None:
    SUBMODULE = RUN_DIR / "SymbolicRegression.jl"
    RESULTS_TSV = RUN_DIR / "results.tsv"
    SANDBOXES = ROOT / "outputs" / "autoresearch_pysr_sandboxes" / RUN_DIR.name
    EVAL_ROOT = RUN_DIR / "eval_results"
else:
    SUBMODULE = ROOT / "SymbolicRegression.jl"
    RESULTS_TSV = ROOT / "autoresearch_sr" / "results.tsv"
    SANDBOXES = ROOT / "outputs" / "autoresearch_pysr_sandboxes"
    EVAL_ROOT = ROOT / "autoresearch_sr" / "eval_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a committed SymbolicRegression.jl autoresearch candidate"
    )
    parser.add_argument("--target", default="latest",
                        help="Commit/branch, latest, best, or expN")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _log_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


def _append(log_path: Path, record: dict) -> None:
    with log_path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT))
    from autoresearch_pysr import resolve_commit

    commit = resolve_commit(args.target, SUBMODULE, RESULTS_TSV)

    run_cfg, log_path, lock_handle, exp_index = {}, None, None, None
    if RUN_DIR is not None:
        run_cfg = json.loads((RUN_DIR / "run.json").read_text())
        log_path = RUN_DIR / "experiment_log.jsonl"
        lock_handle = (RUN_DIR / ".eval.lock").open("w")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("ERROR: another evaluation is already running for this run. "
                     "Evaluations must run one at a time.")
        records = _log_records(log_path)
        seen = []
        for rec in records:
            if rec["event"] == "start" and rec["commit"] != run_cfg["base_commit"] \
                    and rec["commit"] not in seen:
                seen.append(rec["commit"])
        max_exp = int(run_cfg["max_experiments"])
        max_cost = run_cfg.get("max_cost_usd")
        spent = None
        if max_cost is not None:
            sys.path.insert(0, str(ROOT / "autoresearch_sr"))
            from cost_report import run_cost
            spent = run_cost(RUN_DIR)
            print(f"api cost:      ${spent:.2f} of ${float(max_cost):.2f} budget", flush=True)
        if (commit != run_cfg["base_commit"] and commit not in seen
                and spent is not None and spent >= float(max_cost)):
            print(f"BUDGET EXHAUSTED: API cost ${spent:.2f} has reached the "
                  f"${float(max_cost):.2f} budget. No further evaluations are allowed. "
                  "The run is over; stop now.")
            sys.exit(3)
        if commit == run_cfg["base_commit"]:
            exp_index = 0
        elif commit in seen:
            exp_index = seen.index(commit) + 1
        elif len(seen) >= max_exp:
            print(f"BUDGET EXHAUSTED: {len(seen)}/{max_exp} experiments already evaluated. "
                  "No further evaluations are allowed. The run is over; stop now.")
            sys.exit(3)
        else:
            exp_index = len(seen) + 1
        print(f"experiment:    {exp_index} of {max_exp} (baseline = 0)", flush=True)

    output_dir = Path(args.output_dir) if args.output_dir else (
        EVAL_ROOT / commit[:12] / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    command = [
        sys.executable,
        str(ROOT / "evaluate_new_pysr.py"),
        "--autoresearch", commit,
        "--autoresearch-submodule", str(SUBMODULE),
        "--autoresearch-results", str(RESULTS_TSV),
        "--autoresearch-sandboxes", str(SANDBOXES),
        "--splits", TRAIN,
        "--n-runs", str(N_RUNS),
        "--seed", str(SEED),
        "--max-evals", "1000000",
        "--max-samples", "1000",
        "--timeout", "500",
        "--pysr-wall-limit", "600",
        "--max-concurrent-jobs", "300",
        "--time-limit", "01:00:00",
        "--constraint", "il|sr",
        "--fitness-metric", "gt",
        "--random-target-noise",
        "--output-dir", str(output_dir),
    ]
    print(f"commit: {commit}", flush=True)
    print(f"seed: {SEED}  n_runs: {N_RUNS}", flush=True)
    print("command: " + " ".join(command), flush=True)
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "disabled")
    # Every candidate commit is new SymbolicRegression.jl source, so it is compiled
    # once per experiment by the driver's Julia warmup, for the driver's CPU (Ice
    # Lake). Tasks on other CPU families reject that build, recompile it and
    # overwrite the single cache file in turn while every other task waits on the
    # lock (median task load 30 min, 11/9). So workers are pinned to CPU families
    # that can load an Ice Lake build: Ice Lake and Sapphire Rapids (see --constraint).
    # A multi-target JULIA_CPU_TARGET build was tried and abandoned (11/6, 11/9).
    # This requires the driver itself to run on an Ice Lake / Sapphire Rapids node.
    if "gfni" not in Path("/proc/cpuinfo").read_text():
        sys.exit("ERROR: run evaluate.py on an Ice Lake or Sapphire Rapids node "
                 "(e.g. ellis-compute-02); workers are pinned to those CPU families.")
    # Compiled caches go to a run-private depot (first entry); packages and
    # registries still resolve from ~/.julia via the trailing ':' (default depots).
    # The shared ~/.julia compiled slots are rewritten by every other PySR job.
    depot = ROOT / "outputs" / "julia_depots" / (RUN_DIR.name if RUN_DIR else "default")
    depot.mkdir(parents=True, exist_ok=True)
    env.setdefault("JULIA_DEPOT_PATH", f"{depot}:")

    start = time.time()
    if log_path is not None:
        _append(log_path, {"event": "start", "exp": exp_index, "commit": commit,
                           "time": start, "output_dir": str(output_dir)})
    proc = subprocess.run(command, cwd=ROOT, env=env)
    end = time.time()

    score, datasets, datasets_fail = None, 0, 0
    summary_path = output_dir / "eval_summary.json"
    if proc.returncode == 0 and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        name = Path(TRAIN).stem
        values = summary[name].get("per_run_gt_avgs") or []
        details = summary[name].get("result_details") or []
        score = float(np.mean(values)) if values else 0.0
        datasets = len(details)
        datasets_fail = sum(
            detail.get("n_successful_runs", 0) < N_RUNS for detail in details
        )
    if log_path is not None:
        _append(log_path, {"event": "end", "exp": exp_index, "commit": commit,
                           "time": end, "returncode": proc.returncode, "score": score,
                           "datasets": datasets, "datasets_fail": datasets_fail})
    if score is None:
        sys.exit(f"ERROR: evaluation failed (returncode {proc.returncode})")

    print("\n--- autoresearch result ---")
    print(f"commit:        {commit}")
    print(f"score:         {score:.6f}")
    print(f"datasets:      {datasets}")
    print(f"datasets_ok:   {datasets - datasets_fail}")
    print(f"datasets_fail: {datasets_fail}")
    print("metric:        gt")
    print(f"n_runs:        {N_RUNS}")
    print(f"eval_seconds:  {end - start:.0f}")
    print(f"summary:       {summary_path}")


if __name__ == "__main__":
    main()
