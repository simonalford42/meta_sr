#!/usr/bin/env python3
"""Generate baseline-vs-evolved EmpiricalBench jobs for local or SLURM use."""
import argparse
import json
from pathlib import Path

DATASETS = ["empirical_planck", "empirical_rydberg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jobs", default="runs_local/main/jobs.json")
    ap.add_argument("--out-dir", default="runs_local/main")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--max-evals", type=float, default=1e6)
    ap.add_argument("--timeout-seconds", type=float, default=3600,
                    help="PySR soft wall-clock limit; max-evals remains active")
    ap.add_argument("--n-milestones", type=int, default=12)
    ap.add_argument("--wall-limit", type=float, default=3900,
                    help="hard worker guard; must exceed --timeout-seconds")
    ap.add_argument("--evolve-results", default="runs/709715")
    args = ap.parse_args()

    if args.wall_limit <= args.timeout_seconds:
        ap.error("--wall-limit must exceed --timeout-seconds so PySR can save its frontier")

    jobs = []
    out_dir = Path(args.out_dir)
    for ds in DATASETS:
        short = ds.replace("empirical_", "")
        for ri in range(args.n_seeds):
            # baseline
            jobs.append({
                "method": "baseline", "dataset": ds, "run_index": ri,
                "max_evals": int(args.max_evals), "n_milestones": args.n_milestones,
                "timeout_seconds": int(args.timeout_seconds),
                "wall_limit": int(args.wall_limit),
                "out": str(out_dir / f"{short}_baseline_r{ri}.json"),
            })
            # evolved
            jobs.append({
                "method": "evolve", "evolve_results": args.evolve_results,
                "dataset": ds, "run_index": ri,
                "max_evals": int(args.max_evals), "n_milestones": args.n_milestones,
                "timeout_seconds": int(args.timeout_seconds),
                "wall_limit": int(args.wall_limit),
                "out": str(out_dir / f"{short}_evolve_r{ri}.json"),
            })

    Path(args.out_jobs).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_jobs).write_text(json.dumps(jobs, indent=2))
    print(f"Wrote {len(jobs)} jobs to {args.out_jobs}")
    for j in jobs:
        print(f"  {j['method']:9s} {j['dataset']:18s} r{j['run_index']}")


if __name__ == "__main__":
    main()
