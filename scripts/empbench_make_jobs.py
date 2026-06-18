#!/usr/bin/env python3
"""Generate the JSON job list for the EmpiricalBench local study (Phase A: the
baseline-vs-evolved comparison at 1e7 evals on both problems)."""
import argparse
import json
from pathlib import Path

DATASETS = ["empirical_planck", "empirical_rydberg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jobs", default="runs_local/main/jobs.json")
    ap.add_argument("--out-dir", default="runs_local/main")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--max-evals", type=float, default=1e7)
    ap.add_argument("--n-milestones", type=int, default=12)
    ap.add_argument("--wall-limit", type=float, default=21600)
    ap.add_argument("--evolve-results", default="runs/666285")
    args = ap.parse_args()

    jobs = []
    out_dir = Path(args.out_dir)
    for ds in DATASETS:
        short = ds.replace("empirical_", "")
        for ri in range(args.n_seeds):
            # baseline
            jobs.append({
                "method": "baseline", "dataset": ds, "run_index": ri,
                "max_evals": int(args.max_evals), "n_milestones": args.n_milestones,
                "wall_limit": int(args.wall_limit),
                "out": str(out_dir / f"{short}_baseline_r{ri}.json"),
            })
            # evolved
            jobs.append({
                "method": "evolve", "evolve_results": args.evolve_results,
                "dataset": ds, "run_index": ri,
                "max_evals": int(args.max_evals), "n_milestones": args.n_milestones,
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
