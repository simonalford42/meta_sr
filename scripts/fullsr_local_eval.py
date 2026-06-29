#!/usr/bin/env python3
"""Local (no-SLURM) FullSR policy comparison on a split.

Runs `_evaluate_fullsr_task` for one or more built-in policies (basic / pysr /
sr) across every dataset in a split, parallelized over a persistent process
pool sized to the session's reserved cores. Reports per-policy solve counts
(gt match) and mean R²/gt so we can sanity-check the baselines before evolving.

Usage:
    python scripts/fullsr_local_eval.py --split splits/train.txt \
        --policies basic,pysr --n-runs 1 --max-evals 1000000 \
        --max-samples 1000 --workers 7
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from parallel_eval_fullsr import (
    FullSRTaskSpec,
    _evaluate_fullsr_task,
    get_default_engine_kwargs,
)
from slurm_eval import init_worker
from utils import load_dataset_names_from_split


def _run_one(spec_dict):
    spec = FullSRTaskSpec.from_json_dict(spec_dict)
    res = _evaluate_fullsr_task(spec)
    return res.to_json_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="splits/train.txt")
    ap.add_argument("--policies", default="basic,pysr")
    ap.add_argument("--n-runs", type=int, default=1)
    ap.add_argument("--max-evals", type=int, default=1_000_000)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--wall-limit", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--fitness-metric", default="gt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    datasets = load_dataset_names_from_split(args.split)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    engine_kwargs = get_default_engine_kwargs()
    engine_kwargs["max_evals"] = args.max_evals

    specs = []
    for cid, policy in enumerate(policies):
        for ds in datasets:
            for run_idx in range(args.n_runs):
                specs.append(
                    FullSRTaskSpec(
                        config_id=cid,
                        dataset_name=ds,
                        policy_name=policy,
                        engine_kwargs=engine_kwargs,
                        seed=args.seed,
                        data_seed=args.data_seed,
                        max_samples=args.max_samples,
                        run_index=run_idx,
                        target_noise=0.0,
                        fitness_metric=args.fitness_metric,
                        wall_limit=args.wall_limit,
                    ).to_json_dict()
                )

    print(
        f"Split={args.split} datasets={len(datasets)} policies={policies} "
        f"n_runs={args.n_runs} -> {len(specs)} tasks, workers={args.workers}",
        flush=True,
    )
    t0 = time.time()
    results = []
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=({"JULIA_NUM_THREADS": "1"},),
    ) as ex:
        futs = {ex.submit(_run_one, s): s for s in specs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            tag = "OK" if not r.get("error") else r["error"][:60]
            print(
                f"[{done}/{len(specs)}] cfg{r['config_id']} {r['dataset_name']} "
                f"run{r['run_index']} r2={r['r2_score']:.3f} gt={r['gt_match_score']} "
                f"{tag} ({time.time()-t0:.0f}s)",
                flush=True,
            )

    # Aggregate per policy.
    print("\n" + "=" * 70)
    print(f"SUMMARY ({time.time()-t0:.0f}s total)")
    print("=" * 70)
    summary = {}
    for cid, policy in enumerate(policies):
        rs = [r for r in results if r["config_id"] == cid]
        # Per-dataset: solved if ANY run gt-matched; also mean gt/r2.
        per_ds = {}
        for ds in datasets:
            dr = [r for r in rs if r["dataset_name"] == ds]
            gts = [r.get("gt_match_score") or 0.0 for r in dr]
            r2s = [r.get("r2_score") if r.get("r2_score") is not None else -1.0 for r in dr]
            per_ds[ds] = {
                "solved_any": bool(gts) and max(gts) >= 1.0,
                "mean_gt": float(np.mean(gts)) if gts else 0.0,
                "mean_r2": float(np.mean(r2s)) if r2s else -1.0,
                "n_errors": sum(1 for r in dr if r.get("error")),
            }
        n_solved = sum(1 for d in per_ds.values() if d["solved_any"])
        mean_gt = float(np.mean([d["mean_gt"] for d in per_ds.values()]))
        mean_r2 = float(np.mean([d["mean_r2"] for d in per_ds.values()]))
        n_err = sum(d["n_errors"] for d in per_ds.values())
        summary[policy] = {
            "n_solved": n_solved,
            "n_datasets": len(datasets),
            "mean_gt": mean_gt,
            "mean_r2": mean_r2,
            "n_errors": n_err,
            "per_dataset": per_ds,
        }
        print(
            f"\n{policy}: solved {n_solved}/{len(datasets)} "
            f"(mean_gt={mean_gt:.3f}, mean_r2={mean_r2:.3f}, errors={n_err})"
        )
        for ds, d in per_ds.items():
            mark = "SOLVED" if d["solved_any"] else "      "
            print(f"   {mark} {ds:24s} gt={d['mean_gt']:.2f} r2={d['mean_r2']:.3f}"
                  + (f"  ERR={d['n_errors']}" if d["n_errors"] else ""))

    out = args.out or f"scratch_fullsr_eval_{int(t0)}.json"
    Path(out).write_text(json.dumps({"summary": summary, "results": results,
                                     "args": vars(args)}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
