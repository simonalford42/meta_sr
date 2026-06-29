#!/usr/bin/env python3
"""Evaluate an evolved / HPO / baseline PySR method on the *unsolved* EmpiricalBench problems.

EmpiricalBench (Cranmer 2023, arXiv:2305.01582) has 9 historical empirical
equations.  Two of them -- **Planck's law** and the **Rydberg formula** -- were
recovered 0/5 by *every* method in the paper (PySR, Operon, DSR, EQL, QLattice,
SR-Transformer).  This script runs a chosen method on exactly those two problems
and reports symbolic-recovery rate (does the Pareto front contain the true
equation -- the same criterion EmpiricalBench uses), via the pipeline's
``fitness_metric="gt"`` symbolic match.

Method selection mirrors ``srbench_full_eval.py`` (loads via evaluate_new_pysr):

    # baseline PySR
    python empiricalbench_eval.py

    # an evolved bundle
    python empiricalbench_eval.py --evolve-results runs/<run_id>

    # an HPO-tuned config
    python empiricalbench_eval.py --hpo-results runs/<run_id>

The two datasets are materialized on demand by scripts/gen_empirical_bench.py
(faithful reproduction of the paper's generation: log-uniform / integer sampling,
relative noise, and the log-scaled target).  They are auto-generated if missing.
"""

import argparse
import sys
from pathlib import Path

# The two EmpiricalBench problems unsolved by all methods in Cranmer (2023).
EMPIRICAL_DATASETS = ["empirical_planck", "empirical_rydberg"]


def ensure_datasets():
    """Materialize the EmpiricalBench datasets if they are not already present."""
    from utils import PMLB_PATH
    missing = [d for d in EMPIRICAL_DATASETS
               if not (PMLB_PATH / d / f"{d}.tsv.gz").exists()]
    if missing:
        print(f"Generating missing EmpiricalBench datasets: {missing}")
        import scripts.gen_empirical_bench as gen
        # Call generators directly (avoid argparse) and write them.
        writers = {"empirical_planck": gen.gen_planck,
                   "empirical_rydberg": gen.gen_rydberg}
        for name in missing:
            X, y, desc, var_names = writers[name]()
            gen.write_dataset(name, X, y, desc, var_names)


def main():
    parser = argparse.ArgumentParser(
        description="Run a method on the unsolved EmpiricalBench problems (Planck, Rydberg).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evolve-results", type=str, default=None,
                      help="Path / run-id of an evolve run (run_data.json).")
    mode.add_argument("--hpo-results", type=str, default=None,
                      help="Path / run-id of an HPO run (best_params.json).")

    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated override of EmpiricalBench dataset names.")
    parser.add_argument("--max-evals", type=int, default=1_000_000)
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Subsample cap; the datasets have 100/50 points so this is a no-op by default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Trials per problem (the paper used 5).")

    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time-limit", type=str, default="02:00:00")
    parser.add_argument("--mem-per-cpu", type=str, default="8G")
    parser.add_argument("--job-timeout", type=float, default=None)
    parser.add_argument("--pysr-wall-limit", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    print("Executing command: " + " ".join(sys.argv))

    ensure_datasets()

    datasets = ([d.strip() for d in args.datasets.split(",") if d.strip()]
                if args.datasets else list(EMPIRICAL_DATASETS))

    # Reuse the loader/dispatch from srbench_full_eval (baseline | hpo | evolve);
    # it loads evolved/HPO bundles via evaluate_new_pysr.
    from srbench_full_eval import build_config
    config, mode_name, method_meta = build_config(args)
    print(f"Mode: {mode_name}  |  datasets: {datasets}  |  n_runs: {args.n_runs}")

    from parallel_eval_pysr import PySRSlurmEvaluator
    from utils import resolve_run_dir, copy_slurm_log

    output_dir = resolve_run_dir(args.results_dir, label="empiricalbench_eval")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {output_dir}")

    evaluator = PySRSlurmEvaluator(
        results_dir=output_dir,
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=args.max_samples,
        data_seed=args.seed,
        use_cache=not args.no_cache,
        job_timeout=args.job_timeout,
        pysr_wall_limit=args.pysr_wall_limit,
        max_retries=args.max_retries,
    )

    run = None
    if not args.no_wandb:
        from wandb_utils import init_wandb
        run = init_wandb(
            config={
                "mode": mode_name,
                "max_evals": args.max_evals,
                "seed": args.seed,
                "n_runs": args.n_runs,
                "datasets": datasets,
                "benchmark": "empirical_bench_unsolved",
                "method_meta": method_meta,
            },
            script_name="empiricalbench_eval.py",
            output_dir=output_dir,
            extra_tags=["empirical_bench", mode_name],
        )

    # EmpiricalBench data already carries the paper's built-in noise, so we
    # evaluate at noise 0.0 (no extra target noise) and score symbolic recovery.
    target_noise_map = {ds: 0.0 for ds in datasets}
    results = evaluator.evaluate_configs(
        [config], datasets, seed=args.seed, n_runs=args.n_runs,
        target_noise_map=target_noise_map, fitness_metric="gt",
    )

    avg_gt, gt_vector, details = results[0]

    # --- report ---
    print(f"\n{'=' * 64}")
    print(f"  EmpiricalBench (unsolved) — {mode_name}")
    print(f"{'=' * 64}")
    detail_by_ds = {d.get("dataset"): d for d in details}
    total_solved = total_runs = 0
    for ds in datasets:
        d = detail_by_ds.get(ds, {})
        scores = d.get("run_gt_scores", []) or []
        solved = int(sum(1 for s in scores if s and s >= 1.0))
        total_solved += solved
        total_runs += len(scores)
        bits = " ".join("1" if (s and s >= 1.0) else "0" for s in scores)
        print(f"  {ds:20s}  recovered {solved}/{len(scores)}  [{bits}]")
        matched = d.get("gt_matched_equation")
        if matched:
            print(f"  {'':20s}  matched expr: {matched}")
    rate = total_solved / total_runs if total_runs else float("nan")
    print(f"  {'-' * 60}")
    print(f"  TOTAL recovery: {total_solved}/{total_runs}  (rate={rate:.3f})")
    print(f"  (Paper's Table 3: every method scored 0/5 on both.)")

    if run is not None:
        from wandb_utils import log_wandb_summary, finish_wandb
        per_ds_summary = {
            f"{ds}_recovery_rate": (
                (sum(1 for s in (detail_by_ds.get(ds, {}).get("run_gt_scores") or [])
                     if s and s >= 1.0) / len(detail_by_ds[ds]["run_gt_scores"]))
                if detail_by_ds.get(ds, {}).get("run_gt_scores") else 0.0
            )
            for ds in datasets
        }
        log_wandb_summary(run, evaluator=evaluator, extra_summary={
            "overall_recovery_rate": rate,
            "n_solved": total_solved,
            "n_runs_total": total_runs,
            **per_ds_summary,
        })
        finish_wandb(run)

    copy_slurm_log(output_dir)
    print(f"\nDone. Run directory: {output_dir}")


if __name__ == "__main__":
    main()
