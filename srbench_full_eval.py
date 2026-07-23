#!/usr/bin/env python3
"""Submit a full SRBench evaluation (all tasks x seeds x noise levels).

Drives the native evaluator for the loaded method (PySR or FullSR) over every
task in a split file. PySR modes retain caching, SLURM array chunking, retries,
and bad-node handling.

    # baseline PySR
    python srbench_full_eval.py

    # an HPO-tuned config
    python srbench_full_eval.py --hpo-results runs/394789/

    # an evolved bundle
    python srbench_full_eval.py --evolve-results runs/<run_id> --max-evals 1000000

``--evolve-results`` auto-detects both evolve_pysr.py OperatorBundles and
evolve_fullsr.py SkeletonBundles from their run_data.json schema.

Default scale: 133 tasks x 10 seeds x 4 noise levels (0.0, 0.001, 0.01, 0.1).
Results land in ``runs/<run_id>/`` (manifest.json + srbench_full_results.json)
and are logged to the "meta-sr" wandb project as a Table plus solve-rate /
solve-time metrics for srbench(all) / feynman / strogatz per noise level.

Inspect a finished (or in-progress) run with::

    python inspect_srbench_results.py --run-id <run_id>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import srbench_results_io as srio


DEFAULT_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]


def build_config(args):
    """Return (PySRConfig, mode, method_meta) for the selected method mode.

    All three modes go through the single loader (bundle_loader) +
    converter (OperatorBundle.to_pysr_config): evolve/hpo load a bundle from
    the run dir, baseline uses the default (no custom operators) bundle.
    """
    from srbench_eval_source import load_pysr_evaluation_config

    return load_pysr_evaluation_config(args)


def main():
    parser = argparse.ArgumentParser(
        description="Submit a full SRBench evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evolve-results", type=str, default=None,
                      help="Path / run-id of an evolve run (run_data.json).")
    mode.add_argument("--hpo-results", type=str, default=None,
                      help="Path / run-id of an HPO run (best_params.json).")
    parser.add_argument("--select-by", choices=["val", "train"], default="val",
                        help="For evolve runs, select the bundle by validation "
                             "score (default) or training score. Val falls back "
                             "to train for runs without persisted val data.")

    parser.add_argument("--split-file", type=str, default="splits/srbench_all.txt")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset override (for smoke tests).")
    parser.add_argument("--max-evals", type=int, default=1_000_000)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Seeds per task/noise (actual seeds = seed .. seed+n_runs-1).")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS)

    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time-limit", type=str, default="02:00:00")
    parser.add_argument("--mem-per-cpu", type=str, default="8G")
    parser.add_argument("--job-timeout", type=float, default=None)
    parser.add_argument("--pysr-wall-limit", type=int, default=600)
    parser.add_argument("--fullsr-wall-limit", type=int, default=600,
                        help="Hard wall-clock limit for each FullSR fit.")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Retry rounds for transient/missing tasks per batch.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Run directory (default: runs/<SLURM_JOB_ID> or local_*).")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    print("Executing command: " + " ".join(sys.argv))

    from utils import resolve_run_dir, load_dataset_names_from_split, copy_slurm_log
    from srbench_eval_source import load_evaluation_source

    output_dir = resolve_run_dir(args.results_dir, label="srbench_full_eval")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Run directory: {output_dir}")

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = load_dataset_names_from_split(args.split_file)
    print(f"Datasets: {len(datasets)}  |  seeds: {args.n_runs}  |  "
          f"noise levels: {args.noise_levels}  |  total runs: "
          f"{len(datasets) * args.n_runs * len(args.noise_levels)}")

    source = load_evaluation_source(args)
    config, mode_name, method_meta = source.config, source.mode, source.method_meta
    print(f"Mode: {mode_name}  |  backend: {source.backend}")

    if source.backend == "fullsr":
        from parallel_eval_fullsr import FullSRSlurmEvaluator

        evaluator = FullSRSlurmEvaluator(
            results_dir=output_dir,
            partition=args.partition,
            time_limit=args.time_limit,
            mem_per_cpu=args.mem_per_cpu,
            dataset_max_samples=args.max_samples,
            data_seed=args.seed,
            use_cache=not args.no_cache,
            job_timeout=args.job_timeout,
            wall_limit=args.fullsr_wall_limit,
            max_retries=args.max_retries,
            eval_noise_levels=args.noise_levels,
        )
    else:
        from parallel_eval_pysr import PySRSlurmEvaluator

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
                "backend": source.backend,
                "max_evals": args.max_evals,
                "max_samples": args.max_samples,
                "seed": args.seed,
                "n_runs": args.n_runs,
                "noise_levels": args.noise_levels,
                "n_datasets": len(datasets),
                "split_file": args.split_file,
                "method_meta": method_meta,
            },
            script_name="srbench_full_eval.py",
            output_dir=output_dir,
            extra_tags=["srbench_full_eval", mode_name],
        )

    manifest = {
        "mode": mode_name,
        "backend": source.backend,
        "method_meta": method_meta,
        "max_evals": args.max_evals,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "n_runs": args.n_runs,
        "seeds": [args.seed + i for i in range(args.n_runs)],
        "noise_levels": args.noise_levels,
        "split_file": args.split_file,
        "n_datasets": len(datasets),
        "datasets": datasets,
        "unsolvable_tasks": list(srio.UNSOLVABLE_TASKS),
        "batches": [],
    }
    with open(Path(output_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if source.backend == "fullsr":
        # FullSR natively expands all requested noise levels into one array.
        before = set(evaluator.slurm_dir.glob("eval_*"))
        evaluator.evaluate_configs(
            configs=[config],
            dataset_names=datasets,
            seed=args.seed,
            n_runs=args.n_runs,
            fitness_metric="gt",
            fullsr_wall_limit=args.fullsr_wall_limit,
            split_label=args.split_file,
        )
        created = sorted(set(evaluator.slurm_dir.glob("eval_*")) - before)
        if len(created) != 1:
            raise RuntimeError(
                f"Expected one new FullSR batch directory, found {len(created)}"
            )
        batch_dir = created[0]
        manifest["batches"].append({
            "noise": "all",
            "batch_dir": f"slurm_fullsr/{batch_dir.name}",
            "n_tasks": len(datasets) * args.n_runs * len(args.noise_levels),
        })
        with open(Path(output_dir) / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    else:
        # PySR submits one batch per noise level, then waits for all together.
        handles = []
        for noise in args.noise_levels:
            noise_map = {ds: noise for ds in datasets}
            h = evaluator.submit_configs(
                configs=[config],
                dataset_names=datasets,
                seed=args.seed,
                n_runs=args.n_runs,
                target_noise_map=noise_map,
                fitness_metric="gt",
            )
            handles.append((noise, h))
            print(f"  submitted noise={noise}: {h.n_tasks} tasks "
                  f"({h.n_cached} cached) -> {h.batch_dir}")
        total_tasks = sum(h.n_tasks for _, h in handles)
        total_cached = sum(h.n_cached for _, h in handles)
        uncached = total_tasks - total_cached
        print(
            f"\n{total_cached}/{total_tasks} runs already cached; "
            f"{uncached} runs require execution (max_retries={args.max_retries})."
        )
        manifest["batches"] = [
            {
                "noise": noise,
                "batch_dir": f"slurm_pysr/{Path(h.batch_dir).name}",
                "n_tasks": h.n_tasks,
            }
            for noise, h in handles
        ]
        with open(Path(output_dir) / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        evaluator.collect_batches([h for _, h in handles])

    # Build + persist the big keyed JSON from the on-disk batch artifacts.
    keyed = srio.build_keyed_results(output_dir, manifest)
    srio.save_keyed_results(output_dir, keyed, meta={
        "mode": mode_name, "n_datasets": len(datasets),
        "n_runs": args.n_runs, "noise_levels": args.noise_levels,
    })
    n_present = sum(1 for e in keyed.values() if e["present"] and e["error"] is None)
    print(f"\nResults: {n_present}/{len(keyed)} runs present.")

    metrics = srio.aggregate_metrics(keyed, args.noise_levels)
    print(srio.format_metrics_console(metrics, args.noise_levels))

    if run is not None:
        from wandb_utils import log_wandb_summary, finish_wandb
        srio.log_wandb_table_and_metrics(run, keyed, args.noise_levels)
        overall = metrics["all"]["all"]
        log_wandb_summary(run, evaluator=evaluator, extra_summary={
            "n_runs_present": n_present,
            "n_runs_expected": len(keyed),
            "overall_solve_rate_per_run": overall["solve_rate_per_run"],
            "overall_solve_rate_per_task_any": overall["solve_rate_per_task_any"],
        })
        finish_wandb(run)

    copy_slurm_log(output_dir)
    print(f"\nDone. Inspect with: python inspect_srbench_results.py --run-id "
          f"{Path(output_dir).name}")


if __name__ == "__main__":
    main()
