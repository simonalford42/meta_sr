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

Default scale: 133 tasks x 10 seeds x 4 noise levels (0.0, 0.001, 0.01, 0.1)
for ground truth, and 122 datasets x 10 seeds for --black-box. Both come from
--n-trials-per-dataset.
Results land in ``runs/<run_id>/`` (manifest.json + srbench_full_results.json)
and are logged to the "meta-sr" wandb project as a Table plus solve-rate /
solve-time metrics for srbench(all) / feynman / strogatz per noise level.

Inspect a finished (or in-progress) run with::

    python inspect_srbench_results.py --run-id <run_id>
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import srbench_results_io as srio


DEFAULT_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]


def load_black_box_datasets():
    """Return the 122 regression datasets used by SRBench black-box results."""
    csv_path = Path(__file__).parent / "srbench/docs/csv/blackbox_results_datasets.csv"
    with open(csv_path, newline="") as f:
        return sorted({row["dataset"] for row in csv.DictReader(f)})


def _test_pareto(rows):
    """Remove points dominated in (lower complexity, higher test R²)."""
    best = -float("inf")
    frontier = []
    for row in sorted(rows, key=lambda r: (r["complexity"], -r["test_r2"])):
        if row["test_r2"] > best:
            frontier.append(row)
            best = row["test_r2"]
    return frontier


def _envelope(rows, grid):
    """Best-so-far test R² at each complexity in `grid` for one trial frontier."""
    import numpy as np

    values = np.full(len(grid), np.nan)
    best = np.nan
    j = 0
    for i, complexity in enumerate(grid):
        while j < len(rows) and rows[j]["complexity"] <= complexity:
            best = rows[j]["test_r2"]
            j += 1
        values[i] = best
    return values


def save_black_box_results(output_dir, batch_dir, max_train_samples=10_000,
                           n_trials=1):
    """Persist per-dataset per-trial test frontiers and plot their envelope.

    Each dataset is run `n_trials` times (distinct seeds -> distinct 75/25
    splits and searches); results are stored per trial so downstream analysis
    can quantify seed variance. The plot shows each dataset's median-across-
    trials envelope, plus the median of those across datasets.
    """
    with open(Path(batch_dir) / "combined.json") as f:
        raw = json.load(f)
    datasets = {}
    for result in raw:
        if result.get("error") or not result.get("pareto_frontier"):
            continue
        trials = datasets.setdefault(result["dataset_name"], {})
        trials[int(result.get("run_index") or 0)] = _test_pareto(
            result["pareto_frontier"]
        )
    datasets = {
        name: [trials[i] for i in sorted(trials)] for name, trials in datasets.items()
    }

    payload = {
        "protocol": {
            "trials_per_dataset": n_trials,
            "split": "75/25",
            "scale_x": True,
            "scale_y": True,
            "max_train_samples": max_train_samples,
            "selection": "test R2/complexity Pareto frontier",
            "layout": "datasets[name] = list of per-trial frontiers",
        },
        "n_trials_present": {name: len(t) for name, t in datasets.items()},
        "datasets": datasets,
    }
    json_path = Path(output_dir) / "srbench_black_box_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    if datasets:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        max_complexity = max(
            point["complexity"]
            for trials in datasets.values()
            for rows in trials
            for point in rows
        )
        grid = np.arange(1, max_complexity + 1)
        envelopes = []
        fig, ax = plt.subplots(figsize=(8, 5))
        # Low complexities can be NaN for every trial (no frontier point that
        # small), so all-NaN medians are expected -- don't warn about them.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for trials in datasets.values():
                # Median across this dataset's trials, then plot the curve.
                per_trial = np.asarray([_envelope(rows, grid) for rows in trials])
                values = np.nanmedian(per_trial, axis=0)
                envelopes.append(values)
                ax.plot(grid, values, color="tab:blue", alpha=0.06, linewidth=0.7)
            median = np.nanmedian(np.asarray(envelopes), axis=0)
        ax.plot(grid, median, color="black", linewidth=2.5,
                label=f"Median envelope ({len(datasets)} datasets)")
        ax.set(xlabel="Complexity", ylabel="Held-out test R²",
               title="SRBench black-box R²–complexity Pareto frontiers")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(output_dir) / "black_box_r2_complexity_pareto.png", dpi=180)
        plt.close(fig)
    return len(datasets)


def run_black_box(args, output_dir, source, manifest, run):
    datasets = ([d.strip() for d in args.datasets.split(",") if d.strip()]
                if args.datasets else load_black_box_datasets())
    n_trials = args.n_trials_per_dataset
    # Black-box datasets are much larger than the ground-truth ones, so they get
    # their own (higher) per-fit wall limit. <=0 means "no limit": the fit then
    # gets the whole SLURM --time minus a margin, which is the real bound anyway
    # (SLURM would kill the task at --time regardless). Keeping an explicit
    # number rather than None matters because the stall/job watchdogs floor
    # themselves off the wall limit -- with no limit at all a long-but-healthy
    # fit trips the 1800s job watchdog and the array gets cancelled.
    wall_limit = args.black_box_wall_limit
    if wall_limit <= 0:
        from parallel_eval_pysr import _slurm_time_to_seconds
        slurm_s = _slurm_time_to_seconds(args.time_limit)
        if slurm_s is None:
            raise ValueError(
                f"--black-box-wall-limit <= 0 needs a parseable --time-limit "
                f"(got {args.time_limit!r})"
            )
        wall_limit = max(60, int(slurm_s) - 120)
        print(f"  --black-box-wall-limit disabled: using --time-limit "
              f"{args.time_limit} -> {wall_limit}s per fit")
    print(f"Black-box: {len(datasets)} datasets x {n_trials} trials = "
          f"{len(datasets) * n_trials} runs  |  per-fit wall limit: {wall_limit}s")
    if source.backend == "fullsr":
        from parallel_eval_fullsr import FullSRSlurmEvaluator

        evaluator = FullSRSlurmEvaluator(
            results_dir=output_dir,
            partition=args.partition,
            time_limit=args.time_limit,
            mem_per_cpu=args.mem_per_cpu,
            dataset_max_samples=args.black_box_max_samples,
            data_seed=args.seed,
            use_cache=False,
            job_timeout=args.job_timeout,
            wall_limit=wall_limit,
            max_retries=args.max_retries,
        )
        before = set(evaluator.slurm_dir.glob("eval_*"))
        evaluator.evaluate_configs(
            configs=[source.config],
            dataset_names=datasets,
            seed=args.seed,
            n_runs=n_trials,
            fitness_metric="r2",
            fullsr_wall_limit=wall_limit,
            split_label="srbench_black_box",
            black_box=True,
        )
        created = sorted(set(evaluator.slurm_dir.glob("eval_*")) - before)
        if len(created) != 1:
            raise RuntimeError(
                f"Expected one new FullSR black-box batch, found {len(created)}"
            )
        batch_dir = created[0]
    else:
        from parallel_eval_pysr import PySRSlurmEvaluator

        evaluator = PySRSlurmEvaluator(
            results_dir=output_dir,
            partition=args.partition,
            time_limit=args.time_limit,
            mem_per_cpu=args.mem_per_cpu,
            dataset_max_samples=args.black_box_max_samples,
            data_seed=args.seed,
            use_cache=False,
            job_timeout=args.job_timeout,
            pysr_wall_limit=wall_limit,
            max_retries=args.max_retries,
        )
        handle = evaluator.submit_configs(
            configs=[source.config],
            dataset_names=datasets,
            seed=args.seed,
            n_runs=n_trials,
            fitness_metric="r2",
            black_box=True,
        )
        evaluator.collect_batch(handle)
        batch_dir = handle.batch_dir

    manifest["black_box"] = {
        "n_datasets": len(datasets), "datasets": datasets, "n_runs": n_trials,
        "wall_limit": wall_limit,
        "backend": source.backend,
        "batch_dir": (
            f"slurm_fullsr/{Path(batch_dir).name}"
            if source.backend == "fullsr"
            else f"slurm_pysr/{Path(batch_dir).name}"
        ),
        "max_train_samples": args.black_box_max_samples,
    }
    with open(Path(output_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    n_present = save_black_box_results(
        output_dir, batch_dir, args.black_box_max_samples, n_trials=n_trials
    )
    print(f"\nBlack-box results: {n_present}/{len(datasets)} dataset frontiers present.")
    if run is not None:
        run.log({
            "srbench_black_box/n_frontiers": n_present,
            "srbench_black_box/n_expected": len(datasets),
        })
    return evaluator


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
    parser.add_argument("--ground-truth", action="store_true",
                        help="Evaluate ground-truth problems (the default unless "
                             "--black-box is passed alone).")
    parser.add_argument("--black-box", action="store_true",
                        help="Evaluate the 122 SRBench black-box regression datasets "
                             "(supports PySR and FullSR evolution outputs).")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset override (for smoke tests).")
    parser.add_argument("--max-evals", type=int, default=1_000_000)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--black-box-max-samples", type=int, default=10_000,
                        help="Training-row cap for black-box datasets (SRBench protocol).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials-per-dataset", "--n-runs", type=int, default=10,
                        dest="n_trials_per_dataset",
                        help="Seeds per dataset, for both ground-truth (per "
                             "task/noise) and black-box evaluation (actual "
                             "seeds = seed .. seed+n_trials-1).")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS)

    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time-limit", type=str, default="02:00:00")
    parser.add_argument("--mem-per-cpu", type=str, default="8G")
    parser.add_argument("--job-timeout", type=float, default=None)
    parser.add_argument("--pysr-wall-limit", type=int, default=600)
    parser.add_argument("--fullsr-wall-limit", type=int, default=600,
                        help="Hard wall-clock limit for each FullSR fit.")
    parser.add_argument("--black-box-wall-limit", type=int, default=1800,
                        help="Hard wall-clock limit (seconds) per black-box fit. "
                             "Black-box datasets are far larger than the "
                             "ground-truth ones; 0 disables the limit entirely "
                             "(only SLURM --time bounds the fit).")
    parser.add_argument("--max-retries", type=int, default=5,
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
    print(f"Datasets: {len(datasets)}  |  seeds: {args.n_trials_per_dataset}  |  "
          f"noise levels: {args.noise_levels}  |  total runs: "
          f"{len(datasets) * args.n_trials_per_dataset * len(args.noise_levels)}")

    source = load_evaluation_source(args)
    config, mode_name, method_meta = source.config, source.mode, source.method_meta
    print(f"Mode: {mode_name}  |  backend: {source.backend}")
    do_ground_truth = args.ground_truth or not args.black_box

    # Black-box-only mode bypasses all ground-truth result aggregation.
    if not do_ground_truth:
        manifest = {
            "mode": mode_name, "backend": source.backend,
            "method_meta": method_meta, "max_evals": args.max_evals,
            "evaluation_types": ["black_box"], "batches": [],
        }
        with open(Path(output_dir) / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        run = None
        if not args.no_wandb:
            from wandb_utils import init_wandb
            run = init_wandb(
                config={**manifest,
                        "black_box_n_runs": args.n_trials_per_dataset},
                script_name="srbench_full_eval.py", output_dir=output_dir,
                extra_tags=["srbench_full_eval", "black_box", mode_name],
            )
        evaluator = run_black_box(args, output_dir, source, manifest, run)
        if run is not None:
            from wandb_utils import log_wandb_summary, finish_wandb
            log_wandb_summary(run, evaluator=evaluator)
            finish_wandb(run)
        copy_slurm_log(output_dir)
        print(f"\nDone. Results: {output_dir}")
        return

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
                "n_runs": args.n_trials_per_dataset,
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
        "n_runs": args.n_trials_per_dataset,
        "seeds": [args.seed + i for i in range(args.n_trials_per_dataset)],
        "noise_levels": args.noise_levels,
        "split_file": args.split_file,
        "n_datasets": len(datasets),
        "datasets": datasets,
        "unsolvable_tasks": list(srio.UNSOLVABLE_TASKS),
        "batches": [],
        "evaluation_types": ["ground_truth"] + (["black_box"] if args.black_box else []),
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
            n_runs=args.n_trials_per_dataset,
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
            "n_tasks": len(datasets) * args.n_trials_per_dataset * len(args.noise_levels),
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
                n_runs=args.n_trials_per_dataset,
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
        "n_runs": args.n_trials_per_dataset, "noise_levels": args.noise_levels,
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
        if not args.black_box:
            finish_wandb(run)

    if args.black_box:
        evaluator = run_black_box(args, output_dir, source, manifest, run)
        if run is not None:
            from wandb_utils import log_wandb_summary, finish_wandb
            log_wandb_summary(run, evaluator=evaluator)
            finish_wandb(run)

    copy_slurm_log(output_dir)
    print(f"\nDone. Inspect with: python inspect_srbench_results.py --run-id "
          f"{Path(output_dir).name}")


if __name__ == "__main__":
    main()
