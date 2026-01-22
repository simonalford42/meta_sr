"""
Minimal test script to evaluate default bundle on validation split.
Used to debug slowness in evaluate_on_validation.py
"""

import argparse
from meta_evolution import OperatorBundle
from parallel_eval import SlurmEvaluator
from utils import load_dataset_names_from_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='splits/split_val.txt')
    parser.add_argument('--max-samples', type=int, default=1000)
    parser.add_argument('--partition', default='default_partition')
    parser.add_argument('--n-generations', type=int, default=1000)
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=0)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--job-timeout', type=float, default=600.0)
    parser.add_argument('--output-dir', default='test_val_eval_output')
    args = parser.parse_args()

    # Load datasets
    dataset_names = load_dataset_names_from_split(args.split)
    print(f"Datasets ({len(dataset_names)}): {dataset_names}")

    # Create default bundle
    bundle = OperatorBundle.create_default()
    print(f"\nDefault bundle created")
    print(f"  Fitness LOC: {bundle.fitness.lines_of_code}")
    print(f"  Selection LOC: {bundle.selection.lines_of_code}")
    print(f"  Mutation LOC: {bundle.mutation.lines_of_code}")
    print(f"  Crossover LOC: {bundle.crossover.lines_of_code}")

    # SR kwargs - match main.py
    sr_kwargs = {
        'n_generations': args.n_generations,
        'population_size': args.population_size,
        'verbose': False,
    }
    print(f"\nSR kwargs: {sr_kwargs}")

    # Create evaluator - match main.py settings
    evaluator = SlurmEvaluator(
        results_dir=args.output_dir,
        partition=args.partition,
        time_limit='01:00:00',
        mem_per_cpu='4G',
        dataset_max_samples=args.max_samples,
        data_seed=args.data_seed,
        max_retries=3,
        max_concurrent_jobs=200,
        job_timeout=args.job_timeout,
        use_cache=not args.no_cache,
    )

    print(f"\nEvaluator config:")
    print(f"  partition: {args.partition}")
    print(f"  max_samples: {args.max_samples}")
    print(f"  data_seed: {args.data_seed}")
    print(f"  use_cache: {not args.no_cache}")
    print(f"  job_timeout: {args.job_timeout}")

    # Evaluate
    print(f"\n{'='*60}")
    print(f"Evaluating default bundle on {len(dataset_names)} datasets")
    print(f"{'='*60}")

    results = evaluator.evaluate_bundles(
        bundles=[bundle],
        dataset_names=dataset_names,
        sr_kwargs=sr_kwargs,
        seed=args.seed,
        n_runs=1,
    )

    avg_r2, score_vector, trace_feedback = results[0]

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Average R²: {avg_r2:.4f}")
    print(f"\nPer-dataset:")
    for tf in trace_feedback:
        print(f"  {tf['dataset']}: {tf['final_score']:.4f}")


if __name__ == '__main__':
    main()
