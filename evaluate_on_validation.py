"""
Evaluate elite operators on validation set over the course of meta-evolution.

Parses log files to extract elite bundles at each improvement point,
then evaluates them on a held-out validation split using SlurmEvaluator.

Usage:
    # Evaluate baseline and final elite only
    python evaluate_on_validation.py --log out/meta_228219.out --results-dir results/run_20260115_214930

    # Evaluate all intermediate elites too
    python evaluate_on_validation.py --log out/meta_228219.out --results-dir results/run_20260115_214930 --all-elites

    # Specify validation split
    python evaluate_on_validation.py --log out/meta_228219.out --results-dir results/run_20260115_214930 --split splits/split_val.txt
"""

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

from meta_evolution import (
    OperatorBundle,
    create_operator,
    get_default_operator,
    OPERATOR_TYPES,
)
from parallel_eval import SlurmEvaluator
from utils import load_dataset_names_from_split


@dataclass
class EliteSnapshot:
    """Snapshot of elite bundle at a specific generation."""
    generation: int
    avg_r2: float  # R² on training split (from logs)
    evolved_operator: str  # Which operator was evolved this generation
    operator_codes: Dict[str, str]  # operator_type -> full code
    is_baseline: bool = False

    def to_bundle(self) -> OperatorBundle:
        """Reconstruct OperatorBundle from stored codes."""
        operators = {}
        for op_type in OPERATOR_TYPES:
            code = self.operator_codes.get(op_type)
            if code:
                operators[op_type] = create_operator(code, op_type)
            else:
                operators[op_type] = get_default_operator(op_type)

        return OperatorBundle(
            selection=operators['selection'],
            mutation=operators['mutation'],
            crossover=operators['crossover'],
            fitness=operators['fitness'],
        )


def get_default_operator_codes() -> Dict[str, str]:
    """Get the code for all default operators."""
    return {
        op_type: get_default_operator(op_type).code
        for op_type in OPERATOR_TYPES
    }


def parse_elite_snapshots(log_file: str) -> List[EliteSnapshot]:
    """
    Parse log file to extract elite bundle snapshots at each improvement point.

    Args:
        log_file: Path to the meta-evolution log file

    Returns:
        List of EliteSnapshot objects, one per improvement (including baseline)
    """
    with open(log_file, 'r') as f:
        content = f.read()

    snapshots = []

    # Start with baseline (default operators)
    baseline_match = re.search(r'Baseline: n_perfect=(\d+), avg_r2=([\d.]+)', content)
    if baseline_match:
        baseline_r2 = float(baseline_match.group(2))
        snapshots.append(EliteSnapshot(
            generation=0,
            avg_r2=baseline_r2,
            evolved_operator='baseline',
            operator_codes=get_default_operator_codes(),
            is_baseline=True,
        ))

    # Parse NEW ELITE FOUND blocks
    # Pattern: *** NEW ELITE FOUND (avg_r2=X.XXXX) ***
    # Followed by operator code blocks like:
    # --- FITNESS ---
    # <code>
    # --- SELECTION ---
    # <code>
    # etc.

    elite_pattern = r'\*\*\* NEW ELITE FOUND \(avg_r2=([\d.]+)\) \*\*\*\s*\n(.*?)(?=\n\nGeneration \d+ Summary:|\nGeneration \d+/\d+ -|\Z)'

    # We also need to track which generation each elite came from
    # Look for "Generation X Summary:" that follows each elite block

    current_operators = get_default_operator_codes().copy()

    for match in re.finditer(elite_pattern, content, re.DOTALL):
        elite_r2 = float(match.group(1))
        elite_block = match.group(2)

        # Parse each operator from the block
        for op_type in ['FITNESS', 'SELECTION', 'MUTATION', 'CROSSOVER']:
            op_pattern = rf'--- {op_type} ---\s*\n(.*?)(?=\n--- [A-Z]+ ---|\Z)'
            op_match = re.search(op_pattern, elite_block, re.DOTALL)
            if op_match:
                code = op_match.group(1).strip()
                current_operators[op_type.lower()] = code

        # Find the generation number from the summary that follows
        # Look for "Generation X Summary:" after this elite block
        pos = match.end()
        gen_match = re.search(r'Generation (\d+) Summary:', content[pos:pos+500])
        if gen_match:
            gen_num = int(gen_match.group(1))
        else:
            # Fallback: count previous generation summaries
            gen_num = len(snapshots)

        # Find which operator was evolved this generation
        evolved_match = re.search(rf'Generation {gen_num} Summary:.*?Evolved: (\w+)', content, re.DOTALL)
        evolved_op = evolved_match.group(1) if evolved_match else 'unknown'

        snapshots.append(EliteSnapshot(
            generation=gen_num,
            avg_r2=elite_r2,
            evolved_operator=evolved_op,
            operator_codes=current_operators.copy(),
            is_baseline=False,
        ))

    return snapshots


def parse_final_elite_from_results(results_dir: str) -> Optional[EliteSnapshot]:
    """
    Parse the final elite bundle from the results JSON file.

    Args:
        results_dir: Path to the results directory

    Returns:
        EliteSnapshot for the final elite, or None if not found
    """
    results_path = Path(results_dir)

    # Try meta_evolution_results.json first
    results_file = results_path / "meta_evolution_results.json"
    if not results_file.exists():
        results_file = results_path / "full_run_data.json"

    if not results_file.exists():
        print(f"Warning: No results file found in {results_dir}")
        return None

    with open(results_file, 'r') as f:
        data = json.load(f)

    # Extract operator codes
    operator_codes = {}
    for op_type in OPERATOR_TYPES:
        if op_type in data and 'code' in data[op_type]:
            operator_codes[op_type] = data[op_type]['code']

    if not operator_codes:
        # Try nested structure
        if 'final_results' in data:
            for op_type in OPERATOR_TYPES:
                if op_type in data['final_results'] and 'code' in data['final_results'][op_type]:
                    operator_codes[op_type] = data['final_results'][op_type]['code']

    if not operator_codes:
        print(f"Warning: Could not extract operator codes from {results_file}")
        return None

    # Get final R² from bundle metrics
    bundle_data = data.get('bundle', data.get('final_results', {}).get('bundle', {}))
    avg_r2 = bundle_data.get('avg_r2', 0.0)

    return EliteSnapshot(
        generation=-1,  # -1 indicates "final"
        avg_r2=avg_r2,
        evolved_operator='final',
        operator_codes=operator_codes,
        is_baseline=False,
    )


def evaluate_snapshots(
    snapshots: List[EliteSnapshot],
    split_file: str,
    output_dir: str,
    sr_kwargs: Dict = None,
    slurm_partition: str = "default_partition",
    slurm_time_limit: str = "01:00:00",
    seed: int = 42,
    data_seed: int = 0,  # Match main.py default (used for dataset subsampling)
    n_runs: int = 1,
    n_seeds: int = 3,
    max_samples: int = 1000,  # Match main.py default
    job_timeout: float = 600.0,
    max_concurrent_jobs: int = 200,  # Match main.py default
    use_cache: bool = True,
) -> List[Dict]:
    """
    Evaluate elite snapshots on a validation split using SlurmEvaluator.

    Args:
        snapshots: List of EliteSnapshot objects to evaluate
        split_file: Path to split file with dataset names
        output_dir: Directory for evaluation outputs
        sr_kwargs: SR algorithm parameters
        slurm_partition: SLURM partition to use
        slurm_time_limit: Time limit per task
        seed: Random seed
        n_runs: Number of runs per dataset
        max_samples: Max samples per dataset (None = all)
        job_timeout: Timeout for SLURM job

    Returns:
        List of result dicts with generation, train_r2, val_r2, etc.
    """
    if sr_kwargs is None:
        sr_kwargs = {
            'n_generations': 1000,
            'population_size': 100,
            'verbose': False,
        }

    # Load dataset names
    dataset_names = load_dataset_names_from_split(split_file)
    print(f"Evaluating on {len(dataset_names)} datasets from {split_file}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize SLURM evaluator
    evaluator = SlurmEvaluator(
        results_dir=str(output_path),
        partition=slurm_partition,
        time_limit=slurm_time_limit,
        dataset_max_samples=max_samples,
        data_seed=data_seed,
        job_timeout=job_timeout,
        max_concurrent_jobs=max_concurrent_jobs,
        use_cache=use_cache,
    )

    # First, reconstruct all bundles
    print(f"\nReconstructing {len(snapshots)} bundles...")
    bundles = []
    snapshot_errors = {}  # index -> error message

    for i, snapshot in enumerate(snapshots):
        gen_label = "baseline" if snapshot.is_baseline else (
            "final" if snapshot.generation == -1 else f"gen_{snapshot.generation:03d}"
        )
        try:
            bundle = snapshot.to_bundle()
            bundles.append(bundle)
            print(f"  [{i}] {gen_label}: OK (train R²={snapshot.avg_r2:.4f})")
        except Exception as e:
            bundles.append(None)  # Placeholder
            snapshot_errors[i] = str(e)
            print(f"  [{i}] {gen_label}: ERROR - {e}")

    # Filter to valid bundles for evaluation
    valid_indices = [i for i, b in enumerate(bundles) if b is not None]
    valid_bundles = [bundles[i] for i in valid_indices]

    if not valid_bundles:
        print("ERROR: No valid bundles to evaluate")
        return []

    # Evaluate ALL bundles in a single SLURM job
    print(f"\n{'='*60}")
    print(f"Evaluating {len(valid_bundles)} bundles on {len(dataset_names)} datasets")
    seeds = [seed + i for i in range(n_seeds)]
    total_tasks = len(valid_bundles) * len(dataset_names) * n_runs * len(seeds)
    print(f"Total tasks: {len(valid_bundles)} x {len(dataset_names)} x {n_runs} x {len(seeds)} = {total_tasks}")
    print(f"{'='*60}")

    try:
        eval_results_by_seed = []
        for seed_idx, eval_seed in enumerate(seeds):
            print(f"\nRunning seed {seed_idx + 1}/{len(seeds)} (seed={eval_seed})")
            eval_results = evaluator.evaluate_bundles(
                bundles=valid_bundles,
                dataset_names=dataset_names,
                sr_kwargs=sr_kwargs,
                seed=eval_seed,
                n_runs=n_runs,
            )
            eval_results_by_seed.append(eval_results)
    except Exception as e:
        print(f"ERROR: Batch evaluation failed: {e}")
        # Return error results for all snapshots
        results = []
        for snapshot in snapshots:
            gen_label = "baseline" if snapshot.is_baseline else (
                "final" if snapshot.generation == -1 else f"gen_{snapshot.generation:03d}"
            )
            results.append({
                'generation': snapshot.generation,
                'label': gen_label,
                'train_r2': snapshot.avg_r2,
                'val_r2': None,
                'val_score_vector': None,
                'error': str(e),
            })
        return results

    # Map results back to snapshots
    results = []
    eval_idx = 0  # Index into eval_results (only valid bundles)

    for i, snapshot in enumerate(snapshots):
        gen_label = "baseline" if snapshot.is_baseline else (
            "final" if snapshot.generation == -1 else f"gen_{snapshot.generation:03d}"
        )

        if i in snapshot_errors:
            # Bundle reconstruction failed
            results.append({
                'generation': snapshot.generation,
                'label': gen_label,
                'train_r2': snapshot.avg_r2,
                'val_r2': None,
                'val_score_vector': None,
                'error': snapshot_errors[i],
            })
        else:
            # Get result for this bundle
            per_seed = [seed_results[eval_idx] for seed_results in eval_results_by_seed]
            eval_idx += 1

            avg_r2s = [x[0] for x in per_seed]
            score_vectors = [x[1] for x in per_seed]
            trace_feedbacks = [x[2] for x in per_seed]
            avg_r2 = float(np.mean(avg_r2s)) if avg_r2s else None

            score_vector = None
            if score_vectors and all(v is not None for v in score_vectors):
                score_vector = np.mean(np.array(score_vectors, dtype=float), axis=0).tolist()

            per_dataset_avg = {}
            if trace_feedbacks:
                per_seed_maps = []
                for tf in trace_feedbacks:
                    per_seed_maps.append({t['dataset']: t['final_score'] for t in tf})
                for dataset in dataset_names:
                    values = [m[dataset] for m in per_seed_maps if dataset in m]
                    per_dataset_avg[dataset] = float(np.mean(values)) if values else None

            results.append({
                'generation': snapshot.generation,
                'label': gen_label,
                'evolved_operator': snapshot.evolved_operator,
                'train_r2': snapshot.avg_r2,
                'val_r2': avg_r2,
                'val_r2_seeds': avg_r2s,
                'val_score_vector': score_vector,
                'per_dataset': per_dataset_avg,
                'error': None,
            })

    # Print summary inline
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for r in results:
        if r['val_r2'] is not None:
            delta = r['val_r2'] - r['train_r2']
            seeds_str = r.get('val_r2_seeds')
            print(f"  {r['label']}: train={r['train_r2']:.4f} val_avg={r['val_r2']:.4f} val_seeds={seeds_str} (delta={delta:+.4f})")
        else:
            print(f"  {r['label']}: ERROR - {r['error']}")

    # Save results
    results_file = output_path / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return results


def print_summary(results: List[Dict]):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("VALIDATION EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Generation':<15} {'Evolved':<12} {'Train R²':>10} {'Val Avg':>10} {'Delta':>10} {'Val Seeds'}")
    print("-"*60)

    for r in results:
        gen_str = r['label']
        evolved = r.get('evolved_operator', '-')[:10]
        train_r2 = f"{r['train_r2']:.4f}" if r['train_r2'] is not None else "N/A"
        val_r2 = f"{r['val_r2']:.4f}" if r['val_r2'] is not None else "ERROR"
        val_seeds = r.get('val_r2_seeds')

        if r['val_r2'] is not None and r['train_r2'] is not None:
            delta = f"{r['val_r2'] - r['train_r2']:+.4f}"
        else:
            delta = "N/A"

        print(f"{gen_str:<15} {evolved:<12} {train_r2:>10} {val_r2:>10} {delta:>10} {val_seeds}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate elite operators on validation set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--log', type=str, required=True,
                       help='Path to meta-evolution log file (e.g., out/meta_228219.out)')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to results directory (e.g., results/run_20260115_214930)')
    parser.add_argument('--split', type=str, default='splits/split_val.txt',
                       help='Validation split file (default: splits/split_val.txt)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: <results-dir>/validation_eval)')

    # What to evaluate
    parser.add_argument('--all-elites', action='store_true',
                       help='Evaluate all intermediate elites, not just baseline and final')
    parser.add_argument('--baseline-only', action='store_true',
                       help='Only evaluate baseline')
    parser.add_argument('--final-only', action='store_true',
                       help='Only evaluate final elite')

    # SR parameters
    parser.add_argument('--n-generations', type=int, default=1000,
                       help='SR generations (default: 1000)')
    parser.add_argument('--population-size', type=int, default=100,
                       help='SR population size (default: 100)')
    parser.add_argument('--n-runs', type=int, default=1,
                       help='Number of runs per dataset (default: 1)')
    parser.add_argument('--n-seeds', type=int, default=3,
                       help='Number of evaluation seeds per bundle (default: 3)')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Max samples per dataset (default: 1000, matching main.py)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for SR (default: 42)')
    parser.add_argument('--data-seed', type=int, default=0,
                       help='Random seed for dataset subsampling (default: 0, matching main.py)')

    # SLURM parameters
    parser.add_argument('--partition', type=str, default='default_partition',
                       help='SLURM partition')
    parser.add_argument('--time-limit', type=str, default='01:00:00',
                       help='SLURM time limit per task (default: 01:00:00)')
    parser.add_argument('--job-timeout', type=float, default=600.0,
                       help='Overall job timeout in seconds (default: 600)')
    parser.add_argument('--max-concurrent-jobs', type=int, default=200,
                       help='Max concurrent SLURM jobs (default: 200)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable evaluation cache')

    # Dry run
    parser.add_argument('--dry-run', action='store_true',
                       help='Parse and print snapshots without evaluating')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.results_dir) / "validation_eval")

    # Parse elite snapshots from log
    print(f"Parsing elite snapshots from {args.log}...")
    snapshots = parse_elite_snapshots(args.log)
    print(f"Found {len(snapshots)} elite snapshots (including baseline)")

    # Also try to get final elite from results JSON (more reliable for final)
    final_snapshot = parse_final_elite_from_results(args.results_dir)

    # Select which snapshots to evaluate
    if args.baseline_only:
        snapshots_to_eval = [s for s in snapshots if s.is_baseline]
        print("Evaluating baseline only")
    elif args.final_only:
        if final_snapshot:
            snapshots_to_eval = [final_snapshot]
        else:
            # Use last snapshot from logs
            snapshots_to_eval = [snapshots[-1]] if snapshots else []
        print("Evaluating final elite only")
    elif args.all_elites:
        snapshots_to_eval = snapshots.copy()
        # Replace or add final from results JSON if available
        if final_snapshot:
            # Check if we already have this generation
            final_gens = [s.generation for s in snapshots_to_eval]
            if -1 not in final_gens:
                snapshots_to_eval.append(final_snapshot)
        print(f"Evaluating all {len(snapshots_to_eval)} elites")
    else:
        # Default: baseline and final only
        baseline = [s for s in snapshots if s.is_baseline]
        final = [final_snapshot] if final_snapshot else ([snapshots[-1]] if snapshots else [])
        snapshots_to_eval = baseline + final
        print("Evaluating baseline and final elite")

    # Print what we found
    print("\nSnapshots to evaluate:")
    for s in snapshots_to_eval:
        label = "BASELINE" if s.is_baseline else ("FINAL" if s.generation == -1 else f"Gen {s.generation}")
        print(f"  {label}: train R²={s.avg_r2:.4f}, evolved={s.evolved_operator}")
        for op_type, code in s.operator_codes.items():
            preview = code[:60].replace('\n', ' ') + "..." if len(code) > 60 else code.replace('\n', ' ')
            print(f"    {op_type}: {preview}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping evaluation")
        return

    # Build SR kwargs
    # SR kwargs - match main.py format exactly for cache compatibility
    sr_kwargs = {
        'n_generations': args.n_generations,
        'population_size': args.population_size,
        'verbose': False,
    }

    # Run evaluation
    results = evaluate_snapshots(
        snapshots=snapshots_to_eval,
        split_file=args.split,
        output_dir=args.output_dir,
        sr_kwargs=sr_kwargs,
        slurm_partition=args.partition,
        slurm_time_limit=args.time_limit,
        seed=args.seed,
        data_seed=args.data_seed,
        n_runs=args.n_runs,
        n_seeds=args.n_seeds,
        max_samples=args.max_samples,
        job_timeout=args.job_timeout,
        max_concurrent_jobs=args.max_concurrent_jobs,
        use_cache=not args.no_cache,
    )

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
