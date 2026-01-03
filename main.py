import numpy as np
import random
from typing import List, Dict, Optional
import json
import sys
from datetime import datetime
from pathlib import Path
from meta_evolution import (
    Operator,
    OperatorBundle,
    create_and_test_operator,
    semantics_aware_selection,
    mutate_operator,
    crossover_operators,
)
from completions import print_usage
from utils import load_datasets_from_split, TeeLogger, load_dataset_names_from_split
from parallel_eval import SlurmEvaluator


def compute_meta_score(score_vector: List[float]) -> tuple:
    """
    Compute meta-evolution metrics from a vector of R² scores.

    Note: Selection uses avg_r2 directly, but we still track n_perfect for logging.

    Returns:
        (composite_score, n_perfect, avg_r2)
        - composite_score: n_perfect + avg_r2 (legacy, not used for selection)
        - n_perfect: count of tasks with R² = 1.0
        - avg_r2: average R² (clipped to minimum 0)
    """
    n_perfect = sum(1 for s in score_vector if s >= 1.0 - 1e-9)  # Allow small floating point tolerance
    avg_r2 = float(np.mean([max(0.0, s) for s in score_vector]))
    composite_score = n_perfect + avg_r2
    return composite_score, n_perfect, avg_r2


class RunLogger:
    """Tracks and saves all run data including operators and scores."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up tee logging
        self.log_file = self.output_dir / "run.log"
        self.tee = TeeLogger(str(self.log_file))
        sys.stdout = self.tee

        # Initialize run data
        self.run_data = {
            "start_time": datetime.now().isoformat(),
            "config": {},
            "stages": {},
        }

    def set_config(self, config: Dict):
        """Save run configuration."""
        self.run_data["config"] = config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def start_stage(self, operator_type: str):
        """Initialize tracking for a new stage."""
        self.run_data["stages"][operator_type] = {
            "generations": [],
            "best_score_history": [],
            "best_operator": None,
        }

    def log_generation(
        self,
        operator_type: str,
        generation: int,
        population: List[Operator],
        best_operator: Operator,
        full_eval_pct: Optional[float] = None,
    ):
        """Log data for a single generation."""
        stage_data = self.run_data["stages"][operator_type]

        gen_data = {
            "generation": generation,
            "population": [
                {
                    "score": op.score,
                    "n_perfect": op.n_perfect,
                    "avg_r2": op.avg_r2,
                    "score_vector": op.score_vector,
                    "loc": op.lines_of_code,
                    "code": op.code,
                    "quick_eval_only": getattr(op, 'quick_eval_only', False),
                }
                for op in population
            ],
            "best_n_perfect": best_operator.n_perfect,
            "best_avg_r2": best_operator.avg_r2,
            "avg_n_perfect": float(np.mean([op.n_perfect for op in population])),
            "avg_avg_r2": float(np.mean([op.avg_r2 for op in population])),
            "avg_loc": float(np.mean([op.lines_of_code for op in population])),
            "full_eval_pct": full_eval_pct,
        }

        stage_data["generations"].append(gen_data)
        stage_data["best_score_history"].append(best_operator.score)
        stage_data.setdefault("best_n_perfect_history", []).append(best_operator.n_perfect)

        # Save intermediate generation data
        stages_dir = self.output_dir / "stages" / operator_type
        stages_dir.mkdir(parents=True, exist_ok=True)
        gen_file = stages_dir / f"gen_{generation:03d}.json"
        with open(gen_file, "w") as f:
            json.dump(gen_data, f, indent=2)

    def log_stage_complete(self, operator_type: str, best_operator: Operator):
        """Log completion of a stage."""
        stage_data = self.run_data["stages"][operator_type]
        stage_data["best_operator"] = {
            "score": best_operator.score,
            "n_perfect": best_operator.n_perfect,
            "avg_r2": best_operator.avg_r2,
            "score_vector": best_operator.score_vector,
            "loc": best_operator.lines_of_code,
            "code": best_operator.code,
        }

        # Save best operator code
        stages_dir = self.output_dir / "stages" / operator_type
        stages_dir.mkdir(parents=True, exist_ok=True)
        with open(stages_dir / "best_operator.py", "w") as f:
            f.write(best_operator.code)

        # Save stage summary
        stage_file = stages_dir / "summary.json"
        with open(stage_file, "w") as f:
            json.dump(stage_data, f, indent=2)

    def log_bundle_generation(
        self,
        generation: int,
        operator_type: str,
        population: List[OperatorBundle],
        best_bundle: OperatorBundle,
    ):
        """Log data for a single generation in round-robin bundle evolution."""
        gen_data = {
            "generation": generation,
            "evolved_operator_type": operator_type,
            "population": [
                {
                    "score": bundle.score,
                    "n_perfect": bundle.n_perfect,
                    "avg_r2": bundle.avg_r2,
                    "score_vector": bundle.score_vector,
                    "operators": {
                        op_type: {
                            "loc": bundle.get_operator(op_type).lines_of_code,
                            "code_preview": bundle.get_operator(op_type).code[:200],
                        }
                        for op_type in ["fitness", "selection", "mutation", "crossover"]
                    },
                }
                for bundle in population
            ],
            "best_n_perfect": best_bundle.n_perfect,
            "best_avg_r2": best_bundle.avg_r2,
            "avg_n_perfect": float(np.mean([b.n_perfect for b in population])),
            "avg_avg_r2": float(np.mean([b.avg_r2 for b in population])),
        }

        # Store in run_data
        self.run_data.setdefault("generations", []).append(gen_data)

        # Save intermediate generation data
        gen_dir = self.output_dir / "generations"
        gen_dir.mkdir(parents=True, exist_ok=True)
        gen_file = gen_dir / f"gen_{generation:03d}.json"
        with open(gen_file, "w") as f:
            json.dump(gen_data, f, indent=2)

    def save_final_results(self, results: Dict):
        """Save final results."""
        self.run_data["end_time"] = datetime.now().isoformat()
        self.run_data["final_results"] = results

        # Save complete run data
        with open(self.output_dir / "full_run_data.json", "w") as f:
            json.dump(self.run_data, f, indent=2)

        # Also save in the standard format for compatibility
        with open(self.output_dir / "meta_evolution_results.json", "w") as f:
            json.dump(results, f, indent=2)

    def close(self):
        """Clean up and restore stdout."""
        sys.stdout = self.tee.terminal
        self.tee.close()
        print(f"\nAll logs and results saved to: {self.output_dir}/")


# Global logger instance (set during run)
_run_logger: Optional[RunLogger] = None


def _print_bundle_results(bundles: List[OperatorBundle], show_all_runs: bool = True):
    """Print results for all bundles, optionally showing all run scores."""
    for i, bundle in enumerate(bundles):
        skipped = getattr(bundle, 'quick_eval_only', False)
        status = " (SKIPPED - no improvement on quick eval)" if skipped else ""
        print(f"Bundle {i+1}:{status}")
        if hasattr(bundle, 'trace_feedback') and bundle.trace_feedback:
            for tf in bundle.trace_feedback:
                dataset = tf["dataset"]
                score = tf["final_score"]
                run_scores = tf.get("run_scores", [])
                error = tf.get("error")
                if error:
                    print(f"    {dataset}: R²={score:.4f} (Error: {error})")
                elif show_all_runs and run_scores and len(run_scores) > 1:
                    runs_str = " ".join(f"{s:.2f}" for s in run_scores)
                    print(f"    {dataset}: R²={score:.4f} ({runs_str})")
                else:
                    print(f"    {dataset}: R²={score:.4f}")
        print(f"  n_perfect={bundle.n_perfect}, avg_r2={bundle.avg_r2:.4f}")


def _apply_results_to_bundle(bundle: OperatorBundle, avg_score: float, score_vector: List[float], trace_feedback: List[Dict]):
    """Apply evaluation results to a bundle and propagate to operators."""
    bundle.score_vector = score_vector
    bundle.score, bundle.n_perfect, bundle.avg_r2 = compute_meta_score(score_vector)
    bundle.trace_feedback = trace_feedback

    # Propagate bundle scores to individual operators for semantics-aware selection
    for op_type in ["fitness", "selection", "mutation", "crossover"]:
        op = bundle.get_operator(op_type)
        op.score_vector = score_vector
        op.score = bundle.score
        op.n_perfect = bundle.n_perfect
        op.avg_r2 = bundle.avg_r2


def evaluate_bundle_population(
    bundles: List[OperatorBundle],
    label: str,
    datasets: Dict,
    sr_kwargs: Dict,
    slurm_evaluator: SlurmEvaluator,
    seed: Optional[int] = None,
    n_runs: Optional[int] = 1,
    quick_eval_datasets: Optional[List[str]] = None,
    baseline_quick_scores: Optional[Dict[str, float]] = None,
) -> None:
    """
    Evaluate a population of bundles, with optional quick-eval filtering.

    Mutates bundles in-place, setting score, n_perfect, avg_r2, score_vector, trace_feedback.
    Also propagates these scores to individual operators within each bundle for
    semantics-aware selection.

    Args:
        bundles: List of OperatorBundle objects to evaluate
        label: Label for logging (e.g., "Initial Population")
        datasets: Dictionary of datasets to evaluate on
        sr_kwargs: Arguments for symbolic regression algorithm
        slurm_evaluator: SlurmEvaluator for multi-node parallelization
        seed: Base random seed for reproducibility
        n_runs: Number of runs per evaluation
        quick_eval_datasets: If provided, first evaluate on this subset (2 runs).
            Bundles that don't improve on any task vs baseline skip full eval.
        baseline_quick_scores: Dict mapping dataset name -> baseline R² score.
            Required if quick_eval_datasets is provided.
    """
    print(f"\n=== Evaluating {label} ({len(bundles)} bundles) ===")

    if not bundles:
        return

    all_dataset_names = list(datasets.keys())

    # Quick eval mode: first evaluate on subset, filter out non-improving bundles
    if quick_eval_datasets and baseline_quick_scores:
        print(f"  Quick eval: {len(quick_eval_datasets)} datasets, 2 runs each")

        quick_results = slurm_evaluator.evaluate_bundles(
            bundles=bundles,
            dataset_names=quick_eval_datasets,
            sr_kwargs=sr_kwargs,
            seed=seed if seed is not None else 42,
            n_runs=min(2, n_runs) if n_runs is not None else 2,
        )

        # Determine which bundles show improvement
        bundles_to_full_eval = []
        bundles_skipped = []

        for bundle, (avg_score, score_vector, trace_feedback) in zip(bundles, quick_results):
            # Check if any task improved over baseline
            improved = False
            for tf in trace_feedback:
                dataset = tf["dataset"]
                score = tf["final_score"]
                baseline = baseline_quick_scores.get(dataset, -1.0)
                if score > baseline:
                    improved = True
                    break

            if improved:
                bundles_to_full_eval.append(bundle)
            else:
                # Mark as skipped, assign -1 scores for all datasets
                bundle.quick_eval_only = True
                full_score_vector = []
                full_trace_feedback = []
                for ds_name in all_dataset_names:
                    if ds_name in quick_eval_datasets:
                        # Use quick eval results for quick eval datasets
                        idx = quick_eval_datasets.index(ds_name)
                        full_score_vector.append(score_vector[idx])
                        full_trace_feedback.append(trace_feedback[idx])
                    else:
                        # -1 for skipped datasets
                        full_score_vector.append(-1.0)
                        full_trace_feedback.append({
                            "dataset": ds_name,
                            "traces": [],
                            "final_score": -1.0,
                            "run_scores": [],
                            "error": "Skipped (no improvement on quick eval)",
                        })
                _apply_results_to_bundle(bundle, float(np.mean(full_score_vector)), full_score_vector, full_trace_feedback)
                bundles_skipped.append(bundle)

        print(f"  Quick eval complete: {len(bundles_to_full_eval)} bundles improved, {len(bundles_skipped)} skipped")

        if not bundles_to_full_eval:
            # All bundles skipped
            _print_bundle_results(bundles)
            return

        # Full eval only for bundles that improved
        print(f"  Full eval: {len(bundles_to_full_eval)} bundles on {len(all_dataset_names)} datasets")
        full_results = slurm_evaluator.evaluate_bundles(
            bundles=bundles_to_full_eval,
            dataset_names=all_dataset_names,
            sr_kwargs=sr_kwargs,
            seed=seed if seed is not None else 42,
            n_runs=n_runs if n_runs is not None else 1,
        )

        for bundle, (avg_score, score_vector, trace_feedback) in zip(bundles_to_full_eval, full_results):
            bundle.quick_eval_only = False
            _apply_results_to_bundle(bundle, avg_score, score_vector, trace_feedback)

    else:
        # No quick eval - evaluate all bundles on all datasets
        results = slurm_evaluator.evaluate_bundles(
            bundles=bundles,
            dataset_names=all_dataset_names,
            sr_kwargs=sr_kwargs,
            seed=seed if seed is not None else 42,
            n_runs=n_runs if n_runs is not None else 1,
        )

        for bundle, (avg_score, score_vector, trace_feedback) in zip(bundles, results):
            bundle.quick_eval_only = False
            _apply_results_to_bundle(bundle, avg_score, score_vector, trace_feedback)

    # Print results for all bundles
    _print_bundle_results(bundles)


def evaluate_final_bundle(
    final_bundle: OperatorBundle,
    output_dir: str,
    sr_kwargs: Dict,
    seed: int,
    slurm_config: Optional[Dict] = None,
    split_file: str = 'splits/split_val.txt',
    n_runs: int = 1,
) -> Optional[Dict]:
    """
    Evaluate the final bundle on SRBench datasets and save results.

    Args:
        final_bundle: The evolved OperatorBundle to evaluate
        output_dir: Directory to save results
        sr_kwargs: Arguments for symbolic regression algorithm
        seed: Random seed for reproducibility
        slurm_config: SLURM configuration dict with keys: partition, time_limit, mem_per_cpu, etc.
        split_file: Path to file listing dataset names to evaluate on
        n_runs: Number of runs per dataset for averaging (default: 1)

    Returns:
        Dict with evaluation results, or None if SLURM not enabled
    """
    print("\n" + "=" * 80)
    print("EVALUATING FINAL BUNDLE ON SRBENCH")
    print("=" * 80)

    if not slurm_config:
        print("Skipping SRBench evaluation (SLURM not enabled)")
        print("To evaluate, run with --slurm flag or use run_sr_srbench.py separately")
        return None

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use a separate subdirectory for final evaluation to avoid batch ID collision
    # with the main evolution's SLURM evaluator
    final_eval_dir = str(output_path / "final_eval")
    slurm_evaluator = SlurmEvaluator(
        results_dir=final_eval_dir,
        partition=slurm_config.get('partition', 'default_partition'),
        time_limit=slurm_config.get('time_limit', '02:00:00'),
        mem_per_cpu=slurm_config.get('mem_per_cpu', '4G'),
        dataset_max_samples=slurm_config.get('dataset_max_samples'),
        data_seed=slurm_config.get('data_seed', seed),
        max_retries=slurm_config.get('max_retries', 3),
        exclude_nodes=slurm_config.get('exclude_nodes'),
        constraint=slurm_config.get('constraint'),
    )

    srbench_datasets = load_dataset_names_from_split(split_file)
    print(f"Evaluating on {len(srbench_datasets)} SRBench datasets...")

    eval_results = slurm_evaluator.evaluate_bundles(
        bundles=[final_bundle],
        dataset_names=srbench_datasets,
        sr_kwargs=sr_kwargs,
        seed=seed,
        n_runs=n_runs,
    )

    # Extract results for the single bundle (index 0)
    avg_score, score_vector, trace_feedback = eval_results[0]

    # Build per-dataset results
    srbench_results = {
        "avg_r2": avg_score,
        "n_datasets": len(srbench_datasets),
        "sr_kwargs": sr_kwargs,
        "seed": seed,
        "per_dataset": {}
    }
    for i, dataset_name in enumerate(srbench_datasets):
        srbench_results["per_dataset"][dataset_name] = {
            "r2": score_vector[i] if i < len(score_vector) else None,
            "traces": trace_feedback[i].get("traces", []) if i < len(trace_feedback) else [],
            "error": trace_feedback[i].get("error") if i < len(trace_feedback) else None,
        }

    # Save results to main output directory (not the final_eval subdirectory)
    srbench_output_path = output_path / "srbench_evaluation.json"
    with open(srbench_output_path, "w") as f:
        json.dump(srbench_results, f, indent=2)

    print(f"\nSRBench Evaluation Results:")
    print(f"  Average R²: {avg_score:.4f}")
    print(f"  Results saved to: {srbench_output_path}")

    # Print per-dataset summary
    print(f"\nPer-dataset R² scores:")
    for dataset_name in srbench_datasets:
        r2 = srbench_results["per_dataset"][dataset_name]["r2"]
        error = srbench_results["per_dataset"][dataset_name]["error"]
        if error:
            print(f"  {dataset_name}: ERROR - {error}")
        elif r2 is not None:
            print(f"  {dataset_name}: R²={r2:.4f}")

    return srbench_results


def run_meta_evolution(
    n_generations: int,
    population_size: int,
    n_crossover: int,
    n_mutation: int,
    sr_kwargs: Dict,
    model: str,
    output_dir: str,
    slurm_config: Dict,
    datasets: Optional[Dict] = None,
    operator_types: List[str] = None,
    use_trace_feedback: bool = False,
    seed: Optional[int] = None,
    n_runs: int = 1,
    llm_temperature: float = 0.7,
    llm_seed: Optional[int] = None,
) -> OperatorBundle:
    """
    Run meta-evolution with round-robin operator type selection.

    Each generation evolves one operator type, cycling through all types.
    Population consists of full bundles (all 4 operator types together).

    Args:
        n_generations: Total number of generations
        population_size: Number of bundles in population
        n_crossover: Number of offspring generated by crossover per generation
        n_mutation: Number of offspring generated by mutation per generation
        sr_kwargs: Arguments for symbolic regression algorithm
        model: Model to use for LLM calls
        output_dir: Directory to save all logs and results (required).
        slurm_config: SLURM configuration for multi-node evaluation.
        datasets: Dictionary of datasets to evaluate on.
        operator_types: Order of operator types to cycle through (default: fitness, selection, mutation, crossover)
        use_trace_feedback: Whether to include SR evolution traces in mutation/crossover prompts.

    Returns:
        final_bundle: The best OperatorBundle found
    """
    if operator_types is None:
        operator_types = ["fitness", "selection", "mutation", "crossover"]

    # Create run logger for comprehensive logging
    logger = RunLogger(output_dir)

    # Create SLURM evaluator
    slurm_evaluator = SlurmEvaluator(
        results_dir=str(logger.output_dir),
        partition=slurm_config.get('partition', 'default_partition'),
        time_limit=slurm_config.get('time_limit', '01:00:00'),
        mem_per_cpu=slurm_config.get('mem_per_cpu', '4G'),
        dataset_max_samples=slurm_config.get('dataset_max_samples', None),
        data_seed=slurm_config.get('data_seed', 42),
        max_retries=slurm_config.get('max_retries', 3),
        exclude_nodes=slurm_config.get('exclude_nodes'),
        constraint=slurm_config.get('constraint'),
    )

    # Save configuration
    config = {
        "n_generations": n_generations,
        "population_size": population_size,
        "n_crossover": n_crossover,
        "n_mutation": n_mutation,
        "sr_kwargs": sr_kwargs,
        "model": model,
        "operator_types": operator_types,
        "use_trace_feedback": use_trace_feedback,
        "slurm_config": slurm_config,
    }
    logger.set_config(config)

    # Load datasets if not provided (fallback to default split)
    if datasets is None:
        print("No datasets provided; loading default split 'splits/split_train_small.txt'.")
        datasets = load_datasets_from_split('splits/split_train_small.txt', max_samples=1000)
    all_dataset_names = list(datasets.keys())
    print(f"Datasets: {all_dataset_names}")

    # Select 1/4 of datasets for quick evaluation (fixed for entire evolution)
    n_quick = max(1, len(all_dataset_names) // 4)
    rng = random.Random(seed if seed is not None else 42)
    quick_eval_datasets = rng.sample(all_dataset_names, n_quick)
    print(f"Quick eval datasets ({n_quick}/{len(all_dataset_names)}): {quick_eval_datasets}")

    try:
        # === Generation 0: Baseline with default operators ===
        print("\n" + "=" * 80)
        print("Generation 0 - BASELINE (default operators)")
        print("=" * 80)

        default_bundle = OperatorBundle.create_default()
        population = [default_bundle]

        # Evaluate baseline (full eval, no quick filtering for baseline)
        evaluate_bundle_population(
            bundles=population,
            label="Baseline",
            datasets=datasets,
            sr_kwargs=sr_kwargs,
            slurm_evaluator=slurm_evaluator,
            seed=seed,
            n_runs=n_runs,
        )

        best_bundle = default_bundle
        print(f"\nBaseline: n_perfect={best_bundle.n_perfect}, avg_r2={best_bundle.avg_r2:.4f}")

        # Extract baseline scores on quick eval datasets for comparison
        baseline_quick_scores = {}
        for tf in default_bundle.trace_feedback:
            if tf["dataset"] in quick_eval_datasets:
                baseline_quick_scores[tf["dataset"]] = tf["final_score"]
        print(f"Baseline quick eval scores: {baseline_quick_scores}")

        # Log generation 0
        logger.log_bundle_generation(
            generation=0,
            operator_type="baseline",
            population=population,
            best_bundle=best_bundle,
        )

        # === Main evolution loop ===
        for gen in range(n_generations):
            # Round-robin operator type selection
            operator_type = operator_types[gen % len(operator_types)]

            print(f"\n{'='*60}")
            print(f"Generation {gen+1}/{n_generations} - Evolving {operator_type.upper()}")
            print(f"{'='*60}")

            # Elitism: keep the best bundle (by avg_r2, not n_perfect)
            elite = max(population, key=lambda b: b.avg_r2)
            print(f"Elite: n_perfect={elite.n_perfect}, avg_r2={elite.avg_r2:.4f}")

            offspring = []

            # Crossover
            print(f"\nGenerating {n_crossover} offspring via crossover...")
            for i in range(n_crossover):
                print(f"\n--- Crossover {i+1}/{n_crossover} ---")
                print(f"Parent operators for {operator_type}:")
                for attempt in range(10):
                    # Select two parent bundles based on overall score
                    # Use semantics_aware_selection on the operators of the current type
                    parent_operators = [b.get_operator(operator_type) for b in population]
                    # Temporarily attach bundle reference to operators for selection
                    for op, bundle in zip(parent_operators, population):
                        op._parent_bundle = bundle

                    parent_a_op, parent_b_op = semantics_aware_selection(parent_operators)
                    parent_a = parent_a_op._parent_bundle
                    parent_b = parent_b_op._parent_bundle

                    # if attempt == 0:
                        # print(f"  Parent A ({operator_type}):\n{parent_a_op.code[:300]}...")
                        # print(f"  Parent B ({operator_type}):\n{parent_b_op.code[:300]}...")

                    # Crossover the specific operator type
                    code = crossover_operators(
                        parent_a.get_operator(operator_type),
                        parent_b.get_operator(operator_type),
                        operator_type=operator_type,
                        model=model,
                        use_trace_feedback=use_trace_feedback,
                        llm_temperature=llm_temperature,
                        llm_seed=llm_seed,
                    )
                    print(f"  [Attempt {attempt+1}] Generated crossover code:\n{code}...")
                    new_op, passed, error = create_and_test_operator(code, operator_type)

                    if passed:
                        # Create new bundle: use operators from the better-scoring parent (by avg_r2),
                        # except for the crossed-over operator type
                        better_parent = parent_a if parent_a.avg_r2 >= parent_b.avg_r2 else parent_b
                        new_bundle = OperatorBundle(
                            selection=better_parent.selection if operator_type != "selection" else new_op,
                            mutation=better_parent.mutation if operator_type != "mutation" else new_op,
                            crossover=better_parent.crossover if operator_type != "crossover" else new_op,
                            fitness=better_parent.fitness if operator_type != "fitness" else new_op,
                        )
                        offspring.append(new_bundle)
                        print(f"  Crossover {i+1}: Parent A n_perfect={parent_a.n_perfect}, Parent B n_perfect={parent_b.n_perfect}")
                        break
                else:
                    raise ValueError(f"Failed to generate valid {operator_type} crossover after 10 attempts")

            # Mutation
            print(f"\nGenerating {n_mutation} offspring via mutation...")
            for i in range(n_mutation):
                print(f"\n--- Mutation {i+1}/{n_mutation} ---")
                # print(f"  Mutating elite's {operator_type} operator:\n{elite.get_operator(operator_type).code[:300]}...")
                for attempt in range(10):
                    code = mutate_operator(
                        elite.get_operator(operator_type),
                        operator_type=operator_type,
                        model=model,
                        use_trace_feedback=use_trace_feedback,
                        llm_temperature=llm_temperature,
                        llm_seed=llm_seed,
                        sample_index=gen * n_mutation + i * 10 + attempt,
                    )
                    print(f"  [Attempt {attempt+1}] Generated mutation code:\n{code}...")
                    new_op, passed, error = create_and_test_operator(code, operator_type)

                    if passed:
                        # Create new bundle: use elite's operators except for the mutated one
                        new_bundle = OperatorBundle(
                            selection=elite.selection if operator_type != "selection" else new_op,
                            mutation=elite.mutation if operator_type != "mutation" else new_op,
                            crossover=elite.crossover if operator_type != "crossover" else new_op,
                            fitness=elite.fitness if operator_type != "fitness" else new_op,
                        )
                        offspring.append(new_bundle)
                        print(f"  Mutation {i+1}: from elite")
                        break
                else:
                    raise ValueError(f"Failed to generate valid {operator_type} mutation after 10 attempts")

            # Evaluate offspring with quick eval filtering
            if offspring:
                evaluate_bundle_population(
                    bundles=offspring,
                    label=f"{len(offspring)} offspring",
                    datasets=datasets,
                    sr_kwargs=sr_kwargs,
                    slurm_evaluator=slurm_evaluator,
                    seed=seed,
                    n_runs=n_runs,
                    quick_eval_datasets=quick_eval_datasets,
                    baseline_quick_scores=baseline_quick_scores,
                )

            # Update population: elite + offspring
            population = [elite] + offspring

            # Track best (by avg_r2, not n_perfect)
            current_best = max(population, key=lambda b: b.avg_r2)
            if current_best.avg_r2 > best_bundle.avg_r2:
                best_bundle = current_best

            # Generation summary
            print(f"\nGeneration {gen+1} Summary:")
            print(f"  Evolved: {operator_type}")
            print(f"  Best: n_perfect={best_bundle.n_perfect}, avg_r2={best_bundle.avg_r2:.4f}")
            print(f"  Population size: {len(population)}")
            print(f"  Avg population n_perfect: {np.mean([b.n_perfect for b in population]):.2f}")
            print(f"  Avg population avg_r2: {np.mean([b.avg_r2 for b in population]):.4f}")

            # Log generation data
            logger.log_bundle_generation(
                generation=gen + 1,
                operator_type=operator_type,
                population=population,
                best_bundle=best_bundle,
            )

        # === Final results ===
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)

        print(f"\nBest bundle: n_perfect={best_bundle.n_perfect}, avg_r2={best_bundle.avg_r2:.4f}")
        results = {}
        for op_type in operator_types:
            op = best_bundle.get_operator(op_type)
            results[op_type] = {
                "loc": op.lines_of_code,
                "code": op.code,
            }
            print(f"\n{op_type.upper()}:")
            print(f"  LOC: {op.lines_of_code}")
            print(f"  Code:\n{op.code}...")

        # Add bundle-level metrics to results
        results["bundle"] = {
            "score": best_bundle.score,
            "n_perfect": best_bundle.n_perfect,
            "avg_r2": best_bundle.avg_r2,
            "score_vector": best_bundle.score_vector,
        }

        logger.save_final_results(results)

    finally:
        # Always close the logger to restore stdout and save state
        logger.close()

    return best_bundle


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run meta-evolution of SR algorithm',
    )

    # SLURM configuration
    parser.add_argument('--partition', type=str, default='default_partition',
                       help='SLURM partition (default: default_partition)')
    parser.add_argument('--time-limit', type=str, default='01:00:00',
                       help='SLURM time limit per task (default: 01:00:00)')
    parser.add_argument('--mem-per-cpu', type=str, default='4G',
                       help='SLURM memory per CPU (default: 4G)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Max retries for failed SLURM tasks (default: 3)')
    parser.add_argument('--exclude-nodes', type=str, default=None,
                       help='Comma-separated list of SLURM nodes to exclude')
    parser.add_argument('--constraint', type=str, default=None,
                       help='SLURM constraint for node selection (e.g., "avx2")')

    # Meta-evolution parameters
    parser.add_argument('--generations', type=int, default=20, help='Total number of generations (default: 20)')
    parser.add_argument('--population', type=int, default=2, help='Population size (default: 2)')
    parser.add_argument('--n-crossover', type=int, default=1, help='Number of crossover offspring per generation (default: 1)')
    parser.add_argument('--n-mutation', type=int, default=0, help='Number of mutation offspring per generation (default: 0)')

    # SR parameters
    parser.add_argument('--sr-population', type=int, default=100, help='SR population size (default: 100)')
    parser.add_argument('--sr-generations', type=int, default=1000, help='SR generations (default: 1000)')

    # Dataset
    parser.add_argument('--split', type=str, default='splits/split_train.txt',
                       help='Path to split file with dataset names (default: splits/split_train.txt)')
    parser.add_argument('--max-samples', type=int, default=1000, help='Max samples per dataset (default: 1000)')

    # Model
    parser.add_argument('--model', type=str, default='openai/gpt-5-mini',
                       help='LLM model to use (default: openai/gpt-5-mini)')
    parser.add_argument('--temperature', type=float, default=1.0, help='LLM sampling temperature')

    # Operator types to evolve (round-robin order)
    parser.add_argument('--operator-types', type=str, default='fitness,selection,mutation,crossover',
                       help='Comma-separated operator types to evolve in round-robin order')

    # Other options
    parser.add_argument('--no-trace-feedback', action='store_true', help='Disable trace feedback to LLM')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: results/run_TIMESTAMP)')
    parser.add_argument('--n-runs', type=int, default=1, help='Number of runs for meta-evolution (default: 1)')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed for reproducibility')

    args = parser.parse_args()
    assert args.n_crossover + args.n_mutation == args.population - 1, "n_crossover + n_mutation must equal population_size - 1"
    # Global seeding for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # LLM settings
    llm_temperature = args.temperature
    llm_seed = args.seed if (args.temperature == 0) else None

    # Parse operator types
    operator_types = [s.strip() for s in args.operator_types.split(',')]

    # Build SLURM config
    slurm_config = {
        'partition': args.partition,
        'time_limit': args.time_limit,
        'mem_per_cpu': args.mem_per_cpu,
        'dataset_max_samples': args.max_samples,
        'data_seed': args.seed,
        'max_retries': args.max_retries,
        'exclude_nodes': args.exclude_nodes,
        'constraint': args.constraint,
    }

    datasets = load_datasets_from_split(args.split, max_samples=args.max_samples, data_seed=args.seed)
    sr_kwargs={
        'population_size': args.sr_population,
        'n_generations': args.sr_generations,
        'verbose': False,
    }

    # Create output directory (timestamped if not specified)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/run_{timestamp}"

    final_bundle = None
    interrupted = False
    try:
        final_bundle = run_meta_evolution(
            n_generations=args.generations,
            population_size=args.population,
            n_crossover=args.n_crossover,
            n_mutation=args.n_mutation,
            sr_kwargs=sr_kwargs,
            model=args.model,
            output_dir=output_dir,
            slurm_config=slurm_config,
            datasets=datasets,
            operator_types=operator_types,
            use_trace_feedback=not args.no_trace_feedback,
            seed=args.seed,
            n_runs=args.n_runs,
            llm_temperature=llm_temperature,
            llm_seed=llm_seed,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nRun interrupted by user. Partial results have been saved.")

    print_usage()

    # Evaluate the final bundle on val tasks
    if not interrupted and final_bundle is not None:
        sr_kwargs['n_generations'] = 1000
        evaluate_final_bundle(
            final_bundle=final_bundle,
            output_dir=output_dir,
            sr_kwargs=sr_kwargs,
            seed=args.seed,
            slurm_config=slurm_config,
            split_file='splits/split_val.txt',
            n_runs=2,
        )
    else:
        print("Skipping SRBench evaluation due to interruption.")
