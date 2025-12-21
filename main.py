import numpy as np
import random
import hashlib
from typing import List, Dict, Tuple, Optional
import json
import sys
from datetime import datetime
from pathlib import Path
from sr import symbolic_regression
from meta_evolution import (
    Operator,
    OperatorBundle,
    OperatorException,
    create_and_test_operator,
    get_default_operator,
    semantics_aware_selection,
    generate_initial_operator,
    mutate_operator,
    crossover_operators,
)
from operator_templates import OPERATOR_TYPES
from completions import print_usage
from utils import load_datasets_from_split, load_srbench_dataset, load_datasets_from_list
from parallel_eval import SlurmEvaluator


class TeeLogger:
    """Logger that writes to both stdout and a file."""

    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', buffering=1)  # Line buffered

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


class RunLogger:
    """Tracks and saves all run data including operators and scores."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/run_{timestamp}"
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
    ):
        """Log data for a single generation."""
        stage_data = self.run_data["stages"][operator_type]

        gen_data = {
            "generation": generation,
            "population": [
                {
                    "score": op.score,
                    "score_vector": op.score_vector,
                    "loc": op.lines_of_code,
                    "code": op.code,
                }
                for op in population
            ],
            "best_score": best_operator.score,
            "avg_score": float(np.mean([op.score for op in population])),
            "avg_loc": float(np.mean([op.lines_of_code for op in population])),
        }

        stage_data["generations"].append(gen_data)
        stage_data["best_score_history"].append(best_operator.score)

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


def evaluate_operator_on_dataset(
    operator: Operator,
    frozen_bundle: OperatorBundle,
    operator_type: str,
    X,
    y,
    sr_kwargs,
    n_runs=3
) -> Tuple[float, List[str]]:
    """
    Evaluate an operator on a dataset, using frozen operators for other types.

    Returns:
        avg_score: Average R^2 score across multiple runs
        traces: List of trace strings from all runs
    """
    # Create bundle with the operator being evaluated
    bundle = frozen_bundle.copy_with(operator_type, operator)

    scores = []
    mses = []
    all_traces = []

    for run in range(n_runs):
        # Split into train/val
        n_samples = len(y)
        n_train = int(0.8 * n_samples)

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Run symbolic regression with full bundle
        best_ind, trace = symbolic_regression(
            X_train, y_train,
            selection_operator=bundle.selection,
            mutation_operator=bundle.mutation,
            crossover_operator=bundle.crossover,
            fitness_operator=bundle.fitness,
            **sr_kwargs,
        )
        all_traces.extend(trace)

        # Evaluate on validation set
        y_pred = best_ind.evaluate(X_val)
        y_pred = np.clip(y_pred, -1e10, 1e10)

        # Compute R^2 score
        ss_res = np.sum((y_val - y_pred) ** 2)
        mses.append(ss_res / len(y_val))
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))

        print(f"    Run {run+1}/{n_runs}: MSE={mses[-1]:.4e}, R^2={r2:.4f}, Formula={best_ind}")

        scores.append(r2)

    return np.mean(scores), all_traces


def evaluate_operator_on_all_datasets(
    operator: Operator,
    frozen_bundle: OperatorBundle,
    operator_type: str,
    datasets: Dict,
    sr_kwargs: Dict,
    seed: Optional[int] = None,
) -> Tuple[float, List[float], List[Dict]]:
    """
    Evaluate operator on all datasets

    Returns:
        average_score: Average R^2 across all datasets
        score_vector: R^2 score for each dataset
        trace_feedback: List of dicts with dataset name, ground truth, and traces
    """
    scores = []
    trace_feedback = []

    for dataset_name, (X, y, formula) in datasets.items():
        if seed is not None:
            # Deterministic per-dataset seed
            np.random.seed(seed)
            random.seed(seed)
        print(f"  Evaluating on {dataset_name}...")
        try:
            score, traces = evaluate_operator_on_dataset(
                operator, frozen_bundle, operator_type, X, y, sr_kwargs, n_runs=1
            )
            scores.append(score)
            trace_feedback.append({
                "dataset": dataset_name,
                "ground_truth": formula if formula else "Unknown",
                "traces": traces,
                "final_score": score,
            })
        except OperatorException as e:
            print(f"    Operator failed on dataset {dataset_name}, assigning score -1")
            print(f"    Error: {e}")
            score = -1
            scores.append(score)
            trace_feedback.append({
                "dataset": dataset_name,
                "ground_truth": formula if formula else "Unknown",
                "traces": [],
                "final_score": score,
                "error": str(e),
            })
        print(f"    Score: {score:.4f}")

    average_score = np.mean(scores)
    return average_score, scores, trace_feedback


def evaluate_population(
    population: List[Operator],
    label: str,
    operator_type: str,
    frozen_bundle: OperatorBundle,
    datasets: Dict,
    sr_kwargs: Dict,
    seed: Optional[int] = None,
    slurm_evaluator: Optional[SlurmEvaluator] = None,
) -> None:
    """
    Evaluate a population of operators using the configured method.

    Mutates operators in-place, setting their score, score_vector, and trace_feedback.

    Args:
        population: List of operators to evaluate
        label: Label for logging (e.g., "Initial Population")
        operator_type: Which operator type is being evaluated
        frozen_bundle: Bundle with frozen operators for other types
        datasets: Dictionary of datasets to evaluate on
        sr_kwargs: Arguments for symbolic regression algorithm
        seed: Base random seed for reproducibility
        slurm_evaluator: Optional SlurmEvaluator for multi-node parallelization
    """
    print(f"\n=== Evaluating {label} ({len(population)} operators) ===")

    if not population:
        return

    if slurm_evaluator:
        # SLURM job array evaluation using full bundle codes
        dataset_names = list(datasets.keys())
        # Build bundle codes per candidate
        bundles = []
        for i, op in enumerate(population):
            codes = {
                'selection': frozen_bundle.selection.code,
                'mutation': frozen_bundle.mutation.code,
                'crossover': frozen_bundle.crossover.code,
                'fitness': frozen_bundle.fitness.code,
            }
            codes[operator_type] = op.code
            bundles.append((i, codes))
        results = slurm_evaluator.evaluate_bundles(
            bundles=bundles,
            dataset_names=dataset_names,
            sr_kwargs=sr_kwargs,
            seed=seed if seed is not None else 42,
        )
        for i, op in enumerate(population):
            avg_score, score_vector, trace_feedback = results[i]
            op.score = avg_score
            op.score_vector = score_vector
            op.trace_feedback = trace_feedback
            print(f"Operator {i+1}:")
            for tf in trace_feedback:
                dataset = tf["dataset"]
                score = tf["final_score"]
                error = tf.get("error")
                if error:
                    print(f"    {dataset}: Score={score:.4f} (Error: {error})")
                else:
                    print(f"    {dataset}: Score={score:.4f}")
            print(f"  Avg Score = {avg_score:.4f}, LOC = {op.lines_of_code}")
    else:
        # Sequential evaluation (original behavior)
        for i, operator in enumerate(population):
            print(f"Evaluating operator {i+1}...")
            avg_score, score_vector, trace_feedback = evaluate_operator_on_all_datasets(
                operator, frozen_bundle, operator_type, datasets, sr_kwargs, seed=seed
            )
            operator.score = avg_score
            operator.score_vector = score_vector
            operator.trace_feedback = trace_feedback
            print(f"Operator {i+1}: Avg Score = {avg_score:.4f}, LOC = {operator.lines_of_code}")


def run_meta_evolution_stage(
    operator_type: str,
    frozen_bundle: OperatorBundle,
    n_generations: int,
    population_size: int,
    n_crossover: int,
    n_mutation: int,
    sr_kwargs: Dict,
    model: str,
    datasets: Dict,
    logger: Optional[RunLogger] = None,
    use_trace_feedback: bool = False,
    slurm_evaluator: Optional[SlurmEvaluator] = None,
    seed: Optional[int] = None,
    llm_temperature: float = 0.7,
    llm_seed: Optional[int] = None,
) -> Operator:
    """
    Run meta-evolution for a single operator type.

    Uses elitism: the best operator is always kept for the next generation.
    New population = elite (1) + crossover offspring (n_crossover) + mutation offspring (n_mutation)
    So n_crossover + n_mutation should equal population_size - 1.

    Args:
        operator_type: Which operator to evolve ("fitness", "selection", "mutation", "crossover")
        frozen_bundle: Bundle with frozen operators for other types
        n_generations: Number of meta-evolution generations for this stage
        population_size: Number of operators in population (only used for initialization)
        n_crossover: Number of offspring generated by crossover per generation
        n_mutation: Number of offspring generated by mutation per generation
        sr_kwargs: Arguments for symbolic regression algorithm
        model: Model to use for LLM calls
        datasets: Dictionary of datasets to evaluate on
        logger: Optional RunLogger for saving results
        use_trace_feedback: Whether to include SR traces in mutation/crossover prompts
        slurm_evaluator: Optional SlurmEvaluator for multi-node parallelization

    Returns:
        best_operator: The best operator found for this type
    """
    print(f"\n{'='*60}")
    print(f"STAGE: Evolving {operator_type.upper()} operator")
    print(f"{'='*60}")

    eval_mode = "SLURM" if slurm_evaluator else "sequential"
    print(f"Evaluation mode: {eval_mode}")

    if logger:
        logger.start_stage(operator_type)

    # Initialize population
    print(f"\n=== Initializing {operator_type} Population ===")
    population = []

    # Step 1: Generate all operators (LLM calls are sequential)
    for i in range(population_size):
        if i == 0:
            # Include default operator
            print(f"Adding default {operator_type} operator to population")
            operator = get_default_operator(operator_type)
        else:
            print(f"Generating initial {operator_type} operator {i+1}/{population_size}...")
            for j in range(10):
                code = generate_initial_operator(
                    operator_type=operator_type,
                    model=model,
                    llm_temperature=llm_temperature,
                    llm_seed=llm_seed,
                )
                operator, passed, error = create_and_test_operator(code, operator_type)
                if passed:
                    break
                else:
                    print(f"Generated operator failed tests, retrying... ({error})")
            else:
                raise ValueError(f"Failed to generate a valid initial {operator_type} operator after 10 attempts")

        print("*" * 10 + f" Generated {operator_type} operator " + "*" * 10)
        print(operator.code)
        print("*" * 40)
        population.append(operator)

    if len(population) == 0:
        print("Failed to generate initial population!")
        return frozen_bundle.get_operator(operator_type)

    # Step 2: Evaluate initial population
    evaluate_population(
        population=population,
        label="Initial Population",
        operator_type=operator_type,
        frozen_bundle=frozen_bundle,
        datasets=datasets,
        sr_kwargs=sr_kwargs,
        seed=seed,
        slurm_evaluator=slurm_evaluator,
    )

    # Meta-evolution loop
    best_operator = max(population, key=lambda op: op.score)
    best_history = [best_operator.score]

    # Log initial population (generation 0)
    if logger:
        logger.log_generation(operator_type, 0, population, best_operator)

    for gen in range(n_generations):
        print(f"\n=== {operator_type.capitalize()} Generation {gen+1}/{n_generations} ===")

        # Elitism: keep the best operator
        elite = max(population, key=lambda op: op.score)
        print(f"Elite (kept): score={elite.score:.4f}, LOC={elite.lines_of_code}")

        # Step 1: Generate all offspring (LLM calls are sequential)
        offspring = []

        # Crossover
        print(f"Generating {n_crossover} offspring via crossover...")
        for i in range(n_crossover):
            for j in range(10):
                parent_a, parent_b = semantics_aware_selection(population)

                code = crossover_operators(
                    parent_a, parent_b,
                    operator_type=operator_type,
                    model=model,
                    use_trace_feedback=use_trace_feedback,
                    llm_temperature=llm_temperature,
                    llm_seed=llm_seed,
                )
                operator, passed, error = create_and_test_operator(code, operator_type)
                if passed:
                    print(f"  Crossover {i+1}: Parent A score={parent_a.score:.4f}, Parent B score={parent_b.score:.4f}")
                    break
            else:
                raise ValueError(f"Failed to generate a valid {operator_type} offspring via crossover after 10 attempts")

            print("*" * 10 + f" Generated {operator_type} operator " + "*" * 10)
            print(operator.code)
            print("*" * 40)
            offspring.append(operator)

        # Mutation
        print(f"Generating {n_mutation} offspring via mutation...")
        for i in range(n_mutation):
            for j in range(10):
                code = mutate_operator(
                    elite,
                    operator_type=operator_type,
                    model=model,
                    use_trace_feedback=use_trace_feedback,
                    llm_temperature=llm_temperature,
                    llm_seed=llm_seed,
                )
                operator, passed, error = create_and_test_operator(code, operator_type)
                if passed:
                    break
            else:
                raise ValueError(f"Failed to generate a valid {operator_type} offspring via mutation after 10 attempts")

            print("*" * 10 + f" Generated {operator_type} operator " + "*" * 10)
            print(operator.code)
            print("*" * 40)
            offspring.append(operator)

        # Step 2: Evaluate all offspring
        evaluate_population(
            population=offspring,
            label=f"{len(offspring)} offspring",
            operator_type=operator_type,
            frozen_bundle=frozen_bundle,
            datasets=datasets,
            sr_kwargs=sr_kwargs,
            seed=seed,
            slurm_evaluator=slurm_evaluator,
        )

        population = [elite] + offspring

        # Track best
        current_best = max(population, key=lambda op: op.score)
        if current_best.score > best_operator.score:
            best_operator = current_best

        best_history.append(best_operator.score)

        print(f"\n{operator_type.capitalize()} Generation {gen+1} Summary:")
        print(f"  Best score: {best_operator.score:.4f}")
        print(f"  Best LOC: {best_operator.lines_of_code}")
        print(f"  Population size: {len(population)}")
        print(f"  Avg population score: {np.mean([op.score for op in population]):.4f}")
        print(f"  Avg population LOC: {np.mean([op.lines_of_code for op in population]):.1f}")

        # Log this generation
        if logger:
            logger.log_generation(operator_type, gen + 1, population, best_operator)

    print(f"\n=== {operator_type.capitalize()} Stage Complete ===")
    print(f"Best {operator_type} score: {best_operator.score:.4f}")
    print(f"Best {operator_type} LOC: {best_operator.lines_of_code}")
    print(f"\nBest {operator_type} operator code:")
    print("=" * 80)
    print(best_operator.code)
    print("=" * 80)

    # Log stage completion
    if logger:
        logger.log_stage_complete(operator_type, best_operator)

    return best_operator


def run_meta_evolution(
    generations_per_stage: int,
    population_size: int,
    n_crossover: int,
    n_mutation: int,
    sr_kwargs: Dict,
    model: str,
    datasets: Optional[Dict] = None,
    stage_order: List[str] = None,
    output_dir: Optional[str] = None,
    use_trace_feedback: bool = False,
    slurm_config: Optional[Dict] = None,
    seed: Optional[int] = None,
    llm_temperature: float = 0.7,
    llm_seed: Optional[int] = None,
) -> OperatorBundle:
    """
    Run full meta-evolution across all operator types in sequence.

    Args:
        generations_per_stage: Number of generations per operator type
        population_size: Number of operators in population per stage
        n_crossover: Number of crossover offspring per generation
        n_mutation: Number of mutation offspring per generation
        sr_kwargs: Arguments for symbolic regression algorithm
        model: Model to use for LLM calls
        datasets: Dictionary of datasets to evaluate on. If None, uses toy datasets.
                  Format: {name: (X, y, formula)}
        stage_order: Order of operator types to evolve (default: fitness, selection, mutation, crossover)
        output_dir: Directory to save all logs and results. If None, creates timestamped dir.
        use_trace_feedback: Whether to include SR evolution traces in mutation/crossover prompts.
                           When enabled, the LLM sees how expressions evolved and can learn
                           which operations helped reach the ground truth.
        slurm_config: Optional SLURM configuration for multi-node evaluation. Keys:
                     - partition: SLURM partition name
                     - time_limit: Time limit per task (HH:MM:SS)
                     - mem_per_cpu: Memory per CPU

    Returns:
        final_bundle: OperatorBundle with best operators for all types
    """
    if stage_order is None:
        stage_order = ["fitness", "selection", "mutation", "crossover"]

    # Create run logger for comprehensive logging
    logger = RunLogger(output_dir)

    # Create SLURM evaluator if configured
    slurm_evaluator = None
    if slurm_config:
        slurm_evaluator = SlurmEvaluator(
            results_dir=str(logger.output_dir),
            partition=slurm_config.get('partition', 'default_partition'),
            time_limit=slurm_config.get('time_limit', '01:00:00'),
            mem_per_cpu=slurm_config.get('mem_per_cpu', '4G'),
            dataset_max_samples=slurm_config.get('dataset_max_samples', None),
            data_seed=slurm_config.get('data_seed', 42),
        )

    # Save configuration
    config = {
        "generations_per_stage": generations_per_stage,
        "population_size": population_size,
        "n_crossover": n_crossover,
        "n_mutation": n_mutation,
        "sr_kwargs": sr_kwargs,
        "model": model,
        "stage_order": stage_order,
        "use_trace_feedback": use_trace_feedback,
        "slurm_config": slurm_config,
    }
    logger.set_config(config)

    if slurm_evaluator:
        print(f"SLURM mode enabled")

    # Load datasets if not provided (fallback to default split)
    if datasets is None:
        print("No datasets provided; loading default split 'splits/split_train_small.txt'.")
        datasets = load_datasets_from_split('splits/split_train_small.txt', max_samples=1000)
    print(f"Datasets: {list(datasets.keys())}")

    # Start with default bundle
    current_bundle = OperatorBundle.create_default()
    results = {}

    try:
        for operator_type in stage_order:
            print(f"\n{'#'*80}")
            print(f"# Starting stage: {operator_type.upper()}")
            print(f"{'#'*80}")

            best_operator = run_meta_evolution_stage(
                operator_type=operator_type,
                frozen_bundle=current_bundle,
                n_generations=generations_per_stage,
                population_size=population_size,
                n_crossover=n_crossover,
                n_mutation=n_mutation,
                sr_kwargs=sr_kwargs,
                model=model,
                datasets=datasets,
                logger=logger,
                use_trace_feedback=use_trace_feedback,
                slurm_evaluator=slurm_evaluator,
                seed=seed,
                llm_temperature=llm_temperature,
                llm_seed=llm_seed,
            )

            # Freeze this operator and update bundle
            current_bundle = current_bundle.copy_with(operator_type, best_operator)
            results[operator_type] = {
                "score": best_operator.score,
                "score_vector": best_operator.score_vector,
                "loc": best_operator.lines_of_code,
                "code": best_operator.code,
            }

        # Save final results
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)

        for operator_type in stage_order:
            print(f"\n{operator_type.upper()}:")
            print(f"  Score: {results[operator_type]['score']:.4f}")
            print(f"  LOC: {results[operator_type]['loc']}")

        logger.save_final_results(results)

    finally:
        # Always close the logger to restore stdout and save state
        logger.close()

    return current_bundle


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run LLM-Meta-SR: Meta-evolution of SR operators using LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (sequential)
  python main.py

  # Run with SLURM job arrays
  python main.py --slurm --partition gpu --time-limit 00:30:00

  # Custom output directory
  python main.py --output-dir results/my_experiment
        """
    )

    # SLURM mode
    parser.add_argument('--slurm', action='store_true',
                       help='Use SLURM job arrays for multi-node parallelization')
    parser.add_argument('--partition', type=str, default='default_partition',
                       help='SLURM partition (default: default_partition)')
    parser.add_argument('--time-limit', type=str, default='01:00:00',
                       help='SLURM time limit per task (default: 01:00:00)')
    parser.add_argument('--mem-per-cpu', type=str, default='4G',
                       help='SLURM memory per CPU (default: 4G)')

    # Meta-evolution parameters
    parser.add_argument('--generations', type=int, default=5, help='Number of generations per stage (default: 5)')
    parser.add_argument('--population', type=int, default=2, help='Population size (default: 2)')
    parser.add_argument('--n-crossover', type=int, default=1, help='Number of crossover offspring per generation (default: 1)')
    parser.add_argument('--n-mutation', type=int, default=0, help='Number of mutation offspring per generation (default: 0)')

    # SR parameters
    parser.add_argument('--sr-population', type=int, default=100, help='SR population size (default: 100)')
    parser.add_argument('--sr-generations', type=int, default=500, help='SR generations (default: 500)')

    # Dataset
    parser.add_argument('--split', type=str, default='splits/split_train_small.txt',
                       help='Path to split file with dataset names (default: splits/train_small.txt)')
    parser.add_argument('--max-samples', type=int, default=1000, help='Max samples per dataset (default: 1000)')

    # Model
    parser.add_argument('--model', type=str, default='openai/gpt-5-mini',
                       help='LLM model to use (default: openai/gpt-5-mini)')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM sampling temperature (0 for deterministic if supported)')

    # Stage order
    parser.add_argument('--stages', type=str, default='fitness,selection,mutation,crossover')

    # Other options
    parser.add_argument('--no-trace-feedback', action='store_true', help='Disable trace feedback to LLM')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: results/run_TIMESTAMP)')

    parser.add_argument('--seed', type=int, default=0, help='Base random seed for reproducibility')

    args = parser.parse_args()
    assert args.n_crossover + args.n_mutation == args.population - 1, "n_crossover + n_mutation must equal population_size - 1"
    # Global seeding for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # LLM settings
    llm_temperature = args.temperature
    llm_seed = args.seed if (args.temperature == 0) else None

    # Parse stage order
    stage_order = [s.strip() for s in args.stages.split(',')]

    # Build SLURM config if enabled
    slurm_config = None
    if args.slurm:
        slurm_config = {
            'partition': args.partition,
            'time_limit': args.time_limit,
            'mem_per_cpu': args.mem_per_cpu,
            'dataset_max_samples': args.max_samples,
            'data_seed': args.seed,
        }

    datasets = load_datasets_from_split(args.split, max_samples=args.max_samples, data_seed=args.seed)

    try:
        final_bundle = run_meta_evolution(
            generations_per_stage=args.generations,
            population_size=args.population,
            n_crossover=args.n_crossover,
            n_mutation=args.n_mutation,
            sr_kwargs={
                'population_size': args.sr_population,
                'n_generations': args.sr_generations,
                'verbose': False,
            },
            model=args.model,
            datasets=datasets,
            stage_order=stage_order,
            use_trace_feedback=not args.no_trace_feedback,
            output_dir=args.output_dir,
            slurm_config=slurm_config,
            seed=args.seed,
            llm_temperature=llm_temperature,
            llm_seed=llm_seed,
        )
    except KeyboardInterrupt:
        print("\n\nRun interrupted by user. Partial results have been saved.")

    print_usage()
