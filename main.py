"""
Main script for running LLM-Meta-SR on toy problems and SRBench datasets
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from toy_datasets import get_all_toy_datasets
from sr import symbolic_regression
from meta_evolution import (
    Operator,
    OperatorBundle,
    OperatorException,
    create_operator,
    get_default_operator,
    semantics_aware_selection,
    generate_initial_operator,
    mutate_operator,
    crossover_operators,
)
from operator_templates import OPERATOR_TYPES
from completions import print_usage
from utils import load_datasets_from_split, load_srbench_dataset, load_datasets_from_list


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
            output_dir = f"run_{timestamp}"
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
        gen_file = self.output_dir / f"{operator_type}_gen_{generation:03d}.json"
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
        with open(self.output_dir / f"best_{operator_type}_operator.py", "w") as f:
            f.write(best_operator.code)

        # Save stage summary
        stage_file = self.output_dir / f"{operator_type}_summary.json"
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

        # Copy to root for convenience
        with open("meta_evolution_results.json", "w") as f:
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
    sr_kwargs: Dict
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

    Returns:
        best_operator: The best operator found for this type
    """
    print(f"\n{'='*60}")
    print(f"STAGE: Evolving {operator_type.upper()} operator")
    print(f"{'='*60}")

    if logger:
        logger.start_stage(operator_type)

    # Initialize population
    print(f"\n=== Initializing {operator_type} Population ===")
    population = []

    for i in range(population_size):
        if i == 0:
            # Include default operator
            print(f"Adding default {operator_type} operator to population")
            operator = get_default_operator(operator_type)
        else:
            print(f"Generating initial {operator_type} operator {i+1}/{population_size}...")
            for j in range(10):
                code = generate_initial_operator(operator_type=operator_type, model=model)
                operator, passed, error = create_operator(code, operator_type)
                if passed:
                    break
                else:
                    print(f"Generated operator failed tests, retrying... ({error})")
            else:
                raise ValueError(f"Failed to generate a valid initial {operator_type} operator after 10 attempts")

        print("*" * 10 + f" Generated {operator_type} operator " + "*" * 10)
        print(operator.code)
        print("*" * 40)

        # Evaluate on datasets
        print(f"Evaluating operator {i+1}...")
        avg_score, score_vector, trace_feedback = evaluate_operator_on_all_datasets(
            operator, frozen_bundle, operator_type, datasets, sr_kwargs
        )

        operator.score = avg_score
        operator.score_vector = score_vector
        operator.trace_feedback = trace_feedback

        population.append(operator)
        print(f"Operator {i+1}: Avg Score = {avg_score:.4f}, LOC = {operator.lines_of_code}")

    if len(population) == 0:
        print("Failed to generate initial population!")
        return frozen_bundle.get_operator(operator_type)

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

        # Generate offspring to fill the rest of the population
        offspring = []

        # Crossover
        print(f"Generating {n_crossover} offspring via crossover...")
        for i in range(n_crossover):
            for j in range(10):
                parent_a, parent_b = semantics_aware_selection(population)
                print(f"  Crossover {j+1}: Parent A score={parent_a.score:.4f}, Parent B score={parent_b.score:.4f}")

                code = crossover_operators(
                    parent_a, parent_b,
                    operator_type=operator_type,
                    model=model,
                    use_trace_feedback=use_trace_feedback,
                )
                operator, passed, error = create_operator(code, operator_type)
                if passed:
                    break
            else:
                raise ValueError(f"Failed to generate a valid {operator_type} offspring via crossover after 10 attempts")

            print("*" * 10 + f" Generated {operator_type} operator " + "*" * 10)
            print(operator.code)
            print("*" * 40)

            # Evaluate
            avg_score, score_vector, trace_feedback = evaluate_operator_on_all_datasets(
                operator, frozen_bundle, operator_type, datasets, sr_kwargs
            )
            operator.score = avg_score
            operator.score_vector = score_vector
            operator.trace_feedback = trace_feedback

            offspring.append(operator)
            print(f"  Offspring {i+1}: Score = {avg_score:.4f}, LOC = {operator.lines_of_code}")

        # Mutation
        print(f"Generating {n_mutation} offspring via mutation...")

        for i in range(n_mutation):
            for j in range(10):
                code = mutate_operator(
                    elite,
                    operator_type=operator_type,
                    model=model,
                    use_trace_feedback=use_trace_feedback,
                )
                operator, passed, error = create_operator(code, operator_type)
                if passed:
                    break
            else:
                raise ValueError(f"Failed to generate a valid {operator_type} offspring via mutation after 10 attempts")

            print("*" * 10 + f" Generated {operator_type} operator " + "*" * 10)
            print(operator.code)
            print("*" * 40)

            # Evaluate
            avg_score, score_vector, trace_feedback = evaluate_operator_on_all_datasets(
                operator, frozen_bundle, operator_type, datasets, sr_kwargs
            )
            operator.score = avg_score
            operator.score_vector = score_vector
            operator.trace_feedback = trace_feedback

            offspring.append(operator)
            print(f"  Mutant {i+1}: Score = {avg_score:.4f}, LOC = {operator.lines_of_code}")

        # New population = elite + offspring
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


def run_full_meta_evolution(
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

    Returns:
        final_bundle: OperatorBundle with best operators for all types
    """
    if stage_order is None:
        stage_order = ["fitness", "selection", "mutation", "crossover"]

    # Create run logger for comprehensive logging
    logger = RunLogger(output_dir)

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
    }
    logger.set_config(config)

    # Load datasets if not provided
    if datasets is None:
        datasets = get_all_toy_datasets()
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
        print(f"\nResults saved to {logger.output_dir}/")

    finally:
        # Always close the logger to restore stdout and save state
        logger.close()

    return current_bundle


# Backwards compatibility: single-stage evolution for selection only
def run_meta_evolution(
    n_generations,
    population_size,
    n_crossover,
    n_mutation,
    sr_kwargs,
    model,
    datasets: Optional[Dict] = None,
):
    """
    Run meta-evolution for selection operator only (backwards compatible).

    Args:
        datasets: Dictionary of datasets to evaluate on. If None, uses toy datasets.
                  Format: {name: (X, y, formula)}
    """
    if datasets is None:
        datasets = get_all_toy_datasets()
    frozen_bundle = OperatorBundle.create_default()

    best_operator = run_meta_evolution_stage(
        operator_type="selection",
        frozen_bundle=frozen_bundle,
        n_generations=n_generations,
        population_size=population_size,
        n_crossover=n_crossover,
        n_mutation=n_mutation,
        sr_kwargs=sr_kwargs,
        model=model,
        datasets=datasets,
    )

    # Save results (backwards compatible)
    with open("best_operator.py", "w") as f:
        f.write(best_operator.code)

    with open("best_history.json", "w") as f:
        json.dump({
            "final_score": best_operator.score,
            "final_score_vector": best_operator.score_vector,
            "final_loc": best_operator.lines_of_code
        }, f, indent=2)

    return best_operator, [best_operator.score]


if __name__ == "__main__":
    # Load the 4-problem small train split with 1000 samples each
    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    datasets = load_datasets_from_split('split_train_small.txt', max_samples=1000)
    print(f"\nLoaded {len(datasets)} datasets: {list(datasets.keys())}")

    try:
        final_bundle = run_full_meta_evolution(
            generations_per_stage=5,
            population_size=2,
            n_crossover=1,
            n_mutation=0,
            sr_kwargs={
                'population_size': 100,
                'n_generations': 10,
                'verbose': False,
            },
            model="openai/gpt-5-mini",
            datasets=datasets,
            # stage_order=["fitness", "selection", "mutation", "crossover"],
            stage_order=["mutation", "crossover"],
            # Enable trace feedback to show the LLM how SR evolved expressions
            # toward ground truth, so it can design better operators
            use_trace_feedback=True,
        )
    except KeyboardInterrupt:
        print("\n\nRun interrupted by user. Partial results have been saved.")

    print_usage()
