"""
Evaluation script comparing evolved operators vs baseline on train/val splits.
"""
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from sr import BasicSR
from meta_evolution import Operator, OperatorBundle, create_operator
from sr_operators import (
    default_selection_operator,
    default_mutation_operator,
    default_crossover_operator,
    default_fitness_function,
    mse,
)

PMLB_PATH = Path(__file__).parent / 'pmlb' / 'datasets'


def load_evolved_operators(results_path: str = "meta_evolution_results.json") -> OperatorBundle:
    """Load evolved operators from JSON results file."""
    with open(results_path, 'r') as f:
        results = json.load(f)

    operators = {}
    for op_type in ["fitness", "selection", "mutation", "crossover"]:
        code = results[op_type]["code"]
        op, passed, error = create_operator(code, op_type)
        if not passed:
            raise ValueError(f"Failed to load {op_type} operator: {error}")
        op.score = results[op_type]["score"]
        op.score_vector = results[op_type]["score_vector"]
        operators[op_type] = op

    return OperatorBundle(
        selection=operators["selection"],
        mutation=operators["mutation"],
        crossover=operators["crossover"],
        fitness=operators["fitness"],
    )


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load a single dataset, returning (X, y, formula)."""
    dataset_path = PMLB_PATH / dataset_name / f"{dataset_name}.tsv.gz"
    metadata_path = PMLB_PATH / dataset_name / "metadata.yaml"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path, sep='\t', compression='gzip')
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Extract formula from metadata
    formula = ""
    if metadata_path.exists():
        try:
            import yaml
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            if 'description' in metadata:
                desc = metadata['description']
                for line in desc.split('\n'):
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        if ' in [' not in line and ' in (' not in line:
                            formula = line
                            break
        except Exception:
            pass

    return X, y, formula


def evaluate_single_run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray,
    bundle: OperatorBundle,
    n_generations: int,
    population_size: int = 100,
) -> Dict:
    """Run a single SR experiment and return metrics."""

    sr = BasicSR(
        population_size=population_size,
        n_generations=n_generations,
        max_depth=10,
        max_size=40,
        selection_operator=bundle.selection,
        mutation_operator=bundle.mutation,
        crossover_operator=bundle.crossover,
        fitness_operator=bundle.fitness,
        constant_optimization=True,
        optimize_probability=0.01,
        verbose=False,
        save_trace=False,
    )

    start_time = time.time()
    sr.fit(X_train, y_train)
    fit_time = time.time() - start_time

    # Evaluate on full dataset
    y_pred = sr.predict(X_full)
    y_pred = np.clip(y_pred, -1e10, 1e10)

    # Compute metrics
    mse_full = float(np.mean((y_full - y_pred) ** 2))
    ss_res = np.sum((y_full - y_pred) ** 2)
    ss_tot = np.sum((y_full - np.mean(y_full)) ** 2)
    r2_full = float(1 - (ss_res / (ss_tot + 1e-10)))

    return {
        "mse": mse_full,
        "r2": r2_full,
        "time": fit_time,
        "expression": str(sr.best_model_),
        "size": sr.best_model_.size(),
    }


def run_evaluation(
    split_files: List[str],
    n_runs: int = 2,
    n_generations: int = 5000,
    train_samples: int = 1000,
    population_size: int = 100,
    results_path: str = "meta_evolution_results.json",
) -> Dict:
    """
    Run full evaluation comparing evolved vs baseline operators.

    Args:
        split_files: List of split files to evaluate on
        n_runs: Number of runs per dataset
        n_generations: Generations per SR run
        train_samples: Samples to use for training (subsample if larger)
        population_size: Population size for SR
        results_path: Path to evolved operators JSON

    Returns:
        Dictionary with all results
    """
    # Load operators
    print("Loading evolved operators...")
    evolved_bundle = load_evolved_operators(results_path)
    baseline_bundle = OperatorBundle.create_default()

    all_results = {
        "settings": {
            "n_runs": n_runs,
            "n_generations": n_generations,
            "train_samples": train_samples,
            "population_size": population_size,
        },
        "datasets": {},
    }

    # Collect all datasets from split files
    datasets_by_split = {}
    for split_file in split_files:
        split_name = Path(split_file).stem
        with open(split_file, 'r') as f:
            dataset_names = [line.strip() for line in f if line.strip()]
        datasets_by_split[split_name] = dataset_names

    print(f"\n{'='*70}")
    print("EVALUATION CONFIGURATION")
    print(f"{'='*70}")
    print(f"Generations per run: {n_generations}")
    print(f"Runs per dataset: {n_runs}")
    print(f"Training samples: {train_samples}")
    print(f"Evaluation: full dataset (all samples)")
    print(f"\nSplits:")
    for split_name, names in datasets_by_split.items():
        print(f"  {split_name}: {names}")

    # Run evaluation
    for split_name, dataset_names in datasets_by_split.items():
        print(f"\n{'='*70}")
        print(f"EVALUATING SPLIT: {split_name}")
        print(f"{'='*70}")

        for dataset_name in dataset_names:
            print(f"\n{'─'*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'─'*60}")

            try:
                X_full, y_full, formula = load_dataset(dataset_name)
            except Exception as e:
                print(f"  Error loading dataset: {e}")
                continue

            print(f"  Full dataset: {X_full.shape[0]} samples, {X_full.shape[1]} features")
            if formula:
                print(f"  Ground truth: {formula}")

            dataset_results = {
                "n_samples_full": X_full.shape[0],
                "n_features": X_full.shape[1],
                "formula": formula,
                "split": split_name,
                "evolved": [],
                "baseline": [],
            }

            for run_idx in range(n_runs):
                print(f"\n  Run {run_idx + 1}/{n_runs}:")

                # Subsample for training
                if X_full.shape[0] > train_samples:
                    train_idx = np.random.choice(X_full.shape[0], train_samples, replace=False)
                else:
                    train_idx = np.arange(X_full.shape[0])

                X_train = X_full[train_idx]
                y_train = y_full[train_idx]

                # Run evolved operators
                print(f"    Evolved operators...", end=" ", flush=True)
                try:
                    evolved_result = evaluate_single_run(
                        X_train, y_train, X_full, y_full,
                        evolved_bundle, n_generations, population_size
                    )
                    dataset_results["evolved"].append(evolved_result)
                    print(f"R²={evolved_result['r2']:.4f}, MSE={evolved_result['mse']:.4e}, "
                          f"Time={evolved_result['time']:.1f}s")
                except Exception as e:
                    print(f"ERROR: {e}")
                    dataset_results["evolved"].append({"error": str(e)})

                # Run baseline operators
                print(f"    Baseline operators...", end=" ", flush=True)
                try:
                    baseline_result = evaluate_single_run(
                        X_train, y_train, X_full, y_full,
                        baseline_bundle, n_generations, population_size
                    )
                    dataset_results["baseline"].append(baseline_result)
                    print(f"R²={baseline_result['r2']:.4f}, MSE={baseline_result['mse']:.4e}, "
                          f"Time={baseline_result['time']:.1f}s")
                except Exception as e:
                    print(f"ERROR: {e}")
                    dataset_results["baseline"].append({"error": str(e)})

            all_results["datasets"][dataset_name] = dataset_results

    return all_results


def print_summary(results: Dict):
    """Print a summary table of results."""
    print(f"\n{'='*90}")
    print("RESULTS SUMMARY")
    print(f"{'='*90}")

    # Organize by split
    splits = {}
    for dataset_name, data in results["datasets"].items():
        split = data.get("split", "unknown")
        if split not in splits:
            splits[split] = []
        splits[split].append((dataset_name, data))

    # Print header
    print(f"\n{'Dataset':<25} {'Evolved R²':>12} {'Baseline R²':>12} {'Δ R²':>10} {'Winner':>10}")
    print("-" * 75)

    all_evolved_r2 = []
    all_baseline_r2 = []

    for split_name in sorted(splits.keys()):
        print(f"\n[{split_name}]")

        for dataset_name, data in splits[split_name]:
            # Compute average R² across runs
            evolved_r2s = [r.get("r2", float('nan')) for r in data["evolved"] if "error" not in r]
            baseline_r2s = [r.get("r2", float('nan')) for r in data["baseline"] if "error" not in r]

            evolved_mean = np.mean(evolved_r2s) if evolved_r2s else float('nan')
            baseline_mean = np.mean(baseline_r2s) if baseline_r2s else float('nan')

            if not np.isnan(evolved_mean):
                all_evolved_r2.append(evolved_mean)
            if not np.isnan(baseline_mean):
                all_baseline_r2.append(baseline_mean)

            delta = evolved_mean - baseline_mean if not (np.isnan(evolved_mean) or np.isnan(baseline_mean)) else float('nan')

            if np.isnan(delta):
                winner = "N/A"
            elif delta > 0.01:
                winner = "Evolved"
            elif delta < -0.01:
                winner = "Baseline"
            else:
                winner = "Tie"

            print(f"{dataset_name:<25} {evolved_mean:>12.4f} {baseline_mean:>12.4f} {delta:>+10.4f} {winner:>10}")

    # Overall summary
    print(f"\n{'='*75}")
    print("OVERALL")
    print(f"{'='*75}")

    if all_evolved_r2 and all_baseline_r2:
        overall_evolved = np.mean(all_evolved_r2)
        overall_baseline = np.mean(all_baseline_r2)
        overall_delta = overall_evolved - overall_baseline

        print(f"Average Evolved R²:  {overall_evolved:.4f}")
        print(f"Average Baseline R²: {overall_baseline:.4f}")
        print(f"Improvement:         {overall_delta:+.4f} ({100*overall_delta/max(overall_baseline, 1e-10):+.1f}%)")

        wins = sum(1 for e, b in zip(all_evolved_r2, all_baseline_r2) if e > b + 0.01)
        losses = sum(1 for e, b in zip(all_evolved_r2, all_baseline_r2) if e < b - 0.01)
        ties = len(all_evolved_r2) - wins - losses

        print(f"\nWin/Loss/Tie: {wins}/{losses}/{ties}")


def main():
    """Run evaluation with default settings."""
    results = run_evaluation(
        split_files=["split_train_small.txt", "split_val_small.txt"],
        n_runs=2,
        n_generations=1000,
        train_samples=1000,
        population_size=100,
    )

    # Save detailed results
    with open("eval_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print_summary(results)

    print("\n\nDetailed results saved to eval_comparison_results.json")


if __name__ == "__main__":
    main()
