"""
Run PySR on SRBench datasets using the same data loading approach as meta SR.
 
Usage:
    python run_pysr_srbench_output_hof.py --dataset feynman_I_15_10 --max_evals 100000
    python run_pysr_srbench_output_hof.py --split_file splits/train.txt --max_evals 100000
    python run_pysr_srbench_output_hof.py --split_file splits/train.txt --array_index 5
"""
 
import argparse
import os
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from pysr import PySRRegressor
from evaluation import check_symbolic_match, parse_expr_str_to_sympy, parse_ground_truth
 
 
def add_noise(data, noise_level, seed=None):
    """Add Gaussian noise scaled by RMS (SRBench method)."""
    if noise_level <= 0:
        return data
    if seed is not None:
        np.random.seed(seed)
    rms = np.sqrt(np.mean(np.square(data)))
    return data + np.random.normal(0, noise_level * rms, size=data.shape)
 
 
def load_dataset(dataset_name, pmlb_path=None, max_samples=None, seed=42):
    """Load a dataset from the PMLB directory structure."""
    if pmlb_path is None:
        pmlb_path = Path(__file__).parent / 'pmlb' / 'datasets'
    else:
        pmlb_path = Path(pmlb_path)
 
    dataset_path = pmlb_path / dataset_name / f"{dataset_name}.tsv.gz"
    metadata_path = pmlb_path / dataset_name / "metadata.yaml"
 
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
 
    df = pd.read_csv(dataset_path, sep='\t', compression='gzip')
 
    feature_names = [col for col in df.columns if col != 'target']
    X = df[feature_names].values
 
    # Rename reserved sympy names so they don't confuse the parser
    for i, val in enumerate(feature_names):
        if val == 'gamma':
            feature_names[i] = 'x0'
        elif val == 'beta':
            feature_names[i] = 'x1'
        elif val == 'E':
            feature_names[i] = 'x2'
        elif val == 'I':
            feature_names[i] = 'x3'
 
    y = df['target'].values
 
    metadata = {'dataset_name': dataset_name}
    if metadata_path.exists():
        try:
            import yaml
            with open(metadata_path, 'r') as f:
                metadata.update(yaml.safe_load(f))
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
 
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]
 
    return X, y, feature_names, metadata
 
 
def load_split_file(split_file):
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def run_pysr_with_hof_checkpoints(
    X_train, y_train,
    feature_names,
    dataset_name,
    results_dir,
    checkpoints,
    seed=42,
    max_size=40,
    n_cpus=1,
):
    os.makedirs(results_dir, exist_ok=True)
    hof_path = os.path.join(results_dir, f"{dataset_name}_hof.csv")
    pysr_output_dir = os.path.join(results_dir, f"pysr_tmp_{dataset_name}")
    
    milestone_set = sorted([int(float(c)) for c in checkpoints])

    model = PySRRegressor(
        niterations=10_000_000, 
        warm_start=True,        
        verbosity=0, 
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "square"],
        maxsize=max_size,
        procs=n_cpus,
        random_state=seed,
        output_directory=pysr_output_dir,
    )

    # Header for the console
    print(f"\n{'='*60}\n PySR Run: {dataset_name}\n Output: {hof_path}\n{'='*60}")

    for i, milestone in enumerate(milestone_set):
        start_chunk = time.time()
        print(f" >>> Running to {milestone:,} evals...")
        
        model.max_evals = milestone
        model.fit(X_train, y_train, variable_names=feature_names)
        
        chunk_time = time.time() - start_chunk

        if model.equations_ is not None:
            # --- CONSOLE VIEW ---
            print(f"\n Snapshot at {milestone:,} evals:")
            view_cols = ['complexity', 'loss', 'equation']
            print(model.equations_[view_cols].to_string(index=False))
            print(f"{'-'*40}\n")

            # --- FILE OUTPUT ---
            df = model.equations_.copy()
            
            # Organize columns: Milestone and Time first
            df.insert(0, "milestone_evals", milestone)
            df.insert(1, "chunk_runtime", round(chunk_time, 2))
            
            file_exists = os.path.isfile(hof_path)
            with open(hof_path, 'a') as f:
                # 1. Add a visual separator if not the first milestone
                if file_exists:
                    f.write(f"\n# --- MILESTONE: {milestone} ---\n")
                
                # 2. Append the actual data
                # header=not file_exists ensures the CSV header only appears at the very top
                df.to_csv(f, mode='a', index=False, header=not file_exists)
            
    return model
 
def run_pysr_on_dataset(
    dataset_name,
    max_samples=10000,
    results_dir='results_pysr',
    seed=42,
    max_size=40,
    binary_operators=None,
    unary_operators=None,
    verbose=True,
    max_evals=None,
    target_noise=0.0,
    checkpoints=None,
):
    """Run PySR on a single dataset."""
    start_time = time.time()
 
    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/", "^"]
 
    if unary_operators is None:
        unary_operators = []
 
    # CHANGED: checkpoints are now cumulative max_evals targets (not iteration counts).
    # The model runs with warm_start=True and fit() is called once per checkpoint,
    # each time with max_evals raised to the next cumulative target. The HOF is
    # snapshotted and printed to stdout after each fit() call.
    if checkpoints is None:
        checkpoints = [300_000, 600_000, 900_000]
 
    if verbose:
        print(f"=" * 60)
        print(f"Running PySR on: {dataset_name}")
        print(f"Checkpoints: {checkpoints}")
        print(f"=" * 60)
 
    # Load data
    X, y, feature_names, metadata = load_dataset(
        dataset_name,
        max_samples=max_samples,
        seed=seed
    )
 
    if verbose:
        print(f"Data shape: X={X.shape}, y range=[{y.min():.3e}, {y.max():.3e}]")
        print(f"Features: {feature_names}")
        if 'description' in metadata:
            print(f"Description: {metadata.get('description', 'N/A')}...")
 
    # Split into train/test (75/25)
    n_samples = len(y)
    n_train = int(0.75 * n_samples)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
 
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
 
    # Apply noise to training target only (SRBench approach)
    if target_noise > 0:
        noise_seed = seed + 1000
        y_train = add_noise(y_train, target_noise, seed=noise_seed)
        if verbose:
            print(f"Applied target noise: {target_noise} (seed={noise_seed})")
 
    if verbose:
        print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
        print(f"\nStarting PySR fit with checkpoints {checkpoints}...")
 
    try:
        n_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES'))
    except (TypeError, ValueError):
        n_cpus = 1
 
    model = run_pysr_with_hof_checkpoints(
        X_train, y_train,
        feature_names=feature_names,
        dataset_name=dataset_name,
        results_dir=results_dir,
        checkpoints=checkpoints,
        seed=seed,
        max_size=max_size,
        n_cpus=n_cpus,
    )
 
    fit_time = time.time() - start_time
 
    if verbose:
        print(f"\nFit completed in {fit_time:.1f} seconds")
 
    # Evaluate on test set
    y_pred = model.predict(X_test)
 
    mse = np.mean((y_test - y_pred) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
 
    best_eq = model.get_best()
    best_equation_str = str(best_eq['equation'])
 
    results = {
        'dataset': dataset_name,
        'test_mse': float(mse),
        'test_r2': float(r2),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'n_features': X.shape[1],
        'fit_time_seconds': fit_time,
        'best_equation': best_equation_str,
        'seed': seed,
        'max_size': max_size,
        'target_noise': target_noise,
        'checkpoints': checkpoints,
    }
 
    if verbose:
        print(f"\nResults:")
        print(f"  Test MSE: {mse:.4e}")
        print(f"  Test R²:  {r2:.4f}")
        print(f"  Best equation: {best_equation_str}")
 
    if target_noise > 0:
        results_file = os.path.join(results_dir, f"{dataset_name}_n{max_samples}_noise{target_noise}_seed{seed}.json")
    else:
        results_file = os.path.join(results_dir, f"{dataset_name}_n{max_samples}_seed{seed}.json")
 
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
 
    if verbose:
        print(f"\nResults saved to: {results_file}")
 
    return results
 
 
def main():
    parser = argparse.ArgumentParser(
        description='Run PySR on SRBench datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
 
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str,
                       help='Single dataset name to run (e.g., feynman_I_15_10)')
    group.add_argument('--split_file', type=str,
                       help='Path to split file with dataset names')
 
    parser.add_argument('--array_index', type=int, default=None,
                       help='SLURM array task index (0-based).')
 
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples to use from each dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--target_noise', type=float, default=0.0,
                       help='Gaussian noise level for target')
 
    parser.add_argument('--max_evals', type=int, default=int(1e6),
                       help='Maximum evaluations (informational only)')
 
    parser.add_argument('--results_dir', type=str, default='results_pysr',
                       help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
 
    parser.add_argument(
        '--checkpoints',
        type=lambda s: [int(float(x)) for x in s.split(',')],
        # CHANGED: Default was '3,300000,600000,900000'. The '3' caused all HOF
        # files to be missing. Now checkpoints are cumulative max_evals targets —
        # fit() is called once per checkpoint with warm_start=True so the search
        # is continuous. HOF is printed to stdout and saved to CSV after each call.
        default='300000,600000,900000',
        help=(
            'Comma-separated CUMULATIVE max_evals targets at which to snapshot the HOF. '
            'fit() is called once per checkpoint; warm_start=True keeps population state. '
            'Example: "300000,600000,900000" or "3e5,6e5,9e5"'
        )
    )
 
    args = parser.parse_args()
 
    print("Executing command: " + " ".join(os.sys.argv))
    verbose = not args.quiet
 
    if args.dataset:
        datasets = [args.dataset]
        run_index = 0
    else:
        datasets = load_split_file(args.split_file)
 
        if args.array_index is not None:
            run_index = args.array_index
        else:
            run_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
 
        if run_index >= len(datasets):
            print(f"Array index {run_index} is out of range (only {len(datasets)} datasets)")
            return
 
        datasets = [datasets[run_index]]
 
    for dataset_name in datasets:
        try:
            results = run_pysr_on_dataset(
                dataset_name=dataset_name,
                max_samples=args.max_samples,
                results_dir=args.results_dir,
                seed=args.seed,
                verbose=verbose,
                target_noise=args.target_noise,
                checkpoints=args.checkpoints,
            )
 
            if verbose:
                print(f"\nCompleted: {dataset_name}")
                print(f"R² = {results['test_r2']:.4f}")
 
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
 
 
if __name__ == '__main__':
    main()