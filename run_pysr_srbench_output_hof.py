import argparse
import os
import numpy as np
import pandas as pd
import json
import time
import shutil
from pathlib import Path
from pysr import PySRRegressor

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

    # Rename reserved sympy names
    for i, val in enumerate(feature_names):
        if val == 'gamma': feature_names[i] = 'x0'
        elif val == 'beta': feature_names[i] = 'x1'
        elif val == 'E': feature_names[i] = 'x2'
        elif val == 'I': feature_names[i] = 'x3'

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
    milestones,
    seed=42,
    max_size=40,
    n_cpus=1,
    max_evals=1000000,
):
    os.makedirs(results_dir, exist_ok=True)
    hof_path = os.path.join(results_dir, f"{dataset_name}_hof.csv")
    
    # Logic change: Only define a specific temp directory if we are doing milestone logging
    if milestones:
        pysr_output_dir = os.path.join(results_dir, f"pysr_tmp_{dataset_name}")
    else:
        pysr_output_dir = None  # PySR will use a standard system temp location and self-clean

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

    print(f"\n{'='*60}\n PySR Run: {dataset_name}\n{'='*60}")

    try:
        if not milestones:
            print(f" >>> Running single fit (n=0, no trace logging) to {max_evals} evals...")
            model.max_evals = max_evals
            model.fit(X_train, y_train, variable_names=feature_names)
        else:
            for milestone in milestones:
                start_chunk = time.time()
                print(f" >>> Running to {milestone:,} evals...")
                
                model.max_evals = milestone
                model.fit(X_train, y_train, variable_names=feature_names)
                
                chunk_time = time.time() - start_chunk

                if model.equations_ is not None:
                    df = model.equations_.copy()
                    df.insert(0, "milestone_evals", milestone)
                    df.insert(1, "chunk_runtime", round(chunk_time, 2))
                    
                    file_exists = os.path.isfile(hof_path)
                    with open(hof_path, 'a') as f:
                        if file_exists:
                            f.write(f"\n# --- MILESTONE: {milestone} ---\n")
                        df.to_csv(f, mode='a', index=False, header=not file_exists)
    finally:
        # Only attempt deletion if we explicitly created a directory
        if pysr_output_dir and os.path.exists(pysr_output_dir):
            print(f"Cleaning up temporary directory: {pysr_output_dir}")
            shutil.rmtree(pysr_output_dir)
            
    return model

def run_pysr_on_dataset(
    dataset_name,
    max_samples=1000,
    results_dir='results_pysr',
    seed=42,
    max_size=40,
    verbose=True,
    max_evals=1_000_000,
    target_noise=0.0,
    n_steps=3,
):
    start_time = time.time()

    # If n_steps is 0, milestones list is empty
    if n_steps > 0:
        milestones = [int(round(max_evals * (i + 1) / n_steps)) for i in range(n_steps)]
    else:
        milestones = []

    if verbose:
        print(f"=" * 60)
        print(f"Running PySR on: {dataset_name}")
        print(f"Total Max Evals: {max_evals}")
        if milestones:
            print(f"Milestones: {milestones}")
        else:
            print("Mode: Single run (no HOF trace logging)")
        print(f"=" * 60)

    X, y, feature_names, metadata = load_dataset(dataset_name, max_samples=max_samples, seed=seed)

    n_samples = len(y)
    n_train = int(0.75 * n_samples)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_test, y_test = X[indices[n_train:]], y[indices[n_train:]]

    if target_noise > 0:
        y_train = add_noise(y_train, target_noise, seed=seed + 1000)

    try:
        # Standard SLURM CPU detection
        n_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
    except (TypeError, ValueError):
        n_cpus = 1

    model = run_pysr_with_hof_checkpoints(
        X_train, y_train,
        feature_names=feature_names,
        dataset_name=dataset_name,
        results_dir=results_dir,
        milestones=milestones,
        seed=seed,
        max_size=max_size,
        n_cpus=n_cpus,
        max_evals=max_evals
    )

    fit_time = time.time() - start_time
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / (np.sum((y_test - np.mean(y_test)) ** 2) + 1e-10))

    results = {
        'dataset': dataset_name,
        'test_mse': float(mse),
        'test_r2': float(r2),
        'fit_time_seconds': fit_time,
        'best_equation': str(model.get_best()['equation']),
        'milestones': milestones,
    }

    results_file = os.path.join(results_dir, f"{dataset_name}_s{seed}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results

def main():
    parser = argparse.ArgumentParser(
        description='Run PySR on SRBench datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str, help='Single dataset name')
    group.add_argument('--split_file', type=str, help='Path to split file')

    parser.add_argument('--array_index', type=int, default=None)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target_noise', type=float, default=0.0)
    parser.add_argument('--max_evals', type=int, default=1000000, help='Total evals to reach')
    parser.add_argument('--results_dir', type=str, default='results_pysr')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--n', type=int, default=3, 
                        help='Number of checkpoints. If 0, no execution trace hof file is saved.')

    args = parser.parse_args()
    verbose = not args.quiet

    if args.dataset:
        datasets = [args.dataset]
    else:
        full_list = load_split_file(args.split_file)
        idx = args.array_index if args.array_index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        datasets = [full_list[idx]] if idx < len(full_list) else []

    for dataset_name in datasets:
        try:
            run_pysr_on_dataset(
                dataset_name=dataset_name,
                max_samples=args.max_samples,
                results_dir=args.results_dir,
                seed=args.seed,
                verbose=verbose,
                target_noise=args.target_noise,
                max_evals=args.max_evals,
                n_steps=args.n
            )
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

if __name__ == '__main__':
    main()