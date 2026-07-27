#!/usr/bin/env python3
"""
Run PySR on SRBench datasets using the same data loading approach as meta SR.

This script loads data directly from the pmlb/datasets directory structure,
avoiding the SRBench harness that was causing issues.

Usage:
    # Run on a single dataset
    python run_pysr_srbench.py --dataset feynman_I_15_10 --max-evals 100000

    # Run on all datasets in a split file
    python run_pysr_srbench.py --split-file splits/train.txt --max-evals 1000000

    # Run with specific SLURM array task
    python run_pysr_srbench.py --split-file splits/train.txt --array-index 5 --max-evals 1000000
"""

import argparse
import os
import shutil
import sys
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from julia_env import configure_juliapkg_project
configure_juliapkg_project(_REPO_ROOT)

from pysr import PySRRegressor  # noqa: E402  (must come after juliapkg setup)


CONFLICTING_VAR_NAMES = {"gamma", "beta", "E", "I"}


def _build_rename_map(feature_names):
    """Return {original: renamed} for any feature names that conflict with
    Julia/SR builtins (E, I) or special functions (beta, gamma)."""
    rename_map = {}
    for i, name in enumerate(feature_names):
        if name in CONFLICTING_VAR_NAMES:
            rename_map[name] = f"var_{i}"
    return rename_map


def add_noise(data, noise_level, seed=None):
    """
    Add Gaussian noise scaled by RMS (SRBench method).

    This matches SRBench's implementation in experiment/evaluate_model.py:130-143.
    Noise is scaled by the RMS of the data: noise_level * sqrt(mean(x²))

    Args:
        data: Array to add noise to
        noise_level: Noise level (e.g., 0.001, 0.01, 0.1)
        seed: Random seed for reproducibility

    Returns:
        Data with added noise
    """
    if noise_level <= 0:
        return data
    if seed is not None:
        np.random.seed(seed)
    rms = np.sqrt(np.mean(np.square(data)))
    return data + np.random.normal(0, noise_level * rms, size=data.shape)


def load_dataset(dataset_name, pmlb_path=None, max_samples=None, seed=42):
    """
    Load a dataset from the PMLB directory structure.

    Args:
        dataset_name: Name of the dataset (e.g., 'feynman_I_15_10')
        pmlb_path: Path to pmlb/datasets directory. If None, uses default.
        max_samples: Maximum number of samples to use (for faster runs)
        seed: Random seed for subsampling

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        feature_names: List of feature names
        metadata: Dict with dataset info
    """
    from utils import resolve_pmlb_paths

    dataset_path, metadata_path = resolve_pmlb_paths(dataset_name, pmlb_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load data
    df = pd.read_csv(dataset_path, sep='\t', compression='gzip')

    # Extract features and target
    original_feature_names = [col for col in df.columns if col != 'target']
    X = df[original_feature_names].values

    # Some feature names collide with Julia builtins (E, I) or SpecialFunctions
    # (beta, gamma) and crash PySR. Rename them and remember the mapping so
    # downstream symbolic-match checking can substitute back.
    rename_map = _build_rename_map(original_feature_names)
    feature_names = [rename_map.get(n, n) for n in original_feature_names]

    y = df['target'].values

    # Load metadata if available
    metadata = {
        'dataset_name': dataset_name,
        'original_feature_names': original_feature_names,
        'rename_map': rename_map,
    }
    ground_truth_formula = ""
    if metadata_path.exists():
        try:
            import yaml
            with open(metadata_path, 'r') as f:
                metadata.update(yaml.safe_load(f))
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")

    # Extract GT formula from description (mirrors utils.load_srbench_dataset).
    desc = metadata.get('description', '') or ''
    for line in desc.split('\n'):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            if ' in [' not in line and ' in (' not in line:
                from utils import rhs_only
                ground_truth_formula = rhs_only(line)
                break
    metadata['ground_truth_formula'] = ground_truth_formula

    # Subsample if needed
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y, feature_names, metadata


def load_split_file(split_file):
    """Load dataset names from a split file."""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_cpu_count():
    """Get available CPU count, handling SLURM environment."""
    try:
        cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', 0))
        nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        if cpus > 0:
            return cpus * nodes
    except (TypeError, ValueError):
        pass

    # Fall back to os.cpu_count()
    return os.cpu_count() or 4


def run_pysr_with_hof_checkpoints(
    X_train, y_train,
    feature_names,
    dataset_name,
    results_dir,
    milestones,
    model,
    seed=42,
    hof_path=None,
    milestone_kind="evals",
):
    """
    Run PySR with HOF checkpoint logging at each milestone.
    Writes a fresh hall-of-fame CSV with one block per milestone. If milestones
    is empty, runs a single fit with no HOF trace logging.

    milestone_kind:
      "evals" -> `milestones` are cumulative max_evals targets; each chunk fits
                 with warm_start up to that eval count (the eval-budget regime).
      "time"  -> `milestones` are cumulative wall-clock targets (seconds); each
                 chunk fits with warm_start for the time slice since the previous
                 milestone via `timeout_in_seconds` (the time-budget regime).
    The `milestone_evals` CSV column holds the cumulative eval target (evals mode)
    or the cumulative time budget in seconds (time mode).
    """
    os.makedirs(results_dir, exist_ok=True)
    hof_path = hof_path or os.path.join(results_dir, f"{dataset_name}_hof.csv")

    # Only define a specific temp directory if we are doing milestone logging
    if milestones:
        pysr_output_dir = os.path.join(results_dir, f"pysr_tmp_{dataset_name}")
        model.output_directory = pysr_output_dir
        model.warm_start = True
        if os.path.exists(hof_path):
            os.remove(hof_path)

    try:
        if not milestones:
            model.fit(X_train, y_train, variable_names=feature_names)
        else:
            prev_milestone = 0
            for milestone in milestones:
                start_chunk = time.time()

                if milestone_kind == "time":
                    # Per-chunk wall-time slice; timeout_in_seconds resets each
                    # warm-started fit, so pass the increment, not the cumulative.
                    model.timeout_in_seconds = max(1, int(round(milestone - prev_milestone)))
                else:
                    model.max_evals = milestone
                model.fit(X_train, y_train, variable_names=feature_names)

                chunk_time = time.time() - start_chunk
                prev_milestone = milestone

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
        if milestones and model.output_directory and os.path.exists(model.output_directory):
            print(f"Cleaning up temporary directory: {model.output_directory}")
            shutil.rmtree(model.output_directory)

    return model


def run_pysr_on_dataset(
    dataset_name,
    max_samples=10000,
    results_dir='results_pysr',
    seed=42,
    max_size=40,
    # niterations=None,
    binary_operators=None,
    unary_operators=None,
    verbose=True,
    max_evals=None,
    target_noise=0.0,
    hof_n=3,
    evolve_results=None,
):
    """
    Run PySR on a single dataset.

    Args:
        dataset_name: Name of the dataset
        time_minutes: Time limit in minutes
        max_samples: Max samples to use from dataset
        results_dir: Directory to save results
        seed: Random seed
        n_cpus: Number of CPUs to use (None for auto-detect)
        max_size: Maximum expression size
        niterations: Max iterations (usually time is the limit)
        binary_operators: List of binary operators to use
        unary_operators: List of unary operators to use
        verbose: Print progress
        max_evals: Total number of evaluations to reach
        target_noise: Gaussian noise level for target
        hof_n: Number of HOF checkpoints. If 0, no HOF trace logging.

    Returns:
        Dict with results
    """
    start_time = time.time()

    # If hof_n is 0, milestones list is empty
    if hof_n > 0 and max_evals is not None:
        milestones = [int(round(max_evals * (i + 1) / hof_n)) for i in range(hof_n)]
    else:
        milestones = []

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/", "^"]

    if unary_operators is None:
        unary_operators = []  # Match SRBench default

    if verbose:
        print(f"=" * 60)
        print(f"Running PySR on: {dataset_name}")
        if milestones:
            print(f"Milestones: {milestones}")
        print(f"=" * 60)

    # Resolve evolve method (custom mutation/survival/selection/loss + any
    # bundle-tuned PySR hparams). The dynamic Julia code is loaded just before
    # PySRRegressor construction below.
    evolve_weights = {}
    evolve_pysr_overrides = {}
    evolve_extra_code = {}
    if evolve_results is not None:
        from bundle_loader import load_bundle
        bundle = load_bundle(evolve_results)
        # Convert to a PySRConfig with no base pysr_kwargs, so pysr_kwargs holds
        # only the bundle's HPO-tuned overrides (layered onto the model kwargs
        # below) and mutation_weights/custom_*_code carry the operators.
        cfg = bundle.to_pysr_config({})
        evolve_weights = cfg.mutation_weights
        evolve_pysr_overrides = cfg.pysr_kwargs
        evolve_extra_code = {
            "custom_mutation_code": cfg.custom_mutation_code,
            "custom_survival_code": cfg.custom_survival_code,
            "custom_selection_code": cfg.custom_selection_code,
            "custom_loss_code": cfg.custom_loss_code,
        }
        if verbose:
            # Look up whether this dataset was solved during the evolve run.
            # Solved = at least one seed scored gt_match >= 1.0
            # (matches get_solved_tasks in evolution_helpers.py).
            solve_status = "UNKNOWN"
            details = bundle.result_details or []
            for detail in details:
                if detail.get("dataset") == dataset_name:
                    run_gt = detail.get("run_gt_scores") or []
                    solve_status = "SOLVED" if any(g >= 1.0 for g in run_gt) else "NOT SOLVED"
                    break
            print(f"[{dataset_name}] {solve_status} during evolve run")
            score_str = f", train_score {bundle.score:.4f}" if bundle.score is not None else ""
            for t, op in bundle.operators.items():
                if op is not None:
                    print(f"Loaded [{t}] {op.name} (gen {op.generation}){score_str}")
            if bundle.best_hparams:
                print(
                    f"Applied {len(bundle.best_hparams)} HPO-tuned hparam(s) "
                    f"from bundle: {sorted(bundle.best_hparams.keys())}"
                )

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
        noise_seed = seed + 1000  # Derived seed for reproducibility
        y_train = add_noise(y_train, target_noise, seed=noise_seed)
        if verbose:
            print(f"Applied target noise: {target_noise} (seed={noise_seed})")

    if verbose:
        print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # https://stackoverflow.com/a/57474787/4383594
    try:
        n_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES'))
    except (TypeError, ValueError):
        n_cpus = 1

    # Load any custom Julia code from --evolve-results before constructing the
    # model so the new operators are visible to PySR's initialization.
    if evolve_extra_code:
        from parallel_eval_pysr import (
            _load_dynamic_loss,
            _load_dynamic_mutations,
            _load_dynamic_selection,
            _load_dynamic_survival,
        )
        if evolve_extra_code.get("custom_mutation_code"):
            _load_dynamic_mutations(evolve_extra_code["custom_mutation_code"])
        if evolve_extra_code.get("custom_survival_code"):
            _load_dynamic_survival(evolve_extra_code["custom_survival_code"])
        if evolve_extra_code.get("custom_selection_code"):
            _load_dynamic_selection(evolve_extra_code["custom_selection_code"])
        if evolve_extra_code.get("custom_loss_code"):
            _load_dynamic_loss(evolve_extra_code["custom_loss_code"])

    # Configure PySR
    pysr_model_kwargs = dict(
        # timeout_in_seconds=int(time_minutes * 60),
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "square"],
        maxsize=max_size,
        maxdepth=10,
        batching=False,
        # ncycles_per_iteration=10,
        # parallelism='multithreading',
        # procs=n_cpus,
        parallelism='serial',
        deterministic=True,
        # model-selection='score',
        populations=3*n_cpus,
        niterations=1000000000000,
        warm_start=True,
        # population_size=100,
        max_evals=max_evals,
        early_stop_condition=1e-8,
        random_state=seed,
        # verbosity=5,
        output_directory=results_dir,
        constraints={
            **dict(
                sin=9,
                exp=9,
                log=9,
                sqrt=9,
            ),
            **{"/": (-1, 9)}
        },
        nested_constraints=dict(
            sin=dict(
                sin=0,
                exp=1,
                log=1,
                sqrt=1,
            ),
            exp=dict(
                exp=0,
                log=0,
            ),
            log=dict(
                exp=0,
                log=0,
            ),
            sqrt=dict(
                sqrt=0,
            )
        ),
    )
    # Bundle-tuned PySR hparams take precedence over the defaults above; the
    # weight_* mutation weights are layered on top and only include
    # weight_custom_mutation_* when a custom mutation was actually loaded.
    pysr_model_kwargs.update(evolve_pysr_overrides)
    pysr_model_kwargs.update(evolve_weights)
    model = PySRRegressor(**pysr_model_kwargs)

    if verbose:
        print(f"\nStarting PySR fit...")

    # Fit the model
    # model.fit(X_train, y_train, variable_names=feature_names)

    # Fit the model (with optional HOF milestone checkpointing)
    hof_csv_path = os.path.join(results_dir, f"{dataset_name}_hof.csv")
    model = run_pysr_with_hof_checkpoints(
        X_train, y_train,
        feature_names=feature_names,
        dataset_name=dataset_name,
        results_dir=results_dir,
        milestones=milestones,
        model=model,
        seed=seed,
        hof_path=hof_csv_path,
    )

    # Persist rename_map alongside checkpoint.pkl so downstream symbolic
    # match checking can reverse the rename.
    rename_map = metadata.get('rename_map') or {}
    if rename_map and getattr(model, 'output_directory_', None) and getattr(model, 'run_id_', None):
        run_dir = Path(model.output_directory_) / model.run_id_
        # In milestone mode run_pysr_with_hof_checkpoints already deleted the
        # temp output dir (checkpoint.pkl included); recreating it just to hold
        # the sidecar would leave a phantom run dir with no artifacts for the
        # sidecar to describe. Only write beside surviving artifacts.
        if run_dir.is_dir():
            with open(run_dir / 'rename_map.json', 'w') as f:
                json.dump({
                    'rename_map': rename_map,
                    'original_feature_names': metadata.get('original_feature_names', []),
                }, f, indent=2)

    fit_time = time.time() - start_time

    if verbose:
        print(f"\nFit completed in {fit_time:.1f} seconds")

    # Evaluate on test set
    y_pred = model.predict(X_test)

    # Compute metrics
    mse = np.mean((y_test - y_pred) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # Get best equation
    best_eq = model.get_best()

    # import pdb; pdb.set_trace()

    if hasattr(best_eq, 'sympy_format'):
        best_equation_str = str(best_eq.sympy_format)
    else:
        best_equation_str = str(best_eq)

    # Check GT symbolic match across the full Pareto frontier (matches the
    # gt_match_score used by parallel_eval_pysr / evaluate_new_pysr). Any
    # expression in the frontier matching the GT scores 1.0.
    from evaluation import check_pysr_frontier_symbolic_match
    from parallel_eval_pysr import _remap_formula_variables

    gt_formula = metadata.get('ground_truth_formula', '') or ''
    # If feature names were renamed (e.g. gamma -> var_3), apply the same
    # rename to GT formula so symbolic matching uses consistent variables.
    gt_formula_for_match = gt_formula
    if rename_map and gt_formula:
        gt_formula_for_match = _remap_formula_variables(
            gt_formula, metadata['original_feature_names'], feature_names
        )

    best_df_index = best_eq.name if best_eq is not None and hasattr(best_eq, 'name') else None
    gt_match_score = 0.0
    matched_df_index = None
    matched_equation_str = None
    if gt_formula_for_match and model.equations_ is not None and len(model.equations_) > 0:
        try:
            gt_match_result = check_pysr_frontier_symbolic_match(
                equations_df=model.equations_,
                best_df_index=best_df_index,
                ground_truth_str=gt_formula_for_match,
                var_names=feature_names,
                timeout_seconds_per_expression=3,
                predict_fn=lambda idx: model.predict(X_test, index=int(idx)),
                y=y_test,
                min_r2=0.5,
            )
            if gt_match_result.get("match", False):
                gt_match_score = 1.0
                matched_df_index = gt_match_result.get("matched_df_index")
                if matched_df_index is not None:
                    matched_equation_str = str(model.equations_.loc[matched_df_index]["equation"])
        except Exception as e:
            print(f"GT match check failed: {e}")

    if verbose:
        print(f"\nGround truth formula: {gt_formula!r}")
        if gt_formula_for_match != gt_formula:
            print(f"  (after rename: {gt_formula_for_match!r})")
        print(f"GT symbolic match: {bool(gt_match_score)}"
              + (f"  (matched index {matched_df_index})" if matched_df_index is not None else ""))

        # Print Pareto frontier with markers for best and GT match.
        eqs = model.equations_
        if eqs is not None and len(eqs) > 0:
            print("\nPareto frontier:")
            for idx, row in eqs.iterrows():
                tag = ""
                if idx == best_df_index:
                    tag += " [best]"
                if idx == matched_df_index:
                    tag += " [GT MATCH]"
                print(f"  [{idx}] complexity={int(row['complexity']):>3}  "
                      f"loss={float(row['loss']):.4e}  "
                      f"eq={row['equation']}{tag}")

    from parallel_eval_pysr import _get_pysr_num_evaluations
    execution_trace = None
    if milestones:
        from parallel_eval_pysr import _load_execution_trace
        execution_trace = _load_execution_trace([hof_csv_path])

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
        # 'n_cpus': n_cpus,
        'max_size': max_size,
        'target_noise': target_noise,
        'num_evaluations': _get_pysr_num_evaluations(model),
        'milestones': milestones,
        'execution_trace': execution_trace,
        'execution_traces': [execution_trace] if execution_trace else [],
        'rename_map': rename_map,
        'original_feature_names': metadata.get('original_feature_names', []),
        'ground_truth_formula': gt_formula,
        'gt_match_score': float(gt_match_score),
        'gt_match_index': (int(matched_df_index) if matched_df_index is not None else None),
        'gt_match_equation': matched_equation_str,
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Test MSE: {mse:.4e}")
        print(f"  Test R²:  {r2:.4f}")
        print(f"  GT match: {bool(gt_match_score)}")
        print(f"  Best equation: {best_equation_str}")
        if matched_equation_str is not None:
            print(f"  GT-matched equation: {matched_equation_str}")

    # Save results (include noise level in filename when > 0)
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

    # Dataset selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str,
                       help='Single dataset name to run (e.g., feynman_I_15_10)')
    group.add_argument('--split-file', type=str,
                       help='Path to split file with dataset names')

    # For SLURM array jobs
    parser.add_argument('--array-index', type=int, default=None,
                       help='SLURM array task index (0-based). Uses SLURM_ARRAY_TASK_ID if not specified.')

    # Data settings
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples to use from each dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--target-noise', type=float, default=0.0,
                       help='Gaussian noise level for target (SRBench standard levels: 0.0, 0.001, 0.01, 0.1)')

    # PySR settings
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument('--niterations', type=int, default=None,
                    #    help='Maximum iterations')
    parser.add_argument('--max-evals', type=int, default=int(1e6),
                       help='Maximum evaluations')
    parser.add_argument('--hof-n', type=int, default=0,
                       help='Number of HOF checkpoints. If 0, no HOF trace file is saved.')
    parser.add_argument('--evolve-results', type=str, default=None,
                       help='Path to evolve_pysr run_data.json, a run directory, '
                            'or a run id under runs/. Loads custom mutation/'
                            'survival/selection/loss code (and any bundle-tuned '
                            'PySR hparams) into this run.')

    # Output settings
    parser.add_argument('--results-dir', type=str, default='results_pysr',
                       help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    # print script execution command
    print("Executing command: " + " ".join(os.sys.argv))
    verbose = not args.quiet

    # Determine which dataset(s) to run
    if args.dataset:
        datasets = [args.dataset]
        run_index = 0
    else:
        datasets = load_split_file(args.split_file)

        # Get array index from args or SLURM environment
        if args.array_index is not None:
            run_index = args.array_index
        else:
            run_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

        if run_index >= len(datasets):
            print(f"Array index {run_index} is out of range (only {len(datasets)} datasets)")
            return

        datasets = [datasets[run_index]]

    # Run on each dataset
    for dataset_name in datasets:
        try:
            results = run_pysr_on_dataset(
                dataset_name=dataset_name,
                max_samples=args.max_samples,
                results_dir=args.results_dir,
                seed=args.seed,
                # niterations=100000000000,
                max_evals=args.max_evals,
                verbose=verbose,
                target_noise=args.target_noise,
                hof_n=args.hof_n,
                evolve_results=args.evolve_results,
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
