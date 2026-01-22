"""
Script to calculate average/median performance on each ground truth problem
and sort problems from easiest to hardest.
"""
import pandas as pd
import argparse
import os
from pathlib import Path

# Import formula extraction functions
from srbench_formulas import (
    extract_formula_from_metadata,
    extract_operators_from_formula,
)

# Default path relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, 'srbench', 'results')
PMLB_DATASETS_DIR = os.path.join(SCRIPT_DIR, 'pmlb', 'datasets')


def load_data(results_dir=None):
    """Load the ground truth results data."""
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    filepath = os.path.join(results_dir, 'ground-truth_results.feather')
    df = pd.read_feather(filepath)
    return df


def compute_symbolic_solution(df):
    """Compute symbolic solution indicator (same logic as srbench notebook)."""
    df = df.copy()
    df['symbolic_solution'] = df[['symbolic_error_is_zero',
                                   'symbolic_error_is_constant',
                                   'symbolic_fraction_is_constant']
                                  ].apply(any, raw=True, axis=1)
    # Clean up corner cases
    df['symbolic_solution'] = df['symbolic_solution'] & ~df['simplified_symbolic_model'].isna()
    df['symbolic_solution'] = df['symbolic_solution'] & ~(df['simplified_symbolic_model'] == '0')
    df['symbolic_solution'] = df['symbolic_solution'] & ~(df['simplified_symbolic_model'] == 'nan')
    return df


def add_formula_and_operators(df, pmlb_path=None):
    """
    Add ground truth formula and operator columns to a dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'dataset' column containing dataset names
    pmlb_path : str or None
        Path to pmlb/datasets directory. If None, uses default.

    Returns:
    --------
    pd.DataFrame with added columns: 'formula', 'binary_ops', 'unary_ops'
    """
    if pmlb_path is None:
        pmlb_path = Path(PMLB_DATASETS_DIR)
    else:
        pmlb_path = Path(pmlb_path)

    df = df.copy()

    formulas = []
    binary_ops_list = []
    unary_ops_list = []

    for dataset_name in df['dataset']:
        metadata_path = pmlb_path / dataset_name / 'metadata.yaml'
        formula = extract_formula_from_metadata(metadata_path)

        if formula:
            # Extract just the RHS of the equation for cleaner display
            if '=' in formula:
                formula_rhs = formula.split('=', 1)[1].strip()
            else:
                formula_rhs = formula

            ops = extract_operators_from_formula(formula)
            binary_ops = sorted(ops['binary'])
            unary_ops = sorted(ops['unary'])
        else:
            formula_rhs = ''
            binary_ops = []
            unary_ops = []

        formulas.append(formula_rhs)
        binary_ops_list.append(','.join(binary_ops) if binary_ops else '')
        unary_ops_list.append(','.join(unary_ops) if unary_ops else '')

    df['formula'] = formulas
    df['binary_ops'] = binary_ops_list
    df['unary_ops'] = unary_ops_list

    return df


def compute_problem_difficulty(df, noise_level=None):
    """
    Compute difficulty for each problem based on total number of solves.

    Parameters:
    -----------
    df : pd.DataFrame
        The results dataframe (must have symbolic_solution column computed)
    noise_level : float or None
        If specified, filter to this noise level. If None, aggregate across all noise levels.

    Returns:
    --------
    pd.DataFrame with problems sorted from hardest to easiest (by n_solved)
    """
    df = df.copy()

    # Filter by noise level if specified
    if noise_level is not None:
        df = df[df['target_noise'] == noise_level]

    # Compute high accuracy indicator (R^2 > 0.999)
    df['high_acc'] = df['r2_test'] > 0.999

    # Compute total number of solves and high accuracy per dataset
    # (sum over all algorithms, noise levels, and trials)
    problem_difficulty = df.groupby('dataset').agg(
        n_solved=('symbolic_solution', 'sum'),
        n_high_acc=('high_acc', 'sum')
    ).reset_index()

    # Sort by n_solved (ascending = hardest first), then by n_high_acc (ascending = hardest first)
    problem_difficulty = problem_difficulty.sort_values(
        ['n_solved', 'n_high_acc'], ascending=[True, True]
    ).reset_index(drop=True)

    # Add difficulty rank (1 = hardest)
    problem_difficulty['difficulty_rank'] = range(1, len(problem_difficulty) + 1)

    # Add formula and operators columns
    problem_difficulty = add_formula_and_operators(problem_difficulty)

    return problem_difficulty


def get_problems_by_difficulty(noise_level=None):
    """
    Convenience function to get a list of problem names sorted by difficulty.

    Parameters:
    -----------
    noise_level : float or None
        Target noise level (0.0, 0.001, 0.01, 0.1). If None, aggregate across all noise levels.

    Returns:
    --------
    list of str: Problem names sorted from hardest to easiest
    """
    df = load_data()
    df = compute_symbolic_solution(df)
    difficulty = compute_problem_difficulty(df, noise_level=noise_level)
    return difficulty['dataset'].tolist()


# Datasets excluded from run_pysr.sh (problematic datasets)
EXCLUDED_DATASETS = {'feynman_test_10', 'feynman_I_26_2', 'feynman_I_30_5'}


def generate_balanced_split(difficulty_df):
    """
    Generate a balanced 50/50 split of problems using serpentine draft.

    Strategy:
    - Exclude problematic datasets from run_pysr.sh
    - Separate Feynman and Strogatz problems
    - Sort each by difficulty
    - Use serpentine draft (1-2-2-2-...) to alternate between splits

    Parameters:
    -----------
    difficulty_df : pd.DataFrame
        DataFrame with 'dataset', 'n_solved', 'n_high_acc' columns, sorted by difficulty

    Returns:
    --------
    difficulty_df with 'split' column added ('A', 'B', or 'excluded')
    """
    df = difficulty_df.copy()

    # Mark excluded datasets
    df['split'] = ''
    df.loc[df['dataset'].isin(EXCLUDED_DATASETS), 'split'] = 'excluded'

    # Separate Feynman and Strogatz (excluding problematic ones)
    feynman_mask = df['dataset'].str.startswith('feynman') & (df['split'] != 'excluded')
    strogatz_mask = df['dataset'].str.startswith('strogatz') & (df['split'] != 'excluded')

    feynman_indices = df[feynman_mask].index.tolist()
    strogatz_indices = df[strogatz_mask].index.tolist()

    # Simpler serpentine: A, B, B, A, A, B, B, A, ...
    def serpentine_assign_simple(indices):
        """Assign splits: first to A, next two to B, next two to A, etc."""
        splits = []
        current_split = 'A'
        count_in_current = 0
        first = True

        for idx in indices:
            splits.append(current_split)
            count_in_current += 1

            # First assignment is solo, then alternate in pairs
            if first:
                current_split = 'B'
                count_in_current = 0
                first = False
            elif count_in_current == 2:
                current_split = 'A' if current_split == 'B' else 'B'
                count_in_current = 0

        return splits

    # Assign Feynman problems
    feynman_splits = serpentine_assign_simple(feynman_indices)
    for idx, split in zip(feynman_indices, feynman_splits):
        df.loc[idx, 'split'] = split

    # Assign Strogatz problems
    strogatz_splits = serpentine_assign_simple(strogatz_indices)
    for idx, split in zip(strogatz_indices, strogatz_splits):
        df.loc[idx, 'split'] = split

    return df


def print_split_summary(df, split_names=None):
    """Print summary statistics for the split."""
    if split_names is None:
        split_names = ['A', 'B']

    # Filter out excluded
    df_valid = df[df['split'].isin(split_names)]

    for split in split_names:
        split_df = df_valid[df_valid['split'] == split]
        if len(split_df) == 0:
            continue
        feynman_count = split_df['dataset'].str.startswith('feynman').sum()
        strogatz_count = split_df['dataset'].str.startswith('strogatz').sum()

        print(f"\nSplit {split}:")
        print(f"  Total problems: {len(split_df)}")
        print(f"  Feynman: {feynman_count}, Strogatz: {strogatz_count}")
        print(f"  n_solved: mean={split_df['n_solved'].mean():.1f}, median={split_df['n_solved'].median():.1f}, sum={split_df['n_solved'].sum()}")
        print(f"  n_high_acc: mean={split_df['n_high_acc'].mean():.1f}, median={split_df['n_high_acc'].median():.1f}, sum={split_df['n_high_acc'].sum()}")


def generate_train_val_test_split(difficulty_df, train_size=20, val_size=20, test_size=None):
    """
    Generate a balanced train/val/test split of problems using stratified sampling.

    Strategy for balanced difficulty:
    - Exclude problematic datasets
    - Sort all problems by difficulty (hardest first)
    - Divide into strata (difficulty bins)
    - From each stratum, sample proportionally for train/val/test
    - This ensures ALL THREE splits have similar difficulty distributions

    Parameters:
    -----------
    difficulty_df : pd.DataFrame
        DataFrame with 'dataset', 'n_solved', 'n_high_acc' columns, sorted by difficulty
    train_size : int
        Number of problems for training set
    val_size : int
        Number of problems for validation set
    test_size : int or None
        Number of problems for test set. If None, uses all remaining problems.

    Returns:
    --------
    difficulty_df with 'split' column added ('train', 'val', 'test', or 'excluded')
    """
    df = difficulty_df.copy()

    # Mark excluded datasets
    df['split'] = ''
    df.loc[df['dataset'].isin(EXCLUDED_DATASETS), 'split'] = 'excluded'

    # Get indices of includable problems (sorted by difficulty - hardest first)
    includable_mask = df['split'] != 'excluded'
    includable_indices = df[includable_mask].index.tolist()

    n_includable = len(includable_indices)

    # Validate sizes
    if test_size is None:
        test_size = n_includable - train_size - val_size

    total_needed = train_size + val_size + test_size
    if total_needed > n_includable:
        raise ValueError(f"Requested {total_needed} problems but only {n_includable} available")

    train_indices = []
    val_indices = []
    test_indices = []

    # Stratified sampling to ensure all splits have balanced difficulty
    # Use enough strata to get good coverage
    n_strata = max(train_size + val_size, 20)  # At least as many strata as train+val

    # Process each stratum
    stratum_size = n_includable / n_strata

    for stratum_idx in range(n_strata):
        start = int(stratum_idx * stratum_size)
        end = int((stratum_idx + 1) * stratum_size) if stratum_idx < n_strata - 1 else n_includable
        stratum_indices = includable_indices[start:end]

        if not stratum_indices:
            continue

        # Calculate target cumulative counts at this point
        progress = (stratum_idx + 1) / n_strata
        train_target = train_size * progress
        val_target = val_size * progress

        # How many should we have assigned by now?
        train_needed = int(round(train_target)) - len(train_indices)
        val_needed = int(round(val_target)) - len(val_indices)

        # Clamp to what's available in this stratum
        total_from_stratum = len(stratum_indices)
        train_from_stratum = min(train_needed, total_from_stratum)
        if train_from_stratum < 0:
            train_from_stratum = 0

        remaining = total_from_stratum - train_from_stratum
        val_from_stratum = min(val_needed, remaining)
        if val_from_stratum < 0:
            val_from_stratum = 0

        test_from_stratum = remaining - val_from_stratum

        # Assign from this stratum using serpentine within stratum for extra balance
        # Alternate: train, val, test, test, val, train pattern
        assignments = []
        assignments.extend(['train'] * train_from_stratum)
        assignments.extend(['val'] * val_from_stratum)
        assignments.extend(['test'] * test_from_stratum)

        # Interleave assignments within stratum for micro-balance
        for j, idx in enumerate(stratum_indices):
            if j < len(assignments):
                split = assignments[j]
                if split == 'train' and len(train_indices) < train_size:
                    train_indices.append(idx)
                elif split == 'val' and len(val_indices) < val_size:
                    val_indices.append(idx)
                else:
                    test_indices.append(idx)
            else:
                test_indices.append(idx)

    # Assign splits
    for idx in train_indices:
        df.loc[idx, 'split'] = 'train'
    for idx in val_indices:
        df.loc[idx, 'split'] = 'val'
    for idx in test_indices:
        df.loc[idx, 'split'] = 'test'

    # Any remaining unassigned go to test
    remaining = df[(df['split'] == '') & (~df['dataset'].isin(EXCLUDED_DATASETS))].index
    df.loc[remaining, 'split'] = 'test'

    return df


def generate_subset_split(full_split_df, subset_train_size=4, subset_val_size=4):
    """
    Generate a smaller subset split that is contained within a larger split.

    Takes problems from 'train' and 'val' in the input df and creates a smaller
    subset using the same serpentine method to maintain difficulty balance.

    Parameters:
    -----------
    full_split_df : pd.DataFrame
        DataFrame with 'split' column containing 'train', 'val', 'test'
    subset_train_size : int
        Number of problems for the subset training set
    subset_val_size : int
        Number of problems for the subset validation set

    Returns:
    --------
    dict with:
        'train': list of dataset names for subset train
        'val': list of dataset names for subset val
    """
    df = full_split_df.copy()

    # Get train and val problems, maintaining difficulty order
    train_problems = df[df['split'] == 'train']['dataset'].tolist()
    val_problems = df[df['split'] == 'val']['dataset'].tolist()

    if subset_train_size > len(train_problems):
        raise ValueError(f"subset_train_size ({subset_train_size}) > available train ({len(train_problems)})")
    if subset_val_size > len(val_problems):
        raise ValueError(f"subset_val_size ({subset_val_size}) > available val ({len(val_problems)})")

    # Use serpentine selection from each set
    # Since they're already sorted by difficulty, take every N-th problem
    # to maintain difficulty balance

    def select_balanced_subset(problems, n):
        """Select n problems with balanced difficulty from a sorted list."""
        if n >= len(problems):
            return problems

        # Use evenly spaced indices to cover all difficulty levels
        indices = []
        step = len(problems) / n
        for i in range(n):
            idx = int(i * step)
            indices.append(idx)

        return [problems[i] for i in indices]

    subset_train = select_balanced_subset(train_problems, subset_train_size)
    subset_val = select_balanced_subset(val_problems, subset_val_size)

    return {
        'train': subset_train,
        'val': subset_val
    }


def get_train_val_test_splits(train_size=20, val_size=20, test_size=None, noise_level=None):
    """
    Convenience function to get train/val/test splits as lists of problem names.

    Parameters:
    -----------
    train_size : int
        Number of problems for training set
    val_size : int
        Number of problems for validation set
    test_size : int or None
        Number of problems for test set. If None, uses all remaining.
    noise_level : float or None
        Target noise level for difficulty calculation

    Returns:
    --------
    dict with 'train', 'val', 'test' keys, each containing a list of dataset names
    """
    df = load_data()
    df = compute_symbolic_solution(df)
    difficulty = compute_problem_difficulty(df, noise_level=noise_level)
    split_df = generate_train_val_test_split(difficulty, train_size, val_size, test_size)

    return {
        'train': split_df[split_df['split'] == 'train']['dataset'].tolist(),
        'val': split_df[split_df['split'] == 'val']['dataset'].tolist(),
        'test': split_df[split_df['split'] == 'test']['dataset'].tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Compute problem difficulty from SR benchmark results')
    parser.add_argument('--results-dir', default=None,
                        help=f'Directory containing results (default: {DEFAULT_RESULTS_DIR})')
    parser.add_argument('--noise-level', type=float, default=None,
                        help='Filter to specific noise level (0.0, 0.001, 0.01, 0.1)')
    parser.add_argument('--output', default=None, help='Output CSV file')
    parser.add_argument('--top-n', type=int, default=None, help='Show only top N hardest problems')
    parser.add_argument('--split-mode', default='train_val_test',
                        choices=['ab', 'train_val_test'],
                        help='Split mode: ab (50/50) or train_val_test (default)')
    parser.add_argument('--train-size', type=int, default=20,
                        help='Number of training problems (default: 20)')
    parser.add_argument('--val-size', type=int, default=20,
                        help='Number of validation problems (default: 20)')

    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir else DEFAULT_RESULTS_DIR
    print(f"Loading data from {results_dir}...")
    df = load_data(results_dir)

    print("Computing symbolic solutions...")
    df = compute_symbolic_solution(df)

    print("Computing problem difficulty by number of solves...")
    difficulty = compute_problem_difficulty(df, noise_level=args.noise_level)

    if args.split_mode == 'ab':
        print("Generating balanced A/B split...")
        difficulty = generate_balanced_split(difficulty)
        split_names = ['A', 'B']
    else:
        print(f"Generating train/val/test split (train={args.train_size}, val={args.val_size})...")
        difficulty = generate_train_val_test_split(difficulty, args.train_size, args.val_size)
        split_names = ['train', 'val', 'test']

    if args.top_n:
        print(f"\nTop {args.top_n} HARDEST problems:")
        display_df = difficulty.head(args.top_n)
    else:
        print(f"\nAll problems sorted by difficulty (hardest first):")
        display_df = difficulty

    print(display_df.to_string(index=False))

    # Also show easiest problems
    if not args.top_n:
        print(f"\n\nTop 10 EASIEST problems:")
        print(difficulty.tail(10).iloc[::-1].to_string(index=False))

    # Summary statistics
    print(f"\n\nSummary:")
    print(f"  Total problems: {len(difficulty)}")
    print(f"  Excluded: {(difficulty['split'] == 'excluded').sum()}")
    print(f"  n_solved range: {difficulty['n_solved'].min()} - {difficulty['n_solved'].max()}")
    print(f"  n_solved median: {difficulty['n_solved'].median()}")

    # Print split summary
    print(f"\n{'='*60}")
    print("Split Summary:")
    print('='*60)
    print_split_summary(difficulty, split_names)

    # Save to file if requested
    if args.output:
        difficulty.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

    # Save splits to text files
    if args.split_mode == 'ab':
        split_a = difficulty[difficulty['split'] == 'A']['dataset'].tolist()
        split_b = difficulty[difficulty['split'] == 'B']['dataset'].tolist()

        with open('split_A.txt', 'w') as f:
            f.write('\n'.join(split_a) + '\n')
        with open('split_B.txt', 'w') as f:
            f.write('\n'.join(split_b) + '\n')

        print(f"\nSplit files saved:")
        print(f"  split_A.txt: {len(split_a)} datasets")
        print(f"  split_B.txt: {len(split_b)} datasets")
    else:
        # Save train/val/test splits
        for split_name in ['train', 'val', 'test']:
            problems = difficulty[difficulty['split'] == split_name]['dataset'].tolist()
            filename = f'split_{split_name}.txt'
            with open(filename, 'w') as f:
                f.write('\n'.join(problems) + '\n')
            print(f"  {filename}: {len(problems)} datasets")

        # Also generate and save the 4-problem subsets
        print(f"\n{'='*60}")
        print("Generating 4-problem subset splits...")
        print('='*60)
        subset = generate_subset_split(difficulty, subset_train_size=4, subset_val_size=4)

        with open('split_train_small.txt', 'w') as f:
            f.write('\n'.join(subset['train']) + '\n')
        with open('split_val_small.txt', 'w') as f:
            f.write('\n'.join(subset['val']) + '\n')

        print(f"  split_train_small.txt: {len(subset['train'])} datasets")
        print(f"  split_val_small.txt: {len(subset['val'])} datasets")
        print(f"\nSmall train subset: {subset['train']}")
        print(f"Small val subset: {subset['val']}")

    return difficulty


if __name__ == '__main__':
    main()
