"""
Script to calculate average/median performance on each ground truth problem
and sort problems from easiest to hardest.
"""
import pandas as pd
import argparse
import os
import json
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

# Import formula extraction functions
from srbench_formulas import (
    extract_formula_from_metadata,
    extract_operators_from_formula,
)

# Default path relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'srbench', 'results')
PMLB_DATASETS_DIR = os.path.join(SCRIPT_DIR, '..', 'pmlb', 'datasets')


def load_data(results_dir=None):
    """Load the ground truth results data."""
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    filepath = os.path.join(results_dir, 'ground-truth_results.feather')
    df = pd.read_feather(filepath)
    return df


def load_pysr_results_dir(results_dir):
    """Load PySR results from a directory of JSON files."""
    records = []
    for json_path in glob(os.path.join(results_dir, "*.json")):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue
        if "dataset" not in data or "test_r2" not in data:
            continue

        records.append({
            "dataset": data["dataset"],
            "test_r2": float(data["test_r2"]),
        })

    if not records:
        return pd.DataFrame(columns=["dataset", "test_r2"])

    return pd.DataFrame.from_records(records)


def load_pysr_results(
    results_base_dir="results",
    eval_counts=(1_000_000, 10_000_000, 100_000_000),
    noise_levels=(0.0, 0.001, 0.01, 0.1),
):
    """
    Load PySR SRBench-style results across eval counts and noise levels.

    Directory naming conventions:
      - noise=0.0: results_pysr_1e6, results_pysr_1e7, results_pysr_1e8
      - noise>0.0: results_pysr_{noise}_{max_evals}
    """
    eval_label_map = {
        1_000_000: "1e6",
        10_000_000: "1e7",
        100_000_000: "1e8",
    }

    frames = []
    missing_dirs = []

    for eval_count in eval_counts:
        for noise in noise_levels:
            if noise == 0.0:
                label = eval_label_map.get(eval_count, str(eval_count))
                dir_name = f"results_pysr_{label}"
            else:
                dir_name = f"results_pysr_{noise:g}_{int(eval_count)}"

            dir_path = Path(results_base_dir) / dir_name
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
                continue

            df_dir = load_pysr_results_dir(str(dir_path))
            if df_dir.empty:
                continue

            df_dir = df_dir.copy()
            df_dir["max_evals"] = int(eval_count)
            df_dir["target_noise"] = float(noise)
            df_dir["results_dir"] = str(dir_path)
            frames.append(df_dir)

    if not frames:
        return pd.DataFrame(columns=["dataset", "test_r2", "max_evals", "target_noise"]), missing_dirs

    combined = pd.concat(frames, ignore_index=True)
    return combined, missing_dirs


def compute_pysr_difficulty(
    results_base_dir="results",
    eval_counts=(1_000_000, 10_000_000, 100_000_000),
    noise_levels=(0.0, 0.001, 0.01, 0.1),
):
    """
    Compute PySR difficulty by averaging R^2 across eval counts and noise levels.
    """
    pysr_df, missing_dirs = load_pysr_results(
        results_base_dir=results_base_dir,
        eval_counts=eval_counts,
        noise_levels=noise_levels,
    )

    if pysr_df.empty:
        return pd.DataFrame(columns=["dataset", "pysr_avg_r2", "pysr_n_results"]), missing_dirs

    agg = pysr_df.groupby("dataset").agg(
        pysr_avg_r2=("test_r2", "mean"),
        pysr_n_results=("test_r2", "count"),
    ).reset_index()

    # Rank: hardest = lowest avg R^2
    agg = agg.sort_values("pysr_avg_r2", ascending=True).reset_index(drop=True)
    agg["pysr_difficulty_rank"] = range(1, len(agg) + 1)

    return agg, missing_dirs


def compare_srbench_vs_pysr_difficulty(
    srbench_difficulty,
    pysr_difficulty,
    out_dir="plots/pysr_srbench_difficulty",
    top_k=20,
):
    """
    Compare SRBench difficulty ranking with PySR difficulty ranking and plot agreement.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Merge on dataset
    merged = srbench_difficulty.merge(
        pysr_difficulty,
        on="dataset",
        how="inner",
    )

    if merged.empty:
        print("No overlapping datasets between SRBench and PySR results.")
        return merged

    merged = merged.copy()

    # Re-rank PySR within the merged set to align with SRBench coverage
    merged = merged.sort_values("pysr_avg_r2", ascending=True).reset_index(drop=True)
    merged["pysr_difficulty_rank"] = range(1, len(merged) + 1)

    # Compute agreement metrics
    rank_corr = merged[["difficulty_rank", "pysr_difficulty_rank"]].corr(method="spearman").iloc[0, 1]

    # Hardest sets
    top_k = int(top_k)
    pysr_hard = set(merged.nsmallest(top_k, "pysr_avg_r2")["dataset"])
    srbench_hard = set(merged.nsmallest(top_k, "difficulty_rank")["dataset"])
    overlap = pysr_hard & srbench_hard

    print("\n" + "=" * 80)
    print("PySR vs SRBench difficulty comparison")
    print("=" * 80)
    print(f"Overlapping datasets: {len(merged)}")
    print(f"Spearman rank correlation (overall): {rank_corr:.3f}")
    print(f"Top-{top_k} overlap: {len(overlap)} / {top_k}")
    if overlap:
        overlap_list = sorted(
            overlap,
            key=lambda d: merged.loc[merged["dataset"] == d, "pysr_difficulty_rank"].iloc[0],
        )
        print(f"Overlap datasets (sorted by PySR hardness): {overlap_list}")

    # Plot 1: Rank vs Rank scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        merged["difficulty_rank"],
        merged["pysr_difficulty_rank"],
        s=18,
        alpha=0.4,
        color="#8c8c8c",
        label="All datasets",
    )

    # Overlay hard sets
    hard_mask_pysr = merged["dataset"].isin(pysr_hard)
    hard_mask_srbench = merged["dataset"].isin(srbench_hard)

    ax.scatter(
        merged.loc[hard_mask_srbench, "difficulty_rank"],
        merged.loc[hard_mask_srbench, "pysr_difficulty_rank"],
        s=28,
        color="#1f77b4",
        label=f"SRBench top-{top_k}",
    )
    ax.scatter(
        merged.loc[hard_mask_pysr, "difficulty_rank"],
        merged.loc[hard_mask_pysr, "pysr_difficulty_rank"],
        s=28,
        color="#d62728",
        label=f"PySR top-{top_k}",
    )

    # Highlight overlap
    overlap_mask = merged["dataset"].isin(overlap)
    ax.scatter(
        merged.loc[overlap_mask, "difficulty_rank"],
        merged.loc[overlap_mask, "pysr_difficulty_rank"],
        s=45,
        color="#000000",
        label="Overlap",
    )

    max_rank = max(merged["difficulty_rank"].max(), merged["pysr_difficulty_rank"].max())
    ax.plot([1, max_rank], [1, max_rank], linestyle="--", color="#666666", linewidth=1)
    ax.set_xlabel("SRBench difficulty rank (1 = hardest)")
    ax.set_ylabel("PySR difficulty rank (1 = hardest)")
    ax.set_title("Difficulty rank agreement: SRBench vs PySR")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rank_agreement_scatter.png"), dpi=200)
    plt.close(fig)

    # Plot 2: Hard-end comparison for PySR top-K
    hard_df = merged[merged["dataset"].isin(pysr_hard)].copy()
    hard_df = hard_df.sort_values("pysr_avg_r2", ascending=True).reset_index(drop=True)
    hard_df["pysr_hard_rank"] = range(1, len(hard_df) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        hard_df["pysr_hard_rank"],
        hard_df["difficulty_rank"],
        s=30,
        color="#d62728",
    )
    ax.plot(
        [1, len(hard_df)],
        [1, len(hard_df)],
        linestyle="--",
        color="#666666",
        linewidth=1,
    )
    for _, row in hard_df.iterrows():
        ax.text(
            row["pysr_hard_rank"] + 0.1,
            row["difficulty_rank"],
            row["dataset"],
            fontsize=7,
            alpha=0.8,
        )
    ax.set_xlabel(f"PySR hard-rank within top-{top_k}")
    ax.set_ylabel("SRBench difficulty rank (1 = hardest)")
    ax.set_title(f"Hard-end agreement (PySR top-{top_k})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hard_end_agreement.png"), dpi=200)
    plt.close(fig)

    # Plot 3: Overlap vs K (top-K taken from SRBench rankings)
    max_k = min(len(merged), 50)
    ks = list(range(1, max_k + 1))
    overlaps = []
    for k in ks:
        pysr_top = set(merged.nsmallest(k, "pysr_avg_r2")["dataset"])
        sr_top = set(merged.nsmallest(k, "difficulty_rank")["dataset"])
        overlaps.append(len(pysr_top & sr_top))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, overlaps, color="#2ca02c", linewidth=2)
    ax.set_xlabel("K (top-K from SRBench)")
    ax.set_ylabel("# overlapping datasets")
    ax.set_title("Top-K overlap: SRBench vs PySR")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "topk_overlap_curve.png"), dpi=200)
    plt.close(fig)

    # Save merged data
    merged.to_csv(os.path.join(out_dir, "pysr_srbench_difficulty_comparison.csv"), index=False)

    return merged


def parse_eval_list(eval_str):
    values = []
    for part in eval_str.split(','):
        part = part.strip()
        if not part:
            continue
        values.append(int(float(part)))
    return tuple(values)


def parse_noise_list(noise_str):
    values = []
    for part in noise_str.split(','):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return tuple(values)


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


def generate_hard_splits(
    difficulty_df,
    n_hard_total=45,
    n_tail_per_split=5,
    splits=("train_hard", "val_hard", "test_hard"),
):
    """
    Generate hard splits:
      - Top n_hard_total hardest problems split evenly across splits (serpentine draft).
      - Add n_tail_per_split from remaining difficulty distribution to train/val.

    Returns:
    --------
    difficulty_df with 'split_hard' column added.
    """
    df = difficulty_df.copy()
    df["split_hard"] = ""

    # Mark excluded datasets
    df.loc[df["dataset"].isin(EXCLUDED_DATASETS), "split_hard"] = "excluded"

    includable = df[df["split_hard"] != "excluded"].copy()

    # Hard pool
    hard_pool = includable.head(n_hard_total).copy()
    rest_pool = includable.iloc[n_hard_total:].copy()

    # Serpentine assignment for hard pool
    order = []
    direction = 1
    idx = 0
    for _ in range(len(hard_pool)):
        order.append(splits[idx])
        idx += direction
        if idx == len(splits):
            idx = len(splits) - 1
            direction = -1
        elif idx < 0:
            idx = 0
            direction = 1

    hard_pool["split_hard"] = order

    # Tail: stratified sampling across remaining distribution for train/val
    tail_assignments = {s: [] for s in splits}
    if n_tail_per_split > 0 and len(rest_pool) > 0:
        strata_size = len(rest_pool) / n_tail_per_split
        strata = []
        for i in range(n_tail_per_split):
            start = int(i * strata_size)
            end = int((i + 1) * strata_size) if i < n_tail_per_split - 1 else len(rest_pool)
            strata.append(rest_pool.iloc[start:end])

        train_picks = []
        val_picks = []
        for stratum in strata:
            if len(stratum) == 0:
                continue
            datasets = stratum["dataset"].tolist()
            train_pick = datasets[0]
            # Prefer a different dataset for val to avoid overwriting train
            if len(datasets) > 1:
                val_pick = datasets[-1] if datasets[-1] != train_pick else datasets[1]
            else:
                val_pick = None

            train_picks.append(train_pick)
            if val_pick is not None:
                val_picks.append(val_pick)

        tail_assignments["train_hard"] = train_picks
        tail_assignments["val_hard"] = val_picks

    # Apply assignments
    for _, row in hard_pool.iterrows():
        df.loc[df["dataset"] == row["dataset"], "split_hard"] = row["split_hard"]

    for s in ["train_hard", "val_hard"]:
        for d in tail_assignments[s]:
            df.loc[df["dataset"] == d, "split_hard"] = s

    # Assign all remaining includable datasets to test_hard
    remaining_mask = (df["split_hard"] == "") & (~df["dataset"].isin(EXCLUDED_DATASETS))
    df.loc[remaining_mask, "split_hard"] = "test_hard"

    return df


def plot_split_coverage_vs_pysr_topk(
    split_df,
    pysr_difficulty,
    out_dir,
    split_col="split_hard",
    split_names=("train_hard", "val_hard", "test_hard"),
):
    """
    Plot number of datasets from each split in the PySR top-K hardest.
    """
    os.makedirs(out_dir, exist_ok=True)

    merged = split_df.merge(
        pysr_difficulty[["dataset", "pysr_avg_r2"]],
        on="dataset",
        how="inner",
    )

    if merged.empty:
        print("No overlapping datasets between PySR difficulty and splits.")
        return

    merged = merged.sort_values("pysr_avg_r2", ascending=True).reset_index(drop=True)
    merged["pysr_difficulty_rank"] = range(1, len(merged) + 1)

    max_k = len(merged)
    ks = list(range(1, max_k + 1))

    counts = {s: [] for s in split_names}
    for k in ks:
        topk = set(merged.nsmallest(k, "pysr_avg_r2")["dataset"])
        for s in split_names:
            in_split = set(merged[merged[split_col] == s]["dataset"])
            counts[s].append(len(topk & in_split))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "train_hard": "#1f77b4",
        "val_hard": "#ff7f0e",
        "test_hard": "#2ca02c",
    }
    for s in split_names:
        ax.plot(ks, counts[s], label=s, linewidth=2, color=colors.get(s))

    ax.set_xlabel("K (top-K by PySR difficulty)")
    ax.set_ylabel("# tasks from split in PySR top-K")
    ax.set_title("Split coverage in PySR hardest tasks")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pysr_topk_split_coverage.png"), dpi=200)
    plt.close(fig)


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
    parser.add_argument('--compare-pysr', action='store_true',
                        help='Compare SRBench difficulty with PySR difficulty and generate plots')
    parser.add_argument('--pysr-results-base', default='results',
                        help='Base directory containing PySR results_* folders (default: results)')
    parser.add_argument('--pysr-evals', default='1e6,1e7,1e8',
                        help='Comma-separated eval counts for PySR (default: 1e6,1e7,1e8)')
    parser.add_argument('--pysr-noise-levels', default='0,0.001,0.01,0.1',
                        help='Comma-separated target noise levels (default: 0,0.001,0.01,0.1)')
    parser.add_argument('--pysr-top-k', type=int, default=20,
                        help='Top-K hardest problems to emphasize in plots (default: 20)')
    parser.add_argument('--pysr-outdir', default='plots/pysr_srbench_difficulty',
                        help='Output directory for PySR vs SRBench plots (default: plots/pysr_srbench_difficulty)')
    parser.add_argument('--hard-split', action='store_true',
                        help='Generate hard splits train_hard/val_hard/test_hard')
    parser.add_argument('--hard-total', type=int, default=45,
                        help='Total number of hardest problems to distribute across hard splits (default: 45)')
    parser.add_argument('--hard-tail', type=int, default=5,
                        help='Number of non-hard problems to add to train/val from remaining distribution (default: 5)')

    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir else DEFAULT_RESULTS_DIR
    print(f"Loading data from {results_dir}...")
    df = load_data(results_dir)

    print("Computing symbolic solutions...")
    df = compute_symbolic_solution(df)

    print("Computing problem difficulty by number of solves...")
    difficulty = compute_problem_difficulty(df, noise_level=args.noise_level)

    if args.compare_pysr:
        eval_counts = parse_eval_list(args.pysr_evals)
        noise_levels = parse_noise_list(args.pysr_noise_levels)

        pysr_difficulty, missing_dirs = compute_pysr_difficulty(
            results_base_dir=args.pysr_results_base,
            eval_counts=eval_counts,
            noise_levels=noise_levels,
        )

        if missing_dirs:
            print("\nMissing PySR result directories (skipped):")
            for d in missing_dirs:
                print(f"  {d}")

        compare_srbench_vs_pysr_difficulty(
            srbench_difficulty=difficulty,
            pysr_difficulty=pysr_difficulty,
            out_dir=args.pysr_outdir,
            top_k=args.pysr_top_k,
        )

    if args.hard_split:
        print(f"Generating hard splits (hard_total={args.hard_total}, hard_tail={args.hard_tail})...")
        hard_df = generate_hard_splits(
            difficulty,
            n_hard_total=args.hard_total,
            n_tail_per_split=args.hard_tail,
        )

        splits_dir = Path("splits")
        splits_dir.mkdir(exist_ok=True)
        for split_name in ["train_hard", "val_hard", "test_hard"]:
            problems = hard_df[hard_df["split_hard"] == split_name]["dataset"].tolist()
            filename = splits_dir / f"{split_name}.txt"
            with open(filename, "w") as f:
                f.write("\n".join(problems) + "\n")
            print(f"  {filename}: {len(problems)} datasets")

        # Plot PySR top-K coverage for hard splits
        pysr_difficulty, _ = compute_pysr_difficulty(
            results_base_dir=args.pysr_results_base,
            eval_counts=parse_eval_list(args.pysr_evals),
            noise_levels=parse_noise_list(args.pysr_noise_levels),
        )
        plot_split_coverage_vs_pysr_topk(
            split_df=hard_df,
            pysr_difficulty=pysr_difficulty,
            out_dir=args.pysr_outdir,
        )

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

        with open('A.txt', 'w') as f:
            f.write('\n'.join(split_a) + '\n')
        with open('B.txt', 'w') as f:
            f.write('\n'.join(split_b) + '\n')

        print(f"\nSplit files saved:")
        print(f"  A.txt: {len(split_a)} datasets")
        print(f"  B.txt: {len(split_b)} datasets")
    else:
        # Save train/val/test splits
        for split_name in ['train', 'val', 'test']:
            problems = difficulty[difficulty['split'] == split_name]['dataset'].tolist()
            filename = f'{split_name}.txt'
            with open(filename, 'w') as f:
                f.write('\n'.join(problems) + '\n')
            print(f"  {filename}: {len(problems)} datasets")

        # Also generate and save the 4-problem subsets
        print(f"\n{'='*60}")
        print("Generating 4-problem subset splits...")
        print('='*60)
        subset = generate_subset_split(difficulty, subset_train_size=4, subset_val_size=4)

        with open('train_small.txt', 'w') as f:
            f.write('\n'.join(subset['train']) + '\n')
        with open('val_small.txt', 'w') as f:
            f.write('\n'.join(subset['val']) + '\n')

        print(f"  train_small.txt: {len(subset['train'])} datasets")
        print(f"  val_small.txt: {len(subset['val'])} datasets")
        print(f"\nSmall train subset: {subset['train']}")
        print(f"Small val subset: {subset['val']}")

    return difficulty


if __name__ == '__main__':
    main()
