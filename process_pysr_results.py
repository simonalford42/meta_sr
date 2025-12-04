#!/usr/bin/env python
"""
Process PySR results and generate SRBench-style plots

Usage:
    python process_pysr_results.py results_pysr_5min
    python process_pysr_results.py results_pysr_groundtruth
"""
import sys
import os
import pandas as pd
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path

def collate_results(results_dir, output_name=None):
    """Collate individual JSON results into a feather file"""

    if output_name is None:
        output_name = Path(results_dir).name

    print(f"Collating results from: {results_dir}")

    frames = []
    excluded_datasets = [
        'feynman_test_10',
        'feynman_I_26_2',
        'feynman_I_30_5'
    ]
    excluded_cols = ['params']
    fails = []

    json_files = glob(f'{results_dir}/*/*.json')
    print(f"Found {len(json_files)} JSON files")

    for f in tqdm(json_files):
        if 'cv_results' in f:
            continue
        if any([ed in f for ed in excluded_datasets]):
            continue

        try:
            r = json.load(open(f, 'r'))
            if isinstance(r['symbolic_model'], list):
                print(f'WARNING: list returned for model: {f}')
                sm = ['B'+str(i)+'*'+ri for i, ri in enumerate(r['symbolic_model'])]
                sm = '+'.join(sm)
                r['symbolic_model'] = sm

            sub_r = {k: v for k, v in r.items() if k not in excluded_cols}
            frames.append(sub_r)
        except Exception as e:
            fails.append([f, e])

    print(f'{len(frames)} results loaded')
    print(f'{len(fails)} fails')
    for f in fails[:10]:  # Show first 10 failures
        print(f'  {f[0]}: {f[1]}')

    if len(frames) == 0:
        print("ERROR: No results loaded!")
        return None

    # Create dataframe
    df_results = pd.DataFrame.from_records(frames)

    # Cleanup
    df_results = df_results.rename(columns={'time_time': 'training time (s)'})
    df_results.loc[:, 'training time (hr)'] = df_results['training time (s)'] / 3600
    df_results['r2_zero_test'] = df_results['r2_test'].apply(lambda x: max(x, 0))

    # Fill NaN values for symbolic analysis columns
    for col in ['symbolic_error_is_zero', 'symbolic_error_is_constant',
                'symbolic_fraction_is_constant']:
        if col in df_results.columns:
            df_results.loc[:, col] = df_results[col].fillna(False)

    # Clean up algorithm names
    df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('Regressor', ''))
    df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('_5min', ' (5min)'))
    df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('_groundtruth', ''))
    df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('_blackbox', ''))

    # Add data group
    df_results['data_group'] = df_results['dataset'].apply(
        lambda x: 'Feynman' if 'feynman' in x else 'Strogatz'
    )

    # Compute symbolic solutions
    if 'symbolic_error_is_zero' in df_results.columns:
        df_results.loc[:, 'symbolic_solution'] = df_results[[
            'symbolic_error_is_zero',
            'symbolic_error_is_constant',
            'symbolic_fraction_is_constant'
        ]].apply(any, raw=True, axis=1)

        df_results.loc[:, 'symbolic_solution'] = (
            df_results['symbolic_solution'] &
            ~df_results['simplified_symbolic_model'].isna() &
            ~(df_results['simplified_symbolic_model'] == '0') &
            ~(df_results['simplified_symbolic_model'] == 'nan')
        )

    # Save
    output_file = f'{output_name}_results.feather'
    df_results.to_feather(output_file)
    print(f'\nResults saved to: {output_file}')
    print(f'  Shape: {df_results.shape}')
    print(f'  Algorithms: {df_results["algorithm"].unique()}')
    print(f'  Datasets: {df_results["dataset"].nunique()}')

    return df_results

def merge_with_srbench(pysr_df, output_name='combined'):
    """Merge PySR results with existing SRBench results"""

    srbench_file = 'srbench/results/ground-truth_results.feather'
    if not os.path.exists(srbench_file):
        print(f"SRBench results not found at {srbench_file}")
        print("Creating PySR-only results file")
        combined = pysr_df
    else:
        print(f"Loading SRBench results from {srbench_file}")
        srbench_df = pd.read_feather(srbench_file)
        print(f"  SRBench shape: {srbench_df.shape}")
        print(f"  SRBench algorithms: {srbench_df['algorithm'].unique()}")

        # Combine
        combined = pd.concat([srbench_df, pysr_df], ignore_index=True)
        print(f"\nCombined shape: {combined.shape}")
        print(f"Combined algorithms: {sorted(combined['algorithm'].unique())}")

    # Save combined
    output_file = f'{output_name}_results.feather'
    combined.to_feather(output_file)
    print(f'\nCombined results saved to: {output_file}')

    return combined

def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for alg in sorted(df['algorithm'].unique()):
        alg_df = df[df['algorithm'] == alg]
        print(f"\n{alg}:")
        print(f"  Datasets: {alg_df['dataset'].nunique()}")
        print(f"  Results: {len(alg_df)}")

        if 'r2_test' in alg_df.columns:
            print(f"  Mean R² (test): {alg_df['r2_test'].mean():.4f}")
            print(f"  Median R² (test): {alg_df['r2_test'].median():.4f}")
            high_acc = (alg_df['r2_test'] > 0.999).sum()
            print(f"  High accuracy (R² > 0.999): {high_acc} ({high_acc/len(alg_df)*100:.1f}%)")

        if 'symbolic_solution' in alg_df.columns:
            sym_sols = alg_df['symbolic_solution'].sum()
            print(f"  Symbolic solutions: {sym_sols} ({sym_sols/len(alg_df)*100:.1f}%)")

        if 'training time (hr)' in alg_df.columns:
            print(f"  Mean time: {alg_df['training time (hr)'].mean()*60:.1f} min")
            print(f"  Median time: {alg_df['training time (hr)'].median()*60:.1f} min")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python process_pysr_results.py <results_dir> [output_name]")
        print("\nExample:")
        print("  python process_pysr_results.py results_pysr_5min")
        print("  python process_pysr_results.py results_pysr_groundtruth pysr_full")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None

    # Collate results
    df = collate_results(results_dir, output_name)

    if df is not None:
        # Print summary
        print_summary(df)

        # Ask about merging with SRBench
        print("\n" + "="*80)
        merge = input("Merge with SRBench results? [y/N]: ").lower().strip()
        if merge == 'y':
            output_name = output_name or Path(results_dir).name
            combined = merge_with_srbench(df, f'combined_{output_name}')
            print_summary(combined)
