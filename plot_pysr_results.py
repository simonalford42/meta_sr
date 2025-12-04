#!/usr/bin/env python
"""
Generate SRBench-style plots for PySR results

Usage:
    python plot_pysr_results.py combined_results_pysr_5min_results.feather
    python plot_pysr_results.py combined_results_pysr_groundtruth_results.feather
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.3)

def plot_symbolic_solution_rates(df, output_dir='plots'):
    """Plot symbolic solution rates by algorithm and data group"""

    if 'symbolic_solution' not in df.columns:
        print("No symbolic_solution column found, skipping plot")
        return

    # Calculate solution rates
    solution_rates = df.groupby(['algorithm', 'data_group'])['symbolic_solution'].apply(
        lambda x: x.sum() / len(x) * 100
    ).reset_index()
    solution_rates.columns = ['algorithm', 'data_group', 'Symbolic Solution Rate (%)']

    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.catplot(
        data=solution_rates,
        x='algorithm',
        y='Symbolic Solution Rate (%)',
        hue='data_group',
        kind='point',
        height=6,
        aspect=2,
        markers=['o', 's'],
        linestyles=['-', '--'],
        capsize=0.1,
        errwidth=2
    )

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_file = Path(output_dir) / 'symbolic_solution_rates.png'
    output_file.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()

def plot_accuracy_solution_rates(df, output_dir='plots', threshold=0.999):
    """Plot accuracy solution rates (R² > threshold) by algorithm and data group"""

    if 'r2_test' not in df.columns:
        print("No r2_test column found, skipping plot")
        return

    # Calculate accuracy solution rates
    df_copy = df.copy()
    df_copy['accuracy_solution'] = df_copy['r2_test'] > threshold

    accuracy_rates = df_copy.groupby(['algorithm', 'data_group'])['accuracy_solution'].apply(
        lambda x: x.sum() / len(x) * 100
    ).reset_index()
    accuracy_rates.columns = ['algorithm', 'data_group', f'Accuracy Solution Rate (R²>{threshold}, %)']

    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.catplot(
        data=accuracy_rates,
        x='algorithm',
        y=f'Accuracy Solution Rate (R²>{threshold}, %)',
        hue='data_group',
        kind='point',
        height=6,
        aspect=2,
        markers=['o', 's'],
        linestyles=['-', '--'],
        capsize=0.1,
        errwidth=2
    )

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_file = Path(output_dir) / 'accuracy_solution_rates.png'
    output_file.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()

def plot_r2_distribution(df, output_dir='plots'):
    """Plot R² distribution by algorithm"""

    if 'r2_test' not in df.columns:
        print("No r2_test column found, skipping plot")
        return

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='algorithm', y='r2_test', hue='data_group')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('R² (test)')
    plt.title('R² Distribution by Algorithm and Data Group')
    plt.tight_layout()

    output_file = Path(output_dir) / 'r2_distribution.png'
    output_file.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()

def plot_training_time(df, output_dir='plots'):
    """Plot training time by algorithm"""

    if 'training time (hr)' not in df.columns:
        print("No training time column found, skipping plot")
        return

    # Convert to minutes for easier reading
    df_copy = df.copy()
    df_copy['training time (min)'] = df_copy['training time (hr)'] * 60

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_copy, x='algorithm', y='training time (min)', hue='data_group')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Training Time (minutes)')
    plt.yscale('log')
    plt.title('Training Time Distribution by Algorithm and Data Group')
    plt.tight_layout()

    output_file = Path(output_dir) / 'training_time.png'
    output_file.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()

def plot_complexity_vs_accuracy(df, output_dir='plots'):
    """Plot model complexity vs accuracy"""

    if 'model_size' not in df.columns or 'r2_test' not in df.columns:
        print("No model_size or r2_test column found, skipping plot")
        return

    plt.figure(figsize=(10, 8))
    for alg in df['algorithm'].unique():
        alg_df = df[df['algorithm'] == alg]
        plt.scatter(alg_df['model_size'], alg_df['r2_test'], alpha=0.6, label=alg, s=50)

    plt.xlabel('Model Size (complexity)')
    plt.ylabel('R² (test)')
    plt.title('Model Complexity vs Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = Path(output_dir) / 'complexity_vs_accuracy.png'
    output_file.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()

def generate_all_plots(feather_file, output_dir=None):
    """Generate all plots from a feather file"""

    if output_dir is None:
        output_dir = Path(feather_file).stem + '_plots'

    print(f"Loading results from: {feather_file}")
    df = pd.read_feather(feather_file)
    print(f"  Shape: {df.shape}")
    print(f"  Algorithms: {sorted(df['algorithm'].unique())}")
    print(f"  Datasets: {df['dataset'].nunique()}")

    print(f"\nGenerating plots in: {output_dir}")
    print("="*80)

    plot_symbolic_solution_rates(df, output_dir)
    plot_accuracy_solution_rates(df, output_dir)
    plot_r2_distribution(df, output_dir)
    plot_training_time(df, output_dir)
    plot_complexity_vs_accuracy(df, output_dir)

    print("="*80)
    print(f"All plots saved to: {output_dir}/")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_pysr_results.py <feather_file> [output_dir]")
        print("\nExample:")
        print("  python plot_pysr_results.py combined_results_pysr_5min_results.feather")
        print("  python plot_pysr_results.py combined_results_pysr_5min_results.feather my_plots")
        sys.exit(1)

    feather_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(feather_file).exists():
        print(f"ERROR: File not found: {feather_file}")
        sys.exit(1)

    generate_all_plots(feather_file, output_dir)
