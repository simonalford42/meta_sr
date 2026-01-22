"""
Plot complexity comparison between ground truth and discovered expressions.

For each result directory (1e3 to 1e8), plots the percentage of problems where:
discovered_complexity <= ground_truth_complexity + i

where i ranges from 0 to some maximum value on the x-axis.
"""

import json
import yaml
import re
import sympy
from sympy import Symbol, preorder_traversal
from sympy.parsing.sympy_parser import parse_expr
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def complexity(expr):
    """
    Calculate complexity of a sympy expression (number of nodes in expression tree).
    Based on SRBench's complexity function.
    """
    c = 0
    for _ in preorder_traversal(expr):
        c += 1
    return c


def parse_formula(formula_str, var_names=None):
    """
    Parse a formula string to sympy expression.

    Args:
        formula_str: Formula string, may include '=' for assignment
        var_names: List of variable names

    Returns:
        sympy expression (RHS of equation if '=' present)
    """
    # Extract RHS if formula has '='
    if '=' in formula_str:
        formula_str = formula_str.split('=', 1)[1].strip()

    # Create local dict with variable symbols
    local_dict = {}
    if var_names:
        for name in var_names:
            local_dict[name] = Symbol(name)

    # Also add x0, x1, etc.
    for i in range(20):
        local_dict[f'x{i}'] = Symbol(f'x{i}')

    # Add common functions
    local_dict['pi'] = sympy.pi
    local_dict['e'] = sympy.E
    local_dict['sqrt'] = sympy.sqrt
    local_dict['sin'] = sympy.sin
    local_dict['cos'] = sympy.cos
    local_dict['tan'] = sympy.tan
    local_dict['exp'] = sympy.exp
    local_dict['log'] = sympy.log
    local_dict['abs'] = sympy.Abs
    local_dict['tanh'] = sympy.tanh
    local_dict['cosh'] = sympy.cosh
    local_dict['sinh'] = sympy.sinh
    local_dict['arcsin'] = sympy.asin
    local_dict['arccos'] = sympy.acos
    local_dict['arctan'] = sympy.atan
    local_dict['ln'] = sympy.log

    return parse_expr(formula_str, local_dict=local_dict)


def extract_formula_from_metadata(metadata_path):
    """Extract the formula from a metadata.yaml file."""
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    description = metadata.get('description', '')

    # Extract formula line (looks for pattern like "var = expression")
    lines = description.split('\n')
    for line in lines:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            # Check if it looks like a formula (has operators or parentheses)
            rhs = line.split('=', 1)[1].strip()
            if any(c in rhs for c in ['+', '-', '*', '/', '(', '^', '**']):
                return line
            # Also accept simple variable formulas like "F = m*a"
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+', line):
                return line

    return None


def get_var_names_from_metadata(metadata_path):
    """Extract variable names from metadata.yaml."""
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    features = metadata.get('features', [])
    return [f['name'] for f in features if isinstance(f, dict) and 'name' in f]


def load_results_with_complexity(results_dir, pmlb_path):
    """
    Load results from a directory and calculate complexities.

    Returns:
        List of dicts with dataset, ground_truth_complexity, discovered_complexity
    """
    results_dir = Path(results_dir)
    pmlb_path = Path(pmlb_path)

    results = []

    for json_file in sorted(results_dir.glob('*_results.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)

            dataset = data.get('dataset', '')
            best_eq = data.get('best_equation', '')

            if not dataset or not best_eq:
                continue

            # Get metadata path
            metadata_path = pmlb_path / 'datasets' / dataset / 'metadata.yaml'
            if not metadata_path.exists():
                continue

            # Get ground truth formula
            gt_formula = extract_formula_from_metadata(metadata_path)
            if not gt_formula:
                continue

            # Get variable names
            var_names = get_var_names_from_metadata(metadata_path)

            # Calculate ground truth complexity
            try:
                gt_expr = parse_formula(gt_formula, var_names)
                gt_complexity = complexity(gt_expr)
            except Exception as e:
                print(f"Error parsing ground truth for {dataset}: {e}")
                continue

            # Calculate discovered complexity
            try:
                discovered_expr = parse_formula(best_eq, var_names)
                discovered_complexity = complexity(discovered_expr)
            except Exception as e:
                print(f"Error parsing discovered eq for {dataset}: {e}")
                continue

            results.append({
                'dataset': dataset,
                'ground_truth_formula': gt_formula,
                'discovered_formula': best_eq,
                'ground_truth_complexity': gt_complexity,
                'discovered_complexity': discovered_complexity,
                'test_r2': data.get('test_r2', 0),
            })

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    return results


def plot_complexity_gap_cdf(all_results, output_path=None, max_gap=40):
    """
    Plot the percentage of problems where:
    discovered_complexity <= ground_truth_complexity + i

    Args:
        all_results: Dict mapping label (e.g., '1e3') to list of result dicts
        output_path: Path to save the plot
        max_gap: Maximum gap value on x-axis
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_results)))

    x_values = list(range(0, max_gap + 1))

    for (label, results), color in zip(all_results.items(), colors):
        if not results:
            continue

        n_total = len(results)

        # Calculate gap for each problem: discovered - ground_truth
        gaps = [r['discovered_complexity'] - r['ground_truth_complexity'] for r in results]

        # For each x value, count % where gap <= x
        percentages = []
        for x in x_values:
            count = sum(1 for g in gaps if g <= x)
            pct = 100.0 * count / n_total
            percentages.append(pct)

        ax.plot(x_values, percentages, 'o-', label=f'{label} (n={n_total})',
                color=color, markersize=4, linewidth=2)

    ax.set_xlabel('Complexity Gap (i): Discovered - Ground Truth', fontsize=12)
    ax.set_ylabel('% of Problems with Gap ≤ i', fontsize=12)
    ax.set_title('Discovered Expression Complexity vs Ground Truth Complexity', fontsize=14)
    ax.legend(title='Iterations', loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, max_gap)
    ax.set_ylim(0, 105)

    # Add vertical line at x=0 (exact complexity match)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.5, 50, 'Discovered ≤ Ground Truth', rotation=90, va='center', ha='left',
            color='gray', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    return fig, ax


def print_complexity_stats(all_results):
    """Print summary statistics for each result set."""
    print("\n" + "=" * 80)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\n{'Iterations':<12} {'N':>5} {'GT Mean':>8} {'Disc Mean':>10} {'Gap Mean':>9} {'Gap<=0':>8} {'Gap<=5':>8} {'Gap<=10':>9}")
    print("-" * 80)

    for label, results in all_results.items():
        if not results:
            continue

        n = len(results)
        gt_complexities = [r['ground_truth_complexity'] for r in results]
        disc_complexities = [r['discovered_complexity'] for r in results]
        gaps = [r['discovered_complexity'] - r['ground_truth_complexity'] for r in results]

        gt_mean = np.mean(gt_complexities)
        disc_mean = np.mean(disc_complexities)
        gap_mean = np.mean(gaps)

        pct_gap_le_0 = 100.0 * sum(1 for g in gaps if g <= 0) / n
        pct_gap_le_5 = 100.0 * sum(1 for g in gaps if g <= 5) / n
        pct_gap_le_10 = 100.0 * sum(1 for g in gaps if g <= 10) / n

        print(f"{label:<12} {n:>5} {gt_mean:>8.1f} {disc_mean:>10.1f} {gap_mean:>9.1f} {pct_gap_le_0:>7.1f}% {pct_gap_le_5:>7.1f}% {pct_gap_le_10:>8.1f}%")

    print("=" * 80)


def main():
    base_dir = Path('/home/sca63/meta_sr')
    pmlb_path = base_dir / 'pmlb'

    # Result directories to analyze
    result_dirs = [
        ('1e3', base_dir / 'results_pysr_1e3'),
        ('1e4', base_dir / 'results_pysr_1e4'),
        ('1e5', base_dir / 'results_pysr_1e5'),
        ('1e6', base_dir / 'results_pysr_1e6'),
        ('1e7', base_dir / 'results_pysr_1e7'),
        ('1e8', base_dir / 'results_pysr_1e8'),
    ]

    # Load all results
    all_results = {}
    for label, results_dir in result_dirs:
        if not results_dir.exists():
            print(f"Warning: {results_dir} not found, skipping")
            continue

        print(f"Loading {label}...")
        results = load_results_with_complexity(results_dir, pmlb_path)
        all_results[label] = results
        print(f"  Loaded {len(results)} problems")

    # Print statistics
    print_complexity_stats(all_results)

    # Create plot
    plot_complexity_gap_cdf(all_results, output_path=base_dir / 'pysr_complexity_gap.png')

    # Also save detailed results for inspection
    print("\nSample results (first 5 from 1e3):")
    if '1e3' in all_results:
        for r in all_results['1e3'][:5]:
            print(f"  {r['dataset']}: GT={r['ground_truth_complexity']}, Disc={r['discovered_complexity']}, Gap={r['discovered_complexity']-r['ground_truth_complexity']}")


if __name__ == '__main__':
    main()
