"""
Evaluation module for symbolic regression.

Provides functions for:
- Converting Node expressions to sympy
- Comparing predicted expressions with ground truth symbolically (SRBench-style)
- Calculating R², symbolic match (0/1), and complexity
"""

import numpy as np
import sympy
from sympy import Symbol, simplify, Float, Integer, preorder_traversal
from sympy.parsing.sympy_parser import parse_expr
from sklearn.metrics import r2_score


# Mapping from operators to sympy equivalents
SYMPY_OPS = {
    '+': sympy.Add,
    '-': lambda x, y: sympy.Add(x, -y),
    '*': sympy.Mul,
    '/': lambda x, y: sympy.Mul(x, sympy.Pow(y, -1)),
    'abs': sympy.Abs,
    'exp': sympy.exp,
    'log': sympy.log,
    'sqrt': sympy.sqrt,
    'sin': sympy.sin,
    'cos': sympy.cos,
    'tan': sympy.tan,
    'inv': lambda x: sympy.Pow(x, -1),
    'pow2': lambda x: sympy.Pow(x, 2),
    'pow3': lambda x: sympy.Pow(x, 3),
}


def node_to_sympy(node, var_symbols=None):
    """
    Convert a Node expression tree to a sympy expression.

    Args:
        node: Node object from operators.py
        var_symbols: Optional dict of variable name -> Symbol. If None, creates them.

    Returns:
        sympy expression
    """
    if var_symbols is None:
        var_symbols = {}

    # Terminal: numeric constant
    if isinstance(node.value, (int, float)):
        return sympy.Float(node.value)

    # Terminal: variable (e.g., 'x0', 'x1', ...)
    if isinstance(node.value, str) and node.value.startswith('x'):
        if node.value not in var_symbols:
            var_symbols[node.value] = Symbol(node.value)
        return var_symbols[node.value]

    # Operator
    if node.value in SYMPY_OPS:
        op = SYMPY_OPS[node.value]
        if node.right is None:  # Unary
            return op(node_to_sympy(node.left, var_symbols))
        else:  # Binary
            return op(
                node_to_sympy(node.left, var_symbols),
                node_to_sympy(node.right, var_symbols)
            )

    # Unknown - return as symbol
    return Symbol(str(node.value))


def round_floats(expr, precision=3, zero_threshold=0.0001):
    """
    Round floating point numbers in a sympy expression.
    Numbers close to zero are set to 0.

    Based on SRBench's round_floats function.
    """
    result = expr
    for a in preorder_traversal(expr):
        if isinstance(a, Float):
            if abs(a) < zero_threshold:
                result = result.subs(a, Integer(0))
            else:
                result = result.subs(a, Float(round(float(a), precision), precision))
    return result


def complexity(expr):
    """
    Calculate complexity of a sympy expression (number of nodes in expression tree).

    Based on SRBench's complexity function.
    """
    c = 0
    for _ in preorder_traversal(expr):
        c += 1
    return c


def check_symbolic_match(predicted_expr, ground_truth_expr, n_vars=None):
    """
    Check if predicted expression symbolically matches ground truth.

    Uses the SRBench approach:
    - sym_diff = ground_truth - predicted: if this simplifies to 0 or a constant, it's a match
    - sym_frac = predicted / ground_truth: if this simplifies to a constant, it's a match

    Args:
        predicted_expr: sympy expression (predicted model)
        ground_truth_expr: sympy expression (ground truth)
        n_vars: Number of variables (optional, for creating variable symbols)

    Returns:
        dict with keys:
            - 'match': bool, True if expressions match symbolically
            - 'error_is_zero': bool, True if sym_diff == 0
            - 'error_is_constant': bool, True if sym_diff is a constant
            - 'fraction_is_constant': bool, True if sym_frac is a constant
            - 'simplified_predicted': str, simplified predicted expression
            - 'symbolic_error': str, the symbolic difference
    """
    result = {
        'match': False,
        'error_is_zero': False,
        'error_is_constant': False,
        'fraction_is_constant': False,
        'simplified_predicted': str(predicted_expr),
        'symbolic_error': None,
    }

    # Round floats in both expressions
    predicted_clean = round_floats(predicted_expr)
    ground_truth_clean = round_floats(ground_truth_expr)

    # Simplify predicted expression
    predicted_simplified = simplify(predicted_clean, ratio=1)
    result['simplified_predicted'] = str(predicted_simplified)

    # Calculate symbolic difference
    sym_diff = round_floats(ground_truth_clean - predicted_simplified)

    # Calculate symbolic fraction
    sym_frac = round_floats(predicted_simplified / ground_truth_clean)

    # Check if we can skip full simplification
    if not sym_diff.is_constant() or sym_frac.is_constant():
        sym_diff = round_floats(simplify(sym_diff, ratio=1))

    result['symbolic_error'] = str(sym_diff)

    # Check match conditions
    result['error_is_zero'] = str(sym_diff) == '0'
    result['error_is_constant'] = bool(sym_diff.is_constant())
    result['fraction_is_constant'] = bool(sym_frac.is_constant()) if sym_frac.is_constant() is not None else False

    # A match is any of the three conditions
    result['match'] = (
        result['error_is_zero'] or
        result['error_is_constant'] or
        result['fraction_is_constant']
    )

    return result


def parse_ground_truth(ground_truth_str, n_vars=10):
    """
    Parse a ground truth expression string to sympy.

    Args:
        ground_truth_str: String like 'x0**2 + x1'
        n_vars: Maximum number of variables to create symbols for

    Returns:
        sympy expression
    """
    local_dict = {f'x{i}': Symbol(f'x{i}') for i in range(n_vars)}
    return parse_expr(ground_truth_str, local_dict=local_dict)


def evaluate_model(predicted_node, X, y, ground_truth_str=None):
    """
    Comprehensive evaluation of a predicted model.

    Args:
        predicted_node: Node object (the predicted expression tree)
        X: Input data (n_samples, n_features)
        y: Target values (n_samples,)
        ground_truth_str: Optional ground truth expression string

    Returns:
        dict with:
            - 'r2': R² score
            - 'mse': Mean squared error
            - 'complexity': Expression complexity (node count)
            - 'symbolic_match': 0 or 1 (if ground_truth provided)
            - 'symbolic_details': dict with detailed match info (if ground_truth provided)
            - 'predicted_str': String representation of predicted expression
    """
    result = {
        'r2': None,
        'mse': None,
        'complexity': None,
        'symbolic_match': None,
        'symbolic_details': None,
        'predicted_str': str(predicted_node),
    }

    # Calculate predictions and R²/MSE
    try:
        y_pred = predicted_node.evaluate(X)
        if np.any(~np.isfinite(y_pred)):
            result['r2'] = -np.inf
            result['mse'] = np.inf
        else:
            result['r2'] = r2_score(y, y_pred)
            result['mse'] = float(np.mean((y - y_pred) ** 2))
    except Exception as e:
        result['r2'] = -np.inf
        result['mse'] = np.inf
        result['error'] = str(e)

    # Calculate complexity using sympy
    try:
        predicted_sympy = node_to_sympy(predicted_node)
        result['complexity'] = complexity(predicted_sympy)
    except Exception:
        # Fallback to node size
        result['complexity'] = predicted_node.size()

    # Check symbolic match if ground truth provided
    if ground_truth_str is not None:
        try:
            predicted_sympy = node_to_sympy(predicted_node)
            n_vars = X.shape[1]
            ground_truth_sympy = parse_ground_truth(ground_truth_str, n_vars)

            match_result = check_symbolic_match(predicted_sympy, ground_truth_sympy, n_vars)
            result['symbolic_match'] = 1 if match_result['match'] else 0
            result['symbolic_details'] = match_result
        except Exception as e:
            result['symbolic_match'] = 0
            result['symbolic_details'] = {'error': str(e)}

    return result


def evaluate_on_problem(predicted_node, problem_func, seed=42):
    """
    Evaluate a predicted model on a problem function.

    Args:
        predicted_node: Node object (the predicted expression tree)
        problem_func: Problem function from problems.py
        seed: Random seed for generating data

    Returns:
        dict with evaluation metrics (see evaluate_model)
    """
    from problems import get_ground_truth

    X, y = problem_func(seed)
    ground_truth_str = get_ground_truth(problem_func)

    return evaluate_model(predicted_node, X, y, ground_truth_str)


# ============================================================================
# PySR Results Evaluation
# ============================================================================

def parse_pysr_expression(expr_str, var_names=None):
    """
    Parse a PySR expression string to sympy.

    Args:
        expr_str: Expression string from PySR (e.g., '(6.28 * alpha) / (n * d)')
        var_names: List of variable names used in the expression

    Returns:
        sympy expression
    """
    # Create local dict with variable symbols
    local_dict = {}
    if var_names:
        for name in var_names:
            local_dict[name] = Symbol(name)

    # Also add common variable patterns
    for i in range(20):
        local_dict[f'x{i}'] = Symbol(f'x{i}')

    # Add common constants/functions that sympy understands
    local_dict['pi'] = sympy.pi
    local_dict['e'] = sympy.E
    local_dict['sqrt'] = sympy.sqrt
    local_dict['sin'] = sympy.sin
    local_dict['cos'] = sympy.cos
    local_dict['tan'] = sympy.tan
    local_dict['exp'] = sympy.exp
    local_dict['log'] = sympy.log
    local_dict['abs'] = sympy.Abs

    return parse_expr(expr_str, local_dict=local_dict)


def parse_ground_truth_formula(formula_str, var_names=None):
    """
    Parse a ground truth formula string (e.g., 'k = 2*pi*alpha/(n*d)') to sympy.

    Args:
        formula_str: Formula string, may include '=' for assignment
        var_names: List of variable names

    Returns:
        sympy expression (RHS of equation if '=' present)
    """
    # Extract RHS if formula has '='
    if '=' in formula_str:
        formula_str = formula_str.split('=', 1)[1].strip()

    return parse_pysr_expression(formula_str, var_names)


def load_pysr_hall_of_fame(csv_path):
    """
    Load PySR hall of fame CSV file.

    Args:
        csv_path: Path to hall_of_fame.csv

    Returns:
        List of dicts with keys: complexity, loss, equation
    """
    import pandas as pd
    from pathlib import Path

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    results = []
    for _, row in df.iterrows():
        results.append({
            'complexity': int(row['Complexity']),
            'loss': float(row['Loss']),
            'equation': row['Equation'],
        })

    return results


def load_pysr_checkpoint(checkpoint_path):
    """
    Load PySR model from checkpoint.pkl file.

    Args:
        checkpoint_path: Path to checkpoint.pkl

    Returns:
        dict with:
            - model: PySRRegressor object
            - equations: DataFrame with Pareto frontier
            - best_idx: Index of best equation (from model selection)
            - best_equation: Series with best equation info
            - feature_names: List of feature names
    """
    import pickle
    from pathlib import Path

    checkpoint_path = Path(checkpoint_path)
    with open(checkpoint_path, 'rb') as f:
        model = pickle.load(f)

    # Get equations DataFrame
    equations = model.equations_

    # Get best equation (based on model_selection strategy)
    best_eq = model.get_best()
    best_idx = best_eq.name  # The index in the DataFrame

    return {
        'model': model,
        'equations': equations,
        'best_idx': best_idx,
        'best_equation': best_eq,
        'feature_names': list(model.feature_names_in_),
    }


def get_dataset_var_names(dataset_name):
    """
    Get the variable names for a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'feynman_III_15_27')

    Returns:
        List of variable names (column names excluding 'target')
    """
    import pandas as pd
    from pathlib import Path

    pmlb_path = Path(__file__).parent / 'pmlb' / 'datasets'
    dataset_path = pmlb_path / dataset_name / f"{dataset_name}.tsv.gz"

    df = pd.read_csv(dataset_path, sep='\t', compression='gzip', nrows=1)
    return [col for col in df.columns if col != 'target']


def check_pysr_symbolic_match(expr_str, ground_truth_str, var_names=None, timeout_seconds=5):
    """
    Check if a PySR expression symbolically matches ground truth.

    Args:
        expr_str: PySR expression string
        ground_truth_str: Ground truth formula string
        var_names: List of variable names
        timeout_seconds: Timeout for symbolic simplification

    Returns:
        dict with match results (see check_symbolic_match)
    """
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Symbolic match timed out")

    try:
        predicted = parse_pysr_expression(expr_str, var_names)
        ground_truth = parse_ground_truth_formula(ground_truth_str, var_names)

        # Set timeout for complex expressions
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            result = check_symbolic_match(predicted, ground_truth)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        return result

    except TimeoutError:
        return {
            'match': False,
            'error': 'timeout',
            'simplified_predicted': expr_str,
            'symbolic_error': None,
        }


def evaluate_pysr_results(results_path, dataset_name=None, ground_truth_str=None, verbose=True):
    """
    Evaluate PySR results against ground truth.

    Checks:
    1. Best expression (from model selection) for symbolic match
    2. All Pareto frontier expressions for symbolic match

    Args:
        results_path: Path to checkpoint.pkl or hall_of_fame.csv or results directory
        dataset_name: Name of the dataset (for variable names). If None, inferred from checkpoint.
        ground_truth_str: Ground truth formula (if None, loaded from metadata)
        verbose: Print results

    Returns:
        dict with:
            - best_match: bool, whether best expression matches
            - any_match: bool, whether any expression matches
            - best_expr: dict with best expression info
            - matching_exprs: list of matching expressions
            - all_results: list of all expressions with match info
    """
    from utils import load_srbench_dataset
    from pathlib import Path

    results_path = Path(results_path)

    # Determine file type and load accordingly
    if results_path.is_dir():
        # Directory - look for checkpoint.pkl
        checkpoint_path = results_path / 'checkpoint.pkl'
        if checkpoint_path.exists():
            results_path = checkpoint_path
        else:
            # Fall back to hall_of_fame.csv
            results_path = results_path / 'hall_of_fame.csv'

    if results_path.suffix == '.pkl':
        # Load from checkpoint
        checkpoint = load_pysr_checkpoint(results_path)
        var_names = checkpoint['feature_names']
        equations_df = checkpoint['equations']
        best_idx = checkpoint['best_idx']

        # Convert DataFrame to list of dicts
        hof = []
        for idx, row in equations_df.iterrows():
            hof.append({
                'complexity': int(row['complexity']),
                'loss': float(row['loss']),
                'score': float(row['score']),
                'equation': row['equation'],
                'sympy_format': row['sympy_format'],
                'df_index': idx,
            })

        # Find best index in hof list
        best_hof_idx = next(i for i, h in enumerate(hof) if h['df_index'] == best_idx)

    else:
        # Load from CSV (legacy)
        hof = load_pysr_hall_of_fame(results_path)
        var_names = get_dataset_var_names(dataset_name) if dataset_name else None
        # For CSV, best is lowest loss
        best_hof_idx = min(range(len(hof)), key=lambda i: hof[i]['loss'])

    # Get ground truth if not provided
    if ground_truth_str is None and dataset_name:
        _, _, ground_truth_str = load_srbench_dataset(dataset_name, max_samples=10)

    if verbose:
        print(f"Dataset: {dataset_name}")
        print(f"Ground truth: {ground_truth_str}")
        print(f"Variables: {var_names}")
        print(f"Expressions in Pareto frontier: {len(hof)}")
        print("-" * 60)

    # Check all expressions
    all_results = []
    matching_exprs = []

    for i, expr_info in enumerate(hof):
        # Use sympy_format if available (from checkpoint), otherwise parse equation
        if 'sympy_format' in expr_info and expr_info['sympy_format'] is not None:
            # sympy_format is already a sympy expression
            try:
                predicted = expr_info['sympy_format']
                ground_truth = parse_ground_truth_formula(ground_truth_str, var_names)
                match_result = check_symbolic_match(predicted, ground_truth)
            except Exception as e:
                match_result = {'match': False, 'error': str(e)}
        else:
            match_result = check_pysr_symbolic_match(
                expr_info['equation'],
                ground_truth_str,
                var_names
            )

        result = {
            **expr_info,
            'is_best': (i == best_hof_idx),
            'match_result': match_result,
        }
        all_results.append(result)

        if match_result.get('match'):
            matching_exprs.append(result)

    # Check if best matches
    best_match = all_results[best_hof_idx]['match_result'].get('match', False)
    any_match = len(matching_exprs) > 0
    best_expr = hof[best_hof_idx]

    if verbose:
        print(f"\nBest expression (model selection, complexity={best_expr['complexity']}):")
        print(f"  {best_expr['equation']}")
        if 'sympy_format' in best_expr:
            print(f"  Simplified: {best_expr['sympy_format']}")
        print(f"  Symbolic match: {best_match}")

        if matching_exprs:
            print(f"\nMatching expressions ({len(matching_exprs)} found):")
            for expr in matching_exprs:
                print(f"  [complexity={expr['complexity']}, loss={expr['loss']:.2e}]")
                print(f"    {expr['equation']}")
                if 'sympy_format' in expr:
                    print(f"    Simplified: {expr['sympy_format']}")
        else:
            print(f"\nNo symbolic matches found in Pareto frontier.")

        print("-" * 60)

    return {
        'dataset': dataset_name,
        'ground_truth': ground_truth_str,
        'best_match': best_match,
        'any_match': any_match,
        'best_expr': all_results[best_hof_idx],
        'matching_exprs': matching_exprs,
        'all_results': all_results,
    }


def main():
    """CLI for evaluating PySR results."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate PySR results against ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate from checkpoint (preferred)
  python evaluation.py results_pysr/20251220_210555_bodZT6/checkpoint.pkl feynman_III_15_27

  # Evaluate from results directory (auto-finds checkpoint.pkl)
  python evaluation.py results_pysr/20251220_210555_bodZT6 feynman_III_15_27

  # Evaluate from CSV (legacy)
  python evaluation.py results_pysr/20251220_210555_bodZT6/hall_of_fame.csv feynman_III_15_27

  # With explicit ground truth
  python evaluation.py checkpoint.pkl feynman_I_29_16 --ground-truth "sqrt(x0**2 + x1**2)"
        """
    )

    parser.add_argument('results_path', type=str,
                       help='Path to checkpoint.pkl, hall_of_fame.csv, or results directory')
    parser.add_argument('dataset', type=str,
                       help='Dataset name (e.g., feynman_III_15_27)')
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Ground truth formula (optional, loaded from metadata if not provided)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Minimal output')

    args = parser.parse_args()

    result = evaluate_pysr_results(
        args.results_path,
        args.dataset,
        ground_truth_str=args.ground_truth,
        verbose=not args.quiet,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Best expression matches ground truth: {result['best_match']}")
    print(f"Any expression matches ground truth:  {result['any_match']}")

    return 0 if result['any_match'] else 1


def check_12_20_match():
    """
    Check all results in results_pysr/12_20/ and count symbolic matches.
    """
    import json
    from pathlib import Path
    from utils import load_srbench_dataset

    results_dir = Path(__file__).parent / 'results_pysr' / '12_20'
    json_files = list(results_dir.glob('*_results.json'))

    matches = 0
    total = 0
    results = []

    for json_file in sorted(json_files):
        with open(json_file) as f:
            data = json.load(f)

        dataset = data['dataset']
        best_eq = data['best_equation']

        # Get ground truth and variable names
        try:
            _, _, ground_truth = load_srbench_dataset(dataset, max_samples=10)
            var_names = get_dataset_var_names(dataset)
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
            continue

        # Check symbolic match
        match_result = check_pysr_symbolic_match(best_eq, ground_truth, var_names)
        is_match = match_result.get('match', False)

        if is_match:
            matches += 1
        total += 1

        results.append({
            'dataset': dataset,
            'best_equation': best_eq,
            'ground_truth': ground_truth,
            'match': is_match,
        })

        status = '✓' if is_match else '✗'
        print(f"{status} {dataset}: {best_eq}")

    print(f"\n{'='*60}")
    print(f"Symbolic matches: {matches}/{total} ({100*matches/total:.1f}%)")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    exit(main())
