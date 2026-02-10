"""
Evaluation module for symbolic regression.

Provides functions for:
- Converting Node expressions to sympy
- Comparing predicted expressions with ground truth symbolically (SRBench-style)
- Calculating R², symbolic match (0/1), and complexity
"""

import os
import json
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

    # Calculate symbolic difference (and simplify)
    sym_diff = round_floats(ground_truth_clean - predicted_simplified)
    sym_diff = round_floats(simplify(sym_diff, ratio=1))

    # Calculate symbolic fraction only if ground truth is not zero
    sym_frac = None
    ground_truth_is_zero = ground_truth_clean.equals(0)
    if not ground_truth_is_zero:
        sym_frac = round_floats(predicted_simplified / ground_truth_clean)
        sym_frac = round_floats(simplify(sym_frac, ratio=1))

    result['symbolic_error'] = str(sym_diff)

    # Check match conditions
    result['error_is_zero'] = bool(sym_diff.equals(0))
    result['error_is_constant'] = bool(sym_diff.is_constant()) if sym_diff.is_constant() is not None else False
    result['fraction_is_constant'] = (
        bool(sym_frac.is_constant()) if sym_frac is not None and sym_frac.is_constant() is not None else False
    )

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

    # Add common constants/functions that sympy understands, plus common
    # PySR operator aliases (e.g., square) so equivalent forms simplify.
    local_dict['pi'] = sympy.pi
    local_dict['e'] = sympy.E
    local_dict['sqrt'] = sympy.sqrt
    local_dict['sin'] = sympy.sin
    local_dict['cos'] = sympy.cos
    local_dict['tan'] = sympy.tan
    local_dict['exp'] = sympy.exp
    local_dict['log'] = sympy.log
    local_dict['abs'] = sympy.Abs
    local_dict['square'] = lambda x: x**2

    # Extra aliases/common custom ops for robustness across runs.
    local_dict['inv'] = lambda x: 1 / x
    local_dict['pow2'] = lambda x: x**2
    local_dict['pow3'] = lambda x: x**3
    local_dict['cube'] = lambda x: x**3
    local_dict['neg'] = lambda x: -x
    local_dict['sign'] = sympy.sign
    local_dict['relu'] = lambda x: sympy.Max(0, x)
    local_dict['max'] = sympy.Max
    local_dict['min'] = sympy.Min
    local_dict['sinh'] = sympy.sinh
    local_dict['cosh'] = sympy.cosh
    local_dict['tanh'] = sympy.tanh
    local_dict['asin'] = sympy.asin
    local_dict['acos'] = sympy.acos
    local_dict['atan'] = sympy.atan
    local_dict['floor'] = sympy.floor
    local_dict['ceil'] = sympy.ceiling
    local_dict['heaviside'] = sympy.Heaviside
    local_dict['step'] = sympy.Heaviside
    local_dict['sigmoid'] = lambda x: 1 / (1 + sympy.exp(-x))
    local_dict['logistic'] = lambda x: 1 / (1 + sympy.exp(-x))

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


def check_sympy_equivalence_with_llm(
    predicted_expr,
    ground_truth_expr,
    model: str = "openai/gpt-5.2",
    thinking_level: str = "high",
    max_tokens: int = 300,
    use_cache: bool = True,
):
    """
    Ask an LLM whether two sympy expressions are equivalent.

    The decision criterion mirrors SRBench-style symbolic matching:
    expressions are considered equivalent if they are:
      1) exactly equal,
      2) equal up to additive constant, or
      3) equal up to multiplicative constant.

    Args:
        predicted_expr: sympy expression (or string)
        ground_truth_expr: sympy expression (or string)
        model: LLM model identifier
        thinking_level: reasoning effort level passed to API
        max_tokens: max output tokens
        use_cache: whether to use completion cache

    Returns:
        dict with:
            - llm_match: bool
            - raw_response: str
            - reasoning: str
            - model: str
            - error: Optional[str]
    """
    from completions import chat_completion, get_content

    predicted_str = str(predicted_expr)
    ground_truth_str = str(ground_truth_expr)

    system_prompt = (
        "You are a rigorous symbolic math equivalence checker. "
        "Given two expressions, decide if they are equivalent under ANY of: "
        "(a) exact algebraic equality, "
        "(b) differ by an additive constant only, "
        "(c) differ by a multiplicative constant only. "
        "Return strict JSON only."
    )

    user_prompt = (
        "Determine equivalence under the criteria above.\n"
        f"Expression A: {predicted_str}\n"
        f"Expression B: {ground_truth_str}\n\n"
        "Output JSON with keys:\n"
        "  equivalent: true/false\n"
        "  rationale: short string"
    )

    try:
        response = chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            use_cache=use_cache,
            include_default_reasoning=False,
            reasoning={"effort": thinking_level},
        )
        content = get_content(response).strip()

        llm_match = None
        rationale = ""

        # Parse strict JSON first.
        try:
            parsed = json.loads(content)
            llm_match = bool(parsed.get("equivalent", False))
            rationale = str(parsed.get("rationale", ""))
        except Exception:
            # Fallback: minimal robust parse.
            low = content.lower()
            if '"equivalent": true' in low or "equivalent: true" in low:
                llm_match = True
            elif '"equivalent": false' in low or "equivalent: false" in low:
                llm_match = False
            elif low.startswith("true"):
                llm_match = True
            elif low.startswith("false"):
                llm_match = False
            else:
                return {
                    "llm_match": False,
                    "raw_response": content,
                    "reasoning": "",
                    "model": model,
                    "error": "Could not parse LLM equivalence response",
                }

        return {
            "llm_match": bool(llm_match),
            "raw_response": content,
            "reasoning": rationale,
            "model": model,
            "effort": thinking_level,
            "error": None,
        }
    except Exception as e:
        return {
            "llm_match": False,
            "raw_response": "",
            "reasoning": "",
            "model": model,
            "effort": thinking_level,
            "error": str(e),
        }


def check_pysr_symbolic_match_with_llm(
    expr_str,
    ground_truth_str,
    var_names=None,
    timeout_seconds: int = 10,
    llm_model: str = "openai/gpt-5.2",
    llm_thinking_level: str = "high",
    llm_max_tokens: int = 300,
    raise_on_sympy_llm_disagreement: bool = True,
    llm_use_cache: bool = True,
):
    """
    Combined symbolic match check:
      1) Sympy check (with timeout),
      2) LLM equivalence check,
      3) return match = sympy_match OR llm_match.

    Safety guard:
      If sympy_match is True but llm_match is False, raise RuntimeError
      (unless raise_on_sympy_llm_disagreement=False).

    Returns:
        dict with:
            - match: bool  (sympy OR llm)
            - sympy_match: bool
            - llm_match: bool
            - sympy_result: dict
            - llm_result: dict
    """
    # 1) Sympy check first.
    sympy_result = check_pysr_symbolic_match(
        expr_str,
        ground_truth_str,
        var_names=var_names,
        timeout_seconds=timeout_seconds,
    )
    sympy_match = bool(sympy_result.get("match", False))

    # 2) LLM check.
    predicted = parse_pysr_expression(expr_str, var_names)
    ground_truth = parse_ground_truth_formula(ground_truth_str, var_names)
    llm_result = check_sympy_equivalence_with_llm(
        predicted_expr=predicted,
        ground_truth_expr=ground_truth,
        model=llm_model,
        thinking_level=llm_thinking_level,
        max_tokens=llm_max_tokens,
        use_cache=llm_use_cache,
    )
    llm_match = bool(llm_result.get("llm_match", False))

    # 3) Guard unexpected disagreement.
    if sympy_match and not llm_match and raise_on_sympy_llm_disagreement:
        raise RuntimeError(
            "Unexpected disagreement: sympy_match=True but llm_match=False "
            f"for expr='{expr_str}' vs ground_truth='{ground_truth_str}'. "
            f"sympy_result={sympy_result}, llm_result={llm_result}"
        )

    return {
        "match": bool(sympy_match or llm_match),
        "sympy_match": sympy_match,
        "llm_match": llm_match,
        "sympy_result": sympy_result,
        "llm_result": llm_result,
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
    """CLI for evaluating SR/PySR results."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Evaluate symbolic regression results against ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check SR (BasicSR) results directory for symbolic matches
  python evaluation.py --sr results_sr/

  # Compare baseline vs evolved operators
  python evaluation.py --compare results_sr/ results_sr_run_20241220/

  # Evaluate PySR results (legacy mode)
  python evaluation.py results_pysr/checkpoint.pkl feynman_III_15_27

  # Check PySR 12_20 results
  python evaluation.py --pysr-12-20
        """
    )

    # Mode selection
    parser.add_argument('--sr', type=str, metavar='DIR',
                       help='Check SR results directory for symbolic matches')
    parser.add_argument('--compare', nargs=2, metavar=('BASELINE', 'EVOLVED'),
                       help='Compare baseline vs evolved SR results')
    parser.add_argument('--pysr-12-20', action='store_true',
                       help='Check PySR 12_20 results (legacy)')

    # PySR evaluation (legacy positional args)
    parser.add_argument('results_path', type=str, nargs='?',
                       help='Path to PySR checkpoint.pkl or results directory')
    parser.add_argument('dataset', type=str, nargs='?',
                       help='Dataset name for PySR evaluation')
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Ground truth formula (optional)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Minimal output')

    args = parser.parse_args()

    # SR results checking mode
    if args.sr:
        result = check_sr_results(args.sr, verbose=not args.quiet)
        return 0 if result['matches'] > 0 else 1

    # Compare mode
    if args.compare:
        compare_sr_results(args.compare[0], args.compare[1], verbose=not args.quiet)
        return 0

    # PySR 12_20 mode
    if args.pysr_12_20:
        check_12_20_match()
        return 0

    # Legacy PySR evaluation mode
    if args.results_path and args.dataset:
        result = evaluate_pysr_results(
            args.results_path,
            args.dataset,
            ground_truth_str=args.ground_truth,
            verbose=not args.quiet,
        )

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Best expression matches ground truth: {result['best_match']}")
        print(f"Any expression matches ground truth:  {result['any_match']}")

        return 0 if result['any_match'] else 1

    # No valid mode selected
    parser.print_help()
    return 1


def check_12_20_match():
    """
    Check all results in results_pysr/12_20/ and count symbolic matches.
    """
    import json
    from pathlib import Path
    from utils import load_srbench_dataset

    results_dir = Path(__file__).parent / 'results_pysr'
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


# ============================================================================
# SR (BasicSR) Results Evaluation
# ============================================================================

def parse_sr_expression(expr_str, n_vars=10):
    """
    Parse a BasicSR Node expression string to sympy.

    Handles expressions like:
    - "(x0 + x1)"
    - "sin((x0 * x1))"
    - "((x0 + 1.5) * cos(x1))"

    Args:
        expr_str: Expression string from BasicSR Node.__str__()
        n_vars: Maximum number of variables

    Returns:
        sympy expression
    """
    # Create local dict with variable symbols
    local_dict = {f'x{i}': Symbol(f'x{i}') for i in range(n_vars)}

    # Add common functions
    local_dict['sin'] = sympy.sin
    local_dict['cos'] = sympy.cos
    local_dict['tan'] = sympy.tan
    local_dict['exp'] = sympy.exp
    local_dict['log'] = sympy.log
    local_dict['sqrt'] = sympy.sqrt
    local_dict['abs'] = sympy.Abs
    local_dict['square'] = lambda x: x**2
    local_dict['inv'] = lambda x: 1/x
    local_dict['pow2'] = lambda x: x**2
    local_dict['pow3'] = lambda x: x**3
    local_dict['pi'] = sympy.pi
    local_dict['e'] = sympy.E

    # Replace ^ with ** for exponentiation
    expr_str = expr_str.replace('^', '**')

    return parse_expr(expr_str, local_dict=local_dict)


def check_sr_symbolic_match(expr_str, ground_truth_str, n_vars=10, var_names=None, timeout_seconds=5):
    """
    Check if a BasicSR expression symbolically matches ground truth.

    Args:
        expr_str: BasicSR expression string
        ground_truth_str: Ground truth formula string
        n_vars: Number of variables
        var_names: List of variable names in order (e.g., ['n', 'h'] -> x0=n, x1=h).
                   If provided, substitutes ground truth variable names with x0, x1, etc.
        timeout_seconds: Timeout for symbolic simplification

    Returns:
        dict with match results (see check_symbolic_match)
    """
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Symbolic match timed out")

    try:
        predicted = parse_sr_expression(expr_str, n_vars)
        ground_truth = parse_ground_truth_formula(ground_truth_str, var_names)

        # If var_names provided, substitute original variable names with x0, x1, etc.
        if var_names:
            subs_dict = {Symbol(name): Symbol(f'x{i}') for i, name in enumerate(var_names)}
            ground_truth = ground_truth.subs(subs_dict)

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
    except Exception as e:
        return {
            'match': False,
            'error': str(e),
            'simplified_predicted': expr_str,
            'symbolic_error': None,
        }


def check_sr_results(results_dir, verbose=True):
    """
    Check all results in an SR results directory and count symbolic matches.

    Similar to check_12_20_match() but for BasicSR results.

    Args:
        results_dir: Path to directory containing *_results.json files
        verbose: Print per-dataset results

    Returns:
        dict with:
            - matches: number of symbolic matches
            - total: total number of datasets evaluated
            - match_rate: percentage of matches
            - results: list of per-dataset results
    """
    import json
    from pathlib import Path
    from utils import load_srbench_dataset

    results_dir = Path(results_dir)
    json_files = list(results_dir.glob('*_results.json'))

    if not json_files:
        print(f"No result files found in {results_dir}")
        return {'matches': 0, 'total': 0, 'match_rate': 0.0, 'results': []}

    matches = 0
    total = 0
    results = []

    for json_file in sorted(json_files):
        with open(json_file) as f:
            data = json.load(f)

        dataset = data['dataset']
        best_eq = data['best_equation']

        # Get ground truth
        ground_truth = data.get('ground_truth', '')
        if not ground_truth:
            try:
                _, _, ground_truth = load_srbench_dataset(dataset, max_samples=10)
            except Exception as e:
                if verbose:
                    print(f"Error loading {dataset}: {e}")
                continue

        if not ground_truth:
            if verbose:
                print(f"? {dataset}: No ground truth available")
            continue

        # Get number of features and variable names
        n_vars = data.get('n_features', 10)

        # Get actual variable names from dataset to map x0, x1, ... correctly
        try:
            var_names = get_dataset_var_names(dataset)
        except Exception:
            var_names = None

        # Check symbolic match
        match_result = check_sr_symbolic_match(best_eq, ground_truth, n_vars, var_names=var_names)
        is_match = match_result.get('match', False)

        if is_match:
            matches += 1
        total += 1

        results.append({
            'dataset': dataset,
            'best_equation': best_eq,
            'ground_truth': ground_truth,
            'match': is_match,
            'test_r2': data.get('test_r2'),
            'test_mse': data.get('test_mse'),
        })

        if verbose:
            status = 'Y' if is_match else 'X'
            r2 = data.get('test_r2', 0)
            print(f"{status} {dataset}: R2={r2:.4f} | {best_eq}")

    match_rate = 100 * matches / total if total > 0 else 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"Symbolic matches: {matches}/{total} ({match_rate:.1f}%)")
        print(f"{'='*60}")

    return {
        'matches': matches,
        'total': total,
        'match_rate': match_rate,
        'results': results
    }


def compare_sr_results(baseline_dir, evolved_dir, verbose=True):
    """
    Compare SR results between baseline and evolved operators.

    Args:
        baseline_dir: Path to baseline results (default operators)
        evolved_dir: Path to evolved results (meta-evolution operators)
        verbose: Print detailed comparison

    Returns:
        dict with comparison statistics
    """
    baseline = check_sr_results(baseline_dir, verbose=False)
    evolved = check_sr_results(evolved_dir, verbose=False)

    # Create lookup by dataset
    baseline_by_dataset = {r['dataset']: r for r in baseline['results']}
    evolved_by_dataset = {r['dataset']: r for r in evolved['results']}

    # Find common datasets
    common = set(baseline_by_dataset.keys()) & set(evolved_by_dataset.keys())

    if verbose:
        print(f"{'='*80}")
        print(f"Comparison: Baseline vs Evolved Operators")
        print(f"{'='*80}")
        print(f"Baseline results: {baseline_dir}")
        print(f"Evolved results:  {evolved_dir}")
        print(f"Common datasets:  {len(common)}")
        print()

        # Per-dataset comparison
        improved = 0
        degraded = 0
        unchanged = 0

        print(f"{'Dataset':<30} {'Baseline':<12} {'Evolved':<12} {'Change':<10}")
        print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*10}")

        for dataset in sorted(common):
            b = baseline_by_dataset[dataset]
            e = evolved_by_dataset[dataset]

            b_match = 'Y' if b['match'] else 'X'
            e_match = 'Y' if e['match'] else 'X'

            b_r2 = b.get('test_r2', 0)
            e_r2 = e.get('test_r2', 0)

            if e['match'] and not b['match']:
                change = "+MATCH"
                improved += 1
            elif b['match'] and not e['match']:
                change = "-MATCH"
                degraded += 1
            elif e_r2 > b_r2 + 0.01:
                change = f"+{e_r2 - b_r2:.3f}"
                improved += 1
            elif b_r2 > e_r2 + 0.01:
                change = f"{e_r2 - b_r2:.3f}"
                degraded += 1
            else:
                change = "~"
                unchanged += 1

            print(f"{dataset:<30} {b_match} R2={b_r2:.4f}  {e_match} R2={e_r2:.4f}  {change}")

        print()
        print(f"{'='*80}")
        print(f"Summary:")
        print(f"  Baseline: {baseline['matches']}/{baseline['total']} matches ({baseline['match_rate']:.1f}%)")
        print(f"  Evolved:  {evolved['matches']}/{evolved['total']} matches ({evolved['match_rate']:.1f}%)")
        print(f"  Improved: {improved}, Degraded: {degraded}, Unchanged: {unchanged}")
        print(f"{'='*80}")

    return {
        'baseline': baseline,
        'evolved': evolved,
        'common_datasets': len(common),
    }


def compare_seeds(dir1, dir2, r2_threshold=0.999, use_symbolic=True, verbose=True):
    """
    Compare SR results between two different seed runs.

    Analyzes:
    1. R² score variance between runs
    2. Solve rate (both R² threshold and symbolic matching)
    3. Which tasks are solved by both, one, or neither

    Args:
        dir1: Path to first results directory
        dir2: Path to second results directory
        r2_threshold: R² threshold for considering a task "solved" (default 0.999)
        use_symbolic: Also check symbolic matches (default True)
        verbose: Print detailed results

    Returns:
        dict with:
            - r2_stats: dict with mean/median/std/max/min of |Δ R²|
            - solve_rate: dict with solve counts for each directory and overlaps
            - per_task: list of per-task comparison results
    """
    import json
    from pathlib import Path

    dir1 = Path(dir1)
    dir2 = Path(dir2)

    def load_results(dir_path):
        results = {}
        for f in dir_path.glob('*_results.json'):
            with open(f) as fp:
                data = json.load(fp)
                task = data['dataset']
                results[task] = data
        return results

    results1 = load_results(dir1)
    results2 = load_results(dir2)

    # Find common tasks
    common_tasks = set(results1.keys()) & set(results2.keys())

    if verbose:
        print(f"{'='*70}")
        print(f"Seed Comparison: {dir1.name} vs {dir2.name}")
        print(f"{'='*70}")
        print(f"Tasks in {dir1.name}: {len(results1)}")
        print(f"Tasks in {dir2.name}: {len(results2)}")
        print(f"Common tasks: {len(common_tasks)}")

    # Collect R² differences and per-task results
    r2_diffs = []
    per_task = []

    for task in sorted(common_tasks):
        r2_1 = results1[task]['test_r2']
        r2_2 = results2[task]['test_r2']

        # Skip extreme outliers (bad fits with huge negative R²)
        if r2_1 < -10 or r2_2 < -10:
            if verbose:
                print(f"  Skipping outlier {task}: {dir1.name}={r2_1:.2f}, {dir2.name}={r2_2:.2f}")
            continue

        diff = abs(r2_1 - r2_2)
        r2_diffs.append(diff)

        per_task.append({
            'task': task,
            'r2_1': r2_1,
            'r2_2': r2_2,
            'r2_diff': diff,
            'eq_1': results1[task].get('best_equation', ''),
            'eq_2': results2[task].get('best_equation', ''),
            'ground_truth': results1[task].get('ground_truth', ''),
        })

    r2_diffs = np.array(r2_diffs)

    # R² variance statistics
    r2_stats = {
        'mean': float(np.mean(r2_diffs)),
        'median': float(np.median(r2_diffs)),
        'std': float(np.std(r2_diffs)),
        'max': float(np.max(r2_diffs)),
        'min': float(np.min(r2_diffs)),
        'pct_lt_001': float(100 * np.sum(r2_diffs < 0.01) / len(r2_diffs)),
        'pct_lt_005': float(100 * np.sum(r2_diffs < 0.05) / len(r2_diffs)),
        'pct_lt_010': float(100 * np.sum(r2_diffs < 0.10) / len(r2_diffs)),
        'pct_gte_010': float(100 * np.sum(r2_diffs >= 0.10) / len(r2_diffs)),
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"R² Score Difference Statistics (n={len(r2_diffs)} tasks)")
        print(f"{'='*70}")
        print(f"  Mean |Δ R²|:    {r2_stats['mean']:.4f}")
        print(f"  Median |Δ R²|:  {r2_stats['median']:.4f}")
        print(f"  Std of |Δ R²|:  {r2_stats['std']:.4f}")
        print(f"  Max |Δ R²|:     {r2_stats['max']:.4f}")
        print(f"  Min |Δ R²|:     {r2_stats['min']:.6f}")
        print(f"\n  Distribution of |Δ R²|:")
        print(f"    < 0.01:  {r2_stats['pct_lt_001']:.1f}% (stable)")
        print(f"    < 0.05:  {r2_stats['pct_lt_005']:.1f}%")
        print(f"    < 0.10:  {r2_stats['pct_lt_010']:.1f}%")
        print(f"    >= 0.10: {r2_stats['pct_gte_010']:.1f}% (high variance)")

    # Solve rate analysis
    # By R² threshold
    solved_r2_1 = {t['task'] for t in per_task if t['r2_1'] >= r2_threshold}
    solved_r2_2 = {t['task'] for t in per_task if t['r2_2'] >= r2_threshold}

    # By symbolic match (if requested)
    solved_sym_1 = set()
    solved_sym_2 = set()

    if use_symbolic:
        for t in per_task:
            task = t['task']
            ground_truth = t['ground_truth']
            if not ground_truth:
                continue

            # Get variable names
            try:
                var_names = get_dataset_var_names(task)
            except Exception:
                var_names = None

            n_vars = results1[task].get('n_features', 10)

            # Check dir1
            match1 = check_sr_symbolic_match(t['eq_1'], ground_truth, n_vars, var_names)
            if match1.get('match', False):
                solved_sym_1.add(task)
                t['symbolic_match_1'] = True
            else:
                t['symbolic_match_1'] = False

            # Check dir2
            match2 = check_sr_symbolic_match(t['eq_2'], ground_truth, n_vars, var_names)
            if match2.get('match', False):
                solved_sym_2.add(task)
                t['symbolic_match_2'] = True
            else:
                t['symbolic_match_2'] = False

    # Compute overlaps
    n_tasks = len(per_task)
    solve_rate = {
        'r2_threshold': r2_threshold,
        # R² threshold based
        'r2_solved_1': len(solved_r2_1),
        'r2_solved_2': len(solved_r2_2),
        'r2_both': len(solved_r2_1 & solved_r2_2),
        'r2_only_1': len(solved_r2_1 - solved_r2_2),
        'r2_only_2': len(solved_r2_2 - solved_r2_1),
        'r2_neither': n_tasks - len(solved_r2_1 | solved_r2_2),
        'r2_union': len(solved_r2_1 | solved_r2_2),
    }

    if use_symbolic:
        solve_rate.update({
            # Symbolic match based
            'sym_solved_1': len(solved_sym_1),
            'sym_solved_2': len(solved_sym_2),
            'sym_both': len(solved_sym_1 & solved_sym_2),
            'sym_only_1': len(solved_sym_1 - solved_sym_2),
            'sym_only_2': len(solved_sym_2 - solved_sym_1),
            'sym_neither': n_tasks - len(solved_sym_1 | solved_sym_2),
            'sym_union': len(solved_sym_1 | solved_sym_2),
        })

    if verbose:
        print(f"\n{'='*70}")
        print(f"Solve Rate Analysis")
        print(f"{'='*70}")

        print(f"\n  By R² >= {r2_threshold}:")
        print(f"    {dir1.name}: {solve_rate['r2_solved_1']}/{n_tasks} ({100*solve_rate['r2_solved_1']/n_tasks:.1f}%)")
        print(f"    {dir2.name}: {solve_rate['r2_solved_2']}/{n_tasks} ({100*solve_rate['r2_solved_2']/n_tasks:.1f}%)")
        print(f"    ─────────────────────────────")
        print(f"    Both:       {solve_rate['r2_both']}")
        print(f"    Only {dir1.name}: {solve_rate['r2_only_1']}")
        print(f"    Only {dir2.name}: {solve_rate['r2_only_2']}")
        print(f"    Neither:    {solve_rate['r2_neither']}")
        print(f"    Union:      {solve_rate['r2_union']} ({100*solve_rate['r2_union']/n_tasks:.1f}%)")

        if use_symbolic:
            print(f"\n  By Symbolic Match:")
            print(f"    {dir1.name}: {solve_rate['sym_solved_1']}/{n_tasks} ({100*solve_rate['sym_solved_1']/n_tasks:.1f}%)")
            print(f"    {dir2.name}: {solve_rate['sym_solved_2']}/{n_tasks} ({100*solve_rate['sym_solved_2']/n_tasks:.1f}%)")
            print(f"    ─────────────────────────────")
            print(f"    Both:       {solve_rate['sym_both']}")
            print(f"    Only {dir1.name}: {solve_rate['sym_only_1']}")
            print(f"    Only {dir2.name}: {solve_rate['sym_only_2']}")
            print(f"    Neither:    {solve_rate['sym_neither']}")
            print(f"    Union:      {solve_rate['sym_union']} ({100*solve_rate['sym_union']/n_tasks:.1f}%)")

        # Show tasks with high variance
        high_var = [t for t in per_task if t['r2_diff'] >= 0.10]
        if high_var:
            print(f"\n{'='*70}")
            print(f"Tasks with High Variance (|Δ R²| >= 0.10): {len(high_var)}")
            print(f"{'='*70}")
            print(f"{'Task':<30} {dir1.name:<12} {dir2.name:<12} {'|Δ R²|':<10}")
            print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*10}")
            for t in sorted(high_var, key=lambda x: -x['r2_diff'])[:20]:
                print(f"{t['task']:<30} {t['r2_1']:<12.4f} {t['r2_2']:<12.4f} {t['r2_diff']:<10.4f}")

        # Show tasks solved by only one seed
        if use_symbolic:
            only_1 = solved_sym_1 - solved_sym_2
            only_2 = solved_sym_2 - solved_sym_1
            if only_1 or only_2:
                print(f"\n{'='*70}")
                print(f"Tasks Solved by Only One Seed (Symbolic)")
                print(f"{'='*70}")
                if only_1:
                    print(f"\n  Only {dir1.name}:")
                    for task in sorted(only_1):
                        t = next(x for x in per_task if x['task'] == task)
                        print(f"    {task}: R²={t['r2_1']:.4f} vs {t['r2_2']:.4f}")
                if only_2:
                    print(f"\n  Only {dir2.name}:")
                    for task in sorted(only_2):
                        t = next(x for x in per_task if x['task'] == task)
                        print(f"    {task}: R²={t['r2_1']:.4f} vs {t['r2_2']:.4f}")

    return {
        'dir1': str(dir1),
        'dir2': str(dir2),
        'n_tasks': n_tasks,
        'r2_stats': r2_stats,
        'solve_rate': solve_rate,
        'per_task': per_task,
    }


def analyze_r2_across_runs(base_dir: str = ".", verbose: bool = True):
    """
    Analyze R² scores across multiple results_sr{i} directories.

    Shows mean and std of average R² across tasks for:
    1. All tasks (whole split)
    2. Train split (splits/train.txt)
    3. Val split (splits/val.txt)

    Args:
        base_dir: Base directory containing results_sr* directories
        verbose: Whether to print per-problem details

    Returns:
        dict with summary statistics for each split
    """
    import json
    import glob
    from collections import defaultdict
    from pathlib import Path

    def get_results_dirs(base_dir: str) -> list:
        """Find all results_sr* directories."""
        base = Path(base_dir)
        dirs = []

        # Check for results_sr (no number)
        if (base / "results_sr").is_dir():
            dirs.append(str(base / "results_sr"))

        # Check for results_sr2, results_sr3, etc.
        i = 2
        while True:
            dir_path = base / f"results_sr{i}"
            if dir_path.is_dir():
                dirs.append(str(dir_path))
                i += 1
            else:
                break

        return sorted(dirs)

    def load_split(split_path: str) -> set:
        """Load dataset names from a split file."""
        try:
            with open(split_path, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            return set()

    def collect_r2_scores(results_dirs: list, min_r2: float = -100) -> dict:
        """Collect R² scores for each problem across all directories."""
        scores = defaultdict(list)

        for dir_path in results_dirs:
            for json_file in glob.glob(os.path.join(dir_path, "*_results.json")):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    dataset = data.get("dataset", "")
                    test_r2 = data.get("test_r2")

                    if dataset and test_r2 is not None:
                        if test_r2 >= min_r2:
                            scores[dataset].append(test_r2)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not read {json_file}: {e}")

        return dict(scores)

    def calculate_per_problem_stats(scores: list) -> tuple:
        """Calculate mean and sample variance for a single problem."""
        n = len(scores)
        if n == 0:
            return float('nan'), float('nan')

        mean = sum(scores) / n

        if n == 1:
            return mean, float('nan')

        variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
        return mean, variance

    def calculate_split_summary(stats: list, split_name: str) -> dict:
        """Calculate mean and std of avg R² across problems in a split."""
        if not stats:
            return {'n_problems': 0, 'mean': float('nan'), 'std': float('nan')}

        # Get the mean R² for each run (averaged across problems)
        means = [s['mean'] for s in stats if s['mean'] == s['mean']]  # filter NaN

        if not means:
            return {'n_problems': 0, 'mean': float('nan'), 'std': float('nan')}

        n = len(means)
        avg = sum(means) / n
        if n > 1:
            std = (sum((x - avg) ** 2 for x in means) / (n - 1)) ** 0.5
        else:
            std = float('nan')

        return {
            'n_problems': n,
            'mean': avg,
            'std': std,
            'vals': means,
        }

    # Main logic
    base = Path(base_dir)
    results_dirs = get_results_dirs(base_dir)

    if not results_dirs:
        print("No results_sr* directories found.")
        return {}

    print(f"Found {len(results_dirs)} results directories:")
    for d in results_dirs:
        print(f"  - {d}")
    print()

    # Collect all R² scores
    all_scores = collect_r2_scores(results_dirs)
    return all_scores

    if not all_scores:
        print("No results found.")
        return {}

    # Load splits
    train_split = load_split(base / "splits" / "train.txt")
    val_split = load_split(base / "splits" / "val.txt")

    # Calculate per-problem stats
    all_stats = []
    train_stats = []
    val_stats = []

    for problem, r2_values in all_scores.items():
        mean, variance = calculate_per_problem_stats(r2_values)
        stat = {
            'problem': problem,
            'mean': mean,
            'variance': variance,
            'std': variance ** 0.5 if variance == variance else float('nan'),
            'n_runs': len(r2_values),
            'r2_values': r2_values
        }
        all_stats.append(stat)

        if problem in train_split:
            train_stats.append(stat)
        if problem in val_split:
            val_stats.append(stat)

    # Sort by variance
    all_stats.sort(key=lambda x: (x['variance'] != x['variance'], x['variance']))

    # Print per-problem details if verbose
    if verbose:
        print(f"{'Problem':<35} {'Mean R²':>10} {'Std':>10} {'N':>4}  R² values")
        print("-" * 100)

        for s in all_stats:
            std_str = f"{s['std']:.4f}" if s['std'] == s['std'] else "N/A"
            r2_vals = ", ".join(f"{v:.4f}" for v in sorted(s['r2_values'], reverse=True))
            print(f"{s['problem']:<35} {s['mean']:>10.4f} {std_str:>10} {s['n_runs']:>4}  [{r2_vals}]")

        print()

    # Calculate and print summary for each split
    summaries = {}

    print("=" * 70)
    print("SUMMARY: Mean and Std of Avg R² Across Tasks")
    print("=" * 70)
    print(f"{'Split':<20} {'N Problems':>12} {'Mean R²':>12} {'Std R²':>12} {'Vals':>12}")
    print("-" * 58)

    for split_name, stats in [("All", all_stats), ("Train", train_stats), ("Val", val_stats)]:
        summary = calculate_split_summary(stats, split_name)
        summaries[split_name.lower()] = summary

        mean_str = f"{summary['mean']:.4f}" if summary['mean'] == summary['mean'] else "N/A"
        std_str = f"{summary['std']:.4f}" if summary['std'] == summary['std'] else "N/A"
        vals_str = f"{summary['vals']}" if 'vals' in summary else "N/A"

        print(f"{split_name:<20} {summary['n_problems']:>12} {mean_str:>12} {std_str:>12} {vals_str:>12}")

    print("=" * 70)

    return {
        'all': summaries.get('all', {}),
        'train': summaries.get('train', {}),
        'val': summaries.get('val', {}),
        'per_problem': all_stats,
    }


def analyze_best_of_n_match_rate(base_dir: str = ".", min_r2: float = -100):
    """
    Analyze symbolic match rate when selecting the best equation by test R²
    across multiple runs for each problem.

    Also debugs cases where R² = 1.0 but symbolic match fails.

    Args:
        base_dir: Base directory containing results_sr* directories
        min_r2: Minimum R² value to include (filters outliers)

    Returns:
        dict with match statistics
    """
    import json
    import glob
    from collections import defaultdict
    from pathlib import Path

    def get_results_dirs(base_dir: str) -> list:
        """Find all results_sr* directories."""
        base = Path(base_dir)
        dirs = []
        if (base / "results_sr").is_dir():
            dirs.append(str(base / "results_sr"))
        i = 2
        while True:
            dir_path = base / f"results_sr{i}"
            if dir_path.is_dir():
                dirs.append(str(dir_path))
                i += 1
            else:
                break
        return sorted(dirs)

    # Collect all runs for each problem
    results_dirs = get_results_dirs(base_dir)
    if not results_dirs:
        print("No results_sr* directories found.")
        return {}

    print(f"Found {len(results_dirs)} results directories")

    # problem -> list of {test_r2, best_equation, ground_truth, n_features, dir}
    all_runs = defaultdict(list)

    for dir_path in results_dirs:
        for json_file in glob.glob(os.path.join(dir_path, "*_results.json")):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                dataset = data.get("dataset", "")
                test_r2 = data.get("test_r2")

                if dataset and test_r2 is not None and test_r2 >= min_r2:
                    all_runs[dataset].append({
                        'test_r2': test_r2,
                        'best_equation': data.get('best_equation', ''),
                        'ground_truth': data.get('ground_truth', ''),
                        'n_features': data.get('n_features', 10),
                        'dir': dir_path,
                    })
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read {json_file}: {e}")

    print(f"Found {len(all_runs)} problems with results\n")

    # For each problem, select the best run by test_r2 and check symbolic match
    matches = 0
    total = 0
    r2_perfect_match = 0
    r2_perfect_total = 0
    r2_perfect_failures = []

    results = []

    for problem in sorted(all_runs.keys()):
        runs = all_runs[problem]
        if not runs:
            continue

        # Select best run by test_r2
        best_run = max(runs, key=lambda x: x['test_r2'])
        best_eq = best_run['best_equation']
        ground_truth = best_run['ground_truth']
        best_r2 = best_run['test_r2']
        n_features = best_run['n_features']

        if not ground_truth:
            continue

        # Get variable names for symbolic matching
        try:
            var_names = get_dataset_var_names(problem)
        except Exception:
            var_names = None

        # Check symbolic match
        match_result = check_sr_symbolic_match(best_eq, ground_truth, n_features, var_names)
        is_match = match_result.get('match', False)

        total += 1
        if is_match:
            matches += 1

        results.append({
            'problem': problem,
            'best_r2': best_r2,
            'best_equation': best_eq,
            'ground_truth': ground_truth,
            'match': is_match,
            'n_runs': len(runs),
        })

        # Debug: check R² = 1.0 cases
        if abs(best_r2 - 1.0) < 1e-6:  # R² essentially equals 1.0
            r2_perfect_total += 1
            if is_match:
                r2_perfect_match += 1
            else:
                r2_perfect_failures.append({
                    'problem': problem,
                    'best_equation': best_eq,
                    'ground_truth': ground_truth,
                    'r2': best_r2,
                })

    # Print results
    match_rate = 100 * matches / total if total > 0 else 0
    print("=" * 80)
    print("BEST-OF-N SYMBOLIC MATCH RATE (selecting best equation by test R²)")
    print("=" * 80)
    print(f"Total problems: {total}")
    print(f"Symbolic matches: {matches}")
    print(f"Match rate: {match_rate:.1f}%")
    print()

    # Debug: R² = 1.0 analysis
    print("=" * 80)
    print("DEBUG: R² = 1.0 SYMBOLIC MATCH ANALYSIS")
    print("=" * 80)
    if r2_perfect_total > 0:
        r2_perfect_rate = 100 * r2_perfect_match / r2_perfect_total
        print(f"Problems with R² = 1.0: {r2_perfect_total}")
        print(f"Of those, symbolic matches: {r2_perfect_match}")
        print(f"Match rate for R² = 1.0: {r2_perfect_rate:.1f}%")
        print()

        if r2_perfect_failures:
            print(f"FAILURES (R² = 1.0 but no symbolic match): {len(r2_perfect_failures)}")
            print("-" * 80)
            for f in r2_perfect_failures:
                print(f"Problem: {f['problem']}")
                print(f"  R²:         {f['r2']:.6f}")
                print(f"  Predicted:  {f['best_equation']}")
                print(f"  Ground:     {f['ground_truth']}")
                print()
    else:
        print("No problems with R² = 1.0 found.")

    # Print per-problem results sorted by match status then R²
    print("=" * 80)
    print("PER-PROBLEM RESULTS (sorted by match, then R²)")
    print("=" * 80)
    print(f"{'Problem':<30} {'Match':>6} {'Best R²':>10} {'N':>4}")
    print("-" * 55)

    # Sort: matches first, then by R² descending
    results.sort(key=lambda x: (not x['match'], -x['best_r2']))

    for r in results:
        match_str = "YES" if r['match'] else "NO"
        print(f"{r['problem']:<30} {match_str:>6} {r['best_r2']:>10.6f} {r['n_runs']:>4}")

    return {
        'total': total,
        'matches': matches,
        'match_rate': match_rate,
        'r2_perfect_total': r2_perfect_total,
        'r2_perfect_match': r2_perfect_match,
        'r2_perfect_failures': r2_perfect_failures,
        'results': results,
    }


def quick_r2_solve_rate(base_dir: str = ".", r2_threshold: float = 0.9999):
    """
    Quickly print R² >= threshold solve rate for each results directory.

    Args:
        base_dir: Base directory containing results_sr* directories
        r2_threshold: R² threshold to count as "solved" (default 0.9999)
    """
    import json
    import glob
    from pathlib import Path

    base = Path(base_dir)
    dirs = []
    if (base / "results_sr").is_dir():
        dirs.append(("results_sr", str(base / "results_sr")))
    i = 2
    while True:
        dir_name = f"results_sr{i}"
        dir_path = base / dir_name
        if dir_path.is_dir():
            dirs.append((dir_name, str(dir_path)))
            i += 1
        else:
            break

    print(f"R² >= {r2_threshold} solve rate per run:")
    print("-" * 50)

    all_solved = {}  # problem -> list of which runs solved it

    for dir_name, dir_path in dirs:
        solved = 0
        total = 0
        for json_file in glob.glob(os.path.join(dir_path, "*_results.json")):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                dataset = data.get("dataset", "")
                test_r2 = data.get("test_r2")
                if dataset and test_r2 is not None:
                    total += 1
                    if test_r2 >= r2_threshold:
                        solved += 1
                        if dataset not in all_solved:
                            all_solved[dataset] = []
                        all_solved[dataset].append(dir_name)
            except:
                pass

        rate = 100 * solved / total if total > 0 else 0
        print(f"{dir_name:<15} {solved:>3}/{total:<3} ({rate:>5.1f}%)")

    # Best-of-N analysis
    print("-" * 50)
    n_runs = len(dirs)
    total_problems = len(all_solved) if all_solved else 0

    # Count problems solved by at least one run
    solved_by_any = sum(1 for p, runs in all_solved.items() if len(runs) >= 1)
    print(f"Solved by ANY run: {solved_by_any} problems")

    # Count problems solved by all runs
    solved_by_all = sum(1 for p, runs in all_solved.items() if len(runs) == n_runs)
    print(f"Solved by ALL runs: {solved_by_all} problems")


def print_comparison_summary(base_dir: str = ".", pysr_dir: str = None, sr_seeds: list = None,
                             r2_threshold: float = 0.9999, split_file: str = None):
    """
    Print average R² and solve rate (R² >= threshold) for PySR and selected SR seeds.

    Args:
        base_dir: Base directory containing results
        pysr_dir: Path to PySR results directory (default: results_pysr/12_20)
        sr_seeds: List of SR seed directories to compare (default: ['results_sr', 'results_sr2'])
        r2_threshold: R² threshold to count as "solved" (default 0.9999, i.e. ~1.0)
        split_file: Optional path to split file (e.g., splits/val.txt) to filter problems
    """
    import json
    from pathlib import Path

    base = Path(base_dir)

    if pysr_dir is None:
        pysr_dir = base / "results_pysr" / "12_20"
    else:
        pysr_dir = Path(pysr_dir)

    if sr_seeds is None:
        sr_seeds = ['results_sr', 'results_sr2']

    # Load split filter if provided
    filter_problems = None
    split_name = "all"
    if split_file:
        split_path = base / split_file if not Path(split_file).is_absolute() else Path(split_file)
        if split_path.exists():
            with open(split_path) as f:
                filter_problems = set(line.strip() for line in f if line.strip())
            split_name = split_path.stem

    print("=" * 70)
    print(f"COMPARISON: Average R² and Solve Rate (R² >= {r2_threshold})")
    if filter_problems:
        print(f"Filtered to: {split_name} ({len(filter_problems)} problems)")
    print("=" * 70)

    def load_results_from_dir(results_dir, filter_set=None):
        """Load R² values from JSON result files in a directory."""
        r2_values = []
        results_dir = Path(results_dir)

        for json_file in results_dir.glob('*_results.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                dataset = data.get('dataset', '')
                test_r2 = data.get('test_r2')

                # Apply filter if provided
                if filter_set and dataset not in filter_set:
                    continue

                if test_r2 is not None:
                    r2_values.append(test_r2)
            except Exception:
                pass

        return r2_values

    def compute_stats(r2_values):
        """Compute statistics for a set of R² values."""
        n_problems = len(r2_values)
        if n_problems == 0:
            return None

        # Filter outliers for average (keep R² >= -10)
        r2_filtered = [r2 for r2 in r2_values if r2 >= -10]
        avg_r2 = sum(r2_filtered) / len(r2_filtered) if r2_filtered else 0
        n_solved_9999 = sum(1 for r2 in r2_values if r2 >= r2_threshold)
        n_solved_exact = sum(1 for r2 in r2_values if r2 == 1.0)
        n_outliers = n_problems - len(r2_filtered)

        return {
            'n_problems': n_problems,
            'avg_r2': avg_r2,
            'n_outliers': n_outliers,
            'n_solved_9999': n_solved_9999,
            'n_solved_exact': n_solved_exact,
        }

    # Collect all results
    all_results = {}

    # ---- PySR Results ----
    if pysr_dir.exists():
        r2_values = load_results_from_dir(pysr_dir, filter_problems)
        all_results[f"PySR ({pysr_dir.parent.name}/{pysr_dir.name})"] = compute_stats(r2_values)
    else:
        print(f"\nPySR: Directory not found at {pysr_dir}")

    # ---- SR Seed Results ----
    for seed_dir in sr_seeds:
        seed_path = base / seed_dir
        if not seed_path.exists():
            print(f"\n{seed_dir}: Directory not found")
            continue

        r2_values = load_results_from_dir(seed_path, filter_problems)
        all_results[seed_dir] = compute_stats(r2_values)

    # Print results table
    print(f"\n{'Method':<35} {'N':>4}  {'Avg R²':>8}  {'≥0.9999':>12}  {'=1.0':>12}")
    print("-" * 75)

    for name, stats in all_results.items():
        if stats is None:
            print(f"{name:<35} No results found")
            continue

        n = stats['n_problems']
        avg_r2 = stats['avg_r2']
        n_9999 = stats['n_solved_9999']
        n_exact = stats['n_solved_exact']
        rate_9999 = 100 * n_9999 / n
        rate_exact = 100 * n_exact / n

        avg_str = f"{avg_r2:.4f}"
        if stats['n_outliers'] > 0:
            avg_str += "*"

        print(f"{name:<35} {n:>4}  {avg_str:>8}  {n_9999:>3}/{n:<3} ({rate_9999:>5.1f}%)  {n_exact:>3}/{n:<3} ({rate_exact:>5.1f}%)")

    print("\n" + "=" * 70)


def plot_pysr_r2_thresholds(pysr_dirs: list, labels: list = None, base_dir: str = ".",
                             split_file: str = None, output_path: str = None, max_nines: int = 6):
    """
    Plot number of tasks achieving various R² thresholds for different PySR runs.

    Args:
        pysr_dirs: List of PySR result directories
        labels: List of labels for legend (e.g., ['1e3', '1e4', ...]). If None, extracted from dir names.
        base_dir: Base directory containing results
        split_file: Optional path to split file to filter problems
        output_path: Optional path to save the plot (if None, displays interactively)
        max_nines: Maximum number of 9s after decimal point (default 6, i.e., 0.999999).
                   Beyond ~6 nines, floating point precision becomes unreliable.
    """
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path

    base = Path(base_dir)

    # Load split filter if provided
    filter_problems = None
    if split_file:
        split_path = base / split_file if not Path(split_file).is_absolute() else Path(split_file)
        if split_path.exists():
            with open(split_path) as f:
                filter_problems = set(line.strip() for line in f if line.strip())

    def load_r2_values(results_dir, filter_set=None):
        """Load R² values from JSON result files."""
        r2_values = []
        results_dir = Path(results_dir)

        for json_file in results_dir.glob('*_results.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                dataset = data.get('dataset', '')
                test_r2 = data.get('test_r2')

                if filter_set and dataset not in filter_set:
                    continue

                if test_r2 is not None:
                    r2_values.append(test_r2)
            except Exception:
                pass

        return r2_values

    # Generate R² thresholds: 0.9, 0.99, 0.999, ... up to max_nines
    base_thresholds = []
    for n in range(1, max_nines + 1):
        thresh = 1.0 - 10**(-n)  # 0.9, 0.99, 0.999, ...
        base_thresholds.append(thresh)

    # Collect data for all directories
    all_data = {}
    all_r2_values = []

    for i, pysr_dir in enumerate(pysr_dirs):
        pysr_path = base / pysr_dir if not Path(pysr_dir).is_absolute() else Path(pysr_dir)
        if not pysr_path.exists():
            print(f"Warning: {pysr_dir} not found, skipping")
            continue

        r2_values = load_r2_values(pysr_path, filter_problems)
        print(pysr_dir, f"({len(r2_values)} tasks loaded)")
        all_r2_values.extend(r2_values)

        # Determine label
        if labels and i < len(labels):
            label = labels[i]
        else:
            # Try to extract from directory name (e.g., results_pysr_1e4 -> 1e4)
            name = pysr_path.name
            if '_1e' in name:
                label = '1e' + name.split('_1e')[-1]
            else:
                label = name

        all_data[label] = r2_values

    if not all_data:
        print("No data found!")
        return

    # Always include all thresholds plus 1.0 at the end
    thresholds_to_show = base_thresholds + [1.0]

    # Create x-axis labels: 0.9, 0.99, 0.999, ..., 1.0
    def format_threshold(t, n_nines=None):
        if t == 1.0:
            return "1.0"
        if n_nines is not None:
            return "0." + "9" * n_nines
        # Fallback
        return f"{t:.10f}".rstrip('0')

    x_labels = [format_threshold(t, i+1) if i < len(base_thresholds) else "1.0"
                for i, t in enumerate(thresholds_to_show)]
    x_positions = list(range(len(thresholds_to_show)))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_data)))

    for (label, r2_values), color in zip(all_data.items(), colors):
        counts = []
        for thresh in thresholds_to_show:
            if thresh == 1.0:
                count = sum(1 for r2 in r2_values if r2 == 1.0)
            else:
                count = sum(1 for r2 in r2_values if r2 >= thresh)
            counts.append(count)

        ax.plot(x_positions, counts, 'o-', label=label, color=color, markersize=8, linewidth=2)

    ax.set_xlabel('R² Threshold', fontsize=12)
    ax.set_ylabel('Number of Tasks', fontsize=12)
    ax.set_title('Tasks Achieving R² Thresholds by Number of Evaluations', fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend(title='Evaluations', loc='best')
    ax.grid(True, alpha=0.3)

    # Add total tasks annotation
    total_tasks = len(list(all_data.values())[0]) if all_data else 0
    ax.axhline(y=total_tasks, color='gray', linestyle='--', alpha=0.5)
    ax.text(len(x_positions) - 1, total_tasks + 1, f'Total: {total_tasks}', ha='right', va='bottom', color='gray')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    return fig, ax


def print_multi_pysr_comparison(pysr_dirs: list, base_dir: str = ".", sr_seeds: list = None,
                                 r2_threshold: float = 0.9999, split_file: str = None):
    """
    Print average R² and solve rate for multiple PySR directories.

    Args:
        pysr_dirs: List of PySR result directories (relative to base_dir or absolute paths)
        base_dir: Base directory containing results
        sr_seeds: List of SR seed directories to compare (default: None, no SR comparison)
        r2_threshold: R² threshold to count as "solved" (default 0.9999)
        split_file: Optional path to split file to filter problems
    """
    import json
    from pathlib import Path

    base = Path(base_dir)

    if sr_seeds is None:
        sr_seeds = []

    # Load split filter if provided
    filter_problems = None
    split_name = "all"
    if split_file:
        split_path = base / split_file if not Path(split_file).is_absolute() else Path(split_file)
        if split_path.exists():
            with open(split_path) as f:
                filter_problems = set(line.strip() for line in f if line.strip())
            split_name = split_path.stem

    print("=" * 80)
    print(f"COMPARISON: Average R² and Solve Rate (R² >= {r2_threshold})")
    if filter_problems:
        print(f"Filtered to: {split_name} ({len(filter_problems)} problems)")
    print("=" * 80)

    def load_results_from_dir(results_dir, filter_set=None):
        """Load R² values from JSON result files in a directory."""
        r2_values = []
        results_dir = Path(results_dir)

        for json_file in results_dir.glob('*_results.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                dataset = data.get('dataset', '')
                test_r2 = data.get('test_r2')

                if filter_set and dataset not in filter_set:
                    continue

                if test_r2 is not None:
                    r2_values.append(test_r2)
            except Exception:
                pass

        return r2_values

    def compute_stats(r2_values):
        """Compute statistics for a set of R² values."""
        n_problems = len(r2_values)
        if n_problems == 0:
            return None

        r2_filtered = [r2 for r2 in r2_values if r2 >= -10]
        avg_r2 = sum(r2_filtered) / len(r2_filtered) if r2_filtered else 0
        n_solved_9999 = sum(1 for r2 in r2_values if r2 >= r2_threshold)
        n_solved_exact = sum(1 for r2 in r2_values if r2 == 1.0)
        n_outliers = n_problems - len(r2_filtered)

        return {
            'n_problems': n_problems,
            'avg_r2': avg_r2,
            'n_outliers': n_outliers,
            'n_solved_9999': n_solved_9999,
            'n_solved_exact': n_solved_exact,
        }

    all_results = {}

    # ---- PySR Results ----
    for pysr_dir in pysr_dirs:
        pysr_path = base / pysr_dir if not Path(pysr_dir).is_absolute() else Path(pysr_dir)
        if pysr_path.exists():
            r2_values = load_results_from_dir(pysr_path, filter_problems)
            all_results[f"PySR: {pysr_dir}"] = compute_stats(r2_values)
        else:
            print(f"  Warning: {pysr_dir} not found")

    # ---- SR Seed Results ----
    for seed_dir in sr_seeds:
        seed_path = base / seed_dir
        if seed_path.exists():
            r2_values = load_results_from_dir(seed_path, filter_problems)
            all_results[seed_dir] = compute_stats(r2_values)

    # Print results table
    print(f"\n{'Method':<40} {'N':>4}  {'Avg R²':>8}  {'≥0.9999':>12}  {'=1.0':>12}")
    print("-" * 82)

    for name, stats in all_results.items():
        if stats is None:
            print(f"{name:<40} No results found")
            continue

        n = stats['n_problems']
        avg_r2 = stats['avg_r2']
        n_9999 = stats['n_solved_9999']
        n_exact = stats['n_solved_exact']
        rate_9999 = 100 * n_9999 / n
        rate_exact = 100 * n_exact / n

        avg_str = f"{avg_r2:.4f}"
        if stats['n_outliers'] > 0:
            avg_str += "*"

        print(f"{name:<40} {n:>4}  {avg_str:>8}  {n_9999:>3}/{n:<3} ({rate_9999:>5.1f}%)  {n_exact:>3}/{n:<3} ({rate_exact:>5.1f}%)")

    print("\n" + "=" * 80)
    return all_results['PySR: results/results_pysr_1e6']


def plot_pysr_r2_by_noise(results_pattern: str = "results_pysr_*", base_dir: str = ".",
                          split_file: str = None, output_dir: str = None, max_nines: int = 6,
                          no_noise_dirs: list = None):
    """
    Plot R² threshold curves grouped by noise level.

    Creates a separate plot for each noise level, with lines for each eval count.

    Args:
        results_pattern: Glob pattern to find result directories (e.g., "results_pysr_*")
        base_dir: Base directory containing results
        split_file: Optional path to split file to filter problems
        output_dir: Directory to save plots (if None, displays interactively)
        max_nines: Maximum number of 9s after decimal point (default 6)
        no_noise_dirs: List of directories containing noise=0 results (e.g.,
                       ["results/results_pysr_1e3", "results/results_pysr_1e4", ...])
                       These will be added as noise=0.0 data points.

    Directory naming convention expected: results_pysr_{noise}_{max_evals}
    e.g., results_pysr_0.001_1000, results_pysr_0.01_10000
    """
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path
    from collections import defaultdict
    import re

    base = Path(base_dir)

    # Load split filter if provided
    filter_problems = None
    if split_file:
        split_path = base / split_file if not Path(split_file).is_absolute() else Path(split_file)
        if split_path.exists():
            with open(split_path) as f:
                filter_problems = set(line.strip() for line in f if line.strip())

    def load_r2_values(results_dir, filter_set=None):
        """Load R² values and noise level from JSON result files."""
        r2_values = []
        noise_level = None
        max_evals = None
        results_dir = Path(results_dir)

        for json_file in results_dir.glob('*_n*_seed*.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                dataset = data.get('dataset', '')
                test_r2 = data.get('test_r2')

                # Get noise level from file (if available)
                if noise_level is None:
                    noise_level = data.get('target_noise', 0.0)

                if filter_set and dataset not in filter_set:
                    continue

                if test_r2 is not None:
                    r2_values.append(test_r2)
            except Exception:
                pass

        # Also check older format (*_results.json)
        for json_file in results_dir.glob('*_results.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                dataset = data.get('dataset', '')
                test_r2 = data.get('test_r2')

                if noise_level is None:
                    noise_level = data.get('target_noise', 0.0)

                if filter_set and dataset not in filter_set:
                    continue

                if test_r2 is not None:
                    r2_values.append(test_r2)
            except Exception:
                pass

        return r2_values, noise_level

    def parse_dir_name(dir_name):
        """Parse noise level and max_evals from directory name.

        Expected formats:
        - results_pysr_{noise}_{max_evals} (e.g., results_pysr_0.001_1000)
        - results_pysr_1e{exp} (e.g., results_pysr_1e4) -> noise=0, evals=10^exp
        """
        # Try pattern: results_pysr_{noise}_{evals}
        match = re.match(r'results_pysr_(\d+\.?\d*)_(\d+)', dir_name)
        if match:
            noise = float(match.group(1))
            evals = int(match.group(2))
            return noise, evals

        # Try pattern: results_pysr_1e{exp} (no noise)
        match = re.match(r'results_pysr_1e(\d+)', dir_name)
        if match:
            exp = int(match.group(1))
            evals = 10 ** exp
            return 0.0, evals

        return None, None

    # Find all matching directories
    result_dirs = list(base.glob(results_pattern))

    # Also add no_noise_dirs if provided
    if no_noise_dirs:
        for nd in no_noise_dirs:
            nd_path = base / nd if not Path(nd).is_absolute() else Path(nd)
            if nd_path.exists() and nd_path not in result_dirs:
                result_dirs.append(nd_path)

    if not result_dirs:
        print(f"No directories matching '{results_pattern}' found in {base}")
        return

    print(f"Found {len(result_dirs)} result directories")

    # Group by noise level
    # Structure: {noise_level: {max_evals: [r2_values]}}
    data_by_noise = defaultdict(dict)

    for result_dir in sorted(result_dirs):
        dir_name = result_dir.name
        noise_from_name, evals_from_name = parse_dir_name(dir_name)

        r2_values, noise_from_file = load_r2_values(result_dir, filter_problems)

        if not r2_values:
            print(f"  Skipping {dir_name}: no results")
            continue

        # Prefer noise from file, fall back to directory name
        noise = noise_from_file if noise_from_file is not None else noise_from_name
        evals = evals_from_name

        if noise is None or evals is None:
            print(f"  Skipping {dir_name}: couldn't parse noise/evals")
            continue

        print(f"  {dir_name}: noise={noise}, evals={evals}, n_tasks={len(r2_values)}")
        data_by_noise[noise][evals] = r2_values

    if not data_by_noise:
        print("No valid data found!")
        return

    # Generate R² thresholds
    base_thresholds = [1.0 - 10**(-n) for n in range(1, max_nines + 1)]
    thresholds_to_show = base_thresholds + [1.0]

    def format_threshold(t, n_nines=None):
        if t == 1.0:
            return "1.0"
        if n_nines is not None:
            return "0." + "9" * n_nines
        return f"{t:.10f}".rstrip('0')

    x_labels = [format_threshold(t, i+1) if i < len(base_thresholds) else "1.0"
                for i, t in enumerate(thresholds_to_show)]
    x_positions = list(range(len(thresholds_to_show)))

    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Create a plot for each noise level
    figures = {}

    for noise_level in sorted(data_by_noise.keys()):
        evals_data = data_by_noise[noise_level]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort by evals for consistent ordering
        sorted_evals = sorted(evals_data.keys())
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_evals)))

        for evals, color in zip(sorted_evals, colors):
            r2_values = evals_data[evals]

            counts = []
            for thresh in thresholds_to_show:
                if thresh == 1.0:
                    count = sum(1 for r2 in r2_values if r2 == 1.0)
                else:
                    count = sum(1 for r2 in r2_values if r2 >= thresh)
                counts.append(count)

            # Format evals label nicely (1000 -> 1e3, 10000 -> 1e4, etc.)
            if evals >= 1000:
                exp = len(str(evals)) - 1
                label = f"1e{exp}"
            else:
                label = str(evals)

            ax.plot(x_positions, counts, 'o-', label=label, color=color, markersize=8, linewidth=2)

        noise_str = f"{noise_level}" if noise_level > 0 else "0.0 (no noise)"
        ax.set_xlabel('R² Threshold', fontsize=12)
        ax.set_ylabel('Number of Tasks', fontsize=12)
        ax.set_title(f'Tasks Achieving R² Thresholds (Target Noise = {noise_str})', fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend(title='Evaluations', loc='best')
        ax.grid(True, alpha=0.3)

        # Add total tasks annotation
        total_tasks = len(list(evals_data.values())[0]) if evals_data else 0
        ax.axhline(y=total_tasks, color='gray', linestyle='--', alpha=0.5)
        ax.text(len(x_positions) - 1, total_tasks + 1, f'Total: {total_tasks}',
                ha='right', va='bottom', color='gray')

        plt.tight_layout()

        if output_dir:
            # Save with noise level in filename
            noise_fname = str(noise_level).replace('.', 'p')
            out_file = output_path / f"pysr_r2_thresholds_noise_{noise_fname}.png"
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            print(f"Saved: {out_file}")
            plt.close(fig)
        else:
            figures[noise_level] = fig

    # Also create a combined plot with all noise levels (subplots)
    n_noise_levels = len(data_by_noise)
    if n_noise_levels > 1:
        fig, axes = plt.subplots(1, n_noise_levels, figsize=(6 * n_noise_levels, 5), sharey=True)
        if n_noise_levels == 1:
            axes = [axes]

        for ax, noise_level in zip(axes, sorted(data_by_noise.keys())):
            evals_data = data_by_noise[noise_level]
            sorted_evals = sorted(evals_data.keys())
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_evals)))

            for evals, color in zip(sorted_evals, colors):
                r2_values = evals_data[evals]

                counts = []
                for thresh in thresholds_to_show:
                    if thresh == 1.0:
                        count = sum(1 for r2 in r2_values if r2 == 1.0)
                    else:
                        count = sum(1 for r2 in r2_values if r2 >= thresh)
                    counts.append(count)

                if evals >= 1000:
                    exp = len(str(evals)) - 1
                    label = f"1e{exp}"
                else:
                    label = str(evals)

                ax.plot(x_positions, counts, 'o-', label=label, color=color, markersize=6, linewidth=1.5)

            noise_str = f"{noise_level}" if noise_level > 0 else "0.0"
            ax.set_xlabel('R² Threshold', fontsize=10)
            ax.set_title(f'Noise = {noise_str}', fontsize=12)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(title='Evals', loc='best', fontsize=8)

        axes[0].set_ylabel('Number of Tasks', fontsize=10)
        fig.suptitle('Tasks Achieving R² Thresholds by Noise Level', fontsize=14, y=1.02)
        plt.tight_layout()

        if output_dir:
            out_file = output_path / "pysr_r2_thresholds_all_noise.png"
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            print(f"Saved combined plot: {out_file}")
            plt.close(fig)
        else:
            figures['combined'] = fig

    if not output_dir:
        plt.show()

    return figures, data_by_noise


def plot_pysr_r2_by_noise_splits(results_pattern: str, base_dir: str, output_dir: str,
                                 split_files: list, split_labels: list,
                                 no_noise_dirs: list, original_split_file: str,
                                 combined_output_name: str = "pysr_r2_thresholds_all_noise_with_splits.png",
                                 keep_intermediate: bool = False,
                                 combined_title_labels: list = None):
    """
    Create split-filtered versions of pysr_r2_thresholds_all_noise.png and stack them with the original.

    Args:
        results_pattern: Glob pattern to find result directories
        base_dir: Base directory containing results
        output_dir: Directory to save plots
        split_files: List of split file paths (relative to base_dir or absolute)
        split_labels: List of suffix labels for saved images (e.g., ["train_hard", ...])
        no_noise_dirs: List of directories containing noise=0 results
        original_split_file: Split file used for the "original" plot (e.g., splits/srbench_all.txt)
        combined_output_name: Filename for stacked output image
        keep_intermediate: Keep intermediate per-split directories if True
    """
    from pathlib import Path
    import shutil
    from PIL import Image, ImageDraw, ImageFont

    if len(split_files) != len(split_labels):
        raise ValueError("split_files and split_labels must have the same length")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    original_path = output_path / "pysr_r2_thresholds_all_noise.png"
    if not original_path.exists():
        plot_pysr_r2_by_noise(
            results_pattern=results_pattern,
            base_dir=base_dir,
            split_file=original_split_file,
            output_dir=output_dir,
            no_noise_dirs=no_noise_dirs,
        )

    split_images = []
    for split_file, split_label in zip(split_files, split_labels):
        tmp_dir = output_path / f"_tmp_{split_label}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        plot_pysr_r2_by_noise(
            results_pattern=results_pattern,
            base_dir=base_dir,
            split_file=split_file,
            output_dir=str(tmp_dir),
            no_noise_dirs=no_noise_dirs,
        )

        tmp_combined = tmp_dir / "pysr_r2_thresholds_all_noise.png"
        split_out = output_path / f"pysr_r2_thresholds_all_noise_{split_label}.png"
        if tmp_combined.exists():
            shutil.copyfile(tmp_combined, split_out)
            split_images.append(split_out)
        else:
            print(f"Warning: combined plot not found for {split_label} at {tmp_combined}")

        if not keep_intermediate:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    image_paths = [original_path] + split_images
    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")

    if combined_title_labels is None:
        combined_title_labels = ["all"] + split_labels
    if len(combined_title_labels) != len(image_paths):
        raise ValueError("combined_title_labels must match number of images")

    images = []
    for path, title in zip(image_paths, combined_title_labels):
        with Image.open(path) as img:
            base_img = img.copy().convert("RGB")

        font = ImageFont.load_default()
        try:
            bbox = font.getbbox(title)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = font.getsize(title)

        pad_x = 16
        pad_y = 10
        banner_h = text_h + pad_y * 2

        banner = Image.new("RGB", (base_img.width, banner_h), (245, 245, 245))
        draw = ImageDraw.Draw(banner)
        text_x = max(pad_x, (base_img.width - text_w) // 2)
        text_y = (banner_h - text_h) // 2
        draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)

        labeled = Image.new("RGB", (base_img.width, base_img.height + banner_h), (255, 255, 255))
        labeled.paste(banner, (0, 0))
        labeled.paste(base_img, (0, banner_h))
        images.append(labeled)

    max_width = max(img.width for img in images)
    resized = []
    for img in images:
        if img.width != max_width:
            new_height = round(img.height * (max_width / img.width))
            resized.append(img.resize((max_width, new_height), Image.LANCZOS))
        else:
            resized.append(img)

    total_height = sum(img.height for img in resized)
    combined = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    y_offset = 0
    for img in resized:
        combined.paste(img, (0, y_offset))
        y_offset += img.height

    combined_path = output_path / combined_output_name
    combined.save(combined_path)
    print(f"Saved combined rows image: {combined_path}")


if __name__ == "__main__":
    # Plot R² thresholds by noise level
    # Expects directories like: results_pysr_0.001_1000, results_pysr_0.01_10000, etc.
    # Also includes existing no-noise results from results/results_pysr_1e*
    _, _ = plot_pysr_r2_by_noise(
        results_pattern="results/results_pysr_*_*",
        split_file="splits/srbench_all.txt",
        output_dir="plots",  # or None to display interactively
        no_noise_dirs=[
            "results/results_pysr_1e3",
            "results/results_pysr_1e4",
            "results/results_pysr_1e5",
            "results/results_pysr_1e6",
            "results/results_pysr_1e7",
            "results/results_pysr_1e8",
        ]
    )

    plot_pysr_r2_by_noise_splits(
        results_pattern="results/results_pysr_*_*",
        base_dir=".",
        output_dir="plots",
        split_files=[
            "splits/train_hard.txt",
            "splits/val_hard.txt",
            "splits/test_hard.txt",
        ],
        split_labels=[
            "train_hard",
            "val_hard",
            "test_hard",
        ],
        no_noise_dirs=[
            "results/results_pysr_1e3",
            "results/results_pysr_1e4",
            "results/results_pysr_1e5",
            "results/results_pysr_1e6",
            "results/results_pysr_1e7",
            "results/results_pysr_1e8",
        ],
        original_split_file="splits/srbench_all.txt",
    )
