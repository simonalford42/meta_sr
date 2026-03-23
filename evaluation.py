"""
Evaluation module for symbolic regression.

Provides functions for:
- Converting Node expressions to sympy
- Comparing predicted expressions with ground truth symbolically (SRBench-style)
- Calculating R², symbolic match (0/1), and complexity
"""

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

    def _is_constant(expr):
        if expr is None:
            return False
        v = expr.is_constant()
        return bool(v) if v is not None else False

    # Check match conditions
    result['error_is_zero'] = bool(sym_diff.equals(0))
    result['error_is_constant'] = _is_constant(sym_diff)
    result['fraction_is_constant'] = _is_constant(sym_frac)

    # A match is any of the three conditions
    result['match'] = (
        result['error_is_zero'] or
        result['error_is_constant'] or
        result['fraction_is_constant']
    )

    return result


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
    y_pred = predicted_node.evaluate(X)
    if np.any(~np.isfinite(y_pred)):
        result['r2'] = -np.inf
        result['mse'] = np.inf
    else:
        result['r2'] = r2_score(y, y_pred)
        result['mse'] = float(np.mean((y - y_pred) ** 2))

    # Calculate complexity using sympy
    predicted_sympy = node_to_sympy(predicted_node)
    result['complexity'] = complexity(predicted_sympy)

    # Check symbolic match if ground truth provided
    if ground_truth_str is not None:
        n_vars = X.shape[1]
        ground_truth_sympy = parse_expr_str_to_sympy(ground_truth_str)

        match_result = check_symbolic_match(predicted_sympy, ground_truth_sympy, n_vars)
        result['symbolic_match'] = 1 if match_result['match'] else 0
        result['symbolic_details'] = match_result

    return result


# ============================================================================
# PySR Results Evaluation
# ============================================================================

def parse_expr_str_to_sympy(expr_str, var_names=None):
    """
    Parse an expression string to sympy.

    Args:
        expr_str: Expression string (e.g., '(6.28 * alpha) / (n * d)')
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
        predicted = parse_expr_str_to_sympy(expr_str, var_names)
        ground_truth = parse_expr_str_to_sympy(ground_truth_str, var_names)

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


def get_pareto_df_indices_in_best_complexity_order(equations_df, best_df_index):
    """
    Order Pareto expressions by complexity distance from best.

    Ordering:
      1) Best expression first.
      2) Increasing |complexity - best_complexity|.
      3) Ties prefer higher complexity than best.
      4) Then deterministic by df index.
    """
    if equations_df is None or len(equations_df) == 0:
        return []

    if best_df_index not in equations_df.index:
        # Fallback: preserve dataframe order if best index is unavailable.
        return list(equations_df.index)

    best_complexity = int(equations_df.loc[best_df_index]["complexity"])

    def _sort_key(df_index):
        row = equations_df.loc[df_index]
        complexity = int(row["complexity"])
        delta = complexity - best_complexity
        abs_delta = abs(delta)
        # For same abs distance, prefer +delta over -delta.
        # For delta==0 (non-best rows), place after best via primary key.
        tie_pref = 0 if delta > 0 else (1 if delta == 0 else 2)
        return (abs_delta, tie_pref, complexity, str(df_index))

    ordered = sorted(equations_df.index, key=_sort_key)
    # Ensure best is strictly first even when other keys tie.
    ordered = [best_df_index] + [i for i in ordered if i != best_df_index]
    return ordered


def check_pysr_frontier_symbolic_match(
    equations_df,
    best_df_index,
    ground_truth_str,
    var_names=None,
    timeout_seconds_per_expression=3,
):
    """
    Check symbolic match across entire Pareto frontier.

    Returns True if any expression is symbolic match.
    Timeout on an expression is treated as non-match for that expression.
    """
    ordered_indices = get_pareto_df_indices_in_best_complexity_order(
        equations_df, best_df_index
    )

    if not ordered_indices:
        return {
            "match": False,
            "matched_df_index": None,
            "checked_count": 0,
            "timeouts": 0,
            "order": [],
        }

    n_timeouts = 0
    for pos, idx in enumerate(ordered_indices, start=1):
        row = equations_df.loc[idx]
        expr = str(row["equation"])
        res = check_pysr_symbolic_match(
            expr,
            ground_truth_str,
            var_names=var_names,
            timeout_seconds=timeout_seconds_per_expression,
        )
        if res.get("error") == "timeout":
            n_timeouts += 1
        if bool(res.get("match", False)):
            return {
                "match": True,
                "matched_df_index": idx,
                "checked_count": pos,
                "timeouts": n_timeouts,
                "order": ordered_indices,
            }

    return {
        "match": False,
        "matched_df_index": None,
        "checked_count": len(ordered_indices),
        "timeouts": n_timeouts,
        "order": ordered_indices,
    }


def check_sympy_equivalence_with_llm(
    predicted_expr,
    ground_truth_expr,
    model: str = "openai/gpt-5.2",
    thinking_level: str = "high",
    max_tokens=None,
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
        max_tokens: max output tokens (None means do not set a cap in request)
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

    predicted_str = predicted_expr
    ground_truth_str = ground_truth_expr

    system_prompt = (
        "You are a rigorous symbolic math equivalence checker. "
        "Decide if two expressions are equivalent under ANY of: "
        "(a) exact algebraic equality, "
        "(b) differ by an additive constant only, "
        "(c) differ by a multiplicative constant only. "
        "Respond with STRICT JSON only. "
        "The JSON must contain exactly two keys: "
        "{\"explanation\": \"...\", \"equivalent\": true/false}. "
        "Explanation must be 1-3 sentences and must come before the final verdict logically. "
        "No markdown, no extra keys."
    )

    user_prompt = (
        "Determine equivalence under the criteria above.\n"
        f"Expression A: {predicted_str}\n"
        f"Expression B: {ground_truth_str}\n\n"
        "Output exactly one JSON object with keys:\n"
        "  explanation: short 1-3 sentence reason\n"
        "  equivalent: true/false"
    )

    token_budgets = [max_tokens]
    if max_tokens is not None and max_tokens < 1000:
        token_budgets.append(1000)

    last_content = ""
    last_error = None

    for token_budget in token_budgets:
        try:
            response = chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=token_budget,
                temperature=0.0,
                use_cache=use_cache,
                include_default_reasoning=False,
                reasoning={"effort": thinking_level},
            )
            content = get_content(response).strip()
            last_content = content

            llm_match = None

            explanation = ""
            # Parse strict JSON first.
            try:
                # Tolerate accidental fenced JSON.
                if content.startswith("```"):
                    stripped = content.strip("` \n")
                    if stripped.startswith("json\n"):
                        content = stripped[5:].strip()
                    else:
                        content = stripped.strip()
                parsed = json.loads(content)
                llm_match = bool(parsed.get("equivalent", False))
                explanation = str(parsed.get("explanation", "")).strip()
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
                    last_error = (
                        "Could not parse LLM equivalence response "
                        f"(max_tokens={token_budget})"
                    )
                    continue

            return {
                "llm_match": bool(llm_match),
                "raw_response": content,
                "reasoning": explanation,
                "model": model,
                "effort": thinking_level,
                "max_tokens_used": token_budget,
                "error": None,
            }
        except Exception as e:
            last_error = str(e)

    return {
        "llm_match": False,
        "raw_response": last_content,
        "reasoning": "",
        "model": model,
        "effort": thinking_level,
        "max_tokens_used": token_budgets[-1],
        "error": last_error or "Could not parse LLM equivalence response",
    }


def check_pysr_symbolic_match_with_llm(
    expr_str,
    ground_truth_str,
    var_names=None,
    timeout_seconds: int = 10,
    llm_model: str = "openai/gpt-5.2",
    llm_thinking_level: str = "high",
    llm_max_tokens=None,
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
    predicted = parse_expr_str_to_sympy(expr_str, var_names)
    ground_truth = parse_expr_str_to_sympy(ground_truth_str, var_names)
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


# ============================================================================
# SR (BasicSR) Results Evaluation
# ============================================================================

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
    # BasicSR uses ^ for exponentiation
    expr_str = expr_str.replace('^', '**')

    # If var_names provided, substitute original variable names with x0, x1, etc.
    # in the ground truth before passing to check_pysr_symbolic_match
    if var_names:
        gt_expr = parse_expr_str_to_sympy(ground_truth_str, var_names)
        subs_dict = {Symbol(name): Symbol(f'x{i}') for i, name in enumerate(var_names)}
        gt_expr = gt_expr.subs(subs_dict)
        ground_truth_str = str(gt_expr)

    return check_pysr_symbolic_match(expr_str, ground_truth_str, timeout_seconds=timeout_seconds)
