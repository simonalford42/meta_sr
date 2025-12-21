#!/usr/bin/env python3
"""
Extract and analyze ground truth formulas from SRBench datasets.

This script:
1. Reads metadata from all Feynman and Strogatz datasets
2. Extracts the ground truth formulas
3. Analyzes which operators are used across all formulas
"""

import re
import yaml
from pathlib import Path
from collections import Counter


def extract_formula_from_metadata(metadata_path):
    """Extract the ground truth formula from a metadata.yaml file."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)

        if 'description' not in metadata:
            return None

        desc = metadata['description']
        lines = desc.strip().split('\n')

        # Look for lines with '=' that look like formulas
        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Skip variable range definitions like "x1 in [1.0,5.0]"
            if ' in [' in line or ' in (' in line:
                continue
            # Look for assignment-like formulas
            if '=' in line:
                return line

        return None
    except Exception as e:
        return None


# Datasets excluded from SRBench (contain arcsin/arccos which most methods don't support)
EXCLUDED_DATASETS = {
    'feynman_test_10',   # contains arccos
    'feynman_I_26_2',    # contains arcsin
    'feynman_I_30_5',    # contains arcsin
}


def get_ground_truth_datasets(pmlb_path=None, exclude_arcsin_arccos=True):
    """Get list of ground-truth dataset directories (Feynman + Strogatz)."""
    if pmlb_path is None:
        pmlb_path = Path(__file__).parent / 'pmlb' / 'datasets'
    else:
        pmlb_path = Path(pmlb_path)

    datasets = []
    for d in sorted(pmlb_path.iterdir()):
        if d.is_dir() and ('feynman' in d.name or 'strogatz' in d.name):
            if exclude_arcsin_arccos and d.name in EXCLUDED_DATASETS:
                continue
            datasets.append(d)

    return datasets


def extract_operators_from_formula(formula):
    """
    Extract binary and unary operators from a formula string.

    Returns:
        dict with 'binary' and 'unary' operator sets
    """
    binary_ops = set()
    unary_ops = set()

    # Binary operators to look for
    if '+' in formula:
        binary_ops.add('+')
    if '-' in formula:
        binary_ops.add('-')
    if '/' in formula:
        binary_ops.add('/')

    # Check for multiplication
    if re.search(r'\d+\s*\*\s*\w', formula) or re.search(r'\w\s*\*\s*\w', formula):
        binary_ops.add('*')
    # Also check for * not followed by another *
    if re.search(r'\*(?!\*)', formula):
        binary_ops.add('*')

    # Handle power operators: distinguish **2, **3 from general **n
    # Find all **N patterns
    power_matches = re.findall(r'\*\*\s*(\d+(?:\.\d+)?|\([^)]+\)|\w+)', formula)
    has_pow2 = False
    has_pow3 = False
    has_general_power = False

    for match in power_matches:
        match = match.strip()
        if match == '2':
            has_pow2 = True
        elif match == '3':
            has_pow3 = True
        else:
            # Check for fractional powers like 3/2 or (3/2)
            has_general_power = True

    # Also check for patterns like x**2 where 2 is part of expression
    if re.search(r'\*\*\s*2(?!\d)', formula):
        has_pow2 = True
    if re.search(r'\*\*\s*3(?!\d)', formula):
        has_pow3 = True
    # Check for other integer powers (4, 5, etc.) or fractional/variable powers
    if re.search(r'\*\*\s*[456789]', formula) or re.search(r'\*\*\s*\(', formula):
        has_general_power = True

    if has_pow2:
        unary_ops.add('pow2')
    if has_pow3:
        unary_ops.add('pow3')
    if has_general_power:
        binary_ops.add('^')

    # Unary operators (functions)
    # Note: tan(x) = sin(x)/cos(x), cot(x) = cos(x)/sin(x)
    # So we map tan/cot to sin+cos
    unary_patterns = [
        (r'\bsqrt\s*\(', 'sqrt'),
        (r'\bsin\s*\(', 'sin'),
        (r'\bcos\s*\(', 'cos'),
        (r'\bexp\s*\(', 'exp'),
        (r'\blog\s*\(', 'log'),
        (r'\bln\s*\(', 'log'),  # ln is natural log
        (r'\babs\s*\(', 'abs'),
        (r'\barcsin\s*\(', 'arcsin'),
        (r'\barccos\s*\(', 'arccos'),
        (r'\barctan\s*\(', 'arctan'),
        (r'\basin\s*\(', 'arcsin'),  # Alternative notation
        (r'\bacos\s*\(', 'arccos'),
        (r'\batan\s*\(', 'arctan'),
        (r'\btanh\s*\(', 'tanh'),
        (r'\bsinh\s*\(', 'sinh'),
        (r'\bcosh\s*\(', 'cosh'),
    ]

    for pattern, op_name in unary_patterns:
        if re.search(pattern, formula, re.IGNORECASE):
            unary_ops.add(op_name)

    # Handle tan and cot -> map to sin + cos (since tan = sin/cos, cot = cos/sin)
    if re.search(r'\btan\s*\(', formula, re.IGNORECASE):
        unary_ops.add('sin')
        unary_ops.add('cos')
        binary_ops.add('/')  # tan = sin/cos requires division
    if re.search(r'\bcot\s*\(', formula, re.IGNORECASE):
        unary_ops.add('sin')
        unary_ops.add('cos')
        binary_ops.add('/')  # cot = cos/sin requires division

    return {'binary': binary_ops, 'unary': unary_ops}


def print_all_formulas(pmlb_path=None):
    """Print ground truth formulas for all datasets."""
    datasets = get_ground_truth_datasets(pmlb_path)

    print("=" * 80)
    print("GROUND TRUTH FORMULAS FROM SRBENCH")
    print("=" * 80)

    formulas = {}

    for dataset_dir in datasets:
        metadata_path = dataset_dir / 'metadata.yaml'
        if metadata_path.exists():
            formula = extract_formula_from_metadata(metadata_path)
            if formula:
                formulas[dataset_dir.name] = formula
                print(f"\n{dataset_dir.name}:")
                print(f"  {formula}")

    print(f"\n{'=' * 80}")
    print(f"Total datasets with formulas: {len(formulas)}")

    return formulas


def extract_all_exponents(formula):
    """
    Extract all exponents from a formula string.

    Returns a list of exponent strings found (e.g., ['2', '3', '4', '3/2', '(3/2)'])
    """
    # Find all **N patterns where N can be:
    # - integers: **2, **3, **4, **5
    # - fractions in parens: **(3/2)
    # - expressions: **(n+1)

    exponents = []

    # Pattern for ** followed by various exponent forms
    # Match integers
    int_matches = re.findall(r'\*\*\s*(\d+)', formula)
    exponents.extend(int_matches)

    # Match parenthesized expressions like (3/2)
    paren_matches = re.findall(r'\*\*\s*\(([^)]+)\)', formula)
    exponents.extend([f"({m})" for m in paren_matches])

    return exponents


def extract_constants_from_formula(formula):
    """
    Extract numeric constants from a formula string.

    Returns a list of constant strings found (e.g., ['2', '3.14159', '1/2'])
    """
    constants = []

    # Remove the left-hand side of the equation (e.g., "F = ...")
    if '=' in formula:
        formula = formula.split('=', 1)[1]

    # Find floating point numbers (including scientific notation)
    # Match patterns like: 3.14159, 1.0, 0.5, 1e-10, 2.5e3
    float_matches = re.findall(r'(?<![a-zA-Z_])(\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![a-zA-Z_\d])', formula)
    constants.extend(float_matches)

    # Find integers that are standalone constants (not part of variable names like x1, x2)
    # Exclude integers that are part of exponents (handled separately)
    # Also exclude integers that are part of variable names

    # First, remove exponents to avoid double-counting
    formula_no_exp = re.sub(r'\*\*\s*\d+', '', formula)
    # Remove variable names like x1, x2, etc.
    formula_no_vars = re.sub(r'[a-zA-Z_]\d+', '', formula_no_exp)

    # Find standalone integers
    int_matches = re.findall(r'(?<![a-zA-Z_\d.])(\d+)(?![a-zA-Z_\d.])', formula_no_vars)
    constants.extend(int_matches)

    # Find fractions like 1/2, 3/2 (but not in exponents which are handled separately)
    # Look for patterns like (1/2), (3/2) that aren't exponents
    # Use simpler pattern that doesn't require variable-width lookbehind
    fraction_matches = re.findall(r'[\(\s](\d+/\d+)[\)\s]', formula)
    # Filter out exponent fractions (already captured in exponent analysis)
    for frac in fraction_matches:
        # Check if this fraction appears after ** in the original formula
        if f'**({frac})' not in formula and f'** ({frac})' not in formula:
            constants.append(frac)

    return constants


def normalize_constant(const_str):
    """
    Normalize a constant string for comparison.
    Returns a tuple (display_value, numeric_value) for sorting.
    """
    const_str = const_str.strip()

    # Try to evaluate as a number
    try:
        if '/' in const_str:
            parts = const_str.split('/')
            val = float(parts[0]) / float(parts[1])
            return (const_str, val)
        else:
            val = float(const_str)
            # Normalize display: remove trailing zeros
            if val == int(val):
                return (str(int(val)), val)
            return (const_str, val)
    except:
        return (const_str, float('inf'))


def analyze_constants(formulas):
    """Analyze which constants are used across all formulas."""
    all_constants = Counter()
    constant_examples = {}  # constant -> list of (dataset, formula) examples

    print("\n" + "=" * 80)
    print("CONSTANT ANALYSIS")
    print("=" * 80)

    for name, formula in formulas.items():
        # Extract constants - count each unique constant once per formula
        constants = extract_constants_from_formula(formula)

        # Normalize and dedupe within formula
        seen = set()
        for const in constants:
            norm, _ = normalize_constant(const)
            if norm not in seen:
                seen.add(norm)
                all_constants[norm] += 1
                if norm not in constant_examples:
                    constant_examples[norm] = []
                if len(constant_examples[norm]) < 3:  # Keep up to 3 examples
                    constant_examples[norm].append((name, formula))

    print("\nCONSTANTS (sorted by frequency):")
    print("-" * 40)

    # Sort by frequency (descending), then by numeric value
    sorted_constants = sorted(
        all_constants.items(),
        key=lambda x: (-x[1], normalize_constant(x[0])[1])
    )

    for const, count in sorted_constants:
        pct = 100 * count / len(formulas)
        print(f"  {const:15s} : {count:3d} formulas ({pct:5.1f}%)")

    # Group by type
    print("\n" + "-" * 40)
    print("CONSTANTS BY TYPE:")
    print("-" * 40)

    integers = []
    floats = []
    fractions = []

    for const, count in sorted_constants:
        if '/' in const:
            fractions.append((const, count))
        elif '.' in const or 'e' in const.lower():
            floats.append((const, count))
        else:
            integers.append((const, count))

    if integers:
        print("\n  Integers:")
        for const, count in sorted(integers, key=lambda x: (int(x[0]) if x[0].lstrip('-').isdigit() else float('inf'))):
            pct = 100 * count / len(formulas)
            print(f"    {const:12s} : {count:3d} formulas ({pct:5.1f}%)")

    if floats:
        print("\n  Floats:")
        for const, count in sorted(floats, key=lambda x: -x[1]):
            pct = 100 * count / len(formulas)
            print(f"    {const:12s} : {count:3d} formulas ({pct:5.1f}%)")

    if fractions:
        print("\n  Fractions:")
        for const, count in sorted(fractions, key=lambda x: -x[1]):
            pct = 100 * count / len(formulas)
            print(f"    {const:12s} : {count:3d} formulas ({pct:5.1f}%)")

    # Show examples for most common constants
    print("\n" + "-" * 40)
    print("EXAMPLES OF MOST COMMON CONSTANTS:")
    print("-" * 40)

    for const, count in sorted_constants[:5]:  # Top 5
        print(f"\n  Constant '{const}' ({count} formulas):")
        for dataset, formula in constant_examples[const][:2]:
            if len(formula) > 65:
                formula = formula[:62] + "..."
            print(f"    {dataset}:")
            print(f"      {formula}")

    return all_constants


def analyze_operators(formulas):
    """Analyze which operators are used across all formulas."""
    all_binary = Counter()
    all_unary = Counter()
    all_exponents = Counter()  # Count formulas containing each exponent (not occurrences)
    exponent_examples = {}  # exponent -> list of (dataset, formula) examples

    print("\n" + "=" * 80)
    print("OPERATOR ANALYSIS")
    print("=" * 80)

    for name, formula in formulas.items():
        ops = extract_operators_from_formula(formula)
        for op in ops['binary']:
            all_binary[op] += 1
        for op in ops['unary']:
            all_unary[op] += 1

        # Extract exponents - count each unique exponent once per formula
        exponents = set(extract_all_exponents(formula))  # Use set to dedupe within formula
        for exp in exponents:
            all_exponents[exp] += 1
            if exp not in exponent_examples:
                exponent_examples[exp] = []
            if len(exponent_examples[exp]) < 3:  # Keep up to 3 examples
                exponent_examples[exp].append((name, formula))

    print("\nBINARY OPERATORS (sorted by frequency):")
    print("-" * 40)
    for op, count in all_binary.most_common():
        pct = 100 * count / len(formulas)
        print(f"  {op:10s} : {count:3d} formulas ({pct:5.1f}%)")

    print("\nUNARY OPERATORS (sorted by frequency):")
    print("-" * 40)
    for op, count in all_unary.most_common():
        pct = 100 * count / len(formulas)
        print(f"  {op:10s} : {count:3d} formulas ({pct:5.1f}%)")

    # Print exponent analysis
    print("\n" + "=" * 80)
    print("EXPONENT ANALYSIS (all **N patterns)")
    print("=" * 80)

    # Sort exponents: integers first (by value), then others
    def exponent_sort_key(exp_count):
        exp = exp_count[0]
        try:
            return (0, int(exp))  # integers first, sorted numerically
        except ValueError:
            return (1, exp)  # non-integers after, sorted alphabetically

    sorted_exponents = sorted(all_exponents.items(), key=exponent_sort_key)

    for exp, count in sorted_exponents:
        pct = 100 * count / len(formulas)
        print(f"\n  **{exp:10s} : {count:3d} formulas ({pct:5.1f}%)")
        # Show examples
        for dataset, formula in exponent_examples[exp][:2]:
            # Truncate long formulas
            if len(formula) > 60:
                formula = formula[:57] + "..."
            print(f"      Example: {dataset}")
            print(f"               {formula}")

    # Print summary for PySR configuration
    print("\n" + "=" * 80)
    print("RECOMMENDED PYSR CONFIGURATION")
    print("=" * 80)

    binary_list = [op for op, _ in all_binary.most_common()]
    unary_list = [op for op, _ in all_unary.most_common()]

    print(f"\nbinary_operators = {binary_list}")
    print(f"unary_operators = {unary_list}")

    return all_binary, all_unary


def get_operators_for_dataset(dataset_name, pmlb_path=None):
    """Get the operators needed for a specific dataset."""
    if pmlb_path is None:
        pmlb_path = Path(__file__).parent / 'pmlb' / 'datasets'
    else:
        pmlb_path = Path(pmlb_path)

    metadata_path = pmlb_path / dataset_name / 'metadata.yaml'
    formula = extract_formula_from_metadata(metadata_path)

    if formula:
        return extract_operators_from_formula(formula)
    return None


def print_formulas_by_complexity():
    """Print formulas sorted by operator complexity."""
    formulas = {}
    datasets = get_ground_truth_datasets()

    for dataset_dir in datasets:
        metadata_path = dataset_dir / 'metadata.yaml'
        if metadata_path.exists():
            formula = extract_formula_from_metadata(metadata_path)
            if formula:
                ops = extract_operators_from_formula(formula)
                n_ops = len(ops['binary']) + len(ops['unary'])
                formulas[dataset_dir.name] = {
                    'formula': formula,
                    'ops': ops,
                    'n_ops': n_ops,
                    'has_unary': len(ops['unary']) > 0
                }

    print("\n" + "=" * 80)
    print("FORMULAS BY COMPLEXITY (operators used)")
    print("=" * 80)

    # Sort by number of operators
    sorted_formulas = sorted(formulas.items(), key=lambda x: (x[1]['n_ops'], x[1]['has_unary']))

    print("\n--- Simple (binary ops only) ---")
    for name, info in sorted_formulas:
        if not info['has_unary']:
            print(f"{name}: {info['formula']}")
            print(f"  Ops: {info['ops']['binary']}")

    print("\n--- Complex (includes unary ops) ---")
    for name, info in sorted_formulas:
        if info['has_unary']:
            print(f"{name}: {info['formula']}")
            print(f"  Binary: {info['ops']['binary']}, Unary: {info['ops']['unary']}")

    return formulas


if __name__ == '__main__':
    # Print all formulas
    formulas = print_all_formulas()

    # Analyze operators
    binary_ops, unary_ops = analyze_operators(formulas)

    # Analyze constants
    constants = analyze_constants(formulas)

    # Print by complexity
    print_formulas_by_complexity()
