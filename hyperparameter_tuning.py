"""
Hyperparameter tuning for meta-evolved operators.

This module provides functionality to:
1. Identify tunable hyperparameters in operator code via LLM
2. Extract and inject hyperparameter values in code
3. Optimize hyperparameters using black-box optimization (Optuna)
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from completions import chat_completion, get_content
from meta_evolution import (
    Operator,
    OperatorBundle,
    create_and_test_operator,
    OPERATOR_TYPES,
    SR_ALGORITHM_PSEUDOCODE,
)


@dataclass
class HyperparameterSpec:
    """Specification for a tunable hyperparameter."""
    name: str
    param_type: str  # "float", "int", "categorical"
    current_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None  # For categorical
    log_scale: bool = False  # For float/int, whether to use log scale
    description: str = ""


HYPERPARAMETER_IDENTIFICATION_PROMPT = """You are analyzing an operator that is part of an evolutionary symbolic regression (SR) algorithm. Your task is to identify tunable hyperparameters in the operator code so that an automated hyperparameter tuning system can optimize them to improve SR performance.

## Context: Symbolic Regression Algorithm

This operator is used within the following SR algorithm:

```python
{sr_pseudocode}
```

The algorithm evolves mathematical expressions (expression trees) to fit data. The four operator types are:
- **fitness**: Computes how good an expression is (higher = better)
- **selection**: Chooses which expressions to use as parents for the next generation
- **mutation**: Modifies an expression to create a new variant
- **crossover**: Combines two parent expressions to create an offspring

## Operator Being Analyzed

**Operator Type:** {operator_type}

```python
{code}
```

## Your Task

Identify ALL numeric constants and thresholds in this code that could be tuned to improve performance. These are the "hyperparameters" of this operator.

For each hyperparameter, provide:
1. `name`: A descriptive name (snake_case)
2. `line_pattern`: The exact code snippet containing the value (for string matching)
3. `current_value`: The current numeric value in the code (must be a number, not a variable reference)
4. `param_type`: One of "float", "int", or "categorical"
5. `min_value`: Minimum reasonable value (for float/int)
6. `max_value`: Maximum reasonable value (for float/int)
7. `log_scale`: Whether to search on log scale (true for values spanning orders of magnitude)
8. `choices`: List of options (for categorical only)
9. `description`: Brief description of what this parameter controls

## Guidelines

- **Include**: probabilities (0-1), thresholds, multipliers, exponents, size limits, counts, penalty weights
- **Exclude**: loop indices, array indices, string constants, variable references like `self.max_size`
- Be conservative with ranges - suggest reasonable bounds based on the parameter's role
- For probabilities, use min=0.0, max=1.0
- For small positive values (e.g., 0.001 to 1.0), consider log_scale=true

## Output Format

Return a JSON array. Example:
```json
[
    {{
        "name": "tournament_size",
        "line_pattern": "tournament_size = 3",
        "current_value": 3,
        "param_type": "int",
        "min_value": 2,
        "max_value": 10,
        "log_scale": false,
        "choices": null,
        "description": "Number of individuals competing in tournament selection"
    }}
]
```

If there are no tunable hyperparameters, return: `[]`

Return ONLY the JSON array, no additional text."""


def identify_hyperparameters(
    operator: Operator,
    model: str = "openai/gpt-5-mini",
    llm_temperature: float = 0.0,
) -> List[HyperparameterSpec]:
    """
    Use LLM to identify tunable hyperparameters in operator code.

    Args:
        operator: The Operator to analyze
        model: LLM model to use
        llm_temperature: Temperature for LLM (lower = more deterministic)

    Returns:
        List of HyperparameterSpec objects
    """
    prompt = HYPERPARAMETER_IDENTIFICATION_PROMPT.format(
        code=operator.code,
        operator_type=operator.operator_type,
        sr_pseudocode=SR_ALGORITHM_PSEUDOCODE,
    )

    response = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert code analyst. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=llm_temperature,
    )

    content = get_content(response)

    # Parse JSON from response
    try:
        # Try to extract JSON from code block if present
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = content.strip()

        hyperparams_data = json.loads(json_str)

        # Convert to HyperparameterSpec objects
        specs = []
        for hp in hyperparams_data:
            spec = HyperparameterSpec(
                name=hp["name"],
                param_type=hp["param_type"],
                current_value=hp["current_value"],
                min_value=hp.get("min_value"),
                max_value=hp.get("max_value"),
                choices=hp.get("choices"),
                log_scale=hp.get("log_scale", False),
                description=hp.get("description", ""),
            )
            # Store line_pattern for later use
            spec.line_pattern = hp.get("line_pattern", "")
            specs.append(spec)

        # Filter out invalid hyperparameters (e.g., self.max_size references)
        filtered_specs = []
        for spec in specs:
            # Skip if line_pattern references self.* (instance variables, not literals)
            line_pattern = getattr(spec, 'line_pattern', '')
            if 'self.' in line_pattern:
                print(f"  Filtering out '{spec.name}': references instance variable (self.*)")
                continue

            # Skip if current_value is not a valid number
            try:
                if spec.param_type == "int":
                    int(spec.current_value)
                elif spec.param_type == "float":
                    float(spec.current_value)
                elif spec.param_type == "categorical":
                    if not spec.choices:
                        print(f"  Filtering out '{spec.name}': categorical with no choices")
                        continue
            except (ValueError, TypeError):
                print(f"  Filtering out '{spec.name}': current_value '{spec.current_value}' is not a valid number")
                continue

            filtered_specs.append(spec)

        return filtered_specs

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Failed to parse hyperparameter response: {e}")
        print(f"Response was: {content[:500]}...")
        return []


def inject_hyperparameters(
    code: str,
    hyperparams: List[HyperparameterSpec],
    new_values: Dict[str, Any],
) -> str:
    """
    Inject new hyperparameter values into operator code.

    Uses the line_pattern from each hyperparameter spec to find and replace values.

    Args:
        code: Original operator code
        hyperparams: List of HyperparameterSpec (with line_pattern attribute)
        new_values: Dict mapping hyperparameter name to new value

    Returns:
        Modified code with new hyperparameter values
    """
    modified_code = code

    for hp in hyperparams:
        if hp.name not in new_values:
            continue

        new_value = new_values[hp.name]
        old_pattern = getattr(hp, 'line_pattern', None)

        if not old_pattern:
            continue

        # Skip if current_value is None or not a valid number
        old_value = hp.current_value
        if old_value is None:
            continue

        # Verify that old_value is actually a number
        try:
            if hp.param_type == "int":
                int(old_value)
            elif hp.param_type == "float":
                float(old_value)
        except (ValueError, TypeError):
            # old_value is not a valid number (e.g., "self.max_size")
            continue

        # Create new pattern by replacing the old value with new value
        # Handle different value types
        try:
            if hp.param_type == "int":
                new_value_str = str(int(new_value))
            elif hp.param_type == "float":
                # Format float nicely
                if abs(float(new_value)) < 0.01 or abs(float(new_value)) > 1000:
                    new_value_str = f"{float(new_value):.6e}"
                else:
                    new_value_str = f"{float(new_value):.6f}".rstrip('0').rstrip('.')
            else:
                new_value_str = repr(new_value)
        except (ValueError, TypeError):
            continue

        # Try to replace the value in the pattern
        # Look for the current value in the pattern and replace it
        if hp.param_type == "int":
            old_value_pattern = str(int(old_value))
        elif hp.param_type == "float":
            # Match various float representations
            old_value_pattern = str(old_value)
        else:
            old_value_pattern = repr(old_value)

        # Create replacement pattern
        new_pattern = old_pattern.replace(old_value_pattern, new_value_str)

        # Replace in code
        if old_pattern in modified_code:
            modified_code = modified_code.replace(old_pattern, new_pattern, 1)

    return modified_code


def _is_valid_hyperparameter(hp: HyperparameterSpec) -> bool:
    """Check if a hyperparameter has valid numeric values for optimization."""
    if hp.current_value is None:
        return False
    try:
        if hp.param_type == "int":
            int(hp.current_value)
            if hp.min_value is not None:
                int(hp.min_value)
            if hp.max_value is not None:
                int(hp.max_value)
        elif hp.param_type == "float":
            float(hp.current_value)
            if hp.min_value is not None:
                float(hp.min_value)
            if hp.max_value is not None:
                float(hp.max_value)
        elif hp.param_type == "categorical":
            if not hp.choices:
                return False
        return True
    except (ValueError, TypeError):
        return False


def _sample_params_from_trial(
    trial,
    valid_hyperparams_by_type: Dict[str, List[HyperparameterSpec]],
) -> Dict[str, Dict[str, Any]]:
    """Sample hyperparameters from an Optuna trial for all operator types."""
    params_by_type = {}
    for op_type, valid_hyperparams in valid_hyperparams_by_type.items():
        params = {}
        for hp in valid_hyperparams:
            # Prefix param name with operator type to avoid collisions
            param_name = f"{op_type}__{hp.name}"
            if hp.param_type == "float":
                if hp.log_scale:
                    params[hp.name] = trial.suggest_float(
                        param_name, float(hp.min_value), float(hp.max_value), log=True
                    )
                else:
                    params[hp.name] = trial.suggest_float(
                        param_name, float(hp.min_value), float(hp.max_value)
                    )
            elif hp.param_type == "int":
                if hp.log_scale:
                    params[hp.name] = trial.suggest_int(
                        param_name, int(hp.min_value), int(hp.max_value), log=True
                    )
                else:
                    params[hp.name] = trial.suggest_int(
                        param_name, int(hp.min_value), int(hp.max_value)
                    )
            elif hp.param_type == "categorical":
                params[hp.name] = trial.suggest_categorical(param_name, hp.choices)
        params_by_type[op_type] = params
    return params_by_type


def _create_bundle_from_params(
    bundle: OperatorBundle,
    params_by_type: Dict[str, Dict[str, Any]],
    valid_hyperparams_by_type: Dict[str, List[HyperparameterSpec]],
    original_codes: Dict[str, str],
) -> Optional[OperatorBundle]:
    """Create a new bundle with injected hyperparameters. Returns None if any operator fails validation."""
    new_operators = {}
    for op_type, params in params_by_type.items():
        new_code = inject_hyperparameters(
            original_codes[op_type],
            valid_hyperparams_by_type[op_type],
            params
        )
        try:
            new_op, passed, error = create_and_test_operator(new_code, op_type)
            if not passed:
                return None
            new_operators[op_type] = new_op
        except Exception:
            return None

    return OperatorBundle(
        selection=new_operators.get("selection", bundle.selection),
        mutation=new_operators.get("mutation", bundle.mutation),
        crossover=new_operators.get("crossover", bundle.crossover),
        fitness=new_operators.get("fitness", bundle.fitness),
    )


def create_optuna_study_joint(
    bundle: OperatorBundle,
    hyperparams_by_type: Dict[str, List[HyperparameterSpec]],
    batch_evaluate_fn,
    n_trials: int = 50,
    batch_size: int = 1,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """
    Run Optuna optimization to tune hyperparameters for ALL operators jointly.

    Uses ask/tell API to enable batch evaluation of multiple HP configurations
    in parallel (e.g., via SLURM job arrays).

    Args:
        bundle: The OperatorBundle containing operators to tune
        hyperparams_by_type: Dict mapping operator_type to list of hyperparameter specs
        batch_evaluate_fn: Function that takes a list of OperatorBundles and returns
                          a list of scores (higher is better). Invalid bundles should
                          receive score of -inf.
        n_trials: Total number of optimization trials
        batch_size: Number of trials to evaluate in parallel per batch
        seed: Random seed for reproducibility

    Returns:
        (best_params_by_type, best_score): Dict of best params per operator type and best score
    """
    # Filter out invalid hyperparameters for each operator type
    valid_hyperparams_by_type = {}
    for op_type, hyperparams in hyperparams_by_type.items():
        valid_hps = [hp for hp in hyperparams if _is_valid_hyperparameter(hp)]
        if valid_hps:
            valid_hyperparams_by_type[op_type] = valid_hps

    if not valid_hyperparams_by_type:
        print("No valid hyperparameters to tune across any operator")
        return {}, float('-inf')

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter tuning. "
            "Install it with: pip install optuna"
        )

    # Store original codes for each operator type
    original_codes = {
        op_type: bundle.get_operator(op_type).code
        for op_type in valid_hyperparams_by_type
    }

    # Create study
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Enqueue initial trial with current values
    initial_params = {}
    for op_type, valid_hyperparams in valid_hyperparams_by_type.items():
        for hp in valid_hyperparams:
            param_name = f"{op_type}__{hp.name}"
            if hp.param_type == "int":
                initial_params[param_name] = int(hp.current_value)
            elif hp.param_type == "float":
                initial_params[param_name] = float(hp.current_value)
            else:
                initial_params[param_name] = hp.current_value
    study.enqueue_trial(initial_params)

    # Run optimization using ask/tell for batch evaluation
    trials_completed = 0
    batch_num = 0

    while trials_completed < n_trials:
        batch_num += 1
        current_batch_size = min(batch_size, n_trials - trials_completed)

        # Ask for batch_size trial suggestions
        trials = []
        params_list = []
        bundles_to_eval = []
        trial_indices = []  # Track which trials correspond to which bundles

        for i in range(current_batch_size):
            trial = study.ask()
            trials.append(trial)

            # Sample parameters for this trial
            params_by_type = _sample_params_from_trial(trial, valid_hyperparams_by_type)
            params_list.append(params_by_type)

            # Create bundle with these parameters
            new_bundle = _create_bundle_from_params(
                bundle, params_by_type, valid_hyperparams_by_type, original_codes
            )

            if new_bundle is not None:
                bundles_to_eval.append(new_bundle)
                trial_indices.append(i)
            else:
                # Bundle creation failed - immediately tell Optuna
                study.tell(trial, float('-inf'))

        # Batch evaluate all valid bundles
        if bundles_to_eval:
            print(f"  HP tuning batch {batch_num}: evaluating {len(bundles_to_eval)} configurations...")
            try:
                scores = batch_evaluate_fn(bundles_to_eval)
            except Exception as e:
                print(f"  Batch evaluation failed: {e}")
                scores = [float('-inf')] * len(bundles_to_eval)

            # Report results back to Optuna
            for idx, score in zip(trial_indices, scores):
                study.tell(trials[idx], score)

        trials_completed += current_batch_size

        # Progress update
        if study.best_trial is not None:
            print(f"  HP tuning progress: {trials_completed}/{n_trials} trials, best score: {study.best_value:.4f}")

    # Convert best_params back to per-operator format
    best_params_by_type = {op_type: {} for op_type in valid_hyperparams_by_type}
    if study.best_trial is not None:
        for param_name, value in study.best_params.items():
            # Parse "operator_type__param_name" format
            op_type, hp_name = param_name.split("__", 1)
            best_params_by_type[op_type][hp_name] = value
        return best_params_by_type, study.best_value
    else:
        return best_params_by_type, float('-inf')


def create_optuna_study(
    bundle: OperatorBundle,
    operator_type: str,
    hyperparams: List[HyperparameterSpec],
    evaluate_fn,
    n_trials: int = 20,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Run Optuna optimization to tune hyperparameters.

    Args:
        bundle: The OperatorBundle containing the operator to tune
        operator_type: Which operator type to tune
        hyperparams: List of hyperparameter specifications
        evaluate_fn: Function that takes an OperatorBundle and returns a score (higher is better)
        n_trials: Number of optimization trials
        seed: Random seed for reproducibility

    Returns:
        (best_params, best_score): Best hyperparameter values and corresponding score
    """
    # Filter out invalid hyperparameters
    valid_hyperparams = [hp for hp in hyperparams if _is_valid_hyperparameter(hp)]
    if not valid_hyperparams:
        print("No valid hyperparameters to tune")
        return {}, float('-inf')

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter tuning. "
            "Install it with: pip install optuna"
        )

    original_operator = bundle.get_operator(operator_type)
    original_code = original_operator.code

    def objective(trial):
        # Sample hyperparameters
        params = {}
        for hp in valid_hyperparams:
            if hp.param_type == "float":
                if hp.log_scale:
                    params[hp.name] = trial.suggest_float(
                        hp.name, float(hp.min_value), float(hp.max_value), log=True
                    )
                else:
                    params[hp.name] = trial.suggest_float(
                        hp.name, float(hp.min_value), float(hp.max_value)
                    )
            elif hp.param_type == "int":
                if hp.log_scale:
                    params[hp.name] = trial.suggest_int(
                        hp.name, int(hp.min_value), int(hp.max_value), log=True
                    )
                else:
                    params[hp.name] = trial.suggest_int(
                        hp.name, int(hp.min_value), int(hp.max_value)
                    )
            elif hp.param_type == "categorical":
                params[hp.name] = trial.suggest_categorical(hp.name, hp.choices)

        # Inject new values into code
        new_code = inject_hyperparameters(original_code, valid_hyperparams, params)

        # Try to create and test the operator
        try:
            new_op, passed, error = create_and_test_operator(new_code, operator_type)
            if not passed:
                return float('-inf')  # Invalid operator
        except Exception as e:
            return float('-inf')

        # Create new bundle with modified operator
        new_bundle = OperatorBundle(
            selection=bundle.selection if operator_type != "selection" else new_op,
            mutation=bundle.mutation if operator_type != "mutation" else new_op,
            crossover=bundle.crossover if operator_type != "crossover" else new_op,
            fitness=bundle.fitness if operator_type != "fitness" else new_op,
        )

        # Evaluate
        try:
            score = evaluate_fn(new_bundle)
            return score
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return float('-inf')

    # Create and run study
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Add initial trial with current values
    initial_params = {}
    for hp in valid_hyperparams:
        if hp.param_type == "int":
            initial_params[hp.name] = int(hp.current_value)
        elif hp.param_type == "float":
            initial_params[hp.name] = float(hp.current_value)
        else:
            initial_params[hp.name] = hp.current_value
    study.enqueue_trial(initial_params)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study.best_value


def tune_bundle_hyperparameters(
    bundle: OperatorBundle,
    batch_evaluate_fn,
    operator_types: Optional[List[str]] = None,
    n_trials: int = 50,
    n_trials_per_operator: int = 10,
    batch_size: int = 1,
    tune_jointly: bool = True,
    model: str = "openai/gpt-5-mini",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[OperatorBundle, Dict[str, Dict[str, Any]]]:
    """
    Tune hyperparameters for all operators in a bundle.

    This is the main entry point for hyperparameter tuning.

    Args:
        bundle: The OperatorBundle to tune
        batch_evaluate_fn: Function that takes a list of bundles and returns a list of scores.
                          For single-bundle evaluation, wrap your function:
                          `lambda bundles: [evaluate_fn(b) for b in bundles]`
        operator_types: Which operator types to tune (default: all)
        n_trials: Number of optimization trials when tuning jointly (tune_jointly=True)
        n_trials_per_operator: Number of optimization trials per operator (tune_jointly=False)
        batch_size: Number of HP configurations to evaluate in parallel per batch.
                   Higher values = faster tuning if you have parallel evaluation (e.g., SLURM).
        tune_jointly: If True (default), tune all operators' hyperparameters simultaneously.
                      If False, tune each operator's hyperparameters separately in sequence.
        model: LLM model for hyperparameter identification
        seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        (tuned_bundle, all_best_params): Tuned bundle and dict of best params per operator type
    """
    if operator_types is None:
        operator_types = OPERATOR_TYPES

    if tune_jointly:
        return _tune_bundle_jointly(
            bundle, batch_evaluate_fn, operator_types, n_trials, batch_size, model, seed, verbose
        )
    else:
        return _tune_bundle_sequentially(
            bundle, batch_evaluate_fn, operator_types, n_trials_per_operator, model, seed, verbose
        )


def _tune_bundle_jointly(
    bundle: OperatorBundle,
    batch_evaluate_fn,
    operator_types: List[str],
    n_trials: int,
    batch_size: int,
    model: str,
    seed: Optional[int],
    verbose: bool,
) -> Tuple[OperatorBundle, Dict[str, Dict[str, Any]]]:
    """
    Tune hyperparameters for all operators simultaneously in a single optimization.
    Uses batch evaluation to enable parallel HP configuration testing.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Joint hyperparameter tuning for all operators")
        print(f"{'='*60}")

    # Step 1: Identify hyperparameters for all operators
    hyperparams_by_type = {}
    for op_type in operator_types:
        if verbose:
            print(f"\nIdentifying hyperparameters for {op_type}...")
        operator = bundle.get_operator(op_type)
        hyperparams = identify_hyperparameters(operator, model=model)

        if hyperparams:
            hyperparams_by_type[op_type] = hyperparams
            if verbose:
                print(f"  Found {len(hyperparams)} hyperparameters:")
                for hp in hyperparams:
                    print(f"    - {hp.name}: {hp.current_value} ({hp.param_type})")
        else:
            if verbose:
                print(f"  No tunable hyperparameters found")

    if not hyperparams_by_type:
        if verbose:
            print("\nNo tunable hyperparameters found in any operator")
        return bundle, {}

    # Count total hyperparameters
    total_hps = sum(len(hps) for hps in hyperparams_by_type.values())
    if verbose:
        print(f"\nTotal hyperparameters to tune: {total_hps}")
        print(f"Running joint optimization ({n_trials} trials, batch_size={batch_size})...")

    # Step 2: Run joint optimization with batch evaluation
    best_params_by_type, best_score = create_optuna_study_joint(
        bundle,
        hyperparams_by_type,
        batch_evaluate_fn,
        n_trials=n_trials,
        batch_size=batch_size,
        seed=seed,
    )

    if verbose:
        print(f"\nBest score: {best_score:.4f}")
        print("Best params by operator:")
        for op_type, params in best_params_by_type.items():
            if params:
                print(f"  {op_type}: {params}")

    # Step 3: Apply best params to create tuned bundle
    current_bundle = bundle
    for op_type, best_params in best_params_by_type.items():
        if not best_params:
            continue

        operator = bundle.get_operator(op_type)
        tuned_code = inject_hyperparameters(
            operator.code,
            hyperparams_by_type[op_type],
            best_params
        )

        try:
            tuned_op, passed, error = create_and_test_operator(tuned_code, op_type)
            if passed:
                current_bundle = OperatorBundle(
                    selection=current_bundle.selection if op_type != "selection" else tuned_op,
                    mutation=current_bundle.mutation if op_type != "mutation" else tuned_op,
                    crossover=current_bundle.crossover if op_type != "crossover" else tuned_op,
                    fitness=current_bundle.fitness if op_type != "fitness" else tuned_op,
                )
                if verbose:
                    print(f"Applied tuned hyperparameters to {op_type}")
            else:
                if verbose:
                    print(f"Warning: Tuned {op_type} operator failed validation, keeping original")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to create tuned {op_type} operator: {e}")

    return current_bundle, best_params_by_type


def _tune_bundle_sequentially(
    bundle: OperatorBundle,
    batch_evaluate_fn,
    operator_types: List[str],
    n_trials_per_operator: int,
    model: str,
    seed: Optional[int],
    verbose: bool,
) -> Tuple[OperatorBundle, Dict[str, Dict[str, Any]]]:
    """
    Tune hyperparameters for each operator one at a time in sequence.
    Note: Sequential tuning doesn't benefit from batch evaluation,
    so we wrap batch_evaluate_fn for single-bundle calls.
    """
    current_bundle = bundle
    all_best_params = {}

    # Wrap batch_evaluate_fn for single-bundle evaluation
    def single_evaluate_fn(b):
        scores = batch_evaluate_fn([b])
        return scores[0] if scores else float('-inf')

    for op_type in operator_types:
        if verbose:
            print(f"\n--- Tuning {op_type} operator hyperparameters ---")

        operator = current_bundle.get_operator(op_type)

        # Step 1: Identify hyperparameters
        if verbose:
            print(f"  Identifying hyperparameters...")
        hyperparams = identify_hyperparameters(operator, model=model)

        if not hyperparams:
            if verbose:
                print(f"  No tunable hyperparameters found in {op_type} operator")
            continue

        if verbose:
            print(f"  Found {len(hyperparams)} tunable hyperparameters:")
            for hp in hyperparams:
                print(f"    - {hp.name}: {hp.current_value} ({hp.param_type})")

        # Step 2: Optimize
        if verbose:
            print(f"  Running optimization ({n_trials_per_operator} trials)...")

        best_params, best_score = create_optuna_study(
            current_bundle,
            op_type,
            hyperparams,
            single_evaluate_fn,
            n_trials=n_trials_per_operator,
            seed=seed,
        )

        all_best_params[op_type] = best_params

        if verbose:
            print(f"  Best params: {best_params}")
            print(f"  Best score: {best_score:.4f}")

        # Step 3: Apply best params to create tuned operator
        if best_params:
            original_code = operator.code
            tuned_code = inject_hyperparameters(original_code, hyperparams, best_params)

            try:
                tuned_op, passed, error = create_and_test_operator(tuned_code, op_type)
                if passed:
                    # Update current bundle
                    current_bundle = OperatorBundle(
                        selection=current_bundle.selection if op_type != "selection" else tuned_op,
                        mutation=current_bundle.mutation if op_type != "mutation" else tuned_op,
                        crossover=current_bundle.crossover if op_type != "crossover" else tuned_op,
                        fitness=current_bundle.fitness if op_type != "fitness" else tuned_op,
                    )
                    if verbose:
                        print(f"  Successfully applied tuned hyperparameters to {op_type}")
                else:
                    if verbose:
                        print(f"  Warning: Tuned operator failed validation, keeping original")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to create tuned operator: {e}")

    return current_bundle, all_best_params


# Convenience function for testing
def test_hyperparameter_identification():
    """Test hyperparameter identification on default operators."""
    from meta_evolution import get_default_operator

    for op_type in OPERATOR_TYPES:
        print(f"\n{'='*60}")
        print(f"Testing hyperparameter identification for: {op_type}")
        print(f"{'='*60}")

        operator = get_default_operator(op_type)
        print(f"Operator code:\n{operator.code}\n")

        hyperparams = identify_hyperparameters(operator)

        if hyperparams:
            print(f"Found {len(hyperparams)} hyperparameters:")
            for hp in hyperparams:
                print(f"  - {hp.name}:")
                print(f"      current_value: {hp.current_value}")
                print(f"      type: {hp.param_type}")
                if hp.min_value is not None:
                    print(f"      range: [{hp.min_value}, {hp.max_value}]")
                if hp.choices:
                    print(f"      choices: {hp.choices}")
                print(f"      description: {hp.description}")
        else:
            print("No hyperparameters found")


if __name__ == "__main__":
    test_hyperparameter_identification()
