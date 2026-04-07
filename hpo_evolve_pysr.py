"""
Hyperparameter tuning for evolved Julia operators in evolve_pysr.

Tunes two types of hyperparameters:
1. Operator-specific: Numeric constants inside Julia operator code, identified by LLM
2. Base PySR: PySR configuration params (crossover_probability, parsimony, etc.)

Each bundle gets a persistent Optuna study (SQLite-backed) so tuning resumes
across generations.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

from completions import chat_completion, get_content
from hyperparameter_tuning import HyperparameterSpec, inject_hyperparameters
from hpo_pysr import (
    HPO_PARAM_SPECS,
    _filter_active_search_space,
    _split_hpo_params,
    create_param_config_from_trial,
)
from parallel_eval_pysr import PySRConfig, PySRSlurmEvaluator, get_default_mutation_weights

# Default base PySR params to tune (the ones actively used in hpo_pysr.py)
DEFAULT_BASE_HPO_PARAMS = [
    "maxsize",
    "population_size",
    "populations",
    "ncycles_per_iteration",
    "parsimony",
    "optimize_probability",
    "crossover_probability",
    "adaptive_parsimony_scaling",
    "maxdepth",
]

# Julia-adapted version of the hyperparameter identification prompt
JULIA_HPARAM_IDENTIFICATION_PROMPT = """You are analyzing a Julia operator function used within PySR (SymbolicRegression.jl), an evolutionary symbolic regression system. Your task is to identify tunable hyperparameters in the operator code.

## Context: PySR / SymbolicRegression.jl

PySR evolves mathematical expressions (expression trees) to fit data using:
- **Mutation operators**: Modify expression trees (add/delete nodes, change constants, etc.)
- **Survival operators**: Decide which expressions survive to the next generation
- **Selection operators**: Choose parent expressions for breeding

## Operator Being Analyzed

**Operator Type:** {operator_type}

```julia
{code}
```

## Your Task

Identify ALL numeric constants and thresholds in this Julia code that could be tuned to improve performance. These are the "hyperparameters" of this operator.

For each hyperparameter, provide:
1. `name`: A descriptive name (snake_case)
2. `line_pattern`: The exact code snippet containing the value (for string matching/replacement)
3. `current_value`: The current numeric value in the code (must be a literal number, not a variable)
4. `param_type`: One of "float", "int", or "categorical"
5. `min_value`: Minimum reasonable value (for float/int)
6. `max_value`: Maximum reasonable value (for float/int)
7. `log_scale`: Whether to search on log scale (true for values spanning orders of magnitude)
8. `choices`: List of options (for categorical only)
9. `description`: Brief description of what this parameter controls

## Guidelines

- **Include**: probabilities (0-1), thresholds, multipliers, exponents, size limits, counts, penalty weights, temperature values
- **Exclude**: loop indices, array indices, string constants, variable references
- Be conservative with ranges — suggest reasonable bounds based on the parameter's role
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


# =============================================================================
# Hyperparameter Identification
# =============================================================================

def identify_julia_hparams(
    code: str,
    op_type_name: str,
    model: str = "openai/gpt-5-mini",
    llm_temperature: float = 0.0,
) -> List[HyperparameterSpec]:
    """
    Use LLM to identify tunable hyperparameters in Julia operator code.

    Args:
        code: Julia source code of the operator
        op_type_name: Type of operator ("mutation", "survival", "selection")
        model: LLM model to use
        llm_temperature: Temperature for LLM

    Returns:
        List of HyperparameterSpec objects (with line_pattern attribute set)
    """
    prompt = JULIA_HPARAM_IDENTIFICATION_PROMPT.format(
        code=code,
        operator_type=op_type_name,
    )

    response = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert code analyst. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=llm_temperature,
    )

    content = get_content(response)

    try:
        # Extract JSON from code block if present
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = content.strip()

        hyperparams_data = json.loads(json_str)

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
            spec.line_pattern = hp.get("line_pattern", "")
            specs.append(spec)

        # Filter invalid specs
        filtered = []
        for spec in specs:
            line_pattern = getattr(spec, 'line_pattern', '')
            if not line_pattern:
                continue
            # Validate numeric values
            try:
                if spec.param_type == "int":
                    int(spec.current_value)
                elif spec.param_type == "float":
                    float(spec.current_value)
                elif spec.param_type == "categorical":
                    if not spec.choices:
                        continue
            except (ValueError, TypeError):
                continue
            filtered.append(spec)

        return filtered

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  Warning: Failed to parse hyperparameter response: {e}")
        return []


def _specs_to_dicts(specs: List[HyperparameterSpec]) -> List[Dict]:
    """Convert HyperparameterSpec list to serializable dicts (for caching on JuliaOperator)."""
    result = []
    for s in specs:
        d = {
            "name": s.name,
            "param_type": s.param_type,
            "current_value": s.current_value,
            "min_value": s.min_value,
            "max_value": s.max_value,
            "choices": s.choices,
            "log_scale": s.log_scale,
            "description": s.description,
            "line_pattern": getattr(s, "line_pattern", ""),
        }
        result.append(d)
    return result


def _dicts_to_specs(dicts: List[Dict]) -> List[HyperparameterSpec]:
    """Convert cached dicts back to HyperparameterSpec objects."""
    specs = []
    for d in dicts:
        spec = HyperparameterSpec(
            name=d["name"],
            param_type=d["param_type"],
            current_value=d["current_value"],
            min_value=d.get("min_value"),
            max_value=d.get("max_value"),
            choices=d.get("choices"),
            log_scale=d.get("log_scale", False),
            description=d.get("description", ""),
        )
        spec.line_pattern = d.get("line_pattern", "")
        specs.append(spec)
    return specs


# =============================================================================
# Base PySR Search Space
# =============================================================================

def get_base_hpo_search_space(
    param_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return the base PySR HPO search space.

    Args:
        param_names: Specific params to include. If None, uses DEFAULT_BASE_HPO_PARAMS.

    Returns:
        Dict mapping param name to HPOParamSpec.
    """
    names = param_names or DEFAULT_BASE_HPO_PARAMS
    return _filter_active_search_space(names)


# =============================================================================
# Bundle HPO Tuning
# =============================================================================

def _get_or_identify_hparams(
    operator,  # JuliaOperator
    op_type_name: str,
    model: str,
) -> List[HyperparameterSpec]:
    """Get cached hparams or identify them via LLM."""
    if operator.hp_specs is not None:
        return _dicts_to_specs(operator.hp_specs)

    specs = identify_julia_hparams(
        code=operator.code,
        op_type_name=op_type_name,
        model=model,
    )
    # Cache on operator
    operator.hp_specs = _specs_to_dicts(specs)
    return specs


def _sample_trial_params(
    trial,
    base_search_space: Dict,
    op_hparams: Dict[str, List[HyperparameterSpec]],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Sample both base PySR params and operator-specific params from an Optuna trial.

    Returns:
        (base_params, op_params_by_type) where:
        - base_params: Dict of base PySR hparam values
        - op_params_by_type: Dict mapping op_type -> {hp_name: value}
    """
    # Sample base PySR params
    base_params = create_param_config_from_trial(trial, base_search_space)

    # Sample operator-specific params
    op_params_by_type = {}
    for op_type, specs in op_hparams.items():
        params = {}
        for hp in specs:
            param_name = f"op_{op_type}__{hp.name}"
            if hp.param_type == "float":
                params[hp.name] = trial.suggest_float(
                    param_name,
                    float(hp.min_value),
                    float(hp.max_value),
                    log=hp.log_scale,
                )
            elif hp.param_type == "int":
                params[hp.name] = trial.suggest_int(
                    param_name,
                    int(hp.min_value),
                    int(hp.max_value),
                    log=hp.log_scale,
                )
            elif hp.param_type == "categorical":
                params[hp.name] = trial.suggest_categorical(param_name, hp.choices)
        if params:
            op_params_by_type[op_type] = params

    return base_params, op_params_by_type


def _build_trial_config(
    bundle,  # OperatorBundle
    base_params: Dict[str, Any],
    op_params_by_type: Dict[str, Dict[str, Any]],
    op_hparams: Dict[str, List[HyperparameterSpec]],
    pysr_kwargs: Dict,
) -> Optional[PySRConfig]:
    """Build a PySRConfig for one HPO trial, injecting operator hparams into code.

    Returns None if code injection produces invalid code.
    """
    from evolve_pysr import validate_julia_code, OPERATOR_TYPES

    mutation_weights = get_default_mutation_weights()
    config_kwargs = {}

    for op_type_name in ["mutation", "survival", "selection"]:
        op = bundle.operators.get(op_type_name)
        if op is None:
            if op_type_name == "mutation":
                for i in range(1, 6):
                    mutation_weights[f"weight_custom_mutation_{i}"] = 0.0
                config_kwargs["allow_custom_mutations"] = False
            continue

        # Inject operator-specific hparams if we have them
        code = op.code
        if op_type_name in op_params_by_type and op_type_name in op_hparams:
            code = inject_hyperparameters(
                code, op_hparams[op_type_name], op_params_by_type[op_type_name]
            )
            # Validate the modified code before committing to an expensive eval
            op_type_obj = OPERATOR_TYPES[op_type_name]
            is_valid, error = validate_julia_code(op.name, code, op_type_obj)
            if not is_valid:
                return None

        if op_type_name == "mutation":
            weight = op.weight if op.weight is not None else 0.5
            mutation_weights["weight_custom_mutation_1"] = weight
            config_kwargs["custom_mutation_code"] = {op.name: code}
            config_kwargs["allow_custom_mutations"] = True
        elif op_type_name == "survival":
            config_kwargs["custom_survival_code"] = code
        elif op_type_name == "selection":
            config_kwargs["custom_selection_code"] = code

    # Merge base HPO params into pysr_kwargs / mutation_weights
    hpo_mutation_weights, hpo_pysr_kwargs = _split_hpo_params(base_params)
    mutation_weights.update(hpo_mutation_weights)
    merged_pysr_kwargs = {**pysr_kwargs, **hpo_pysr_kwargs}

    name_parts = []
    for t in ["mutation", "survival", "selection"]:
        op = bundle.operators.get(t)
        name_parts.append(op.name if op else "default")
    name = "__".join(name_parts) + "_hpo"

    return PySRConfig(
        mutation_weights=mutation_weights,
        pysr_kwargs=merged_pysr_kwargs,
        name=name,
        **config_kwargs,
    )


def tune_bundle(
    bundle,  # OperatorBundle
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    operator_type_names: List[str],
    n_trials: int,
    seed: int,
    output_dir: str,
    model: str = "openai/gpt-5-mini",
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
    fitness_metric: str = "gt",
    base_hpo_params: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[Any, float]:  # Returns (OperatorBundle, best_score)
    """
    Run HPO trials on a single bundle.

    Tunes both operator-specific hparams (in Julia code) and base PySR hparams.
    Uses a persistent Optuna study (SQLite) so tuning can resume across generations.

    Args:
        bundle: The OperatorBundle to tune
        evaluator: PySRSlurmEvaluator for running evaluations
        dataset_names: Datasets to evaluate on
        pysr_kwargs: Base PySR configuration
        operator_type_names: Which operator types are in play
        n_trials: Number of new Optuna trials to run
        seed: Random seed
        output_dir: Directory for Optuna study storage
        model: LLM model for hparam identification
        n_runs: Number of runs per config per dataset
        target_noise_map: Per-dataset noise levels
        fitness_metric: "r2" or "gt"
        base_hpo_params: Base PySR params to tune (None = defaults)
        verbose: Print progress

    Returns:
        (updated_bundle, best_score): Bundle with best hparams applied, and best score
    """
    if optuna is None:
        raise ImportError("optuna is required for HPO. Install with: pip install optuna")

    # Step 1: Identify operator-specific hparams for each operator in the bundle
    op_hparams: Dict[str, List[HyperparameterSpec]] = {}
    for op_type_name in operator_type_names:
        op = bundle.operators.get(op_type_name)
        if op is None:
            continue
        specs = _get_or_identify_hparams(op, op_type_name, model)
        if specs:
            op_hparams[op_type_name] = specs
            if verbose:
                print(f"    {op_type_name}: {len(specs)} tunable hparams: "
                      f"{', '.join(s.name for s in specs)}")

    # Step 2: Get base PySR search space
    base_search_space = get_base_hpo_search_space(base_hpo_params)

    total_params = len(base_search_space) + sum(len(s) for s in op_hparams.values())
    if total_params == 0:
        if verbose:
            print("    No hyperparameters to tune")
        return bundle, bundle.score if bundle.score is not None else float('-inf')

    if verbose:
        print(f"    Tuning {total_params} params "
              f"({len(base_search_space)} base + "
              f"{sum(len(s) for s in op_hparams.values())} operator-specific) "
              f"for {n_trials} trials")

    # Step 3: Create or load Optuna study
    study_dir = Path(output_dir) / "hp_studies"
    study_dir.mkdir(parents=True, exist_ok=True)

    # Use a sanitized bundle name for the study
    study_name = bundle.display_name.replace(" | ", "_").replace(" ", "_")
    db_path = study_dir / f"{study_name}.db"
    storage = f"sqlite:///{db_path}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # Enqueue initial trial with current values if study is fresh
    if len(study.trials) == 0:
        initial_params = {}
        # Base params: use current pysr_kwargs values or defaults
        for name, spec in base_search_space.items():
            if spec.kind == "categorical" or spec.kind == "bool":
                # For categorical, find the label matching the default
                choice_map = dict(spec.choices)
                reverse_map = {v: k for k, v in choice_map.items()}
                default_label = reverse_map.get(spec.default)
                if default_label is not None:
                    initial_params[name] = default_label
                else:
                    initial_params[name] = spec.choices[0][0]
            else:
                initial_params[name] = spec.default

        # Operator-specific params: use current values
        for op_type, specs in op_hparams.items():
            for hp in specs:
                param_name = f"op_{op_type}__{hp.name}"
                initial_params[param_name] = hp.current_value

        try:
            study.enqueue_trial(initial_params)
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not enqueue initial trial: {e}")

    # Step 4: Run n_trials
    from evolve_pysr import _evaluate_configs_with_noise_map

    best_score = bundle.score if bundle.score is not None else float('-inf')
    best_base_params = None
    best_op_params = None

    for trial_idx in range(n_trials):
        trial = study.ask()

        try:
            base_params, op_params = _sample_trial_params(
                trial, base_search_space, op_hparams
            )
        except Exception as e:
            study.tell(trial, float('-inf'))
            if verbose:
                print(f"    Trial {trial_idx + 1}: param sampling failed: {e}")
            continue

        config = _build_trial_config(
            bundle, base_params, op_params, op_hparams, pysr_kwargs
        )

        if config is None:
            study.tell(trial, float('-inf'))
            if verbose:
                print(f"    Trial {trial_idx + 1}: config build failed")
            continue

        try:
            results = _evaluate_configs_with_noise_map(
                evaluator=evaluator,
                configs=[config],
                dataset_names=dataset_names,
                seed=seed,
                n_runs=n_runs,
                target_noise_map=target_noise_map,
                fitness_metric=fitness_metric,
            )
            score = results[0][0]  # avg score
        except Exception as e:
            score = float('-inf')
            if verbose:
                print(f"    Trial {trial_idx + 1}: evaluation failed: {e}")

        study.tell(trial, score)

        if verbose:
            print(f"    Trial {trial_idx + 1}/{n_trials}: score={score:.4f}")

        if score > best_score:
            best_score = score
            best_base_params = base_params
            best_op_params = op_params

    # Step 5: Apply best params to bundle
    if best_base_params is not None:
        # Merge best base params + operator hparam values into best_hparams
        all_best = dict(best_base_params)
        # Also store operator-specific param values for reference
        if best_op_params:
            for op_type, params in best_op_params.items():
                for hp_name, val in params.items():
                    all_best[f"op_{op_type}__{hp_name}"] = val

        bundle.best_hparams = all_best

        # Inject best operator hparams into operator code
        if best_op_params:
            for op_type, params in best_op_params.items():
                op = bundle.operators.get(op_type)
                if op is None or op_type not in op_hparams:
                    continue
                new_code = inject_hyperparameters(
                    op.code, op_hparams[op_type], params
                )
                op.code = new_code
                # Update hp_specs to reflect new current values
                for spec_dict in (op.hp_specs or []):
                    if spec_dict["name"] in params:
                        spec_dict["current_value"] = params[spec_dict["name"]]

        bundle.score = best_score

        if verbose:
            print(f"    HPO improved score to {best_score:.4f}")
    else:
        if verbose:
            print(f"    HPO did not improve score (best: {best_score:.4f})")

    return bundle, best_score


def tune_population(
    population: List,  # List[OperatorBundle]
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    pysr_kwargs: Dict,
    operator_type_names: List[str],
    n_trials: int,
    seed: int,
    output_dir: str,
    model: str = "openai/gpt-5-mini",
    n_runs: int = 1,
    target_noise_map: Optional[Dict[str, float]] = None,
    fitness_metric: str = "gt",
    base_hpo_params: Optional[List[str]] = None,
    verbose: bool = True,
) -> List:  # Returns List[OperatorBundle]
    """
    Run HPO tuning on each bundle in the population.

    Args:
        population: List of OperatorBundles to tune
        (other args same as tune_bundle)

    Returns:
        Updated population (same list, modified in-place), re-sorted by score
    """
    print(f"\n  HPO tuning: {n_trials} trials per bundle, {len(population)} bundles")

    for i, bundle in enumerate(population):
        print(f"\n  Bundle {i + 1}/{len(population)}: {bundle.display_name}")
        tune_bundle(
            bundle=bundle,
            evaluator=evaluator,
            dataset_names=dataset_names,
            pysr_kwargs=pysr_kwargs,
            operator_type_names=operator_type_names,
            n_trials=n_trials,
            seed=seed,
            output_dir=output_dir,
            model=model,
            n_runs=n_runs,
            target_noise_map=target_noise_map,
            fitness_metric=fitness_metric,
            base_hpo_params=base_hpo_params,
            verbose=verbose,
        )

    # Re-sort by score
    population.sort(key=lambda b: b.score if b.score is not None else -1, reverse=True)
    return population
