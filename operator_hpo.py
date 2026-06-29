"""
Hyperparameter tuning for evolved Julia operators (operator-internal constants only).

Identifies the most influential numeric constant inside each operator's Julia
code via an LLM call, then runs Optuna to tune those constants. Base PySR
hyperparameters (parsimony, niterations, etc.) are NOT tuned here — that's
hpo_pysr.py's job. This file is the operator-internal-constant counterpart.

Each bundle gets a persistent Optuna study (SQLite-backed) so tuning resumes
across generations.
"""

import json
import os
import re
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
from parallel_eval_pysr import PySRConfig, PySRSlurmEvaluator, get_default_mutation_weights

# Julia-adapted version of the hyperparameter identification prompt
JULIA_HPARAM_IDENTIFICATION_PROMPT = """You are analyzing a Julia operator function used within PySR (SymbolicRegression.jl), an evolutionary symbolic regression system. Your task is to identify the SINGLE most important tunable hyperparameter in the operator code.

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

Pick the ONE numeric constant in this Julia code that most influences operator behavior and is most worth tuning. Only one — the single most important. Prefer parameters that control the core behavior of the operator (tournament size, selection pressure, key probability/threshold, dominant multiplier) over cosmetic or safety-clamp values.

Provide:
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

- **Eligible**: probabilities (0-1), thresholds, multipliers, exponents, size limits, counts, penalty weights, temperature values
- **Ineligible**: loop indices, array indices, string constants, variable references
- Be conservative with ranges — suggest reasonable bounds based on the parameter's role
- For probabilities, use min=0.0, max=1.0
- For small positive values (e.g., 0.001 to 1.0), consider log_scale=true

## Output Format

Return a JSON array with exactly ONE element (the most important hyperparameter). Example:
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

If there are no tunable hyperparameters in this code, return: `[]`

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
    Use LLM to identify the single most important tunable hyperparameter in Julia
    operator code.

    Returns a list with at most one HyperparameterSpec (the most important param,
    per the LLM). If the LLM returns multiple, only the first valid one is kept.
    The list shape is preserved for compatibility with the rest of the HPO pipeline.

    Args:
        code: Julia source code of the operator
        op_type_name: Type of operator ("mutation", "survival", "selection", "loss")
        model: LLM model to use
        llm_temperature: Temperature for LLM

    Returns:
        List of HyperparameterSpec with 0 or 1 element (line_pattern attribute set).
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

        # Filter invalid specs, then keep only the first (most important) one.
        for spec in specs:
            line_pattern = getattr(spec, 'line_pattern', '')
            if not line_pattern:
                continue
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
            return [spec]

        return []

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
    op_hparams: Dict[str, List[HyperparameterSpec]],
) -> Dict[str, Dict[str, Any]]:
    """Sample operator-specific params from an Optuna trial.

    Returns op_params_by_type: Dict mapping op_type -> {hp_name: value}.
    """
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

    return op_params_by_type


def _build_trial_config(
    bundle,  # OperatorBundle
    op_params_by_type: Dict[str, Dict[str, Any]],
    op_hparams: Dict[str, List[HyperparameterSpec]],
    pysr_kwargs: Dict,
) -> Optional[PySRConfig]:
    """Build a PySRConfig for one HPO trial, injecting operator hparams into code.

    Returns None if code injection produces invalid code. `pysr_kwargs` is
    passed through unmodified — base PySR hparams are NOT tuned here.
    """
    from operator_types import validate_julia_code, OPERATOR_TYPES

    mutation_weights = get_default_mutation_weights()
    config_kwargs = {}

    for op_type_name in ["mutation", "survival", "selection", "loss"]:
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
        elif op_type_name == "loss":
            config_kwargs["custom_loss_code"] = code

    name_parts = []
    for t in ["mutation", "survival", "selection", "loss"]:
        op = bundle.operators.get(t)
        name_parts.append(op.name if op else "default")
    name = "__".join(name_parts) + "_hpo"

    return PySRConfig(
        mutation_weights=mutation_weights,
        pysr_kwargs=pysr_kwargs,
        name=name,
        **config_kwargs,
    )



# =============================================================================
# Operator HPO Loop (LLM-extracted operator-internal constants only)
# =============================================================================

def run_operator_hpo(
    n_trials: int,
    n_parallel: int,
    n_runs: int,
    dataset_names: List[str],
    baseline_bundle,
    seed: int,
    output_dir: str,
    pysr_kwargs: Dict[str, Any],
    slurm_partition: str,
    slurm_time_limit: str,
    slurm_mem_per_cpu: str,
    max_samples: int,
    job_timeout: float,
    max_concurrent_jobs: Optional[int],
    fitness_metric: str = "gt",
    use_cache: bool = True,
    continue_from: Optional[str] = None,
    wandb_run: Any = None,
    target_noise_map: Optional[Dict[str, float]] = None,
    operator_type_names: Optional[List[str]] = None,
    model: str = "openai/gpt-5-mini",
    max_op_hparams: int = 2,
) -> Tuple[Dict[str, Any], float, Any]:
    """HPO over LLM-extracted operator-internal numeric constants.

    `pysr_kwargs` is held fixed — base PySR hparam tuning is hpo_pysr.py's job.
    Each trial samples values for operator-internal numeric constants, rewrites
    the operator code via line-pattern injection, and evaluates the resulting
    bundle. Returns (best_combined_params, best_score, final_bundle).
    """
    import copy as _copy
    import optuna
    from optuna.samplers import TPESampler
    from hpo_pysr import HPOLogger, HPOTrialResult
    from wandb_utils import log_cpu_usage, log_wandb_summary

    if operator_type_names is None:
        operator_type_names = ["mutation", "survival", "selection", "loss"]

    np.random.seed(seed)

    # Phase 0: identify operator-specific hparams via LLM (cached on operator.hp_specs).
    print("=" * 60)
    print("Identifying operator-specific hyperparameters via LLM...")
    print("=" * 60)
    op_hparams: Dict[str, List] = {}
    for op_type_name in operator_type_names:
        op = baseline_bundle.operators.get(op_type_name)
        if op is None:
            continue
        specs = _get_or_identify_hparams(op, op_type_name, model)
        if max_op_hparams is not None and len(specs) > max_op_hparams:
            specs = specs[:max_op_hparams]
        if specs:
            op_hparams[op_type_name] = specs
            for s in specs:
                if s.param_type in ("int", "float"):
                    rng = f"range=[{s.min_value}, {s.max_value}]"
                else:
                    rng = f"choices={s.choices}"
                print(f"  {op_type_name}.{s.name} ({s.param_type}, "
                      f"current={s.current_value}, {rng})")
        else:
            print(f"  {op_type_name}: no tunable hparams found by LLM")

    n_op_params = sum(len(s) for s in op_hparams.values())
    if n_op_params == 0:
        raise ValueError("No tunable operator-internal constants identified by the LLM.")
    print(f"  Total operator-internal params to tune: {n_op_params}")

    metric_label = "R²" if fitness_metric == "r2" else "GT match rate"
    logger = HPOLogger(output_dir, fitness_metric=fitness_metric)

    try:
        logger.set_config({
            "n_trials": n_trials,
            "n_parallel": n_parallel,
            "n_runs": n_runs,
            "n_datasets": len(dataset_names),
            "dataset_names": dataset_names,
            "seed": seed,
            "pysr_kwargs": pysr_kwargs,
            "max_samples": max_samples,
            "max_concurrent_jobs": max_concurrent_jobs,
            "fitness_metric": fitness_metric,
            "mode": "operator",
            "max_op_hparams": max_op_hparams,
            "baseline_bundle": baseline_bundle.to_dict(),
            "operator_hparams": {
                op_type: _specs_to_dicts(specs) for op_type, specs in op_hparams.items()
            },
        })

        evaluator = PySRSlurmEvaluator(
            results_dir=output_dir,
            partition=slurm_partition,
            time_limit=slurm_time_limit,
            mem_per_cpu=slurm_mem_per_cpu,
            dataset_max_samples=max_samples,
            data_seed=seed,
            job_timeout=job_timeout,
            max_concurrent_jobs=max_concurrent_jobs,
            use_cache=use_cache,
        )

        # Phase 1: baseline (bundle as-is, no HPO overrides).
        print("\n" + "=" * 60)
        loaded_types = [t for t, op in baseline_bundle.operators.items() if op is not None]
        print(f"Phase 1: Evaluating baseline (bundle {', '.join(loaded_types)}, no HPO overrides)...")
        print("=" * 60)
        baseline_config = _build_trial_config(
            baseline_bundle, op_params_by_type={}, op_hparams=op_hparams,
            pysr_kwargs=pysr_kwargs,
        )
        if baseline_config is None:
            raise RuntimeError("Failed to build baseline config from bundle (code injection failed)")
        baseline_config.name = "baseline"
        baseline_results = evaluator.evaluate_configs(
            [baseline_config], dataset_names, seed=seed, n_runs=n_runs,
            target_noise_map=target_noise_map, fitness_metric=fitness_metric,
        )
        baseline_score, baseline_vector, baseline_details = baseline_results[0]
        if n_runs > 1 and baseline_details:
            score_key = "run_r2_scores" if fitness_metric == "r2" else "run_gt_scores"
            per_run_avgs = []
            for run_idx in range(n_runs):
                rs = [d[score_key][run_idx] for d in baseline_details
                      if len(d.get(score_key, [])) > run_idx]
                if rs:
                    per_run_avgs.append(float(np.mean(rs)))
            runs_str = ", ".join(f"{s:.2f}" for s in per_run_avgs)
            print(f"Baseline avg {metric_label}: {baseline_score:.4f} [{runs_str}]")
        else:
            print(f"Baseline avg {metric_label}: {baseline_score:.4f}")
        logger.log_baseline(
            baseline_score, baseline_vector, get_default_mutation_weights(), baseline_details,
        )

        if wandb_run is not None:
            import wandb
            wandb.log({"baseline_score": baseline_score, "best_score": baseline_score, "trial": -1})
            log_cpu_usage(wandb_run)

        # Phase 2: Optuna HPO loop (operator constants only).
        print("\n" + "=" * 60)
        print(f"Phase 2: Operator HPO ({n_trials} trials, {n_parallel} parallel)...")
        print("=" * 60)

        db_path = Path(output_dir) / "optuna_study.db"
        storage_url = f"sqlite:///{db_path}"
        study_name = "pysr_operator_hpo"

        prior_db_path = Path(continue_from) / "optuna_study.db" if continue_from else None
        if prior_db_path and prior_db_path.exists():
            import shutil
            shutil.copy2(prior_db_path, db_path)
            print(f"  Loaded Optuna study DB from {prior_db_path}")
            study = optuna.load_study(
                study_name=study_name, storage=storage_url, sampler=TPESampler(seed=seed),
            )
            print(f"  Resuming with {len(study.trials)} prior trials")
            if study.best_trial:
                print(f"  Best prior trial: {study.best_trial.number} (score: {study.best_trial.value:.4f})")
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=seed),
                study_name=study_name,
                storage=storage_url,
            )

        # Enqueue defaults (each operator hparam at its current value) as the first trial.
        if len(study.trials) == 0:
            initial = {}
            for op_type, specs in op_hparams.items():
                for hp in specs:
                    initial[f"op_{op_type}__{hp.name}"] = hp.current_value
            try:
                study.enqueue_trial(initial)
            except Exception as e:
                print(f"  Warning: could not enqueue initial trial: {e}")

        trials_completed = 0
        best_score = baseline_score
        best_combined_params: Dict[str, Any] = {}
        best_op_params: Optional[Dict[str, Dict[str, Any]]] = None

        while trials_completed < n_trials:
            batch_size = min(n_parallel, n_trials - trials_completed)
            print(f"\n--- Batch {trials_completed // n_parallel + 1}: "
                  f"trials {trials_completed + 1}-{trials_completed + batch_size} ---")

            asked: List = []
            op_params_list: List[Dict[str, Dict[str, Any]]] = []
            configs: List[PySRConfig] = []

            for _ in range(batch_size):
                trial = study.ask()
                try:
                    op_params = _sample_trial_params(trial, op_hparams)
                except Exception as e:
                    print(f"  Trial {trial.number}: param sampling failed: {e}")
                    study.tell(trial, float('-inf'))
                    continue

                config = _build_trial_config(
                    baseline_bundle, op_params, op_hparams, pysr_kwargs,
                )
                if config is None:
                    print(f"  Trial {trial.number}: code injection produced invalid Julia")
                    study.tell(trial, float('-inf'))
                    continue
                config.name = f"hpo_trial_{trial.number}"
                asked.append(trial)
                op_params_list.append(op_params)
                configs.append(config)

            if configs:
                try:
                    results = evaluator.evaluate_configs(
                        configs, dataset_names,
                        seed=seed, n_runs=n_runs,
                        target_noise_map=target_noise_map,
                        fitness_metric=fitness_metric,
                    )

                    for trial, op_params, (avg_score, score_vec, result_details) in zip(
                        asked, op_params_list, results,
                    ):
                        study.tell(trial, avg_score)

                        all_params: Dict[str, Any] = {}
                        for op_type, params in op_params.items():
                            for hp_name, val in params.items():
                                all_params[f"op_{op_type}__{hp_name}"] = val

                        improvement = avg_score - baseline_score
                        logger.log_trial(HPOTrialResult(
                            trial_number=trial.number,
                            params=all_params,
                            avg_score=avg_score,
                            score_vector=score_vec,
                            result_details=result_details,
                            improvement_vs_baseline=improvement,
                        ))

                        sign = "+" if improvement >= 0 else ""
                        if n_runs > 1 and result_details:
                            score_key = "run_r2_scores" if fitness_metric == "r2" else "run_gt_scores"
                            per_run_avgs = []
                            for run_idx in range(n_runs):
                                rs = [d[score_key][run_idx] for d in result_details
                                      if len(d.get(score_key, [])) > run_idx]
                                if rs:
                                    per_run_avgs.append(float(np.mean(rs)))
                            runs_str = ", ".join(f"{s:.2f}" for s in per_run_avgs)
                            print(f"  Trial {trial.number}: {metric_label}={avg_score:.4f} [{runs_str}] "
                                  f"({sign}{improvement:.4f} vs baseline)")
                        else:
                            print(f"  Trial {trial.number}: {metric_label}={avg_score:.4f} "
                                  f"({sign}{improvement:.4f} vs baseline)")

                        if avg_score > best_score:
                            best_score = avg_score
                            best_combined_params = all_params.copy()
                            best_op_params = {k: v.copy() for k, v in op_params.items()}
                            logger.log_best_trial(trial.number, avg_score)
                            print("    *** New best! ***")

                        if wandb_run is not None:
                            import wandb
                            wandb.log({
                                "trial": trial.number,
                                "trial_score": avg_score,
                                "best_score": best_score,
                            })

                except Exception as e:
                    print(f"  Batch evaluation failed: {e}")
                    for trial in asked:
                        study.tell(trial, float('-inf'))

            trials_completed += batch_size

        # Phase 3: bake best operator hparams back into a bundle and persist.
        print("\n" + "=" * 60)
        print("Phase 3: Final Results")
        print("=" * 60)

        try:
            best_trial = study.best_trial
            print(f"\nBest trial: {best_trial.number}")
            print(f"Best {metric_label}: {best_trial.value:.4f}")
        except Exception:
            pass
        print(f"Baseline {metric_label}: {baseline_score:.4f}")
        print(f"Improvement: {best_score - baseline_score:+.4f}")

        final_bundle = _copy.deepcopy(baseline_bundle)
        if best_op_params:
            for op_type, params in best_op_params.items():
                op = final_bundle.operators.get(op_type)
                if op is None or op_type not in op_hparams:
                    continue
                op.code = inject_hyperparameters(op.code, op_hparams[op_type], params)
                if op.hp_specs:
                    for spec_dict in op.hp_specs:
                        if spec_dict["name"] in params:
                            spec_dict["current_value"] = params[spec_dict["name"]]
        final_bundle.best_hparams = best_combined_params if best_combined_params else None
        final_bundle.score = best_score

        bundle_path = Path(output_dir) / "best_bundle.json"
        with open(bundle_path, "w") as f:
            json.dump(final_bundle.to_dict(), f, indent=2)
        print(f"\nBest bundle saved to: {bundle_path}")

        # Embed best_bundle into the logger's in-memory dict so evaluate_new_pysr.py
        # --evolve-results can consume it from run_data.json. Must be set before
        # logger.finalize() (which writes run_data.json from the in-memory dict).
        logger.run_data["best_bundle"] = final_bundle.to_dict()
        logger.finalize(best_combined_params, best_score, baseline_score)

        log_wandb_summary(
            wandb_run,
            evaluator=evaluator,
            extra_summary={
                "best_score": best_score,
                "baseline_score": baseline_score,
                "improvement": best_score - baseline_score,
            },
        )

        return best_combined_params, best_score, final_bundle
    finally:
        logger.close()


# =============================================================================
# Main CLI
# =============================================================================

def main():
    import argparse
    from datetime import datetime

    from utils import load_dataset_names_from_split, copy_slurm_log
    from wandb_utils import init_wandb, finish_wandb
    from bundle_loader import load_bundle
    from parallel_eval_pysr import get_default_pysr_kwargs
    from hpo_pysr import _build_target_noise_map, TARGET_NOISE_LEVELS

    parser = argparse.ArgumentParser(
        description="HPO over LLM-extracted operator-internal numeric constants. "
                    "Tunes only operator code; base PySR hparams are held fixed "
                    "(use hpo_pysr.py to tune those separately).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # HPO settings
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Total Optuna trials")
    parser.add_argument("--n-parallel", type=int, default=4,
                        help="Configs to evaluate per SLURM batch")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Seeds per config per dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed for reproducibility")
    parser.add_argument("--fitness-metric", type=str, default="gt", choices=["r2", "gt"],
                        help="Fitness metric to optimize")

    # Bundle / operator-HPO settings
    parser.add_argument("--evolved-bundle", type=str, required=True,
                        help="Path to an evolved bundle (run_data.json or output dir from "
                             "evolve_pysr.py, openevolve best_program.py, or a raw .jl file). "
                             "The operator code in the bundle is what gets its constants tuned.")
    parser.add_argument("--max-op-hparams", type=int, default=2,
                        help="Max operator-internal hparams per operator type "
                             "(LLM extraction is capped here).")
    parser.add_argument("--llm-model", type=str, default="openai/gpt-5-mini",
                        help="LLM used to identify operator-internal numeric constants.")

    # Dataset settings
    parser.add_argument("--split", type=str, default="splits/train.txt",
                        help="Dataset split file")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples per dataset")
    parser.add_argument("--target-noise", type=float, default=0.0,
                        help="Fixed Gaussian noise level for target")
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument("--random-target-noise", action="store_true",
                             help="Assign per-dataset target noise from {0.0, 0.001, 0.01, 0.1} using the seed")
    noise_group.add_argument("--no-random-target-noise", dest="random_target_noise", action="store_false")
    parser.set_defaults(random_target_noise=False)

    # PySR settings (frozen — passed through to every trial)
    parser.add_argument("--max-evals", type=int, default=1000000,
                        help="Max evaluations per PySR run")
    parser.add_argument("--timeout", type=int, default=300,
                        help="PySR timeout in seconds")

    # SLURM settings
    parser.add_argument("--partition", type=str, default="default_partition")
    parser.add_argument("--time-limit", type=str, default="00:30:00")
    parser.add_argument("--mem-per-cpu", type=str, default="8G")
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)

    # Output / continuation
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/operator_hpo_TIMESTAMP)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable evaluation caching")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Continue a previous operator HPO run (output dir of a prior run).")

    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/operator_hpo_{timestamp}"

    # Load datasets
    dataset_names = load_dataset_names_from_split(args.split)
    print(f"Loaded {len(dataset_names)} datasets from {args.split}")

    # Build target noise map
    target_noise_map = None
    if args.random_target_noise:
        target_noise_map = _build_target_noise_map(dataset_names, args.seed, TARGET_NOISE_LEVELS)
    elif args.target_noise > 0:
        target_noise_map = {name: args.target_noise for name in dataset_names}

    # Set up frozen PySR kwargs
    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    pysr_kwargs["timeout_in_seconds"] = args.timeout

    # Load bundle
    baseline_bundle = load_bundle(args.evolved_bundle)
    if baseline_bundle is None or not any(op is not None for op in baseline_bundle.operators.values()):
        parser.error(
            f"--evolved-bundle {args.evolved_bundle} did not yield a bundle with any operator code."
        )

    # Initialize wandb
    wandb_config = {
        "n_trials": args.n_trials,
        "n_parallel": args.n_parallel,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "fitness_metric": args.fitness_metric,
        "split": args.split,
        "max_samples": args.max_samples,
        "max_evals": args.max_evals,
        "timeout": args.timeout,
        "partition": args.partition,
        "target_noise": args.target_noise,
        "random_target_noise": args.random_target_noise,
        "evolved_bundle": args.evolved_bundle,
        "continue_from": args.continue_from,
        "no_cache": args.no_cache,
        "max_op_hparams": args.max_op_hparams,
        "llm_model": args.llm_model,
    }
    wandb_run = init_wandb(
        config=wandb_config,
        script_name="operator_hpo.py",
        output_dir=args.output_dir,
    )

    print(f"\nOperator HPO mode: tuning operator-internal constants in {args.evolved_bundle}")
    best_params, best_score, _final_bundle = run_operator_hpo(
        n_trials=args.n_trials,
        n_parallel=args.n_parallel,
        n_runs=args.n_runs,
        dataset_names=dataset_names,
        baseline_bundle=baseline_bundle,
        seed=args.seed,
        output_dir=args.output_dir,
        pysr_kwargs=pysr_kwargs,
        slurm_partition=args.partition,
        slurm_time_limit=args.time_limit,
        slurm_mem_per_cpu=args.mem_per_cpu,
        max_samples=args.max_samples,
        job_timeout=args.job_timeout,
        max_concurrent_jobs=args.max_concurrent_jobs,
        fitness_metric=args.fitness_metric,
        use_cache=not args.no_cache,
        continue_from=args.continue_from,
        wandb_run=wandb_run,
        target_noise_map=target_noise_map,
        model=args.llm_model,
        max_op_hparams=args.max_op_hparams,
    )

    run_data_path = Path(args.output_dir) / "run_data.json"
    print(
        f"\nTo evaluate the best bundle:\n"
        f"  python evaluate_new_pysr.py --evolve-results {run_data_path} "
        f"--splits splits/train.txt splits/val.txt --n-runs 10"
    )

    finish_wandb(wandb_run)

    print(f"\nResults saved to: {args.output_dir}")
    copy_slurm_log(args.output_dir)


if __name__ == "__main__":
    main()
