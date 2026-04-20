#!/usr/bin/env python3
"""
Hyperparameter optimization of PySR hyperparameters using Optuna.

This script serves as a baseline comparison for the LLM-based mutation evolution
in `evolve_pysr.py`. It can optimize arbitrary PySRRegressor constructor
hyperparameters exposed through the Python interface, including the built-in
mutation weights, without requiring changes to SymbolicRegression.jl.

Usage:
    python hpo_pysr.py \
        --n-trials 50 \
        --n-parallel 4 \
        --n-runs 3 \
        --split splits/train.txt \
        --partition <slurm_partition>
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    get_default_mutation_weights,
    get_default_pysr_kwargs,
)
from utils import load_dataset_names_from_split, TeeLogger, copy_slurm_log
from wandb_utils import init_wandb, log_wandb_summary, log_cpu_usage, finish_wandb


TARGET_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]


def _stable_target_noise(dataset_name: str, seed: int, noise_levels: List[float]) -> float:
    """Deterministically assign a target noise level based on dataset name + seed."""
    digest = hashlib.sha256(f"{seed}:{dataset_name}".encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "little") % len(noise_levels)
    return noise_levels[idx]


def _build_target_noise_map(
    dataset_names: List[str],
    seed: int,
    noise_levels: List[float],
) -> Dict[str, float]:
    """Map each dataset name to a deterministic target noise level."""
    return {name: _stable_target_noise(name, seed, noise_levels) for name in dataset_names}


# =============================================================================
# Constants: Generic HPO Search Specs
# =============================================================================

PYSR_HPARAM_IMPORTANCE_ORDER = [
    "binary_operators",
    "unary_operators",
    "constraints",
    "nested_constraints",
    "maxsize",
    "population_size",
    "populations",
    "ncycles_per_iteration",
    "parsimony",
    "weight_optimize",
    "batching",
    "batch_size",
    "loss_function",
    "elementwise_loss",
    "model_selection",
    "adaptive_parsimony_scaling",
    "warmup_maxsize_by",
    "maxdepth",
    "complexity_of_operators",
    "complexity_of_constants",
    "complexity_of_variables",
    "use_frequency",
    "use_frequency_in_tournament",
    "turbo",
    "precision",
    "crossover_probability",
    "alpha",
    "annealing",
    "tournament_selection_n",
    "tournament_selection_p",
    "fraction_replaced",
    "fraction_replaced_hof",
    "fraction_replaced_guesses",
    "topn",
    "migration",
    "hof_migration",
    "should_simplify",
    "weight_add_node",
    "weight_insert_node",
    "weight_delete_node",
    "weight_do_nothing",
    "weight_mutate_constant",
    "weight_mutate_operator",
    "weight_mutate_feature",
    "weight_swap_operands",
    "weight_rotate_tree",
    "weight_randomize",
    "weight_simplify",
    "should_optimize_constants",
    "optimize_probability",
    "optimizer_algorithm",
    "optimizer_nrestarts",
    "optimizer_iterations",
    "optimizer_f_calls_limit",
    "perturbation_factor",
    "probability_negate_constant",
    "loss_scale",
    "complexity_mapping",
    "dimensional_constraint_penalty",
    "dimensionless_constants_only",
    "skip_mutation_failures",
    "fast_cycle",
    "bumper",
    "autodiff_backend",
    "select_k_features",
    "denoise",
    "early_stop_condition",
    "niterations",
    "max_evals",
    "timeout_in_seconds",
    "parallelism",
    "procs",
    "cluster_manager",
    "heap_size_hint_in_bytes",
    "worker_timeout",
    "worker_imports",
    "random_state",
    "deterministic",
    "warm_start",
    "guesses",
    "verbosity",
    "update_verbosity",
    "print_precision",
    "progress",
    "logger_spec",
    "input_stream",
    "run_id",
    "output_directory",
    "temp_equation_file",
    "tempdir",
    "delete_tempfiles",
    "update",
    "output_jax_format",
    "output_torch_format",
    "extra_sympy_mappings",
    "extra_torch_mappings",
    "extra_jax_mappings",
    "operators",
    "expression_spec",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class HPOParamSpec:
    """Search spec for a single PySR hyperparameter."""
    name: str
    kind: str  # "float", "int", "categorical", "bool"
    default: Any
    low: Optional[float] = None
    high: Optional[float] = None
    scale: str = "linear"
    choices: Optional[List[Tuple[str, Any]]] = None


@dataclass
class HPOTrialResult:
    """Result from a single HPO trial."""
    trial_number: int
    params: Dict[str, Any]
    avg_score: float
    score_vector: List[float]
    result_details: List[Dict]
    improvement_vs_baseline: float


# =============================================================================
# Logging
# =============================================================================

class HPOLogger:
    """Tracks and saves HPO run data."""

    def __init__(self, output_dir: str, fitness_metric: str = "r2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fitness_metric = fitness_metric

        # Set up tee logging
        self.log_file = self.output_dir / "run.log"
        self.tee = TeeLogger(str(self.log_file))
        sys.stdout = self.tee

        # Initialize run data
        self.run_data = {
            "start_time": datetime.now().isoformat(),
            "fitness_metric": fitness_metric,
            "config": {},
            "baseline": {},
            "trials": [],
            "best_trial": None,
            "final_summary": None,
        }

    def set_config(self, config: Dict):
        """Save run configuration."""
        self.run_data["config"] = config
        self._save()

    def log_baseline(
        self,
        avg_score: float,
        score_vector: List[float],
        weights: Dict[str, float],
        result_details: List[Dict],
    ):
        """Log baseline results."""
        self.run_data["baseline"] = {
            "avg_score": avg_score,
            "score_vector": score_vector,
            "weights": weights,
            "result_details": result_details,
        }
        self._save()

    def log_trial(self, trial_result: HPOTrialResult):
        """Log a single trial result."""
        entry = {
            "trial_number": trial_result.trial_number,
            "params": trial_result.params,
            "avg_score": trial_result.avg_score,
            "score_vector": trial_result.score_vector,
            "result_details": trial_result.result_details,
            "improvement_vs_baseline": trial_result.improvement_vs_baseline,
        }
        self.run_data["trials"].append(entry)

        trials_dir = self.output_dir / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)
        trial_path = trials_dir / f"trial_{trial_result.trial_number:04d}.json"
        with open(trial_path, "w") as f:
            json.dump(entry, f, indent=2)

        self._save()

    def log_best_trial(self, trial_number: int, avg_score: float):
        """Log the best trial so far."""
        self.run_data["best_trial"] = {
            "trial_number": trial_number,
            "avg_score": avg_score,
        }
        self._save()

    def _save(self):
        """Save run data to JSON."""
        with open(self.output_dir / "run_data.json", "w") as f:
            json.dump(self.run_data, f, indent=2)

    def finalize(self, best_params: Dict[str, Any], best_score: float, baseline_score: float):
        """Save final results."""
        self.run_data["end_time"] = datetime.now().isoformat()

        improvement = best_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0

        self.run_data["final_summary"] = {
            "best_score": best_score,
            "baseline_score": baseline_score,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
        }
        self._save()

        # Save best params as standalone file for easy loading
        best_params_file = self.output_dir / "best_params.json"
        with open(best_params_file, "w") as f:
            json.dump({
                "params": best_params,
                "fitness_metric": self.fitness_metric,
                "avg_score": best_score,
                "baseline_score": baseline_score,
                "improvement": improvement,
            }, f, indent=2)
        print(f"\nBest params saved to: {best_params_file}")

    def close(self):
        """Clean up and restore stdout."""
        sys.stdout = self.tee.terminal
        self.tee.close()
        print(f"\nAll logs and results saved to: {self.output_dir}/")


# =============================================================================
# Core Functions
# =============================================================================

def _make_float_spec(
    name: str,
    default: float,
    low: float,
    high: float,
    scale: str = "linear",
) -> HPOParamSpec:
    return HPOParamSpec(
        name=name,
        kind="float",
        default=default,
        low=low,
        high=high,
        scale=scale,
    )


def _make_int_spec(
    name: str,
    default: int,
    low: int,
    high: int,
    scale: str = "linear",
) -> HPOParamSpec:
    return HPOParamSpec(
        name=name,
        kind="int",
        default=default,
        low=low,
        high=high,
        scale=scale,
    )


def _make_bool_spec(name: str, default: bool) -> HPOParamSpec:
    return HPOParamSpec(
        name=name,
        kind="bool",
        default=default,
        choices=[("false", False), ("true", True)],
    )


def _make_categorical_spec(name: str, default: Any, choices: List[Any]) -> HPOParamSpec:
    if default not in choices:
        choices = [default] + list(choices)
    labeled_choices = []
    for idx, value in enumerate(choices):
        if isinstance(value, (str, int, float, bool)) or value is None:
            label = repr(value)
        else:
            label = f"choice_{idx}"
        labeled_choices.append((label, value))
    return HPOParamSpec(
        name=name,
        kind="categorical",
        default=default,
        choices=labeled_choices,
    )


HPO_PARAM_SPECS = {
    # Search space / operators
    "binary_operators": _make_categorical_spec(
        "binary_operators",
        ["+", "-", "*", "/"],
        [
            ["+", "-", "*", "/"],
            ["+", "*", "/"],
            ["+", "-", "*", "/", "^"],
        ],
    ),
    "unary_operators": _make_categorical_spec(
        "unary_operators",
        ["sin", "cos", "exp", "log", "sqrt", "square"],
        [
            ["sin", "cos", "exp", "log", "sqrt", "square"],
            ["sin", "exp", "log", "sqrt", "square"],
            ["sin", "cos", "sqrt", "square"],
            ["sin", "cos", "exp", "log", "sqrt", "square", "abs"],
        ],
    ),
    "constraints": _make_categorical_spec(
        "constraints",
        {
            "sin": 9,
            "cos": 9,
            "exp": 9,
            "log": 9,
            "sqrt": 9,
            "/": (-1, 9),
        },
        [
            {
                "sin": 9,
                "cos": 9,
                "exp": 9,
                "log": 9,
                "sqrt": 9,
                "/": (-1, 9),
            },
            {
                "sin": 7,
                "cos": 7,
                "exp": 7,
                "log": 7,
                "sqrt": 7,
                "/": (-1, 7),
            },
            {
                "sin": 11,
                "cos": 11,
                "exp": 11,
                "log": 11,
                "sqrt": 11,
                "/": (-1, 11),
            },
        ],
    ),
    "nested_constraints": _make_categorical_spec(
        "nested_constraints",
        {
            "sin": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
            "cos": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
            "exp": {"exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
            "sqrt": {"sqrt": 0},
        },
        [
            {
                "sin": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
                "cos": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
                "sqrt": {"sqrt": 0},
            },
            {
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
                "sqrt": {"sqrt": 0},
            },
            {},
        ],
    ),
    "maxsize": _make_int_spec("maxsize", 30, 10, 80),
    "population_size": _make_int_spec("population_size", 27, 16, 256, scale="log"),
    "populations": _make_int_spec("populations", 31, 4, 128, scale="log"),
    "ncycles_per_iteration": _make_int_spec("ncycles_per_iteration", 380, 50, 10000, scale="log"),
    "parsimony": _make_float_spec("parsimony", 0.0, 1e-6, 1e-1, scale="log"),
    "weight_optimize": _make_float_spec("weight_optimize", 0.0, 0.0, 0.1),
    "batching": _make_categorical_spec("batching", False, [False, True, "auto"]),
    "batch_size": _make_int_spec("batch_size", 50, 16, 1024, scale="log"),
    "loss_function": _make_categorical_spec(
        "loss_function",
        None,
        [
            None,
            "loss(prediction, target) = (prediction - target)^2",
            "loss(prediction, target) = abs(prediction - target)",
        ],
    ),
    "elementwise_loss": _make_categorical_spec(
        "elementwise_loss",
        None,
        [
            None,
            "L2DistLoss()",
            "L1DistLoss()",
            "HuberLoss(1.0)",
        ],
    ),
    "model_selection": _make_categorical_spec(
        "model_selection", "best", ["best", "accuracy", "score"]
    ),
    "adaptive_parsimony_scaling": _make_float_spec(
        "adaptive_parsimony_scaling", 1040.0, 1.0, 5000.0, scale="log"
    ),
    "warmup_maxsize_by": _make_categorical_spec(
        "warmup_maxsize_by", None, [None, 0.25, 0.5, 0.75]
    ),
    "maxdepth": _make_categorical_spec("maxdepth", None, [None, 6, 8, 10, 12, 16]),
    "complexity_of_operators": _make_categorical_spec(
        "complexity_of_operators",
        None,
        [
            None,
            {"^": 3, "exp": 2, "log": 2},
            {"exp": 2, "log": 2, "/": 2},
        ],
    ),
    "complexity_of_constants": _make_categorical_spec(
        "complexity_of_constants", None, [None, 1, 2, 3]
    ),
    "complexity_of_variables": _make_categorical_spec(
        "complexity_of_variables", None, [None, 1, 2, 3]
    ),
    "use_frequency": _make_bool_spec("use_frequency", True),
    "use_frequency_in_tournament": _make_bool_spec("use_frequency_in_tournament", True),
    "turbo": _make_bool_spec("turbo", False),
    "precision": _make_categorical_spec("precision", 32, [16, 32, 64]),
    # Evolutionary search
    "crossover_probability": _make_float_spec("crossover_probability", 0.0259, 1e-4, 0.5, scale="log"),
    "alpha": _make_float_spec("alpha", 3.17, 0.1, 20.0, scale="log"),
    "annealing": _make_bool_spec("annealing", False),
    "tournament_selection_n": _make_int_spec("tournament_selection_n", 15, 2, 64),
    "tournament_selection_p": _make_float_spec("tournament_selection_p", 0.982, 0.5, 0.9999),
    "fraction_replaced": _make_float_spec("fraction_replaced", 0.00036, 1e-5, 0.1, scale="log"),
    "fraction_replaced_hof": _make_float_spec("fraction_replaced_hof", 0.0614, 1e-4, 0.5, scale="log"),
    "fraction_replaced_guesses": _make_float_spec("fraction_replaced_guesses", 0.001, 1e-5, 0.1, scale="log"),
    "topn": _make_int_spec("topn", 12, 1, 64),
    "migration": _make_bool_spec("migration", True),
    "hof_migration": _make_bool_spec("hof_migration", True),
    "should_simplify": _make_bool_spec("should_simplify", True),
    "weight_add_node": _make_float_spec("weight_add_node", 2.47, 0.001, 20.0, scale="log"),
    "weight_insert_node": _make_float_spec("weight_insert_node", 0.0112, 0.0001, 5.0, scale="log"),
    "weight_delete_node": _make_float_spec("weight_delete_node", 0.87, 0.001, 20.0, scale="log"),
    "weight_do_nothing": _make_float_spec("weight_do_nothing", 0.273, 0.0001, 5.0, scale="log"),
    "weight_mutate_constant": _make_float_spec("weight_mutate_constant", 0.0346, 0.0001, 5.0, scale="log"),
    "weight_mutate_operator": _make_float_spec("weight_mutate_operator", 0.293, 0.0001, 5.0, scale="log"),
    "weight_mutate_feature": _make_float_spec("weight_mutate_feature", 0.1, 0.0001, 5.0, scale="log"),
    "weight_swap_operands": _make_float_spec("weight_swap_operands", 0.198, 0.0001, 5.0, scale="log"),
    "weight_rotate_tree": _make_float_spec("weight_rotate_tree", 4.26, 0.0001, 20.0, scale="log"),
    "weight_randomize": _make_float_spec("weight_randomize", 0.000502, 1e-6, 0.1, scale="log"),
    "weight_simplify": _make_float_spec("weight_simplify", 0.00209, 1e-6, 0.1, scale="log"),
    # Constant optimization
    "should_optimize_constants": _make_bool_spec("should_optimize_constants", True),
    "optimize_probability": _make_float_spec("optimize_probability", 0.14, 1e-4, 1.0, scale="log"),
    "optimizer_algorithm": _make_categorical_spec(
        "optimizer_algorithm", "BFGS", ["BFGS", "NelderMead"]
    ),
    "optimizer_nrestarts": _make_int_spec("optimizer_nrestarts", 2, 0, 20),
    "optimizer_iterations": _make_int_spec("optimizer_iterations", 8, 1, 100, scale="log"),
    "optimizer_f_calls_limit": _make_categorical_spec(
        "optimizer_f_calls_limit", None, [None, 32, 64, 128, 256]
    ),
    "perturbation_factor": _make_float_spec("perturbation_factor", 0.129, 1e-4, 1.0, scale="log"),
    "probability_negate_constant": _make_float_spec(
        "probability_negate_constant", 0.00743, 1e-5, 0.5, scale="log"
    ),
    # Lower-priority algorithmic knobs
    "loss_scale": _make_categorical_spec("loss_scale", "log", ["log", "linear"]),
    "complexity_mapping": _make_categorical_spec(
        "complexity_mapping",
        None,
        [
            None,
            "complexity_mapping(tree) = compute_complexity(tree)",
        ],
    ),
    "dimensional_constraint_penalty": _make_categorical_spec(
        "dimensional_constraint_penalty", None, [None, 10.0, 100.0, 1000.0]
    ),
    "dimensionless_constants_only": _make_bool_spec("dimensionless_constants_only", False),
    "skip_mutation_failures": _make_bool_spec("skip_mutation_failures", True),
    "fast_cycle": _make_bool_spec("fast_cycle", False),
    "bumper": _make_bool_spec("bumper", False),
    "autodiff_backend": _make_categorical_spec(
        "autodiff_backend", None, [None, "Zygote", "Mooncake", "Enzyme"]
    ),
    "select_k_features": _make_categorical_spec(
        "select_k_features", None, [None, 4, 8, 16]
    ),
    "denoise": _make_bool_spec("denoise", False),
    "early_stop_condition": _make_categorical_spec(
        "early_stop_condition",
        None,
        [
            None,
            1e-3,
            "stop_if(loss, complexity) = loss < 1e-6 && complexity < 20",
        ],
    ),
}


def _filter_active_search_space(active_param_names: List[str]) -> Dict[str, HPOParamSpec]:
    active_specs = {}
    for name in active_param_names:
        if name not in HPO_PARAM_SPECS:
            raise KeyError(
                f"HPO parameter '{name}' does not have a search spec. "
                "Add it to HPO_PARAM_SPECS before enabling it."
            )
        active_specs[name] = HPO_PARAM_SPECS[name]
    return active_specs


def _serialize_param_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_serialize_param_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_param_value(v) for k, v in value.items()}
    return value


def _split_hpo_params(params: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    mutation_weights = {}
    pysr_kwargs = {}
    for name, value in params.items():
        if name.startswith("weight_"):
            mutation_weights[name] = value
        else:
            pysr_kwargs[name] = value
    return mutation_weights, pysr_kwargs


def _build_optuna_distributions(search_space: Dict[str, 'HPOParamSpec']) -> Dict:
    """Build Optuna distribution objects from search space specs."""
    import optuna

    distributions = {}
    for name, spec in search_space.items():
        if spec.kind == "float":
            distributions[name] = optuna.distributions.FloatDistribution(
                spec.low, spec.high, log=(spec.scale == "log"),
            )
        elif spec.kind == "int":
            distributions[name] = optuna.distributions.IntDistribution(
                int(spec.low), int(spec.high), log=(spec.scale == "log"),
            )
        elif spec.kind in ("bool", "categorical"):
            choices = [label for label, _ in spec.choices]
            distributions[name] = optuna.distributions.CategoricalDistribution(choices)
    return distributions


def _replay_trials_into_study(
    study,
    prior_trials: List[Dict],
    search_space: Dict[str, 'HPOParamSpec'],
) -> int:
    """Replay saved trial data into an Optuna study so TPE has full history.

    Returns number of trials successfully replayed.
    """
    import optuna

    distributions = _build_optuna_distributions(search_space)
    replayed = 0

    for trial_data in prior_trials:
        params = trial_data.get("params", {})
        score = trial_data.get("avg_score", trial_data.get("avg_r2"))
        if score is None:
            continue

        # Map params to Optuna's internal representation
        # For categorical/bool params, we need the label not the value
        optuna_params = {}
        for name, spec in search_space.items():
            if name not in params:
                continue
            if spec.kind in ("bool", "categorical"):
                # Reverse-map value to label
                value = params[name]
                for label, val in spec.choices:
                    if val == value:
                        optuna_params[name] = label
                        break
                else:
                    optuna_params[name] = value
            else:
                optuna_params[name] = params[name]

        # Only include distributions for params we have
        trial_distributions = {k: v for k, v in distributions.items() if k in optuna_params}

        try:
            frozen = optuna.trial.create_trial(
                params=optuna_params,
                distributions=trial_distributions,
                values=[score],
                state=optuna.trial.TrialState.COMPLETE,
            )
            study.add_trial(frozen)
            replayed += 1
        except Exception as e:
            print(f"  Warning: could not replay trial {trial_data.get('trial_number', '?')}: {e}")

    return replayed


def create_param_config_from_trial(
    trial,
    search_space: Dict[str, HPOParamSpec],
) -> Dict[str, Any]:
    """Create a generic PySR hyperparameter dict from an Optuna trial."""
    params = {}
    for name, spec in search_space.items():
        if spec.kind == "float":
            params[name] = trial.suggest_float(
                name,
                spec.low,
                spec.high,
                log=(spec.scale == "log"),
            )
        elif spec.kind == "int":
            if spec.scale == "log":
                params[name] = trial.suggest_int(name, int(spec.low), int(spec.high), log=True)
            else:
                params[name] = trial.suggest_int(name, int(spec.low), int(spec.high))
        elif spec.kind == "bool":
            label = trial.suggest_categorical(name, [choice_label for choice_label, _ in spec.choices])
            choice_map = dict(spec.choices)
            params[name] = choice_map[label]
        elif spec.kind == "categorical":
            label = trial.suggest_categorical(name, [choice_label for choice_label, _ in spec.choices])
            choice_map = dict(spec.choices)
            params[name] = choice_map[label]
        else:
            raise ValueError(f"Unsupported spec kind '{spec.kind}' for param '{name}'")
    return params


def _apply_baseline_to_config(
    config: PySRConfig,
    baseline_bundle,
) -> PySRConfig:
    """Merge baseline operator(s) into a PySRConfig."""
    if baseline_bundle is None:
        return config

    mut = baseline_bundle.operators.get("mutation")
    if mut is not None:
        weight = mut.weight if mut.weight is not None else 0.5
        config.mutation_weights["weight_custom_mutation_1"] = weight
        config.custom_mutation_code = {mut.name: mut.code}
        config.allow_custom_mutations = True

    surv = baseline_bundle.operators.get("survival")
    if surv is not None:
        config.custom_survival_code = surv.code

    sel = baseline_bundle.operators.get("selection")
    if sel is not None:
        config.custom_selection_code = sel.code

    return config


def evaluate_param_configs_batch(
    param_configs: List[Dict[str, Any]],
    trial_numbers: List[int],
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    base_pysr_kwargs: Dict[str, Any],
    seed: int,
    n_runs: int,
    fitness_metric: str,
    baseline_bundle=None,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> List[Tuple[float, List[float], List[Dict]]]:
    """
    Evaluate multiple hyperparameter configurations in a single SLURM batch.

    Args:
        param_configs: List of hyperparameter dicts
        evaluator: PySRSlurmEvaluator instance
        dataset_names: List of dataset names
        base_pysr_kwargs: Base PySR configuration
        seed: Random seed
        n_runs: Number of runs per config per dataset
        baseline_bundle: Optional OperatorBundle with custom operators to include

    Returns:
        List of (avg_r2, r2_vector, result_details) tuples
    """
    # Convert HPO param dicts to PySRConfig objects
    configs = []
    for i, params in enumerate(param_configs):
        mutation_weights = get_default_mutation_weights()
        hpo_mutation_weights, hpo_pysr_kwargs = _split_hpo_params(params)
        mutation_weights.update(hpo_mutation_weights)

        config = PySRConfig(
            mutation_weights=mutation_weights,
            pysr_kwargs={**base_pysr_kwargs, **hpo_pysr_kwargs},
            custom_mutation_code=None,
            allow_custom_mutations=False,
            name=f"hpo_trial_{trial_numbers[i]}",
        )
        config = _apply_baseline_to_config(config, baseline_bundle)
        configs.append(config)

    # Evaluate all configs in parallel via SLURM
    return evaluator.evaluate_configs(
        configs,
        dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
    )


def evaluate_baseline(
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    base_pysr_kwargs: Dict[str, Any],
    seed: int,
    n_runs: int,
    fitness_metric: str,
    baseline_bundle=None,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[float], List[Dict], Dict[str, float]]:
    """
    Evaluate PySR with default mutation weights (baseline).

    If baseline_bundle is provided, the custom operator(s) are included in the
    baseline config so that HPO optimizes hyperparams *on top of* the operator.

    Returns:
        (avg_r2, r2_vector, result_details, explicit_weights_passed)
    """
    # Use PySR's built-in defaults by not overriding weights
    mutation_weights = get_default_mutation_weights()

    config = PySRConfig(
        mutation_weights=mutation_weights,
        pysr_kwargs=base_pysr_kwargs,
        custom_mutation_code=None,
        allow_custom_mutations=False,
        name="baseline",
    )
    config = _apply_baseline_to_config(config, baseline_bundle)

    results = evaluator.evaluate_configs(
        [config],
        dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
    )
    avg_r2, r2_vector, result_details = results[0]
    return avg_r2, r2_vector, result_details, mutation_weights.copy()


# =============================================================================
# Main HPO Loop
# =============================================================================

def run_hpo(
    n_trials: int,
    n_parallel: int,
    n_runs: int,
    dataset_names: List[str],
    active_hpo_params: List[str],
    seed: int,
    output_dir: str,
    base_pysr_kwargs: Dict[str, Any],
    slurm_partition: str,
    slurm_time_limit: str,
    slurm_mem_per_cpu: str,
    max_samples: int,
    job_timeout: float,
    max_concurrent_jobs: Optional[int],
    fitness_metric: str = "gt",
    use_cache: bool = True,
    baseline_bundle=None,
    prior_trials: Optional[List[Dict]] = None,
    continue_from: Optional[str] = None,
    wandb_run: Any = None,
    target_noise_map: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Run hyperparameter optimization for generic PySR hyperparameters.

    Args:
        n_trials: Total number of Optuna trials
        n_parallel: Number of configs to evaluate per SLURM batch
        n_runs: Seeds per config per dataset
        dataset_names: List of dataset names to evaluate on
        seed: Master seed for reproducibility
        output_dir: Directory for outputs
        base_pysr_kwargs: Base PySR configuration
        slurm_*: SLURM job configuration
        max_samples: Max samples per dataset
        job_timeout: SLURM job timeout in seconds
        fitness_metric: Optimization objective ('r2' or 'gt')
        use_cache: Whether to use evaluation caching

    Returns:
        (best_params, best_score) tuple
    """
    # Lazy import of optuna
    import optuna
    from optuna.samplers import TPESampler

    np.random.seed(seed)

    search_space = _filter_active_search_space(active_hpo_params)
    if not search_space:
        raise ValueError("No active HPO hyperparameters selected.")
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
            "active_hpo_params": active_hpo_params,
            "base_pysr_kwargs": base_pysr_kwargs,
            "max_samples": max_samples,
            "max_concurrent_jobs": max_concurrent_jobs,
            "fitness_metric": fitness_metric,
            "search_space": {
                name: {
                    "kind": spec.kind,
                    "default": _serialize_param_value(spec.default),
                    "min": spec.low,
                    "max": spec.high,
                    "scale": spec.scale,
                    "choices": [
                        {"label": label, "value": _serialize_param_value(value)}
                        for label, value in (spec.choices or [])
                    ],
                }
                for name, spec in search_space.items()
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

        # Phase 1: Evaluate baseline
        print("=" * 60)
        if baseline_bundle:
            loaded_types = [t for t, op in baseline_bundle.operators.items() if op is not None]
            print(f"Phase 1: Evaluating baseline (with custom {', '.join(loaded_types)}, no HPO overrides)...")
        else:
            print("Phase 1: Evaluating baseline (current base PySR config, no HPO overrides)...")
        print("=" * 60)

        baseline_r2, baseline_vector, baseline_details, baseline_weights = evaluate_baseline(
            evaluator, dataset_names, base_pysr_kwargs, seed, n_runs, fitness_metric,
            baseline_bundle=baseline_bundle,
            target_noise_map=target_noise_map,
        )
        if n_runs > 1 and baseline_details:
            per_run_avgs = []
            for run_idx in range(n_runs):
                score_key = "run_r2_scores" if fitness_metric == "r2" else "run_gt_scores"
                run_scores = [d[score_key][run_idx] for d in baseline_details
                              if len(d.get(score_key, [])) > run_idx]
                if run_scores:
                    per_run_avgs.append(float(np.mean(run_scores)))
            runs_str = ", ".join(f"{s:.2f}" for s in per_run_avgs)
            print(f"Baseline avg {metric_label}: {baseline_r2:.4f} [{runs_str}]")
        else:
            print(f"Baseline avg {metric_label}: {baseline_r2:.4f}")
        print("Baseline uses the base PySR kwargs from this script with no additional HPO overrides.")
        logger.log_baseline(baseline_r2, baseline_vector, baseline_weights, baseline_details)

        if wandb_run is not None:
            import wandb
            wandb.log({"baseline_score": baseline_r2, "best_score": baseline_r2, "trial": -1})
            log_cpu_usage(wandb_run)

        # Phase 2: Optuna HPO Loop
        print("\n" + "=" * 60)
        print(f"Phase 2: Running HPO ({n_trials} trials, {n_parallel} parallel)...")
        print("=" * 60)

        db_path = Path(output_dir) / "optuna_study.db"
        storage_url = f"sqlite:///{db_path}"
        study_name = "pysr_hyperparameter_hpo"

        # Check if we're resuming from a prior run's DB
        prior_db_path = Path(continue_from) / "optuna_study.db" if continue_from else None
        if prior_db_path and prior_db_path.exists():
            import shutil
            shutil.copy2(prior_db_path, db_path)
            print(f"  Loaded Optuna study DB from {prior_db_path}")
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_url,
                sampler=TPESampler(seed=seed),
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
            # Fall back to JSON replay if no DB available
            if prior_trials:
                n_replayed = _replay_trials_into_study(study, prior_trials, search_space)
                print(f"  Replayed {n_replayed}/{len(prior_trials)} prior trials into Optuna study (from JSON)")
                if n_replayed > 0:
                    best_prior = study.best_trial
                    print(f"  Best prior trial: {best_prior.number} (score: {best_prior.value:.4f})")

        trials_completed = 0
        best_score = baseline_r2
        best_params = {}
        trial_param_map = {}

        while trials_completed < n_trials:
            batch_size = min(n_parallel, n_trials - trials_completed)

            print(f"\n--- Batch {trials_completed // n_parallel + 1}: "
                  f"Trials {trials_completed + 1}-{trials_completed + batch_size} ---")

            trials = [study.ask() for _ in range(batch_size)]
            param_configs = [create_param_config_from_trial(t, search_space) for t in trials]
            trial_numbers = [t.number for t in trials]
            trial_param_map.update({trial.number: params for trial, params in zip(trials, param_configs)})

            for trial in trials:
                print(f"  Trial {trial.number}: evaluating...")

            try:
                results = evaluate_param_configs_batch(
                    param_configs,
                    trial_numbers,
                    evaluator,
                    dataset_names,
                    base_pysr_kwargs,
                    seed,
                    n_runs,
                    fitness_metric,
                    baseline_bundle=baseline_bundle,
                    target_noise_map=target_noise_map,
                )

                for trial, params, (avg_score, score_vector, result_details) in zip(trials, param_configs, results):
                    study.tell(trial, avg_score)

                    improvement = avg_score - baseline_r2
                    trial_result = HPOTrialResult(
                        trial_number=trial.number,
                        params=params,
                        avg_score=avg_score,
                        score_vector=score_vector,
                        result_details=result_details,
                        improvement_vs_baseline=improvement,
                    )
                    logger.log_trial(trial_result)

                    sign = "+" if improvement >= 0 else ""
                    if n_runs > 1 and result_details:
                        per_run_avgs = []
                        for run_idx in range(n_runs):
                            score_key = "run_r2_scores" if fitness_metric == "r2" else "run_gt_scores"
                            run_scores = [d[score_key][run_idx] for d in result_details
                                          if len(d.get(score_key, [])) > run_idx]
                            if run_scores:
                                per_run_avgs.append(float(np.mean(run_scores)))
                        runs_str = ", ".join(f"{s:.2f}" for s in per_run_avgs)
                        print(f"  Trial {trial.number}: {metric_label}={avg_score:.4f} [{runs_str}] "
                              f"({sign}{improvement:.4f} vs baseline)")
                    else:
                        print(f"  Trial {trial.number}: {metric_label}={avg_score:.4f} ({sign}{improvement:.4f} vs baseline)")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = params.copy()
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
                for trial in trials:
                    study.tell(trial, float("-inf"))

            trials_completed += batch_size

        # Phase 3: Final Results
        print("\n" + "=" * 60)
        print("Phase 3: Final Results")
        print("=" * 60)

        best_trial = study.best_trial
        best_params = trial_param_map.get(best_trial.number, best_params)
        print(f"\nBest trial: {best_trial.number}")
        print(f"Best {metric_label}: {best_trial.value:.4f}")
        print(f"Baseline {metric_label}: {baseline_r2:.4f}")
        print(f"Improvement: {best_trial.value - baseline_r2:+.4f} "
              f"({(best_trial.value - baseline_r2) / baseline_r2 * 100:+.2f}%)")

        print(f"\nBest params:")
        for name, value in best_params.items():
            spec = search_space[name]
            default = spec.default
            if isinstance(value, (int, float)) and isinstance(default, (int, float)):
                pct_change = ((value - default) / default * 100) if default != 0 else float("inf")
                print(f"  {name}: {value} (default: {default}, {pct_change:+.1f}%)")
            else:
                print(f"  {name}: {value!r} (default: {default!r})")

        logger.finalize(best_params, best_trial.value, baseline_r2)

        log_wandb_summary(
            wandb_run,
            evaluator=evaluator,
            extra_summary={
                "best_score": best_trial.value,
                "baseline_score": baseline_r2,
                "improvement": best_trial.value - baseline_r2,
                "best_trial_number": best_trial.number,
            },
        )

        return best_params, best_trial.value
    finally:
        logger.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization of PySR hyperparameters using Optuna",
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
    parser.add_argument("--fitness_metric", type=str, default="gt", choices=["r2", "gt"],
                        help="Fitness metric to optimize")

    # Dataset settings
    parser.add_argument("--split", type=str, default="splits/train.txt",
                        help="Dataset split file")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples per dataset")
    parser.add_argument("--target-noise", type=float, default=0.0,
                        help="Fixed Gaussian noise level for target (SRBench standard levels: 0.0, 0.001, 0.01, 0.1)")
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument("--random-target-noise", action="store_true",
                             help="Assign per-dataset target noise from {0.0, 0.001, 0.01, 0.1} using the seed")
    noise_group.add_argument("--no-random-target-noise", dest="random_target_noise", action="store_false",
                             help="Disable per-dataset target noise and use --target-noise instead")
    parser.set_defaults(random_target_noise=False)

    # PySR settings
    parser.add_argument("--max-evals", type=int, default=1000000,
                        help="Max evaluations per PySR run")
    parser.add_argument("--timeout", type=int, default=300,
                        help="PySR timeout in seconds")

    # SLURM settings
    parser.add_argument("--partition", type=str, default="default_partition",
                        help="SLURM partition")
    parser.add_argument("--time-limit", type=str, default="00:30:00",
                        help="Time limit per job")
    parser.add_argument("--mem-per-cpu", type=str, default="8G",
                        help="Memory per CPU")
    parser.add_argument("--job-timeout", type=float, default=1800.0,
                        help="Max wait for SLURM completion")
    parser.add_argument("--max-concurrent-jobs", type=int, default=None,
                        help="Max concurrent SLURM array tasks (passed to job array % limit)")

    # Output settings
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/hpo_pysr_TIMESTAMP)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable evaluation caching")

    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to a baseline operator to include in all HPO trials. "
                             "Accepts: evolve_pysr output dir or run_data.json, "
                             "openevolve best_program.py, or a raw .jl file.")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Continue a previous HPO run. Pass the output directory of a prior run "
                             "(e.g., outputs/hpo_pysr_20260407_152534). Replays saved trials into "
                             "the Optuna study so TPE benefits from prior history.")

    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/hpo_pysr_{timestamp}"

    # Load datasets
    dataset_names = load_dataset_names_from_split(args.split)
    print(f"Loaded {len(dataset_names)} datasets from {args.split}")

    # Build target noise map
    target_noise_map = None
    if args.random_target_noise:
        target_noise_map = _build_target_noise_map(dataset_names, args.seed, TARGET_NOISE_LEVELS)
    elif args.target_noise > 0:
        target_noise_map = {name: args.target_noise for name in dataset_names}

    # Set up PySR kwargs
    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    pysr_kwargs["timeout_in_seconds"] = args.timeout

    # Uncomment entries in this list to include them in HPO.
    # The order follows the estimated importance ranking in PYSR_HPARAM_IMPORTANCE_ORDER.
    active_hpo_params = [
        # "maxsize",
        "population_size",
        "populations",
        "ncycles_per_iteration",
        "parsimony",
        "optimize_probability",
        "crossover_probability",
        "adaptive_parsimony_scaling",
        # "maxdepth",
        # "binary_operators",
        # "unary_operators",
        # "constraints",
        # "nested_constraints",
        # "batching",
        # "batch_size",
        # "loss_function",
        # "elementwise_loss",
        # "model_selection",
        # "warmup_maxsize_by",
        # "complexity_of_operators",
        # "complexity_of_constants",
        # "complexity_of_variables",
        # "use_frequency",
        # "use_frequency_in_tournament",
        # "turbo",
        # "precision",
        # "alpha",
        # "annealing",
        # "tournament_selection_n",
        # "tournament_selection_p",
        # "fraction_replaced",
        # "fraction_replaced_hof",
        # "fraction_replaced_guesses",
        # "topn",
        # "migration",
        # "hof_migration",
        # "should_simplify",
        # "weight_optimize",
        # "weight_add_node",
        # "weight_insert_node",
        # "weight_delete_node",
        # "weight_do_nothing",
        # "weight_mutate_constant",
        # "weight_mutate_operator",
        # "weight_mutate_feature",
        # "weight_swap_operands",
        # "weight_rotate_tree",
        # "weight_randomize",
        # "weight_simplify",
        # "should_optimize_constants",
        # "optimizer_algorithm",
        # "optimizer_nrestarts",
        # "optimizer_iterations",
        # "optimizer_f_calls_limit",
        # "perturbation_factor",
        # "probability_negate_constant",
        # "loss_scale",
        # "complexity_mapping",
        # "dimensional_constraint_penalty",
        # "dimensionless_constants_only",
        # "skip_mutation_failures",
        # "fast_cycle",
        # "bumper",
        # "autodiff_backend",
        # "select_k_features",
        # "denoise",
        # "early_stop_condition",
    ]

    # Load baseline operator if specified
    baseline_bundle = None
    if args.baseline:
        from baseline_loader import load_baseline_bundle
        baseline_bundle = load_baseline_bundle(args.baseline)

    # Load prior trials from JSON as fallback (for runs predating DB persistence)
    prior_trials = None
    if args.continue_from:
        prior_db_path = Path(args.continue_from) / "optuna_study.db"
        if not prior_db_path.exists():
            prior_run_data_path = Path(args.continue_from) / "run_data.json"
            if not prior_run_data_path.exists():
                parser.error(f"Cannot find optuna_study.db or run_data.json in {args.continue_from}")
            with open(prior_run_data_path) as f:
                prior_run_data = json.load(f)
            prior_trials = prior_run_data.get("trials", [])
            print(f"Continuing from {args.continue_from}: {len(prior_trials)} prior trials (JSON fallback)")
        else:
            print(f"Continuing from {args.continue_from}: found optuna_study.db")

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
        "baseline": args.baseline,
        "continue_from": args.continue_from,
        "no_cache": args.no_cache,
        "active_hpo_params": active_hpo_params,
    }
    wandb_run = init_wandb(
        config=wandb_config,
        script_name="hpo_pysr.py",
        output_dir=args.output_dir,
    )

    # Run HPO
    best_params, best_score = run_hpo(
        n_trials=args.n_trials,
        n_parallel=args.n_parallel,
        n_runs=args.n_runs,
        dataset_names=dataset_names,
        active_hpo_params=active_hpo_params,
        seed=args.seed,
        output_dir=args.output_dir,
        base_pysr_kwargs=pysr_kwargs,
        slurm_partition=args.partition,
        slurm_time_limit=args.time_limit,
        slurm_mem_per_cpu=args.mem_per_cpu,
        max_samples=args.max_samples,
        job_timeout=args.job_timeout,
        max_concurrent_jobs=args.max_concurrent_jobs,
        fitness_metric=args.fitness_metric,
        use_cache=not args.no_cache,
        baseline_bundle=baseline_bundle,
        prior_trials=prior_trials,
        continue_from=args.continue_from,
        wandb_run=wandb_run,
        target_noise_map=target_noise_map,
    )

    # Final evaluation on train + val (10 seeds)
    best_params_path = Path(args.output_dir) / "best_params.json"
    if best_params_path.exists():
        try:
            from evaluate_new_pysr import run_final_evaluation
            # Build noise map for final eval splits
            final_noise_map = None
            if args.random_target_noise:
                all_splits = ["splits/train.txt", "splits/val.txt"]
                all_datasets = []
                for sp in all_splits:
                    all_datasets.extend(load_dataset_names_from_split(sp))
                final_noise_map = _build_target_noise_map(
                    list(dict.fromkeys(all_datasets)), args.seed, TARGET_NOISE_LEVELS,
                )
            run_final_evaluation(
                output_dir=args.output_dir,
                method_source="hpo",
                method_path=str(best_params_path),
                partition=args.partition,
                n_runs=10,
                seed=args.seed,
                max_samples=args.max_samples,
                max_evals=args.max_evals,
                timeout=args.timeout,
                time_limit=args.time_limit,
                mem_per_cpu=args.mem_per_cpu,
                job_timeout=args.job_timeout,
                use_cache=not args.no_cache,
                wandb_run=wandb_run,
                target_noise_map=final_noise_map,
            )
        except Exception as e:
            print(f"\nFinal evaluation failed: {e}")

    finish_wandb(wandb_run)

    print(f"\nResults saved to: {args.output_dir}")
    copy_slurm_log(args.output_dir)


if __name__ == "__main__":
    main()
