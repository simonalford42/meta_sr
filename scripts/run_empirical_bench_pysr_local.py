#!/usr/bin/env python3
"""Local single-core PySR runs for the two EmpiricalBench tasks.

This is intentionally separate from empiricalbench_eval.py because that entry
point submits through PySRSlurmEvaluator. This script runs one local process,
records checkpoint fronts, and stops once the existing symbolic-match criterion
finds a GT match, or after --max-evals.
"""

import argparse
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JULIA_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parent.parent

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bundle_loader import load_bundle
from empiricalbench_eval import EMPIRICAL_DATASETS, ensure_datasets
from evaluation import check_pysr_frontier_symbolic_match, get_dataset_var_names
from operator_types import OperatorBundle
from parallel_eval_pysr import (
    PySRConfig,
    _get_pysr_num_evaluations,
    _import_pysr_regressor,
    _load_dynamic_loss,
    _load_dynamic_mutations,
    _load_dynamic_selection,
    _load_dynamic_survival,
    _remap_formula_variables,
    get_default_pysr_kwargs,
)
from utils import load_srbench_dataset, resolve_run_dir


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def make_milestones(max_evals: int, n: int) -> List[int]:
    if n <= 1:
        return [max_evals]
    log_part = np.geomspace(max(1000, max_evals // 10000), max_evals, num=n)
    milestones = sorted({int(round(x)) for x in log_part if x > 0})
    if milestones[-1] != max_evals:
        milestones.append(max_evals)
    return milestones


def clear_dynamic_operators() -> None:
    """Clear Julia dynamic operator slots so one local trial cannot contaminate the next."""
    from juliacall import Main as jl

    jl.seval("using SymbolicRegression")
    try:
        jl.seval("using SymbolicRegression.CustomMutationsModule")
        jl.seval("using SymbolicRegression.CoreModule: CUSTOM_MUTATION_NAMES")
        jl.seval("clear_dynamic_mutations!()")
        for i in range(1, 6):
            jl.seval(f"CUSTOM_MUTATION_NAMES[:custom_mutation_{i}] = :none")
        jl.seval("reload_custom_mutations!()")
    except Exception:
        pass
    for mod, clear_fn in [
        ("CustomSelectionModule", "clear_dynamic_selections!"),
        ("CustomSurvivalModule", "clear_dynamic_survivals!"),
        ("CustomLossModule", "clear_dynamic_losses!"),
    ]:
        try:
            jl.seval(f"using SymbolicRegression.{mod}")
            jl.seval(f"{clear_fn}()")
        except Exception:
            pass


def base_kwargs(max_evals: int, output_directory: Path) -> Dict[str, Any]:
    kwargs = get_default_pysr_kwargs()
    kwargs["max_evals"] = max_evals
    kwargs["niterations"] = max(10_000_000, max_evals)
    kwargs["parallelism"] = "serial"
    kwargs["procs"] = 0
    kwargs["deterministic"] = True
    kwargs["progress"] = False
    kwargs["verbosity"] = 0
    kwargs["temp_equation_file"] = False
    kwargs["delete_tempfiles"] = False
    kwargs["output_directory"] = str(output_directory)
    kwargs.pop("timeout_in_seconds", None)
    return kwargs


def apply_variant(kwargs: Dict[str, Any], variant: str) -> Dict[str, Any]:
    kwargs = deepcopy(kwargs)
    if variant == "default":
        return kwargs
    if variant == "inv":
        kwargs["unary_operators"] = kwargs["unary_operators"] + ["inv(x)=1/x"]
        kwargs["extra_sympy_mappings"] = {"inv": lambda x: 1 / x}
    elif variant == "cube":
        kwargs["unary_operators"] = kwargs["unary_operators"] + ["cube(x)=x*x*x"]
        kwargs["extra_sympy_mappings"] = {"cube": lambda x: x**3}
    elif variant == "inv_cube":
        kwargs["unary_operators"] = kwargs["unary_operators"] + [
            "inv(x)=1/x",
            "cube(x)=x*x*x",
        ]
        kwargs["extra_sympy_mappings"] = {"inv": lambda x: 1 / x, "cube": lambda x: x**3}
    elif variant == "larger":
        kwargs["maxsize"] = 60
        kwargs["maxdepth"] = 15
        kwargs["populations"] = 30
        kwargs["population_size"] = 50
    elif variant == "inv_cube_larger":
        kwargs = apply_variant(kwargs, "inv_cube")
        kwargs["maxsize"] = 60
        kwargs["maxdepth"] = 15
        kwargs["populations"] = 30
        kwargs["population_size"] = 50
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return kwargs


def build_config(mode: str, max_evals: int, output_directory: Path, variant: str) -> PySRConfig:
    kwargs = apply_variant(base_kwargs(max_evals, output_directory), variant)
    if mode == "baseline" or mode == "variant":
        bundle = OperatorBundle.create_default()
    elif mode == "evolved":
        bundle = load_bundle("runs/666285")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    config = bundle.to_pysr_config(kwargs)
    config.name = f"{mode}_{variant}"
    return config


def _best_and_match(model, X_val, y_val, ground_truth: str, variable_names: List[str]) -> Dict[str, Any]:
    best = model.get_best()
    best_index = best.name if best is not None else None
    best_equation = str(best["equation"]) if best is not None else None
    best_loss = float(best["loss"]) if best is not None else None
    y_pred = np.asarray(model.predict(X_val))
    y_pred = np.clip(y_pred, -1e10, 1e10)
    ss_res = float(np.sum((y_val - y_pred) ** 2))
    ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
    r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))
    match_result = check_pysr_frontier_symbolic_match(
        equations_df=model.equations_,
        best_df_index=best_index,
        ground_truth_str=ground_truth,
        var_names=variable_names,
        timeout_seconds_per_expression=3,
        predict_fn=lambda idx: model.predict(X_val, index=int(idx)),
        y=y_val,
        min_r2=0.5,
    )
    matched_equation = None
    matched_idx = match_result.get("matched_df_index")
    if matched_idx is not None and model.equations_ is not None:
        matched_equation = str(model.equations_.loc[matched_idx]["equation"])
    return {
        "best_equation": best_equation,
        "best_loss": best_loss,
        "validation_r2": r2,
        "match": bool(match_result.get("match", False)),
        "matched_equation": matched_equation,
        "match_result": match_result,
    }


def run_trial(
    config: PySRConfig,
    dataset: str,
    max_evals: int,
    milestones: List[int],
    seed: int,
    run_index: int,
    result_dir: Path,
    stop_on_solve: bool,
) -> Dict[str, Any]:
    run_seed = seed + run_index
    np.random.seed(42)
    random.seed(42)
    X, y, ground_truth = load_srbench_dataset(dataset, max_samples=1000)

    np.random.seed(run_seed)
    random.seed(run_seed)
    indices = np.random.permutation(len(y))
    n_train = int(0.8 * len(y))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    variable_names = [f"x{i}" for i in range(X.shape[1])]
    try:
        source_names = get_dataset_var_names(dataset)
        ground_truth_match = _remap_formula_variables(ground_truth, source_names, variable_names)
    except Exception:
        ground_truth_match = ground_truth

    PySRRegressor = _import_pysr_regressor()
    clear_dynamic_operators()
    if config.custom_mutation_code:
        _load_dynamic_mutations(config.custom_mutation_code)
    if config.custom_selection_code:
        _load_dynamic_selection(config.custom_selection_code)
    if config.custom_survival_code:
        _load_dynamic_survival(config.custom_survival_code)
    if config.custom_loss_code:
        _load_dynamic_loss(config.custom_loss_code)

    model_kwargs = deepcopy(config.pysr_kwargs)
    for key, value in config.mutation_weights.items():
        if config.allow_custom_mutations or "custom_mutation" not in key:
            model_kwargs[key] = value
    model_kwargs["random_state"] = run_seed
    model_kwargs["warm_start"] = True
    model_kwargs["run_id"] = f"{config.name}_{dataset}_seed{run_seed}".replace("/", "_")

    model = PySRRegressor(**model_kwargs)
    trial_start = time.time()
    checkpoints: List[Dict[str, Any]] = []
    solved = False
    solved_at: Optional[Dict[str, Any]] = None
    error = None

    try:
        for milestone in milestones:
            chunk_start = time.time()
            model.max_evals = int(milestone)
            model.fit(X_train, y_train, variable_names=variable_names)
            elapsed = time.time() - trial_start
            actual_evals = _get_pysr_num_evaluations(model)
            status = _best_and_match(model, X_val, y_val, ground_truth_match, variable_names)
            eq_rows = []
            if model.equations_ is not None:
                eq_rows = model.equations_.to_dict(orient="records")
            checkpoint = {
                "milestone_max_evals": int(milestone),
                "actual_num_evaluations": actual_evals,
                "elapsed_seconds": elapsed,
                "chunk_seconds": time.time() - chunk_start,
                "n_equations": len(eq_rows),
                "equations": eq_rows,
                **status,
            }
            checkpoints.append(checkpoint)
            with open(result_dir / "checkpoints.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(checkpoint, default=_json_default) + "\n")
            if status["match"]:
                solved = True
                solved_at = checkpoint
                if stop_on_solve:
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    total_elapsed = time.time() - trial_start
    final = checkpoints[-1] if checkpoints else {}
    result = {
        "config_name": config.name,
        "dataset": dataset,
        "seed": seed,
        "run_index": run_index,
        "run_seed": run_seed,
        "max_evals": max_evals,
        "milestones": milestones,
        "solved": solved,
        "solved_at": solved_at,
        "error": error,
        "total_elapsed_seconds": total_elapsed,
        "final": final,
        "ground_truth": ground_truth,
        "ground_truth_match": ground_truth_match,
    }
    with open(result_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=_json_default)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "evolved", "variant"], required=True)
    parser.add_argument("--variant", default="default")
    parser.add_argument("--datasets", default=",".join(EMPIRICAL_DATASETS))
    parser.add_argument("--max-evals", type=int, default=10_000_000)
    parser.add_argument("--n-milestones", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--no-stop-on-solve", action="store_true")
    args = parser.parse_args()

    ensure_datasets()
    output_dir = Path(resolve_run_dir(args.results_dir, label="empirical_pysr_local"))
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    milestones = make_milestones(args.max_evals, args.n_milestones)

    all_results = []
    for dataset in datasets:
        trial_dir = output_dir / f"{args.mode}_{args.variant}_{dataset}_seed{args.seed + args.run_index}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config = build_config(
            args.mode,
            args.max_evals,
            trial_dir / "pysr_outputs",
            args.variant,
        )
        meta = {
            "args": vars(args),
            "config": config.to_json_dict(),
            "dataset": dataset,
            "trial_dir": str(trial_dir),
        }
        with open(trial_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=_json_default)
        result = run_trial(
            config=config,
            dataset=dataset,
            max_evals=args.max_evals,
            milestones=milestones,
            seed=args.seed,
            run_index=args.run_index,
            result_dir=trial_dir,
            stop_on_solve=not args.no_stop_on_solve,
        )
        all_results.append(result)
        print(json.dumps({
            "dataset": dataset,
            "solved": result["solved"],
            "solved_evals": None if not result["solved_at"] else result["solved_at"]["actual_num_evaluations"],
            "solved_time": None if not result["solved_at"] else result["solved_at"]["elapsed_seconds"],
            "error": result["error"],
        }, default=_json_default), flush=True)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
