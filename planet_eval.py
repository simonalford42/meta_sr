#!/usr/bin/env python3
"""Evaluate baseline/evolved PySR on the planet_eqs 11003 f2 task.

Submit this through planet_eval.sh, which requests the SLURM allocation
matching planet_eqs/sr.sh (-N 1, -n 32, 200G, default_partition). Driver mode
runs inside that allocation: it first uses the new_bnn environment to export
the 11003 NN-derived SR arrays, then uses the meta_sr environment to run PySR
and log resonant-test metrics to the meta-sr wandb project.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


META_ROOT = Path(__file__).resolve().parent
DEFAULT_PLANET_ROOT = META_ROOT.parent / "planet_eqs"
CONDA_SH = Path("/home/sca63/mambaforge/etc/profile.d/conda.sh")
PHASE_ENV = "PLANET_EVAL_PHASE"
CONFIG_ENV = "PLANET_EVAL_CONFIG"

DEFAULT_NN_VERSION = 11003
DEFAULT_TARGET = "f2"
DEFAULT_LOSS_FN = "mse"
DEFAULT_TIME_IN_HOURS = 8
DEFAULT_NITERATIONS = 500000
DEFAULT_MAX_SIZE = 30
DEFAULT_N = 10000
DEFAULT_BATCH_SIZE = 1000


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _repo_imports() -> None:
    if str(META_ROOT) not in sys.path:
        sys.path.insert(0, str(META_ROOT))


def _planet_imports(planet_root: Path) -> None:
    if str(planet_root) not in sys.path:
        sys.path.insert(0, str(planet_root))
    os.chdir(planet_root)


def _method_label(config: Dict[str, Any]) -> str:
    return "evolve" if config.get("evolve_results") else "baseline"


def _source_label(path: Optional[str]) -> str:
    if not path:
        return "baseline"
    p = Path(path).expanduser()
    if p.name == "run_data.json":
        p = p.parent
    return p.name or "evolve"


def _safe_float(x: Any) -> Optional[float]:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def slurm_num_cpus(default: int = 10) -> int:
    """Mirror planet_eqs/sr.py's PySR worker count from the SLURM allocation."""
    try:
        return int(os.environ.get("SLURM_CPUS_ON_NODE")) * int(
            os.environ.get("SLURM_JOB_NUM_NODES")
        )
    except (TypeError, ValueError):
        return default


def build_planet_sr_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Match the default planet_eqs/sr.py PySR configuration."""
    num_cpus = int(config["num_cpus"])
    equation_file = str(Path(config["output_dir"]) / "planet_pysr_equations.csv")
    return {
        "procs": num_cpus,
        "populations": 3 * num_cpus,
        "batching": True,
        "batch_size": int(config["batch_size"]),
        "equation_file": equation_file,
        "niterations": int(config["niterations"]),
        "binary_operators": ["+", "*", "/", "-", "^"],
        "maxsize": int(config["max_size"]),
        "timeout_in_seconds": int(60 * 60 * float(config["time_in_hours"])),
        "constraints": {"^": (-1, 1)},
        "ncyclesperiteration": 1000,
        "random_state": int(config["seed"]),
    }


def export_data(config_path: Path) -> None:
    """Run under new_bnn: save train and resonant-test arrays from planet_eqs."""
    config = _read_json(config_path)
    planet_root = Path(config["planet_root"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    cache_dir = output_dir / "planet_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    _planet_imports(planet_root)

    import torch
    import sr as planet_sr
    import spock_reg_model

    sr_args = argparse.Namespace(
        no_log=True,
        nn_version=int(config["nn_version"]),
        version=0,
        time_in_hours=float(config["time_in_hours"]),
        niterations=int(config["niterations"]),
        max_size=int(config["max_size"]),
        seed=int(config["seed"]),
        target=config["target"],
        residual=False,
        n=int(config["n"]),
        batch_size=int(config["batch_size"]),
        sr_residual=False,
        loss_fn=config["loss_fn"],
    )
    sr_config = planet_sr.get_config(sr_args)
    X_train, y_train, variable_names, nn_std_arr = planet_sr.load_inputs_and_targets(sr_config)

    model = spock_reg_model.load(version=int(config["nn_version"]))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.make_dataloaders(train=True, plot=True)
    dataloader = model._val_dataloader

    summaries: List[np.ndarray] = []
    truths: List[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            out = model(
                x_batch.to(device),
                noisy_val=False,
                deterministic=True,
                return_intermediates=True,
            )
            summaries.append(out["summary_stats"].detach().cpu().numpy())
            truths.append(y_batch.detach().cpu().numpy())

    X_test = np.concatenate(summaries, axis=0).astype(np.float32)
    y_test = np.concatenate(truths, axis=0).astype(np.float32)

    data_path = cache_dir / "planet_sr_data.npz"
    np.savez_compressed(
        data_path,
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.float32),
        X_test=X_test,
        y_test=y_test,
        nn_std_arr=(
            np.asarray(nn_std_arr, dtype=np.float32)
            if nn_std_arr is not None
            else np.asarray([], dtype=np.float32)
        ),
    )
    _write_json(
        cache_dir / "planet_sr_data.json",
        {
            "data_path": str(data_path),
            "nn_version": config["nn_version"],
            "target": config["target"],
            "loss_fn": config["loss_fn"],
            "n_train": int(np.asarray(X_train).shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_test.shape[1]),
            "variable_names": list(variable_names),
            "train_shape": list(np.asarray(X_train).shape),
            "target_shape": list(np.asarray(y_train).shape),
            "test_shape": list(X_test.shape),
            "test_truth_shape": list(y_test.shape),
        },
    )
    print(f"Saved planet arrays to {data_path}", flush=True)


def safe_log_erf_np(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf

    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    under = x < -1
    xu = x[under]
    out[under] = (
        0.485660082730562 * xu
        + 0.643278438654541 * np.exp(xu)
        + 0.00200084619923262 * xu**3
        - 0.643250926022749
        - 0.955350621183745 * xu**2
    )
    xo = x[~under]
    out[~under] = np.log1p(erf(xo))
    return out


def planet_ll(truths_full: np.ndarray, mu_pred: np.ndarray, fixed_std: float = 1.0) -> float:
    """Copy planet_eqs/evaluation.py lossfnc sign convention: return LL."""
    y = np.asarray(truths_full, dtype=np.float64)
    mu = np.asarray(mu_pred, dtype=np.float64).reshape(-1, 1)
    std = np.ones_like(mu) * float(fixed_std)
    var = std**2
    t_greater_9 = y >= 9

    regression_loss = -(y - mu) ** 2 / (2 * var)
    regression_loss += -np.log(std)
    regression_loss += -safe_log_erf_np((mu - 4) / np.sqrt(2 * var))

    classifier_loss = safe_log_erf_np((mu - 9) / np.sqrt(2 * var))

    safe_regression = np.where(np.isfinite(regression_loss), regression_loss, -100.0)
    safe_classifier = np.where(np.isfinite(classifier_loss), classifier_loss, -100.0)

    total_regression = -(safe_regression * (~t_greater_9)).sum() / len(y)
    total_classifier = -(safe_classifier * t_greater_9).sum() / len(y)
    return float(-(total_regression + total_classifier))


def calculate_planet_metrics(truths_full: np.ndarray, mu_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    truths_full = np.asarray(truths_full, dtype=np.float64)
    roc_preds = np.asarray(mu_pred, dtype=np.float64).reshape(-1)
    preds = np.clip(roc_preds, 4, 9)
    truths = np.average(truths_full, axis=1) if truths_full.ndim == 2 else truths_full

    pred_stable = preds >= 9
    true_stable = truths >= 9
    unstable = ~true_stable

    rmse = float(np.sqrt(np.mean(np.square(truths[unstable] - preds[unstable]))))
    full_rmse = float(np.sqrt(np.mean(np.square(truths - preds))))
    acc = float(np.mean((preds >= 9) == (truths >= 9)))
    bias = float(np.mean(preds[truths < 9] - truths[truths < 9]))

    fpr = float(np.mean(~pred_stable[true_stable])) if np.any(true_stable) else float("nan")
    fnr = float(np.mean(pred_stable[~true_stable])) if np.any(~true_stable) else float("nan")
    roc_true_unstable = truths < 9
    unstable_score = -roc_preds
    roc_auc = (
        float(roc_auc_score(roc_true_unstable, unstable_score))
        if len(np.unique(roc_true_unstable)) == 2
        else float("nan")
    )

    return {
        "rmse": rmse,
        "full_rmse": full_rmse,
        "acc": acc,
        "ll": planet_ll(truths_full, roc_preds, fixed_std=1.0),
        "roc_auc": roc_auc,
        "fpr": fpr,
        "fnr": fnr,
        "bias": bias,
    }


def build_method(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return (pysr_kwargs, extra_code, method_meta)."""
    _repo_imports()
    base_kwargs = build_planet_sr_kwargs(config)
    method_meta: Dict[str, Any] = {"mode": _method_label(config)}

    if not config.get("evolve_results"):
        return base_kwargs, {}, method_meta

    from evaluate_new_pysr import build_evolve_kwargs, load_evolve_results
    from parallel_eval_pysr import get_default_mutation_weights

    source = str(Path(config["evolve_results"]).expanduser())
    method = load_evolve_results(source, None)
    weights, pysr_kwargs, extra_code, items = build_evolve_kwargs(
        method,
        get_default_mutation_weights(),
        base_kwargs,
    )
    allow_custom = bool(extra_code.get("allow_custom_mutations", False))
    mutation_kwargs = {}
    for key, value in weights.items():
        if not key.startswith("weight_"):
            key = f"weight_{key}"
        if "custom_mutation" in key and not allow_custom:
            continue
        mutation_kwargs[key] = value
    pysr_kwargs = {**mutation_kwargs, **pysr_kwargs}

    method_meta = {
        "mode": "evolve",
        "source": source,
        "operators": [
            {
                "operator_type": item.operator_type,
                "name": item.name,
                "generation": item.generation,
                "train_score": item.train_score,
                "weight": item.weight,
            }
            for item in items
        ],
        "best_hparams": items[0].best_hparams,
    }
    return pysr_kwargs, extra_code, method_meta


def _load_dynamic_code(extra_code: Dict[str, Any]) -> None:
    from parallel_eval_pysr import (
        _load_dynamic_loss,
        _load_dynamic_mutations,
        _load_dynamic_selection,
        _load_dynamic_survival,
    )

    if extra_code.get("custom_mutation_code"):
        _load_dynamic_mutations(extra_code["custom_mutation_code"])
    if extra_code.get("custom_selection_code"):
        _load_dynamic_selection(extra_code["custom_selection_code"])
    if extra_code.get("custom_survival_code"):
        _load_dynamic_survival(extra_code["custom_survival_code"])
    if extra_code.get("custom_loss_code"):
        _load_dynamic_loss(extra_code["custom_loss_code"])
    else:
        try:
            from juliacall import Main as jl

            jl.seval("using SymbolicRegression.CustomLossModule")
            jl.seval("clear_dynamic_losses!()")
        except Exception:
            pass


def fit_eval(config_path: Path) -> None:
    """Run under meta_sr: fit PySR, evaluate resonant test, log wandb table."""
    _repo_imports()
    config = _read_json(config_path)
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    data_meta = _read_json(output_dir / "planet_data" / "planet_sr_data.json")
    data = np.load(data_meta["data_path"], allow_pickle=True)
    X_train = np.asarray(data["X_train"], dtype=np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.float32).reshape(-1)
    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.float32)
    variable_names = list(data_meta["variable_names"])

    from parallel_eval_pysr import _import_pysr_regressor

    PySRRegressor = _import_pysr_regressor()
    pysr_kwargs, extra_code, method_meta = build_method(config)
    _load_dynamic_code(extra_code)

    print("PySR kwargs:", json.dumps(pysr_kwargs, indent=2, default=_json_default), flush=True)
    model = PySRRegressor(**pysr_kwargs)
    model.fit(X_train, y_train, variable_names=variable_names)
    print("Done running planet PySR", flush=True)

    best = model.get_best()
    best_equation = str(best["equation"]) if best is not None else None
    best_loss = _safe_float(best["loss"]) if best is not None and "loss" in best else None
    best_complexity = _safe_float(best["complexity"]) if best is not None and "complexity" in best else None

    preds = np.asarray(model.predict(X_test))
    if preds.ndim == 2:
        preds = preds[:, 0]
    preds = preds.reshape(-1)
    preds = np.where(np.isfinite(preds), preds, 4.0)

    metrics = calculate_planet_metrics(y_test, preds)

    equations_path = output_dir / "planet_equations.csv"
    try:
        model.equations_.to_csv(equations_path, index=False)
    except Exception as exc:
        print(f"WARNING: could not save equations CSV: {exc}", flush=True)

    result = {
        "mode": _method_label(config),
        "method_meta": method_meta,
        "nn_version": config["nn_version"],
        "target": config["target"],
        "loss_fn": config["loss_fn"],
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "variable_names": variable_names,
        "equation": best_equation,
        "best_loss": best_loss,
        "best_complexity": best_complexity,
        "metrics": metrics,
        "equations_path": str(equations_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    result_path = output_dir / "planet_eval_result.json"
    _write_json(result_path, result)

    print("\nPlanet resonant-test metrics")
    for key in ["rmse", "acc", "ll", "roc_auc", "fpr", "fnr", "bias"]:
        print(f"  {key}: {metrics[key]}")
    print(f"  equation: {best_equation}", flush=True)

    if not config.get("no_wandb"):
        log_planet_wandb(config, result)


def log_planet_wandb(config: Dict[str, Any], result: Dict[str, Any]) -> None:
    from wandb_utils import finish_wandb, init_wandb, log_wandb_summary

    mode = _method_label(config)
    source = config.get("evolve_results")
    run = init_wandb(
        config={
            "mode": mode,
            "evolve_results": source,
            "nn_version": config["nn_version"],
            "target": config["target"],
            "loss_fn": config["loss_fn"],
            "max_size": config["max_size"],
            "time_in_hours": config["time_in_hours"],
            "n": config["n"],
            "seed": config["seed"],
            "method_meta": result.get("method_meta", {}),
        },
        script_name="planet_eval.py",
        output_dir=config["output_dir"],
        extra_tags=["planet_eval", mode, _source_label(source)],
    )
    if run is None:
        return

    import wandb

    columns = [
        "mode",
        "source",
        "nn_version",
        "target",
        "loss_fn",
        "n_train",
        "n_test",
        "equation",
        "best_loss",
        "best_complexity",
        "rmse",
        "acc",
        "ll",
        "roc_auc",
        "fpr",
        "fnr",
        "bias",
        "slurm_job_id",
    ]
    metrics = result["metrics"]
    table = wandb.Table(columns=columns)
    table.add_data(
        result["mode"],
        source or "baseline",
        result["nn_version"],
        result["target"],
        result["loss_fn"],
        result["n_train"],
        result["n_test"],
        result["equation"],
        result["best_loss"],
        result["best_complexity"],
        metrics["rmse"],
        metrics["acc"],
        metrics["ll"],
        metrics["roc_auc"],
        metrics["fpr"],
        metrics["fnr"],
        metrics["bias"],
        result.get("slurm_job_id"),
    )
    wandb.log({"planet_eval/results_table": table})
    wandb.log({f"planet_eval/{k}": v for k, v in metrics.items()})
    log_wandb_summary(
        run,
        extra_summary={
            f"planet_eval_{k}": v
            for k, v in {
                **metrics,
                "best_loss": result.get("best_loss"),
                "best_complexity": result.get("best_complexity"),
            }.items()
        },
    )
    finish_wandb(run)


def run_worker_phase(worker: str, conda_env: str, cwd: Path, config_path: Path) -> None:
    script_path = Path(__file__).resolve()
    env = os.environ.copy()
    env[PHASE_ENV] = worker
    env[CONFIG_ENV] = str(config_path)
    shell_cmd = " && ".join(
        [
            "set -euo pipefail",
            f"source {shlex.quote(str(CONDA_SH))}",
            f"conda activate {shlex.quote(conda_env)}",
            f"cd {shlex.quote(str(cwd))}",
            f"python -u {shlex.quote(str(script_path))}",
        ]
    )
    print(f"\n[planet_eval] running {worker} in conda env {conda_env}", flush=True)
    subprocess.run(["bash", "-lc", shell_cmd], cwd=str(META_ROOT), env=env, check=True)


def summarize_result(output_dir: Path) -> None:
    result_path = output_dir / "planet_eval_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"planet_eval did not write {result_path}")

    result = _read_json(result_path)
    print(f"Result: {result_path}", flush=True)
    for key in ["rmse", "acc", "ll", "roc_auc", "fpr", "fnr", "bias"]:
        print(f"  {key}: {result['metrics'].get(key)}", flush=True)
    print(f"  equation: {result.get('equation')}", flush=True)


def get_config(args: argparse.Namespace) -> Dict[str, Any]:
    _repo_imports()
    from utils import resolve_run_dir

    if args.baseline and args.evolve_results:
        raise SystemExit("--baseline and --evolve-results are mutually exclusive")

    output_dir = Path(resolve_run_dir(args.results_dir, label="planet_eval")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    evolve_results = str(Path(args.evolve_results).expanduser()) if args.evolve_results else None
    return {
        "output_dir": str(output_dir),
        "meta_root": str(META_ROOT),
        "planet_root": str(DEFAULT_PLANET_ROOT.expanduser().resolve()),
        "baseline": bool(args.baseline or not evolve_results),
        "evolve_results": evolve_results,
        "nn_version": DEFAULT_NN_VERSION,
        "target": DEFAULT_TARGET,
        "loss_fn": DEFAULT_LOSS_FN,
        "max_size": int(args.max_size),
        "time_in_hours": float(args.time_in_hours),
        "niterations": DEFAULT_NITERATIONS,
        "n": DEFAULT_N,
        "batch_size": DEFAULT_BATCH_SIZE,
        "seed": int(args.seed),
        "num_cpus": slurm_num_cpus(),
        "no_wandb": bool(args.no_log),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_name": os.environ.get("SLURM_JOB_NAME"),
    }


def driver(args: argparse.Namespace) -> None:
    config = get_config(args)
    output_dir = Path(config["output_dir"])
    config_path = output_dir / "planet_eval_config.json"
    _write_json(config_path, config)

    print(f"Run directory: {output_dir}", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"PySR workers: {config['num_cpus']}", flush=True)
    if args.dry_run:
        print("Dry run requested; not running planet_eval worker phases.", flush=True)
        return

    print(f"planet_eval job {os.environ.get('SLURM_JOB_ID', 'local')} on {os.uname().nodename}", flush=True)
    run_worker_phase("export-data", "new_bnn", Path(config["planet_root"]), config_path)
    run_worker_phase("fit-eval", "meta_sr", META_ROOT, config_path)
    summarize_result(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate planet_eqs PySR baseline or evolved operators inside planet_eval.sh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--evolve-results", type=str, default=None, help="Path/run id with run_data.json.")
    parser.add_argument("--baseline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-size", "--max_size", dest="max_size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument(
        "--time-in-hours",
        "--time_in_hours",
        dest="time_in_hours",
        type=float,
        default=DEFAULT_TIME_IN_HOURS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-log", "--no_log", dest="no_log", action="store_true", help="disable wandb logging")
    parser.add_argument("--results-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-wandb", dest="no_log", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    phase = os.environ.pop(PHASE_ENV, None)
    if phase:
        config_raw = os.environ.pop(CONFIG_ENV, None)
        if not config_raw:
            raise SystemExit(f"{PHASE_ENV} requires {CONFIG_ENV}")
        config_path = Path(config_raw).expanduser().resolve()
        if phase == "export-data":
            export_data(config_path)
        elif phase == "fit-eval":
            fit_eval(config_path)
        else:
            raise SystemExit(f"Unknown {PHASE_ENV}: {phase}")
        return

    args = parse_args()
    driver(args)


if __name__ == "__main__":
    main()
