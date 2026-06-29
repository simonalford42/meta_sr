#!/usr/bin/env python3
"""Evaluate baseline/evolved PySR on the planet_eqs 24880 f2 task.

Submit this through planet_eval.sh, which requests the SLURM allocation
matching planet_eqs/sr.sh (-N 1, -n 32, 200G, default_partition). The
pre-exported planet_eval_data.pkl cache is created once outside this script;
this script only runs PySR and logs resonant-test metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


META_ROOT = Path(__file__).resolve().parent
PLANET_EVAL_DATA_PATH = META_ROOT / "planet_eval_data.pkl"

DEFAULT_NN_VERSION = 24880
DEFAULT_TARGET = "f2"
DEFAULT_LOSS_FN = "mse"
DEFAULT_TIME_IN_HOURS = 8
DEFAULT_NITERATIONS = 500000
DEFAULT_MAX_SIZE = 30
DEFAULT_N = 10000
DEFAULT_BATCH_SIZE = 1000
DEFAULT_CHECKPOINT_EVALS = 1_000_000


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
    return {
        "procs": num_cpus,
        "populations": 3 * num_cpus,
        "batching": True,
        "batch_size": int(config["batch_size"]),
        "output_directory": str(config["output_dir"]),
        "niterations": int(config["niterations"]),
        "binary_operators": ["+", "*", "/", "-", "^"],
        "maxsize": int(config["max_size"]),
        # timeout_in_seconds / max_evals are set per warm-start chunk in
        # run_checkpointed_fit so we can snapshot at 1e6 evals / 1h / 8h.
        "constraints": {"^": (-1, 1)},
        "ncyclesperiteration": 1000,
        "random_state": int(config["seed"]),
        # Keep the SLURM log clean: verbosity=0 silences PySR's periodic
        # hall-of-fame printing (and forces the progress bar off). We log the
        # frontier + per-checkpoint metrics ourselves and write hall_of_fame.csv.
        "verbosity": 0,
        "progress": False,
    }


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
        "acc": acc,
        "ll": planet_ll(truths_full, roc_preds, fixed_std=1.0),
        "roc_auc": roc_auc,
        "fpr": fpr,
        "fnr": fnr,
        "bias": bias,
    }


def _predict_at_index(model: Any, X: np.ndarray, index: Any) -> np.ndarray:
    preds = np.asarray(model.predict(X, index=index))
    if preds.ndim == 2:
        preds = preds[:, 0]
    preds = preds.reshape(-1)
    return np.where(np.isfinite(preds), preds, 4.0)


def evaluate_frontier(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[list, Optional[Dict[str, Any]]]:
    """Evaluate every Pareto-frontier equation on val + test.

    Returns (records, best). Each record has complexity, loss (train), equation,
    and full val/test metric dicts. ``best`` is the record maximizing validation
    log-likelihood (mirrors planet_eqs/evaluation.py best-complexity selection).
    """
    records: list = []
    try:
        equations = model.equations_
    except Exception as exc:
        print(f"WARNING: could not read frontier: {exc}", flush=True)
        return records, None

    for idx, row in equations.iterrows():
        rec = {
            "complexity": _safe_float(row.get("complexity")),
            "loss": _safe_float(row.get("loss")),
            "equation": str(row.get("equation")),
            "val_metrics": None,
            "test_metrics": None,
        }
        try:
            rec["val_metrics"] = calculate_planet_metrics(
                y_val, _predict_at_index(model, X_val, idx)
            )
            rec["test_metrics"] = calculate_planet_metrics(
                y_test, _predict_at_index(model, X_test, idx)
            )
        except Exception as exc:
            print(f"WARNING: frontier predict failed at index {idx}: {exc}", flush=True)
        records.append(rec)

    scored = [
        r
        for r in records
        if r["val_metrics"] is not None and _safe_float(r["val_metrics"].get("ll")) is not None
    ]
    best = max(scored, key=lambda r: r["val_metrics"]["ll"]) if scored else None
    return records, best


def _frontier_rows(records: list) -> list:
    """Flatten records into JSON/CSV-friendly rows for logging."""
    rows = []
    for r in records:
        val = r.get("val_metrics") or {}
        test = r.get("test_metrics") or {}
        rows.append(
            {
                "complexity": r.get("complexity"),
                "loss": r.get("loss"),
                "equation": r.get("equation"),
                "val_ll": _safe_float(val.get("ll")),
                "val_rmse": _safe_float(val.get("rmse")),
                "test_rmse": _safe_float(test.get("rmse")),
                "test_ll": _safe_float(test.get("ll")),
                "test_acc": _safe_float(test.get("acc")),
            }
        )
    return rows


def _best_summary(best: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Headline summary for a val-selected equation."""
    if best is None:
        return {"equation": None, "complexity": None, "val_ll": None, "metrics": None}
    return {
        "equation": best.get("equation"),
        "complexity": best.get("complexity"),
        "val_ll": _safe_float((best.get("val_metrics") or {}).get("ll")),
        "metrics": best.get("test_metrics"),
    }


def _fmt_evals(n: int) -> str:
    n = int(n)
    for suffix, base in (("e6", 1_000_000), ("e3", 1000)):
        if n >= base and n % base == 0:
            return f"{n // base}{suffix}"
    return str(n)


def build_checkpoints(config: Dict[str, Any]) -> list:
    """Ordered warm-start checkpoints: 1e6 evals, 1h, then the full budget."""
    total_seconds = int(60 * 60 * float(config["time_in_hours"]))
    checkpoints = [
        {"label": f"evals_{_fmt_evals(config['checkpoint_evals'])}", "kind": "evals", "value": int(config["checkpoint_evals"])},
        {"label": "time_1h", "kind": "time", "value": 3600},
        {"label": f"time_{config['time_in_hours']:g}h", "kind": "time", "value": total_seconds},
    ]
    # Drop time checkpoints at/after the final budget except the final one, and
    # any non-positive budget, keeping order.
    pruned = []
    for ck in checkpoints:
        if ck["kind"] == "time" and ck["value"] >= total_seconds and ck["label"] != checkpoints[-1]["label"]:
            continue
        pruned.append(ck)
    return pruned


# Sentinel "no eval cap" for time-bounded warm-start chunks.
_EVALS_UNBOUNDED = 10 ** 12


def run_checkpointed_fit(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    variable_names: list,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    checkpoints: list,
) -> list:
    """Warm-start PySR through each checkpoint, evaluating the frontier between.

    Returns a list of per-checkpoint dicts with the val-selected best equation's
    test metrics and the full frontier at that point. Checkpoints are cumulative;
    if a time budget is already exceeded we snapshot the current state instead of
    fitting further (chosen "skip overshoot" behavior).
    """
    import time

    total_seconds = max(
        (ck["value"] for ck in checkpoints if ck["kind"] == "time"), default=None
    )
    start = time.time()
    results: list = []

    for i, ck in enumerate(checkpoints):
        model.warm_start = i > 0
        elapsed = time.time() - start
        did_fit = True

        if ck["kind"] == "evals":
            model.max_evals = int(ck["value"])
            if total_seconds is not None:
                model.timeout_in_seconds = max(1, int(total_seconds - elapsed))
        else:  # time
            model.max_evals = _EVALS_UNBOUNDED
            remaining = ck["value"] - elapsed
            if remaining <= 1:
                did_fit = False  # already past this time budget; snapshot as-is
            else:
                model.timeout_in_seconds = int(remaining)

        print(
            f"\n=== checkpoint {ck['label']} ({ck['kind']}={ck['value']}) "
            f"elapsed={elapsed:.0f}s fit={did_fit} ===",
            flush=True,
        )
        if did_fit:
            model.fit(X_train, y_train, variable_names=variable_names)

        elapsed_after = time.time() - start
        records, best = evaluate_frontier(model, X_val, y_val, X_test, y_test)
        summary = _best_summary(best)
        results.append(
            {
                "label": ck["label"],
                "kind": ck["kind"],
                "value": ck["value"],
                "elapsed_s": round(elapsed_after, 1),
                "did_fit": did_fit,
                "best": summary,
                "frontier": _frontier_rows(records),
            }
        )
        m = summary.get("metrics") or {}
        print(
            f"  best complexity={summary.get('complexity')} val_ll={summary.get('val_ll')} "
            f"test_rmse={m.get('rmse')} test_ll={m.get('ll')} test_acc={m.get('acc')}",
            flush=True,
        )
        print(f"  equation: {summary.get('equation')}", flush=True)

    return results


def build_method(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return (pysr_kwargs, extra_code, method_meta)."""
    _repo_imports()
    base_kwargs = build_planet_sr_kwargs(config)
    method_meta: Dict[str, Any] = {"mode": _method_label(config)}

    if not config.get("evolve_results"):
        return base_kwargs, {}, method_meta

    from bundle_loader import load_bundle

    source = str(Path(config["evolve_results"]).expanduser())
    bundle = load_bundle(source)
    cfg = bundle.to_pysr_config(base_kwargs)
    allow_custom = bool(cfg.allow_custom_mutations)
    extra_code = {
        "custom_mutation_code": cfg.custom_mutation_code,
        "custom_survival_code": cfg.custom_survival_code,
        "custom_selection_code": cfg.custom_selection_code,
        "custom_loss_code": cfg.custom_loss_code,
        "allow_custom_mutations": allow_custom,
    }
    mutation_kwargs = {}
    for key, value in cfg.mutation_weights.items():
        if not key.startswith("weight_"):
            key = f"weight_{key}"
        if "custom_mutation" in key and not allow_custom:
            continue
        mutation_kwargs[key] = value
    pysr_kwargs = {**mutation_kwargs, **cfg.pysr_kwargs}

    method_meta = {
        "mode": "evolve",
        "source": source,
        "operators": [
            {
                "operator_type": t,
                "name": op.name,
                "generation": op.generation,
                "weight": op.weight,
            }
            for t, op in bundle.operators.items() if op is not None
        ],
        "best_hparams": bundle.best_hparams,
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


def load_planet_eval_data(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build it once with: "
            "conda activate new_bnn && python scripts/export_planet_eval_data.py"
        )

    with open(path, "rb") as f:
        data = pickle.load(f)

    meta = data.get("meta", {})
    expected = {
        "nn_version": DEFAULT_NN_VERSION,
        "target": DEFAULT_TARGET,
        "loss_fn": DEFAULT_LOSS_FN,
        "n": DEFAULT_N,
        "batch_size": DEFAULT_BATCH_SIZE,
    }
    mismatches = {
        key: (meta.get(key), value)
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path} metadata does not match planet_eval.py defaults: {mismatches}")

    if "X_val" not in data or "y_val" not in data:
        raise ValueError(
            f"{path} has no validation split. Re-export it with: "
            "conda activate new_bnn && python scripts/export_planet_eval_data.py"
        )

    return data


def fit_eval(config: Dict[str, Any]) -> None:
    """Fit PySR, evaluate resonant test, log wandb table."""
    _repo_imports()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    data = load_planet_eval_data(Path(config["data_path"]).expanduser().resolve())
    X_train = np.asarray(data["X_train"], dtype=np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.float32).reshape(-1)
    X_val = np.asarray(data["X_val"], dtype=np.float32)
    y_val = np.asarray(data["y_val"], dtype=np.float32)
    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.float32)
    variable_names = list(data["variable_names"])

    from parallel_eval_pysr import _import_pysr_regressor

    PySRRegressor = _import_pysr_regressor()
    pysr_kwargs, extra_code, method_meta = build_method(config)
    _load_dynamic_code(extra_code)

    print("PySR kwargs:", json.dumps(pysr_kwargs, indent=2, default=_json_default), flush=True)
    model = PySRRegressor(**pysr_kwargs)

    checkpoints = build_checkpoints(config)
    print(f"Checkpoints: {[c['label'] for c in checkpoints]}", flush=True)
    checkpoint_results = run_checkpointed_fit(
        model, X_train, y_train, variable_names, X_val, y_val, X_test, y_test, checkpoints
    )
    print("Done running planet PySR", flush=True)

    # Final checkpoint (full budget) carries the headline frontier + selection.
    final = checkpoint_results[-1] if checkpoint_results else {"best": _best_summary(None), "frontier": []}
    best = final["best"]
    frontier = final["frontier"]
    metrics = best.get("metrics") or {}

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
        "selection": "val_ll",
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "variable_names": variable_names,
        "equation": best.get("equation"),
        "best_complexity": best.get("complexity"),
        "best_val_ll": best.get("val_ll"),
        "metrics": metrics,
        "frontier": frontier,
        "checkpoints": checkpoint_results,
        "equations_path": str(equations_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    result_path = output_dir / "planet_eval_result.json"
    _write_json(result_path, result)

    print("\nFinal val-selected equation (test metrics)")
    for key in ["rmse", "acc", "ll", "roc_auc", "fpr", "fnr", "bias"]:
        print(f"  {key}: {metrics.get(key)}")
    print(f"  complexity: {best.get('complexity')}  val_ll: {best.get('val_ll')}")
    print(f"  equation: {best.get('equation')}", flush=True)

    print("\nFinal Pareto frontier (complexity | val_ll | test_rmse | equation)")
    for rec in frontier:
        print(
            f"  {rec.get('complexity')} | {rec.get('val_ll')} | "
            f"{rec.get('test_rmse')} | {rec.get('equation')}",
            flush=True,
        )

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

    metrics = result["metrics"] or {}

    columns = [
        "mode",
        "source",
        "nn_version",
        "target",
        "loss_fn",
        "n_train",
        "n_val",
        "n_test",
        "equation",
        "best_complexity",
        "best_val_ll",
        "rmse",
        "acc",
        "ll",
        "roc_auc",
        "fpr",
        "fnr",
        "bias",
        "slurm_job_id",
    ]
    table = wandb.Table(columns=columns)
    table.add_data(
        result["mode"],
        source or "baseline",
        result["nn_version"],
        result["target"],
        result["loss_fn"],
        result["n_train"],
        result.get("n_val"),
        result["n_test"],
        result["equation"],
        result.get("best_complexity"),
        result.get("best_val_ll"),
        metrics.get("rmse"),
        metrics.get("acc"),
        metrics.get("ll"),
        metrics.get("roc_auc"),
        metrics.get("fpr"),
        metrics.get("fnr"),
        metrics.get("bias"),
        result.get("slurm_job_id"),
    )
    wandb.log({"planet_eval/results_table": table})

    # Per-checkpoint table: val-selected best equation's test metrics at each of
    # 1e6 evals / 1h / 8h, plus scalar series keyed by checkpoint label.
    checkpoints = result.get("checkpoints") or []
    if checkpoints:
        ck_columns = [
            "label", "kind", "value", "elapsed_s", "complexity",
            "equation", "test_rmse", "test_ll", "test_acc", "val_ll",
        ]
        ck_table = wandb.Table(columns=ck_columns)
        for ck in checkpoints:
            best = ck.get("best") or {}
            m = best.get("metrics") or {}
            ck_table.add_data(
                ck.get("label"), ck.get("kind"), ck.get("value"), ck.get("elapsed_s"),
                best.get("complexity"), best.get("equation"),
                _safe_float(m.get("rmse")), _safe_float(m.get("ll")),
                _safe_float(m.get("acc")), best.get("val_ll"),
            )
            label = ck.get("label")
            wandb.log({
                f"planet_eval/{label}/test_rmse": _safe_float(m.get("rmse")),
                f"planet_eval/{label}/test_ll": _safe_float(m.get("ll")),
                f"planet_eval/{label}/test_acc": _safe_float(m.get("acc")),
                f"planet_eval/{label}/val_ll": best.get("val_ll"),
                f"planet_eval/{label}/complexity": best.get("complexity"),
                f"planet_eval/{label}/elapsed_s": ck.get("elapsed_s"),
            })
        wandb.log({"planet_eval/checkpoints_table": ck_table})

    frontier = result.get("frontier") or []
    if frontier:
        frontier_columns = [
            "complexity", "loss", "val_ll", "val_rmse",
            "test_rmse", "test_ll", "test_acc", "equation",
        ]
        frontier_table = wandb.Table(columns=frontier_columns)
        for rec in frontier:
            frontier_table.add_data(
                rec.get("complexity"), rec.get("loss"),
                rec.get("val_ll"), rec.get("val_rmse"),
                rec.get("test_rmse"), rec.get("test_ll"), rec.get("test_acc"),
                rec.get("equation"),
            )
        wandb.log({"planet_eval/frontier_table": frontier_table})

    wandb.log({f"planet_eval/{k}": v for k, v in metrics.items()})
    log_wandb_summary(
        run,
        extra_summary={
            f"planet_eval_{k}": v
            for k, v in {
                **metrics,
                "best_complexity": result.get("best_complexity"),
                "best_val_ll": result.get("best_val_ll"),
            }.items()
        },
    )
    finish_wandb(run)


def summarize_result(output_dir: Path) -> None:
    result_path = output_dir / "planet_eval_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"planet_eval did not write {result_path}")

    result = _read_json(result_path)
    metrics = result.get("metrics") or {}
    print(f"Result: {result_path}", flush=True)
    for ck in result.get("checkpoints") or []:
        m = (ck.get("best") or {}).get("metrics") or {}
        print(
            f"  [{ck.get('label')}] complexity={(ck.get('best') or {}).get('complexity')} "
            f"test_rmse={m.get('rmse')} test_ll={m.get('ll')} test_acc={m.get('acc')}",
            flush=True,
        )
    print("  final (val-selected) test metrics:", flush=True)
    for key in ["rmse", "acc", "ll", "roc_auc", "fpr", "fnr", "bias"]:
        print(f"    {key}: {metrics.get(key)}", flush=True)
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
        "data_path": str(PLANET_EVAL_DATA_PATH),
        "baseline": bool(args.baseline or not evolve_results),
        "evolve_results": evolve_results,
        "nn_version": DEFAULT_NN_VERSION,
        "target": DEFAULT_TARGET,
        "loss_fn": DEFAULT_LOSS_FN,
        "max_size": int(args.max_size),
        "time_in_hours": float(args.time_in_hours),
        "checkpoint_evals": int(args.checkpoint_evals),
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
    print(f"Data: {config['data_path']}", flush=True)
    print(f"PySR workers: {config['num_cpus']}", flush=True)
    if args.dry_run:
        print("Dry run requested; not fitting PySR.", flush=True)
        return

    print(f"planet_eval job {os.environ.get('SLURM_JOB_ID', 'local')} on {os.uname().nodename}", flush=True)
    fit_eval(config)
    summarize_result(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate planet_eqs PySR baseline or evolved operators inside planet_eval.sh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--evolve-results", type=str, default=None, help="Path/run id with run_data.json.")
    parser.add_argument("--baseline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--time-in-hours", type=float, default=DEFAULT_TIME_IN_HOURS)
    parser.add_argument(
        "--checkpoint-evals", type=int, default=DEFAULT_CHECKPOINT_EVALS,
        help="Cumulative eval count for the first warm-start checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-log", "--no_log", dest="no_log", action="store_true", help="disable wandb logging")
    parser.add_argument("--results-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-wandb", dest="no_log", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    driver(args)


if __name__ == "__main__":
    main()
