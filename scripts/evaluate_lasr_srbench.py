#!/usr/bin/env python3
"""Run LaSR on the canonical SRBench ground-truth grid.

The archived LaSR experiment driver only accepts its bundled equation CSV. This
adapter keeps LaSR's published search configuration while feeding it the same
PMLB rows, train/validation split, target-noise convention, and whole-frontier
ground-truth check used by ``srbench_full_eval.py``.

Typical workflow (``submit`` is the only subcommand that calls ``sbatch``)::

    python scripts/evaluate_lasr_srbench.py plan
    python scripts/evaluate_lasr_srbench.py submit --output-dir runs/lasr_srbench_nemo_1seed
    python inspect_srbench_results.py --run-id lasr_srbench_nemo_1seed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LASR_ROOT = PROJECT_ROOT / "LaSR.jl"
LASR_PYTHON = PROJECT_ROOT / ".venv-lasr" / "bin" / "python"
DEFAULT_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]
DEFAULT_MODEL = "mistralai/mistral-nemo"
DEFAULT_MODEL_URL = "https://openrouter.ai/api/v1"

# Measured from the 2026-08-31 local one-iteration OpenRouter smoke test:
# 3 completed responses plus one failed request at weights 0.0001, using a
# 256-token completion cap, cost $0.00010236. LaSR's LLM event rate scales
# approximately linearly with iterations and the three LLM mutation weights.
CALIBRATION = {
    "iterations": 1,
    "weight_sum": 0.0003,
    "max_tokens": 256,
    "attempted_calls": 4,
    "completed_calls": 3,
    "cost_usd": 0.00010236,
}


def _load_project_env() -> None:
    """Load missing values from the gitignored project ``.env`` file."""
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_datasets(split_file: str | Path) -> list[str]:
    path = Path(split_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    datasets = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(datasets) != len(set(datasets)):
        raise ValueError(f"Duplicate datasets in {path}")
    return datasets


def _tasks(datasets: list[str], seed: int, noise_levels: list[float],
           max_samples: int) -> list[dict[str, Any]]:
    tasks = []
    for dataset in datasets:
        for noise in noise_levels:
            tasks.append({
                "task_index": len(tasks),
                "dataset_name": dataset,
                "seed": int(seed),
                "run_index": 0,
                "target_noise": float(noise),
                "max_samples": int(max_samples),
            })
    return tasks


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "model_url": args.model_url.rstrip("/"),
        "num_iterations": int(args.num_iterations),
        "max_tokens": int(args.max_tokens),
        "llm_mutate_weight": float(args.llm_mutate_weight),
        "llm_crossover_weight": float(args.llm_crossover_weight),
        "llm_gen_random_weight": float(args.llm_gen_random_weight),
        "num_pareto_context": int(args.num_pareto_context),
        "idea_threshold": int(args.idea_threshold),
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
        "noise_levels": [float(x) for x in args.noise_levels],
        "split_file": str(Path(args.split_file)),
    }


def _estimate(args: argparse.Namespace, n_fits: int) -> dict[str, Any]:
    weight_sum = (
        args.llm_mutate_weight
        + args.llm_crossover_weight
        + args.llm_gen_random_weight
    )
    scale_per_fit = (
        args.num_iterations / CALIBRATION["iterations"]
        * weight_sum / CALIBRATION["weight_sum"]
    )
    attempted = n_fits * CALIBRATION["attempted_calls"] * scale_per_fit
    completed = n_fits * CALIBRATION["completed_calls"] * scale_per_fit
    floor_cost = n_fits * CALIBRATION["cost_usd"] * scale_per_fit
    # Input charges do not grow with max_tokens, so scaling the entire measured
    # charge is deliberately a conservative ceiling rather than a point guess.
    token_cap_scale = max(1.0, args.max_tokens / CALIBRATION["max_tokens"])
    ceiling_cost = floor_cost * token_cap_scale
    smoke_log_bytes = 6289
    floor_log_gb = completed / CALIBRATION["completed_calls"] * smoke_log_bytes / 1e9
    ceiling_log_gb = floor_log_gb * token_cap_scale
    return {
        "calibration": CALIBRATION,
        "estimated_attempted_llm_calls": round(attempted),
        "estimated_completed_llm_calls": round(completed),
        "estimated_cost_usd_floor": floor_cost,
        "estimated_cost_usd_conservative_ceiling": ceiling_cost,
        "estimated_llm_log_gb_floor": floor_log_gb,
        "estimated_llm_log_gb_conservative_ceiling": ceiling_log_gb,
        "note": (
            "Empirical extrapolation from one stochastic smoke run. The floor "
            "assumes completion lengths remain at or below 256 tokens; the "
            "ceiling pessimistically scales the entire charge and log size by "
            "the configured completion-token cap."
        ),
    }


def _print_plan(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    datasets = _read_datasets(args.split_file)
    tasks = _tasks(datasets, args.seed, args.noise_levels, args.max_samples)
    missing = []
    for dataset in datasets:
        data = PROJECT_ROOT / "pmlb" / "datasets" / dataset / f"{dataset}.tsv.gz"
        metadata = data.with_name("metadata.yaml")
        if not data.exists() or not metadata.exists():
            missing.append(dataset)
    if missing:
        raise FileNotFoundError(f"Missing PMLB data/metadata for: {', '.join(missing)}")

    estimate = _estimate(args, len(tasks))
    print("LaSR full SRBench GT plan")
    print(f"  datasets: {len(datasets)}")
    print(f"  seeds: 1 ({args.seed})")
    print(f"  noise levels: {args.noise_levels}")
    print(f"  fits: {len(tasks)}")
    print(f"  model: {args.model}")
    print(f"  LaSR iterations: {args.num_iterations}")
    print(
        "  LLM weights (mutate/crossover/random): "
        f"{args.llm_mutate_weight:g}/{args.llm_crossover_weight:g}/"
        f"{args.llm_gen_random_weight:g}"
    )
    print(
        "  estimated LLM calls: "
        f"{estimate['estimated_completed_llm_calls']:,} completed "
        f"({estimate['estimated_attempted_llm_calls']:,} attempts)"
    )
    print(
        "  estimated OpenRouter cost: "
        f"${estimate['estimated_cost_usd_floor']:.0f}–"
        f"${estimate['estimated_cost_usd_conservative_ceiling']:.0f}"
    )
    print(
        "  estimated raw LLM logs: "
        f"{estimate['estimated_llm_log_gb_floor']:.0f}–"
        f"{estimate['estimated_llm_log_gb_conservative_ceiling']:.0f} GB"
    )
    print("  inverse-trig tasks retained as canonical unsolvable cases: 3")
    return datasets, tasks


def _prepare(args: argparse.Namespace) -> Path:
    datasets, tasks = _print_plan(args)
    output_dir = Path(args.output_dir).resolve()
    batch_dir = output_dir / "lasr_batch"
    results_dir = batch_dir / "results"
    logs_dir = batch_dir / "logs"
    slurm_dir = output_dir / "slurm"
    for path in (results_dir, logs_dir, slurm_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = _run_config(args)
    manifest = {
        "mode": "lasr",
        "backend": "lasr-paper-artifact",
        "created_at": _utc_now(),
        "evaluation_types": ["ground_truth"],
        "srbench_edition": 2021,
        "datasets": datasets,
        "n_datasets": len(datasets),
        "seed": args.seed,
        "n_runs": 1,
        "seeds": [args.seed],
        "noise_levels": args.noise_levels,
        "max_samples": args.max_samples,
        "split_file": args.split_file,
        "unsolvable_tasks": [
            "feynman_test_10", "feynman_I_26_2", "feynman_I_30_5"
        ],
        "lasr_config": config,
        "cost_estimate": _estimate(args, len(tasks)),
        "batches": [{
            "noise": "all",
            "batch_dir": "lasr_batch",
            "n_tasks": len(tasks),
        }],
    }
    manifest_path = output_dir / "manifest.json"
    tasks_path = batch_dir / "tasks.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        old_comparable = {
            key: old.get(key)
            for key in ("datasets", "seeds", "noise_levels", "max_samples", "lasr_config")
        }
        new_comparable = {
            key: manifest.get(key)
            for key in ("datasets", "seeds", "noise_levels", "max_samples", "lasr_config")
        }
        if old_comparable != new_comparable:
            raise RuntimeError(
                f"{manifest_path} already exists with a different configuration; "
                "choose another --output-dir"
            )
        print(f"Resuming compatible run directory: {output_dir}")
    else:
        _write_json_atomic(manifest_path, manifest)
        _write_json_atomic(tasks_path, tasks)
        print(f"Prepared run directory: {output_dir}")
    if not tasks_path.exists():
        _write_json_atomic(tasks_path, tasks)
    return output_dir


def _sbatch_commands(args: argparse.Namespace, output_dir: Path,
                     n_tasks: int, array_job_id: str | None = None) -> tuple[list[str], list[str]]:
    script = str(Path(__file__).resolve())
    run_sh = str(PROJECT_ROOT / "run.sh")
    array = [
        "sbatch", "--parsable",
        f"--array=0-{n_tasks - 1}%{args.max_concurrent}",
        f"--partition={args.partition}",
        f"--time={args.time_limit}",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.mem}",
        "--job-name=lasr-srb-gt",
        f"--output={output_dir}/slurm/%A_%a.out",
        f"--error={output_dir}/slurm/%A_%a.err",
        run_sh, script, "worker", "--output-dir", str(output_dir),
        "--task-index-env",
    ]
    dependency = array_job_id or "<array-job-id>"
    aggregate = [
        "sbatch", "--parsable", f"--dependency=afterany:{dependency}",
        f"--partition={args.partition}", "--time=00:20:00",
        "--cpus-per-task=1", "--mem=4G", "--job-name=lasr-srb-final",
        f"--output={output_dir}/slurm/final_%j.out",
        f"--error={output_dir}/slurm/final_%j.err",
        run_sh, script, "aggregate", "--output-dir", str(output_dir),
    ]
    return array, aggregate


def _submit(args: argparse.Namespace) -> None:
    _load_project_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set and was not found in the project .env."
        )
    if not LASR_PYTHON.exists():
        raise SystemExit("LaSR environment missing; run scripts/setup_lasr.sh first")
    output_dir = _prepare(args)
    tasks = json.loads((output_dir / "lasr_batch" / "tasks.json").read_text())
    array_cmd, _ = _sbatch_commands(args, output_dir, len(tasks))
    array_job_id = subprocess.check_output(array_cmd, text=True).strip().split(";")[0]
    _, aggregate_cmd = _sbatch_commands(
        args, output_dir, len(tasks), array_job_id=array_job_id
    )
    aggregate_job_id = subprocess.check_output(aggregate_cmd, text=True).strip().split(";")[0]
    print(f"Submitted LaSR array {array_job_id} and aggregator {aggregate_job_id}")
    print(f"Results: {output_dir}")


def _ensure_lasr_python() -> None:
    if not LASR_PYTHON.exists():
        raise SystemExit("LaSR environment missing; run scripts/setup_lasr.sh first")
    if Path(sys.executable).resolve() != LASR_PYTHON.resolve():
        env = os.environ.copy()
        env.setdefault("JULIA_DEPOT_PATH", str(PROJECT_ROOT / ".julia_depot"))
        julia = subprocess.check_output(["bash", "-lc", "command -v julia"], text=True).strip()
        env.setdefault("PYTHON_JULIACALL_EXE", julia)
        os.execve(
            str(LASR_PYTHON),
            [str(LASR_PYTHON), "-u", str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )


def _safe_prompt_name(name: str, index: int) -> str:
    # LaSR reserves capital C as its arbitrary-constant placeholder.
    if name == "C":
        return "C_var"
    if name.isidentifier():
        return name
    return f"variable_{index}"


def _llm_stats(log_path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "attempted": 0,
        "completed": 0,
        "failed_records": 0,
        "completed_by_mode": {},
    }
    if not log_path.exists():
        return stats
    with log_path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("[llm_input|"):
                stats["attempted"] += 1
            elif line.startswith("[llm_output|"):
                stats["completed"] += 1
                mode = line.split("|", 1)[1].split("]", 1)[0]
                stats["completed_by_mode"][mode] = (
                    stats["completed_by_mode"].get(mode, 0) + 1
                )
            elif line.startswith("[") and "|failed]" in line:
                stats["failed_records"] += 1
    return stats


def _finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return value if math.isfinite(value) else None


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def _run_worker(args: argparse.Namespace) -> None:
    _ensure_lasr_python()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY was not exported to the worker")

    # Import the installed frozen PySR before exposing the source artifact on
    # sys.path; this avoids the artifact's machine-specific juliapkg.json.
    import numpy as np
    from pysr import PySRRegressor

    sys.path.append(str(LASR_ROOT))
    from experiments.model import custom_loss
    from evaluation import get_dataset_var_names
    from parallel_eval_pysr import _remap_formula_variables, add_noise
    from domains import get_domain
    from utils import load_srbench_dataset

    output_dir = Path(args.output_dir).resolve()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    tasks = json.loads((output_dir / "lasr_batch" / "tasks.json").read_text())
    if args.task_index_env:
        try:
            task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        except KeyError as exc:
            raise SystemExit("SLURM_ARRAY_TASK_ID is not set") from exc
    else:
        if args.task_index is None:
            raise SystemExit("Pass --task-index or --task-index-env")
        task_index = args.task_index
    if not 0 <= task_index < len(tasks):
        raise SystemExit(f"Task index {task_index} is outside 0..{len(tasks)-1}")

    task = tasks[task_index]
    result_path = output_dir / "lasr_batch" / "results" / f"task_{task_index:06d}.json"
    if result_path.exists() and not args.force:
        existing = json.loads(result_path.read_text())
        if existing.get("error") is None:
            print(f"Task {task_index} already complete; skipping")
            return

    dataset_name = task["dataset_name"]
    run_seed = int(task["seed"]) + int(task.get("run_index", 0))
    noise = float(task["target_noise"])
    config = manifest["lasr_config"]
    task_log_dir = output_dir / "lasr_batch" / "logs" / f"task_{task_index:06d}"
    task_log_dir.mkdir(parents=True, exist_ok=True)
    llm_log = task_log_dir / "llm_calls.txt"
    llm_log.touch(exist_ok=True)

    started = time.monotonic()
    result: dict[str, Any] = {
        "task_index": task_index,
        "dataset_name": dataset_name,
        "seed": run_seed,
        "target_noise": noise,
        "model": config["model"],
        "error": None,
    }
    try:
        X, y, ground_truth = load_srbench_dataset(
            dataset_name,
            max_samples=int(task["max_samples"]),
            data_seed=int(task["seed"]),
        )
        if not ground_truth:
            raise ValueError(f"No ground-truth formula found for {dataset_name}")
        feature_names = get_dataset_var_names(dataset_name)
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"Feature-name mismatch: {len(feature_names)} names for {X.shape[1]} columns"
            )
        finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y = X[finite], y[finite]
        rng = np.random.RandomState(run_seed)
        indices = rng.permutation(len(y))
        n_train = int(0.8 * len(y))
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train_base, y_val = y[train_idx], y[val_idx]
        y_train = add_noise(y_train_base.copy(), noise, seed=run_seed + 1000)

        variable_names = [f"x{i}" for i in range(X.shape[1])]
        prompt_names = [_safe_prompt_name(name, i) for i, name in enumerate(feature_names)]
        var_order = dict(zip(variable_names, prompt_names))
        ground_truth_mapped = _remap_formula_variables(
            ground_truth, feature_names, variable_names
        )
        llm_options = {
            "active": True,
            "weights": {
                "llm_mutate": config["llm_mutate_weight"],
                "llm_crossover": config["llm_crossover_weight"],
                "llm_gen_random": config["llm_gen_random_weight"],
            },
            "prompt_evol": True,
            "prompt_concepts": True,
            "num_pareto_context": config["num_pareto_context"],
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "model": config["model"],
            "api_kwargs": {
                "max_tokens": config["max_tokens"],
                "url": config["model_url"],
            },
            "http_kwargs": {"retries": 5, "readtimeout": 360},
            "llm_recorder_dir": str(task_log_dir),
            "idea_threshold": config["idea_threshold"],
            "prompts_dir": str(LASR_ROOT / "prompts") + "/",
            "is_parametric": False,
            "llm_context": "",
            "var_order": var_order,
        }
        pysr_temp = task_log_dir / "pysr_tmp"
        pysr_temp.mkdir(exist_ok=True)
        model = PySRRegressor(
            niterations=config["num_iterations"],
            ncyclesperiteration=550,
            populations=15,
            population_size=33,
            maxsize=30,
            binary_operators=["+", "*", "-", "/", "^"],
            unary_operators=["exp", "log", "sqrt", "sin", "cos"],
            full_objective=custom_loss,
            verbosity=0,
            temp_equation_file=True,
            tempdir=str(pysr_temp),
            delete_tempfiles=True,
            llm_options=llm_options,
            weight_randomize=0.1,
            should_simplify=True,
            random_state=run_seed,
            constraints={
                "sin": 10, "cos": 10, "exp": 20, "log": 20,
                "sqrt": 20, "pow": (-1, 20),
            },
            nested_constraints={
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
                "sqrt": {"sqrt": 0},
            },
        )
        print(
            f"[{task_index}/{len(tasks)}] {dataset_name} seed={run_seed} noise={noise:g} "
            f"rows={len(X_train)}/{len(X_val)} features={X.shape[1]}",
            flush=True,
        )
        search_started = time.monotonic()
        model.fit(X_train, y_train)
        search_seconds = time.monotonic() - search_started

        best = model.get_best()
        best_equation = str(best["equation"]) if best is not None else None
        best_index = best.name if best is not None else None
        best_loss = _finite_or_none(best["loss"] if best is not None else None)
        best_complexity = int(best["complexity"]) if best is not None else None
        y_pred = np.asarray(model.predict(X_val))
        y_pred = np.clip(y_pred, -1e10, 1e10)
        ss_res = float(np.sum((y_val - y_pred) ** 2))
        ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        domain = get_domain("srbench")
        match = domain.check_solved(
            equations_df=model.equations_,
            best_df_index=best_index,
            target=ground_truth_mapped,
            var_names=variable_names,
            predict_fn=lambda idx: model.predict(X_val, index=int(idx)),
            y_val=y_val,
            predict_on=lambda idx, Xq: model.predict(Xq, index=int(idx)),
            dataset_name=dataset_name,
        )
        matched_idx = match.get("matched_df_index")
        matched_equation = None
        if matched_idx is not None:
            matched_equation = str(model.equations_.loc[matched_idx]["equation"])
        frontier = []
        for idx, row in model.equations_.iterrows():
            frontier.append({
                "index": int(idx),
                "equation": str(row["equation"]),
                "loss": _finite_or_none(row.get("loss")),
                "score": _finite_or_none(row.get("score")),
                "complexity": int(row["complexity"]),
            })
        result.update({
            "gt_formula": ground_truth,
            "gt_formula_mapped": ground_truth_mapped,
            "feature_names": feature_names,
            "prompt_variable_names": prompt_names,
            "n_train": len(X_train),
            "n_validation": len(X_val),
            "best_equation": best_equation,
            "best_loss": best_loss,
            "best_complexity": best_complexity,
            "r2_score": _finite_or_none(r2),
            "gt_match_score": 1.0 if match.get("match") else 0.0,
            "gt_match_details": match,
            "gt_matched_equation": matched_equation,
            "search_seconds": search_seconds,
            "pareto_frontier": frontier,
        })
    except Exception as exc:
        import traceback
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        result["runtime_seconds"] = time.monotonic() - started
        result["llm_calls"] = _llm_stats(llm_log)
        result["completed_at"] = _utc_now()
        _write_json_atomic(result_path, result)
        print(f"Wrote {result_path} (error={result['error']!r})", flush=True)
    if result["error"] is not None:
        raise SystemExit(1)


def _aggregate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    sys.path.insert(0, str(PROJECT_ROOT))
    import srbench_results_io as srio

    keyed = srio.build_keyed_results(output_dir, manifest)
    srio.save_keyed_results(output_dir, keyed, meta={
        "mode": "lasr",
        "n_datasets": manifest["n_datasets"],
        "n_runs": 1,
        "noise_levels": manifest["noise_levels"],
        "lasr_config": manifest["lasr_config"],
    })
    metrics = srio.aggregate_metrics(keyed, manifest["noise_levels"])
    result_paths = sorted((output_dir / "lasr_batch" / "results").glob("task_*.json"))
    attempted = completed = failed_records = 0
    errored = 0
    for path in result_paths:
        row = json.loads(path.read_text())
        calls = row.get("llm_calls") or {}
        attempted += int(calls.get("attempted", 0))
        completed += int(calls.get("completed", 0))
        failed_records += int(calls.get("failed_records", 0))
        errored += int(row.get("error") is not None)
    n_success = sum(1 for row in keyed.values() if row["present"] and row["error"] is None)
    summary = {
        "n_expected": len(keyed),
        "n_result_files": len(result_paths),
        "n_successful": n_success,
        "n_errored": errored,
        "n_missing": len(keyed) - len(result_paths),
        "llm_attempted_calls": attempted,
        "llm_completed_calls": completed,
        "llm_failed_records": failed_records,
        "metrics": metrics,
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    print(
        f"Results: {n_success}/{len(keyed)} successful, {errored} errored, "
        f"{len(keyed) - len(result_paths)} missing"
    )
    print(
        f"LLM: {completed:,} completed responses / {attempted:,} attempts; "
        f"{failed_records:,} failure records"
    )
    print(srio.format_metrics_console(metrics, manifest["noise_levels"]))
    print(f"\nInspect with: python inspect_srbench_results.py --run-id {output_dir.name}")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-file", default="splits/srbench_all.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--num-iterations", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--llm-mutate-weight", type=float, default=0.01)
    parser.add_argument("--llm-crossover-weight", type=float, default=0.01)
    parser.add_argument("--llm-gen-random-weight", type=float, default=0.01)
    parser.add_argument("--num-pareto-context", type=int, default=3)
    parser.add_argument("--idea-threshold", type=int, default=30)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the LaSR paper artifact on SRBench GT tasks."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Validate and print task/cost estimates; submits nothing.")
    _add_run_args(plan)

    prepare = sub.add_parser("prepare", help="Create a run manifest; submits nothing.")
    _add_run_args(prepare)
    prepare.add_argument("--output-dir", required=True)

    submit = sub.add_parser("submit", help="Prepare and submit the worker array and aggregator.")
    _add_run_args(submit)
    submit.add_argument("--output-dir", required=True)
    submit.add_argument("--partition", default="default_partition")
    submit.add_argument("--time-limit", default="24:00:00")
    submit.add_argument("--cpus-per-task", type=int, default=4)
    submit.add_argument("--mem", default="8G")
    submit.add_argument("--max-concurrent", type=int, default=32)

    worker = sub.add_parser("worker", help="Run one task from a prepared manifest.")
    worker.add_argument("--output-dir", required=True)
    worker.add_argument("--task-index", type=int)
    worker.add_argument("--task-index-env", action="store_true")
    worker.add_argument("--force", action="store_true")

    aggregate = sub.add_parser("aggregate", help="Aggregate completed task JSON files.")
    aggregate.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "plan":
        _print_plan(args)
    elif args.command == "prepare":
        _prepare(args)
    elif args.command == "submit":
        _submit(args)
    elif args.command == "worker":
        _run_worker(args)
    elif args.command == "aggregate":
        _aggregate(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
