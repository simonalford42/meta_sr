#!/usr/bin/env python3
"""Run MOD-SR (ICML 2026, baselines/mod-sr) on the canonical SRBench GT grid.

MOD-SR's own benchmark loader re-samples 200 points from each equation string,
so it is not comparable to our numbers.  This adapter feeds MOD-SR the same
PMLB rows, train/validation split and target-noise convention as
``evaluate_lasr_srbench.py`` / ``srbench_full_eval.py``, then scores its
refined candidates with our gated whole-frontier GT-match check.

Pipeline (two conda envs):
  prepare   [meta_sr]  write tasks.json + inputs/task_N.npz
  predict   [modsr]    baselines/modsr_adapter.py -> predictions/task_N.json
  score     [meta_sr]  candidates -> Pareto frontier -> GT match -> results/task_N.json
  aggregate [meta_sr]  srbench_results_io summary

``shard`` does predict+score for one shard (what each SLURM array task runs);
``submit`` is the only subcommand that calls ``sbatch``.

    python evaluate_modsr_srbench.py prepare --output-dir runs/modsr_srbench_1seed
    python evaluate_modsr_srbench.py shard --output-dir ... --task-indices 0 1 --cpu
    python evaluate_modsr_srbench.py submit --output-dir runs/modsr_srbench_1seed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
ADAPTER = PROJECT_ROOT / "baselines" / "modsr_adapter.py"
MODSR_PYTHON = Path.home() / ".conda" / "envs" / "modsr" / "bin" / "python"
DEFAULT_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]
BATCH = "modsr_batch"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    import numpy as np
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=_json_default))
    os.replace(tmp, path)


def _finite_or_none(v):
    import math
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _read_datasets(split_file: str) -> list[str]:
    path = Path(split_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    datasets = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    if len(datasets) != len(set(datasets)):
        raise ValueError(f"Duplicate datasets in {path}")
    return datasets


def _config(args) -> dict[str, Any]:
    return {
        "inference": "infer_modsr (bypass DFEX), e2e encoder, modsr_10d.pth",
        "top_k": args.top_k,
        "ddim_steps": args.ddim_steps,
        "n_encoder_points": args.n_encoder_points,
    }


# --------------------------------------------------------------------- prepare
def _prepare(args) -> Path:
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT))
    from evaluation import get_dataset_var_names
    from parallel_eval_pysr import _remap_formula_variables, add_noise
    from utils import load_srbench_dataset

    datasets = _read_datasets(args.split_file)
    output_dir = Path(args.output_dir).resolve()
    batch_dir = output_dir / BATCH
    for sub in ("inputs", "predictions", "results"):
        (batch_dir / sub).mkdir(parents=True, exist_ok=True)
    (output_dir / "slurm").mkdir(exist_ok=True)

    manifest = {
        "mode": "modsr",
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
        "modsr_config": _config(args),
        "batches": [{"noise": "all", "batch_dir": BATCH, "n_tasks": None}],
    }
    manifest_path = output_dir / "manifest.json"
    keys = ("datasets", "seeds", "noise_levels", "max_samples", "modsr_config")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if {k: old.get(k) for k in keys} != {k: manifest.get(k) for k in keys}:
            raise RuntimeError(f"{manifest_path} exists with a different config")
        print(f"Resuming compatible run directory: {output_dir}")
        return output_dir

    tasks = []
    for dataset in datasets:
        X, y, gt = load_srbench_dataset(dataset, max_samples=args.max_samples,
                                        data_seed=args.seed)
        if not gt:
            raise ValueError(f"No ground truth for {dataset}")
        feature_names = get_dataset_var_names(dataset)
        finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y = X[finite], y[finite]
        variable_names = [f"x{i}" for i in range(X.shape[1])]
        gt_mapped = _remap_formula_variables(gt, feature_names, variable_names)
        for noise in args.noise_levels:
            run_seed = args.seed
            rng = np.random.RandomState(run_seed)
            perm = rng.permutation(len(y))
            n_train = int(0.8 * len(y))
            tr, va = perm[:n_train], perm[n_train:]
            y_train = add_noise(y[tr].copy(), noise, seed=run_seed + 1000)
            idx = len(tasks)
            np.savez(batch_dir / "inputs" / f"task_{idx:06d}.npz",
                     X_train=X[tr], y_train=y_train, X_val=X[va], y_val=y[va])
            tasks.append({
                "task_index": idx, "dataset_name": dataset, "seed": args.seed,
                "run_index": 0, "target_noise": float(noise),
                "max_samples": args.max_samples, "n_features": int(X.shape[1]),
                "gt_formula": gt, "gt_formula_mapped": gt_mapped,
                "feature_names": feature_names,
            })
    manifest["batches"][0]["n_tasks"] = len(tasks)
    _write_json_atomic(batch_dir / "tasks.json", tasks)
    _write_json_atomic(manifest_path, manifest)
    too_wide = sorted({t["dataset_name"] for t in tasks if t["n_features"] > 10})
    print(f"Prepared {len(tasks)} tasks ({len(datasets)} datasets) in {output_dir}")
    if too_wide:
        print(f"  >10 features (MOD-SR unsupported, will error): {too_wide}")
    return output_dir


# --------------------------------------------------------------------- predict
def _predict(args, output_dir: Path, indices: list[int] | None) -> None:
    manifest = json.loads((output_dir / "manifest.json").read_text())
    cfg = manifest["modsr_config"]
    cmd = [str(MODSR_PYTHON), "-u", str(ADAPTER),
           "--batch-dir", str(output_dir / BATCH),
           "--top-k", str(cfg["top_k"]), "--ddim-steps", str(cfg["ddim_steps"]),
           "--n-encoder-points", str(cfg["n_encoder_points"])]
    if indices:
        cmd += ["--task-indices", *map(str, indices)]
    if args.cpu:
        cmd.append("--cpu")
    if args.force:
        cmd.append("--force")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------- score
def _complexity(expr) -> int:
    import sympy as sp
    return sum(1 for _ in sp.preorder_traversal(expr))


def _score_task(batch_dir: Path, task: dict, force: bool) -> dict | None:
    import numpy as np
    import pandas as pd
    import sympy as sp
    sys.path.insert(0, str(PROJECT_ROOT))
    from domains import get_domain

    i = task["task_index"]
    res_path = batch_dir / "results" / f"task_{i:06d}.json"
    pred_path = batch_dir / "predictions" / f"task_{i:06d}.json"
    if res_path.exists() and not force:
        return None
    if not pred_path.exists():
        print(f"[{i}] no prediction yet; skipping score")
        return None
    pred = json.loads(pred_path.read_text())
    t0 = time.monotonic()
    result = {
        "task_index": i, "dataset_name": task["dataset_name"],
        "seed": task["seed"], "target_noise": task["target_noise"],
        "gt_formula": task["gt_formula"],
        "gt_formula_mapped": task["gt_formula_mapped"],
        "error": pred.get("error"),
        "predict_seconds": pred.get("predict_seconds"),
        "sample_seconds": pred.get("sample_seconds"),
        "refine_seconds": pred.get("refine_seconds"),
        "device": pred.get("device"),
    }
    try:
        if pred.get("error"):
            raise RuntimeError(pred["error"])
        data = np.load(batch_dir / "inputs" / f"task_{i:06d}.npz")
        X_tr, y_tr = data["X_train"], data["y_train"]
        X_va, y_va = data["X_val"], data["y_val"]
        d = X_tr.shape[1]
        syms = sp.symbols([f"x{j}" for j in range(d)])
        local = {f"x{j}": syms[j] for j in range(d)}

        rows, fns = [], []
        for c in pred.get("candidates", []):
            try:
                expr = sp.sympify(c["equation"], locals=local)
                fn = sp.lambdify(syms, expr, modules="numpy")

                def predict(Xq, fn=fn):
                    with np.errstate(all="ignore"):
                        out = fn(*[Xq[:, j] for j in range(d)])
                    return np.broadcast_to(np.asarray(out, dtype=float),
                                           (Xq.shape[0],)).copy()
                yp = predict(X_tr)
                loss = float(np.mean((y_tr - yp) ** 2))
                if not np.isfinite(loss):
                    continue
            except Exception:
                continue
            rows.append({"equation": c["equation"], "complexity": _complexity(expr),
                         "loss": loss, "rank": c.get("rank")})
            fns.append(predict)
        if not rows:
            raise RuntimeError("no valid candidates")

        # Pareto frontier over (complexity, train MSE), like PySR's hall of fame.
        order = sorted(range(len(rows)), key=lambda k: (rows[k]["complexity"], rows[k]["loss"]))
        frontier, best_loss_so_far = [], float("inf")
        for k in order:
            if rows[k]["loss"] < best_loss_so_far:
                frontier.append(k)
                best_loss_so_far = rows[k]["loss"]
        df = pd.DataFrame([rows[k] for k in frontier], index=frontier)
        best_idx = min(frontier, key=lambda k: rows[k]["loss"])

        def predict_fn(idx):
            return np.clip(fns[int(idx)](X_va), -1e10, 1e10)

        domain = get_domain("srbench")
        var_names = [f"x{j}" for j in range(d)]
        match = domain.check_solved(
            equations_df=df, best_df_index=best_idx,
            target=task["gt_formula_mapped"], var_names=var_names,
            predict_fn=predict_fn, y_val=y_va,
            predict_on=lambda idx, Xq: fns[int(idx)](Xq),
            dataset_name=task["dataset_name"],
        )
        y_pred = predict_fn(best_idx)
        ss_res = float(np.sum((y_va - y_pred) ** 2))
        ss_tot = float(np.sum((y_va - np.mean(y_va)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        frontier_out = []
        for k in frontier:
            yp = predict_fn(k)
            frontier_out.append({
                "index": int(k), "equation": rows[k]["equation"],
                "complexity": rows[k]["complexity"],
                "loss": _finite_or_none(rows[k]["loss"]),
                "val_r2": _finite_or_none(
                    1.0 - float(np.sum((y_va - yp) ** 2)) / (ss_tot + 1e-10)),
            })
        matched = match.get("matched_df_index")
        result.update({
            "n_candidates": len(rows),
            "best_equation": rows[best_idx]["equation"],
            "best_loss": _finite_or_none(rows[best_idx]["loss"]),
            "best_complexity": rows[best_idx]["complexity"],
            "r2_score": _finite_or_none(r2),
            "gt_match_score": 1.0 if match.get("match") else 0.0,
            "gt_match_details": {k: v for k, v in match.items() if k != "order"},
            "gt_matched_equation": rows[int(matched)]["equation"] if matched is not None else None,
            "pareto_frontier": frontier_out,
            "all_candidates": rows,
            "error": None,
        })
    except Exception as exc:
        import traceback
        result["error"] = result["error"] or f"{type(exc).__name__}: {exc}"
        if "traceback" not in result:
            result["traceback"] = traceback.format_exc()
    result["score_seconds"] = time.monotonic() - t0
    # runtime_seconds = model wall time (sampling + refinement), excluding scoring.
    result["runtime_seconds"] = pred.get("predict_seconds")
    result["completed_at"] = _utc_now()
    _write_json_atomic(res_path, result)
    print(f"[{i}] {task['dataset_name']} noise={task['target_noise']:g} "
          f"gt={result.get('gt_match_score')} r2={result.get('r2_score')} "
          f"best={result.get('best_equation')} err={result['error']}", flush=True)
    return result


def _score(args, output_dir: Path, indices: list[int] | None) -> None:
    batch_dir = output_dir / BATCH
    tasks = json.loads((batch_dir / "tasks.json").read_text())
    for i in (indices if indices else range(len(tasks))):
        _score_task(batch_dir, tasks[i], args.force)


def _shard_indices(output_dir: Path, args) -> list[int]:
    if args.task_indices:
        return args.task_indices
    tasks = json.loads((output_dir / BATCH / "tasks.json").read_text())
    shard = args.shard
    if args.shard_env:
        shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
    return [i for i in range(len(tasks)) if i % args.n_shards == shard]


# ------------------------------------------------------------------- aggregate
def _aggregate(args) -> None:
    output_dir = Path(args.output_dir).resolve()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    sys.path.insert(0, str(PROJECT_ROOT))
    import srbench_results_io as srio
    keyed = srio.build_keyed_results(output_dir, manifest)
    srio.save_keyed_results(output_dir, keyed, meta={
        "mode": "modsr", "n_datasets": manifest["n_datasets"], "n_runs": 1,
        "noise_levels": manifest["noise_levels"],
        "modsr_config": manifest["modsr_config"],
    })
    metrics = srio.aggregate_metrics(keyed, manifest["noise_levels"])
    paths = sorted((output_dir / BATCH / "results").glob("task_*.json"))
    errored = sum(json.loads(p.read_text()).get("error") is not None for p in paths)
    n_ok = sum(1 for r in keyed.values() if r["present"] and r["error"] is None)
    summary = {"n_expected": len(keyed), "n_result_files": len(paths),
               "n_successful": n_ok, "n_errored": errored,
               "n_missing": len(keyed) - len(paths), "metrics": metrics}
    _write_json_atomic(output_dir / "summary.json", summary)
    print(f"Results: {n_ok}/{len(keyed)} successful, {errored} errored, "
          f"{len(keyed) - len(paths)} missing")
    print(srio.format_metrics_console(metrics, manifest["noise_levels"]))


# ---------------------------------------------------------------------- submit
def _sbatch_commands(args, output_dir: Path, array_job_id: str | None = None):
    script = str(Path(__file__).resolve())
    run_sh = str(PROJECT_ROOT / "run.sh")
    array = [
        "sbatch", "--parsable", "--export=ALL",
        f"--array=0-{args.n_shards - 1}",
        f"--partition={args.partition}", f"--time={args.time_limit}",
        f"--cpus-per-task={args.cpus_per_task}", f"--mem={args.mem}",
        f"--gres=gpu:{args.gpu_type + ':' if args.gpu_type else ''}1",
        "--job-name=modsr-srb-gt",
        f"--output={output_dir}/slurm/%A_%a.out",
        f"--error={output_dir}/slurm/%A_%a.err",
        run_sh, script, "shard", "--output-dir", str(output_dir),
        "--n-shards", str(args.n_shards), "--shard-env",
    ]
    aggregate = [
        "sbatch", "--parsable",
        f"--dependency=afterany:{array_job_id or '<array-job-id>'}",
        f"--partition={args.partition}", "--time=00:20:00",
        "--cpus-per-task=1", "--mem=4G", "--job-name=modsr-srb-final",
        f"--output={output_dir}/slurm/final_%j.out",
        f"--error={output_dir}/slurm/final_%j.err",
        run_sh, script, "aggregate", "--output-dir", str(output_dir),
    ]
    return array, aggregate


def _submit(args) -> None:
    output_dir = _prepare(args)
    array_cmd, _ = _sbatch_commands(args, output_dir)
    print(" ".join(array_cmd))
    if args.dry_run:
        return
    job = subprocess.check_output(array_cmd, text=True).strip().split(";")[0]
    _, agg_cmd = _sbatch_commands(args, output_dir, array_job_id=job)
    agg = subprocess.check_output(agg_cmd, text=True).strip().split(";")[0]
    print(f"Submitted MOD-SR array {job} and aggregator {agg}")


# ------------------------------------------------------------------------ main
def _add_run_args(p) -> None:
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split-file", default="splits/srbench_all.txt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--n-encoder-points", type=int, default=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_args(sub.add_parser("prepare"))
    s = sub.add_parser("submit")
    _add_run_args(s)
    s.add_argument("--n-shards", type=int, default=4)
    s.add_argument("--partition", default="default_partition")
    s.add_argument("--gpu-type", default="")
    s.add_argument("--time-limit", default="12:00:00")
    s.add_argument("--cpus-per-task", type=int, default=4)
    s.add_argument("--mem", default="32G")
    s.add_argument("--dry-run", action="store_true")
    for name in ("predict", "score", "shard"):
        q = sub.add_parser(name)
        q.add_argument("--output-dir", required=True)
        q.add_argument("--task-indices", type=int, nargs="*")
        q.add_argument("--shard", type=int, default=0)
        q.add_argument("--n-shards", type=int, default=1)
        q.add_argument("--shard-env", action="store_true")
        q.add_argument("--cpu", action="store_true")
        q.add_argument("--force", action="store_true")
    a = sub.add_parser("aggregate")
    a.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        _prepare(args)
    elif args.command == "submit":
        _submit(args)
    elif args.command == "aggregate":
        _aggregate(args)
    else:
        output_dir = Path(args.output_dir).resolve()
        indices = _shard_indices(output_dir, args)
        if args.command in ("predict", "shard"):
            _predict(args, output_dir, indices)
        if args.command in ("score", "shard"):
            _score(args, output_dir, indices)


if __name__ == "__main__":
    main()
