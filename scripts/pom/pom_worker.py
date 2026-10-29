#!/usr/bin/env python3
"""Run a list of PySR jobs (one process, Julia loaded once) for the
pattern-oriented-modelling (POM) loss experiment.

Each job (one JSON line in --jobs):
  {"id": str, "dataset": str, "seed": int, "timeout": int,
   "loss": {"type": "mse"} | {"type": "stat", "families": [...], "alpha": float},
   "early_stop": bool}

All arms share the same data draw (data_seed=0, 80/20 split seed 0), so the
training statistics are identical across arms. Output per job:
  <out>/<id>.json  — frontier (equation, complexity, loss, train/val R²), GT match
  <out>/<id>.npz   — X_train, y_train, frontier predictions on X_train
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("JULIA_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

POM_JL = Path(__file__).resolve().parent / "PomStats.jl"
N_SAMPLES = 1000

POM_LOSS = """
function pom_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    pred, ok = eval_tree_array(tree, dataset.X, options)
    !ok && return L(Inf)
    v = Main.PomStats.objective(pred, dataset.y)
    return isfinite(v) ? L(v) : L(Inf)
end
"""


def load_split(dataset):
    from utils import load_srbench_dataset
    from parallel_eval_pysr import _remap_formula_variables
    from evaluation import get_dataset_var_names

    X, y, gt = load_srbench_dataset(dataset, max_samples=N_SAMPLES, data_seed=0)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))
    n_train = int(0.8 * len(y))
    tr, va = idx[:n_train], idx[n_train:]
    var_names = [f"x{i}" for i in range(X.shape[1])]
    try:
        dvn = get_dataset_var_names(dataset)
        if len(dvn) == X.shape[1]:
            gt = _remap_formula_variables(gt, dvn, var_names)
    except Exception:
        pass
    return X[tr], y[tr], X[va], y[va], gt, var_names


def r2(y, p):
    p = np.asarray(p, dtype=float)
    if p.shape != y.shape or not np.all(np.isfinite(p)):
        return None
    return float(1 - np.sum((y - p) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-30))


def run_job(job, out_dir, PySRRegressor, jl):
    from parallel_eval_pysr import get_default_pysr_kwargs
    from domains import get_domain

    Xtr, ytr, Xva, yva, gt, var_names = load_split(job["dataset"])
    kw = get_default_pysr_kwargs()
    kw.pop("niterations", None)
    kw.update(
        niterations=10_000_000,
        timeout_in_seconds=int(job["timeout"]),
        random_state=int(job["seed"]),
        verbosity=0,
        output_directory=str(out_dir / "pysr_tmp"),
    )
    if not job.get("early_stop", True):
        kw["early_stop_condition"] = None
    loss = job["loss"]
    if loss["type"] == "stat":
        jl.seval("PomStats.configure!")(Xtr, ytr)
        jl.seval("PomStats.configure_loss!")(loss["families"], ytr, float(loss["alpha"]))
        kw["loss_function"] = POM_LOSS
        kw["early_stop_condition"] = None
    model = PySRRegressor(**kw)
    t0 = time.time()
    model.fit(Xtr, ytr, variable_names=var_names)
    fit_s = time.time() - t0

    eq = model.equations_
    best = model.get_best()
    frontier, preds = [], []
    for idx, row in eq.iterrows():
        try:
            ptr = np.asarray(model.predict(Xtr, index=int(idx)), dtype=float)
            pva = np.asarray(model.predict(Xva, index=int(idx)), dtype=float)
        except Exception:
            ptr = np.full_like(ytr, np.nan)
            pva = np.full_like(yva, np.nan)
        preds.append(ptr)
        frontier.append({
            "index": int(idx), "complexity": int(row["complexity"]),
            "equation": str(row["equation"]), "loss": float(row["loss"]),
            "r2_train": r2(ytr, ptr), "r2_val": r2(yva, pva),
        })
    domain = get_domain("srbench")
    try:
        m = domain.check_solved(
            equations_df=eq, best_df_index=best.name, target=gt,
            var_names=var_names,
            predict_fn=lambda i: model.predict(Xva, index=int(i)),
            y_val=yva, predict_on=None, dataset_name=job["dataset"])
        solved = bool(m.get("match", False))
        matched = m.get("matched_df_index")
    except Exception as exc:
        solved, matched = False, None
        print(f"GT check failed: {exc}", flush=True)
    res = dict(job, gt=gt, solved=solved,
               matched_index=None if matched is None else int(matched),
               best_index=int(best.name), fit_seconds=fit_s,
               num_evals=_num_evals(model), frontier=frontier)
    np.savez_compressed(out_dir / f"{job['id']}.npz", X_train=Xtr, y_train=ytr,
                        preds=np.array(preds), index=np.array([f["index"] for f in frontier]))
    tmp = out_dir / f"{job['id']}.json.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    tmp.rename(out_dir / f"{job['id']}.json")
    return solved


def _num_evals(model):
    try:
        from parallel_eval_pysr import _get_pysr_num_evaluations
        return _get_pysr_num_evaluations(model)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(l) for l in open(args.jobs) if l.strip()]
    jobs = [j for k, j in enumerate(jobs) if k % args.nworkers == args.worker]

    from parallel_eval_pysr import _import_pysr_regressor
    PySRRegressor = _import_pysr_regressor()
    from juliacall import Main as jl
    jl.seval("using SymbolicRegression")
    jl.seval(f'include("{POM_JL}")')

    for job in jobs:
        if (out_dir / f"{job['id']}.json").exists():
            continue
        t0 = time.time()
        try:
            s = run_job(job, out_dir, PySRRegressor, jl)
            print(f"[w{args.worker}] {job['id']} solved={s} {time.time()-t0:.0f}s", flush=True)
        except Exception:
            print(f"[w{args.worker}] {job['id']} FAILED\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
