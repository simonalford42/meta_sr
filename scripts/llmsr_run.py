#!/usr/bin/env python3
"""Run ONE PySR fit (baseline or evolved bundle) on ONE LLM-SR benchmark problem.

LLM-SR (Shojaee et al., ICLR 2025, arXiv:2404.18400) custom problems:
oscillator1, oscillator2, bactgrow, stressstrain. Data from
github.com/deep-symbolic-mathematics/LLM-SR (cloned to baselines/llm-sr-src/LLM-SR).

Protocol (paper App. A.1): PySR with populations=15, population_size=33, other
settings default; trained on train.csv; reported metric is NMSE = MSE/Var(y)
on test_id.csv and test_ood.csv. We time-box each fit (single core) instead of
"2M iterations".

Every frontier equation is scored on train / ID / OOD, both raw and with a
train-fit affine rescaling a + b*f (Keijzer linear scaling; needed for the
evolved bundle's affine-profile loss, which only fits f up to scale/offset).

    python scripts/llmsr_run.py --method baseline --problem oscillator1 \
        --run-index 0 --timeout-seconds 1800 --out runs/x/osc1_baseline_r0.json
    python scripts/llmsr_run.py --method evolve --evolve-results runs/709715 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JULIA_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import pandas as pd

DATA_DIR = REPO / "baselines" / "llm-sr-src" / "LLM-SR" / "data"
PROBLEMS = ("oscillator1", "oscillator2", "bactgrow", "stressstrain")


def load_split(problem, split):
    df = pd.read_csv(DATA_DIR / problem / f"{split}.csv")
    cols = list(df.columns)
    return df[cols[:-1]].to_numpy(float), df[cols[-1]].to_numpy(float), cols[:-1]


def nmse(yp, y):
    yp = np.asarray(yp, float).reshape(-1)
    if yp.shape != y.shape or not np.all(np.isfinite(yp)):
        return float("inf")
    return float(np.mean((yp - y) ** 2) / np.var(y))


def affine_fit(f, y):
    fm, ym = float(np.mean(f)), float(np.mean(y))
    df = f - fm
    sxx = float(np.dot(df, df))
    b = float(np.dot(df, y - ym)) / sxx if sxx > 0 else 0.0
    if not np.isfinite(b):
        b = 0.0
    return ym - b * fm, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["baseline", "evolve"], required=True)
    ap.add_argument("--evolve-results", default=None)
    ap.add_argument("--problem", choices=PROBLEMS, required=True)
    ap.add_argument("--run-index", type=int, default=0)
    ap.add_argument("--seed-base", type=int, default=10000)
    ap.add_argument("--timeout-seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from empbench_run import build_pysr_kwargs_and_code, load_dynamic
    from parallel_eval_pysr import _import_pysr_regressor, _get_pysr_num_evaluations

    # Reuse the EmpiricalBench builder (bundle -> kwargs + custom Julia code).
    bargs = argparse.Namespace(
        method=args.method, evolve_results=args.evolve_results,
        max_evals=10**12, binary_ops=None, unary_ops=None, maxsize=None,
        populations=None, population_size=None, extra_kwargs=None,
        parallelism="serial", procs=0, timeout_seconds=args.timeout_seconds,
    )
    pysr_kwargs, custom_code, method_meta = build_pysr_kwargs_and_code(bargs)
    pysr_kwargs.pop("max_evals", None)  # wall-clock only
    # Paper protocol: populations=15, population_size=33 (also our defaults).
    pysr_kwargs["populations"] = 15
    pysr_kwargs["population_size"] = 33

    X_tr, y_tr, names = load_split(args.problem, "train")
    X_id, y_id, _ = load_split(args.problem, "test_id")
    X_ood, y_ood, _ = load_split(args.problem, "test_ood")
    xvars = [f"x{i}" for i in range(X_tr.shape[1])]

    seed = args.seed_base + args.run_index
    PySRRegressor = _import_pysr_regressor()
    load_dynamic(custom_code)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.parent / f"_pysrtmp_{out.stem}_{os.getpid()}"
    model_kwargs = dict(pysr_kwargs)
    model_kwargs.update(random_state=seed, output_directory=str(tmp_out),
                        temp_equation_file=False, verbosity=0, progress=False)
    model_kwargs.pop("delete_tempfiles", None)
    model = PySRRegressor(**model_kwargs)

    t0 = time.time()
    model.fit(X_tr, y_tr, variable_names=xvars)
    wall = time.time() - t0

    eqs = model.equations_
    best = model.get_best()
    best_idx = int(best.name) if best is not None else None
    frontier = []
    for idx in eqs.index:
        row = {"complexity": int(eqs.loc[idx, "complexity"]),
               "loss": float(eqs.loc[idx, "loss"]),
               "equation": str(eqs.loc[idx, "equation"]),
               "pysr_best": int(idx) == best_idx}
        try:
            p_tr = np.asarray(model.predict(X_tr, index=int(idx)), float).reshape(-1)
            p_id = np.asarray(model.predict(X_id, index=int(idx)), float).reshape(-1)
            p_ood = np.asarray(model.predict(X_ood, index=int(idx)), float).reshape(-1)
        except Exception as e:  # noqa: BLE001
            row["error"] = repr(e)
            frontier.append(row)
            continue
        a, b = affine_fit(p_tr, y_tr) if np.all(np.isfinite(p_tr)) else (0.0, 1.0)
        row.update(
            nmse_train=nmse(p_tr, y_tr), nmse_id=nmse(p_id, y_id), nmse_ood=nmse(p_ood, y_ood),
            affine_a=a, affine_b=b,
            aff_nmse_train=nmse(a + b * p_tr, y_tr), aff_nmse_id=nmse(a + b * p_id, y_id),
            aff_nmse_ood=nmse(a + b * p_ood, y_ood),
        )
        frontier.append(row)

    result = {
        "problem": args.problem, "variables": names, "method": args.method,
        "run_index": args.run_index, "seed": seed, "method_meta": method_meta,
        "timeout_seconds": args.timeout_seconds, "wallclock_s": round(wall, 1),
        "num_evals": _get_pysr_num_evaluations(model),
        "pysr_kwargs": {k: v for k, v in pysr_kwargs.items()
                        if k in ("binary_operators", "unary_operators", "maxsize", "maxdepth",
                                 "populations", "population_size", "elementwise_loss",
                                 "early_stop_condition")},
        "frontier": frontier, "completed": True,
    }
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1, default=str)
    os.replace(tmp, out)
    sel = min((r for r in frontier if "nmse_train" in r), key=lambda r: r["nmse_train"], default=None)
    print(f"[{args.problem}/{args.method}/r{args.run_index}] t={wall:.0f}s "
          f"evals={result['num_evals']} best-train: {sel and sel['equation']} "
          f"ID={sel and sel['nmse_id']:.3g} OOD={sel and sel['nmse_ood']:.3g}", flush=True)


if __name__ == "__main__":
    main()
