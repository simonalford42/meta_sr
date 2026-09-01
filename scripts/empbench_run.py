#!/usr/bin/env python3
"""Run ONE PySR search on ONE EmpiricalBench problem, locally, single-core.

Reuses the production scoring path (check_pysr_frontier_symbolic_match with the
R²>0.5 gate, exactly as parallel_eval_pysr._evaluate_pysr_task) AND adds a robust
numeric recovery check (scripts/empbench_lib) so we can report both the official
(buggy-for-Planck) metric and the true recovery.

Warm-start milestones give evals-to-solve: at each cumulative-eval milestone we
score the live Pareto front. Designed to be launched many-at-once (each process
single-core, JULIA_NUM_THREADS=1) to use all cores without SLURM.

Examples:
    python scripts/empbench_run.py --method baseline --dataset empirical_rydberg \
        --run-index 0 --max-evals 10000000 --out runs_local/x.json
    python scripts/empbench_run.py --method evolve --evolve-results runs/666285 \
        --dataset empirical_planck --run-index 0 --max-evals 10000000 --out ...
    python scripts/empbench_run.py --method custom --binary-ops '+,-,*,/' \
        --unary-ops 'log,square,exp' --dataset empirical_rydberg --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from empbench_lib import numeric_recovery, _gt_xform, clean_grid  # noqa: E402


def build_pysr_kwargs_and_code(args):
    """Return (pysr_kwargs, custom_code_dict, method_meta)."""
    from parallel_eval_pysr import (get_default_pysr_kwargs,
                                    get_default_mutation_weights)

    base = get_default_pysr_kwargs()
    base["max_evals"] = int(args.max_evals)
    # We drive milestones ourselves; keep niterations effectively unbounded.
    base["niterations"] = 10_000_000
    base["progress"] = False
    base["verbosity"] = 0
    base["deterministic"] = True
    base["procs"] = 0
    base["parallelism"] = "serial"

    custom_code = {"custom_mutation_code": None, "allow_custom_mutations": False,
                   "custom_survival_code": None, "custom_selection_code": None,
                   "custom_loss_code": None}
    method_meta = {"method": args.method}
    mutation_weights = get_default_mutation_weights()

    if args.method == "evolve":
        from bundle_loader import load_bundle
        bundle = load_bundle(args.evolve_results)
        cfg = bundle.to_pysr_config(base)
        allow = bool(cfg.allow_custom_mutations)
        custom_code = {
            "custom_mutation_code": cfg.custom_mutation_code,
            "allow_custom_mutations": allow,
            "custom_survival_code": cfg.custom_survival_code,
            "custom_selection_code": cfg.custom_selection_code,
            "custom_loss_code": cfg.custom_loss_code,
        }
        mutation_weights = dict(cfg.mutation_weights)
        base = dict(cfg.pysr_kwargs)
        base["max_evals"] = int(args.max_evals)
        base["progress"] = False
        base["verbosity"] = 0
        base["deterministic"] = True
        base["procs"] = 0
        base["parallelism"] = "serial"
        method_meta["operators"] = {t: op.name for t, op in bundle.operators.items()
                                    if op is not None}
        method_meta["evolve_results"] = args.evolve_results

    # custom / baseline op-set + hparam overrides
    if args.binary_ops:
        base["binary_operators"] = [s for s in args.binary_ops.split(",") if s]
    if args.unary_ops is not None:
        base["unary_operators"] = [s for s in args.unary_ops.split(",") if s]
    if args.maxsize is not None:
        base["maxsize"] = int(args.maxsize)
    if args.populations is not None:
        base["populations"] = int(args.populations)
    if args.population_size is not None:
        base["population_size"] = int(args.population_size)
    if args.extra_kwargs:
        extra = json.loads(args.extra_kwargs)
        base.update(extra)
    # Drop constraints/nested_constraints that reference operators we removed,
    # so PySR doesn't error on an unknown-operator constraint key.
    ops = set(base.get("binary_operators", [])) | set(base.get("unary_operators", []))
    for ckey in ("constraints", "nested_constraints"):
        if isinstance(base.get(ckey), dict):
            base[ckey] = {k: v for k, v in base[ckey].items()
                          if k in ops or k in ("/",) and "/" in ops}
            # for nested_constraints, also prune inner keys
            if ckey == "nested_constraints":
                base[ckey] = {k: {ik: iv for ik, iv in v.items() if ik in ops}
                              for k, v in base[ckey].items()}

    # Parallelism / wall-clock overrides (PySR-paper protocol: multi-core,
    # time-boxed). Keep max_evals active as a second stopping condition: when
    # both limits are supplied, PySR stops at whichever is reached first.
    if args.parallelism is not None or args.procs is not None or args.timeout_seconds is not None:
        par = args.parallelism or "serial"
        base["parallelism"] = par
        # procs only applies to multiprocessing; multithreading uses JULIA_NUM_THREADS
        base["procs"] = (int(args.procs) if args.procs is not None
                         else (0 if par != "multiprocessing" else 8))
        # PySR forbids deterministic with non-serial parallelism
        base["deterministic"] = (par == "serial")
        if args.timeout_seconds is not None:
            base["timeout_in_seconds"] = float(args.timeout_seconds)
            base["niterations"] = 10_000_000  # let the wall clock govern

    method_meta["binary_operators"] = base.get("binary_operators")
    method_meta["unary_operators"] = base.get("unary_operators")
    method_meta["parallelism"] = base.get("parallelism")
    method_meta["timeout_in_seconds"] = base.get("timeout_in_seconds")

    # Merge mutation weights into kwargs (PySR takes weight_* as kwargs)
    mut_kwargs = {}
    for k, v in mutation_weights.items():
        if not k.startswith("weight_"):
            k = f"weight_{k}"
        if "custom_mutation" in k and not custom_code["allow_custom_mutations"]:
            continue
        mut_kwargs[k] = v
    pysr_kwargs = {**mut_kwargs, **base}
    # NOTE: `allow_custom_mutations` is NOT a PySRRegressor kwarg. Custom
    # mutations are activated purely by (a) _load_dynamic_mutations(code) and
    # (b) passing weight_custom_mutation_1 (done above when allow=True). Never
    # forward allow_custom_mutations / the bundle's non-PySR fields to the model.
    for bad in ("allow_custom_mutations",):
        pysr_kwargs.pop(bad, None)
    return pysr_kwargs, custom_code, method_meta


def load_dynamic(custom_code):
    from parallel_eval_pysr import (_load_dynamic_loss, _load_dynamic_mutations,
                                    _load_dynamic_selection, _load_dynamic_survival)
    if custom_code.get("custom_mutation_code"):
        _load_dynamic_mutations(custom_code["custom_mutation_code"])
    if custom_code.get("custom_selection_code"):
        _load_dynamic_selection(custom_code["custom_selection_code"])
    if custom_code.get("custom_survival_code"):
        _load_dynamic_survival(custom_code["custom_survival_code"])
    if custom_code.get("custom_loss_code"):
        _load_dynamic_loss(custom_code["custom_loss_code"])
    else:
        try:
            from juliacall import Main as jl
            jl.seval("using SymbolicRegression.CustomLossModule")
            jl.seval("clear_dynamic_losses!()")
        except Exception:
            pass


def milestone_schedule(max_evals, n):
    """Geometric-ish increasing cumulative-eval milestones up to max_evals."""
    lo = max(2000, int(max_evals / 1000))
    ms = np.unique(np.round(np.geomspace(lo, max_evals, n)).astype(int)).tolist()
    if ms[-1] != int(max_evals):
        ms[-1] = int(max_evals)
    return ms


def score_frontier(model, ds, X_val, y_val, gt_x, xvars, min_r2=0.5):
    """Return (official_match, robust_match, best_val_r2, matched_official,
    matched_robust, frontier rows)."""
    from evaluation import check_pysr_frontier_symbolic_match
    eqs = model.equations_
    rows = []
    best_r2 = -np.inf
    if eqs is None or len(eqs) == 0:
        return dict(official=False, robust=False, best_r2=None,
                    matched_official=None, matched_robust=None, frontier=[])

    ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
    robust_match = False
    matched_robust = None
    for idx in eqs.index:
        expr = str(eqs.loc[idx]["equation"])
        try:
            yp = np.clip(np.asarray(model.predict(X_val, index=int(idx))), -1e10, 1e10)
            ss_res = float(np.sum((y_val - yp) ** 2))
            r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        except Exception:
            r2 = float("nan")
        if np.isfinite(r2):
            best_r2 = max(best_r2, r2)
        rec = numeric_recovery(expr, ds)
        rows.append({"complexity": int(eqs.loc[idx].get("complexity", -1)),
                     "loss": float(eqs.loc[idx].get("loss", float("nan"))),
                     "val_r2": float(r2) if np.isfinite(r2) else None,
                     "equation": expr,
                     "robust_match": bool(rec["match"]),
                     "robust_kind": rec["kind"]})
        # Robust recovery does NOT require a raw val-R² gate: it is "equal to GT
        # up to an additive/multiplicative constant" verified on a clean grid,
        # which is self-validating. Gating on raw val R² would wrongly reject a
        # shape-correct expression that the evolved scale-calibrating loss leaves
        # offset/scaled (poor raw R², but a genuine recovery up to a constant).
        if rec["match"] and not robust_match:
            robust_match = True
            matched_robust = expr

    best = model.get_best()
    best_idx = best.name if best is not None else None
    off = check_pysr_frontier_symbolic_match(
        equations_df=eqs, best_df_index=best_idx, ground_truth_str=gt_x,
        var_names=xvars, timeout_seconds_per_expression=3,
        predict_fn=lambda i: model.predict(X_val, index=int(i)),
        y=y_val, min_r2=min_r2)
    matched_official = None
    if off.get("match") and off.get("matched_df_index") is not None:
        try:
            matched_official = str(eqs.loc[off["matched_df_index"]]["equation"])
        except Exception:
            pass
    return dict(official=bool(off.get("match")), robust=robust_match,
                best_r2=(float(best_r2) if np.isfinite(best_r2) else None),
                matched_official=matched_official, matched_robust=matched_robust,
                frontier=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["baseline", "evolve", "custom"], required=True)
    ap.add_argument("--evolve-results", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-index", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="fraction of data used for training; 1.0 = PySR-paper "
                         "protocol (train on the full dataset, val=train)")
    ap.add_argument("--timeout-seconds", type=float, default=None,
                    help="Run a single fit with this PySR wall-clock limit and "
                         "--max-evals both active, then score the final Pareto "
                         "front once (whichever budget is reached first).")
    ap.add_argument("--parallelism", default=None,
                    choices=[None, "serial", "multithreading", "multiprocessing"],
                    help="PySR parallelism. Use 'multithreading' for multi-core "
                         "with custom (evolved) operators — threads share the one "
                         "Julia process so the dynamically-loaded operators apply; "
                         "multiprocessing workers would NOT see them.")
    ap.add_argument("--procs", type=int, default=None,
                    help="worker count for multiprocessing (ignored for "
                         "multithreading, which uses JULIA_NUM_THREADS)")
    ap.add_argument("--max-evals", type=float, default=1e7)
    ap.add_argument("--n-milestones", type=int, default=12)
    ap.add_argument("--binary-ops", default=None)
    ap.add_argument("--unary-ops", default=None)
    ap.add_argument("--maxsize", type=int, default=None)
    ap.add_argument("--populations", type=int, default=None)
    ap.add_argument("--population-size", type=int, default=None)
    ap.add_argument("--extra-kwargs", default=None, help="JSON dict merged into pysr_kwargs")
    ap.add_argument("--wall-limit", type=float, default=14400, help="seconds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.time()
    run_seed = args.seed_base + args.run_index

    from parallel_eval_pysr import _import_pysr_regressor, _get_pysr_num_evaluations
    from utils import load_srbench_dataset

    pysr_kwargs, custom_code, method_meta = build_pysr_kwargs_and_code(args)

    # data + split (mirror _evaluate_pysr_task)
    np.random.seed(args.data_seed)
    X, y, _ = load_srbench_dataset(args.dataset, max_samples=1000)
    np.random.seed(run_seed)
    n = len(y)
    idx = np.random.permutation(n)
    if args.train_frac >= 1.0:
        # PySR-paper protocol: feed the ENTIRE dataset for training (the test is
        # the output expression, scored on a clean grid by the robust check, not
        # a held-out split). val = train so the in-sample R² gate still works.
        tr = va = idx
        ntr = n
    else:
        ntr = int(args.train_frac * n)
        tr, va = idx[:ntr], idx[ntr:]
    X_train, y_train = X[tr], y[tr]
    X_val, y_val = X[va], y[va]
    n_features = X_train.shape[1]
    xvars = [f"x{i}" for i in range(n_features)]
    gt_x, _ = _gt_xform(args.dataset)

    PySRRegressor = _import_pysr_regressor()
    load_dynamic(custom_code)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = out_dir / f"_pysrtmp_{args.dataset}_{args.method}_{args.run_index}_{os.getpid()}"

    model_kwargs = dict(pysr_kwargs)
    model_kwargs["random_state"] = run_seed
    model_kwargs["output_directory"] = str(tmp_out)
    model_kwargs["warm_start"] = True
    model_kwargs["temp_equation_file"] = False
    model_kwargs.pop("delete_tempfiles", None)
    model = PySRRegressor(**model_kwargs)

    milestones = milestone_schedule(int(args.max_evals), args.n_milestones)

    result = {
        "dataset": args.dataset, "method": args.method, "run_index": args.run_index,
        "seed": run_seed, "max_evals": int(args.max_evals),
        "method_meta": method_meta, "milestones": [],
        "official_solved_at": None, "robust_solved_at": None,
        "n_train": int(ntr), "n_val": int(n - ntr), "gt": gt_x,
        "pysr_kwargs": {k: v for k, v in pysr_kwargs.items()
                        if k in ("binary_operators", "unary_operators", "maxsize",
                                 "populations", "population_size", "max_evals",
                                 "timeout_in_seconds")},
    }

    import signal

    def _alarm(signum, frame):
        raise TimeoutError(f"wall limit {args.wall_limit}s")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(args.wall_limit))

    result["completed"] = False

    def _dump():
        # Atomic checkpoint so a later SIGSEGV/OOM/timeout can't corrupt or lose
        # the JSON: write to a temp file then rename.
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp, args.out)

    try:
        if args.timeout_seconds is not None:
            # WALL-CLOCK mode: one time-boxed fit, score the final front once.
            t0 = time.time()
            model.fit(X_train, y_train, variable_names=xvars)
            chunk_t = time.time() - t0
            try:
                from juliacall import Main as jl
                nthreads = int(jl.seval("Threads.nthreads()"))
            except Exception:
                nthreads = None
            sc = score_frontier(model, args.dataset, X_val, y_val, gt_x, xvars)
            rec = {"milestone": "timeout", "wallclock_s": round(chunk_t, 1),
                   "nthreads": nthreads,
                   "parallelism": method_meta.get("parallelism"),
                   "timeout_in_seconds": args.timeout_seconds,
                   "official": sc["official"], "robust": sc["robust"],
                   "best_val_r2": sc["best_r2"],
                   "matched_official": sc["matched_official"],
                   "matched_robust": sc["matched_robust"],
                   "frontier": [{"complexity": row["complexity"],
                                 "val_r2": row["val_r2"],
                                 "equation": row["equation"]}
                                for row in sc["frontier"]]}
            result["milestones"].append(rec)
            result["final_frontier"] = sc["frontier"]
            result["wallclock_s"] = round(chunk_t, 1)
            result["nthreads"] = nthreads
            if sc["official"]:
                result["official_solved_at"] = "timeout"
            if sc["robust"]:
                result["robust_solved_at"] = "timeout"
            print(f"[{args.dataset}/{args.method}/r{args.run_index}] "
                  f"WALLCLOCK t={chunk_t:.0f}s threads={nthreads} "
                  f"off={sc['official']} rob={sc['robust']} "
                  f"bestR2={sc['best_r2']} robmatch={sc['matched_robust']}", flush=True)
            _dump()
            result["completed"] = True
            raise StopIteration  # skip the milestone loop below
        for ms in milestones:
            t0 = time.time()
            model.max_evals = int(ms)
            model.fit(X_train, y_train, variable_names=xvars)
            chunk_t = time.time() - t0
            num_evals = _get_pysr_num_evaluations(model)
            sc = score_frontier(model, args.dataset, X_val, y_val, gt_x, xvars)
            rec = {"milestone": int(ms), "num_evals": num_evals,
                   "chunk_time_s": round(chunk_t, 2),
                   "official": sc["official"], "robust": sc["robust"],
                   "best_val_r2": sc["best_r2"],
                   "matched_official": sc["matched_official"],
                   "matched_robust": sc["matched_robust"],
                   # Full frontier (trimmed) at THIS milestone, so verified
                   # recovery / evals-to-solve is fully derivable offline under
                   # any later scoring refinement — no re-run needed.
                   "frontier": [{"complexity": row["complexity"],
                                 "val_r2": row["val_r2"],
                                 "equation": row["equation"]}
                                for row in sc["frontier"]]}
            result["milestones"].append(rec)
            ne = num_evals if num_evals is not None else ms
            if sc["official"] and result["official_solved_at"] is None:
                result["official_solved_at"] = ne
            if sc["robust"] and result["robust_solved_at"] is None:
                result["robust_solved_at"] = ne
            print(f"[{args.dataset}/{args.method}/r{args.run_index}] ms={ms:>9} "
                  f"nevals={num_evals} t={chunk_t:5.1f}s off={sc['official']} "
                  f"rob={sc['robust']} bestR2={sc['best_r2']} "
                  f"robmatch={sc['matched_robust']}", flush=True)
            # Always keep the LATEST frontier snapshot + checkpoint to disk, so a
            # crash mid-run still leaves the most recent frontier + milestones.
            result["final_frontier"] = sc["frontier"]
            _dump()
        result["completed"] = True
    except StopIteration:
        pass  # wall-clock branch finished; skip the milestone loop
    except (TimeoutError, KeyboardInterrupt) as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"[{args.dataset}/{args.method}/r{args.run_index}] STOPPED: {e}", flush=True)
    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        signal.alarm(0)
        try:
            import shutil
            if tmp_out.exists():
                shutil.rmtree(tmp_out)
        except Exception:
            pass

    result["total_time_s"] = round(time.time() - t_start, 1)
    _dump()
    print(f"[{args.dataset}/{args.method}/r{args.run_index}] DONE in "
          f"{result['total_time_s']}s -> {args.out} "
          f"(official_solved_at={result['official_solved_at']}, "
          f"robust_solved_at={result['robust_solved_at']})", flush=True)


if __name__ == "__main__":
    main()
