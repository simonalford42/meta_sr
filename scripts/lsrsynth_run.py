#!/usr/bin/env python3
"""PySR (baseline or evolved bundle) on LLM-SRBench LSR-Synth.

LSR-Synth (Shojaee et al., LLM-SRBench, ICML 2025, arXiv:2504.10415): 129
synthetic problems in 4 domains (chem_react 36, bio_pop_growth 24, phys_osc 44,
matsci 25), each with train / ID test / OOD test splits. Data must first be
converted with scripts/lsrsynth_prepare_data.py.

Metrics follow the official pipeline (baselines/llm-srbench-src/bench/
pipelines.py): NMSE = mean((y-ŷ)^2)/var(y) over non-NaN predictions, plus
Acc_tau = 1[max_i |ŷ_i-y_i|/|y_i| <= tau] for tau in {0.1, 0.01, 0.001}
(LLM-SRBench / PiT-PO), and the 5%-trimmed acc95_tau used by SR-Scientist /
RESTART. See reports/lsr_synth_literature.md for which paper uses which.
Every frontier equation is scored on train/ID/OOD, raw and with a train-fit
affine rescaling a+b*f (the evolved bundle's loss is affine-invariant);
selection happens in scripts/lsrsynth_summarize.py.

Subcommands
    run        one fit:     --method baseline|evolve --domain D --name N --run-index K --out F
    make-jobs  jobs.json for (problem x method x seed); --output-dir runs/<name>
    worker     run jobs[--index or $SLURM_ARRAY_TASK_ID] (resume-safe)

Submit jobs.json as an array with
    python scripts/empbench_slurm_submit.py --worker scripts/lsrsynth_run.py --jobs runs/<name>/jobs.json ...
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

DATA_DIR = Path(os.environ.get("LSRSYNTH_DATA_DIR", REPO / "data" / "llm-srbench" / "lsr_synth"))
DOMAINS = ("chem_react", "bio_pop_growth", "phys_osc", "matsci")
TAUS = (0.1, 0.01, 0.001)
SPLITS = ("train", "id_test", "ood_test")


def load_manifest():
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        sys.exit(f"{path} missing: run scripts/lsrsynth_prepare_data.py first")
    return json.loads(path.read_text())


def load_problem(domain, name):
    d = np.load(DATA_DIR / domain / f"{name}.npz")
    # column 0 is the target, the rest are inputs (official convention)
    return {k: (d[k][:, 1:], d[k][:, 0]) for k in SPLITS if k in d}


def metrics(yp, y):
    """Official LLM-SRBench NMSE (NaN preds dropped) + Acc_tau + R2 + validity."""
    yp = np.asarray(yp, float).reshape(-1)
    if yp.shape != y.shape:
        return {"nmse": float("inf"), "r2": float("-inf"), "n_valid": 0,
                **{f"acc{v}_{t}": False for t in TAUS for v in ("", "95")}}
    ok = ~np.isnan(yp)
    out = {"n_valid": int(ok.sum()), "frac_valid": float(ok.mean())}
    if ok.sum() == 0:
        out.update(nmse=float("inf"), r2=float("-inf"),
                   **{f"acc{v}_{t}": False for t in TAUS for v in ("", "95")})
        return out
    err = yp[ok] - y[ok]
    with np.errstate(all="ignore"):
        mse = float(np.mean(err ** 2))
        out["nmse"] = mse / float(np.var(y[ok]))
        out["r2"] = 1.0 - mse / float(np.var(y[ok]))
        # Acc_tau uses all points: any NaN/inf prediction -> not accurate.
        rel = np.abs(yp - y) / np.abs(y)
        maxrel = float(np.max(rel)) if np.all(np.isfinite(yp)) else float("inf")
    out["max_rel_err"] = maxrel if np.isfinite(maxrel) else float("inf")
    for t in TAUS:
        out[f"acc_{t}"] = bool(maxrel <= t)
    # SR-Scientist (arXiv:2510.11661, inference/infer/get_results.py): "discard the
    # 5% worst" = more than 95% of points have rel. err < tau; y~0 points count
    # only if yhat~0 too. Non-finite predictions count as misses.
    with np.errstate(all="ignore"):
        yz = np.isclose(y, 0)
        good_rel = np.abs(yp - y) / np.where(yz, 1.0, np.abs(y))
        fin = np.isfinite(yp)
        for t in TAUS:
            hit = np.where(yz, np.isclose(yp, 0, atol=1e-9), good_rel < t) & fin
            out[f"acc95_{t}"] = bool(hit.mean() > 0.95)
    for k in ("nmse", "r2"):
        if not np.isfinite(out[k]):
            out[k] = float("inf") if k == "nmse" else float("-inf")
    return out


def affine_fit(f, y):
    fm, ym = float(np.mean(f)), float(np.mean(y))
    df = f - fm
    sxx = float(np.dot(df, df))
    b = float(np.dot(df, y - ym)) / sxx if sxx > 0 else 0.0
    if not np.isfinite(b):
        b = 0.0
    return ym - b * fm, b


def cmd_run(args):
    from empbench_run import build_pysr_kwargs_and_code, load_dynamic
    from parallel_eval_pysr import _import_pysr_regressor, _get_pysr_num_evaluations

    meta = next(m for m in load_manifest() if m["domain"] == args.domain and m["name"] == args.name)
    data = load_problem(args.domain, args.name)
    X_tr, y_tr = data["train"]
    rng = np.random.default_rng(args.seed_base + args.run_index)
    n_train_total = len(y_tr)
    if args.max_train_samples and n_train_total > args.max_train_samples:
        idx = rng.choice(n_train_total, args.max_train_samples, replace=False)
        X_fit, y_fit = X_tr[idx], y_tr[idx]
    else:
        X_fit, y_fit = X_tr, y_tr

    bargs = argparse.Namespace(
        method=args.method, evolve_results=args.evolve_results,
        max_evals=args.max_evals or 10**12, binary_ops=args.binary_ops, unary_ops=args.unary_ops,
        maxsize=args.maxsize, populations=None, population_size=None,
        extra_kwargs=args.extra_kwargs, parallelism="serial", procs=0,
        timeout_seconds=args.timeout_seconds,
    )
    pysr_kwargs, custom_code, method_meta = build_pysr_kwargs_and_code(bargs)
    if not args.max_evals:
        pysr_kwargs.pop("max_evals", None)  # wall-clock only
    # Early stop at ~exact fit is fine for accuracy metrics and saves compute.

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

    xvars = [f"x{i}" for i in range(X_fit.shape[1])]
    t0 = time.time()
    model.fit(X_fit, y_fit, variable_names=xvars)
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
            preds = {k: np.asarray(model.predict(X, index=int(idx)), float).reshape(-1)
                     for k, (X, _) in data.items()}
        except Exception as e:  # noqa: BLE001
            row["error"] = repr(e)
            frontier.append(row)
            continue
        p_tr = preds["train"]
        a, b = affine_fit(p_tr, y_tr) if np.all(np.isfinite(p_tr)) else (0.0, 1.0)
        row["affine_a"], row["affine_b"] = a, b
        for k, (_, y) in data.items():
            row[k] = metrics(preds[k], y)
            row["aff_" + k] = metrics(a + b * preds[k], y)
        frontier.append(row)

    result = {
        "domain": args.domain, "name": args.name, "gt_expression": meta["expression"],
        "symbols": meta["symbols"], "method": args.method, "run_index": args.run_index,
        "seed": seed, "method_meta": method_meta, "timeout_seconds": args.timeout_seconds,
        "max_evals": args.max_evals, "n_train_total": n_train_total, "n_train_fit": len(y_fit),
        "wallclock_s": round(wall, 1), "num_evals": _get_pysr_num_evaluations(model),
        "pysr_kwargs": {k: v for k, v in pysr_kwargs.items()
                        if k in ("binary_operators", "unary_operators", "maxsize", "maxdepth",
                                 "populations", "population_size", "elementwise_loss",
                                 "early_stop_condition", "timeout_in_seconds", "max_evals")},
        "frontier": frontier, "completed": True,
    }
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1, default=str)
    os.replace(tmp, out)
    pre = "aff_" if args.method == "evolve" else ""
    rows = [r for r in frontier if "train" in r]
    sel = min(rows, key=lambda r: r[pre + "train"]["nmse"], default=None)
    if sel:
        print(f"[{args.domain}/{args.name}/{args.method}/r{args.run_index}] t={wall:.0f}s "
              f"evals={result['num_evals']} {sel['equation'][:120]}  "
              f"ID nmse={sel[pre+'id_test']['nmse']:.3g} acc0.1={sel[pre+'id_test']['acc_0.1']} "
              f"OOD nmse={sel.get(pre+'ood_test', {}).get('nmse', float('nan')):.3g}", flush=True)


def cmd_make_jobs(args):
    out_dir = Path(args.output_dir)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    probs = [m for m in load_manifest() if not args.domains or m["domain"] in args.domains]
    if args.limit:  # first N problems of each domain (smoke tests)
        probs = [m for d in DOMAINS for m in [p for p in probs if p["domain"] == d][: args.limit]]
    jobs = []
    for r in range(args.n_runs):
        for m in probs:
            for method in args.methods:
                job = {"method": method, "domain": m["domain"], "name": m["name"],
                       "run_index": r, "seed_base": args.seed_base,
                       "timeout_seconds": args.timeout_seconds,
                       "max_train_samples": args.max_train_samples,
                       "out": str(out_dir / "results" / f"{m['domain']}__{m['name']}__{method}__r{r}.json")}
                if method == "evolve":
                    job["evolve_results"] = args.evolve_results
                if args.max_evals:
                    job["max_evals"] = args.max_evals
                jobs.append(job)
    (out_dir / "jobs.json").write_text(json.dumps(jobs, indent=1))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=1, default=str))
    print(f"wrote {len(jobs)} jobs -> {out_dir/'jobs.json'}")


def cmd_worker(args):
    jobs = json.loads(Path(args.jobs).read_text())
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    if index >= len(jobs):
        print(f"index {index} >= n_jobs {len(jobs)}; nothing to do")
        return
    job = jobs[index]
    if Path(job["out"]).exists():
        try:
            if json.loads(Path(job["out"]).read_text()).get("completed"):
                print(f"SKIP (already completed): {job['out']}", flush=True)
                return
        except Exception:  # noqa: BLE001
            pass
    argv = ["run"]
    for k, v in job.items():
        if v is not None:
            argv += ["--" + k.replace("_", "-"), str(v)]
    print("RUN:", " ".join(argv), flush=True)
    main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--method", choices=["baseline", "evolve"], required=True)
    r.add_argument("--evolve-results", default=None)
    r.add_argument("--domain", choices=DOMAINS, required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--run-index", type=int, default=0)
    r.add_argument("--seed-base", type=int, default=10000)
    r.add_argument("--timeout-seconds", type=float, required=True)
    r.add_argument("--max-evals", type=int, default=None)
    r.add_argument("--max-train-samples", type=int, default=None)
    r.add_argument("--binary-ops", default=None)
    r.add_argument("--unary-ops", default=None)
    r.add_argument("--maxsize", type=int, default=None)
    r.add_argument("--extra-kwargs", default=None)
    r.add_argument("--out", required=True)

    j = sub.add_parser("make-jobs")
    j.add_argument("--output-dir", required=True)
    j.add_argument("--methods", nargs="+", default=["baseline", "evolve"])
    j.add_argument("--evolve-results", default="runs/709715")
    j.add_argument("--domains", nargs="*", default=None)
    j.add_argument("--n-runs", type=int, default=1)
    j.add_argument("--seed-base", type=int, default=10000)
    j.add_argument("--timeout-seconds", type=float, required=True)
    j.add_argument("--max-evals", type=int, default=None)
    j.add_argument("--max-train-samples", type=int, default=None)
    j.add_argument("--limit", type=int, default=None, help="first N problems per domain (smoke tests)")

    w = sub.add_parser("worker")
    w.add_argument("--jobs", required=True)
    w.add_argument("--index", type=int, default=None)

    args = ap.parse_args(argv)
    if args.cmd == "make-jobs" and "evolve" in args.methods and not args.evolve_results:
        ap.error("--evolve-results required for method evolve")
    {"run": cmd_run, "make-jobs": cmd_make_jobs, "worker": cmd_worker}[args.cmd](args)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("run", "make-jobs", "worker", "-h", "--help"):
        # empbench_slurm_submit calls `<worker> --jobs J --index I`
        main(["worker"] + sys.argv[1:])
    else:
        main()
