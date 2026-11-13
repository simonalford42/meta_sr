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
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("JULIA_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

from lsr_synth import (DATA_DIR, DOMAINS, SPLITS, TAUS, affine_fit,  # noqa: E402,F401
                       load_manifest, load_problem, metrics, pysr_opset)

# SR-Scientist (arXiv:2510.11661) App. A PySR baseline config; still capped by --timeout-seconds.
SRSCI_KWARGS = {
    "niterations": 125, "ncycles_per_iteration": 550, "populations": 15, "population_size": 33,
    "maxsize": 30, "maxdepth": None, "weight_randomize": 0.1,
    "binary_operators": ["+", "-", "*", "/", "^"],
    "unary_operators": ["exp", "log", "sqrt", "sin", "cos"],
    "constraints": {"sin": 10, "cos": 10, "exp": 20, "log": 20, "sqrt": 20, "^": (-1, 20)},
    "nested_constraints": {"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0},
                           "exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0},
                           "sqrt": {"sqrt": 0}},
}
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
        method="baseline" if args.method == "srsci" else args.method, evolve_results=args.evolve_results,
        max_evals=args.max_evals or 10**12, binary_ops=args.binary_ops, unary_ops=args.unary_ops,
        maxsize=args.maxsize, populations=None, population_size=None,
        extra_kwargs=args.extra_kwargs, parallelism="serial", procs=0,
        timeout_seconds=args.timeout_seconds,
    )
    pysr_kwargs, custom_code, method_meta = build_pysr_kwargs_and_code(bargs)
    if not args.max_evals:
        pysr_kwargs.pop("max_evals", None)  # wall-clock only
    if args.method == "srsci":
        pysr_kwargs.update(SRSCI_KWARGS)
        if pysr_kwargs.get("maxdepth") is None:
            pysr_kwargs.pop("maxdepth")
        method_meta["config"] = "SR-Scientist App. A"
    if args.opset == "full":
        # every LSR-Synth ground-truth term expressible (lsr_synth.pysr_opset)
        pysr_kwargs.update(pysr_opset())
    method_meta["opset"] = args.opset
    # Early stop at ~exact fit is fine for accuracy metrics and saves compute.

    seed = args.seed_base + args.run_index
    PySRRegressor = _import_pysr_regressor()
    load_dynamic(custom_code)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.parent.parent / "pysr_tmp" / f"{out.stem}_{os.getpid()}"
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
    for r in range(max(args.n_runs, args.srsci_n_runs or 0)):
        for m in probs:
            for method in args.methods:
                n_m = (args.srsci_n_runs or args.n_runs) if method == "srsci" else args.n_runs
                if r >= n_m:
                    continue
                job = {"method": method, "domain": m["domain"], "name": m["name"],
                       "run_index": r, "seed_base": args.seed_base,
                       "timeout_seconds": args.timeout_seconds,
                       "max_train_samples": args.max_train_samples, "opset": args.opset,
                       "out": str(out_dir / "results" / f"{m['domain']}__{m['name']}__{method}__r{r}.json")}
                if method == "evolve":
                    job["evolve_results"] = args.evolve_results
                if args.max_evals:
                    job["max_evals"] = args.max_evals
                jobs.append(job)
    # method-major order, so few --group-size groups mix methods (the worker
    # splits those into one subprocess per method)
    order = {m: i for i, m in enumerate(args.methods)}
    jobs.sort(key=lambda j: order[j["method"]])
    (out_dir / "jobs.json").write_text(json.dumps(jobs, indent=1))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=1, default=str))
    print(f"wrote {len(jobs)} jobs -> {out_dir/'jobs.json'}")


def cmd_worker(args):
    jobs = json.loads(Path(args.jobs).read_text())
    indices = args.index if args.index else [int(os.environ["SLURM_ARRAY_TASK_ID"])]
    if args.group_size > 1:  # array task i runs jobs [i*G, (i+1)*G)
        if len(indices) != 1:
            sys.exit("--group-size takes a single --index (the array task id)")
        g = indices[0]
        indices = list(range(g * args.group_size, min((g + 1) * args.group_size, len(jobs))))
    methods = {jobs[i]["method"] for i in indices if i < len(jobs)}
    if len(methods) > 1:
        # custom Julia code is process-global: a group straddling a method
        # boundary runs each method's jobs in its own subprocess
        rc = 0
        for m in sorted(methods):
            idx = [str(i) for i in indices if i < len(jobs) and jobs[i]["method"] == m]
            rc |= subprocess.call([sys.executable, "-u", __file__, "worker",
                                   "--jobs", args.jobs, "--index", *idx])
        sys.exit(rc)
    for index in indices:
        run_one(jobs, index)


def run_one(jobs, index):
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
    r.add_argument("--method", choices=["baseline", "evolve", "srsci"], required=True)
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
    r.add_argument("--opset", choices=["default", "full"], default="default",
                   help="'full' = lsr_synth.pysr_opset(): adds ^, abs, cbrt, cube")
    r.add_argument("--out", required=True)

    j = sub.add_parser("make-jobs")
    j.add_argument("--output-dir", required=True)
    j.add_argument("--methods", nargs="+", default=["baseline", "evolve"],
                   choices=["baseline", "evolve", "srsci"])
    j.add_argument("--srsci-n-runs", type=int, default=None,
                   help="seeds for the srsci method (default: --n-runs)")
    j.add_argument("--evolve-results", default="runs/709715")
    j.add_argument("--domains", nargs="*", default=None)
    j.add_argument("--n-runs", type=int, default=1)
    j.add_argument("--seed-base", type=int, default=10000)
    j.add_argument("--timeout-seconds", type=float, required=True)
    j.add_argument("--max-evals", type=int, default=None)
    j.add_argument("--max-train-samples", type=int, default=None)
    j.add_argument("--opset", choices=["default", "full"], default="default")
    j.add_argument("--limit", type=int, default=None, help="first N problems per domain (smoke tests)")

    w = sub.add_parser("worker")
    w.add_argument("--jobs", required=True)
    w.add_argument("--index", type=int, nargs="*", default=None,
                   help="one or more job indices, run sequentially in this process (reuses the compiled Julia)")
    w.add_argument("--group-size", type=int, default=1,
                   help="treat --index (or $SLURM_ARRAY_TASK_ID) as a group id over G consecutive jobs")

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
