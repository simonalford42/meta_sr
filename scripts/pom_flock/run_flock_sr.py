#!/usr/bin/env python3
"""PySR search for the flocking kernel f(r) (scripts/pom_flock/FlockSim.jl).

Arms (--arm):
  oracle   individual-level data: plain MSE to f_true on the r grid (upper bound;
           not available in the POM setting)
  <fams>   population patterns only, comma-separated FlockSim families, e.g.
           polar | gr | polar,nn,gr,velcorr
The tree is evaluated on FlockSim.RGRID (the 1-column dataset X), the result is
used as a lookup table for f, the model is simulated with common random numbers
(--crn seeds), and the loss is the z-scored mismatch to the "observed" patterns
(true model, independent seeds 1001-1016).

After the fit, every frontier equation is scored on
  f-error        RMS(f_hat - f_true) / RMS(f_true) over r in [0.3, 3]
  held-out       all 6 pattern families on fresh seeds (2001-2016) in the
                 training regime, and in a held-out regime (--heldout-*)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JULIA_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

FLOCK_JL = Path(__file__).resolve().parent / "FlockSim.jl"

PATTERN_LOSS = """
function flock_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    pred, ok = eval_tree_array(tree, dataset.X, options)
    !ok && return L(Inf)
    v = Main.FlockSim.pattern_loss(pred)
    return isfinite(v) ? L(v) : L(Inf)
end
"""

HELDOUT_REGIME = dict(L=14.0, ETA=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--crn", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm.replace(',', '+').replace('@', '_')}__s{args.seed}"

    from parallel_eval_pysr import _import_pysr_regressor
    PySRRegressor = _import_pysr_regressor()
    from juliacall import Main as jl
    jl.seval("using SymbolicRegression")
    jl.seval(f'include("{FLOCK_JL}")')
    FS = jl.FlockSim

    r = np.array(FS.RGRID, dtype=float)
    f_true = (r - 1) * np.exp(-r)
    X = r.reshape(-1, 1)

    kw = dict(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "square"],
        maxsize=15, populations=4, population_size=30,
        # one iteration = populations * ncycles * ~population_size evals; keep it
        # small so timeout_in_seconds (checked between iterations) is honoured
        ncycles_per_iteration=5,
        niterations=10_000_000, timeout_in_seconds=args.timeout,
        procs=0, parallelism="serial", deterministic=True, random_state=args.seed,
        progress=False, verbosity=0, temp_equation_file=True, delete_tempfiles=True,
    )
    if args.arm != "oracle":
        # "<fams>@2reg" fits the patterns in two observed regimes
        fams, _, nreg = args.arm.partition("@")
        regimes = "[(;), (; L=10.0, ETA=0.8)]" if nreg == "2reg" else "[(;)]"
        jl.seval(f"(f, s) -> FlockSim.configure!(f, s; regimes={regimes})")(fams.split(","), args.crn)
        kw["loss_function"] = PATTERN_LOSS
    else:
        kw["early_stop_condition"] = 1e-12

    model = PySRRegressor(**kw)
    t0 = time.time()
    model.fit(X, f_true, variable_names=["r"])
    fit_s = time.time() - t0
    n_evals = int(jl.seval("FlockSim.NEVAL[]"))

    # ---- evaluation ----
    mask = r >= 0.3
    rms_true = np.sqrt(np.mean(f_true[mask] ** 2))
    fam_losses = jl.seval("""
        (fv, kw) -> begin
            p = FlockSim.Params(; kw...)
            tg, sc = FlockSim.observe(p)
            Dict(FlockSim.family_losses(fv, p, collect(2001:2016), tg, sc))
        end""")
    nt_train = jl.seval("(;)")
    nt_held = jl.seval("(; L=14.0, ETA=0.3)")

    def held(fv, nt):
        return {str(k): float(v) for k, v in dict(fam_losses(fv, nt)).items()}

    rows = []
    for idx, row in model.equations_.iterrows():
        fh = np.asarray(model.predict(X, index=int(idx)), dtype=float)
        ok = np.all(np.isfinite(fh))
        rows.append({
            "index": int(idx), "complexity": int(row["complexity"]),
            "equation": str(row["equation"]), "loss": float(row["loss"]),
            "f_err": float(np.sqrt(np.mean((fh[mask] - f_true[mask]) ** 2)) / rms_true) if ok else None,
            "heldout_train_regime": held(fh, nt_train) if ok else None,
            "heldout_new_regime": held(fh, nt_held) if ok else None,
        })
    truth = {"train_regime": held(f_true, nt_train), "new_regime": held(f_true, nt_held)}
    res = {"arm": args.arm, "seed": args.seed, "timeout": args.timeout, "crn": args.crn,
           "fit_seconds": fit_s, "n_sim_evals": n_evals, "best_index": int(model.get_best().name),
           "frontier": rows, "truth_heldout": truth}
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"{tag}: done in {fit_s:.0f}s, {n_evals} pattern-loss evals")
    for rw in rows:
        print(f"  c={rw['complexity']:2d} loss={rw['loss']:.3g} f_err={rw['f_err']}  {rw['equation']}")


if __name__ == "__main__":
    main()
