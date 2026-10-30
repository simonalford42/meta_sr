#!/usr/bin/env python3
"""POM vs derivative / trajectory losses for recovering one unknown ODE component.

Job (JSON line): {"id", "system", "regime", "arm", "seed", "timeout"}
arms:
  deriv       PySR MSE on finite-difference derivatives of the noisy observations
  traj        short-horizon (H obs steps) trajectory MSE, simulated in the loss
  pom_quant   single pattern: marginal distributions (quantiles) only
  pom_multi   multi-scale patterns: quant + diff + acf + cond + peaks
  pom_traj    0.5 * pom_multi + 0.5 * traj
The data (true trajectory, observation noise) depend only on (system, regime),
never on the arm or seed, so all arms see identical observations.
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
import sympy as sp

POM_JL = Path(__file__).resolve().parent / "PomODE.jl"

# unknown is 1-based (Julia); truth is written in x0.. variables
SYSTEMS = {
    "lorenz": dict(D=3, unknown=3, truth="x0*x1 - 8/3*x2", dt=0.01, ic=[1.0, 1.0, 25.0], burn=20.0,
                   regimes={"clean": dict(sigma=0.01, dt_obs=0.02, T=40.0),
                            "noisy": dict(sigma=0.2, dt_obs=0.1, T=40.0)}),
    "rossler": dict(D=3, unknown=3, truth="0.2 + x2*(x0 - 5.7)", dt=0.01, ic=[1.0, 1.0, 0.0], burn=100.0,
                    regimes={"clean": dict(sigma=0.01, dt_obs=0.1, T=300.0),
                             "noisy": dict(sigma=0.2, dt_obs=0.5, T=300.0)}),
    "vdp": dict(D=2, unknown=2, truth="1.5*(1 - x0**2)*x1 - x0", dt=0.02, ic=[2.0, 0.0], burn=30.0,
                regimes={"clean": dict(sigma=0.01, dt_obs=0.1, T=60.0),
                         "noisy": dict(sigma=0.2, dt_obs=0.4, T=60.0)}),
}
POM_K = int(os.environ.get("POM_K", 20))  # number of observation windows used for pattern statistics
TRAJ_H = 2       # trajectory-loss horizon in observation steps
TRAJ_M = 300     # max starting states for the trajectory loss
POM_GROUPS = {"pom_quant": ["quant"], "pom_multi": ["quant", "diff", "acf", "cond", "peaks"],
              "pom_traj": ["quant", "diff", "acf", "cond", "peaks"]}


def true_rhs(sysname, U):
    if sysname == "lorenz":
        x, y, z = U
        return np.array([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z])
    if sysname == "rossler":
        x, y, z = U
        return np.array([-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)])
    x, y = U
    return np.array([y, 1.5 * (1 - x ** 2) * y - x])


def integrate(sysname, u0, dt, n_steps, record_every):
    u = np.array(u0, float)
    out = [u.copy()]
    for i in range(1, n_steps + 1):
        k1 = true_rhs(sysname, u)
        k2 = true_rhs(sysname, u + 0.5 * dt * k1)
        k3 = true_rhs(sysname, u + 0.5 * dt * k2)
        k4 = true_rhs(sysname, u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % record_every == 0:
            out.append(u.copy())
    return np.array(out)  # [n, D]


def make_data(sysname, regime):
    s = SYSTEMS[sysname]
    r = s["regimes"][regime]
    spo = int(round(r["dt_obs"] / s["dt"]))
    burn = integrate(sysname, s["ic"], s["dt"], int(s["burn"] / s["dt"]), int(s["burn"] / s["dt"]))[-1]
    n_obs = int(r["T"] / r["dt_obs"])
    n_obs -= n_obs % POM_K
    clean = integrate(sysname, burn, s["dt"], (n_obs - 1) * spo, spo)[:n_obs]
    scale = clean.std(0)
    rng = np.random.default_rng(12345)
    obs = clean + r["sigma"] * scale * rng.standard_normal(clean.shape)
    # held-out states for function-space error: a fresh trajectory continuing on
    test = integrate(sysname, clean[-1], s["dt"], int(50 / s["dt"]), max(1, int(0.05 / s["dt"])))
    sim_noise = r["sigma"] * scale[:, None, None] * rng.standard_normal((s["D"], n_obs // POM_K, POM_K))
    return dict(obs=obs, clean=clean, scale=scale, spo=spo, test=test, sim_noise=sim_noise,
                dt_obs=r["dt_obs"])


def evaluate_frontier(eqdf, sysname, data):
    s = SYSTEMS[sysname]
    D = s["D"]
    xs = sp.symbols([f"x{i}" for i in range(D)])
    truth = sp.expand(sp.sympify(s["truth"], locals=dict(zip([str(x) for x in xs], xs))))
    T = data["test"]
    cols = [T[:, i] for i in range(D)]
    f_true = np.broadcast_to(sp.lambdify(xs, truth)(*cols), (len(T),)).astype(float)
    sd = f_true.std()
    true_terms = _terms(truth, xs, cols, sd)
    rows = []
    for _, row in eqdf.iterrows():
        e = sp.expand(row["sympy_format"])
        try:
            with np.errstate(all="ignore"):
                f = np.broadcast_to(sp.lambdify(xs, e)(*cols), (len(T),)).astype(float)
            err = float(np.sqrt(np.mean((f - f_true) ** 2)) / sd) if np.all(np.isfinite(f)) else None
        except Exception:
            err = None
        terms = _terms(e, xs, cols, sd)
        rows.append({"complexity": int(row["complexity"]), "loss": float(row["loss"]),
                     "equation": str(row["equation"]), "rel_err": err,
                     "terms": None if terms is None else sorted(terms),
                     "structure_match": terms == true_terms})
    return rows, sorted(true_terms)


def _terms(e, xs, cols, sd):
    """Monomials whose contribution on the test states exceeds 1% of std(f)."""
    try:
        p = sp.Poly(e, *xs)
    except Exception:
        return None
    out = set()
    for monom, c in p.terms():
        m = np.prod([cols[i] ** k for i, k in enumerate(monom)], axis=0) if any(monom) else 1.0
        mag = abs(float(c)) * float(np.sqrt(np.mean(np.square(m))))
        if mag > 0.01 * sd:
            out.add(str(sp.Mul(*[x ** k for x, k in zip(xs, monom)])))
    return out


def run_job(job, out_dir, PySRRegressor, jl):
    s = SYSTEMS[job["system"]]
    data = make_data(job["system"], job["regime"])
    obs, D, u = data["obs"], s["D"], s["unknown"] - 1
    kw = dict(binary_operators=["+", "-", "*", "/"], unary_operators=[], maxsize=20,
              populations=15, population_size=33, niterations=10_000_000,
              timeout_in_seconds=int(job["timeout"]), random_state=int(job["seed"]),
              deterministic=True, procs=0, parallelism="serial", verbosity=0, progress=False,
              temp_equation_file=True, delete_tempfiles=True, batching=False)
    names = [f"x{i}" for i in range(D)]
    if job["arm"] == "deriv":
        X = obs[1:-1]
        y = (obs[2:, u] - obs[:-2, u]) / (2 * data["dt_obs"])
    else:
        X, y = obs, np.zeros(len(obs))
        n = len(obs)
        starts = np.linspace(0, n - TRAJ_H - 1, min(TRAJ_M, n - TRAJ_H)).astype(int)
        tr_x0 = obs[starts].T.copy()
        tr_y = np.stack([obs[starts + h].T for h in range(1, TRAJ_H + 1)], axis=1)  # D x H x M
        W = obs.reshape(POM_K, n // POM_K, D).transpose(2, 1, 0).copy()        # D x nobs x K
        groups = POM_GROUPS.get(job["arm"], ["quant"])
        alpha = {"traj": 0.0, "pom_traj": 0.5}.get(job["arm"], 1.0)
        jl.seval("PomODE.configure!")(job["system"], s["unknown"], s["dt"], data["spo"],
                                      data["scale"], tr_x0, tr_y, W, data["sim_noise"],
                                      groups, alpha)
        kw["loss_function"] = ("function pom_ode_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}\n"
                               "    return L(Main.PomODE.objective(tree, options))\nend")
    model = PySRRegressor(**kw)
    t0 = time.time()
    model.fit(X, y, variable_names=names)
    fit_s = time.time() - t0
    eq = model.equations_
    rows, true_terms = evaluate_frontier(eq, job["system"], data)
    best = model.get_best()
    best_row = next(r for r, (_, e) in zip(rows, eq.iterrows()) if e.name == best.name)
    res = dict(job, fit_seconds=fit_s, truth=s["truth"], true_terms=true_terms,
               best=best_row, frontier=rows,
               any_structure_match=any(r["structure_match"] for r in rows),
               min_rel_err=min((r["rel_err"] for r in rows if r["rel_err"] is not None), default=None))
    tmp = out_dir / f"{job['id']}.json.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    tmp.rename(out_dir / f"{job['id']}.json")
    return res


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
            r = run_job(job, out_dir, PySRRegressor, jl)
            print(f"[w{args.worker}] {job['id']} struct={r['any_structure_match']} "
                  f"best_err={r['best']['rel_err']} eq={r['best']['equation']} {time.time()-t0:.0f}s",
                  flush=True)
        except Exception:
            print(f"[w{args.worker}] {job['id']} FAILED\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
