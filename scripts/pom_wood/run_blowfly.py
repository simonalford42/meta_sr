"""Blowfly experiments: fit Wood's stochastic delay model to Nicholson's bf1 series (or a synthetic
series simulated from Wood's parameters) by synthetic likelihood (MCMC) and by two MSE baselines,
then compare how well each fitted model reproduces the data's dynamics/patterns.

Usage:
  python run_blowfly.py real --out runs/pom-wood-blowfly
  python run_blowfly.py synthetic --seed 0 --out runs/pom-wood-blowfly
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blowfly as bf
from sl_common import mh_mcmc

BOUNDS = np.log(np.array([[1e-3, 2.0], [0.1, 300.0], [10.0, 1e5], [1e-3, 4.0], [2.0, 40.0], [1e-3, 4.0]]))
INIT = np.log([0.1, 8.0, 600.0, 0.2, 17.0, 0.3])  # synlik vignette starting values


def log_prior(th):
    return 0.0 if np.all((th >= BOUNDS[:, 0]) & (th <= BOUNDS[:, 1])) else -np.inf


def evaluate(name, theta_det, x, rng, noise=None, n_sims=500):
    """Pattern diagnostics for a fitted parameter set.

    theta_det: dict(delta, P, N0, tau). noise: (var_p, var_d) or None -> profile SL over a grid.
    """
    out = {"name": name}
    nat = np.array([theta_det["delta"], theta_det["P"], theta_det["N0"], theta_det["tau"]])
    det = bf.det_simulate(nat, len(x), 2, h=180.0, n_burn=1000)
    out["skeleton"] = bf.period_and_amp(det)
    out["skeleton_series"] = det.tolist()
    s_obs = bf.stats(x[None], x)[0]
    if noise is None:
        grid = [0.01, 0.03, 0.1, 0.3, 1.0, 2.0]
        best = (-np.inf, None)
        for vp in grid:
            for vd in grid:
                th = np.log([nat[0], nat[1], nat[2], vp, nat[3], vd])
                ll = bf.sl_loglik(th, x, rng)
                if ll > best[0]:
                    best = (ll, (vp, vd))
        noise = best[1]
        out["noise_profiled"] = True
    th = np.log([nat[0], nat[1], nat[2], noise[0], nat[3], noise[1]])
    out["noise"] = list(noise)
    out["sl_loglik"] = float(np.median([bf.sl_loglik(th, x, rng) for _ in range(5)]))
    sims = bf.simulate(th, n_sims, rng, len(x))
    S = bf.stats(sims, x)
    lo, hi = np.quantile(S, 0.025, 0), np.quantile(S, 0.975, 0)
    z = (s_obs - S.mean(0)) / S.std(0)
    out["stat_z"] = dict(zip(bf.STAT_NAMES, z.tolist()))
    out["frac_stats_in_95"] = float(np.mean((s_obs >= lo) & (s_obs <= hi)))
    pa = [bf.period_and_amp(s) for s in sims[:200]]
    out["stoch_summary"] = {k: float(np.median([p[k] for p in pa])) for k in pa[0]}
    out["example_sims"] = sims[:3].tolist()
    # trajectory MSE of the stochastic model's mean path is meaningless; report skeleton-vs-data MSE
    out["skeleton_mse_vs_data_after_burn"] = float(((det - x) ** 2).mean())
    return out


def run(x, xd, out_path, rng, n_iter, burn, truth=None, refresh=0):
    res = {"x": x.tolist(), "data_patterns": bf.period_and_amp(x)}
    if truth is not None:
        res["truth"] = dict(zip(bf.NAMES, np.exp(truth).tolist()))
    t0 = time.time()
    if xd is not None:
        res["onestep"] = bf.fit_onestep_mse(xd)
        print("onestep", res["onestep"], f"{time.time()-t0:.0f}s", flush=True)
    res["traj"] = bf.fit_traj_mse(x, rng=np.random.default_rng(1))
    print("traj", {k: v for k, v in res["traj"].items() if k != "sse_by_tau"}, f"{time.time()-t0:.0f}s", flush=True)
    chain, lls, acc = mh_mcmc(lambda th: bf.sl_loglik(th, x, rng), INIT, np.full(6, 0.03), n_iter, rng,
                              log_prior=log_prior, adapt_until=burn, verbose_every=1000,
                              refresh_every=refresh)
    post = chain[burn:]
    res["sl"] = dict(mean=dict(zip(bf.NAMES, np.exp(post.mean(0)).tolist())),
                     q025=dict(zip(bf.NAMES, np.exp(np.quantile(post, 0.025, 0)).tolist())),
                     q975=dict(zip(bf.NAMES, np.exp(np.quantile(post, 0.975, 0)).tolist())),
                     acc=acc, thin_chain=chain[::10].tolist(), thin_ll=lls[::10].tolist())
    print("SL", res["sl"]["mean"], f"acc={acc:.2f} {time.time()-t0:.0f}s", flush=True)
    m = res["sl"]["mean"]
    evals = [evaluate("synthetic likelihood", m, x, rng, noise=(m["var_p"], m["var_d"]))]
    if xd is not None:
        evals.append(evaluate("one-step MSE", res["onestep"], x, rng))
    evals.append(evaluate("trajectory MSE", res["traj"], x, rng))
    if truth is not None:
        tn = dict(zip(bf.NAMES, np.exp(truth)))
        evals.append(evaluate("truth", tn, x, rng, noise=(tn["var_p"], tn["var_d"])))
    res["eval"] = evals
    for e in evals:
        print(e["name"], "SL", round(e["sl_loglik"], 1), "in95", round(e["frac_stats_in_95"], 2),
              "skeleton", {k: round(v) for k, v in e["skeleton"].items()},
              "stoch", {k: round(v) for k, v in e["stoch_summary"].items()}, flush=True)
    with open(out_path, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "synthetic"])
    ap.add_argument("--out", default="runs/pom-wood-blowfly")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_iter", type=int, default=12000)
    ap.add_argument("--burn", type=int, default=4000)
    ap.add_argument("--refresh", type=int, default=0, help="MCWM: re-estimate current SL every k iters")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(777 + a.seed)
    if a.mode == "real":
        _, x = bf.load_bf1(2)
        _, xd = bf.load_bf1(1)
        run(x, xd, os.path.join(a.out, "real_bf1.json"), rng, a.n_iter, a.burn, refresh=a.refresh)
    else:
        # daily synthetic series from Wood's parameters; observed every 2 days (+ daily for one-step)
        xd = bf.simulate(bf.WOOD_THETA, 1, np.random.default_rng(5000 + a.seed), 275, steps=1)[0]
        x = xd[::2]
        run(x, xd, os.path.join(a.out, f"synthetic_seed{a.seed}.json"), rng, a.n_iter, a.burn, truth=bf.WOOD_THETA,
            refresh=a.refresh)
