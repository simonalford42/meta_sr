"""Ricker experiments: (1) likelihood/loss landscapes over log r (Wood Fig. 1), (2) replicate study.

Usage:
  python run_ricker.py landscape --out runs/pom-wood-ricker
  python run_ricker.py replicates --out runs/pom-wood-ricker --seeds 0-39 --shard 0 --nshards 2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ricker import (TRUE_THETA, simulate, sl_loglik, pf_loglik, det_traj, traj_sse_profile,
                    fit_traj_mse, fit_onestep_mse, onestep_pred, N_BURN, N_OBS)
from sl_common import mh_mcmc

LANDSCAPE_SEED = 20100826


def landscape(out):
    rng = np.random.default_rng(LANDSCAPE_SEED)
    y, Ntrue = simulate(TRUE_THETA, 1, rng, return_state=True)
    y = y[0]
    sig, phi = np.exp(TRUE_THETA[1]), np.exp(TRUE_THETA[2])
    res = dict(y=y.tolist(), N=Ntrue[0].tolist())

    # (b) deterministic model, Wood's construction: N_0 = 1 at start of burn-in, sigma = 0, phi = truth
    lr_fine = np.linspace(2.5, 4.5, 4001)
    tr = det_traj(lr_fine, 1.0, n_obs=N_BURN + 1 + N_OBS)[:, N_BURN + 1:]
    lam = np.maximum(phi * tr, 1e-300)
    from scipy.special import gammaln
    res["det_logr"] = lr_fine.tolist()
    res["det_pois_ll"] = (y * np.log(lam) - lam - gammaln(y + 1)).sum(1).tolist()
    res["det_mse"] = ((y - phi * tr) ** 2).mean(1).tolist()
    # (b') trajectory MSE profiled over the initial state N_1 and phi (what an optimizer sees)
    n0_grid = np.geomspace(1e-3, 25.0, 2000)
    prof = []
    for lr in lr_fine[::4]:
        trj = det_traj(np.full(n0_grid.shape, lr), n0_grid)
        s, _ = traj_sse_profile(trj, y)
        prof.append(s.min() / N_OBS)
    res["prof_logr"] = lr_fine[::4].tolist()
    res["prof_mse"] = prof
    # one-step conditional MSE (phi at truth, and profiled)
    res["onestep_mse"] = [float(((y[1:] - onestep_pred((lr, np.log(phi)), y)) ** 2).mean()) for lr in lr_fine]

    # (c) synthetic log-likelihood and (d) particle-filter log-likelihood, other params at truth
    lr_c = np.linspace(2.5, 4.5, 401)
    t0 = time.time()
    res["sl_logr"] = lr_c.tolist()
    res["sl_ll"] = [sl_loglik(np.array([lr, TRUE_THETA[1], TRUE_THETA[2]]), y, rng, 500) for lr in lr_c]
    res["pf_ll"] = [pf_loglik(np.array([lr, TRUE_THETA[1], TRUE_THETA[2]]), y, rng, 2000) for lr in lr_c]
    print(f"landscape SL/PF done in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(out, "landscape.json"), "w") as f:
        json.dump(res, f)


BOUNDS = np.array([[1.0, 6.0], [np.log(0.01), np.log(2.0)], [0.0, 5.0]])


def log_prior(th):
    return 0.0 if np.all((th >= BOUNDS[:, 0]) & (th <= BOUNDS[:, 1])) else -np.inf


def one_dataset(seed, n_iter, burn, truth=TRUE_THETA, methods=("traj", "onestep", "sl", "pf"), refresh=0):
    rng = np.random.default_rng(1000 + seed)
    y = simulate(truth, 1, rng)[0]
    rec = dict(seed=seed, y=y.tolist())
    t0 = time.time()
    rec["truth"] = list(truth)
    if "traj" in methods:
        rec["traj"] = fit_traj_mse(y)
    rec["onestep"] = fit_onestep_mse(y)
    t1 = time.time()
    theta0 = np.array([3.2, np.log(0.5), np.log(15.0)])  # common, deliberately off-truth start
    for name, fn in [("sl", lambda th: sl_loglik(th, y, rng, 500)),
                     ("pf", lambda th: pf_loglik(th, y, rng, 1000))]:
        if name not in methods:
            continue
        chain, lls, acc = mh_mcmc(fn, theta0, [0.05, 0.15, 0.05], n_iter, rng, log_prior=log_prior,
                                  adapt_until=burn, refresh_every=refresh)
        post = chain[burn:]
        rec[name] = dict(mean=post.mean(0).tolist(), median=np.median(post, 0).tolist(),
                         sd=post.std(0).tolist(), q025=np.quantile(post, 0.025, 0).tolist(),
                         q975=np.quantile(post, 0.975, 0).tolist(), acc=acc,
                         thin_chain=chain[::10].tolist())
    rec["time_fit_s"] = t1 - t0
    rec["time_mcmc_s"] = time.time() - t1
    return rec


def replicates(out, seeds, shard, nshards, n_iter, burn, truth=TRUE_THETA, methods=("traj", "onestep", "sl", "pf"),
               refresh=0):
    os.makedirs(os.path.join(out, "replicates"), exist_ok=True)
    for s in seeds[shard::nshards]:
        fn = os.path.join(out, "replicates", f"seed{s:03d}.json")
        if os.path.exists(fn):
            continue
        rec = one_dataset(s, n_iter, burn, truth, methods, refresh)
        with open(fn, "w") as f:
            json.dump(rec, f)
        print(f"seed {s}: " + " ".join(f"{m}={rec[m]['logr'] if 'logr' in rec[m] else rec[m]['mean'][0]:.3f}"
                                         for m in methods if m in rec), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["landscape", "replicates"])
    ap.add_argument("--out", default="runs/pom-wood-ricker")
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n_iter", type=int, default=6000)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--phi", type=float, default=10.0, help="observation scale phi of the data-generating model")
    ap.add_argument("--methods", default="traj,onestep,sl,pf")
    ap.add_argument("--refresh", type=int, default=0, help="MCWM: re-estimate current SL every k iters")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "landscape":
        landscape(a.out)
    else:
        lo, hi = map(int, a.seeds.split("-"))
        truth = np.array([TRUE_THETA[0], TRUE_THETA[1], np.log(a.phi)])
        replicates(a.out, list(range(lo, hi + 1)), a.shard, a.nshards, a.n_iter, a.burn, truth,
                   tuple(a.methods.split(",")), a.refresh)
