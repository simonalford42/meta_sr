"""Model selection among candidate interaction structures (Lukeman et al. Fig. 3 / SI Fig. S8,
plus a topological-neighbour variant): MSE (leave-one-flock-out CV of individual-level
regression) vs POM (fit each candidate by pattern ABC, then posterior-predictive pattern
discrepancy on fresh simulation seeds).

python select_models.py --truth full [--R 10] [--nrep 12]
Outputs runs/pom-lukeman-selection/<truth>.npz and <truth>.txt
"""
import argparse
import time

import numpy as np

from common import CONDS, FAMILIES, K_FLOCKS, MODELS, OBS_DIR, RUNS, TABLE_DIR, cond_label, jl_table
from pomlib import SAABC, PaperE, lofo_cv

ap = argparse.ArgumentParser()
ap.add_argument("--truth", default="full")
ap.add_argument("--R", type=int, default=10)
ap.add_argument("--nrep", type=int, default=12)
ap.add_argument("--conds", default="0,5,8")
ap.add_argument("--summarize", action="store_true", help="only rebuild the text summary from the saved npz")
a = ap.parse_args()
OUT = RUNS / "pom-lukeman-selection"
OUT.mkdir(parents=True, exist_ok=True)
SEL = [int(c) for c in a.conds.split(",")]

obs = np.load(OBS_DIR / f"{a.truth}.npz")
obs_cidx = list(obs["cidx"])
hs = list(obs["hs"])
R = min(a.R, obs["S"].shape[0])
tables = {m: np.load(TABLE_DIR / f"{m}.npz") for m in MODELS}


def discrepancy(s_obs, reps):
    """mean over pattern families of mean z^2, z from replicate mean/sd (sd floored)."""
    mu, sd = reps.mean(0), reps.std(0, ddof=1)
    sd = np.maximum(sd, 0.02 * np.abs(mu) + 1e-3)
    z2 = ((s_obs - mu) / sd) ** 2
    fam = {f: z2[list(r)].mean() for f, r in FAMILIES.items()}
    return np.mean(list(fam.values())), fam


if a.summarize:
    _d = np.load(OUT / f"{a.truth}.npz")
    res = {k: _d[k] for k in ["mse", "pom", "pomE", "pomfam", "theta"]}
    R = res["pom"].shape[1]
else:
    res = {"mse": np.zeros((len(SEL), R, len(hs), len(MODELS))),
           "pom": np.zeros((len(SEL), R, len(MODELS))),
           "pomE": np.zeros((len(SEL), R, len(MODELS))),
           "pomfam": np.zeros((len(SEL), R, len(MODELS), len(FAMILIES))),
           "theta": np.zeros((len(SEL), R, len(MODELS), 5))}
    t0 = time.time()
    for ic, c in enumerate(SEL):
        oc = obs_cidx.index(c)
        fitters = {}
        for m in MODELS:
            tc = list(tables[m]["cidx"]).index(c)
            fitters[m] = SAABC(tables[m]["theta"], tables[m]["S"][:, :, tc], m)
        for r in range(R):
            s_obs = obs["S"][r, :, oc]
            for j, m in enumerate(MODELS):
                for ih in range(len(hs)):
                    res["mse"][ic, r, ih, j] = lofo_cv(obs["G"][r, oc, j, ih], obs["NR"][r, oc, j, ih], m)
                th = fitters[m].estimate(s_obs)[0]
                res["theta"][ic, r, j] = th
                seed0 = 900_000_000 + 1_000_000 * ic + 10_000 * r + 1000 * j
                reps = jl_table(np.tile(th, (a.nrep, 1)), m, K_FLOCKS, seed0, conds=[CONDS[c]])[:, :, 0]
                D, fam = discrepancy(s_obs, reps)
                res["pom"][ic, r, j] = D
                res["pomfam"][ic, r, j] = list(fam.values())
                # paper's Eq. 12 objective between observed and replicate-mean patterns
                mu = reps.mean(0)
                ang = slice(FAMILIES["angular"].start, FAMILIES["angular"].stop)
                res["pomE"][ic, r, j] = (((PaperE._rho(mu) - PaperE._rho(s_obs)) ** 2).mean()
                                         + ((mu[ang] - s_obs[ang]) ** 2).mean())
            print(f"{cond_label(c)} dataset {r + 1}/{R}: POM D=" + " ".join(
                f"{m}:{v:.2f}" for m, v in zip(MODELS, res["pom"][ic, r])) + f"  ({time.time() - t0:.0f}s)", flush=True)

    np.savez(OUT / f"{a.truth}.npz", sel=np.array(SEL), models=np.array(MODELS), hs=np.array(hs),
             fams=np.array(list(FAMILIES)), **res)

lines = [f"truth={a.truth}  R={R} datasets  nrep={a.nrep} PPC replicates/candidate",
         "counts = how often each candidate is selected (lowest criterion)"]
for ic, c in enumerate(SEL):
    lines.append(f"\n== {cond_label(c)}")
    lines.append(f"{'criterion':22s} " + " ".join(f"{m:>6s}" for m in MODELS))
    for ih, h in enumerate(hs):
        win = np.bincount(res["mse"][ic, :, ih].argmin(1), minlength=len(MODELS))
        lines.append(f"{'MSE LOFO-CV h=' + str(h):22s} " + " ".join(f"{v:6d}" for v in win))
    for key, lab in [("pom", "POM multi-pattern PPC"), ("pomE", "POM paper-E PPC")]:
        win = np.bincount(res[key][ic].argmin(1), minlength=len(MODELS))
        lines.append(f"{lab:22s} " + " ".join(f"{v:6d}" for v in win))
    lines.append("median POM D: " + " ".join(f"{m}={v:.2f}" for m, v in zip(MODELS, np.median(res["pom"][ic], 0))))
    rel = res["mse"][ic, :, 0] / res["mse"][ic, :, 0, MODELS.index(a.truth.split("_")[0])][:, None] - 1
    lines.append("median MSE-CV(h=1) excess vs truth model: " + " ".join(
        f"{m}={v:+.2e}" for m, v in zip(MODELS, np.median(rel, 0))))
    fm = np.median(res["pomfam"][ic], 0)
    lines.append("median per-family z2 (rows=models): " + " ".join(FAMILIES))
    for j, m in enumerate(MODELS):
        lines.append(f"   {m:6s} " + " ".join(f"{v:7.2f}" for v in fm[j]))
open(OUT / f"{a.truth}.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
