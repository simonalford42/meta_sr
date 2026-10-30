"""Parameter recovery: MSE (individual-level least squares) vs POM (pattern ABC) for the
true model structure, across observation noise sigma and frame rate.

Outputs runs/pom-lukeman-recovery/{estimates.npz, summary.txt}.
"""
import json
import time

import numpy as np

from common import CONDS, OBS_DIR, PNAMES, RUNS, TABLE_DIR, cond_label
from pomlib import SAABC, PaperE, mse_coef, mse_fit

OUT = RUNS / "pom-lukeman-recovery"
OUT.mkdir(parents=True, exist_ok=True)
tab = np.load(TABLE_DIR / "full.npz")
obs = np.load(OBS_DIR / "full.npz")
theta_true = obs["theta"]
R = obs["S"].shape[0]
hs = list(obs["hs"])
models = list(obs["models"])
im = models.index("full")

METHODS = ["MSE-FD", "MSE-smooth*", "POM-paperE", "POM-1pattern", "POM-multi"]
est = {m: np.full((len(CONDS), R, 5), np.nan) for m in METHODS}
ci = np.full((len(CONDS), R, 2, 5), np.nan)
info = {}
t0 = time.time()
for c, (sig, stride) in enumerate(CONDS):
    S_tab = tab["S"][:, :, c]
    sa = SAABC(tab["theta"], S_tab, "full")
    sa1 = SAABC(tab["theta"], S_tab, "full", fams=["radial", "nnd"])
    pe = PaperE(tab["theta"], S_tab, "full")
    info[cond_label(c)] = {"r2_multi": sa.r2.round(3).tolist(), "r2_1pattern": sa1.r2.round(3).tolist()}
    # smoothing half-width chosen with hindsight (min median error of w_rep..w_front) = optimistic MSE
    best_h, best_err, best_est = None, np.inf, None
    for ih, h in enumerate(hs):
        e = np.array([mse_coef(obs["G"][r, c, im, ih], "full") for r in range(R)])
        err = np.median(np.abs(e / theta_true[:4] - 1))
        if err < best_err:
            best_h, best_err, best_est = h, err, e
    info[cond_label(c)]["best_h"] = int(best_h)
    for r in range(R):
        s_obs = obs["S"][r, :, c]
        est["MSE-FD"][c, r] = mse_fit(obs["G"][r, c, im, 0], obs["NR"][r, c, im, 0], "full", sig, stride)
        est["MSE-smooth*"][c, r, :4] = best_est[r]
        est["POM-paperE"][c, r] = pe.estimate(s_obs)[0]
        est["POM-1pattern"][c, r] = sa1.estimate(s_obs)[0]
        m, lo, hi = sa.estimate(s_obs)
        est["POM-multi"][c, r] = m
        ci[c, r, 0], ci[c, r, 1] = lo, hi
    print(f"{cond_label(c)} done {time.time() - t0:.0f}s  r2={sa.r2.round(2)}", flush=True)

np.savez(OUT / "estimates.npz", theta_true=theta_true, ci=ci, methods=np.array(METHODS),
         conds=np.array(CONDS), **{m.replace("*", "_star"): v for m, v in est.items()})
json.dump(info, open(OUT / "info.json", "w"), indent=1)

lines = [f"truth {dict(zip(PNAMES, theta_true))}; R={R} datasets x 5 flocks x 100 birds x 20 s",
         "cells: median over datasets of |est/true - 1| (relative error); MSE-smooth* has no w_xi"]
for c in range(len(CONDS)):
    lines.append(f"\n== {cond_label(c)} (MSE-smooth h*={info[cond_label(c)]['best_h']})")
    lines.append(f"{'method':14s} " + " ".join(f"{p:>9s}" for p in PNAMES) + "   mean4  | median est")
    for m in METHODS:
        e = est[m][c]
        rel = np.median(np.abs(e / theta_true - 1), 0)
        med = np.median(e, 0)
        lines.append(f"{m:14s} " + " ".join(f"{v:9.3f}" for v in rel) + f"   {rel[:4].mean():.3f}  | "
                     + " ".join(f"{v:.3g}" for v in med))
    cov = ((ci[c, :, 0] <= theta_true) & (theta_true <= ci[c, :, 1])).mean(0)
    lines.append(f"POM-multi 90% CI coverage: " + " ".join(f"{v:.1f}" for v in cov))
open(OUT / "summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
