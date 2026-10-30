"""Robustness of POM to a misspecified observation model: fit observed data from condition
(sigma_true, fps) with the reference table simulated under (sigma_assumed, fps).
Output runs/pom-lukeman-recovery/obs_misspec.txt"""
import numpy as np

from common import CONDS, OBS_DIR, PNAMES, RUNS, SIGMAS, STRIDES, TABLE_DIR
from pomlib import SAABC

tab = np.load(TABLE_DIR / "full.npz")
obs = np.load(OBS_DIR / "full.npz")
tt = obs["theta"]
R = obs["S"].shape[0]
lines = ["rows: true sigma -> assumed sigma (same frame rate); cells: median |est/true-1| per param, mean of 5"]
for k in [10, 3]:
    fitters = {s: SAABC(tab["theta"], tab["S"][:, :, CONDS.index((s, k))], "full") for s in SIGMAS}
    for st in SIGMAS:
        co = CONDS.index((st, k))
        for sa in SIGMAS:
            e = np.array([fitters[sa].estimate(obs["S"][r, :, co])[0] for r in range(R)])
            rel = np.median(np.abs(e / tt - 1), 0)
            lines.append(f"{30 // k}fps sigma {st:g} -> {sa:g}: " + " ".join(f"{p}={v:.3f}" for p, v in zip(PNAMES, rel))
                         + f"  mean={rel.mean():.3f}")
# sigma unknown: pool the four sigma-tables at this frame rate and infer sigma jointly
lines.append("\nsigma unknown (inferred jointly as a nuisance parameter from the pooled tables)")
for k in [10, 3]:
    th = np.concatenate([tab["theta"]] * len(SIGMAS))
    S = np.concatenate([tab["S"][:, :, CONDS.index((s, k))] for s in SIGMAS])
    sig = np.repeat(SIGMAS, len(tab["theta"]))[:, None].astype(float)
    f = SAABC(th, S, "full", extra=sig)
    for st in SIGMAS:
        co = CONDS.index((st, k))
        e, sh = [], []
        for r in range(R):
            e.append(f.estimate(obs["S"][r, :, co])[0]); sh.append(f.last_extra[0])
        rel = np.median(np.abs(np.array(e) / tt - 1), 0)
        lines.append(f"{30 // k}fps sigma {st:g} -> inferred {np.median(sh):.3f}: "
                     + " ".join(f"{p}={v:.3f}" for p, v in zip(PNAMES, rel)) + f"  mean={rel.mean():.3f}")
out = RUNS / "pom-lukeman-recovery" / "obs_misspec.txt"
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
