"""Synthetic 'observed' datasets (K=5 flocks of N=100, 20 s each) from a ground-truth model.

For each dataset: pattern summaries under each observation condition, plus per-flock Gram
matrices of the individual-level regression design for every candidate model and smoothing
half-width h (h=1: central finite differences; h>1: local-quadratic smoothing).

python make_obs.py --truth full --R 10            (all 12 conditions)
python make_obs.py --truth topo --R 10 --conds 0,5,8
"""
import argparse
import time

import numpy as np

from common import CONDS, K_FLOCKS, MODELS, OBS_DIR, THETA_TRUE, THETA_PAPER, jl_observed

HS = [1, 2, 4, 8]
ap = argparse.ArgumentParser()
ap.add_argument("--truth", default="full")
ap.add_argument("--theta", default="true", choices=["true", "paper"])
ap.add_argument("--R", type=int, default=10)
ap.add_argument("--conds", default="all")
a = ap.parse_args()
cidx = list(range(len(CONDS))) if a.conds == "all" else [int(c) for c in a.conds.split(",")]
conds = [CONDS[c] for c in cidx]
theta = THETA_TRUE.copy() if a.theta == "true" else THETA_PAPER.copy()
if a.truth in ("ARA", "blind", "angw"):
    theta[3] = 0.0
name = f"{a.truth}" + ("" if a.theta == "true" else "_paper")
OBS_DIR.mkdir(parents=True, exist_ok=True)
S, G, NR = [], [], []
t0 = time.time()
for r in range(a.R):
    seeds = [500_000 * (1 + MODELS.index(a.truth)) + 1000 * (a.theta == "paper") + r * K_FLOCKS + k
             for k in range(K_FLOCKS)]
    s, g, n = jl_observed(theta, a.truth, seeds, MODELS, conds=conds, hs=HS)
    S.append(s); G.append(g); NR.append(n)
    print(f"{name} dataset {r + 1}/{a.R} {time.time() - t0:.0f}s", flush=True)
np.savez(OBS_DIR / f"{name}.npz", S=np.array(S), G=np.array(G), NR=np.array(NR), theta=theta,
         cidx=np.array(cidx), hs=np.array(HS), models=np.array(MODELS))
print("saved", OBS_DIR / f"{name}.npz")
