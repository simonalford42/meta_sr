"""Reference table for POM/ABC: prior draws -> pattern summaries under all observation conditions.

python make_table.py --model full --M 4000
Saves runs/pom-lukeman-tables/<model>.npz with theta (M,5), S (M,NSUMM,ncond). Resumable by chunks.
"""
import argparse
import time

import numpy as np

from common import CONDS, K_FLOCKS, MODELS, TABLE_DIR, jl_table, sample_prior

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--M", type=int, default=4000)
ap.add_argument("--chunk", type=int, default=150)
ap.add_argument("--conds", default="all", help="comma-separated CONDS indices or all")
a = ap.parse_args()
cidx = list(range(len(CONDS))) if a.conds == "all" else [int(c) for c in a.conds.split(",")]
conds = [CONDS[c] for c in cidx]

TABLE_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1000 + MODELS.index(a.model))
theta = sample_prior(a.M, a.model, rng)
seed_base = 10_000_000 * (MODELS.index(a.model) + 1)
parts = TABLE_DIR / f"{a.model}_parts"
parts.mkdir(exist_ok=True)
t0 = time.time()
for c0 in range(0, a.M, a.chunk):
    f = parts / f"{c0:06d}.npy"
    if f.exists():
        continue
    th = theta[c0:c0 + a.chunk]
    S = jl_table(th, a.model, K_FLOCKS, seed_base + c0 * K_FLOCKS, conds=conds)
    np.save(f, S)
    print(f"{a.model} {c0 + len(th)}/{a.M} {time.time() - t0:.0f}s", flush=True)
S = np.concatenate([np.load(parts / f"{c0:06d}.npy") for c0 in range(0, a.M, a.chunk)])
np.savez(TABLE_DIR / f"{a.model}.npz", theta=theta, S=S, cidx=np.array(cidx))
print("done", S.shape)
