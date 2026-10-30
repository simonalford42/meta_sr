"""Reference patterns of each candidate structure at the ground-truth weights (20 flocks,
sigma=0.03 BL, 3 fps) for the pattern figure. Output runs/pom-lukeman-selection/ref_patterns.npz"""
import numpy as np

from common import CONDS, MODELS, RUNS, THETA_PAPER, THETA_TRUE, jl_table

out = RUNS / "pom-lukeman-selection"
out.mkdir(parents=True, exist_ok=True)
c = CONDS.index((0.03, 10))
S = {}
for m in MODELS:
    th = THETA_TRUE.copy()
    S[m] = jl_table(th[None], m, 20, 777_000 + 100 * MODELS.index(m), conds=[CONDS[c]])[0, :, 0]
S["full_paper"] = jl_table(THETA_PAPER[None], "full", 20, 778_000, conds=[CONDS[c]])[0, :, 0]
np.savez(out / "ref_patterns.npz", **S)
print({k: np.round(v[70:79], 2) for k, v in S.items()})
