"""Shared setup for the Lukeman et al. (2010) POM-vs-MSE reproduction (reports/pom_lukeman2010.md)."""
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "3")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
TABLE_DIR = RUNS / "pom-lukeman-tables"
OBS_DIR = RUNS / "pom-lukeman-obs"

PNAMES = ["w_rep", "w_att", "w_al", "w_front", "w_xi"]
# Ground truth used here (see report: the paper's w_front=0.08, w_xi=0.37 give no visible
# frontal pattern in our integration convention; we keep w_rep/w_att/w_al from the paper).
THETA_TRUE = np.array([8.5, 1.2, 0.74, 0.5, 0.15])
THETA_PAPER = np.array([8.5, 1.2, 0.74, 0.08, 0.37])

# prior: (low, high, log-uniform?)
PRIOR = {
    "w_rep": (2.0, 30.0, True),
    "w_att": (0.3, 5.0, True),
    "w_al": (0.2, 2.5, True),
    "w_front": (0.0, 1.5, False),
    "w_xi": (0.05, 0.6, True),
}
MODELS = ["full", "ARA", "AR", "blind", "angw", "topo"]
FREE = {  # which parameters are free in each candidate structure
    "full": [0, 1, 2, 3, 4], "topo": [0, 1, 2, 3, 4],
    "ARA": [0, 1, 2, 4], "blind": [0, 1, 2, 4], "angw": [0, 1, 2, 4],
    "AR": [0, 1, 4],
}

SIGMAS = [0.0, 0.03, 0.1, 0.2]       # position noise, body lengths
STRIDES = [1, 3, 10]                 # frame interval = stride/30 s -> 30, 10, 3 fps
CONDS = [(s, k) for s in SIGMAS for k in STRIDES]
K_FLOCKS = 5                          # flocks (N=100, 20 s) per dataset

FAMILIES = {"radial": range(0, 16), "dens2d": range(16, 70), "angular": range(70, 79),
            "nnd": range(79, 84), "align": range(84, 91), "global": range(91, 94),
            "temporal": range(94, 96)}
NSUMM = 96


def cond_label(c):
    s, k = CONDS[c]
    return f"sigma={s:g},{30 // k}fps"


def sample_prior(M, model, rng):
    th = np.zeros((M, 5))
    for j, p in enumerate(PNAMES):
        lo, hi, lg = PRIOR[p]
        if j not in FREE[model]:
            continue
        u = rng.random(M)
        th[:, j] = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if lg else lo + u * (hi - lo)
    return th


def to_unc(th, model):
    """natural -> unconstrained (log for log-uniform params) for the free params."""
    cols = []
    for j in FREE[model]:
        lg = PRIOR[PNAMES[j]][2]
        cols.append(np.log(th[..., j]) if lg else th[..., j])
    return np.stack(cols, -1)


def from_unc(z, model):
    th = np.zeros(z.shape[:-1] + (5,))
    for i, j in enumerate(FREE[model]):
        lg = PRIOR[PNAMES[j]][2]
        th[..., j] = np.exp(z[..., i]) if lg else z[..., i]
    return th


_JL = None


def jl():
    global _JL
    if _JL is None:
        from juliacall import Main
        Main.seval(f'include("{HERE / "LukemanSim.jl"}")')
        _JL = Main
    return _JL


def jl_table(thetas, model, K, seed0, conds=None):
    J = jl()
    conds = CONDS if conds is None else conds
    sig = [float(s) for s, _ in conds]
    stri = [int(k) for _, k in conds]
    out = J.LukemanSim.table(J.convert(J.Matrix[J.Float64], np.ascontiguousarray(thetas)), model,
                             int(K), int(seed0), J.convert(J.Vector[J.Float64], sig),
                             J.convert(J.Vector[J.Int], stri))
    return np.asarray(out)


def jl_observed(theta, truename, seeds, mnames, conds=None, hs=(1,)):
    J = jl()
    conds = CONDS if conds is None else conds
    sig = [float(s) for s, _ in conds]
    stri = [int(k) for _, k in conds]
    s, g, n = J.LukemanSim.observed(J.convert(J.Vector[J.Float64], list(map(float, theta))), truename,
                                    J.convert(J.Vector[J.Int], list(map(int, seeds))),
                                    J.convert(J.Vector[J.Float64], sig), J.convert(J.Vector[J.Int], stri),
                                    J.convert(J.Vector[J.String], list(mnames)),
                                    J.convert(J.Vector[J.Int], [int(h) for h in hs]))
    return np.asarray(s), np.asarray(g), np.asarray(n)
