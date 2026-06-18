#!/usr/bin/env python3
"""Diagnose num_evals readback + whether max_evals is honored, with warm_start."""
import os, sys
os.environ.setdefault("JULIA_NUM_THREADS", "1")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np
from parallel_eval_pysr import get_default_pysr_kwargs, _import_pysr_regressor
from utils import load_srbench_dataset

PySRRegressor = _import_pysr_regressor()
X, y, _ = load_srbench_dataset("empirical_rydberg", max_samples=1000)
np.random.seed(42); idx = np.random.permutation(len(y)); X, y = X[idx[:40]], y[idx[:40]]
xv = ["x0", "x1"]

k = get_default_pysr_kwargs()
k.update(dict(progress=False, verbosity=0, procs=0, parallelism="serial",
              deterministic=True, random_state=0, warm_start=True,
              output_directory="runs_local/_diagtmp", temp_equation_file=False))
k.pop("delete_tempfiles", None)
m = PySRRegressor(**k)

from juliacall import Main as jl

def readbacks(model):
    out = {}
    try:
        st = model.julia_state_
        out["type"] = str(type(st))
        try: out["len"] = len(st)
        except Exception as e: out["len"] = f"err:{e}"
    except Exception as e:
        out["state_err"] = str(e); return out
    # try several indexings
    for name, fn in {
        "state[0].num_evals": lambda: float(jl.seval("s -> sum(sum, s.num_evals)")(st[0])),
        "state[1].num_evals": lambda: float(jl.seval("s -> sum(sum, s.num_evals)")(st[1])),
    }.items():
        try: out[name] = fn()
        except Exception as e: out[name] = f"err:{type(e).__name__}:{str(e)[:60]}"
    return out

for me in [5000, 50000, 200000]:
    m.max_evals = me
    m.fit(X, y, variable_names=xv)
    rb = readbacks(m)
    print(f"max_evals={me:>8} -> {rb}", flush=True)
import shutil; shutil.rmtree("runs_local/_diagtmp", ignore_errors=True)
