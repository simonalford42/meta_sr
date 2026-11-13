"""Shared LSR-Synth (LLM-SRBench, arXiv:2504.10415) pieces.

Used by both the benchmark runner (scripts/lsrsynth_run.py) and the
``lsr_synth`` evolution domain (domains.py), so evolution fitness and the
reported benchmark numbers are computed the same way:

* data: data/llm-srbench/lsr_synth/<domain>/<name>.npz (train / id_test /
  ood_test, column 0 = target), built by scripts/lsrsynth_prepare_data.py.
* operator set: every LSR-Synth ground-truth term is expressible
  (abs, cbrt, cube, and ``^`` with a constant-or-variable exponent).
* scoring: per run, the reported equation is the frontier member with the
  lowest NMSE on all train points (optionally after a train-fit affine
  rescale a + b*f), scored on the test split with NMSE, strict Acc_tau
  (LLM-SRBench / PiT-PO) and 5%-trimmed acc95_tau (SR-Scientist).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LSRSYNTH_DATA_DIR", REPO / "data" / "llm-srbench" / "lsr_synth"))
DOMAINS = ("chem_react", "bio_pop_growth", "phys_osc", "matsci")
TAUS = (0.1, 0.01, 0.001)
SPLITS = ("train", "id_test", "ood_test")
# Evolution / benchmark headline metric: SR-Scientist's 5%-trimmed Acc_0.01.
SOLVED_KEY = "acc95_0.01"
DATASET_PREFIX = "lsr:"


def pysr_opset():
    """Operators + constraints under which every LSR-Synth ground truth is expressible.

    Defaults (+ - * / sin cos exp log sqrt square) plus:
      ^     ε^0.6, P/t^2.6, ε^n·exp(-Q/RT); exponent limited to complexity 1
            (a constant or a variable) and no nested powers, so power laws stay
            interpretable
      abs   |v|^(1/3), log(|v|+1), exp(-|x|)
      cbrt  A^(1/3), P^(1/3), |v|^(1/3) without the negative-base NaN of ^
      cube  x^3, v^3, ε^3 (safe for negative bases, unlike ^ with a fitted 3.0)
    """
    return {
        "binary_operators": ["+", "-", "*", "/", "^"],
        "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "square",
                            "cube", "cbrt", "abs"],
        "constraints": {"sin": 9, "cos": 9, "exp": 9, "log": 9, "sqrt": 9,
                        "/": (-1, 9), "^": (-1, 1)},
        "nested_constraints": {
            "sin": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
            "cos": {"sin": 0, "cos": 0, "exp": 1, "log": 1, "sqrt": 1, "square": 1},
            "exp": {"exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
            "sqrt": {"sqrt": 0},
            "^": {"^": 0},
        },
    }


def load_manifest():
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/lsrsynth_prepare_data.py first")
    return json.loads(path.read_text())


def dataset_name(domain, name):
    return f"{DATASET_PREFIX}{domain}/{name}"


def parse_dataset_name(ds):
    if not ds.startswith(DATASET_PREFIX):
        raise ValueError(f"not an LSR-Synth dataset name: {ds!r}")
    domain, name = ds[len(DATASET_PREFIX):].split("/", 1)
    return domain, name


@lru_cache(maxsize=256)
def _load_npz(domain, name):
    d = np.load(DATA_DIR / domain / f"{name}.npz")
    return {k: (d[k][:, 1:].astype(np.float64), d[k][:, 0].astype(np.float64))
            for k in SPLITS if k in d}


def load_problem(domain, name):
    """{'train'|'id_test'|'ood_test': (X, y)}; column 0 of the npz is the target."""
    return _load_npz(domain, name)


def subsample(X, y, n, seed):
    if n is None or len(y) <= n:
        return X, y
    idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
    return X[idx], y[idx]


def metrics(yp, y):
    """NMSE (NaN preds dropped, as in llm-srbench) + strict Acc_tau + SR-Scientist acc95_tau."""
    yp = np.asarray(yp, float).reshape(-1)
    fail = {f"acc{v}_{t}": False for t in TAUS for v in ("", "95")}
    if yp.shape != y.shape:
        return {"nmse": float("inf"), "r2": float("-inf"), "n_valid": 0, **fail}
    ok = ~np.isnan(yp)
    out = {"n_valid": int(ok.sum()), "frac_valid": float(ok.mean())}
    if ok.sum() == 0:
        out.update(nmse=float("inf"), r2=float("-inf"), **fail)
        return out
    err = yp[ok] - y[ok]
    with np.errstate(all="ignore"):
        mse = float(np.mean(err ** 2))
        out["nmse"] = mse / float(np.var(y[ok]))
        out["r2"] = 1.0 - mse / float(np.var(y[ok]))
        rel = np.abs(yp - y) / np.abs(y)
        maxrel = float(np.max(rel)) if np.all(np.isfinite(yp)) else float("inf")
    out["max_rel_err"] = maxrel if np.isfinite(maxrel) else float("inf")
    for t in TAUS:
        out[f"acc_{t}"] = bool(maxrel <= t)
    # SR-Scientist (inference/infer/get_results.py): more than 95% of points have
    # rel. err < tau; y~0 points count only if yhat~0. Non-finite preds = misses.
    with np.errstate(all="ignore"):
        yz = np.isclose(y, 0)
        good_rel = np.abs(yp - y) / np.where(yz, 1.0, np.abs(y))
        fin = np.isfinite(yp)
        for t in TAUS:
            hit = np.where(yz, np.isclose(yp, 0, atol=1e-9), good_rel < t) & fin
            out[f"acc95_{t}"] = bool(hit.mean() > 0.95)
    for k in ("nmse", "r2"):
        if not np.isfinite(out[k]):
            out[k] = float("inf") if k == "nmse" else float("-inf")
    return out


def affine_fit(f, y):
    fm, ym = float(np.mean(f)), float(np.mean(y))
    df = f - fm
    sxx = float(np.dot(df, df))
    b = float(np.dot(df, y - ym)) / sxx if sxx > 0 else 0.0
    if not np.isfinite(b):
        b = 0.0
    return ym - b * fm, b


def select_and_score(pred_train, pred_test, y_train, y_test, affine):
    """Benchmark selection rule over one run's frontier.

    pred_train / pred_test: {frontier_index: predictions}. Picks the member with
    the lowest train NMSE (after a train-fit affine rescale when ``affine``) and
    returns (index, test metrics) or (None, None) if nothing is finite.
    """
    best = None
    for idx, p_tr in pred_train.items():
        p_tr = np.asarray(p_tr, float).reshape(-1)
        if p_tr.shape != y_train.shape or not np.all(np.isfinite(p_tr)):
            continue
        a, b = affine_fit(p_tr, y_train) if affine else (0.0, 1.0)
        n = metrics(a + b * p_tr, y_train)["nmse"]
        if np.isfinite(n) and (best is None or n < best[0]):
            best = (n, idx, a, b)
    if best is None:
        return None, None
    _, idx, a, b = best
    m = metrics(a + b * np.asarray(pred_test[idx], float).reshape(-1), y_test)
    m.update(train_nmse=best[0], affine_a=a, affine_b=b)
    return idx, m
