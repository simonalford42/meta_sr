#!/usr/bin/env python3
"""Independent re-scoring of LSR-Synth result JSONs (audit of lsrsynth_summarize).

Shares no scoring code with lsr_synth.py / lsrsynth_summarize.py: re-parses each
saved frontier equation string, evaluates it on the raw npz data in float64,
re-selects the lowest-train-NMSE member (after a train-only affine fit a + b*f
when --affine), and recomputes SR-Scientist's acc95_tau from its definition.

    python scripts/lsrsynth_audit_rescore.py RESULT_GLOB [--affine] [--out x.json]
"""
import argparse
import glob
import json
import re
import sys

import numpy as np

DATA = "data/llm-srbench/lsr_synth"
FUNCS = {
    "sin": np.sin, "cos": np.cos, "exp": np.exp, "abs": np.abs,
    "square": np.square, "cube": lambda v: v ** 3, "cbrt": np.cbrt,
    # PySR's safe log/sqrt return NaN outside the domain
    "log": lambda v: np.where(v > 0, np.log(np.abs(v) + (v <= 0)), np.nan),
    "sqrt": lambda v: np.where(v >= 0, np.sqrt(np.abs(v)), np.nan),
}


def evaluate(eq, X):
    expr = eq.replace("^", "**")
    ns = dict(FUNCS)
    for i in range(X.shape[1]):
        ns[f"x{i}"] = X[:, i]
    if re.search(r"[A-Za-z_]\w*", re.sub(r"\b(x\d+|" + "|".join(FUNCS) + r")\b|e[-+]?\d", "", expr)):
        raise ValueError(f"unknown token in {eq!r}")
    with np.errstate(all="ignore"):
        out = eval(expr, {"__builtins__": {}}, ns)  # noqa: S307 (our own saved equations)
    return np.broadcast_to(np.asarray(out, float), (X.shape[0],)).copy()


def acc95(yp, y, tau):
    with np.errstate(all="ignore"):
        zero = np.isclose(y, 0)
        rel = np.abs(yp - y) / np.where(zero, 1.0, np.abs(y))
        hit = np.where(zero, np.isclose(yp, 0, atol=1e-9), rel < tau) & np.isfinite(yp)
    return bool(hit.mean() > 0.95)


def rescore(path, affine):
    r = json.load(open(path))
    d = np.load(f"{DATA}/{r['domain']}/{r['name']}.npz")
    Xtr, ytr = d["train"][:, 1:], d["train"][:, 0]
    Xte, yte = d["id_test"][:, 1:], d["id_test"][:, 0]
    best = None
    for fr in r["frontier"]:
        try:
            p = evaluate(fr["equation"], Xtr)
        except Exception:  # noqa: BLE001
            continue
        if not np.all(np.isfinite(p)):
            continue
        a, b = 0.0, 1.0
        if affine:
            dp = p - p.mean()
            s = dp @ dp
            b = (dp @ (ytr - ytr.mean())) / s if s > 0 else 0.0
            a = ytr.mean() - b * p.mean()
        nmse = np.mean((a + b * p - ytr) ** 2) / np.var(ytr)
        if np.isfinite(nmse) and (best is None or nmse < best[0]):
            best = (nmse, fr["equation"], a, b)
    if best is None:
        return {"solved_0.01": False, "solved_0.001": False, "equation": None}
    _, eq, a, b = best
    yp = a + b * evaluate(eq, Xte)
    return {"solved_0.01": acc95(yp, yte, 0.01), "solved_0.001": acc95(yp, yte, 0.001),
            "equation": eq, "nmse": float(np.mean((yp - yte) ** 2) / np.var(yte))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob")
    ap.add_argument("--affine", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    res = {f: rescore(f, args.affine) for f in files}
    n = len(res)
    print(f"{n} runs  acc95_0.01 {100*np.mean([v['solved_0.01'] for v in res.values()]):.1f}%  "
          f"acc95_0.001 {100*np.mean([v['solved_0.001'] for v in res.values()]):.1f}%")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
