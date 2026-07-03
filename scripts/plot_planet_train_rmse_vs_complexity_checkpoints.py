#!/usr/bin/env python3
"""Plot planet TRAIN RMSE vs complexity, baseline vs evolved, at each checkpoint.

The PySR `loss` field is not comparable across runs (the evolved run uses a
custom normalized-MSE loss while the baseline uses plain MSE), so we recompute
the actual training RMSE: evaluate each frontier equation on X_train and apply
the same planet unstable-systems RMSE used for the test plots. One subplot per
checkpoint (1e6 evals / 1h / 8h); the star marks the val-selected equation.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import sympy as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

META = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(META))
from planet_eval import calculate_planet_metrics  # noqa: E402

BASELINE = ("Baseline (834989)", META / "runs/834989/planet_eval_result.json", "tab:gray")
NEW = ("Evolved 120459 (397143)", META / "runs/397143/planet_eval_result.json", "tab:purple")

CHECKPOINT_ORDER = ["evals_1e6", "time_1h", "time_8h"]
CHECKPOINT_TITLES = {"evals_1e6": "1e6 evals", "time_1h": "1 hour", "time_8h": "8 hours"}

data = pickle.load(open(META / "planet_eval_data.pkl", "rb"))
X_train = np.asarray(data["X_train"], dtype=np.float64)
y_train = np.asarray(data["y_train"], dtype=np.float64)
var_names = list(data["variable_names"])
syms = sp.symbols(var_names)
cols = [X_train[:, i] for i in range(len(var_names))]

_cache = {}


def train_rmse(equation_str):
    if equation_str in _cache:
        return _cache[equation_str]
    try:
        expr = sp.sympify(equation_str.replace("^", "**"), locals=dict(zip(var_names, syms)))
        f = sp.lambdify(syms, expr, modules="numpy")
        preds = np.broadcast_to(np.asarray(f(*cols), dtype=np.float64), (X_train.shape[0],)).copy()
        preds = np.where(np.isfinite(preds), preds, 4.0)
        rmse = float(calculate_planet_metrics(y_train, preds)["rmse"])
    except Exception as exc:
        print(f"  eval failed for {equation_str[:40]!r}: {exc}")
        rmse = None
    _cache[equation_str] = rmse
    return rmse


def load_checkpoints(path):
    result = json.load(open(path))
    out = {}
    for ck in result.get("checkpoints") or []:
        rows = []
        for r in ck.get("frontier") or []:
            c, eq = r.get("complexity"), r.get("equation")
            if c is None or not eq:
                continue
            rmse = train_rmse(eq)
            if rmse is not None:
                rows.append((float(c), rmse))
        rows.sort()
        best = ck.get("best") or {}
        best_pt = None
        if best.get("complexity") is not None and best.get("equation"):
            r = train_rmse(best["equation"])
            if r is not None:
                best_pt = (float(best["complexity"]), r)
        out[ck["label"]] = {"frontier": rows, "best": best_pt}
    return out


baseline = load_checkpoints(BASELINE[1])
new = load_checkpoints(NEW[1])

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, label in zip(axes, CHECKPOINT_ORDER):
    for name, ckdata, color in [
        (BASELINE[0], baseline.get(label), BASELINE[2]),
        (NEW[0], new.get(label), NEW[2]),
    ]:
        if not ckdata or not ckdata["frontier"]:
            continue
        xs, ys = zip(*ckdata["frontier"])
        ax.plot(xs, ys, marker="o", color=color, label=name)
        if ckdata["best"]:
            ax.scatter([ckdata["best"][0]], [ckdata["best"][1]], color=color, s=160,
                       marker="*", zorder=5, edgecolor="black", linewidth=0.5)
    ax.set_title(CHECKPOINT_TITLES.get(label, label))
    ax.set_xlabel("Complexity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Train RMSE (unstable systems)")
fig.suptitle("Planet f2: TRAIN RMSE vs complexity  (★ = val-selected equation)", y=1.02)
fig.tight_layout()

out = META / "plots/planet_train_rmse_vs_complexity_834989_vs_397143_checkpoints.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
