#!/usr/bin/env python3
"""Plot planet TRAIN + TEST RMSE vs complexity, baseline vs evolved, per checkpoint.

Four lines per subplot: test = solid full-shade, train = dashed lighter-shade,
gray = baseline, purple = evolved. Test RMSE comes from the stored frontier;
train RMSE is recomputed per equation on X_train with the same planet
unstable-systems RMSE (PySR `loss` isn't comparable: the evolved run uses a
custom normalized-MSE loss). Star = val-selected equation (on the test line).
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

# (name, result.json, test_color_solid, train_color_light)
BASELINE = ("Baseline (834989)", META / "runs/834989/planet_eval_result.json", "tab:gray", "silver")
NEW = ("Evolved 120459 (397143)", META / "runs/397143/planet_eval_result.json", "tab:purple", "#c9a8e6")

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
        test_rows, train_rows = [], []
        for r in ck.get("frontier") or []:
            c, eq = r.get("complexity"), r.get("equation")
            if c is None:
                continue
            if r.get("test_rmse") is not None:
                test_rows.append((float(c), float(r["test_rmse"])))
            if eq:
                tr = train_rmse(eq)
                if tr is not None:
                    train_rows.append((float(c), tr))
        test_rows.sort()
        train_rows.sort()
        best = ck.get("best") or {}
        bm = best.get("metrics") or {}
        best_test = None
        if best.get("complexity") is not None and bm.get("rmse") is not None:
            best_test = (float(best["complexity"]), float(bm["rmse"]))
        out[ck["label"]] = {"test": test_rows, "train": train_rows, "best_test": best_test}
    return out


baseline = load_checkpoints(BASELINE[1])
new = load_checkpoints(NEW[1])

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, label in zip(axes, CHECKPOINT_ORDER):
    for name, ckd, test_color, train_color in [
        (BASELINE[0], baseline.get(label), BASELINE[2], BASELINE[3]),
        (NEW[0], new.get(label), NEW[2], NEW[3]),
    ]:
        if not ckd:
            continue
        if ckd["test"]:
            xs, ys = zip(*ckd["test"])
            ax.plot(xs, ys, marker="o", color=test_color, label=f"{name} — test")
        if ckd["train"]:
            xs, ys = zip(*ckd["train"])
            ax.plot(xs, ys, marker="o", markersize=4, linestyle="--", color=train_color,
                    label=f"{name} — train")
        if ckd["best_test"]:
            ax.scatter([ckd["best_test"][0]], [ckd["best_test"][1]], color=test_color, s=160,
                       marker="*", zorder=5, edgecolor="black", linewidth=0.5)
    ax.set_title(CHECKPOINT_TITLES.get(label, label))
    ax.set_xlabel("Complexity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

axes[0].set_ylabel("RMSE (unstable systems)")
fig.suptitle("Planet f2: train (dashed) + test (solid) RMSE vs complexity  (★ = val-selected)", y=1.02)
fig.tight_layout()

out = META / "plots/planet_train_test_rmse_vs_complexity_834989_vs_397143_checkpoints.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
