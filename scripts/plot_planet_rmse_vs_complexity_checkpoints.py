#!/usr/bin/env python3
"""Plot planet test RMSE vs complexity, baseline vs evolved, at each checkpoint.

Reads the per-checkpoint frontiers from planet_eval_result.json (the new
checkpointed planet_eval.py output) and draws one subplot per checkpoint
(1e6 evals / 1h / 8h). The star marks each frontier's val-selected equation.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

META = Path(__file__).resolve().parents[1]

BASELINE = ("Baseline (834989)", META / "runs/834989/planet_eval_result.json", "tab:gray")
NEW = ("Evolved 120459 (397143)", META / "runs/397143/planet_eval_result.json", "tab:purple")

CHECKPOINT_ORDER = ["evals_1e6", "time_1h", "time_8h"]
CHECKPOINT_TITLES = {"evals_1e6": "1e6 evals", "time_1h": "1 hour", "time_8h": "8 hours"}


def load_checkpoints(path):
    """Return {label: {'frontier': [(complexity, test_rmse)...], 'best': (c, rmse)}}."""
    result = json.load(open(path))
    out = {}
    for ck in result.get("checkpoints") or []:
        rows = [
            (float(r["complexity"]), float(r["test_rmse"]))
            for r in ck.get("frontier") or []
            if r.get("test_rmse") is not None and r.get("complexity") is not None
        ]
        rows.sort()
        best = ck.get("best") or {}
        bm = best.get("metrics") or {}
        best_pt = None
        if best.get("complexity") is not None and bm.get("rmse") is not None:
            best_pt = (float(best["complexity"]), float(bm["rmse"]))
        out[ck["label"]] = {"frontier": rows, "best": best_pt}
    return out


baseline = load_checkpoints(BASELINE[1])
new = load_checkpoints(NEW[1])

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, label in zip(axes, CHECKPOINT_ORDER):
    for name, data, color in [
        (BASELINE[0], baseline.get(label), BASELINE[2]),
        (NEW[0], new.get(label), NEW[2]),
    ]:
        if not data or not data["frontier"]:
            continue
        xs, ys = zip(*data["frontier"])
        ax.plot(xs, ys, marker="o", color=color, label=name)
        if data["best"]:
            ax.scatter([data["best"][0]], [data["best"][1]], color=color, s=160,
                       marker="*", zorder=5, edgecolor="black", linewidth=0.5)
    ax.set_title(CHECKPOINT_TITLES.get(label, label))
    ax.set_xlabel("Complexity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Test RMSE (unstable systems)")
fig.suptitle("Planet f2: test RMSE vs complexity  (★ = val-selected equation)", y=1.02)
fig.tight_layout()

out = META / "plots/planet_rmse_vs_complexity_834989_vs_397143_checkpoints.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
