#!/usr/bin/env python3
"""Inspect saved supernova frontiers and compare them with a fitted Bazin family.

No SR searches, API calls, or Slurm submissions. Fits only four reference-model
parameters locally. All numerical comparisons use the 236 existing training rows.
"""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
TASK = "first_principles_supernovae_zr"
RUNS = {
    "Baseline single": "runs/srbench2_9-8_gt_baseline_1core_l1_60m_no_warmup",
    "Baseline portfolio": "runs/srbench2_9-8_gt_baseline_1core_l1_portfolio_1m_no_warmup",
    "Evolved single": "runs/709715/srbench2_9-8_ground_truth_1core_60m_no_warmup",
    "Evolved portfolio": "runs/709715/srbench2_9-8_ground_truth_1core_portfolio_1m_no_warmup",
}


def predict(equation, x):
    # Preserve the printed tree's floating-point operations: symbolic
    # simplification can hide cancellation in ill-conditioned evolved formulas.
    binary = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
              ast.Div: np.divide, ast.Pow: np.power}
    unary = {"square": lambda t:t*t, "cube": lambda t:t*t*t,
             "log": np.log, "exp": np.exp, "sqrt": np.sqrt}
    def visit(n):
        if isinstance(n, ast.Constant):
            return np.float64(n.value)
        if isinstance(n, ast.Name) and n.id == "x0":
            return x
        if isinstance(n, ast.BinOp) and type(n.op) in binary:
            return binary[type(n.op)](visit(n.left), visit(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -visit(n.operand) if isinstance(n.op, ast.USub) else visit(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and len(n.args) == 1:
            return unary[n.func.id](visit(n.args[0]))
        raise ValueError(ast.dump(n))
    with np.errstate(all="ignore"):
        p = visit(ast.parse(equation, mode="eval").body)
    return np.broadcast_to(np.asarray(p, dtype=float), x.shape)


def score(p, y):
    dy = y-y.mean()
    dp = p-p.mean()
    norm = np.dot(dy, dy)
    a = np.dot(dp, dy)/np.dot(dp, dp) if np.dot(dp, dp) > 0 else 0.
    b = y.mean()-a*p.mean()
    calibrated = y.mean()+a*dp
    raw_nmse = np.sum((p-y)**2)/norm
    affine_nmse = np.sum((calibrated-y)**2)/norm
    raw_nrmse = np.sqrt(raw_nmse)
    loss = np.sqrt(np.clip(affine_nmse, 0, 1)) + raw_nrmse/(1+raw_nrmse)/256
    return {"raw_r2": float(1-raw_nmse), "affine_r2": float(1-affine_nmse),
            "affine_scale": float(a), "affine_offset": float(b), "evolved_loss": float(loss)}


def bazin(theta, x):
    log_a, log_decay, log_rise, midpoint = theta
    decay, rise = np.exp(log_decay), np.exp(log_rise)
    # A*exp(-decay*t)/(1+exp(-rise*(t-midpoint))).
    return np.exp(np.clip(log_a-decay*x, -700, 700))*expit(rise*(x-midpoint))


def main():
    data = pd.read_csv(ROOT / f"pmlb/datasets/{TASK}/{TASK}.tsv.gz", sep="\t")
    x, y = data.iloc[:, 0].to_numpy(), data.target.to_numpy()
    fits = []
    for midpoint in [-12, -5, 0, 5]:
        for rise in [.15, .4, 1.]:
            f = least_squares(lambda theta: bazin(theta, x)-y,
                [0, np.log(.04), np.log(rise), midpoint],
                bounds=([-8, -7, -4, -40], [8, 0, 2, 40]), max_nfev=1500)
            fits.append(f)
    fit = min(fits, key=lambda f: np.sum(f.fun**2))
    reference = {"theta": fit.x.tolist(), **score(bazin(fit.x, x), y)}
    def shape_residual(theta):
        p = bazin(np.r_[0., theta], x)
        s = score(p, y)
        return s["affine_scale"]*p+s["affine_offset"]-y
    shape_fits = [least_squares(shape_residual, f.x[1:],
        bounds=([-7, -4, -40], [0, 2, 40]), max_nfev=1500) for f in fits]
    shape_fit = min(shape_fits, key=lambda f: np.sum(f.fun**2))
    shape_reference = {"theta": np.r_[0., shape_fit.x].tolist(),
                       **score(bazin(np.r_[0., shape_fit.x], x), y)}
    output = {"dataset": TASK, "n_rows": len(x), "time_range": [float(x.min()), float(x.max())],
              "evaluation": "All comparisons are in-sample, on the same rows used by SR.",
              "bazin_least_squares_fit": reference, "bazin_affine_profile_fit": shape_reference,
              "methods": {}, "per_seed": []}
    representative = {}
    for label, rel in RUNS.items():
        results = json.loads((ROOT/rel/"srbench_full_results.json").read_text())["results"]
        seed_summaries = []
        for r in sorted(results.values(), key=lambda r:r["seed"]):
            if r["dataset"] != TASK:
                continue
            candidates = []
            for f in r["pareto_frontier"]:
                p = predict(f["equation"], x)
                if np.isfinite(p).all():
                    candidates.append({"equation": f["equation"], "complexity": f["complexity"],
                                       "saved_loss": f["loss"], **score(p, y)})
            best_shape = max(candidates, key=lambda c:c["affine_r2"])
            best_raw = max(candidates, key=lambda c:c["raw_r2"])
            low_loss = min(candidates, key=lambda c:c["saved_loss"])
            if label.startswith("Evolved"):
                loss_difference = abs(low_loss["saved_loss"]-low_loss["evolved_loss"])
                if loss_difference > 1e-5:
                    print("Numerically sensitive expression:", label, r["seed"], loss_difference)
            entry = {"method": label, "seed": r["seed"], "runtime_seconds": r["runtime_seconds"],
                     "best_shape": best_shape, "best_raw": best_raw, "lowest_search_loss": low_loss,
                     "successful_restarts": (r.get("portfolio") or {}).get("restart_count_successful")}
            if label.startswith("Evolved"):
                entry["saved_vs_recomputed_loss_difference"] = loss_difference
            seed_summaries.append(entry)
            output["per_seed"].append(entry)
            if r["seed"] == 10009:
                representative[label] = best_shape
        output["methods"][label] = {
            "median_best_raw_r2": float(np.median([s["best_raw"]["raw_r2"] for s in seed_summaries])),
            "median_best_affine_r2": float(np.median([s["best_shape"]["affine_r2"] for s in seed_summaries])),
            "median_best_shape_complexity": float(np.median([s["best_shape"]["complexity"] for s in seed_summaries])),
            "successful_restarts": [s["successful_restarts"] for s in seed_summaries],
            "seeds_shape_better_than_fitted_bazin": sum(s["best_shape"]["affine_r2"] > shape_reference["affine_r2"] for s in seed_summaries),
        }
        print(label, output["methods"][label], flush=True)
    print("Fitted Bazin:", reference)
    print("Affine-profile-fitted Bazin:", shape_reference)
    out = ROOT / "analysis/supernova_zr_diagnostic_2026-09-08.json"
    out.write_text(json.dumps(output, indent=2)+"\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    grid = np.linspace(x.min(), x.max(), 1000)
    for ax, label in zip(axes, ["Baseline portfolio", "Evolved portfolio"]):
        c = representative[label]
        pred = predict(c["equation"], grid)
        ax.scatter(x, y, s=12, c="black", alpha=.5, label="Observed flux", zorder=3)
        ax.plot(grid, bazin(fit.x, grid), color="#1976b6", lw=2, label="Bazin family, refitted")
        if label.startswith("Evolved"):
            ax.plot(grid, pred, color="#ce7722", alpha=.6, ls=":", label="Evolved expression, raw")
        ax.plot(grid, c["affine_offset"]+c["affine_scale"]*pred, color="#ce7722", lw=2,
                label="Saved expression, affine calibrated")
        ax.set_title(f"{label}, seed 10009\nBest shape on saved frontier")
        ax.set_xlabel("Days relative to observed peak")
        ax.set_ylim(-.12, 1.15)
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Normalized flux")
    fig.suptitle("Supernova ZR: fitting the observations vs recovering the reference family")
    fig.tight_layout()
    fig.savefig(ROOT/"analysis/supernova_zr_diagnostic_2026-09-08.png", dpi=180)


if __name__ == "__main__":
    main()
