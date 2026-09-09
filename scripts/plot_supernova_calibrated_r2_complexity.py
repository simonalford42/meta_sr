#!/usr/bin/env python3
"""Plot the two saved portfolios' calibrated-R2 complexity-budget curves.

At each integer budget, take each seed's best calibrated R2 among saved
expressions of complexity <= budget. Then aggregate the ten seed values.
This avoids dropping seeds with no expression at a particular complexity.
No searches, Slurm submissions, or model-fitting API calls.
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_supernova_zr_2026_09_08 import ROOT, RUNS, TASK, predict, score

STEM = "supernova_zr_calibrated_r2_vs_complexity"


def main():
    data = pd.read_csv(ROOT/f"pmlb/datasets/{TASK}/{TASK}.tsv.gz", sep="\t")
    x, y = data.iloc[:, 0].to_numpy(), data.target.to_numpy()
    grid = np.arange(1, 31)
    summaries, seed_rows = [], []
    curves = {}
    for label in ["Baseline portfolio", "Evolved portfolio"]:
        source = ROOT/RUNS[label]/"srbench_full_results.json"
        results = json.loads(source.read_text())["results"]
        runs = sorted((r for r in results.values() if r["dataset"] == TASK),
                      key=lambda r:r["seed"])
        assert [r["seed"] for r in runs] == list(range(10000, 10010))
        matrix = []
        for r in runs:
            candidates = []
            for f in r["pareto_frontier"]:
                p = predict(f["equation"], x)
                assert np.isfinite(p).all(), (label, r["seed"], f["equation"])
                calibrated = score(p, y)["affine_r2"]
                assert -1e-10 <= calibrated <= 1+1e-10
                # Only remove machine-roundoff excursions outside [0,1].
                candidates.append((f["complexity"], float(np.clip(calibrated, 0, 1))))
            assert min(c for c, _ in candidates) == 1
            curve = np.array([max(v for c, v in candidates if c <= budget) for budget in grid])
            assert np.all(np.diff(curve) >= 0)
            matrix.append(curve)
            for budget, value in zip(grid, curve):
                seed_rows.append({"method": label, "seed": r["seed"],
                                  "complexity_budget": int(budget), "calibrated_r2": value})
        matrix = np.array(matrix)
        mean, std = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        assert matrix.shape == (10, 30)
        curves[label] = (mean, std)
        for budget, mu, sd in zip(grid, mean, std):
            summaries.append({"method": label, "complexity_budget": int(budget),
                              "n_seeds": 10, "mean_calibrated_r2": mu,
                              "sample_stdev": sd, "lower": mu-sd, "upper": mu+sd})
        print(label, "at complexity 30:", f"{mean[-1]:.6f} +/- {std[-1]:.6f}")

    out = ROOT/"analysis"
    for suffix, rows in [("summary", summaries), ("per_seed", seed_rows)]:
        with (out/f"{STEM}_{suffix}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(left=.10, right=.97, bottom=.19, top=.83)
    colors = {"Baseline portfolio": "#2878AD", "Evolved portfolio": "#D27928"}
    for label, (mean, std) in curves.items():
        ax.fill_between(grid, mean-std, mean+std, color=colors[label], alpha=.15,
                        step="post", linewidth=0)
        ax.step(grid, mean, where="post", color=colors[label], lw=2.2, label=label)
    ax.set_xlim(1, 30)
    ax.set_ylim(-.035, 1.035)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Complexity budget (maximum expression complexity)")
    ax.set_ylabel(r"Best calibrated $R^2$ within budget")
    ax.grid(alpha=.17)
    ax.legend(loc="center left", bbox_to_anchor=(.035, .35), frameon=False, fontsize=10)

    inset = ax.inset_axes([.52, .14, .43, .44])
    for label, (mean, std) in curves.items():
        inset.fill_between(grid, mean-std, mean+std, color=colors[label], alpha=.15,
                           step="post", linewidth=0)
        inset.step(grid, mean, where="post", color=colors[label], lw=1.7)
    inset.set_xlim(12, 30)
    inset.set_ylim(.95, 1.002)
    inset.set_xticks([15, 20, 25, 30])
    inset.set_yticks([.95, .975, 1.0])
    inset.tick_params(labelsize=8)
    inset.set_title("Higher-complexity detail", fontsize=9, pad=6)
    inset.grid(alpha=.15)
    fig.suptitle("Supernova ZR: fit quality versus complexity", fontsize=16, x=.10, ha="left", y=.965)
    fig.text(.10, .898, "Portfolio searches · 1 core · 60 minutes per seed · no max-size warmup",
             fontsize=10, color="#52616B")
    fig.text(.10, .065, "Mean over 10 seeds; shading is ±1 sample standard deviation (not a confidence interval).",
             fontsize=9, color="#52616B")
    fig.text(.10, .03, "Each seed contributes its best saved expression within the budget. Calibration and scoring use the same 236 fit-data rows.",
             fontsize=8.2, color="#52616B")
    fig.savefig(out/f"{STEM}.png", dpi=200)
    fig.savefig(out/f"{STEM}.pdf")
    (out/f"{STEM}.md").write_text(
        "# Supernova ZR: calibrated R² versus complexity\n\n"
        f"![Complexity curves]({STEM}.png)\n\n"
        "The figure compares baseline and evolved 709715 portfolios from September 8: "
        "one core, 60 minutes per seed, 1-million-evaluation restarts, and no max-size warmup. "
        "Seeds are 10000–10009. For each integer complexity budget 1–30, each seed contributes "
        "the maximum calibrated R² among its saved frontier equations with complexity at or "
        "below that budget. Values are carried forward across gaps; there is no linear "
        "interpolation. Thus all ten seeds contribute at every budget. Curves show the mean, "
        "and bands show ±1 sample standard deviation (ddof=1), not SEM or confidence intervals. "
        "Band values are not clipped.\n\n"
        "Calibration fits an external least-squares scale and offset, and R² is evaluated on "
        "the same 236 training observations. It is not raw predictive R², held-out performance, "
        "or Bazin-family recovery. These curves describe saved frontiers, not every candidate "
        "ever searched.\n\n"
        f"- [PDF]({STEM}.pdf)\n- [Summary data]({STEM}_summary.csv)\n"
        f"- [Per-seed curves]({STEM}_per_seed.csv)\n"
        "- [Reproduction script](../scripts/plot_supernova_calibrated_r2_complexity.py)\n")


if __name__ == "__main__":
    main()
