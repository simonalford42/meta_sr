#!/usr/bin/env python3
"""Plot the two saved portfolios' calibrated-R2 complexity-budget curves.

At each integer budget, take each seed's best calibrated R2 among saved
expressions of complexity <= budget. Then aggregate the ten seed values.
This avoids dropping seeds with no expression at a particular complexity.
No searches, Slurm submissions, or model-fitting API calls.
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_supernova_zr_2026_09_08 import ROOT, RUNS, TASK, predict, score
from post_calibration_complexity import calibrate, count, parse

STEM = "supernova_zr_calibrated_r2_vs_complexity"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-calibration", action="store_true",
                        help="Count a*f+b after affine constant folding; save a separate plot.")
    post = parser.parse_args().post_calibration
    stem = STEM if not post else "supernova_zr_calibrated_r2_vs_post_calibration_complexity"
    data = pd.read_csv(ROOT/f"pmlb/datasets/{TASK}/{TASK}.tsv.gz", sep="\t")
    x, y = data.iloc[:, 0].to_numpy(), data.target.to_numpy()
    grid = np.arange(1, 35 if post else 31)
    summaries, seed_rows, candidate_rows = [], [], []
    curves = {}
    for label in ["Baseline portfolio", "Evolved portfolio"]:
        source = ROOT/RUNS[label]/"srbench_full_results.json"
        results = json.loads(source.read_text())["results"]
        runs = sorted((r for r in results.values() if r["dataset"] == TASK),
                      key=lambda r:r["seed"])
        assert [r["seed"] for r in runs] == list(range(10000, 10010))
        matrix = []
        for r in runs:
            # Identical, explicitly supplied intercept-only null model for both
            # methods makes low budgets defined even when calibration needs 5 nodes.
            candidates = [(1, 0.0)] if post else []
            for index, f in enumerate(r["pareto_frontier"]):
                p = predict(f["equation"], x)
                assert np.isfinite(p).all(), (label, r["seed"], f["equation"])
                metrics = score(p, y)
                calibrated = metrics["affine_r2"]
                assert -1e-10 <= calibrated <= 1+1e-10
                # Only remove machine-roundoff excursions outside [0,1].
                complexity = f["complexity"]
                if post:
                    assert count(parse(f["equation"])) == complexity
                    if np.ptp(p) == 0:
                        adjusted_complexity, expression = 1, repr(float(y.mean()))
                    else:
                        adjusted_complexity, expression = calibrate(
                            f["equation"], metrics["affine_scale"], metrics["affine_offset"])
                    q = predict(expression, x)
                    assert np.isfinite(q).all()
                    actual_r2 = score(q, y)["raw_r2"]
                    assert abs(actual_r2-calibrated) < 1e-8, (label, r["seed"], index, actual_r2, calibrated)
                    assert adjusted_complexity <= complexity+4
                    candidate_rows.append({"method": label, "seed": r["seed"], "frontier_index": index,
                        "original_complexity": complexity, "post_calibration_complexity": adjusted_complexity,
                        "affine_scale": metrics["affine_scale"], "affine_offset": metrics["affine_offset"],
                        "calibrated_r2": calibrated, "rewritten_raw_r2": actual_r2,
                        "original_equation": f["equation"], "calibrated_equation": expression})
                    complexity = adjusted_complexity
                candidates.append((complexity, float(np.clip(calibrated, 0, 1))))
            assert min(c for c, _ in candidates) == 1
            curve = np.array([max(v for c, v in candidates if c <= budget) for budget in grid])
            assert np.all(np.diff(curve) >= 0)
            matrix.append(curve)
            for budget, value in zip(grid, curve):
                seed_rows.append({"method": label, "seed": r["seed"],
                                  "complexity_budget": int(budget), "calibrated_r2": value})
        matrix = np.array(matrix)
        mean, std = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        assert matrix.shape == (10, len(grid))
        curves[label] = (mean, std)
        for budget, mu, sd in zip(grid, mean, std):
            summaries.append({"method": label, "complexity_budget": int(budget),
                              "n_seeds": 10, "mean_calibrated_r2": mu,
                              "sample_stdev": sd, "lower": mu-sd, "upper": mu+sd})
        print(label, f"at complexity {grid[-1]}:", f"{mean[-1]:.6f} +/- {std[-1]:.6f}")

    out = ROOT/"analysis"
    tables = [("summary", summaries), ("per_seed", seed_rows)]
    if post:
        tables.append(("candidates", candidate_rows))
    for suffix, rows in tables:
        with (out/f"{stem}_{suffix}.csv").open("w", newline="") as f:
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
    ax.set_xlim(1, grid[-1])
    ax.set_ylim(-.035, 1.035)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30] + ([34] if post else []))
    ax.set_xlabel("Post-calibration complexity budget" if post else "Complexity budget (maximum expression complexity)")
    ax.set_ylabel(r"Best calibrated $R^2$ within budget")
    ax.grid(alpha=.17)
    ax.legend(loc="center left", bbox_to_anchor=(.18, .23) if post else (.035, .35),
              frameon=False, fontsize=10)

    inset = ax.inset_axes([.52, .14, .43, .44])
    for label, (mean, std) in curves.items():
        inset.fill_between(grid, mean-std, mean+std, color=colors[label], alpha=.15,
                           step="post", linewidth=0)
        inset.step(grid, mean, where="post", color=colors[label], lw=1.7)
    inset.set_xlim(12 if not post else 15, grid[-1])
    inset.set_ylim(.95, 1.002)
    inset.set_xticks([15, 20, 25, 30] + ([34] if post else []))
    inset.set_yticks([.95, .975, 1.0])
    inset.tick_params(labelsize=8)
    inset.set_title("Higher-complexity detail", fontsize=9, pad=6)
    inset.grid(alpha=.15)
    fig.suptitle("Supernova ZR: complexity after calibration" if post else "Supernova ZR: fit quality versus complexity",
                 fontsize=16, x=.10, ha="left", y=.965)
    fig.text(.10, .898, "Portfolio searches · 1 core · 60 minutes per seed · no max-size warmup",
             fontsize=10, color="#52616B")
    fig.text(.10, .065, "Mean over 10 seeds; shading is ±1 sample standard deviation (not a confidence interval).",
             fontsize=9, color="#52616B")
    fig.text(.10, .03, "Counts nodes in a·f(x)+b after affine constant folding; includes an intercept-only null model. In-sample calibrated R²."
             if post else "Each seed contributes its best saved expression within the budget. Calibration and scoring use the same 236 fit-data rows.",
             fontsize=8.2, color="#52616B")
    fig.savefig(out/f"{stem}.png", dpi=200)
    fig.savefig(out/f"{stem}.pdf")
    if post:
        (out/f"{stem}.md").write_text(
            "# Supernova ZR: post-calibration complexity\n\n"
            f"![Post-calibration curves]({stem}.png)\n\n"
            "Same two portfolios and ten seeds as the original complexity plot. Each expression "
            "is transformed to its least-squares affine calibration a*f(x)+b. We then count its "
            "nodes, with each variable, constant, binary operation and unary function costing one. "
            "The original node counts were checked against all 390 saved expressions.\n\n"
            "Counting uses constant folding, identities with exact zero/one, and affine rewrites "
            "through sums, products, and quotients to absorb scale/offset into existing constants. "
            "For example, a*(c*x+d)+b remains five nodes, while a*x+b costs five rather than one. "
            "No coefficient is dropped because it is small. This is a bounded simplification "
            "policy, not a claim of globally minimal algebraic complexity; transformations through "
            "log/exp/sqrt identities are not searched. The rewritten expressions' raw R² agrees "
            "with the original calibrated R² to within 1e-8 for every candidate.\n\n"
            "Both methods also have an explicit target-mean constant predictor (complexity 1, "
            "R²=0), so all ten seeds contribute even below the smallest nonconstant calibrated "
            "expression. For budgets 1–34, take each seed's best score at or below the budget, "
            "then plot the mean ±1 sample standard deviation (ddof=1). Stepwise curves, no "
            "interpolation, no clipping of standard-deviation bands. Calibration and scoring "
            "use all 236 fit-data rows; there is no held-out evaluation.\n\n"
            f"- [PDF]({stem}.pdf)\n- [Summary]({stem}_summary.csv)\n"
            f"- [Per-seed curves]({stem}_per_seed.csv)\n"
            f"- [Every calibrated equation and node count]({stem}_candidates.csv)\n"
            "- Reproduce: `python scripts/plot_supernova_calibrated_r2_complexity.py --post-calibration`\n")
        return
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
