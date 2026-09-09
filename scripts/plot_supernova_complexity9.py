#!/usr/bin/env python3
"""Show the log singularity in the saved, calibrated complexity-budget-9 fit."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_supernova_zr_2026_09_08 import ROOT, TASK, predict


def main():
    table = pd.read_csv(ROOT/"analysis/supernova_zr_calibrated_r2_vs_post_calibration_complexity_candidates.csv")
    frame = pd.read_csv(ROOT/f"pmlb/datasets/{TASK}/{TASK}.tsv.gz", sep="\t").sort_values("Xaxis0")
    t, flux = frame.iloc[:, 0].to_numpy(), frame.target.to_numpy()
    selected = {}
    for method, rows in table[(table.seed == 10009) & (table.post_calibration_complexity <= 9)].groupby("method"):
        selected[method] = rows.loc[rows.calibrated_r2.idxmax()].to_dict()
    center = 4.234760003602217
    lower, upper = t[t<center].max(), t[t>center].min()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    colors = {"Baseline portfolio": "#2878AD", "Evolved portfolio": "#D27928"}
    for ax, interval, ylim in zip(axes, [(-20, 90), (-3, 11)], [(-.15, 2.35), (.5, 2.35)]):
        grid = np.r_[np.linspace(interval[0], center-.0001, 1800), np.nan,
                     np.linspace(center+.0001, interval[1], 1800)]
        ax.axvspan(lower, upper, color="#D27928", alpha=.08, label="No measurements")
        ax.scatter(t, flux, s=15, color="#222222", alpha=.65, zorder=4, label="Observed flux")
        for method, row in selected.items():
            p = predict(row["calibrated_equation"], grid)
            ax.plot(grid, p, color=colors[method], lw=2, label=method.replace(" portfolio", ""))
        ax.axvline(center, color=colors["Evolved portfolio"], ls=":", lw=1.4)
        ax.set_xlim(*interval)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Days relative to observed peak")
        ax.grid(alpha=.15)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Normalized flux after affine calibration")
    axes[0].set_title("Full observed interval")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].set_title("Peak region: singularity inside a sampling gap")
    axes[1].annotate(r"$f(t)\to+\infty$ at day 4.235", xy=(center, 2.3),
                     xytext=(6, 2.1), arrowprops={"arrowstyle": "->", "color": colors["Evolved portfolio"]},
                     fontsize=9, ha="left")
    fig.suptitle("Supernova ZR: complexity budget 9, portfolio seed 10009", fontsize=14)
    fig.text(.5, .02, "Baseline: 8 nodes, calibrated R² = 0.791.  Evolved: 9 nodes, calibrated R² = 0.902.  Both scored only at measurement times.",
             ha="center", fontsize=8.5, color="#52616B")
    fig.tight_layout(rect=(0, .045, 1, .95))
    stem = ROOT/"analysis/supernova_zr_complexity9_singularity"
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    info = {"seed": 10009, "complexity_budget": 9, "selected": selected,
            "singularity_day": center, "measurement_gap": [float(lower), float(upper)],
            "n_observations": len(t), "n_observations_after_day20": int(sum(t>20))}
    stem.with_suffix(".json").write_text(json.dumps(info, indent=2)+"\n")


if __name__ == "__main__":
    main()
