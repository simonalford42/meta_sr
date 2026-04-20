"""Compare policies as a function of discovery difficulty (plateau).

Two main plots:
  1. `plateau_smart_advantage`: smart-policy advantage over best-fixed-N at each (σ, plateau).
  2. `plateau_per_policy`: curve of final true_of_declared vs plateau for each policy,
     one panel per σ.

Usage:
  python scripts/plot_plateau_analysis.py --sweep-dir outputs/synthetic_pol/plateau
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_finals(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(sweep_dir.glob("*.json")):
        r = json.loads(f.read_text())
        cfg = r["cfg"]
        task = r.get("task", {})
        plateau = task.get("plateau", float("inf"))
        try:
            plateau = float(plateau)
        except Exception:
            plateau = float("inf")
        last = r["trajectory"][-1]
        rows.append({
            "policy": cfg["policy"],
            "noise": float(cfg["noise_std"]),
            "plateau": plateau,
            "seed": int(cfg["seed"]),
            "best_true": last["best_true_ever"],
            "true_of_decl": last["true_of_declared_best"],
            "inflation": last["noisy_of_declared_best"] - last["true_of_declared_best"],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=str, required=True)
    args = ap.parse_args()
    sweep = Path(args.sweep_dir)
    out_dir = sweep / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_finals(sweep)
    df.to_csv(out_dir / "finals.csv", index=False)

    # convert plateau=inf to a finite for plotting (use max+extra)
    df_plot = df.copy()
    finite_max = df_plot.loc[np.isfinite(df_plot["plateau"]), "plateau"].max()
    df_plot["plateau_plot"] = df_plot["plateau"].replace(float("inf"), finite_max * 3)

    # per-policy curve: y = mean true_of_decl, x = plateau, panel per σ
    noises = sorted(df["noise"].unique())
    policies = sorted(df["policy"].unique())
    ncols = min(3, len(noises))
    nrows = (len(noises) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False, sharey=True)
    axes = axes.ravel()
    for ax in axes[len(noises):]:
        ax.axis("off")
    colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))
    for ax, sigma in zip(axes, noises):
        sub = df_plot[df_plot["noise"] == sigma]
        for c, pol in zip(colors, policies):
            s2 = sub[sub["policy"] == pol]
            if s2.empty:
                continue
            agg = s2.groupby("plateau_plot")["true_of_decl"].agg(["mean", "std"]).reset_index()
            agg = agg.sort_values("plateau_plot")
            ax.errorbar(agg["plateau_plot"], agg["mean"], yerr=agg["std"],
                        label=pol, color=c, lw=1.6, capsize=2)
        ax.set_xscale("log")
        ax.set_title(f"σ = {sigma}")
        ax.set_xlabel("plateau (higher = easier discovery; ∞ = no clipping)")
        ax.grid(alpha=0.3, which="both")
        ax.set_ylabel("final true fitness of declared-best (higher = better)")
    axes[0].legend(loc="lower right", fontsize=8, ncol=2)
    fig.suptitle("Policy comparison as discovery difficulty varies")
    fig.tight_layout()
    out = out_dir / "plateau_per_policy.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out}")

    # smart-policy advantage: best-smart minus best-fixed at each (sigma, plateau)
    fixed_policies = [p for p in policies if p.startswith("fixed")]
    smart_policies = [p for p in policies if not p.startswith("fixed")]

    agg = df.groupby(["policy", "noise", "plateau"])["true_of_decl"].mean().reset_index()

    def best_of(df, group_policies, key):
        return df[df["policy"].isin(group_policies)].loc[
            df[df["policy"].isin(group_policies)]
            .groupby(key)["true_of_decl"].idxmax()
        ]

    best_fixed = agg[agg["policy"].isin(fixed_policies)].groupby(["noise", "plateau"]).agg(
        best_fixed=("true_of_decl", "max"),
        best_fixed_policy=("true_of_decl", lambda x: agg.loc[x.index[x.argmax()], "policy"])
    ).reset_index()
    best_smart = agg[agg["policy"].isin(smart_policies)].groupby(["noise", "plateau"]).agg(
        best_smart=("true_of_decl", "max"),
        best_smart_policy=("true_of_decl", lambda x: agg.loc[x.index[x.argmax()], "policy"])
    ).reset_index()
    merged = best_fixed.merge(best_smart, on=["noise", "plateau"])
    merged["advantage"] = merged["best_smart"] - merged["best_fixed"]
    merged["plateau_plot"] = merged["plateau"].replace(float("inf"), finite_max * 3)
    merged = merged.sort_values(["noise", "plateau_plot"])
    merged.to_csv(out_dir / "smart_vs_fixed.csv", index=False)
    print(merged.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for sigma in noises:
        sub = merged[merged["noise"] == sigma].sort_values("plateau_plot")
        ax.plot(sub["plateau_plot"], sub["advantage"], "-o",
                label=f"σ = {sigma}", lw=1.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("plateau (smaller = harder discovery; ∞ = no clip)")
    ax.set_ylabel("best smart policy  −  best fixed-N\n(positive = smart policy wins)")
    ax.set_title("Smart-policy advantage over best fixed-N, by task difficulty")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "smart_advantage_vs_plateau.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
