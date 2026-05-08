"""Histogram of (child score − parent score) with fitted Normal overlay.

Models offspring fitness as N(μ_parent, σ). Reports:
  - μ̂ = mean(c − p)        (negative due to parent-selection bias)
  - σ̂_naive = std(c − p)   (includes evaluation noise on parent and child)
  - σ̂_corr  = √(σ̂_naive² − 2·SE²) where SE is meas SE of a 10-seed mean
                              (subtracts evaluation-noise contribution)
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_parent_child_score import collect_pairs


COLORS = {"explore": "C0", "refine": "C1"}


def fitted_normal(diff, n_seeds=10, sigma_per_seed=0.0704):
    se = sigma_per_seed / np.sqrt(n_seeds)
    meas_var = 2 * se ** 2
    mu = float(diff.mean())
    sigma_naive = float(diff.std(ddof=1))
    sigma_corr = float(np.sqrt(max(0.0, sigma_naive ** 2 - meas_var)))
    return mu, sigma_naive, sigma_corr, se


def plot_hist(ax, diff, *, color, label, bins, n_seeds=10, sigma_per_seed=0.0704):
    mu, sig_naive, sig_corr, se = fitted_normal(
        diff, n_seeds=n_seeds, sigma_per_seed=sigma_per_seed)
    counts, edges, _ = ax.hist(diff, bins=bins, color=color, alpha=0.55,
                               edgecolor="white", label=label)
    bin_width = edges[1] - edges[0]
    xs = np.linspace(edges[0], edges[-1], 400)
    pdf_naive = (1 / (sig_naive * np.sqrt(2 * np.pi))
                 * np.exp(-0.5 * ((xs - mu) / sig_naive) ** 2))
    ax.plot(xs, pdf_naive * len(diff) * bin_width, color=color, linewidth=2,
            label=f"  N(μ={mu:+.3f}, σ_naive={sig_naive:.3f})")
    if sig_corr > 0:
        pdf_corr = (1 / (sig_corr * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((xs - mu) / sig_corr) ** 2))
        ax.plot(xs, pdf_corr * len(diff) * bin_width, color=color,
                linewidth=1.5, linestyle="--",
                label=f"  N(μ={mu:+.3f}, σ_corr={sig_corr:.3f})")


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "947961"
    data = json.loads((Path("runs") / job / "run_data.json").read_text())
    rows = collect_pairs(data)
    if not rows:
        print("No parent/child pairs found")
        return
    diff = np.array([r[4] - r[3] for r in rows])
    modes = np.array([r[2] for r in rows])

    bins = np.linspace(diff.min() - 0.005, diff.max() + 0.005, 40)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: overall.
    ax = axes[0]
    plot_hist(ax, diff, color="C0", label=f"all (n={len(diff)})", bins=bins)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("child score − parent score")
    ax.set_ylabel("count")
    ax.set_title("Offspring − parent fitness, all pairs")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 2: split by mode.
    ax = axes[1]
    for m in sorted(set(modes)):
        mask = modes == m
        if mask.sum() < 5:
            continue
        plot_hist(ax, diff[mask], color=COLORS.get(m, "gray"),
                  label=f"{m} (n={int(mask.sum())})", bins=bins)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("child score − parent score")
    ax.set_ylabel("count")
    ax.set_title("Offspring − parent fitness, by creation mode")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"{job} — offspring − parent score, with fitted Normal "
        f"(σ_corr subtracts 10-seed eval noise)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path("plots") / f"{job}_offspring_minus_parent.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")

    # Print numerical summary.
    mu, sig_n, sig_c, se = fitted_normal(diff)
    print(f"\nall    n={len(diff):4d}  μ={mu:+.4f}  "
          f"σ_naive={sig_n:.4f}  σ_corr={sig_c:.4f}  (meas-SE={se:.4f})")
    for m in sorted(set(modes)):
        mask = modes == m
        if mask.sum() < 5:
            continue
        mu, sig_n, sig_c, _ = fitted_normal(diff[mask])
        print(f"{m:6s} n={int(mask.sum()):4d}  μ={mu:+.4f}  "
              f"σ_naive={sig_n:.4f}  σ_corr={sig_c:.4f}")


if __name__ == "__main__":
    main()
