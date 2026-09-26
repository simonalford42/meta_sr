"""Offspring fitness vs estimated seed-noise std, N_init=10 GT runs.

Runs: the three n10 ablation runs (90s, barely_unsolvable, topk, --reeval none),
seeds 1-3. Each offspring has run_gt_scores of shape (20 datasets, 10 seeds);
fitness = mean of that matrix, noise std = std (ddof=1) across seeds of the
per-seed score (mean over datasets), i.e. the std of a single-seed fitness
estimate. Offspring whose evaluation errored are dropped.

Run: python figures/plot_offspring_noise_vs_fitness.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1603675/-home-sca63-meta-sr/1aa53368-9077-4159-acac-024006dcb380/scratchpad")
RUNS = ["64604", "64605", "192865"]
N_BINS = 8


def collect(run):
    d = json.load(open(REPO / f"runs/{run}/run_data.json"))
    fit, std = [], []
    for gen in d["generations"]:
        for o in gen["offspring"]:
            rd = o.get("result_details") or []
            if not rd or any(x.get("errors") or x["n_successful_runs"] < x["n_total_runs"] for x in rd):
                continue
            mat = np.array([x["run_gt_scores"] for x in rd], dtype=float)
            assert mat.shape == (20, 10) and np.isclose(mat.mean(), o["score"]), (run, mat.shape)
            per_seed = mat.mean(axis=0)
            fit.append(per_seed.mean())
            std.append(per_seed.std(ddof=1))
    return fit, std


def main():
    fit, std = [], []
    for run in RUNS:
        f, s = collect(run)
        print(f"{run}: {len(f)} offspring")
        fit += f
        std += s
    fit, std = np.array(fit), np.array(std)

    # 8 equal-count bins over offspring sorted by fitness (quantile bins that
    # stay equal-sized when fitness values tie).
    order = np.argsort(fit, kind="stable")
    bins = np.array_split(order, N_BINS)
    bx = np.array([fit[b].mean() for b in bins])
    by = np.array([std[b].mean() for b in bins])

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.scatter(fit, std, s=14, color="#4c72b0", alpha=0.35, linewidths=0,
               label="Offspring", zorder=2)
    ax.axhline(std.mean(), color="0.35", ls="--", lw=1.2,
               label=f"Mean = {std.mean():.3f}", zorder=1)
    ax.plot(bx, by, color="#c44e52", marker="o", ms=6, lw=1.8,
            markeredgecolor="white", markeredgewidth=0.8,
            label="Binned mean", zorder=3)

    ax.set_xlabel("Offspring fitness (GT match rate)")
    ax.set_ylabel("Noise std across seeds")
    ax.set_ylim(0, std.max() * 1.3)  # headroom for the legend
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=9, loc="upper left", frameon=False, handletextpad=0.4)
    fig.tight_layout()
    out = REPO / "figures/offspring_noise_vs_fitness.pdf"
    fig.savefig(out)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCRATCH / "offspring_noise_vs_fitness.png", dpi=150)  # preview only
    print(f"mean std={std.mean():.4f}, median={np.median(std):.4f}")
    for x, y, b in zip(bx, by, bins):
        print(f"  bin fit={x:.3f} [{fit[b].min():.3f},{fit[b].max():.3f}] n={len(b)}: std={y:.4f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
