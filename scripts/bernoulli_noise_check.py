"""Test whether per-bundle aggregate noise is consistent with the model
  "each (bundle, task) cell is Bernoulli(p_bt), seeds IID across tasks."

Per-seed scores are literally 0/1, so the per-cell Bernoulli claim is
automatic (within-cell std is mechanically sqrt(p̂(1-p̂) n/(n-1))). The
non-trivial test is across-task independence:

  - If for bundle b, on task t, x_{s,t} ~ Bern(p_bt) IID over seeds s,
    and seeds are independent across tasks, then the per-seed
    bundle-score (mean over T tasks) has variance (1/T²) Σ_t p_bt(1-p_bt).

We compare predicted std to observed std-across-seeds per bundle.

Bonus: per-seed-index marginal success rate (under the model, all seeds
exchangeable → equal rates).
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "947961"
    path = Path("runs") / job / "run_data.json"
    data = json.loads(path.read_text())

    bundles = {}
    for gen in data["generations"]:
        for entry in list(gen["population"]) + list(gen["offspring"]):
            ops = entry["operators"]
            key = tuple(sorted((slot, v["name"]) for slot, v in ops.items()))
            if key in bundles:
                continue
            rd = entry.get("result_details") or []
            if not rd:
                continue
            try:
                mat = np.array([d["run_gt_scores"] for d in rd], dtype=float)
            except Exception:
                continue
            if mat.ndim != 2 or mat.shape[1] < 2:
                continue
            bundles[key] = mat  # shape (n_tasks, n_seeds)

    print(f"{len(bundles)} bundles")

    # Verify binary.
    sample = np.concatenate([m.ravel() for m in list(bundles.values())[:50]])
    print(f"binary check on first ~50 bundles: unique values = {np.unique(sample)}")

    obs_means, obs_stds, pred_stds = [], [], []
    seed_success = None  # (n_seeds,) — cumulative successes per seed index
    seed_n = 0
    for mat in bundles.values():
        n_tasks, n_seeds = mat.shape
        per_seed = mat.mean(axis=0)
        obs_means.append(per_seed.mean())
        obs_stds.append(per_seed.std(ddof=1))
        p_hat = mat.mean(axis=1)  # per-task success rate
        # Plug-in estimator. (Unbiased version multiplies by n/(n-1).)
        var_pred = (p_hat * (1 - p_hat)).sum() / (n_tasks ** 2)
        # Bias correction: E[p̂(1-p̂)] = p(1-p)(n-1)/n
        var_pred *= n_seeds / (n_seeds - 1)
        pred_stds.append(np.sqrt(var_pred))
        if seed_success is None:
            seed_success = np.zeros(n_seeds)
        if n_seeds == seed_success.size:
            seed_success += mat.sum(axis=0)  # successes per seed across tasks
            seed_n += n_tasks

    obs_means = np.array(obs_means)
    obs_stds = np.array(obs_stds)
    pred_stds = np.array(pred_stds)

    # Per-seed marginal success rate (averaged across bundles & tasks).
    # seed_n accumulated n_tasks per bundle → total opportunities per seed = seed_n.
    seed_rate = seed_success / seed_n
    print(f"\nPer-seed marginal success rate (across all bundles × tasks):")
    for i, r in enumerate(seed_rate):
        print(f"  seed {i}: {r:.4f}")
    print(f"  mean across seeds: {seed_rate.mean():.4f}  "
          f"std across seeds: {seed_rate.std(ddof=1):.4f}")
    # Under Bernoulli+exchangeable seeds, the SE of each seed's marginal rate is
    # sqrt(p(1-p)/N_cells). Sanity check.
    p_overall = seed_rate.mean()
    se_per_seed = np.sqrt(p_overall * (1 - p_overall) / seed_n)
    print(f"  expected SE per seed under null: {se_per_seed:.4f}")
    # Chi-square: are the n_seeds rates significantly different?
    counts = seed_success
    expected_per_seed = counts.sum() / len(counts)
    chi2 = ((counts - expected_per_seed) ** 2 / expected_per_seed).sum()
    from scipy.stats import chi2 as chi2_dist
    p_chi2 = 1 - chi2_dist.cdf(chi2, df=len(counts) - 1)
    print(f"  chi² test of equal-rates (df={len(counts)-1}): "
          f"chi²={chi2:.2f}, p={p_chi2:.3e}")

    # Aggregate fit stats.
    valid = pred_stds > 0
    ratio = obs_stds[valid] / pred_stds[valid]
    print(f"\nObserved mean σ across bundles  = {obs_stds.mean():.4f}")
    print(f"Predicted mean σ under Bernoulli = {pred_stds.mean():.4f}")
    print(f"Median ratio observed/predicted  = {np.median(ratio):.3f}")
    print(f"Mean   ratio observed/predicted  = {ratio.mean():.3f}")

    # Number of tasks per bundle (assume constant).
    T = next(iter(bundles.values())).shape[0]

    # Plot.
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    ax = axes[0]
    ax.scatter(pred_stds, obs_stds, s=22, alpha=0.5, color="C0", edgecolor="white")
    lim = max(obs_stds.max(), pred_stds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x (perfect fit)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("predicted std  (Bernoulli + seed-independence)")
    ax.set_ylabel("observed std (across seeds)")
    ax.set_title(f"Per-bundle noise: model vs observation  (n={len(bundles)})")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    ax = axes[1]
    seeds = np.arange(len(seed_rate))
    ax.bar(seeds, seed_rate, color="C0", edgecolor="white",
           yerr=se_per_seed, capsize=4, label="observed (±SE under null)")
    ax.axhline(p_overall, color="C3", linestyle="--",
               label=f"overall mean = {p_overall:.3f}")
    ax.set_xticks(seeds)
    ax.set_xlabel("seed index")
    ax.set_ylabel("marginal success rate")
    ax.set_title(f"Seed exchangeability  (chi²={chi2:.1f}, p={p_chi2:.2e})")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3, axis="y")
    # Tighten y-range to make small differences visible.
    span = max(seed_rate.max() - seed_rate.min(), 4 * se_per_seed)
    mid = seed_rate.mean()
    ax.set_ylim(mid - span * 1.5, mid + span * 1.5)

    # Per-(bundle, task) scatter: solve rate p̂_t vs observed std-across-seeds.
    cell_p, cell_std, n_seeds_any = [], [], None
    for mat in bundles.values():
        n_seeds_any = mat.shape[1]
        cell_p.append(mat.mean(axis=1))
        cell_std.append(mat.std(axis=1, ddof=1))
    cell_p = np.concatenate(cell_p)
    cell_std = np.concatenate(cell_std)

    ax = axes[2]
    # Jitter x to reveal density at the 11 discrete k/n values.
    jitter = (np.random.default_rng(0).uniform(-0.5, 0.5, len(cell_p))
              / n_seeds_any * 0.6)
    ax.scatter(cell_p + jitter, cell_std, s=6, alpha=0.15, color="C0",
               edgecolor="none",
               label=f"(bundle, task) cells (n={len(cell_p)}, x jittered)")
    ps = np.linspace(0, 1, 400)
    ax.plot(ps, np.sqrt(ps * (1 - ps) * n_seeds_any / (n_seeds_any - 1)),
            color="C1", linewidth=2,
            label=f"√(p(1−p)·n/(n−1)),  n={n_seeds_any}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, np.sqrt(0.25 * n_seeds_any / (n_seeds_any - 1)) * 1.1)
    ax.set_xlabel("task solve rate p̂  (per bundle × task)")
    ax.set_ylabel("observed std across seeds")
    ax.set_title("Per-(bundle, task) cell std vs solve rate")
    ax.legend(fontsize=9, loc="lower center"); ax.grid(alpha=0.3)

    fig.suptitle(f"{job} — is the noise Bernoulli?  "
                 f"obs mean σ={obs_stds.mean():.3f}, pred={pred_stds.mean():.3f}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path("plots") / f"{job}_bernoulli_noise_check.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
