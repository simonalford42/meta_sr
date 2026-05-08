"""Estimate per-bundle score-noise std from `runs/<job>/run_data.json`.

Each evaluated bundle has `result_details[d].run_gt_scores` of shape
(n_datasets, n_seeds). The bundle score equals the mean of that matrix across
both axes. We estimate noise by computing the per-seed score (mean across
datasets), then taking the std across seeds — that is the std of a single-seed
score estimate. Standard error of the reported mean = std / sqrt(n_seeds).

Outputs:
  - plots/<job>_operator_noise.png  : two-panel figure
  - prints Levene-test stats for "are all bundles' noise stds equal?"
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def collect_bundles(run_data: dict):
    """Return dict bundle_key -> dict(per_seed_scores, n_seeds, mean_score)."""
    bundles = {}
    for gen in run_data["generations"]:
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
            except (KeyError, TypeError, ValueError):
                continue
            if mat.ndim != 2 or mat.shape[1] < 2:
                continue
            per_seed = mat.mean(axis=0)  # shape (n_seeds,)
            bundles[key] = {
                "per_seed": per_seed,
                "n_seeds": int(mat.shape[1]),
                "n_datasets": int(mat.shape[0]),
                "mean": float(per_seed.mean()),
                "std": float(per_seed.std(ddof=1)) if mat.shape[1] > 1 else 0.0,
                "score_reported": entry.get("score"),
            }
    return bundles


def homogeneity_tests(per_seed_list):
    """Run Levene (Brown-Forsythe) and Bartlett tests."""
    # Drop bundles with no variance? Levene handles fine, but require >=2.
    samples = [s for s in per_seed_list if len(s) >= 2]
    if len(samples) < 2:
        return None
    levene = stats.levene(*samples, center="median")
    try:
        bartlett = stats.bartlett(*samples)
    except Exception:
        bartlett = None
    return {"levene": levene, "bartlett": bartlett, "k": len(samples)}


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "947961"
    path = Path("runs") / job / "run_data.json"
    data = json.loads(path.read_text())
    bundles = collect_bundles(data)
    if not bundles:
        print("No evaluated bundles found")
        return

    means = np.array([b["mean"] for b in bundles.values()])
    stds = np.array([b["std"] for b in bundles.values()])
    n_seeds = np.array([b["n_seeds"] for b in bundles.values()])
    per_seed_list = [b["per_seed"] for b in bundles.values()]

    print(f"{len(bundles)} unique evaluated bundles")
    print(f"  seeds per bundle: min={n_seeds.min()}, max={n_seeds.max()}, "
          f"median={int(np.median(n_seeds))}")
    print(f"  per-bundle std: mean={stds.mean():.4f}, median={np.median(stds):.4f}, "
          f"min={stds.min():.4f}, max={stds.max():.4f}")
    print(f"  pooled std (mean over bundles) = {stds.mean():.4f}")
    print(f"  implied SE of reported score (mean / sqrt(n_seeds)): "
          f"mean={np.mean(stds / np.sqrt(n_seeds)):.4f}")

    test = homogeneity_tests(per_seed_list)
    if test is not None:
        lev = test["levene"]
        print(f"\nHomogeneity of variance (H0: all bundles share one noise std):")
        print(f"  Levene (Brown-Forsythe, median): "
              f"W={lev.statistic:.3f}, p={lev.pvalue:.3e}, k={test['k']}")
        if test["bartlett"] is not None:
            bt = test["bartlett"]
            print(f"  Bartlett (assumes normality):    "
                  f"chi2={bt.statistic:.3f}, p={bt.pvalue:.3e}")
        verdict = "REJECT H0" if lev.pvalue < 0.05 else "fail to reject H0"
        print(f"  At alpha=0.05: {verdict}.")

    # Plotting.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.scatter(means, stds, s=22, alpha=0.6, color="C0", edgecolor="white")
    # Trend: rolling-mean std vs mean score, in 8 quantile bins.
    if len(means) >= 8:
        order = np.argsort(means)
        m_sorted = means[order]
        s_sorted = stds[order]
        n_bins = 8
        edges = np.quantile(m_sorted, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        centers, bin_means, bin_stderr = [], [], []
        for i in range(n_bins):
            mask = (m_sorted >= edges[i]) & (m_sorted < edges[i + 1])
            if mask.sum() == 0:
                continue
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            bin_means.append(s_sorted[mask].mean())
            bin_stderr.append(s_sorted[mask].std() / max(1, np.sqrt(mask.sum())))
        ax.errorbar(centers, bin_means, yerr=bin_stderr, fmt="o-", color="C3",
                    capsize=4, linewidth=1.8, markersize=7,
                    label=f"binned mean std (8 quantile bins)")
    ax.axhline(stds.mean(), color="k", linestyle="--", linewidth=1, alpha=0.5,
               label=f"overall mean std = {stds.mean():.3f}")
    ax.set_xlabel("bundle mean score")
    ax.set_ylabel("estimated noise std (across seeds)")
    ax.set_title(f"Per-bundle noise std vs mean score  (n={len(bundles)})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    bin_edges = np.linspace(0, max(stds.max(), 0.15) + 1e-3, 31)
    ax.hist(stds, bins=bin_edges, color="C0", edgecolor="white",
            label=f"observed (n={len(stds)})")
    ax.axvline(stds.mean(), color="C3", linewidth=1.5,
               label=f"observed mean = {stds.mean():.3f}")
    ax.axvline(np.median(stds), color="C2", linewidth=1.5, linestyle="--",
               label=f"observed median = {np.median(stds):.3f}")

    # Null overlay: if every bundle has the same true std σ, the sample std
    # over n seeds follows s = σ √(χ²_{n-1}/(n-1)). Plot that density,
    # rescaled to match the observed histogram's counts.
    n_seeds_null = int(np.median(n_seeds))
    sigma_null = float(stds.mean())
    df = n_seeds_null - 1
    xs = np.linspace(bin_edges[0], bin_edges[-1], 400)
    from scipy.special import gammaln
    log_pdf = (np.log(2) + (df / 2) * np.log(df / 2) - gammaln(df / 2)
               + (df - 1) * np.log(np.maximum(xs, 1e-12) / sigma_null)
               - df * xs ** 2 / (2 * sigma_null ** 2)
               - np.log(sigma_null))
    pdf = np.exp(log_pdf)
    bin_width = bin_edges[1] - bin_edges[0]
    ax.plot(xs, pdf * len(stds) * bin_width, color="C1", linewidth=2,
            label=f"null: σ={sigma_null:.3f} for all (n_seeds={n_seeds_null})")

    ax.set_xlabel("estimated noise std")
    ax.set_ylabel("count")
    title = f"Distribution of per-bundle noise stds"
    if test is not None:
        title += f"\nLevene p = {test['levene'].pvalue:.2e}"
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"{job} — score-noise std per bundle (10-seed eval)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path("plots") / f"{job}_operator_noise.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
