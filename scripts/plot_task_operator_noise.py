"""Per-task version of plot_operator_noise.py.

For each dataset in the (20-task train) split, compute per-bundle noise on
that single task — instead of averaging seeds-scores across datasets first.

For task t and bundle b: per-seed scores = result_details[t].run_gt_scores
(shape n_seeds). Then:
  bundle mean on task = mean over seeds
  bundle noise std on task = std over seeds (ddof=1)

Outputs:
  - plots/<job>_task_noise/task_<i>_operator_noise.png  (one per task)
  - prints summary table: per-task mean and std of bundle noise stds.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln


def collect_per_task_bundles(run_data: dict):
    """Return (task_names, per_task) where per_task[t] = list of dicts.

    Each dict has keys: per_seed (np.ndarray), n_seeds, mean, std. Deduplicated
    by bundle-operator key across the whole run (first occurrence wins).
    """
    task_names = None
    per_task = None
    seen_keys = set()
    for gen in run_data["generations"]:
        for entry in list(gen["population"]) + list(gen["offspring"]):
            ops = entry["operators"]
            key = tuple(sorted((slot, v["name"]) for slot, v in ops.items()))
            if key in seen_keys:
                continue
            rd = entry.get("result_details") or []
            if not rd:
                continue
            if task_names is None:
                task_names = [d["dataset"] for d in rd]
                per_task = [[] for _ in task_names]
            else:
                names = [d["dataset"] for d in rd]
                if names != task_names:
                    continue
            try:
                mat = np.array([d["run_gt_scores"] for d in rd], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            if mat.ndim != 2 or mat.shape[1] < 2:
                continue
            seen_keys.add(key)
            for t in range(mat.shape[0]):
                ps = mat[t]
                per_task[t].append({
                    "per_seed": ps,
                    "n_seeds": int(ps.size),
                    "mean": float(ps.mean()),
                    "std": float(ps.std(ddof=1)),
                })
    return task_names, per_task


def homogeneity_tests(per_seed_list):
    samples = [s for s in per_seed_list if len(s) >= 2]
    if len(samples) < 2:
        return None
    levene = stats.levene(*samples, center="median")
    try:
        bartlett = stats.bartlett(*samples)
    except Exception:
        bartlett = None
    return {"levene": levene, "bartlett": bartlett, "k": len(samples)}


def plot_task(job: str, task_idx: int, task_name: str, bundles, out_path: Path):
    means = np.array([b["mean"] for b in bundles])
    stds = np.array([b["std"] for b in bundles])
    n_seeds = np.array([b["n_seeds"] for b in bundles])
    per_seed_list = [b["per_seed"] for b in bundles]
    test = homogeneity_tests(per_seed_list)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.scatter(means, stds, s=22, alpha=0.6, color="C0", edgecolor="white")
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
                    label="binned mean std (8 quantile bins)")
    ax.axhline(stds.mean(), color="k", linestyle="--", linewidth=1, alpha=0.5,
               label=f"overall mean std = {stds.mean():.3f}")
    ax.set_xlabel("bundle mean score (this task)")
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

    n_seeds_null = int(np.median(n_seeds))
    sigma_null = float(stds.mean())
    df = n_seeds_null - 1
    xs = np.linspace(bin_edges[0], bin_edges[-1], 400)
    if sigma_null > 0 and df > 0:
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
    title = "Distribution of per-bundle noise stds"
    if test is not None:
        title += f"\nLevene p = {test['levene'].pvalue:.2e}"
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"{job} — task {task_idx}: {task_name} — score-noise std per bundle",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {
        "task_idx": task_idx,
        "task_name": task_name,
        "n_bundles": len(bundles),
        "noise_mean": float(stds.mean()),
        "noise_std": float(stds.std(ddof=1)) if len(stds) > 1 else 0.0,
        "noise_median": float(np.median(stds)),
        "noise_min": float(stds.min()),
        "noise_max": float(stds.max()),
        "levene_p": float(test["levene"].pvalue) if test is not None else float("nan"),
    }


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "947961"
    path = Path("runs") / job / "run_data.json"
    data = json.loads(path.read_text())
    task_names, per_task = collect_per_task_bundles(data)
    if not task_names:
        print("No evaluated bundles found")
        return

    out_dir = Path("plots") / f"{job}_task_noise"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for t, name in enumerate(task_names):
        bundles = per_task[t]
        if not bundles:
            print(f"task {t} ({name}): no bundles, skipping")
            continue
        out_path = out_dir / f"task_{t}_operator_noise.png"
        summary = plot_task(job, t, name, bundles, out_path)
        summaries.append(summary)

    # Print summary table.
    print(f"\n{'idx':>3}  {'task':<28}  {'n':>4}  {'mean(σ)':>8}  "
          f"{'std(σ)':>8}  {'median':>8}  {'min':>7}  {'max':>7}  {'Levene p':>10}")
    for s in summaries:
        print(f"{s['task_idx']:>3}  {s['task_name']:<28}  {s['n_bundles']:>4}  "
              f"{s['noise_mean']:>8.4f}  {s['noise_std']:>8.4f}  "
              f"{s['noise_median']:>8.4f}  {s['noise_min']:>7.4f}  "
              f"{s['noise_max']:>7.4f}  {s['levene_p']:>10.2e}")

    overall_mean = np.mean([s["noise_mean"] for s in summaries])
    overall_std = np.std([s["noise_mean"] for s in summaries], ddof=1)
    print(f"\nAcross tasks: per-task mean(σ) — mean={overall_mean:.4f}, "
          f"std={overall_std:.4f}, min={min(s['noise_mean'] for s in summaries):.4f}, "
          f"max={max(s['noise_mean'] for s in summaries):.4f}")
    print(f"\nWrote {len(summaries)} plots to {out_dir}/")


if __name__ == "__main__":
    main()
