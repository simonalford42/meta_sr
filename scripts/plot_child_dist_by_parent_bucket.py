"""Overlay child-score distributions, one per parent-score bucket.

Bins parents into N quantile buckets and plots a KDE of the child scores in
each bucket. A dashed vertical line at the bucket-mean parent score (same
color) makes it easy to read off the conditional shift E[child | parent].

Usage: python scripts/plot_child_dist_by_parent_bucket.py [job] [n_bins]
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent))
from plot_parent_child_score import collect_pairs, collect_pairs_from_out


def quantile_buckets(parent_scores, n_bins):
    """Return list of (lo, hi, mask) using quantile edges."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(parent_scores, qs)
    edges[-1] += 1e-9
    out = []
    for i in range(n_bins):
        mask = (parent_scores >= edges[i]) & (parent_scores < edges[i + 1])
        out.append((edges[i], edges[i + 1], mask))
    return out


def plot_buckets(rows, job: str, out_dir: Path, n_bins: int = 8):
    ps = np.array([r[3] for r in rows], dtype=float)
    cs = np.array([r[4] for r in rows], dtype=float)

    buckets = quantile_buckets(ps, n_bins)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Common evaluation grid for KDE across all panels.
    lo = min(ps.min(), cs.min()) - 0.02
    hi = max(ps.max(), cs.max()) + 0.02
    xs = np.linspace(lo, hi, 400)

    # Panel 1: KDE of child score per parent bucket.
    ax = axes[0]
    for i, (b_lo, b_hi, mask) in enumerate(buckets):
        n = int(mask.sum())
        if n < 3:
            continue
        color = cmap(i / max(1, n_bins - 1))
        c_in_bucket = cs[mask]
        p_mean = ps[mask].mean()
        try:
            kde = gaussian_kde(c_in_bucket)
            ax.plot(xs, kde(xs), color=color, linewidth=2,
                    label=f"parent ∈ [{b_lo:.3f}, {b_hi:.3f}]  "
                          f"⟨p⟩={p_mean:.3f}, n={n}")
        except np.linalg.LinAlgError:
            # Degenerate (e.g., all identical child scores) — show as a spike.
            ax.axvline(c_in_bucket[0], color=color, linewidth=2,
                       label=f"parent ∈ [{b_lo:.3f}, {b_hi:.3f}]  "
                             f"n={n}, c={c_in_bucket[0]:.3f}")
        # Dashed vertical at parent-mean for this bucket, same color.
        ax.axvline(p_mean, color=color, linestyle="--", linewidth=1.2, alpha=0.8)

    ax.set_xlabel("child score")
    ax.set_ylabel("density (KDE)")
    ax.set_title(f"P(child score | parent bucket)  —  {n_bins} quantile buckets\n"
                 "dashed vertical = bucket-mean parent score")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.3)
    ax.set_xlim(lo, hi)

    # Panel 2: same buckets as histograms (stacked subplots in one axis with
    # vertical offset for readability — ridgeline-style).
    ax = axes[1]
    n_active = sum(1 for _, _, mask in buckets if mask.sum() >= 3)
    bins = np.linspace(lo, hi, 40)
    bin_width = bins[1] - bins[0]
    offset_step = 1.0  # density-units between bucket ridges
    row_idx = 0
    yticks, ylabels = [], []
    for i, (b_lo, b_hi, mask) in enumerate(buckets):
        n = int(mask.sum())
        if n < 3:
            continue
        color = cmap(i / max(1, n_bins - 1))
        c_in_bucket = cs[mask]
        p_mean = ps[mask].mean()
        c_mean = c_in_bucket.mean()
        offset = (n_active - 1 - row_idx) * offset_step
        # Filled histogram normalized to density, then offset vertically.
        counts, edges = np.histogram(c_in_bucket, bins=bins, density=True)
        ax.fill_between(0.5 * (edges[:-1] + edges[1:]), offset,
                        offset + counts, color=color, alpha=0.55,
                        step="mid")
        ax.plot(0.5 * (edges[:-1] + edges[1:]), offset + counts,
                color=color, linewidth=1.5, drawstyle="steps-mid")
        # Reference markers: parent-mean (dashed) and child-mean (solid tick).
        ax.vlines(p_mean, offset, offset + offset_step * 0.9,
                  color=color, linestyle="--", linewidth=1.2)
        ax.vlines(c_mean, offset, offset + offset_step * 0.5,
                  color=color, linestyle="-", linewidth=2)
        yticks.append(offset + offset_step * 0.3)
        ylabels.append(f"p∈[{b_lo:.2f},{b_hi:.2f}] n={n}")
        row_idx += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("child score")
    ax.set_title("Ridgeline view  —  dashed = parent-mean, solid = child-mean")
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(lo, hi)

    fig.suptitle(
        f"{job} — child-score distribution conditioned on parent-score bucket "
        f"(n={len(rows)} pairs)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / f"{job}_child_dist_by_parent_bucket.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "947961"
    n_bins = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    path_arg = Path(job)
    if path_arg.suffix == ".out" or path_arg.is_file():
        path = path_arg if path_arg.exists() else Path("out") / path_arg.name
        job_name = path.stem
        rows = collect_pairs_from_out(path)
    else:
        job_name = job
        path = Path("runs") / job / "run_data.json"
        rows = collect_pairs(json.loads(path.read_text()))

    if not rows:
        print(f"No parent/child pairs found for {job}")
        return
    print(f"Loaded {len(rows)} parent/child pairs")

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    plot_buckets(rows, job_name, out_dir, n_bins=n_bins)


if __name__ == "__main__":
    main()
