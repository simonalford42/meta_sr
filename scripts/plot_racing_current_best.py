"""Reconstruct a 'current best of all time' plot for a --racing run.

Unlike the monotonic `eval_running_best` logged to wandb, this tracks the
maximum over each candidate's *most recent* score, so if the current leader's
score drops as more seeds are accumulated, the line can go down.
"""
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

EVAL_RE = re.compile(r"^\s{2}(?P<name>.+?): Avg (?P<score>-?\d+\.\d+)\s*\(seeds=(?P<seeds>\d+)\)")


def parse(path: Path, min_seeds_filter: int = 10):
    latest, latest_seeds = {}, {}
    xs, ys, running_max, leaders = [], [], [], []
    prev_leader_scores = []
    # filtered view: only candidates with seeds >= min_seeds_filter
    ys_f, leaders_f, prev_leader_scores_f = [], [], []
    cur_max = float("-inf")
    idx = 0
    prev_leader = None
    prev_leader_f = None
    for line in path.read_text().splitlines():
        m = EVAL_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        score = float(m.group("score"))
        seeds = int(m.group("seeds"))
        latest[name] = score
        latest_seeds[name] = seeds
        idx += 1
        leader = max(latest, key=lambda k: latest[k])
        cur_max = max(cur_max, score)
        xs.append(idx)
        ys.append(latest[leader])
        running_max.append(cur_max)
        leaders.append(leader)
        prev_leader_scores.append(float("nan") if prev_leader is None else latest[prev_leader])
        prev_leader = leader

        # filtered
        eligible = [k for k, s in latest_seeds.items() if s >= min_seeds_filter]
        if eligible:
            leader_f = max(eligible, key=lambda k: latest[k])
            ys_f.append(latest[leader_f])
            leaders_f.append(leader_f)
            prev_leader_scores_f.append(
                float("nan") if prev_leader_f is None or prev_leader_f not in latest
                else latest[prev_leader_f]
            )
            prev_leader_f = leader_f
        else:
            ys_f.append(float("nan"))
            leaders_f.append(None)
            prev_leader_scores_f.append(float("nan"))
    return (np.array(xs), np.array(ys), np.array(running_max),
            leaders, np.array(prev_leader_scores),
            np.array(ys_f), leaders_f, np.array(prev_leader_scores_f),
            latest)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/499255.out")
    min_seeds = 10
    (xs, ys, running_max, leaders, prev_leader_scores,
     ys_f, leaders_f, prev_leader_scores_f, latest) = parse(path, min_seeds_filter=min_seeds)
    print(f"Parsed {len(xs)} eval updates from {len(latest)} unique candidates")
    print(f"Final current-best: {ys[-1]:.4f}  |  monotonic running best: {running_max[-1]:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    # --- Top: plain current-best + monotonic running best ---
    ax = axes[0]
    ax.plot(xs, running_max, color="gray", linewidth=1.2, label="monotonic running best")
    ax.plot(xs, ys, color="C0", linewidth=1.5, label="current best (latest score)")
    ax.set_title("Current best vs monotonic running best")
    ax.set_ylabel("score"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    def draw_segments(ax, xs, ys, leaders, prev_leader_scores, title):
        ax.plot(xs, running_max, color="lightgray", linewidth=1.0, label="monotonic running best")
        # valid indices (skip NaN ys, which mean no eligible candidate yet)
        valid = ~np.isnan(ys)
        # find change points considering only valid steps
        changes = [i for i in range(1, len(leaders))
                   if valid[i] and valid[i-1] and leaders[i] != leaders[i-1]]
        # first valid index is effectively a segment start
        first_valid = int(np.argmax(valid)) if valid.any() else 0
        colors = plt.get_cmap("tab20").colors
        seg_starts = [first_valid] + changes
        seg_ends = changes + [len(xs)]
        for k, (s, e) in enumerate(zip(seg_starts, seg_ends)):
            ax.plot(xs[s:e], ys[s:e], color=colors[k % len(colors)], linewidth=1.8)
        for ci in changes:
            x0, y0 = xs[ci - 1], ys[ci - 1]
            x1, y1 = xs[ci], prev_leader_scores[ci]
            if np.isnan(y1):
                continue
            seg_idx = seg_starts.index(ci) - 1
            c = colors[seg_idx % len(colors)]
            ax.plot([x0, x1], [y0, y1], color=c, linewidth=1.0, linestyle="--", alpha=0.8)
            ax.scatter([x1], [y1], s=22, facecolor="white", edgecolor=c, linewidth=1.2, zorder=4)
        ax.set_title(f"{title} ({len(changes)} leader changes; dashed = dethroned leader's new score)")
        ax.set_ylabel("score"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    draw_segments(axes[1], xs, ys, leaders, prev_leader_scores,
                  "Segment color = current leader")
    draw_segments(axes[2], xs, ys_f, leaders_f, prev_leader_scores_f,
                  f"Filtered: only candidates with seeds ≥ {min_seeds}")
    axes[2].set_xlabel("eval update index")

    fig.suptitle(f"{path.name} — racing", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path("plots") / f"{path.stem}_racing_current_best.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
