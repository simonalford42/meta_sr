"""Plot per-generation evolution metrics for a bundle-evolution run.

Reads `runs/<RUN_ID>/run_data.json` and produces a 4-panel summary:
  * Population mean score
  * Offspring mean score
  * Best-in-population score
  * Per-task best, averaged across tasks (population coverage)

Usage:
    python scripts/plot_run_metrics.py 947961
    python scripts/plot_run_metrics.py 947961 --output out/947961_metrics.png
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_run(run_id: str):
    path = REPO_ROOT / "runs" / run_id / "run_data.json"
    with open(path) as f:
        return json.load(f)


def per_gen_metrics(data):
    gens = data["generations"]
    rows = []
    for g in gens:
        pop = g.get("population", []) or []
        off = g.get("offspring", []) or []

        pop_scores = [b["score"] for b in pop if b.get("score") is not None]
        off_scores = [b["score"] for b in off if b.get("score") is not None]

        score_vectors = [b["score_vector"] for b in pop if b.get("score_vector")]
        if score_vectors:
            n_tasks = len(score_vectors[0])
            per_task_best = [
                max(v[t] for v in score_vectors) for t in range(n_tasks)
            ]
            pt_avg = statistics.mean(per_task_best)
        else:
            pt_avg = float("nan")

        rows.append({
            "generation": g["generation"],
            "avg_pop": statistics.mean(pop_scores) if pop_scores else float("nan"),
            "avg_off": statistics.mean(off_scores) if off_scores else float("nan"),
            "best": g.get("best_score", float("nan")),
            "pt_avg": pt_avg,
        })
    return rows


def plot(run_id: str, output: Path):
    data = load_run(run_id)
    rows = per_gen_metrics(data)

    cfg = data.get("config", {})
    n_pop = cfg.get("population_size", "?")
    n_off = cfg.get("n_offspring", "?")
    n_tasks = cfg.get("n_datasets", "?")
    n_runs = cfg.get("n_runs", "?")
    op_types = "+".join(cfg.get("operator_types", []))
    baseline = data.get("baseline", {}).get("avg_r2")

    gens = [r["generation"] for r in rows]
    avg_pop = [r["avg_pop"] for r in rows]
    avg_off = [r["avg_off"] for r in rows]
    best = [r["best"] for r in rows]
    pt_avg = [r["pt_avg"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    suptitle = (
        f"Run {run_id} — bundle evolution per-generation metrics\n"
        f"operators: {op_types}   |   pop={n_pop}   offspring/gen={n_off}   "
        f"tasks={n_tasks}   runs/task={n_runs}"
    )
    fig.suptitle(suptitle, fontsize=12)

    panels = [
        (axes[0, 0], best, "tab:green",
         "Best-in-population score",
         "GT match rate of the top bundle (monotone non-decreasing)"),
        (axes[0, 1], avg_pop, "tab:blue",
         "Population mean score",
         f"mean GT match rate across the {n_pop} surviving bundles"),
        (axes[1, 0], avg_off, "tab:orange",
         "Offspring mean score",
         f"mean GT match rate of the {n_off} newly proposed bundles"),
        (axes[1, 1], pt_avg, "tab:red",
         "Per-task best, averaged across tasks",
         "for each task, best score in the population, then averaged\n"
         "(population coverage / diversity, not a single bundle)"),
    ]

    for ax, ys, color, title, subtitle in panels:
        ax.plot(gens, ys, marker="o", markersize=3, color=color, linewidth=1.2)
        if baseline is not None:
            ax.axhline(
                baseline, color="gray", linestyle="--", linewidth=0.8,
                label=f"baseline avg = {baseline:.3f}",
            )
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc="left")
        ax.set_xlabel("Generation", fontsize=9)
        ax.set_ylabel("GT match rate (avg over tasks)", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    print(f"wrote {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    out = args.output or REPO_ROOT / "out" / f"{args.run_id}_metrics.png"
    plot(args.run_id, out)


if __name__ == "__main__":
    main()
