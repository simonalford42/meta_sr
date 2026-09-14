"""Reconstruct periodic GT diagnostics from saved tasks; never launches evaluations.

Usage: python figures/plot_finetuning_gt_by_noise.py --run 140185
Generation is the generation submitted, not asynchronous completion time.
Errors count as zero, matching parallel_eval_pysr.py. Each point averages
20 datasets x 3 fresh seeds per noise level for run 140185.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="140185")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    log = (root / "out" / f"{args.run}.out").read_text()
    n_runs = int(re.search(r"--val-n-runs (\d+)", log)[1])
    batches = sorted(set(re.findall(
        r"PySR SLURM eval: \d+ tasks in batch (eval_\d+) "
        rf"\(1 configs x \d+ datasets x {n_runs} runs x 4 noise levels\)", log
    )))
    logged = {}
    for kind, gen, score in re.findall(
        r"\[(val eval|train reeval)\] gen (\d+).*?: "
        r"(?:avg|reeval) GT match rate=([\d.]+)", log
    ):
        logged[(kind, int(gen))] = float(score)
    rows = []
    for batch in batches:
        directory = root / "runs" / args.run / "slurm_pysr" / batch
        tasks = json.loads((directory / "tasks.json").read_text())
        first_index = min(t["run_index"] for t in tasks)
        if 100_000 <= first_index < 200_000:
            kind, offset = "train reeval", 100_000
        elif 200_000 <= first_index < 300_000:
            kind, offset = "val eval", 200_000
        else:
            continue
        gen, remainder = divmod(first_index - offset, n_runs)
        assert remainder == 0 and (kind, gen) in logged, (batch, first_index)
        results = json.loads((directory / "combined.json").read_text())
        assert len(tasks) == len(results), batch
        grouped = defaultdict(list)
        errors = defaultdict(int)
        seen = set()
        for task_index, (task, result) in enumerate(zip(tasks, results)):
            # collect_task_results preserves task order but missing-file
            # placeholders have dataset_name='unknown' and run_index=0.
            missing = result.get("error") == f"Result file missing for task {task_index}"
            if not missing:
                for key in ("dataset_name", "run_index", "config_id"):
                    assert task[key] == result[key], (batch, key)
            noise = task["target_noise"]
            identity = (task["dataset_name"], task["run_index"], noise)
            assert identity not in seen, (batch, identity)
            seen.add(identity)
            score = result.get("gt_match_score")
            failed = bool(result.get("error"))
            if failed or score is None or not math.isfinite(score):
                score = 0.0
            grouped[noise].append(score)
            errors[noise] += failed
        assert set(grouped) == {0, 0.1, 0.01, 0.001}, batch
        assert len({len(v) for v in grouped.values()}) == 1, batch
        overall = sum(map(sum, grouped.values())) / len(tasks)
        assert abs(overall - logged[(kind, gen)]) < 0.000051, (batch, overall)
        for noise, scores in grouped.items():
            rows.append(dict(kind=kind, generation=gen, noise=noise,
                             avg_gt=sum(scores) / len(scores), n_tasks=len(scores),
                             n_errors=errors[noise], batch=batch))
    assert {(r["kind"], r["generation"]) for r in rows} == set(logged)
    assert len(rows) == len(logged) * 4
    rows.sort(key=lambda r: (r["kind"], r["generation"], r["noise"]))
    output = root / "figures" / f"{args.run}_gt_by_noise"
    output.mkdir(exist_ok=True)
    with (output / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    colors = ["#333333", "#D55E00", "#0072B2", "#009E73"]
    for kind, title, filename in [
        ("train reeval", "Train reevaluation GT", "train_reeval_gt"),
        ("val eval", "Validation average GT", "val_avg_gt"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
        for noise, color in zip([0, 0.1, 0.01, 0.001], colors):
            series = [r for r in rows if r["kind"] == kind and r["noise"] == noise]
            ax.plot([r["generation"] for r in series],
                    [r["avg_gt"] for r in series], color=color,
                    marker="o", markersize=4, linewidth=1.7, label=f"Noise {noise:g}")
            early = [r["avg_gt"] for r in series if r["generation"] <= 3]
            late = [r["avg_gt"] for r in series if r["generation"] >= 17]
            print(f"{kind}, noise={noise:g}: gen0={series[0]['avg_gt']:.4f}, "
                  f"gen20={series[-1]['avg_gt']:.4f}; "
                  f"gen0–3={sum(early)/len(early):.4f}, "
                  f"gen17–20={sum(late)/len(late):.4f}")
        ax.set(xlabel="Fine-tuning generation (submitted)",
               ylabel="Ground-truth match rate", ylim=(0.4, 0.9), xlim=(-0.3, 20.3))
        ax.set_title(f"Run {args.run} · {title}\n"
                     "Generation 0: checkpoint from 709715 · 20 datasets × 3 fresh seeds / noise",
                     fontsize=11, pad=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncol=4, loc="upper right", frameon=False)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{filename}.{extension}", dpi=180)
        plt.close(fig)
    print(f"Validated {len(logged)} diagnostics against the log; saved to {output}")


if __name__ == "__main__":
    main()
