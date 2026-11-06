#!/usr/bin/env python3
"""Append noise-free PySR final-evaluation results and logged search durations."""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_fullsr_gt_r2 import first_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "final_eval_summary.json").read_text())
    comparison = json.loads(args.comparison.read_text())
    model = {"model": f"{args.run.name} final (validation-selected)", "engine": "PySR",
             "selection": summary["method"], "target_noise": 0.0, "matches": []}
    for path in sorted((args.run / "final_eval/slurm_pysr").glob("*/tasks.json")):
        task = first_task(path)
        if task["target_noise"] != 0:
            continue
        tasks = json.loads(path.read_text())
        records = json.loads(path.with_name("combined.json").read_text())
        assert len(tasks) == len(records) == 200
        fields = ["custom_mutation_code", "custom_selection_code", "custom_survival_code", "custom_loss_code"]
        for t in tasks:
            assert t["target_noise"] == 0
            assert all(t[f] == task[f] for f in fields)
        for op in summary["operators"]:
            assert op["name"] in str(task[f"custom_{op['type']}_code"])
        assert {(r["dataset_name"], r["run_index"]) for r in records} == {
            (t["dataset_name"], t["run_index"]) for t in tasks}
        datasets = sorted({r["dataset_name"] for r in records})
        kind = next(m["kind"] for m in comparison[0]["matches"] if m["datasets"] == datasets)
        assert all(sum(r["dataset_name"] == d for r in records) == 10 for d in datasets)
        times = {}
        for log in (path.parent / "logs").glob("*.out"):
            text = log.read_text()
            index = re.search(r"PySR Worker starting: task=(\d+)", text)
            duration = re.search(r"PySR search complete.*?in ([0-9.]+)s", text)
            if index and duration:
                times[int(index[1])] = float(duration[1])
        assert set(times) == set(range(200))
        gt = statistics.mean(r["gt_match_score"] for r in records)
        r2 = statistics.mean(r["r2_score"] for r in records)
        split = "barely_unsolvable" if kind == "train_reeval" else "barely_unsolvable_val2"
        expected = summary["multi_noise"][split]["per_noise_level"]["0.0"]
        assert abs(gt - expected["avg_gt"]) < 1e-10
        assert abs(r2 - expected["avg_r2"]) < 1e-10
        model["matches"].append({
            "kind": kind, "source": str(path.parent), "n": 200, "datasets": datasets,
            "gt": gt, "frontier_r2": statistics.mean(r["r2_frontier_score"] for r in records),
            "best_equation_r2": r2, "errors": sum(bool(r.get("error")) for r in records),
            "timing": {"search_n": 200, "search_mean_s": statistics.mean(times.values()),
                       "search_median_s": statistics.median(times.values()),
                       "task_total_mean_s": statistics.mean(r["runtime_seconds"] for r in records)},
            "settings": {"timeout": task["pysr_kwargs"].get("timeout_in_seconds"),
                         "max_evals": task["pysr_kwargs"].get("max_evals"),
                         "seed": task["seed"], "data_seed": task["data_seed"],
                         "target_noise": 0.0, "run_indices": sorted({t["run_index"] for t in tasks})},
        })
    assert sorted(m["kind"] for m in model["matches"]) == ["train_reeval", "val"]
    comparison = [m for m in comparison if m["model"] != model["model"]] + [model]
    args.comparison.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()
