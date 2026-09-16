#!/usr/bin/env python3
"""Append a pinned PySR generation and completed reevaluation pair to a comparison."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from operator_types import OperatorBundle
from evolution_helpers import code_loc
from parallel_eval_pysr import select_run_scores
from scripts.compare_fullsr_gt_r2 import first_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--generation", required=True, type=int)
    parser.add_argument("--reeval-generation", required=True, type=int)
    parser.add_argument("--comparison", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads((args.run / "run_data.json").read_text())
    snapshot = next(g for g in data["generations"] if g["generation"] == args.generation)
    bundle = max(snapshot["population"], key=lambda b: b["score"])
    older = next(g for g in data["generations"] if g["generation"] == args.reeval_generation)
    older_bundle = max(older["population"], key=lambda b: b["score"])
    config = OperatorBundle.from_dict(bundle).to_pysr_config(data["config"]["pysr_kwargs"])
    old_config = OperatorBundle.from_dict(older_bundle).to_pysr_config(data["config"]["pysr_kwargs"])
    fields = ["custom_mutation_code", "custom_selection_code", "custom_survival_code",
              "custom_loss_code", "mutation_weights", "allow_custom_mutations"]
    for field in fields:
        assert getattr(config, field) == getattr(old_config, field), field
    model = {"model": f"{args.run.name} gen {args.generation}", "engine": "PySR",
             "snapshot_latest_generation": data["generations"][-1]["generation"],
             "run_finished_at_snapshot": data.get("end_time") is not None,
             "reeval_generations": {k: args.reeval_generation for k in ("val", "train_reeval")},
             "live_score": bundle["score"],
             "loc": sum(code_loc(op["code"]) for op in bundle["operators"].values() if op),
             "matches": []}
    for path in sorted((args.run / "slurm_pysr").glob("*/tasks.json")):
        task = first_task(path)
        if task["run_index"] not in (100000 + 10 * args.reeval_generation, 200000 + 10 * args.reeval_generation):
            continue
        tasks = json.loads(path.read_text())
        records = json.loads(path.with_name("combined.json").read_text())
        assert len(tasks) == len(records) == 200
        for t in tasks:
            for field in fields:
                assert t[field] == getattr(config, field), (path, field)
        assert {(r["dataset_name"], r["run_index"]) for r in records} == {
            (t["dataset_name"], t["run_index"]) for t in tasks}
        datasets = sorted({r["dataset_name"] for r in records})
        assert len(datasets) == 20
        assert all(sum(r["dataset_name"] == d for r in records) == 10 for d in datasets)
        kind = "val" if task["run_index"] >= 200000 else "train_reeval"
        gt = [r["gt_match_score"] or 0.0 for r in records]
        r2 = [r["r2_score"] if r["r2_score"] is not None else -1.0 for r in records]
        r2c = [r["r2_frontier_score"] if r.get("r2_frontier_score") is not None else v for r, v in zip(records, r2)]
        search = [r["search_runtime_seconds"] for r in records]
        assert all(t is not None and t > 0 for t in search)
        model["matches"].append({
            "kind": kind, "source": str(path.parent), "n": 200, "datasets": datasets,
            "gt": statistics.mean(gt), "frontier_r2": statistics.mean(r2c),
            "best_equation_r2": statistics.mean(r2),
            "combined_score": statistics.mean(select_run_scores(r2, gt, r2c, task["fitness_metric"])),
            "errors": sum(bool(r.get("error")) for r in records),
            "timing": {"search_n": len(search), "search_mean_s": statistics.mean(search),
                       "search_median_s": statistics.median(search),
                       "task_total_mean_s": statistics.mean(r["runtime_seconds"] for r in records)},
            "settings": {"fitness_metric": task["fitness_metric"],
                         "timeout": task["pysr_kwargs"].get("timeout_in_seconds"),
                         "max_evals": task["pysr_kwargs"].get("max_evals"),
                         "seed": task["seed"], "data_seed": task["data_seed"],
                         "run_indices": sorted({t["run_index"] for t in tasks})},
        })
    assert sorted(m["kind"] for m in model["matches"]) == ["train_reeval", "val"]
    comparison = json.loads(args.comparison.read_text())
    comparison = [m for m in comparison if not m["model"].startswith(args.run.name + " gen ")]
    for match in model["matches"]:
        reference = next(m for m in comparison[0]["matches"] if m["kind"] == match["kind"])
        assert match["datasets"] == reference["datasets"]
    comparison.append(model)
    args.comparison.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()
