#!/usr/bin/env python3
"""Extract saved first 90-second restarts for independent recovery review.

No searches or API calls. Outputs use the standard evaluation layout so both
review_srbench2_frontiers.py and inspect_srbench_results.py can read them.
Actual restart runtimes may slightly exceed their nominal 90-second limit.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srbench_results_io import expected_keys


def extract(source: Path, output: Path) -> None:
    manifest = json.loads((source / "manifest.json").read_text())
    raw = (source / "srbench_full_results.json").read_bytes()
    payload = json.loads(raw)
    if manifest.get("srbench_edition") != 2025:
        raise ValueError("Expected an SRBench2 evaluation")
    if set(payload["results"]) != set(expected_keys(manifest, payload["results"])):
        raise ValueError("Source does not contain the complete expected trial grid")
    results, tasks, restarts = {}, [], []
    for key, row in sorted(payload["results"].items()):
        first = copy.deepcopy(row["portfolio"]["restarts"][0])
        if (first.get("restart_index") != 0 or first.get("soft_timeout_seconds") != 90
                or first.get("error") or not first.get("pareto_frontier")):
            raise ValueError(f"{key}: missing successful first 90-second restart")
        elapsed = float(first["search_runtime_seconds"])
        first["source_portfolio_seed"] = row["seed"]
        restarts.append(first)
        tasks.append({"dataset_name": row["dataset"], "seed": row["seed"],
                      "run_index": 0, "target_noise": row["noise"]})
        result = {field: row[field] for field in
                  ("dataset", "family", "seed", "run_index", "noise", "config_id")}
        result.update(present=True, error=None, solved=False, gt_match_score=None,
                      test_r2=first.get("r2_score"),
                      runtime_seconds=first["runtime_seconds"], solve_time=elapsed,
                      solve_time_source="first_restart_search_runtime",
                      best_equation=first.get("best_equation"), best_loss=first.get("best_loss"),
                      pareto_frontier=first["pareto_frontier"],
                      execution_trace=first.get("execution_trace"),
                      source_restart_seed=first["seed"])
        results[key] = result

    manifest.update(timeout_in_seconds=90, timeout_source="saved_first_restart",
                    n_runs=len(manifest["seeds"]), merge_run_frontiers=False,
                    evaluation_types=["ground_truth"],
                    batches=[{"batch_dir": "first_restart", "n_tasks": len(tasks)}])
    manifest.pop("black_box", None)
    manifest.pop("serial_restart_portfolio", None)
    manifest["derived_from"] = {
        "run_dir": str(source.resolve()), "results_sha256": hashlib.sha256(raw).hexdigest(),
        "restart_index": 0, "nominal_search_budget_seconds": 90,
        "actual_search_seconds_range": [min(r["search_runtime_seconds"] for r in restarts),
                                        max(r["search_runtime_seconds"] for r in restarts)],
        "note": "Restart-end frontier; actual search can overshoot the nominal limit.",
    }
    # Fail before overwriting any existing extraction or review state.
    output.mkdir(parents=True, exist_ok=False)

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")

    write(output / "first_restart/tasks.json", tasks)
    for index, first in enumerate(restarts):
        write(output / "first_restart/results" / f"task_{index:06d}.json", first)
    write(output / "srbench_full_results.json", {
        "meta": {"mode": manifest["mode"], "derived_from": manifest["derived_from"]},
        "results": results,
    })
    write(output / "manifest.json", manifest)
    print(f"Extracted {len(results)} first-restart frontiers to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.source, args.output)
