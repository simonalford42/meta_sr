#!/usr/bin/env python3
"""Decompose saved FullSR reevaluations; does not submit evaluation jobs.

Matches task policy code to each generation's best model by rendered-code hash,
then verifies reconstructed combined scores against the saved reevaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skeleton_operator_types import SkeletonBundle, render_sr_module_body
from evolution_helpers import code_loc
from parallel_eval_pysr import select_run_scores


def content_key(body):
    return hashlib.sha1(body.encode()).hexdigest()[:16]


def first_task(path):
    """Read only the first task from potentially large task arrays."""
    with path.open() as handle:
        text = ""
        while chunk := handle.read(131072):
            text += chunk
            try:
                return json.JSONDecoder().raw_decode(text.lstrip()[1:].lstrip())[0]
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Cannot read first task: {path}")


def targets():
    result = []
    for run, gens in [("229869", [30, 60, 90]), ("243303", [30])]:
        data = json.loads((ROOT / "runs" / run / "run_data.json").read_text())
        for gen in gens:
            snapshot = next(g for g in data["generations"] if g["generation"] == gen)
            bundle = max(snapshot["population"], key=lambda b: b["score"])
            body = render_sr_module_body(SkeletonBundle.from_dict(bundle))
            key = content_key(body)
            loc = code_loc(bundle["raw_module_body"]) if bundle.get("raw_module_body") else sum(
                code_loc(f["code"]) for f in bundle["functions"].values())
            result.append({
                "model": f"{'150815' if run == '229869' else run} gen {gen}",
                "content_key": key, "loc": loc, "live_score": bundle["score"],
                "saved": data["val_results"][key], "matches": [],
            })
        del data
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save the comparison as JSON")
    args = parser.parse_args()
    models = targets()
    by_key = {m["content_key"]: m for m in models}
    for run in ["150815", "150815-simplify-30-best2", "229869", "243303"]:
        for path in sorted((ROOT / "runs" / run / "slurm_fullsr").glob("eval_*/tasks.json")):
            task = first_task(path)
            key = content_key(task.get("policy_module_code") or "")
            if key not in by_key or not path.with_name("combined.json").exists():
                continue
            # Evolution batches use three runs; background reevaluations use ten.
            tasks = json.loads(path.read_text())
            if len(tasks) != 200 or any(content_key(t.get("policy_module_code") or "") != key for t in tasks):
                continue
            records = json.loads(path.with_name("combined.json").read_text())
            assert len(records) == len(tasks)
            assert {(r["dataset_name"], r["run_index"]) for r in records} == {
                (t["dataset_name"], t["run_index"]) for t in tasks}
            datasets = sorted({r["dataset_name"] for r in records})
            assert len(datasets) == 20
            assert all(sum(r["dataset_name"] == d for r in records) == 10 for d in datasets)
            kind = "train_reeval" if task["run_index"] >= 100000 else "val"
            gt = [r["gt_match_score"] or 0.0 for r in records]
            r2 = [r["r2_score"] if r["r2_score"] is not None else -1.0 for r in records]
            r2c = [r["r2_frontier_score"] if r.get("r2_frontier_score") is not None else v
                   for r, v in zip(records, r2)]
            score = statistics.mean(select_run_scores(r2, gt, r2c, task["fitness_metric"]))
            expected = by_key[key]["saved"].get(kind, {}).get("avg_score")
            if expected is None or abs(score - expected) > 1e-10:
                continue
            search_times = {}
            for log in sorted((path.parent / "logs").glob("*.out")):
                text = log.read_text()
                index = re.search(r"FullSR Worker starting: task=(\d+)", text)
                duration = re.search(r"Search finished in ([0-9.]+)s", text)
                if index and duration:
                    search_times[int(index[1])] = float(duration[1])
            # Cache-only batches have no worker logs; report no search timing
            # for those copies rather than mixing total task and fit timers.
            if search_times:
                assert set(search_times) == set(range(len(tasks))), path
            timing = {
                "search_n": len(search_times),
                "search_mean_s": statistics.mean(search_times.values()) if search_times else None,
                "search_median_s": statistics.median(search_times.values()) if search_times else None,
                "task_total_mean_s": statistics.mean(r["runtime_seconds"] for r in records),
                "mean_n_evals": statistics.mean(r["n_evals"] for r in records if r.get("n_evals") is not None),
            }
            by_key[key]["matches"].append({
                "kind": kind, "source": str(path.parent.relative_to(ROOT)),
                "n": len(records), "datasets": datasets,
                "gt": statistics.mean(gt), "frontier_r2": statistics.mean(r2c),
                "clipped_frontier_r2": statistics.mean(max(v, 0) for v in r2c),
                "best_equation_r2": statistics.mean(r2), "combined_score": score,
                "errors": sum(bool(r.get("error")) for r in records),
                "timing": timing,
                "settings": {"fitness_metric": task["fitness_metric"],
                             "timeout": task["engine_kwargs"].get("timeout_in_seconds"),
                             "max_evals": task["engine_kwargs"].get("max_evals"),
                             "wall_limit": task.get("wall_limit"),
                             "seed": task["seed"], "data_seed": task["data_seed"],
                             "run_indices": sorted({t["run_index"] for t in tasks})},
            })
    for model in models:
        assert {m["kind"] for m in model["matches"]} == {"train_reeval", "val"}, model["model"]
        model["reeval_generations"] = {k: model["saved"][k]["gen_submitted"] for k in ("train_reeval", "val")}
        del model["saved"]
    for kind in ("train_reeval", "val"):
        dataset_sets = {tuple(m["datasets"]) for model in models for m in model["matches"] if m["kind"] == kind}
        assert len(dataset_sets) == 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(models, indent=2) + "\n")
    for model in models:
        print(model["model"], "LOC", model["loc"])
        for kind in ("train_reeval", "val"):
            matches = [m for m in model["matches"] if m["kind"] == kind]
            selected = next((m for m in matches if m["timing"]["search_n"] == 200), matches[0])
            print(kind, json.dumps({k: selected[k] for k in (
                "gt", "frontier_r2", "best_equation_r2", "combined_score", "timing", "source")}))


if __name__ == "__main__":
    main()
