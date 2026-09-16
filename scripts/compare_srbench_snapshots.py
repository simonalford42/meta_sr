"""Summarize seed variation and score cumulative snapshot recovery offline.

Uses the same three-second symbolic checker as summarize_frontier_snapshot_times.
Deduplicates equation checks across methods, noise levels, and seeds per dataset.
Checkpoints each dataset; no searches or Slurm jobs are submitted.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import srbench_results_io as srio

RUNS = {
    "old_baseline": "srbench_gt_baseline_90s",
    "old_evolved": "709715/srbench_gt_90s",
    "baseline": "srbench_gt_baseline_9-16_10seed_allnoise_90s_snap10",
    "evolved": "709715-srbench_gt_9-16_10seed_allnoise_90s_snap10",
}
PREVIOUS = [
    "srbench_gt_baseline_9-15_1seed_90s_snap10",
    "709715-srbench_gt_9-14_1seed_90s_snap10",
]


def seed_stats(rows, field="solved"):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(bool(row[field]))
    rates = {seed: 100 * statistics.mean(values) for seed, values in sorted(by_seed.items())}
    return {"mean": statistics.mean(rates.values()), "sd": statistics.stdev(rates.values()),
            "se": statistics.stdev(rates.values()) / len(rates) ** .5,
            "seed_rates": rates, "n": len(rows)}


def score_dataset(item):
    dataset, records, cache, out = item
    path = Path(out) / f"{dataset}.json"
    if path.exists():
        return json.loads(path.read_text())
    from evaluation import check_pysr_symbolic_match, get_dataset_var_names
    from parallel_eval_pysr import _remap_formula_variables
    from utils import get_dataset_gt_formula
    names = get_dataset_var_names(dataset)
    variables = [f"x{i}" for i in range(len(names))]
    target = _remap_formula_variables(get_dataset_gt_formula(dataset), names, variables)
    if not target:
        raise ValueError(f"Missing target for {dataset}")
    results = []
    for record in records:
        result = {k: v for k, v in record.items() if k != "trace"}
        result.update(first_scheduled=None, first_elapsed=None, unresolved_checks=0,
                      missing_snapshots=[], unavailable_snapshots=[])
        trace = sorted(record["trace"], key=lambda o: o["elapsed_seconds"])
        present = {o.get("scheduled_seconds") for o in trace}
        result["missing_snapshots"] = [t for t in range(10, 91, 10) if t not in present]
        result["unavailable_snapshots"] = [o["scheduled_seconds"] for o in trace if o.get("status") != "ok"]
        for snapshot in trace:
            if snapshot.get("status") != "ok":
                continue
            # Check cached matches first; a positive suffices to score a snapshot.
            equations = sorted(snapshot.get("equations", []),
                               key=lambda row: not cache.get(row["equation"], {}).get("match"))
            for row in equations:
                eq = row["equation"]
                if eq not in cache:
                    decision = check_pysr_symbolic_match(eq, target, var_names=variables, timeout_seconds=3)
                    cache[eq] = {"match": bool(decision.get("match")), "error": decision.get("error")}
                decision = cache[eq]
                result["unresolved_checks"] += bool(decision.get("error"))
                if decision["match"]:
                    result.update(first_scheduled=snapshot["scheduled_seconds"],
                                  first_elapsed=snapshot["elapsed_seconds"], equation=eq)
                    break
            if result["first_scheduled"] is not None:
                break
        results.append(result)
    payload = {"dataset": dataset, "records": results, "equation_checks": cache}
    path.write_text(json.dumps(payload) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/srbench_9-16_snapshot_comparison")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Write per-dataset inputs without scoring")
    parser.add_argument("--dataset-index", type=int, help="Score one prepared dataset (0 through 129)")
    parser.add_argument("--aggregate-only", action="store_true", help="Require all dataset scores and write tables")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.output_dir / "inputs"
    checkpoints = args.output_dir / "datasets"
    if args.dataset_index is not None:
        item = json.loads((inputs_dir / f"{args.dataset_index:03d}.json").read_text())
        result = score_dataset(item)
        print(f"Scored {result['dataset']}: {len(result['records'])} trials", flush=True)
        return
    if args.aggregate_only:
        records = []
        for i in range(130):
            dataset = json.loads((inputs_dir / f"{i:03d}.json").read_text())[0]
            records.extend(json.loads((checkpoints / f"{dataset}.json").read_text())["records"])
        write_tables(records, args.output_dir)
        return
    grouped = defaultdict(list)
    final_stats = {}
    for label, name in RUNS.items():
        root = ROOT / "runs" / name
        manifest = srio.load_manifest(root)
        raw = srio.load_keyed_results(root)
        _, keyed = srio.standard_ground_truth_view(manifest, raw)
        rows = list(keyed.values())
        assert len(rows) == 5200 and all(r["present"] and not r.get("error") for r in rows)
        final_stats[label] = {str(noise): seed_stats([r for r in rows if noise == "all" or r["noise"] == noise])
                              for noise in ["all", 0, .001, .01, .1]}
        print(label, json.dumps(final_stats[label]["all"]), flush=True)
        if label.startswith("old_") or args.stats_only:
            continue
        for r in rows:
            grouped[r["dataset"]].append({
                "method": label, "dataset": r["dataset"], "seed": r["seed"], "noise": r["noise"],
                "trace": [o for o in r["execution_trace"] if not o.get("final")
                          and o.get("scheduled_seconds") is not None and o["scheduled_seconds"] <= 90],
            })
    (args.output_dir / "final_seed_stats.json").write_text(json.dumps(final_stats, indent=2) + "\n")
    if args.stats_only:
        return
    caches = defaultdict(dict)
    for name in PREVIOUS:
        for r in json.loads((ROOT / "runs" / name / "snapshot_solve_times.json").read_text())["records"]:
            caches[r["dataset"]].update(r["equation_checks"])
    checkpoints.mkdir(exist_ok=True)
    inputs = [(ds, rows, caches[ds], str(checkpoints)) for ds, rows in sorted(grouped.items())]
    assert len(inputs) == 130
    if args.prepare:
        inputs_dir.mkdir(exist_ok=True)
        for i, item in enumerate(inputs):
            (inputs_dir / f"{i:03d}.json").write_text(json.dumps(item) + "\n")
        print(f"Prepared {len(inputs)} dataset inputs in {inputs_dir}", flush=True)
        return
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_dataset, item) for item in inputs]
        for i, future in enumerate(as_completed(futures), 1):
            payload = future.result()
            records.extend(payload["records"])
            print(f"Scored {i}/130 datasets: {payload['dataset']}", flush=True)
    write_tables(records, args.output_dir)


def write_tables(records, output_dir):
    assert len(records) == 10400
    assert len({(r['method'], r['dataset'], r['seed'], r['noise']) for r in records}) == 10400
    tables = {}
    lines = ["Cumulative observed symbolic recovery; scheduled fit-wall snapshots.\n",
             "All rates are percentages; SD is across ten seed-level rates.\n"]
    for noise in ["all", 0, .001, .01, .1]:
        tables[str(noise)] = {}
        lines += [f"\nNoise: {noise}\n", "| Seconds | Baseline mean ± SD | Evolved mean ± SD |",
                  "|---:|---:|---:|"]
        for t in range(10, 91, 10):
            stats = {}
            for method in ["baseline", "evolved"]:
                subset = [dict(r, solved=r["first_scheduled"] is not None and r["first_scheduled"] <= t)
                          for r in records if r["method"] == method and (noise == "all" or r["noise"] == noise)]
                stats[method] = seed_stats(subset)
            tables[str(noise)][t] = stats
            b, e = stats["baseline"], stats["evolved"]
            lines.append(f"| {t} | {b['mean']:.2f} ± {b['sd']:.2f} | {e['mean']:.2f} ± {e['sd']:.2f} |")
    output = {"tables": tables, "records": records,
              "unresolved_checks": sum(r["unresolved_checks"] for r in records),
              "records_with_missing_snapshots": sum(bool(r["missing_snapshots"]) for r in records)}
    (output_dir / "snapshot_comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    (output_dir / "snapshot_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
