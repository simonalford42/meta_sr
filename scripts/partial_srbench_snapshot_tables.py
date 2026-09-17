"""Write comparable snapshot tables using only fully scored datasets."""
import json
from pathlib import Path

from scripts.compare_srbench_snapshots import ROOT, seed_stats


def main():
    root = ROOT / "runs/srbench_9-16_snapshot_comparison"
    records, included, omitted = [], [], []
    for source in sorted((root / "inputs").glob("*.json")):
        dataset, expected, _, _ = json.loads(source.read_text())
        complete = root / "datasets" / f"{dataset}.json"
        parts = [root / "binary_shards" / f"{dataset}.{i:03d}.json" for i in range(10)]
        if complete.exists():
            rows = json.loads(complete.read_text())["records"]
        elif all(p.exists() for p in parts):
            rows = [r for p in parts for r in json.loads(p.read_text())["records"]]
        else:
            omitted.append(dataset)
            continue
        key = lambda r: (r["method"], r["dataset"], r["seed"], r["noise"])
        assert len(rows) == 80 and {key(r) for r in rows} == {key(r) for r in expected}
        records.extend(rows)
        included.append(dataset)
    assert included
    approximate = sorted({r["dataset"] for r in records if r.get("scoring_method") == "binary_search_final_gate"})
    lines = [f"Coverage: {len(included)}/130 fully scored datasets; ten seeds and four noise levels per method.",
             f"Omitted: {', '.join(omitted) or 'none'}.",
             f"Approximate binary-search datasets: {', '.join(approximate) or 'none'}.",
             "Entries are mean solve percentage ± sample SD across ten seed-level rates.",
             "Scheduled snapshot times include fit startup. Final frontiers are excluded.", ""]
    tables = {}
    for noise in ["all", 0, .001, .01, .1]:
        lines += [f"Noise: {noise}", "", "| Time | Baseline (%) | Evolved 709715 (%) |",
                  "|---:|---:|---:|"]
        tables[str(noise)] = {}
        for t in range(10, 91, 10):
            stats = {}
            for method in ["baseline", "evolved"]:
                subset = [dict(r, solved=r["first_scheduled"] is not None and r["first_scheduled"] <= t)
                          for r in records if r["method"] == method and (noise == "all" or r["noise"] == noise)]
                stats[method] = seed_stats(subset)
            tables[str(noise)][t] = stats
            b, e = stats["baseline"], stats["evolved"]
            lines.append(f"| {t}s | {b['mean']:.2f} ± {b['sd']:.2f} | {e['mean']:.2f} ± {e['sd']:.2f} |")
        lines.append("")
    (root / "partial_snapshot_tables.md").write_text("\n".join(lines))
    (root / "partial_snapshot_tables.json").write_text(json.dumps(
        dict(included=included, omitted=omitted, approximate_datasets=approximate, tables=tables), indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
