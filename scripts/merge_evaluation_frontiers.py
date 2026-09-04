#!/usr/bin/env python3
"""Create merged frontiers from a compatible completed evaluation run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from frontier_aggregation import FrontierMergeError, merge_frontiers


RESULT_FILES = ("neuron_results.json", "boolean_results.json", "empbench_results.json")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir
    output = args.output or run_dir / "merged_frontiers.json"

    source = next((run_dir / name for name in RESULT_FILES if (run_dir / name).exists()), None)
    if source is None:
        raise SystemExit(
            "No compatible NeuronBench, Boolean, or EmpiricalBench result file found. "
            "Older standard SRBench/MIPS runs did not retain mergeable frontiers."
        )
    payload = json.loads(source.read_text())
    domain = (payload.get("protocol") or {}).get("domain", source.stem)
    key_field = "world" if source.name.startswith("neuron") else (
        "problem" if source.name.startswith("boolean") else "dataset"
    )
    grouped = defaultdict(list)
    for record in payload.get("runs", []):
        if record.get("status") == "complete" and record.get("frontier"):
            grouped[record[key_field]].append(record)

    merged = {}
    for name, records in grouped.items():
        # Preserve the algorithm's own objective. PySR++ may evolve a custom
        # loss, so replacing it with MSE changes the method being evaluated.
        loss_key = "loss"
        try:
            frontier = merge_frontiers(
                [record["frontier"] for record in records],
                sources=[{
                    "source_run_index": record.get("run_index"),
                    "source_seed": record.get("seed"),
                } for record in records],
                loss_key=loss_key,
            )
        except FrontierMergeError as exc:
            raise SystemExit(f"Cannot merge {name}: {exc}") from exc
        merged[name] = {
            "frontier": frontier,
            "n_searches": len(records),
            "loss_source": loss_key,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "source": str(source),
        "domain": domain,
        "selection": "minimum training loss per complexity, then Pareto prune",
        "datasets": merged,
    }, indent=2) + "\n")
    print(f"Wrote {len(merged)} merged frontiers to {output}")


if __name__ == "__main__":
    main()
