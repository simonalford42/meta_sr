#!/usr/bin/env python3
"""Import successful black-box batch artifacts into the shared PySR cache.

This is intended for evaluations created before black-box cache reuse was
enabled. Failed/error results and results without a test Pareto frontier are
left uncached so a subsequent evaluation reruns them.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_cache import get_pysr_cache
from parallel_eval_pysr import (
    PySRTaskResult,
    PySRTaskSpec,
    _build_pysr_cache_entries,
)


def import_run(run_dir: Path) -> tuple[int, int]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    black_box = manifest.get("black_box") or {}
    relative_batch = black_box.get("batch_dir")
    if not relative_batch:
        raise ValueError(f"No black-box batch recorded in {manifest_path}")

    batch_dir = run_dir / relative_batch
    tasks = json.loads((batch_dir / "tasks.json").read_text())
    entries = []
    skipped = 0
    for index, task_dict in enumerate(tasks):
        result_path = batch_dir / "results" / f"task_{index:06d}.json"
        if not result_path.exists():
            skipped += 1
            continue
        result = PySRTaskResult.from_json_dict(json.loads(result_path.read_text()))
        if result.error is not None or not result.pareto_frontier:
            skipped += 1
            continue
        task = PySRTaskSpec.from_json_dict(task_dict)
        if not task.black_box:
            raise ValueError(f"Task {index} in {batch_dir} is not black-box")
        entries.extend(_build_pysr_cache_entries(task, result))

    cache = get_pysr_cache()
    if cache is None:
        raise RuntimeError("PySR cache is disabled")
    return cache.store_many(entries), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    total_imported = 0
    total_skipped = 0
    for run_dir in args.run_dirs:
        imported, skipped = import_run(run_dir)
        total_imported += imported
        total_skipped += skipped
        print(f"{run_dir}: imported {imported}, left uncached {skipped}")
    print(f"Total: imported {total_imported}, left uncached {total_skipped}")


if __name__ == "__main__":
    main()
