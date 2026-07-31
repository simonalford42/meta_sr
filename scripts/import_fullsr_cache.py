#!/usr/bin/env python3
"""Import successful FullSR batch artifacts into the shared task cache.

Use this for runs created before FullSR cache population was implemented, or
for a controller that crashed after workers had already written result JSONs.
Missing, errored, and incomplete black-box/trace results stay uncached so the
next evaluation reruns them.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_cache import get_fullsr_cache, set_fullsr_cache_path
from parallel_eval_fullsr import (
    FullSRTaskResult,
    FullSRTaskSpec,
    _build_fullsr_cache_entry,
)


def _batch_dirs(run_dir: Path) -> list[Path]:
    """Return manifest-referenced FullSR batches, with scan fallback."""
    candidates: list[Path] = []
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for batch in manifest.get("batches") or []:
            relative = batch.get("batch_dir")
            if relative and "slurm_fullsr" in relative:
                candidates.append(run_dir / relative)
        black_box = manifest.get("black_box") or {}
        relative = black_box.get("batch_dir")
        if relative and "slurm_fullsr" in relative:
            candidates.append(run_dir / relative)
    if not candidates:
        candidates.extend(sorted((run_dir / "slurm_fullsr").glob("eval_*")))

    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def import_run(run_dir: Path) -> tuple[int, int]:
    cache = get_fullsr_cache()
    if cache is None:
        raise RuntimeError("FullSR cache is disabled")

    entries = []
    skipped = 0
    batches = _batch_dirs(run_dir)
    if not batches:
        raise ValueError(f"No FullSR batches found under {run_dir}")

    for batch_dir in batches:
        tasks_path = batch_dir / "tasks.json"
        if not tasks_path.exists():
            continue
        tasks = json.loads(tasks_path.read_text())
        for index, task_dict in enumerate(tasks):
            result_path = batch_dir / "results" / f"task_{index:06d}.json"
            if not result_path.exists():
                skipped += 1
                continue
            try:
                task = FullSRTaskSpec.from_json_dict(task_dict)
                result = FullSRTaskResult.from_json_dict(
                    json.loads(result_path.read_text())
                )
                entry = _build_fullsr_cache_entry(task, result, cache)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                entry = None
            if entry is None:
                skipped += 1
            else:
                entries.append(entry)

    return cache.store_many(entries), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--cache-path",
        type=Path,
        help="Override caches/fullsr_evaluation_cache.db (mainly for testing).",
    )
    args = parser.parse_args()
    if args.cache_path is not None:
        set_fullsr_cache_path(str(args.cache_path))

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
