#!/usr/bin/env python3
"""Summarize PySR SRBench smoke-test status files.

Usage:
    python scripts/collect_smoke_results.py \\
        --results_dir results_pysr_smoke \\
        --split_file splits/srbench_all.txt
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results_pysr_smoke")
    parser.add_argument("--split_file", type=str, default="splits/srbench_all.txt")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    expected = [line.strip() for line in open(args.split_file) if line.strip()]

    ok, errors, missing = [], [], []
    for ds in expected:
        path = results_dir / f"{ds}_status.json"
        if not path.exists():
            missing.append(ds)
            continue
        with open(path) as f:
            rec = json.load(f)
        if rec.get("status") == "ok":
            ok.append(ds)
        else:
            errors.append((ds, rec.get("error_type", "?"), rec.get("error", "")))

    print(f"Total expected: {len(expected)}")
    print(f"  OK:      {len(ok)}")
    print(f"  Errors:  {len(errors)}")
    print(f"  Missing: {len(missing)}")

    if errors:
        print("\n--- Errors ---")
        for ds, etype, msg in errors:
            print(f"  {ds:<40s} {etype}: {msg.splitlines()[0] if msg else ''}")
    if missing:
        print("\n--- Missing (no status file written) ---")
        for ds in missing:
            print(f"  {ds}")

    sys.exit(0 if not errors and not missing else 1)


if __name__ == "__main__":
    main()
