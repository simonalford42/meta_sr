#!/usr/bin/env python3
import os
import re
from pathlib import Path


def find_logs(root: Path):
    """Yield paths to all log directories under results/run_*/slurm/*/logs."""
    results_dir = root / "results"
    if not results_dir.exists():
        return
    for run_dir in results_dir.glob("run_*"):
        slurm_dir = run_dir / "slurm"
        if not slurm_dir.exists():
            continue
        for eval_dir in slurm_dir.iterdir():
            logs_dir = eval_dir / "logs"
            if logs_dir.is_dir():
                yield logs_dir


def parse_hostname_from_out(out_path: Path) -> str | None:
    """Parse hostname from a corresponding .out file line: 'Task X running on node: HOST'"""
    try:
        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.search(r"running on node:\s*(\S+)", line)
                if m:
                    return m.group(1)
    except FileNotFoundError:
        return None
    return None


def main():
    root = Path(__file__).resolve().parents[1]
    illegal_nodes: dict[str, int] = {}

    for logs_dir in find_logs(root):
        # Check all .err files for 'Illegal instruction'
        for err_path in logs_dir.glob("*.err"):
            try:
                with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            if "Illegal instruction" not in content:
                continue

            # Try to map to corresponding .out file to get hostname
            out_path = err_path.with_suffix("")  # remove .err
            out_path = out_path.with_suffix(".out")  # add .out

            hostname = parse_hostname_from_out(out_path)
            if not hostname:
                # Fallback: try to inspect any .out file in same logs dir with same task index suffix
                # Pattern examples: task_12.err -> task_12.out, retry1_task_12.err -> retry1_task_12.out
                # If not found, leave as unknown to surface issues
                hostname = "<unknown>"

            illegal_nodes[hostname] = illegal_nodes.get(hostname, 0) + 1

    if not illegal_nodes:
        print("No 'Illegal instruction' occurrences found in logs.")
        return

    # Print unique hostnames (and counts for context)
    print("Nodes with 'Illegal instruction' crashes:")
    for host in sorted(illegal_nodes.keys()):
        count = illegal_nodes[host]
        print(f"- {host} ({count} occurrences)")


if __name__ == "__main__":
    main()

