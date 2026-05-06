#!/usr/bin/env python3
"""Extract PySR SLURM-array timing points from driver logs.

This is intended for post-hoc analysis of evolve_pysr.py/evaluate_new_pysr.py
logs in out/<slurm_id>.out.  It filters to 1e6 PySR max_evals batches and, for
evolve_pysr.py, keeps only the older all-configs-in-one-eval submission style.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PYSR_EVAL_RE = re.compile(
    r"PySR SLURM eval: (?P<total>\d+) tasks in batch (?P<batch>\S+) "
    r"\((?P<configs>\d+) configs x (?P<datasets>\d+) datasets x (?P<runs>\d+) runs\)"
)
SUBMIT_RE = re.compile(r"Submitted SLURM job array: (?P<job>\d+) \((?P<detail>.*?)(?P<tasks>\d+) tasks\)")
SCRIPT_RE = re.compile(r"Script: (?P<script>.+)")
ALL_DONE_RE = re.compile(r"All (?P<tasks>\d+) tasks completed in (?P<seconds>[0-9.]+)s")
ALL_INITIAL_DONE_RE = re.compile(r"All (?P<tasks>\d+) initial tasks completed in (?P<seconds>[0-9.]+)s")
PROGRESS_RE = re.compile(r"Progress(?: \([^)]*\))?: \d+/\d+ tasks complete \((?P<seconds>[0-9.]+)s elapsed")
TIMEOUT_RE = re.compile(r"TIMEOUT: .* exceeded (?P<seconds>[0-9.]+)s")
STALL_RE = re.compile(r"STALL: .* no progress for (?P<seconds>[0-9.]+)s")
RETRY_DONE_RE = re.compile(r"Retry completed in (?P<seconds>[0-9.]+)s")
RETRY_PROGRESS_RE = re.compile(r"Retry progress: \d+/\d+ tasks complete \((?P<seconds>[0-9.]+)s elapsed\)")
COMMAND_RE = re.compile(r"(?:^|Running: .*)(?P<cmd>(?:\S*/)?(?:python(?:\d+(?:\.\d+)?)?\s+(?:-u\s+)?)?(?:\S*/)?(?:evolve_pysr|evaluate_new_pysr)\.py\b.*)")


@dataclass
class Point:
    parent_slurm_id: str
    source: str
    command: str
    log_line: int
    batch: str
    total_tasks: int
    submitted_tasks: int
    n_configs: int
    n_datasets: int
    n_runs: int
    t_seconds: float
    initial_seconds: float
    retry_seconds: float
    duration_source: str
    initial_job_ids: str
    script: str
    tasks_json: str


def _normalize_path(path_text: str, repo_root: Path) -> Path:
    path_text = path_text.strip()
    path = Path(path_text)
    if path.exists():
        return path
    marker = "/meta_sr/"
    if marker in path_text:
        suffix = path_text.split(marker, 1)[1]
        mapped = repo_root / suffix
        if mapped.exists():
            return mapped
    return path


def _find_command(lines: list[str]) -> str:
    for line in lines[:80]:
        stripped = line.strip()
        if stripped.startswith("evolve_pysr.py ") or stripped.startswith("evaluate_new_pysr.py "):
            return stripped
    for line in lines[:120]:
        match = COMMAND_RE.search(line.strip())
        if match:
            return match.group("cmd").strip()
    return ""


def _infer_source(command: str, lines: list[str]) -> str:
    if "evaluate_new_pysr.py" in command:
        return "evaluate_new_pysr.py"
    if "evolve_pysr.py" in command:
        return "evolve_pysr.py"
    sample = "\n".join(lines[:80])
    if "evaluate_new_pysr.py" in sample:
        return "evaluate_new_pysr.py"
    if "Evolving:" in sample and "hpo_pysr" not in sample and "openevolve" not in sample:
        return "evolve_pysr.py"
    return ""


def _is_new_evolve_submission(lines: list[str]) -> bool:
    text = "\n".join(lines)
    return bool(
        re.search(r"Submitted SLURM job array: \d+ \(batch eval_", text)
        or re.search(r"Waiting on \d+ .* batches", text)
        or "submit_bundle_future" in text
    )


def _batch_tasks_json(script_path: str, repo_root: Path) -> tuple[Path | None, Path | None]:
    script = _normalize_path(script_path, repo_root)
    if not script.exists():
        return script, None
    batch_dir = script.parent
    tasks_json = batch_dir / "tasks.json"
    return script, tasks_json if tasks_json.exists() else None


def _max_evals_from_tasks(tasks_json: Path) -> set[int | None]:
    with tasks_json.open("r", encoding="utf-8", errors="replace") as f:
        tasks = json.load(f)
    vals: set[int | None] = set()
    for task in tasks:
        kwargs = task.get("pysr_kwargs") or {}
        val = kwargs.get("max_evals")
        vals.add(None if val is None else int(val))
    return vals


def _retry_seconds(block: list[str]) -> float:
    total = 0.0
    active = False
    current_progress = 0.0
    for line in block:
        if "Submitted retry job:" in line:
            if active:
                total += current_progress
            active = True
            current_progress = 0.0
            continue
        match = RETRY_PROGRESS_RE.search(line)
        if active and match:
            current_progress = max(current_progress, float(match.group("seconds")))
            continue
        match = RETRY_DONE_RE.search(line)
        if match:
            if active:
                total += float(match.group("seconds"))
                active = False
                current_progress = 0.0
            else:
                total += float(match.group("seconds"))
            continue
        if active and "Retry job" in line and "ended with status" in line:
            total += current_progress
            active = False
            current_progress = 0.0
    if active:
        total += current_progress
    return total


def _initial_seconds(block: list[str], total_tasks: int) -> tuple[float | None, str]:
    done_matches: list[tuple[int, float, str]] = []
    for line in block:
        for regex, source in ((ALL_DONE_RE, "all_completed"), (ALL_INITIAL_DONE_RE, "all_initial_completed")):
            match = regex.search(line)
            if match:
                done_matches.append((int(match.group("tasks")), float(match.group("seconds")), source))
    if done_matches:
        exact = [m for m in done_matches if m[0] == total_tasks]
        chosen = exact[-1] if exact else done_matches[-1]
        return chosen[1], chosen[2]

    timeout_matches = [float(m.group("seconds")) for line in block if (m := TIMEOUT_RE.search(line))]
    if timeout_matches:
        return timeout_matches[-1], "timeout_limit"

    stall_matches = [float(m.group("seconds")) for line in block if (m := STALL_RE.search(line))]
    if stall_matches:
        return stall_matches[-1], "stall_limit"

    progress_matches = [float(m.group("seconds")) for line in block if (m := PROGRESS_RE.search(line))]
    if progress_matches:
        return progress_matches[-1], "last_progress_lower_bound"

    return None, "missing"


def _blocks(lines: list[str]) -> Iterable[tuple[int, int, re.Match[str]]]:
    starts: list[tuple[int, re.Match[str]]] = []
    for idx, line in enumerate(lines):
        match = PYSR_EVAL_RE.search(line)
        if match:
            starts.append((idx, match))
    for pos, (start, match) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        yield start, end, match


def extract_points(log_dir: Path, repo_root: Path) -> tuple[list[Point], dict[str, dict[str, str]]]:
    points: list[Point] = []
    job_commands: dict[str, dict[str, str]] = {}

    for log_path in sorted(log_dir.glob("*.out")):
        if not re.fullmatch(r"\d+", log_path.stem):
            continue
        parent_slurm_id = log_path.stem
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not any("PySR SLURM eval:" in line for line in lines):
            continue

        command = _find_command(lines)
        source = _infer_source(command, lines)
        if source not in {"evolve_pysr.py", "evaluate_new_pysr.py"}:
            continue
        if source == "evolve_pysr.py" and _is_new_evolve_submission(lines):
            continue

        if not command:
            command = "(not logged; inferred from run output)"

        job_commands[parent_slurm_id] = {"source": source, "command": command}

        for start, end, eval_match in _blocks(lines):
            block = lines[start:end]
            submissions = [m for line in block if (m := SUBMIT_RE.search(line)) and "retry" not in line.lower()]
            if not submissions:
                continue
            script_matches = [m.group("script").strip() for line in block if (m := SCRIPT_RE.search(line))]
            script_text = script_matches[0] if script_matches else ""
            script_path, tasks_json = _batch_tasks_json(script_text, repo_root) if script_text else (None, None)
            if tasks_json is None:
                continue
            try:
                max_evals = _max_evals_from_tasks(tasks_json)
            except Exception:
                continue
            if max_evals != {1_000_000}:
                continue

            total_tasks = int(eval_match.group("total"))
            submitted_tasks = sum(int(m.group("tasks")) for m in submissions)
            initial, duration_source = _initial_seconds(block, total_tasks)
            if initial is None:
                continue
            retries = _retry_seconds(block)
            points.append(
                Point(
                    parent_slurm_id=parent_slurm_id,
                    source=source,
                    command=command,
                    log_line=start + 1,
                    batch=eval_match.group("batch"),
                    total_tasks=total_tasks,
                    submitted_tasks=submitted_tasks,
                    n_configs=int(eval_match.group("configs")),
                    n_datasets=int(eval_match.group("datasets")),
                    n_runs=int(eval_match.group("runs")),
                    t_seconds=round(initial + retries, 1),
                    initial_seconds=round(initial, 1),
                    retry_seconds=round(retries, 1),
                    duration_source=duration_source,
                    initial_job_ids=";".join(m.group("job") for m in submissions),
                    script=str(script_path) if script_path else script_text,
                    tasks_json=str(tasks_json),
                )
            )

    used_ids = {p.parent_slurm_id for p in points}
    job_commands = {jid: val for jid, val in job_commands.items() if jid in used_ids}
    return points, job_commands


def summarize(points: list[Point]) -> list[dict[str, float | int]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for point in points:
        grouped[point.submitted_tasks].append(point.t_seconds)

    rows: list[dict[str, float | int]] = []
    for n in sorted(grouped):
        vals = grouped[n]
        mean = statistics.fmean(vals)
        var = statistics.pvariance(vals) if len(vals) > 1 else 0.0
        rows.append(
            {
                "N": n,
                "n_points": len(vals),
                "mean_s": round(mean, 3),
                "var_s2": round(var, 3),
                "min_s": round(min(vals), 3),
                "max_s": round(max(vals), 3),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_job_commands(path: Path, job_commands: dict[str, dict[str, str]]) -> None:
    rows = [
        {"parent_slurm_id": jid, "source": val["source"], "command": val["command"]}
        for jid, val in sorted(job_commands.items(), key=lambda item: int(item[0]))
    ]
    write_csv(path, rows)


def make_plot(points: list[Point], stats_rows: list[dict[str, float | int]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ns = [int(row["N"]) for row in stats_rows]
    n_to_pos = {n: i for i, n in enumerate(ns)}
    per_n_seen: dict[int, int] = defaultdict(int)
    xs = []
    for point in points:
        n = point.submitted_tasks
        idx = per_n_seen[n]
        per_n_seen[n] += 1
        count = max(1, sum(1 for p in points if p.submitted_tasks == n))
        jitter = 0.0 if count == 1 else ((idx / (count - 1)) - 0.5) * 0.55
        xs.append(n_to_pos[n] + jitter)
    ys = [p.t_seconds / 60.0 for p in points]

    fig, (ax_time, ax_count) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_time.scatter(xs, ys, alpha=0.45, s=26, label="eval batch")
    means = [float(row["mean_s"]) / 60.0 for row in stats_rows]
    mins = [float(row["min_s"]) / 60.0 for row in stats_rows]
    maxs = [float(row["max_s"]) / 60.0 for row in stats_rows]
    lower = [m - lo for m, lo in zip(means, mins)]
    upper = [hi - m for m, hi in zip(means, maxs)]
    positions = list(range(len(ns)))
    ax_time.errorbar(positions, means, yerr=[lower, upper], fmt="o-", color="black", lw=1.5, capsize=4, label="mean with min/max")
    ax_time.set_ylabel("T including retries (minutes)")
    ax_time.set_title("1e6 max_evals PySR SLURM eval timing, old all-at-once style")
    ax_time.grid(True, axis="y", alpha=0.25)
    ax_time.legend(loc="best")

    counts = [int(row["n_points"]) for row in stats_rows]
    ax_count.bar(positions, counts, width=0.65, color="#5B8FF9", alpha=0.85)
    ax_count.set_ylabel("points")
    ax_count.set_xlabel("N submitted SLURM array tasks")
    ax_count.grid(True, axis="y", alpha=0.25)
    ax_count.set_xticks(positions)
    ax_count.set_xticklabels([str(n) for n in ns], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_markdown(points: list[Point], stats_rows: list[dict[str, float | int]], job_commands: dict[str, dict[str, str]]) -> None:
    print("Relevant parent SLURM jobs")
    print("| slurm_id | source | command |")
    print("|---:|---|---|")
    for jid, val in sorted(job_commands.items(), key=lambda item: int(item[0])):
        print(f"| {jid} | {val['source']} | `{val['command']}` |")
    print()

    print("Per-N summary")
    print("| N | points | mean_s | var_s2 | min_s | max_s |")
    print("|---:|---:|---:|---:|---:|---:|")
    for row in stats_rows:
        print(
            f"| {row['N']} | {row['n_points']} | {row['mean_s']:.1f} | "
            f"{row['var_s2']:.1f} | {row['min_s']:.1f} | {row['max_s']:.1f} |"
        )
    print()

    print("Data points")
    print("| parent_slurm_id | line | batch | array_job_ids | N | T_s | source |")
    print("|---:|---:|---|---|---:|---:|---|")
    for point in sorted(points, key=lambda p: (int(p.parent_slurm_id), p.batch)):
        print(
            f"| {point.parent_slurm_id} | {point.log_line} | {point.batch} | {point.initial_job_ids} | "
            f"{point.submitted_tasks} | {point.t_seconds:.1f} | {point.source} |"
        )


def markdown_text(points: list[Point], stats_rows: list[dict[str, float | int]], job_commands: dict[str, dict[str, str]]) -> str:
    from io import StringIO
    import contextlib

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        print_markdown(points, stats_rows, job_commands)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("out"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--points-csv", type=Path, default=Path("plots/pysr_slurm_parallel_eval_1e6_oldstyle_points.csv"))
    parser.add_argument("--stats-csv", type=Path, default=Path("plots/pysr_slurm_parallel_eval_1e6_oldstyle_stats.csv"))
    parser.add_argument("--jobs-csv", type=Path, default=Path("plots/pysr_slurm_parallel_eval_1e6_oldstyle_jobs.csv"))
    parser.add_argument("--plot", type=Path, default=Path("plots/pysr_slurm_parallel_eval_1e6_oldstyle_timing.png"))
    parser.add_argument("--report-md", type=Path, default=Path("plots/pysr_slurm_parallel_eval_1e6_oldstyle_report.md"))
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    points, job_commands = extract_points(args.log_dir, repo_root)
    stats_rows = summarize(points)

    write_csv(args.points_csv, [asdict(p) for p in points])
    write_csv(args.stats_csv, stats_rows)
    write_job_commands(args.jobs_csv, job_commands)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(markdown_text(points, stats_rows, job_commands), encoding="utf-8")
    if points:
        make_plot(points, stats_rows, args.plot)

    print(f"points: {len(points)}")
    print(f"jobs: {len(job_commands)}")
    print(f"points_csv: {args.points_csv}")
    print(f"stats_csv: {args.stats_csv}")
    print(f"jobs_csv: {args.jobs_csv}")
    print(f"report_md: {args.report_md}")
    print(f"plot: {args.plot if points else '(not written; no points)'}")
    if args.markdown:
        print()
        print_markdown(points, stats_rows, job_commands)


if __name__ == "__main__":
    main()
