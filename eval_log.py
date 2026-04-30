"""Append-only bundle evaluation log for monitoring cluster throughput.

Every completed bundle evaluation (PySR or MiniSR) writes one JSON line to a
shared log file. Safe for concurrent writers on Linux: each record is a single
os.write() < PIPE_BUF (4096), so O_APPEND guarantees the line won't interleave
with writes from other processes.

Location override: $META_SR_EVAL_LOG (default ~/.cache/meta_sr/eval_log.jsonl).

View with: python3 view_eval_log.py            (newest first)
          tail -f ~/.cache/meta_sr/eval_log.jsonl   (as JSONL)
"""

from __future__ import annotations

import json
import os
import socket
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_DEFAULT_LOG_PATH = Path.home() / ".cache" / "meta_sr" / "eval_log.jsonl"
_MAX_LINE_BYTES = 3800  # stay under PIPE_BUF=4096 to keep appends atomic
_LABEL_MAX = 200


def get_log_path() -> Path:
    override = os.environ.get("META_SR_EVAL_LOG")
    return Path(override) if override else _DEFAULT_LOG_PATH


def log_bundle_eval(
    source: str,
    bundle: str,
    n_tasks: int,
    n_cached: int,
    n_executed: int,
    n_errors: int,
    n_timed_out: int,
    bundle_wall_s: float,
    task_runtime_s: Sequence[float],
    label: str = "",
    split: Optional[str] = None,
    n_datasets: Optional[int] = None,
    n_runs: Optional[int] = None,
    n_configs: Optional[int] = None,
    n_retries: int = 0,
    results_dir: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one bundle-eval record. Never raises — logging is best-effort."""
    try:
        path = get_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": source,
            "bundle": bundle,
        }
        if split is not None:
            record["split"] = split
        if n_datasets is not None:
            record["n_datasets"] = int(n_datasets)
        if n_runs is not None:
            record["n_runs"] = int(n_runs)
        if n_configs is not None:
            record["n_configs"] = int(n_configs)
        record["n_tasks"] = int(n_tasks)
        record["n_cached"] = int(n_cached)
        record["n_executed"] = int(n_executed)
        record["n_errors"] = int(n_errors)
        record["n_timed_out"] = int(n_timed_out)
        record["n_retries"] = int(n_retries)
        record["bundle_wall_s"] = round(float(bundle_wall_s), 1)

        runtimes = [float(t) for t in task_runtime_s if t and t > 0]
        if runtimes:
            record["task_mean_s"] = round(statistics.fmean(runtimes), 1)
            record["task_median_s"] = round(statistics.median(runtimes), 1)
            record["task_max_s"] = round(max(runtimes), 1)
            record["task_total_s"] = round(sum(runtimes), 1)
            # Effective parallelism = total compute / wall clock. Higher = cluster
            # was responsive; lower = tasks queued serially / cluster congested.
            if bundle_wall_s > 0:
                record["parallelism"] = round(sum(runtimes) / bundle_wall_s, 2)

        record["host"] = socket.gethostname().split(".")[0]
        record["pid"] = os.getpid()
        run_id = os.environ.get("SLURM_JOB_ID")
        if run_id:
            record["run_id"] = run_id

        if label:
            if len(label) > _LABEL_MAX:
                label = label[: _LABEL_MAX - 3] + "..."
            record["label"] = label
        if results_dir:
            record["results_dir"] = str(results_dir)
        if extra:
            record.update(extra)

        line = json.dumps(record, separators=(",", ":")) + "\n"
        payload = line.encode("utf-8")
        if len(payload) > _MAX_LINE_BYTES:
            # Last-ditch truncation so the append stays atomic.
            record["label"] = (record.get("label", "") or "")[:50] + "...[TRUNC]"
            line = json.dumps(record, separators=(",", ":")) + "\n"
            payload = line.encode("utf-8")
            if len(payload) > _MAX_LINE_BYTES:
                return

        fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except Exception:
        # Never let logging failures break the evaluator.
        pass


def iter_log_records(path: Optional[Path] = None):
    """Yield parsed records from a JSONL log file, skipping malformed lines."""
    if path is None:
        path = get_log_path()
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
