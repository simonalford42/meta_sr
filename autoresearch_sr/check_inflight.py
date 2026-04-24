"""Check in-flight experiments for a given run tag.

Reads runs/<tag>/inflight.json and reports each entry's status:
  RUNNING  — process alive, no score in evaluate.log yet
  DONE     — evaluate.log contains a 'score:' line (regardless of process state)
  CRASHED  — process dead, no score in evaluate.log

Usage:
  python check_inflight.py --tag apr23
  python check_inflight.py --tag apr23 --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

AUTORESEARCH_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(AUTORESEARCH_DIR))
from submit_experiment import inflight_path, inflight_lock, _load_inflight  # noqa: E402


_SCORE_RE = re.compile(r"^score:\s+([-\d.]+)", re.MULTILINE)
_DATASETS_FAIL_RE = re.compile(r"^datasets_fail:\s+(\d+)", re.MULTILINE)


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _parse_log(log_path: Path) -> tuple[Optional[float], Optional[int]]:
    """Return (score, datasets_fail) if summary block is present, else (None, None)."""
    if not log_path.exists():
        return None, None
    try:
        content = log_path.read_text()
    except Exception:
        return None, None
    m_score = _SCORE_RE.search(content)
    m_fail = _DATASETS_FAIL_RE.search(content)
    score = float(m_score.group(1)) if m_score else None
    fail = int(m_fail.group(1)) if m_fail else None
    return score, fail


def _status(entry: dict) -> dict:
    pid = entry.get("pid", 0)
    log = Path(entry.get("evaluate_log", ""))
    score, datasets_fail = _parse_log(log)
    alive = _process_alive(pid)

    if score is not None:
        status = "DONE"
    elif alive:
        status = "RUNNING"
    else:
        status = "CRASHED"

    return {
        "exp_num": entry.get("exp_num"),
        "slot": entry.get("slot"),
        "pid": pid,
        "status": status,
        "score": score,
        "datasets_fail": datasets_fail,
        "elapsed_s": int(time.time() - entry.get("started_at", time.time())),
        "evaluate_log": str(log),
        "exp_dir": entry.get("exp_dir"),
        "slot_path": entry.get("slot_path"),
        "base_sr_sha": entry.get("base_sr_sha"),
        "slot_sr_sha": entry.get("slot_sr_sha"),
        "description": entry.get("description", ""),
    }


def _print_table(rows: list) -> None:
    if not rows:
        print("(no in-flight experiments)")
        return
    header = f"{'exp':>4}  {'slot':>4}  {'status':<8}  {'score':>8}  {'fail':>4}  {'elapsed':>7}  description"
    print(header)
    print("-" * len(header))
    for r in rows:
        score = f"{r['score']:.4f}" if r['score'] is not None else "—"
        fail = str(r['datasets_fail']) if r['datasets_fail'] is not None else "—"
        elapsed = f"{r['elapsed_s']}s"
        desc = (r['description'] or "")[:60]
        print(f"{r['exp_num']:>4}  {r['slot']:>4}  {r['status']:<8}  {score:>8}  {fail:>4}  {elapsed:>7}  {desc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = parser.parse_args()

    with inflight_lock(args.tag):
        entries = _load_inflight(args.tag)

    rows = [_status(e) for e in entries]
    rows.sort(key=lambda r: (r["exp_num"] if r["exp_num"] is not None else 0))

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
