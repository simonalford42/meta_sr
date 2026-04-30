#!/usr/bin/env python3
"""View the bundle eval log as a table (newest first).

Usage:
    python3 view_eval_log.py                  # last 30 entries
    python3 view_eval_log.py -n 100           # last 100
    python3 view_eval_log.py --source pysr    # filter by source
    python3 view_eval_log.py --split barely_unsolvable
    python3 view_eval_log.py --since 1h       # last hour (30m/2h/1d)
    python3 view_eval_log.py --raw            # print JSONL verbatim

Log path: $META_SR_EVAL_LOG or ~/.cache/meta_sr/eval_log.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from eval_log import get_log_path, iter_log_records


def _fmt_dur(sec):
    if sec is None or sec != sec:
        return "-"
    sec = float(sec)
    if sec < 0:
        return "-"
    if sec >= 3600:
        return f"{sec/3600:.1f}h"
    if sec >= 60:
        return f"{sec/60:.1f}m"
    return f"{sec:.0f}s"


def _parse_ts(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_since(s):
    m = re.match(r"^(\d+)([smhd])$", s)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    delta = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit] * n
    return datetime.now(timezone.utc) - timedelta(seconds=delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=30, help="Number of most-recent entries")
    ap.add_argument("--source", choices=["pysr", "minisr"], help="Filter by source")
    ap.add_argument("--split", help="Filter by split name (substring match)")
    ap.add_argument("--since", help="Only entries newer than this (e.g. 30m, 2h, 1d)")
    ap.add_argument("--bundle", help="Filter by bundle name substring")
    ap.add_argument("--raw", action="store_true", help="Print JSONL verbatim (newest last)")
    ap.add_argument("--path", help="Override log path")
    ap.add_argument("--reverse", action="store_true",
                    help="Print oldest first (default is newest first)")
    args = ap.parse_args()

    path = Path(args.path) if args.path else get_log_path()
    if not path.exists():
        print(f"No log at {path}", file=sys.stderr)
        return 1

    since_dt = _parse_since(args.since) if args.since else None

    records = []
    for rec in iter_log_records(path):
        if args.source and rec.get("source") != args.source:
            continue
        if args.split and args.split not in (rec.get("split") or ""):
            continue
        if args.bundle and args.bundle not in (rec.get("bundle") or ""):
            continue
        if since_dt is not None:
            ts = _parse_ts(rec.get("ts", ""))
            if ts is None or ts < since_dt:
                continue
        records.append(rec)

    if args.raw:
        sel = records[-args.num:] if args.num > 0 else records
        import json
        for rec in sel:
            print(json.dumps(rec, separators=(",", ":")))
        return 0

    sel = records[-args.num:] if args.num > 0 else records
    if not args.reverse:
        sel = list(reversed(sel))

    # Fixed-width table
    header = (
        f"{'TIMESTAMP (UTC)':<20} "
        f"{'SRC':<6} "
        f"{'BUNDLE':<11} "
        f"{'SPLIT':<18} "
        f"{'NT':>4} "
        f"{'EXC':>4} "
        f"{'CAC':>4} "
        f"{'ERR':>3} "
        f"{'TO':>2} "
        f"{'RTR':>3} "
        f"{'WALL':>6} "
        f"{'MEAN':>5} "
        f"{'MAX':>6} "
        f"{'PAR':>4} "
        f"LABEL"
    )
    print(header)
    print("-" * len(header))
    for rec in sel:
        ts = rec.get("ts", "-")[:19].replace("T", " ")
        src = rec.get("source", "-")
        bundle = rec.get("bundle", "-")
        split = (rec.get("split") or "-")[:18]
        nt = rec.get("n_tasks", "-")
        exc = rec.get("n_executed", "-")
        cac = rec.get("n_cached", "-")
        err = rec.get("n_errors", "-")
        to  = rec.get("n_timed_out", "-")
        rtr = rec.get("n_retries", "-")
        wall = _fmt_dur(rec.get("bundle_wall_s"))
        mean = _fmt_dur(rec.get("task_mean_s"))
        mx   = _fmt_dur(rec.get("task_max_s"))
        par  = rec.get("parallelism", "-")
        par_s = f"{par:.1f}" if isinstance(par, (int, float)) else str(par)
        label = (rec.get("label") or "").strip()
        print(
            f"{ts:<20} "
            f"{src:<6} "
            f"{bundle:<11} "
            f"{split:<18} "
            f"{nt:>4} "
            f"{exc:>4} "
            f"{cac:>4} "
            f"{err:>3} "
            f"{to:>2} "
            f"{rtr:>3} "
            f"{wall:>6} "
            f"{mean:>5} "
            f"{mx:>6} "
            f"{par_s:>4} "
            f"{label}"
        )

    print("-" * len(header))
    print(f"({len(sel)} of {len(records)} records shown, path={path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
