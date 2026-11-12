#!/usr/bin/env python3
"""Append Claude Code stream-json events from stdin to a JSONL file, stamping
each with the wall-clock time it was received (field ``_recv_time``).

Write errors (e.g. a full disk, 11/11) are retried instead of killing the
logger, which would otherwise leave the agent writing into a dead pipe."""

import json
import sys
import time

path = sys.argv[1]
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        event = {"type": "_raw", "text": line}
    event["_recv_time"] = time.time()
    record = json.dumps(event) + "\n"
    while True:
        try:
            with open(path, "a") as out:
                out.write(record)
            break
        except OSError as exc:
            print(f"stream_logger: write failed ({exc}); retrying in 60 s", file=sys.stderr, flush=True)
            time.sleep(60)
