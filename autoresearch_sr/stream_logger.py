#!/usr/bin/env python3
"""Append Claude Code stream-json events from stdin to a JSONL file, stamping
each with the wall-clock time it was received (field ``_recv_time``)."""

import json
import sys
import time

with open(sys.argv[1], "a") as out:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {"type": "_raw", "text": line}
        event["_recv_time"] = time.time()
        out.write(json.dumps(event) + "\n")
        out.flush()
