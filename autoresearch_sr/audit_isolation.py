#!/usr/bin/env python3
"""Flag agent commands that may have read other autoresearch runs' results.

    python autoresearch_sr/audit_isolation.py autoresearch_sr/live/<tag>

Scans every tool call in <run_dir>/agent_stream.jsonl and prints those whose
input mentions another run's directory or branch, the archived runs, or a
cross-branch git query. Heuristic: review flagged commands by hand.
"""

import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
tag = json.loads((run_dir / "run.json").read_text())["tag"]

PATTERNS = [
    (re.compile(r"autoresearch_sr/runs\b|/runs/(apr|sep|aug|iso)"), "archived runs"),
    (re.compile(r"autoresearch_sr/live(?!/" + re.escape(tag) + r"\b)"), "another run directory"),
    (re.compile(r"autoresearch_sr/(?!live\b|runs\b|evaluate\.py|program\.md|" + re.escape(tag) + r"\b)[\w.-]+"), "another autoresearch branch"),
    (re.compile(r"autoresearch_pysr_sandboxes(?!/" + re.escape(tag) + r"\b)"), "another run's sandboxes"),
    (re.compile(r"julia_depots(?!/" + re.escape(tag) + r"\b)"), "another run's depot"),
    (re.compile(r"git\b[^|;&]*\s--all\b|git\b[^|;&]*\s--branches\b|git\s+branch\s+(-a|-r|--all)"), "cross-branch git query"),
    (re.compile(r"meta_sr/SymbolicRegression\.jl|meta_sr_agent_loop/SymbolicRegression\.jl"), "shared SR repo (holds other runs' branches)"),
    (re.compile(r"\.\./"), "parent-directory access (check target)"),
    (re.compile(r"\.claude/projects|\.claude/CLAUDE|/memory/"), "Claude Code memory (shared across runs)"),
    (re.compile(r"\b(find|bfs|fd|locate)\s+(/|~|\$HOME|/home/sca63)(\s|/?\s|$)|grep\s+-\w*r\w*\s+[^|]*\s(/|~|/home/sca63)(\s|$)"), "filesystem-wide search"),
]

flagged = 0
n_calls = 0
for line in (run_dir / "agent_stream.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:  # partial line (live run or disk-full write)
        continue
    if event.get("type") != "assistant":
        continue
    for block in (event.get("message") or {}).get("content") or []:
        if block.get("type") != "tool_use":
            continue
        n_calls += 1
        text = json.dumps(block.get("input"))
        hits = sorted({label for pattern, label in PATTERNS if pattern.search(text)})
        if hits:
            flagged += 1
            print(f"[{', '.join(hits)}] {block.get('name')}: {text[:300]}")

print(f"isolation audit ({tag}): {flagged} of {n_calls} tool calls flagged")
