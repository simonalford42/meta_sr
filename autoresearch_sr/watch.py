#!/usr/bin/env python3
"""Readable view of an autoresearch run.

    python autoresearch_sr/watch.py <tag>            # status + the agent's last 15 actions
    python autoresearch_sr/watch.py <tag> -f         # ...then follow live (Ctrl-C to stop)
    python autoresearch_sr/watch.py <tag> -n 50      # show more history

Shows the agent's messages and commands from agent_stream.jsonl, evaluation
starts/ends from experiment_log.jsonl, the SLURM state of the current
evaluation, and spend so far.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

AR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AR_DIR))
from cost_report import run_cost  # noqa: E402


def fmt_time(t):
    return datetime.fromtimestamp(t).strftime("%H:%M:%S") if t else "--:--:--"


def stream_lines(event):
    """Human-readable lines for one agent_stream event."""
    t = event.get("_recv_time")
    if event.get("type") == "assistant":
        for block in (event.get("message") or {}).get("content") or []:
            if block.get("type") == "text" and block.get("text", "").strip():
                yield f"{fmt_time(t)}  agent: {block['text'].strip()[:300]}"
            elif block.get("type") == "tool_use":
                inp = block.get("input") or {}
                what = inp.get("description") or ""
                cmd = inp.get("command") or inp.get("file_path") or json.dumps(inp)
                cmd = " ".join(str(cmd).split())
                yield f"{fmt_time(t)}  {block.get('name')}: {what + ' | ' if what else ''}{cmd[:200]}"
    elif event.get("type") == "_launch":
        yield f"{fmt_time(t)}  [launcher] agent (re)started, restart {event.get('restart')}"
    elif event.get("type") == "result":
        yield f"{fmt_time(t)}  [agent session ended]"


def eval_lines(run_dir):
    p = run_dir / "experiment_log.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r["event"] == "start":
            out.append((r["time"], f"{fmt_time(r['time'])}  >> eval exp{r['exp']} started ({r['commit'][:8]})"))
        else:
            score = "crashed" if r.get("score") is None else f"score {r['score']:.3f}, {r.get('datasets_fail', 0)} failed datasets"
            out.append((r["time"], f"{fmt_time(r['time'])}  << eval exp{r['exp']} done: {score}"))
    return out


def status(run_dir):
    cfg = json.loads((run_dir / "run.json").read_text())
    print(f"== {cfg['tag']} ({cfg['model']}) ==")
    log = run_dir / "experiment_log.jsonl"
    recs = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    done = {r["commit"] for r in recs if r["event"] == "end" and r["commit"] != cfg["base_commit"]}
    scores = [r["score"] for r in recs if r["event"] == "end" and r.get("score") is not None]
    print(f"experiments done: {len(done)}/{cfg['max_experiments']}   best score: "
          f"{max(scores):.3f}" if scores else f"experiments done: {len(done)}/{cfg['max_experiments']}")
    print(f"API spend: ${run_cost(run_dir):.2f} of ${cfg.get('max_cost_usd', float('nan'))}")
    starts = {r["output_dir"]: r for r in recs if r["event"] == "start"}
    ended = {(r["exp"], r["commit"]) for r in recs if r["event"] == "end"}
    running = [r for r in starts.values() if (r["exp"], r["commit"]) not in ended]
    if running:
        r = running[-1]
        batch = Path(r["output_dir"]) / "slurm_pysr" / "eval_0000"
        n_res = len(list((batch / "results").glob("*.json"))) if (batch / "results").exists() else 0
        n_started = len(list((batch / "logs").glob("*.out"))) if (batch / "logs").exists() else 0
        try:
            q = subprocess.run(["squeue", "-h", "-u", "sca63", "-o", "%T %Z"], capture_output=True, text=True).stdout
            pending = sum(1 for l in q.splitlines() if str(run_dir.parent.parent.parent) in l and l.startswith("PENDING"))
        except OSError:
            pending = "?"
        mins = (time.time() - r["time"]) / 60
        print(f"current eval: exp{r['exp']} ({r['commit'][:8]}), running {mins:.0f} min: "
              f"{n_started} tasks started, {n_res}/200 results, {pending} SLURM job(s) pending")
    else:
        print("current eval: none (agent is thinking/editing)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("-f", "--follow", action="store_true")
    ap.add_argument("-n", type=int, default=15)
    args = ap.parse_args()
    run_dir = AR_DIR / "live" / args.tag
    if not (run_dir / "run.json").exists():
        sys.exit(f"no run {args.tag!r} in {AR_DIR / 'live'}")
    status(run_dir)

    stream = run_dir / "agent_stream.jsonl"
    pos, items = 0, []
    def read_new():
        nonlocal pos
        new = []
        if stream.exists():
            with stream.open() as fh:
                fh.seek(pos)
                for line in fh:
                    if not line.endswith("\n"):
                        break
                    pos += len(line.encode())
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for text in stream_lines(ev):
                        new.append((ev.get("_recv_time") or 0, text))
        return new

    items = read_new() + eval_lines(run_dir)
    items.sort(key=lambda x: x[0])
    for _, text in items[-args.n:]:
        print(text)
    if not args.follow:
        return
    seen_evals = len(eval_lines(run_dir))
    try:
        while True:
            time.sleep(5)
            new = read_new()
            evs = eval_lines(run_dir)
            new += evs[seen_evals:]
            seen_evals = len(evs)
            for _, text in sorted(new, key=lambda x: x[0]):
                print(text, flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
