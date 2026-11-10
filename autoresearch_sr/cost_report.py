#!/usr/bin/env python3
"""Per-experiment API cost and cumulative wall-clock for a budgeted autoresearch run.

    python autoresearch_sr/cost_report.py autoresearch_sr/live/<tag> [--csv out.csv]

Inputs (both written automatically during a run):
  <run_dir>/agent_stream.jsonl    Claude Code stream-json, stamped with _recv_time
  <run_dir>/experiment_log.jsonl  start/end of every evaluate.py call

Cost attribution: the agent's tokens between the launch of evaluation k-1 and the
launch of evaluation k are charged to experiment k (the reasoning/editing that
produced it). Experiment 0 (baseline) includes setup. Tokens after the last
evaluation launch are reported as "tail". Re-evaluations of an already-seen
commit (e.g. after an infra failure) are folded into that experiment.

Per-message cost is computed from each assistant message's usage at list prices;
the sum of Claude Code's own `result.total_cost_usd` (which also covers side
calls such as compaction) is printed as a cross-check.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# $ per million tokens: (input, output). Cache reads 0.1x input,
# 5-minute cache writes 1.25x input, 1-hour cache writes 2x input.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def _price(model: str) -> tuple[float, float]:
    for key in sorted(PRICES, key=len, reverse=True):
        if model.startswith(key):
            return PRICES[key]
    raise KeyError(f"No price for model {model!r}; add it to PRICES")


def message_cost(model: str, usage: dict) -> float:
    p_in, p_out = _price(model)
    cc = usage.get("cache_creation") or {}
    w5 = cc.get("ephemeral_5m_input_tokens")
    w1h = cc.get("ephemeral_1h_input_tokens")
    if w5 is None and w1h is None:
        w5, w1h = usage.get("cache_creation_input_tokens", 0), 0
    return (usage.get("input_tokens", 0) * p_in
            + usage.get("output_tokens", 0) * p_out
            + usage.get("cache_read_input_tokens", 0) * p_in * 0.1
            + (w5 or 0) * p_in * 1.25
            + (w1h or 0) * p_in * 2.0) / 1e6


def load_messages(stream_path: Path):
    """Return [(time, model, usage)] deduplicated by message id, plus result totals."""
    msgs: dict[str, dict] = {}
    result_cost, first_time = 0.0, None
    for line in stream_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:  # partial last line while the run is live
            continue
        t = event.get("_recv_time")
        if first_time is None and t is not None:
            first_time = t
        if event.get("type") == "result":
            result_cost += event.get("total_cost_usd") or 0.0
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        mid, usage = message.get("id"), message.get("usage")
        if not mid or not usage or message.get("model") == "<synthetic>":
            continue
        entry = msgs.setdefault(mid, {"time": t, "model": message.get("model"), "usage": {}})
        # Content blocks of one message arrive as separate events; keep the max per field.
        for k, v in usage.items():
            if isinstance(v, (int, float)):
                entry["usage"][k] = max(entry["usage"].get(k, 0), v)
            elif isinstance(v, dict) and k == "cache_creation":
                cc = entry["usage"].setdefault(k, {})
                for kk, vv in v.items():
                    cc[kk] = max(cc.get(kk, 0), vv or 0)
    return list(msgs.values()), result_cost, first_time


def run_cost(run_dir: Path) -> float:
    """Estimated API cost (USD) of the agent so far, at list prices."""
    stream = Path(run_dir) / "agent_stream.jsonl"
    if not stream.exists():
        return 0.0
    messages, _, _ = load_messages(stream)
    return sum(message_cost(m["model"], m["usage"]) for m in messages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--csv", default=None, help="default: <run_dir>/cost_report.csv")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_cfg = json.loads((run_dir / "run.json").read_text())
    messages, result_cost, t0 = load_messages(run_dir / "agent_stream.jsonl")
    records = [json.loads(l) for l in (run_dir / "experiment_log.jsonl").read_text().splitlines()
               if l.strip()] if (run_dir / "experiment_log.jsonl").exists() else []

    # One row per experiment index; boundaries are the first launch of each experiment.
    exps: dict[int, dict] = {}
    for rec in records:
        row = exps.setdefault(rec["exp"], {"exp": rec["exp"], "commit": rec["commit"][:12],
                                           "n_evals": 0, "eval_seconds": 0.0})
        if rec["event"] == "start":
            row.setdefault("launch", rec["time"])
            row["n_evals"] += 1
            row["_start"] = rec["time"]
        else:
            row["end"] = rec["time"]
            row["eval_seconds"] += rec["time"] - row.get("_start", rec["time"])
            row["score"] = rec.get("score")
    rows = sorted(exps.values(), key=lambda r: r["launch"])
    starts = [t for t in [t0] + [r["launch"] for r in rows] if t is not None]
    t0 = min(starts) if starts else 0.0

    boundaries = [r["launch"] for r in rows]
    costs = [0.0] * (len(rows) + 1)  # last slot = tail
    for m in messages:
        idx = next((i for i, b in enumerate(boundaries) if m["time"] <= b), len(rows))
        costs[idx] += message_cost(m["model"], m["usage"])

    cum_cost, best = 0.0, None
    out_rows = []
    for row, cost in zip(rows, costs):
        cum_cost += cost
        score = row.get("score")
        if score is not None:
            best = score if best is None else max(best, score)
        out_rows.append({
            "exp": row["exp"], "commit": row["commit"],
            "score": "" if score is None else f"{score:.6f}",
            "best_so_far": "" if best is None else f"{best:.6f}",
            "n_evals": row["n_evals"],
            "eval_minutes": f"{row['eval_seconds'] / 60:.1f}",
            "cum_wall_hours": f"{(row.get('end', row['launch']) - t0) / 3600:.3f}",
            "cost_usd": f"{cost:.4f}", "cum_cost_usd": f"{cum_cost:.4f}",
        })

    csv_path = Path(args.csv) if args.csv else run_dir / "cost_report.csv"
    if out_rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
            writer.writeheader()
            writer.writerows(out_rows)

    n_candidates = sum(1 for r in out_rows if r["exp"] != 0)
    total = sum(costs)
    print(f"run: {run_cfg['tag']}  model: {run_cfg['model']}")
    print(f"experiments evaluated: {n_candidates}/{run_cfg['max_experiments']} (+ baseline)")
    if run_cfg.get("max_cost_usd") is not None:
        print(f"cost budget: ${total:.2f} of ${float(run_cfg['max_cost_usd']):.2f}")
    print(f"estimated API cost: ${total:.2f} (tail after last eval launch: ${costs[-1]:.2f})")
    print(f"Claude Code reported cost (sum of result events): ${result_cost:.2f}")
    if n_candidates:
        print(f"mean cost per experiment: ${total / n_candidates:.3f}")
    if out_rows:
        print(f"wall clock: {out_rows[-1]['cum_wall_hours']} h; best score: {out_rows[-1]['best_so_far']}")
        print(f"per-experiment table: {csv_path}")


if __name__ == "__main__":
    main()
