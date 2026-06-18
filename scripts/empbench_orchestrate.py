#!/usr/bin/env python3
"""Run a batch of empbench_run.py jobs with a concurrency cap (no SLURM).

Each job is an isolated single-core process (JULIA_NUM_THREADS=1). We keep
`--max-parallel` running at once to use the node's cores. A manifest of all jobs
and their exit status / result-path is written so progress can be inspected
mid-flight.

Job spec file: JSON list of dicts, each a kwargs set for empbench_run.py, e.g.
  [{"method":"baseline","dataset":"empirical_rydberg","run_index":0,
    "max_evals":1e7,"out":"runs_local/main/rydberg_baseline_r0.json"}, ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def build_cmd(job):
    cmd = [sys.executable, str(REPO / "scripts" / "empbench_run.py")]
    for k, v in job.items():
        if k == "log":
            continue
        flag = "--" + k.replace("_", "-")
        cmd += [flag, str(v)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSON job list")
    ap.add_argument("--max-parallel", type=int, default=6)
    ap.add_argument("--log-dir", default="runs_local/logs")
    ap.add_argument("--manifest", default="runs_local/manifest.json")
    args = ap.parse_args()

    jobs = json.loads(Path(args.jobs).read_text())
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)

    running = {}  # proc -> (job, logf, start)
    done = []
    # Resume: skip jobs whose result already exists and is marked completed.
    pending = []
    skipped = 0
    for i, job in enumerate(jobs):
        out = job.get("out")
        if out and Path(out).exists():
            try:
                r = json.loads(Path(out).read_text())
                if r.get("completed"):
                    skipped += 1
                    done.append({"job": f"{job.get('dataset')}_{job.get('method')}_r{job.get('run_index')}",
                                 "out": out, "returncode": 0, "seconds": 0,
                                 "official_solved_at": r.get("official_solved_at"),
                                 "robust_solved_at": r.get("robust_solved_at"),
                                 "resumed": True})
                    continue
            except Exception:
                pass
        pending.append((i, job))
    if skipped:
        print(f"[resume] skipping {skipped} already-completed jobs; "
              f"{len(pending)} to run", flush=True)
    t0 = time.time()

    def label(job):
        return f"{job.get('dataset','?')}_{job.get('method','?')}_r{job.get('run_index',0)}"

    def write_manifest():
        Path(args.manifest).write_text(json.dumps({
            "n_jobs": len(jobs), "n_done": len(done),
            "n_running": len(running), "elapsed_s": round(time.time() - t0, 1),
            "done": done,
            "running": [label(j) for j, _, _ in running.values()],
        }, indent=2))

    while pending or running:
        while pending and len(running) < args.max_parallel:
            ji, job = pending.pop(0)
            lbl = label(job) + f"_{ji}"
            logp = Path(args.log_dir) / f"{lbl}.log"
            logf = open(logp, "w")
            cmd = build_cmd(job)
            env = dict(os.environ, JULIA_NUM_THREADS="1", OMP_NUM_THREADS="1")
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                 cwd=str(REPO), env=env)
            running[p] = (job, logf, time.time())
            print(f"[launch] {lbl}  pid={p.pid}  (running={len(running)}, "
                  f"pending={len(pending)})", flush=True)
            write_manifest()

        time.sleep(5)
        for p in list(running.keys()):
            if p.poll() is not None:
                job, logf, st = running.pop(p)
                logf.close()
                dt = round(time.time() - st, 1)
                rec = {"job": label(job), "out": job.get("out"),
                       "returncode": p.returncode, "seconds": dt}
                # pull solved-at from result if present
                try:
                    r = json.loads(Path(job["out"]).read_text())
                    rec["official_solved_at"] = r.get("official_solved_at")
                    rec["robust_solved_at"] = r.get("robust_solved_at")
                    rec["error"] = r.get("error")
                except Exception:
                    pass
                done.append(rec)
                print(f"[done]   {rec['job']}  rc={p.returncode}  {dt}s  "
                      f"robust_solved_at={rec.get('robust_solved_at')}  "
                      f"official_solved_at={rec.get('official_solved_at')}", flush=True)
                write_manifest()

    write_manifest()
    print(f"\nALL DONE: {len(done)} jobs in {round(time.time()-t0,1)}s", flush=True)
    # concise summary
    for rec in done:
        print(f"  {rec['job']:42s} rc={rec['returncode']} "
              f"rob@{rec.get('robust_solved_at')} off@{rec.get('official_solved_at')}")


if __name__ == "__main__":
    main()
