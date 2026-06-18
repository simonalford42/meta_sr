#!/usr/bin/env python3
"""SLURM array worker: run jobs[$index] from a jobs.json via empbench_run.main().

Resume-safe: if the job's --out already exists and is marked completed, skip.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # so `import empbench_run` works
sys.path.insert(0, str(HERE.parent))     # repo root


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--index", type=int, required=True)
    a = ap.parse_args()

    jobs = json.loads(Path(a.jobs).read_text())
    if a.index >= len(jobs):
        print(f"index {a.index} >= n_jobs {len(jobs)}; nothing to do")
        return
    job = jobs[a.index]
    out = job.get("out")
    if out and Path(out).exists():
        try:
            r = json.loads(Path(out).read_text())
            if r.get("completed"):
                print(f"SKIP (already completed): {out}", flush=True)
                return
        except Exception:
            pass

    argv = ["empbench_run.py"]
    for k, v in job.items():
        if k == "log":
            continue
        argv += ["--" + k.replace("_", "-"), str(v)]
    print("RUN:", " ".join(argv), flush=True)
    sys.argv = argv
    import empbench_run
    empbench_run.main()


if __name__ == "__main__":
    main()
