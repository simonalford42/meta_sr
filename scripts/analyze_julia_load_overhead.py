#!/usr/bin/env python3
"""Per-task Julia/PySR load time ("PySR loaded in Xs") across SLURM eval batches.

Usage: python scripts/analyze_julia_load_overhead.py LABEL=GLOB [LABEL=GLOB ...]
  GLOB matches slurm_pysr/eval_* batch directories (each with logs/*.out).
Prints, per label, over all batches: #tasks, #tasks that never finished loading,
and load-time quantiles; plus the per-batch first-load (max) distribution, which
is where a per-commit recompile shows up.
"""
import glob
import re
import sys

import numpy as np

PAT = re.compile(r"PySR loaded in ([0-9.]+)s")


def batch_loads(batch):
    loads, stuck = [], 0
    for f in glob.glob(f"{batch}/logs/*.out"):
        text = open(f, errors="replace").read()
        found = [float(x) for x in PAT.findall(text)]
        if found:
            loads.append(found[0])
        elif "Loading PySR" in text:
            stuck += 1
    return loads, stuck


for arg in sys.argv[1:]:
    label, pattern = arg.split("=", 1)
    batches = sorted(glob.glob(pattern))
    all_loads, all_stuck, batch_max = [], 0, []
    for b in batches:
        loads, stuck = batch_loads(b)
        all_loads += loads
        all_stuck += stuck
        if loads:
            batch_max.append(max(loads))
    if not all_loads:
        print(f"{label}: no load lines in {len(batches)} batches")
        continue
    q = np.percentile(all_loads, [50, 90, 99])
    bm = np.percentile(batch_max, [50, 90]) if batch_max else [np.nan] * 2
    print(f"{label}: batches={len(batches)} tasks={len(all_loads)} never_loaded={all_stuck} "
          f"load p50={q[0]:.0f}s p90={q[1]:.0f}s p99={q[2]:.0f}s max={max(all_loads):.0f}s | "
          f"per-batch max p50={bm[0]:.0f}s p90={bm[1]:.0f}s")
