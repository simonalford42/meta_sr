#!/usr/bin/env python3
"""How long does the ground-truth (sympy) equivalence check take?

FullSR worker logs bracket ``domain.check_solved`` with two elapsed-stamped
lines::

    [ds] Search returned N frontier rows, n_evals=... (elapsed=A)
    [ds] Validation R²=..., gt=..., best=...            (elapsed=B)

B - A is the GT check plus a cheap numeric R² pass, so it upper-bounds the
sympy cost. This script reports the distribution over every task log of one or
more FullSR eval runs, and lists the tasks whose logs stop after "Search
returned" (i.e. the check never finished before the job was killed).

Usage: python scripts/gt_check_timing.py runs/93797 runs/93798 [...]
"""
import re
import sys
from pathlib import Path
from statistics import mean, median

RET = re.compile(r"\] Search returned .*\(elapsed=([\d.]+)s\)")
VAL = re.compile(r"\] Validation R²=.*\(elapsed=([\d.]+)s\)")
DS = re.compile(r"dataset=(\S+),")


def scan(run_dir: Path, batch: str = "*"):
    deltas, hangs = [], []
    for log in run_dir.glob(f"slurm_fullsr/{batch}/logs/*.out"):
        text = log.read_text(errors="replace")
        ret = RET.search(text)
        if not ret:
            continue
        val = VAL.search(text)
        ds = DS.search(text)
        name = ds.group(1) if ds else "?"
        if val:
            deltas.append((float(val.group(1)) - float(ret.group(1)), name, log))
        else:
            hangs.append((name, log))
    return deltas, hangs


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--batch=")]
    batch = next((a.split("=", 1)[1] for a in sys.argv[1:]
                  if a.startswith("--batch=")), "*")
    runs = [Path(a) for a in argv] or [Path("runs/93797")]
    all_d, all_h = [], []
    for run in runs:
        d, h = scan(run, batch)
        all_d += d
        all_h += h
        vals = [x[0] for x in d]
        if vals:
            print(f"{run.name}: n={len(vals)} mean={mean(vals):.2f}s "
                  f"median={median(vals):.2f}s p99={pct(vals, 99):.2f}s "
                  f"max={max(vals):.2f}s  unfinished={len(h)}")
    vals = [x[0] for x in all_d]
    if not vals:
        print("no timed GT checks found")
        return
    print(f"\nALL n={len(vals)}")
    for p in (50, 90, 95, 99, 99.9):
        print(f"  p{p:<5} {pct(vals, p):8.2f}s")
    print(f"  max    {max(vals):8.2f}s   mean {mean(vals):.2f}s")
    for thresh in (1, 5, 10, 30, 60, 120, 300):
        n = sum(1 for v in vals if v > thresh)
        print(f"  >{thresh:>4}s: {n:6d}  ({100 * n / len(vals):.3f}%)")
    print("\nSlowest finished checks:")
    for v, name, log in sorted(all_d, reverse=True)[:15]:
        print(f"  {v:8.1f}s  {name:22s} {log}")
    if all_h:
        print(f"\nNever finished ({len(all_h)}):")
        for name, log in all_h[:20]:
            print(f"  {name:22s} {log}")


if __name__ == "__main__":
    main()
