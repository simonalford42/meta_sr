#!/usr/bin/env python3
"""Aggregate empbench_run result JSONs into a recovery / evals-to-solve summary.

Reports BOTH the official metric (buggy for Planck) and the robust numeric
recovery, plus evals-to-solve and frontiers for manual inspection.

Usage:
    python scripts/empbench_analyze.py runs_local/main [runs_local/phaseB ...]
    python scripts/empbench_analyze.py --frontiers runs_local/main   # dump frontiers
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict


def variant_key(r, path):
    """Group key: (dataset, method-or-variant-label)."""
    ds = r.get("dataset", "?")
    method = r.get("method", "?")
    if method == "custom":
        # derive variant label from filename: <short>_<label>_r<seed>.json
        base = os.path.basename(path)
        m = re.match(r"(planck|rydberg)_(.+)_r\d+\.json", base)
        label = m.group(2) if m else "custom"
        # include op-set in label
        mm = r.get("method_meta", {})
        uops = mm.get("unary_operators")
        ps = r.get("pysr_kwargs", {})
        extra = f" maxsize={ps.get('maxsize')}" if ps.get("maxsize") not in (None, 40) else ""
        return ds, f"custom:{label}", f"unary={uops}{extra}"
    return ds, method, ""


def fmt_evals(v):
    if v is None:
        return "  --   "
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}k"
    return str(int(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--frontiers", action="store_true",
                    help="also print final frontiers (manual inspection)")
    ap.add_argument("--robust-only-frontiers", action="store_true")
    a = ap.parse_args()

    files = []
    for d in a.dirs:
        files += sorted(glob.glob(os.path.join(d, "*.json")))
    files = [f for f in files if os.path.basename(f) not in
             ("jobs.json", "jobs_evolve.json", "manifest.json")]

    groups = defaultdict(list)
    meta = {}
    for f in files:
        try:
            r = json.loads(open(f).read())
        except Exception:
            continue
        if "milestones" not in r:
            continue
        ds, label, note = variant_key(r, f)
        groups[(ds, label)].append((f, r))
        meta[(ds, label)] = note

    print("=" * 100)
    print(f"{'dataset':17s} {'method/variant':26s} {'n':>2s} {'ROBUST':>7s} "
          f"{'rob_evals':>9s} {'OFFICIAL':>8s} {'off_evals':>9s} {'bestR2(med)':>11s}  note")
    print("-" * 100)
    for (ds, label) in sorted(groups):
        runs = groups[(ds, label)]
        n = len(runs)
        rob_solved = [r for _, r in runs if r.get("robust_solved_at") is not None]
        off_solved = [r for _, r in runs if r.get("official_solved_at") is not None]
        rob_evals = sorted(r["robust_solved_at"] for r in rob_solved)
        off_evals = sorted(r["official_solved_at"] for r in off_solved)
        med_rob = rob_evals[len(rob_evals)//2] if rob_evals else None
        med_off = off_evals[len(off_evals)//2] if off_evals else None
        # best val r2 across final milestone of each run
        bestr2s = []
        for _, r in runs:
            ms = r.get("milestones", [])
            vals = [m.get("best_val_r2") for m in ms if m.get("best_val_r2") is not None]
            if vals:
                bestr2s.append(max(vals))
        bestr2s.sort()
        med_r2 = bestr2s[len(bestr2s)//2] if bestr2s else float("nan")
        ncomplete = sum(1 for _, r in runs if r.get("completed"))
        print(f"{ds:17s} {label:26s} {n:>2d} {len(rob_solved):>3d}/{n:<3d} "
              f"{fmt_evals(med_rob):>9s} {len(off_solved):>3d}/{n:<3d} {fmt_evals(med_off):>9s} "
              f"{med_r2:>11.4f}  {meta.get((ds,label),'')}"
              f"{'' if ncomplete==n else f'  [{ncomplete}/{n} complete]'}")

    # robust-matched expressions (the actual discovered laws)
    print("\n" + "=" * 100)
    print("ROBUSTLY-RECOVERED expressions (true recovery):")
    any_rob = False
    for (ds, label) in sorted(groups):
        for f, r in groups[(ds, label)]:
            for m in r.get("milestones", []):
                if m.get("robust") and m.get("matched_robust"):
                    any_rob = True
                    print(f"  [{ds} / {label} / seed{r.get('seed')}] @~{fmt_evals(m.get('num_evals') or m.get('milestone'))} evals:")
                    print(f"      {m['matched_robust']}")
                    break
    if not any_rob:
        print("  (none — no run robustly recovered either law)")

    if a.frontiers:
        print("\n" + "=" * 100)
        print("FINAL FRONTIERS (manual inspection):")
        for (ds, label) in sorted(groups):
            for f, r in groups[(ds, label)]:
                print(f"\n--- {ds} / {label} / seed{r.get('seed')}  ({os.path.basename(f)}) ---")
                print(f"    GT: {r.get('gt')}")
                for row in r.get("final_frontier", []):
                    if a.robust_only_frontiers and not row.get("robust_match"):
                        continue
                    flag = "  <== ROBUST MATCH" if row.get("robust_match") else ""
                    print(f"    c={row['complexity']:>2} R2={row['val_r2']} | {row['equation']}{flag}")


if __name__ == "__main__":
    main()
