#!/usr/bin/env python3
"""Comprehensive, verified EmpiricalBench report.

For every run JSON: determine VERIFIED recovery (numeric match on the sound grid
AND structural/symbolic confirmation via scripts/empbench_verify), the recovering
expression, verified evals-to-solve (first milestone whose matched_robust
verifies), the official-metric result, and best val R². Then aggregate per
(dataset, method/variant).

Usage: python scripts/empbench_report.py runs_local/main runs_local/phaseB
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from empbench_verify import verdict as _verdict  # noqa: E402
from empbench_lib import numeric_recovery as _numrec  # noqa: E402

# Memoize the expensive sympy-parsing calls — the same equation strings recur
# across many milestones, so this turns ~500k calls into a few thousand.
_NR_CACHE = {}
_V_CACHE = {}


def numeric_recovery(eq, ds):
    k = (ds, eq)
    if k not in _NR_CACHE:
        _NR_CACHE[k] = _numrec(eq, ds)
    return _NR_CACHE[k]


def verdict(eq, ds):
    k = (ds, eq)
    if k not in _V_CACHE:
        _V_CACHE[k] = _verdict(eq, ds)
    return _V_CACHE[k]


def vkey(r, path):
    ds = r.get("dataset", "?")
    method = r.get("method", "?")
    if method == "custom":
        base = os.path.basename(path)
        # filenames are "<short>_<label>_r<seed>.json" (phaseB) or
        # "<label>_r<seed>.json" (rydextra); strip optional planck_/rydberg_
        # prefix so the same variant from both sources groups together.
        m = re.match(r"(?:(?:planck|rydberg)_)?(.+?)_r\d+\.json", base)
        return ds, "custom:" + (m.group(1) if m else "custom")
    return ds, method


def fmt_evals(v):
    if v is None:
        return "  --  "
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}k"
    return str(int(v))


def _first_verified_expr(frontier, ds, cache):
    """Return the simplest VERIFIED-recovering expr on a frontier, else None.
    Cheap numeric prefilter (add_rel<5e-3) before the (slow, refit-based)
    verdict(); cache verdicts by equation string across milestones."""
    for row in sorted(frontier, key=lambda d: d.get("complexity", 999)):
        eq = row["equation"]
        if eq not in cache:
            nr = numeric_recovery(eq, ds)
            ar = nr.get("add_rel")
            cache[eq] = (verdict(eq, ds)["TRUE_RECOVERY"]
                         if (ar is not None and ar < 2e-3) else False)
        if cache[eq]:
            return eq
    return None


def verified_recovery_for_run(r):
    """Return (recovered_bool, recovering_expr, verified_solved_at_evals).

    Uses per-milestone frontiers when present (precise evals-to-solve); falls
    back to final_frontier + matched_robust for older JSONs.
    """
    ds = r["dataset"]
    ms_list = r.get("milestones", [])
    cache = {}
    have_ms_frontiers = any("frontier" in m for m in ms_list)

    if have_ms_frontiers:
        solved_at = None
        rec_expr = None
        for m in ms_list:
            e = _first_verified_expr(m.get("frontier", []), ds, cache)
            if e is not None:
                solved_at = m.get("num_evals") or m.get("milestone")
                rec_expr = e
                break
        # count as recovered-by-1e7 if verified at ANY milestone
        final_rec = rec_expr is not None
        if rec_expr is None:
            rec_expr = _first_verified_expr(ms_list[-1].get("frontier", []), ds, cache) if ms_list else None
            final_rec = rec_expr is not None
        return final_rec, rec_expr, solved_at

    # ---- fallback (older JSONs without per-milestone frontiers) ----
    rec_expr = _first_verified_expr(r.get("final_frontier", []), ds, cache)
    final_rec = rec_expr is not None
    solved_at = None
    for m in ms_list:
        mr = m.get("matched_robust")
        if mr and verdict(mr, ds)["TRUE_RECOVERY"]:
            solved_at = m.get("num_evals") or m.get("milestone")
            break
    if final_rec and solved_at is None and ms_list:
        solved_at = ms_list[-1].get("milestone")
    return final_rec, rec_expr, solved_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--save", default=None, help="write per-run verdicts JSON here")
    a = ap.parse_args()

    files = []
    for d in a.dirs:
        files += sorted(glob.glob(os.path.join(d, "*.json")))
    files = [f for f in files if os.path.basename(f) not in
             ("jobs.json", "jobs_evolve.json", "manifest.json")]

    per_run = []
    groups = defaultdict(list)
    for f in files:
        try:
            r = json.loads(open(f).read())
        except Exception:
            continue
        if "milestones" not in r:
            continue
        rec, expr, solved = verified_recovery_for_run(r)
        bestr2 = max([m.get("best_val_r2") for m in r["milestones"]
                      if m.get("best_val_r2") is not None] or [float("nan")])
        # proximity-to-recovery: smallest additive-residual any frontier expr got
        # (across all milestones). For Planck, restrict to exprs WITH exp so we
        # measure proximity of the *right structure*, not of a smooth approximation.
        best_add = float("inf")
        for m in r.get("milestones", []):
            for row in m.get("frontier", []):
                nr = numeric_recovery(row["equation"], r["dataset"])
                ar = nr.get("add_rel")
                if ar is None:
                    continue
                if r["dataset"] == "empirical_planck":
                    from empbench_verify import planck_has_structure
                    if not planck_has_structure(row["equation"]).get("structure_plausible"):
                        continue
                best_add = min(best_add, ar)
        rr = {"file": os.path.basename(f), "dataset": r["dataset"],
              "key": vkey(r, f)[1], "seed": r.get("seed"),
              "completed": r.get("completed"),
              "verified_recovery": rec, "recovering_expr": expr,
              "verified_solved_at": solved,
              "official_solved_at": r.get("official_solved_at"),
              "best_add_rel": best_add,
              "best_val_r2": bestr2}
        per_run.append(rr)
        groups[(r["dataset"], vkey(r, f)[1])].append(rr)

    def _med(xs):
        xs = sorted(xs)
        return xs[len(xs)//2] if xs else float("nan")

    print("=" * 116)
    print(f"{'dataset':17s} {'method/variant':27s} {'n':>2s} {'VERIFIED':>8s} "
          f"{'evals→solve':>11s} {'closest add_rel':>16s} {'OFFICIAL':>8s} {'bestR2(med)':>11s}")
    print("-" * 116)
    for (ds, key) in sorted(groups):
        runs = groups[(ds, key)]
        n = len(runs)
        rec = [x for x in runs if x["verified_recovery"]]
        offi = [x for x in runs if x["official_solved_at"] is not None]
        med_solve = _med([x["verified_solved_at"] for x in rec if x["verified_solved_at"]]) if rec else None
        med_r2 = _med([x["best_val_r2"] for x in runs])
        finite_add = [x["best_add_rel"] for x in runs if x["best_add_rel"] != float("inf")]
        med_add = _med(finite_add) if finite_add else float("inf")
        best_add = min(finite_add) if finite_add else float("inf")
        nc = sum(1 for x in runs if x["completed"])
        tag = "" if nc == n else f" [{nc}/{n}]"
        add_s = f"{med_add:.1e}(min {best_add:.0e})" if finite_add else "  none  "
        print(f"{ds:17s} {key:27s} {n:>2d} {len(rec):>3d}/{n:<3d}  "
              f"{fmt_evals(med_solve) if rec else '   --   ':>11s} {add_s:>16s} "
              f"{len(offi):>3d}/{n:<3d}  {med_r2:>10.4f}{tag}")

    print("\n" + "=" * 104)
    print("VERIFIED recovering expressions:")
    any_rec = False
    for (ds, key) in sorted(groups):
        for x in groups[(ds, key)]:
            if x["verified_recovery"]:
                any_rec = True
                print(f"  [{ds} / {key} / seed{x['seed']}] @~{fmt_evals(x['verified_solved_at'])} evals:")
                print(f"      {x['recovering_expr']}")
    if not any_rec:
        print("  (none verified)")

    if a.save:
        json.dump(per_run, open(a.save, "w"), indent=2)
        print(f"\nwrote {a.save}")


if __name__ == "__main__":
    main()
