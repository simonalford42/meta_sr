#!/usr/bin/env python3
"""Dump experiments sorted by change complexity (1=hyperparam .. 4=big algo change)."""
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "SymbolicRegression.jl"
TSV = HERE / "results.tsv"
OUT = HERE / "experiments_sorted.diff"

STATUS_MARK = {"keep": "[KEEP]", "discard": "[DISCARD]", "crash": "[CRASH]"}

LEVEL_NAME = {
    1: "LEVEL 1 - hyperparameter change",
    2: "LEVEL 2 - tweak approach",
    3: "LEVEL 3 - new approach added",
    4: "LEVEL 4 - big algorithmic change",
}

# Ratings derived from commit descriptions + git shortstats.
# 1 = pure numeric knob, 2 = small logic tweak, 3 = new logic block, 4 = big algo change.
RATING = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,
    6: 3,         # log-scale constant restarts (new restart logic)
    7: 1, 8: 1, 9: 1, 10: 1, 11: 1,
    12: 3,        # 30% worst-cost replacement (new replacement strategy)
    13: 2,        # temperature-dependent crossover formula
    14: 1,
    15: 2,        # elitist 2nd crossover parent
    16: 2,        # scale-aware constant restart perturbation
    17: 2,        # operator-biased crossover selection
    18: 1,
    19: 3,        # alternating log-scale + scale-aware restart scheme
    20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1,
    26: 2,        # random 2nd crossover parent
    27: 1,
    28: 2,        # hybrid random/tournament 2nd parent
    29: 3,        # crossover replaces oldest-with-better (refactor of replacement)
    30: 2,        # mutate_constant bump + replacement logic touched
    31: 1, 32: 1,
    33: 2,        # wide random 2nd restart T(2)*eps
    34: 1,
    35: 2,        # warm-start 2nd BFGS restart
    36: 1, 37: 1, 38: 1,
    39: 3,        # 10% worst-replacement survival (new survival rule)
    40: 1, 41: 1,
    42: 3,        # non-leaf-biased crossover point selection
    43: 1, 44: 1, 45: 1, 46: 1, 47: 1, 48: 1, 49: 1,
    50: 3,        # BFGS on crossover babies (new optimization hook)
    51: 1, 52: 1, 53: 1, 54: 1, 55: 1,
    56: 2,        # progressive BFGS restart scales
    57: 3,        # constant-biased crossover (new crossover variant)
    58: 2,        # early BFGS restart termination
    59: 1, 60: 1,
}


def git_show(commit: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "show", "--stat", "--patch", commit],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        return f"(git show failed: {e.stderr.strip()})\n"


def main() -> int:
    with TSV.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    for r in rows:
        r["_exp"] = int(r["exp"])
        r["_rating"] = RATING.get(r["_exp"], 1)

    # Sort: most complex first; within a level, keep exp order.
    rows.sort(key=lambda r: (-r["_rating"], r["_exp"]))

    counts = Counter(r["_rating"] for r in rows)

    with OUT.open("w") as f:
        f.write("=" * 100 + "\n")
        f.write(f"  AUTORESEARCH EXPERIMENTS, SORTED BY CHANGE COMPLEXITY  ({len(rows)} total)\n")
        f.write("=" * 100 + "\n\n")

        f.write("Complexity counts:\n")
        for lvl in (4, 3, 2, 1):
            f.write(f"  {LEVEL_NAME[lvl]:<40}  {counts.get(lvl, 0):>3}\n")
        f.write("\n")

        f.write(f"{'lvl':>3}  {'exp':>4}  {'commit':<10}  {'score':>8}  {'ok/fail':>8}  {'status':<8}  description\n")
        f.write("-" * 100 + "\n")
        for r in rows:
            f.write(
                f"{r['_rating']:>3}  {r['exp']:>4}  {r['commit']:<10}  {r['score']:>8}  "
                f"{r['datasets_ok']+'/'+r['datasets_fail']:>8}  "
                f"{r['status']:<8}  {r['description']}\n"
            )
        f.write("\n")

        current_level = None
        for r in rows:
            if r["_rating"] != current_level:
                current_level = r["_rating"]
                f.write("\n" + "#" * 100 + "\n")
                f.write(f"#  {LEVEL_NAME[current_level]}  ({counts[current_level]} experiments)\n")
                f.write("#" * 100 + "\n")

            mark = STATUS_MARK.get(r["status"], f"[{r['status'].upper()}]")
            header = (
                f"exp {r['exp']}  |  {r['commit']}  |  score {r['score']}  "
                f"|  ok {r['datasets_ok']} / fail {r['datasets_fail']}  |  {mark}  "
                f"|  level {r['_rating']}"
            )
            f.write("\n" + "=" * 100 + "\n")
            f.write(header + "\n")
            f.write(f"description: {r['description']}\n")
            f.write("=" * 100 + "\n\n")
            f.write(git_show(r["commit"]))
            f.write("\n")

    print(f"wrote {OUT}  ({len(rows)} experiments)")
    for lvl in (4, 3, 2, 1):
        print(f"  {LEVEL_NAME[lvl]:<40}  {counts.get(lvl, 0):>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
