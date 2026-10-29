#!/usr/bin/env python3
"""Per-arm solve counts for POM rerun experiments.

Usage: python scripts/pom/summarize.py runs/pom-arms-...
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    d = Path(sys.argv[1])
    rows = defaultdict(lambda: defaultdict(list))  # ds -> arm -> [solved]
    r2s = defaultdict(lambda: defaultdict(list))
    for jf in d.glob("*.json"):
        r = json.loads(jf.read_text())
        arm = r.get("arm", r["loss"]["type"])
        rows[r["dataset"]][arm].append(r["solved"])
        # val R² of the most accurate frontier entry
        r2s[r["dataset"]][arm].append(max((f["r2_val"] or -1) for f in r["frontier"]))
    arms = sorted({a for v in rows.values() for a in v})
    print(f"{'dataset':28s} " + " ".join(f"{a:>14s}" for a in arms))
    tot = defaultdict(lambda: [0, 0])
    for ds in sorted(rows):
        cells = []
        for a in arms:
            s = rows[ds].get(a, [])
            tot[a][0] += sum(s)
            tot[a][1] += len(s)
            med = np.median(r2s[ds][a]) if r2s[ds].get(a) else float("nan")
            cells.append(f"{sum(s)}/{len(s)} r2={med:.4f}".rjust(14 + 8))
        print(f"{ds:28s} " + " ".join(cells))
    print()
    for a in arms:
        n_tasks = sum(1 for ds in rows if any(rows[ds].get(a, [])))
        print(f"{a:12s} runs solved {tot[a][0]}/{tot[a][1]}  tasks solved≥1: {n_tasks}")


if __name__ == "__main__":
    main()
