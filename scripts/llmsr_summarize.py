#!/usr/bin/env python3
"""Summarize scripts/llmsr_run.py results against LLM-SR paper Table 1 (NMSE).

Per run, the reported equation is the frontier equation with the lowest train
NMSE (raw for baseline; affine-rescaled a+b*f for evolved, whose loss is
affine-invariant; `--raw-evolve` to disable). Across the 5 seeds we report the
median and the "paper-style" best-of-5 (seed chosen by train NMSE).

    python scripts/llmsr_summarize.py runs/llmsr-pysr-30min
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PAPER = {  # Table 1, (ID, OOD)
    "PySR (paper)": {"oscillator1": (0.0009, 0.3106), "oscillator2": (0.0002, 0.0098),
                     "bactgrow": (0.0376, 1.0141), "stressstrain": (0.0331, 0.1304)},
    "uDSR (paper)": {"oscillator1": (0.0003, 0.0007), "oscillator2": (0.0032, 0.0015),
                     "bactgrow": (0.3322, 5.4584), "stressstrain": (0.0502, 0.1761)},
    "LLM-SR GPT-3.5 (paper)": {"oscillator1": (4.65e-7, 0.0005), "oscillator2": (2.12e-7, 3.81e-5),
                               "bactgrow": (0.0214, 0.0264), "stressstrain": (0.0210, 0.0516)},
    "LLM-SR Mixtral (paper)": {"oscillator1": (7.89e-8, 0.0002), "oscillator2": (0.0030, 0.0291),
                               "bactgrow": (0.0026, 0.0037), "stressstrain": (0.0162, 0.0946)},
}
PROBLEMS = ("oscillator1", "oscillator2", "bactgrow", "stressstrain")


def pick(run, affine):
    pre = "aff_" if affine else ""
    rows = [r for r in run["frontier"] if np.isfinite(r.get(pre + "nmse_train", np.inf))]
    r = min(rows, key=lambda r: r[pre + "nmse_train"])
    return r[pre + "nmse_train"], r[pre + "nmse_id"], r[pre + "nmse_ood"], r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--raw-evolve", action="store_true")
    args = ap.parse_args()
    runs = defaultdict(list)
    for f in sorted(Path(args.run_dir).glob("*_r*.json")):
        d = json.loads(f.read_text())
        if d.get("completed"):
            runs[(d["method"], d["problem"])].append(d)

    def fmt(x):
        return f"{x:.3g}" if np.isfinite(x) else "inf"

    short = {"oscillator1": "osc1", "oscillator2": "osc2", "bactgrow": "ecoli", "stressstrain": "stress"}
    hdr = f"{'method':28s}" + "".join(f"{short[p]+' ID':>15s}{short[p]+' OOD':>15s}" for p in PROBLEMS)
    print("NMSE (lower is better)\n" + hdr)
    for name, tab in PAPER.items():
        print(f"{name:28s}" + "".join(f"{fmt(tab[p][0]):>15s}{fmt(tab[p][1]):>15s}" for p in PROBLEMS))
    for method, label in (("baseline", "PySR (ours)"), ("evolve", "PySR++ 709715")):
        aff = method == "evolve" and not args.raw_evolve
        for stat in ("median", "best-of-n"):
            line = f"{label + ' ' + stat:28s}"
            for p in PROBLEMS:
                rs = [pick(r, aff) for r in runs.get((method, p), [])]
                if not rs:
                    line += f"{'-':>15s}{'-':>15s}"
                    continue
                if stat == "median":
                    i, o = np.median([r[1] for r in rs]), np.median([r[2] for r in rs])
                else:
                    b = min(rs, key=lambda r: r[0])
                    i, o = b[1], b[2]
                line += f"{fmt(i):>15s}{fmt(o):>15s}"
            print(line)
    print("\nseeds per cell:", {f"{m}/{p}": len(v) for (m, p), v in sorted(runs.items())})
    print("\nBest-train equation per method/problem:")
    for (m, p), v in sorted(runs.items()):
        tr, i, o, r = min((pick(x, m == "evolve" and not args.raw_evolve) for x in v), key=lambda t: t[0])
        print(f"  {m:8s} {p:12s} train={fmt(tr)} ID={fmt(i)} OOD={fmt(o)} c={r['complexity']}  {r['equation'][:160]}")


if __name__ == "__main__":
    main()
