#!/usr/bin/env python3
"""Summarize scripts/lsrsynth_run.py results next to published LSR-Synth numbers.

Per run, the reported equation is the frontier equation with the lowest train
NMSE: train-only selection, the same rule as SR-Scientist and llm-srbench's LLM-SR
searcher. The evolved bundle's loss is affine-invariant, so its equations are
affine-rescaled on train first (`--raw-evolve` disables this). Acc values are
percent of problems, averaged over seeds; "overall" is pooled over all 129
problems. NMSE is the median over problems, taking each problem's seed-mean of
log10 NMSE.

Each block reports the metric that the papers it's compared against use (see
reports/lsr_synth_literature.md):
  A  LLM-SRBench / PiT-PO: strict Acc_0.1 + NMSE, ID test
  B  SR-Scientist: 5%-trimmed Acc_0.01 / Acc_0.001, ID test
  C  OOD test (no paper has a full numeric table; RESTART's numbers are unverified and leave out chem)

    python scripts/lsrsynth_summarize.py runs/<name> [--md runs/<name>/summary.md]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DOMAINS = ("chem_react", "bio_pop_growth", "phys_osc", "matsci")
SHORT = {"chem_react": "Chem", "bio_pop_growth": "Bio", "phys_osc": "Phys", "matsci": "MatSci"}

# LLM-SRBench Table 1 (Acc0.1 %, NMSE) and App. F (PySR, Acc only); PiT-PO Table 2.
PAPER_A = {
    "PySR [LLM-SRBench App.F]": {"chem_react": (41.67, None), "bio_pop_growth": (25.0, None),
                                 "phys_osc": (29.55, None), "matsci": (68.0, None)},
    "LaSR GPT-4o-mini [LLM-SRBench]": {"chem_react": (38.92, 9.11e-5), "bio_pop_growth": (20.83, 1.53e-4),
                                       "phys_osc": (31.81, 9.94e-4), "matsci": (72.04, 9.23e-6)},
    "LLM-SR Llama-3.1-8B [LLM-SRBench]": {"chem_react": (66.66, 8.01e-6), "bio_pop_growth": (58.33, 1.04e-6),
                                          "phys_osc": (34.09, 1.23e-4), "matsci": (88.12, 1.15e-7)},
    "LLM-SR GPT-4o-mini [LLM-SRBench]": {"chem_react": (52.77, 4.12e-6), "bio_pop_growth": (29.16, 3.06e-6),
                                         "phys_osc": (36.36, 7.62e-5), "matsci": (88.28, 3.21e-9)},
    "PiT-PO Llama-3.1-8B [PiT-PO]": {"chem_react": (77.78, 4.13e-7), "bio_pop_growth": (70.83, 9.37e-8),
                                     "phys_osc": (40.91, 6.57e-5), "matsci": (92.00, 1.18e-8)},
}
# SR-Scientist Table 1: (Acc0.01, Acc0.001) %, overall + per domain, mean of 3 runs.
PAPER_B = {
    "PySR [SR-Scientist]": {"overall": (29.46, 14.47), "matsci": (53.33, 22.67), "chem_react": (25.93, 11.11),
                            "bio_pop_growth": (16.67, 6.95), "phys_osc": (25.76, 16.67)},
    "uDSR [SR-Scientist]": {"overall": (29.46, 12.40), "matsci": (36.00, 8.00), "chem_react": (50.00, 25.00),
                            "bio_pop_growth": (29.17, 8.33), "phys_osc": (9.09, 6.82)},
    "LLM-SR Qwen3-480B [SR-Scientist]": {"overall": (41.08, 18.09), "matsci": (80.00, 52.00),
                                         "chem_react": (36.11, 9.26), "bio_pop_growth": (30.56, 18.06),
                                         "phys_osc": (28.79, 6.06)},
    "SR-Scientist GPT-OSS-120B": {"overall": (63.57, 49.35), "matsci": (74.67, 60.00),
                                  "chem_react": (81.48, 64.81), "bio_pop_growth": (66.67, 43.05),
                                  "phys_osc": (40.91, 34.09)},
    "SR-Scientist Qwen3-480B": {"overall": (49.09, 24.55), "matsci": (86.67, 69.33),
                                "chem_react": (40.74, 5.56), "bio_pop_growth": (50.00, 26.39),
                                "phys_osc": (34.09, 13.64)},
}


def pick(run, affine):
    pre = "aff_" if affine else ""
    rows = [r for r in run["frontier"] if "train" in r and np.isfinite(r[pre + "train"]["nmse"])]
    if not rows:
        return None
    r = min(rows, key=lambda r: r[pre + "train"]["nmse"])
    return {"id": r[pre + "id_test"], "ood": r.get(pre + "ood_test"), "row": r}


def load(run_dir, raw_evolve):
    per = defaultdict(lambda: defaultdict(list))  # method -> (domain,name) -> [sel]
    for f in sorted(Path(run_dir, "results").glob("*.json")):
        run = json.loads(f.read_text())
        if not run.get("completed"):
            continue
        sel = pick(run, run["method"] == "evolve" and not raw_evolve)
        per[run["method"]][(run["domain"], run["name"])].append(sel)
    return per


def acc(per_m, domain, split, key):
    """% problems accurate (seed-averaged); failed runs count as misses."""
    vals = []
    for (d, _), sels in per_m.items():
        if domain in (d, "overall"):
            vals.append(np.mean([bool(s and s[split] and s[split][key]) for s in sels]))
    return 100 * np.mean(vals) if vals else None


def nmse_med(per_m, domain, split):
    vals = []
    for (d, _), sels in per_m.items():
        if domain in (d, "overall"):
            ls = [np.log10(max(s[split]["nmse"], 1e-300)) if s and s[split] else np.inf for s in sels]
            vals.append(np.mean(ls))
    return 10 ** np.median(vals) if vals else None


def f(v, pct=True):
    if v is None:
        return "-"
    return f"{v:.2f}" if pct else f"{v:.3g}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--raw-evolve", action="store_true")
    ap.add_argument("--md", default=None, help="also write markdown here")
    args = ap.parse_args()

    per = load(args.run_dir, args.raw_evolve)
    label = {"baseline": "PySR (ours)", "evolve": "PySR evolved (ours)"}
    counts = {m: {SHORT[d]: sum(1 for (dd, _) in v if dd == d) for d in DOMAINS} for m, v in per.items()}
    seeds = {m: sorted({len(s) for s in v.values()}) for m, v in per.items()}
    parts = [f"Problems with >=1 completed run: {counts}; seeds per problem: {seeds}"]

    hdr = ["Method"] + [f"{SHORT[d]} {k}" for d in DOMAINS for k in ("Acc0.1", "NMSE")]
    rows = [[n] + [f(t[d][0]) if k == 0 else f(t[d][1], False) for d in DOMAINS for k in (0, 1)]
            for n, t in PAPER_A.items()]
    for m in ("baseline", "evolve"):
        if m in per:
            rows.append([f"**{label[m]}**"] + [x for d in DOMAINS for x in
                        (f(acc(per[m], d, "id", "acc_0.1")), f(nmse_med(per[m], d, "id"), False))])
    parts.append("### A. LLM-SRBench convention: strict Acc_0.1 (%) and NMSE, ID\n" + table(hdr, rows))

    cols = ["overall"] + list(DOMAINS)
    hdr = ["Method"] + [f"{SHORT.get(c, 'Overall')} {k}" for c in cols for k in ("0.01", "0.001")]
    rows = [[n] + [f(t[c][i]) for c in cols for i in (0, 1)] for n, t in PAPER_B.items()]
    for m in ("baseline", "evolve"):
        if m in per:
            rows.append([f"**{label[m]}**"] + [f(acc(per[m], c, "id", f"acc95_{t}"))
                                               for c in cols for t in (0.01, 0.001)])
    parts.append("### B. SR-Scientist convention: 5%-trimmed Acc (%), ID\n" + table(hdr, rows))

    hdr = ["Method", "split"] + [f"{SHORT.get(c, 'Overall')} {k}" for c in cols
                                 for k in ("Acc0.1", "Acc95_0.01", "NMSE")]
    rows = []
    for m in ("baseline", "evolve"):
        for split in ("id", "ood"):
            if m in per:
                rows.append([label[m], split] + [x for c in cols for x in (
                    f(acc(per[m], c, split, "acc_0.1")), f(acc(per[m], c, split, "acc95_0.01")),
                    f(nmse_med(per[m], c, split), False))])
    parts.append("### C. ID vs OOD (ours)\n" + table(hdr, rows))

    text = "\n\n".join(parts)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
