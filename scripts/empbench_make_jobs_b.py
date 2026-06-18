#!/usr/bin/env python3
"""Generate Phase B job list: operator-set / hparam variants of *vanilla* PySR,
to find the minimal change that lets baseline PySR recover each law.

Each variant is method=custom (baseline algorithm, modified function set / hparams).
Rationale per problem:
  Rydberg log(1/(R_H(1/n1^2-1/n2^2))) needs {log, square, /, -}. Default set has
    them but also sin/cos/exp/sqrt distractors + maxsize 40 (overfit room).
    -> test distractor removal and smaller maxsize.
  Planck log(2h nu^3/c^2/(exp(h nu/(kB T))-1)) needs {log, exp, cube, /, -} plus a
    tiny (~5e-11) constant inside exp and the exp(.)-1 structure.
    -> test adding cube and focusing the set.
"""
import argparse
import json
from pathlib import Path

# (variant_label, dict-of-overrides) ; overrides may set binary_ops, unary_ops,
# maxsize, populations, population_size, extra_kwargs(JSON str).
RYDBERG_VARIANTS = [
    ("min_logsq",        {"unary_ops": "log,square", "binary_ops": "+,-,*,/"}),
    ("min_logsq_sqrt",   {"unary_ops": "log,square,sqrt", "binary_ops": "+,-,*,/"}),
    ("default_maxsize20",{"maxsize": 20}),
    ("min_logsq_maxsize20",{"unary_ops": "log,square", "binary_ops": "+,-,*,/", "maxsize": 20}),
]
PLANCK_VARIANTS = [
    ("add_cube",         {"unary_ops": "sin,cos,exp,log,sqrt,square,cube"}),
    ("focused_logexpcube",{"unary_ops": "log,exp,square,cube", "binary_ops": "+,-,*,/"}),
    ("focused_logexp",   {"unary_ops": "log,exp,cube", "binary_ops": "+,-,*,/"}),
    ("default_maxsize25",{"maxsize": 25}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jobs", default="runs_local/phaseB/jobs.json")
    ap.add_argument("--out-dir", default="runs_local/phaseB")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--max-evals", type=float, default=1e7)
    ap.add_argument("--n-milestones", type=int, default=12)
    ap.add_argument("--wall-limit", type=float, default=21600)
    ap.add_argument("--only", default=None, help="comma list of dataset short names: planck,rydberg")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    plan = {"rydberg": ("empirical_rydberg", RYDBERG_VARIANTS),
            "planck": ("empirical_planck", PLANCK_VARIANTS)}
    only = set(args.only.split(",")) if args.only else set(plan)

    jobs = []
    for short, (ds, variants) in plan.items():
        if short not in only:
            continue
        for vlabel, ov in variants:
            for ri in range(args.n_seeds):
                job = {"method": "custom", "dataset": ds, "run_index": ri,
                       "max_evals": int(args.max_evals),
                       "n_milestones": args.n_milestones,
                       "wall_limit": int(args.wall_limit),
                       "out": str(out_dir / f"{short}_{vlabel}_r{ri}.json")}
                job.update(ov)
                jobs.append(job)

    Path(args.out_jobs).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_jobs).write_text(json.dumps(jobs, indent=2))
    print(f"Wrote {len(jobs)} Phase-B jobs to {args.out_jobs}")
    seen = set()
    for j in jobs:
        key = (j["dataset"], j["out"].split("_r")[0])
        if key not in seen:
            seen.add(key)
            print(f"  {j['dataset']:18s} {Path(j['out']).name.rsplit('_r',1)[0]:28s} "
                  f"unary={j.get('unary_ops','<default>')} maxsize={j.get('maxsize','40')}")


if __name__ == "__main__":
    main()
