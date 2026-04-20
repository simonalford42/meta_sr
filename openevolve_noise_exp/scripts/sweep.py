"""Launch the full noise × averaging sweep.

Runs each (noise, n_trials, seed) combination. By default runs them sequentially;
set --parallel N to launch N concurrent openevolve processes at the OS level.

Each openevolve run already uses its own worker-pool parallelism (see config.yaml
`evaluator.parallel_evaluations`), so external parallelism stacks on top.
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_ONE = REPO / "scripts" / "run_one.py"


def run_one(noise, n_trials, seed, budget, iters, out_root, base_config):
    tag = f"n{n_trials}_sigma{noise}_seed{seed}"
    out_dir = Path(out_root) / tag
    if (out_dir / "run_status.json").exists():
        print(f"[sweep] skipping existing run: {out_dir}")
        return (tag, 0, 0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUN_ONE),
        "--noise", str(noise),
        "--n-trials", str(n_trials),
        "--seed", str(seed),
        "--out", str(out_dir),
        "--base-config", str(base_config),
    ]
    if iters is not None:
        cmd += ["--iters", str(iters)]
    else:
        cmd += ["--budget", str(budget)]
    t0 = time.time()
    proc = subprocess.run(cmd)
    return (tag, proc.returncode, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, default=str(REPO / "outputs" / "sweep"))
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--iters", type=int, default=None, help="fixed iteration count (overrides --budget)")
    ap.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0])
    ap.add_argument("--n-trials", type=int, nargs="+", default=[1, 3, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--parallel", type=int, default=1, help="how many openevolve runs in parallel at OS level")
    ap.add_argument("--base-config", type=str, default=str(REPO / "configs" / "base.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    configs = []
    for noise in args.noises:
        for n in args.n_trials:
            for s in args.seeds:
                configs.append((noise, n, s))

    print(f"[sweep] total runs: {len(configs)}")
    print(f"[sweep] out_root: {args.out_root}")
    print(f"[sweep] budget={args.budget} parallel={args.parallel}")
    for c in configs:
        print("   -", c)
    if args.dry_run:
        return 0

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = []
    if args.parallel <= 1:
        for noise, n, s in configs:
            results.append(run_one(noise, n, s, args.budget, args.iters, args.out_root, args.base_config))
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futs = [
                ex.submit(run_one, noise, n, s, args.budget, args.iters, args.out_root, args.base_config)
                for (noise, n, s) in configs
            ]
            for fut in as_completed(futs):
                results.append(fut.result())

    elapsed = time.time() - t0
    print(f"\n[sweep] all done in {elapsed:.1f}s")
    for tag, rc, t in results:
        mark = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  {mark}  {tag}  {t:.1f}s")
    bad = [r for r in results if r[1] != 0]
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
