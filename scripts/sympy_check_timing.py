#!/usr/bin/env python3
"""Time the sympy equivalence check on real stored SRBench results.

Replays ``check_pysr_symbolic_match(best_equation, ground_truth)`` for a sample
of the (dataset, seed, noise) results in a completed run and reports the
per-expression wall time. Note the full ``check_solved`` calls this once per
R²-passing frontier row (up to ~20), so per-task cost is a multiple of this.

Usage: python scripts/sympy_check_timing.py runs/737094 [n_sample] [timeout_s]
"""
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _gt_formula(dataset):
    from utils import load_srbench_dataset
    from evaluation import get_dataset_var_names
    from parallel_eval_pysr import _remap_formula_variables
    X, y, gt = load_srbench_dataset(dataset, max_samples=10)
    names = [f"x{i}" for i in range(X.shape[1])]
    try:
        ds_names = get_dataset_var_names(dataset)
        if len(ds_names) == X.shape[1]:
            gt = _remap_formula_variables(gt, ds_names, names)
    except Exception:
        pass
    return gt, names


_CACHE = {}


def _time_one(rec):
    from evaluation import check_pysr_symbolic_match
    dataset, expr, timeout = rec
    if dataset not in _CACHE:
        try:
            _CACHE[dataset] = _gt_formula(dataset)
        except Exception:
            _CACHE[dataset] = (None, None)
    gt, names = _CACHE[dataset]
    if gt is None or not expr:
        return None
    t0 = time.time()
    try:
        res = check_pysr_symbolic_match(expr, gt, var_names=names,
                                        timeout_seconds=timeout)
        err = res.get("error")
    except Exception as e:
        err = f"exc:{type(e).__name__}"
    return (time.time() - t0, dataset, err, len(expr))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/737094")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    results = json.load(open(run / "srbench_full_results.json"))["results"]
    if isinstance(results, dict):
        results = list(results.values())
    pool_in = [(r["dataset"], r.get("best_equation"), timeout)
               for r in results if r.get("present") and r.get("best_equation")]
    random.seed(0)
    random.shuffle(pool_in)
    pool_in = pool_in[:n]
    print(f"{run}: timing {len(pool_in)} expressions "
          f"(per-expression timeout={timeout}s)")
    with Pool(7) as p:
        out = [r for r in p.map(_time_one, pool_in, chunksize=4) if r]
    ts = [r[0] for r in out]
    print(f"n={len(ts)} mean={mean(ts):.2f}s median={median(ts):.3f}s")
    for q in (50, 75, 90, 95, 99, 100):
        print(f"  p{q:<4} {pct(ts, q):7.3f}s")
    n_to = sum(1 for r in out if r[2] == "timeout")
    print(f"  timeouts: {n_to} ({100*n_to/len(out):.1f}%)")
    print("  slowest:")
    for t, ds, err, L in sorted(out, reverse=True)[:10]:
        print(f"    {t:7.2f}s  {ds:22s} err={err} len={L}")


if __name__ == "__main__":
    main()
