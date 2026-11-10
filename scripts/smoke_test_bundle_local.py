#!/usr/bin/env python3
"""Run one short local PySR fit with a bundle to check its Julia operators compile and run.

Usage: python scripts/smoke_test_bundle_local.py BUNDLE [--dataset feynman_I_6_2a] [--timeout 30] [--affine-scale-frontier]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bundle_loader import load_bundle
from parallel_eval_pysr import PySRTaskSpec, _evaluate_pysr_task, get_default_pysr_kwargs

p = argparse.ArgumentParser()
p.add_argument("bundle")
p.add_argument("--dataset", default="feynman_I_6_2a")
p.add_argument("--timeout", type=int, default=30)
p.add_argument("--max-evals", type=int, default=200_000)
p.add_argument("--affine-scale-frontier", action="store_true")
args = p.parse_args()

bundle = load_bundle(args.bundle, select_by="train")
kwargs = get_default_pysr_kwargs()
kwargs.update(max_evals=args.max_evals, timeout_in_seconds=args.timeout)
if args.affine_scale_frontier:
    kwargs["_affine_scale_frontier"] = True
cfg = bundle.to_pysr_config(kwargs)
spec = PySRTaskSpec(
    config_id=0, dataset_name=args.dataset, pysr_kwargs=cfg.pysr_kwargs,
    mutation_weights=cfg.mutation_weights, seed=0, data_seed=0, max_samples=1000,
    custom_mutation_code=cfg.custom_mutation_code,
    allow_custom_mutations=cfg.allow_custom_mutations,
    custom_selection_code=cfg.custom_selection_code,
    custom_survival_code=cfg.custom_survival_code,
    custom_loss_code=cfg.custom_loss_code,
    fitness_metric="gt", pysr_wall_limit=args.timeout * 4,
)
res = _evaluate_pysr_task(spec, use_cache=False)
for k in ("score", "r2", "gt", "error", "best_equation", "equation"):
    if hasattr(res, k):
        print(f"{k}: {getattr(res, k)}")
print(res)
