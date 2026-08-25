#!/usr/bin/env python3
"""Full train/val evaluation of a simplify run's Pareto frontier plus slot ablations.

Builds one evaluation config per candidate, writes each as a minimal
``run_data.json`` that ``bundle_loader`` can load, and submits an independent
``evaluate_new_pysr.py`` SLURM job per config so they all run in parallel.

Two families of candidates:

1. ``frontier_*`` — every bundle on the LOC/score Pareto frontier. LOC is
   ``evolution_helpers._bundle_loc`` (code only, no docstrings or comments).

2. ``ablate_*`` — the lowest-LOC frontier bundle decomposed into its custom
   slots: the all-default baseline, each slot alone, each pair, and all three.
   The survival slot is dropped from these because the run's survival operator
   (``age_regularized_survival``) is byte-identical to SR.jl's
   ``default_survival``, so it is a no-op. That makes ``ablate_<all three>``
   semantically identical to the lowest-LOC frontier bundle, and the gap
   between their two scores is a free read on the seed noise floor.

Budget, splits, domain and metric are all inherited from the source run so the
numbers are comparable to its own ``final_eval_summary.json``.

Usage:
    # Preview the configs and sbatch commands without submitting
    python scripts/eval_simplify_candidates.py --dry-run

    # Build the configs and submit every job
    python scripts/eval_simplify_candidates.py

    # Once the jobs land, print the results table
    python scripts/eval_simplify_candidates.py --collect
"""

import argparse
import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evolution_helpers import _bundle_loc  # noqa: E402
from operator_types import OperatorBundle  # noqa: E402

# Slots ablated in family 2. Survival is excluded on purpose — see module docstring.
ABLATION_SLOTS = ("mutation", "loss", "selection")


def load_run_data(run: str) -> Tuple[Path, Dict[str, Any]]:
    path = Path(run)
    if not path.exists() and len(path.parts) == 1:
        path = REPO_ROOT / "runs" / path.name
    if path.is_dir():
        path = path / "run_data.json"
    if not path.exists():
        raise FileNotFoundError(f"No run_data.json at {path}")
    with open(path) as f:
        return path.parent, json.load(f)


def pareto_frontier(bundles: List[OperatorBundle]) -> List[OperatorBundle]:
    """Bundles that no smaller-LOC bundle matches or beats on score."""
    ranked = sorted(bundles, key=lambda b: (_bundle_loc(b), -(b.score or 0.0)))
    front: List[OperatorBundle] = []
    best = float("-inf")
    for b in ranked:
        if (b.score or float("-inf")) > best:
            front.append(b)
            best = b.score
    return front


def collect_bundles(data: Dict[str, Any], scope: str) -> List[OperatorBundle]:
    """Scored, de-duplicated bundles from the requested generations."""
    gens = data.get("generations") or []
    if not gens:
        raise ValueError("Run has no generations")
    if scope == "final-population":
        gens = gens[-1:]

    by_name: Dict[str, OperatorBundle] = {}
    for gen in gens:
        for key in ("population", "offspring"):
            for entry in gen.get(key) or []:
                if entry.get("score") is None:
                    continue
                bundle = OperatorBundle.from_dict(entry)
                prior = by_name.get(bundle.display_name)
                # Keep the best-scored copy: a bundle re-evaluated across
                # generations accumulates seeds and its score drifts.
                if prior is None or bundle.score > prior.score:
                    by_name[bundle.display_name] = bundle
    return list(by_name.values())


def subset_bundle(source: OperatorBundle, slots: Tuple[str, ...]) -> OperatorBundle:
    """A copy of `source` keeping only `slots`; every other slot falls back to PySR's default."""
    return OperatorBundle(
        operators={
            slot: copy.deepcopy(op)
            for slot, op in source.operators.items()
            if slot in slots and op is not None
        },
        best_hparams=copy.deepcopy(source.best_hparams) if source.best_hparams else None,
    )


def build_configs(
    data: Dict[str, Any], scope: str,
) -> Tuple[List[Tuple[str, OperatorBundle]], OperatorBundle]:
    """Return [(config_name, bundle)] for both families, plus the simplest frontier bundle."""
    front = pareto_frontier(collect_bundles(data, scope))
    if not front:
        raise ValueError("Pareto frontier is empty")

    configs: List[Tuple[str, OperatorBundle]] = [
        (f"frontier_loc{_bundle_loc(b)}", b) for b in front
    ]

    # Family 2 decomposes the smallest frontier bundle (front is LOC-ascending).
    simplest = front[0]
    present = tuple(s for s in ABLATION_SLOTS if simplest.operators.get(s) is not None)
    for size in range(len(present) + 1):
        for slots in itertools.combinations(present, size):
            name = "ablate_baseline" if not slots else "ablate_" + "_".join(slots)
            configs.append((name, subset_bundle(simplest, slots)))
    return configs, simplest


def write_config_dir(
    out_dir: Path, name: str, bundle: OperatorBundle, run_config: Dict[str, Any],
) -> Path:
    """Write a minimal run_data.json holding exactly this bundle.

    ``bundle_loader`` returns ``best_bundle`` verbatim when a run has no
    ``val_results``, and ``evaluate_new_pysr`` reads ``config`` off the same
    file for the domain, metric, and PySR kwargs — so copying the source run's
    config keeps every evaluation on the original run's settings.
    """
    config_dir = out_dir / name
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": run_config,
        "generations": [],
        "val_results": {},
        "best_bundle": bundle.to_dict(),
    }
    run_data_path = config_dir / "run_data.json"
    with open(run_data_path, "w") as f:
        json.dump(payload, f, indent=1)
    return run_data_path


def eval_command(
    run_data_path: Path, config_dir: Path, name: str, args: argparse.Namespace,
    splits: List[str], budget: Dict[str, Any],
) -> List[str]:
    cmd = [
        "sbatch", "--parsable", "-J", f"{args.job_prefix}-{name}",
        "run.sh", "evaluate_new_pysr.py",
        "--evolve-results", str(run_data_path),
        "--output-dir", str(config_dir / "eval"),
        "--splits", *splits,
        "--n-runs", str(args.n_runs),
        "--seed", str(args.seed),
        "--max-samples", str(budget["max_samples"]),
        "--max-evals", str(budget["max_evals"]),
        "--timeout", str(budget["timeout"]),
        "--pysr-wall-limit", str(args.pysr_wall_limit),
        "--time-limit", args.time_limit,
        "--job-timeout", str(args.job_timeout),
        "--mem-per-cpu", args.mem_per_cpu,
        "--partition", args.partition,
        "--max-concurrent-jobs", str(args.max_concurrent_jobs),
    ]
    if budget["domain"]:
        cmd += ["--domain", budget["domain"]]
    if budget["fitness_metric"]:
        cmd += ["--fitness-metric", budget["fitness_metric"]]
    return cmd


def collect(out_dir: Path, splits: List[str]) -> None:
    """Print the results table from whatever per-config evals have finished."""
    split_keys = [Path(s).stem for s in splits]
    rows = []
    for config_dir in sorted(out_dir.iterdir()):
        summary = config_dir / "eval" / "final_eval_summary.json"
        if not summary.is_dir() and summary.exists():
            with open(summary) as f:
                data = json.load(f)
            scores = []
            for key in split_keys:
                section = data.get(key) or {}
                gts = section.get("per_run_gt_avgs")
                scores.append(sum(gts) / len(gts) if gts else None)
            rows.append((config_dir.name, data.get("evolve_train_score"), scores))
        elif (config_dir / "run_data.json").exists():
            rows.append((config_dir.name, None, [None] * len(split_keys)))

    if not rows:
        print(f"No configs found under {out_dir}")
        return

    width = max(len(r[0]) for r in rows)
    header = f"{'config':<{width}}  {'evolve':>7}" + "".join(f"  {k:>22}" for k in split_keys)
    print(header)
    print("-" * len(header))
    for name, train, scores in rows:
        train_str = f"{train:.4f}" if train is not None else "-"
        cells = "".join(f"  {s:>22.4f}" if s is not None else f"  {'pending':>22}" for s in scores)
        print(f"{name:<{width}}  {train_str:>7}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", default="runs/252289",
                        help="Source evolve run (directory, run_data.json, or bare run id)")
    parser.add_argument("--output-dir", default=None,
                        help="Where config dirs are written (default: <run>/candidate_evals)")
    parser.add_argument("--frontier-scope", choices=["final-population", "all-generations"],
                        default="final-population",
                        help="Take the Pareto frontier over the last generation's "
                             "population+offspring, or over every generation in the run")
    parser.add_argument("--n-runs", type=int, default=10, help="Seeds per dataset per config")
    parser.add_argument("--seed", type=int, default=192,
                        help="Base eval seed (192 matches evolve_pysr's own final eval, "
                             "so already-evaluated bundles hit the cache)")
    parser.add_argument("--partition", default="ellis")
    parser.add_argument("--time-limit", default="00:15:00", help="SLURM time per PySR task")
    parser.add_argument("--mem-per-cpu", default="8G")
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    parser.add_argument("--pysr-wall-limit", type=int, default=600)
    parser.add_argument("--max-concurrent-jobs", type=int, default=50,
                        help="Concurrent PySR tasks per config; every config runs at "
                             "once, so the cluster-wide ceiling is this times the "
                             "number of configs")
    parser.add_argument("--job-prefix", default="simp-cand", help="SLURM job name prefix")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write config dirs and print sbatch commands without submitting")
    parser.add_argument("--collect", action="store_true",
                        help="Skip submission; print results from finished evals")
    args = parser.parse_args()

    run_dir, data = load_run_data(args.run)
    run_config = data.get("config") or {}
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "candidate_evals"

    train_split = run_config.get("split") or (
        f"splits/{run_config['split_label']}.txt" if run_config.get("split_label") else None
    )
    splits = [s for s in (train_split, run_config.get("val_split"),
                          run_config.get("test_split")) if s]
    if not splits:
        raise ValueError(f"Could not resolve any splits from {run_dir}/run_data.json")

    if args.collect:
        collect(out_dir, splits)
        return

    pysr_kwargs = run_config.get("pysr_kwargs") or {}
    budget = {
        "max_evals": pysr_kwargs.get("max_evals", 1000000),
        "timeout": pysr_kwargs.get("timeout_in_seconds", 500),
        "max_samples": run_config.get("max_samples", 1000),
        "domain": run_config.get("domain"),
        "fitness_metric": run_config.get("fitness_metric"),
    }

    configs, simplest = build_configs(data, args.frontier_scope)
    print(f"Source run:  {run_dir}")
    print(f"Splits:      {', '.join(splits)}  ({args.n_runs} seeds, base seed {args.seed})")
    print(f"Budget:      max_evals={budget['max_evals']:,} timeout={budget['timeout']}s "
          f"domain={budget['domain']} metric={budget['fitness_metric']}")
    print(f"Output:      {out_dir}")
    print(f"Ablating:    {simplest.display_name}\n")

    name_width = max(len(n) for n, _ in configs)
    print(f"{'config':<{name_width}}  {'LOC':>4}  {'evolve':>7}  bundle")
    print("-" * (name_width + 80))
    for name, bundle in configs:
        score = f"{bundle.score:.4f}" if bundle.score is not None else "-"
        print(f"{name:<{name_width}}  {_bundle_loc(bundle):>4}  {score:>7}  {bundle.display_name}")
    print()

    submitted: List[Tuple[str, str]] = []
    for name, bundle in configs:
        config_dir = out_dir / name
        run_data_path = write_config_dir(out_dir, name, bundle, run_config)
        cmd = eval_command(run_data_path, config_dir, name, args, splits, budget)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        job_id = subprocess.run(
            cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        submitted.append((name, job_id))
        print(f"  submitted {job_id}  {name}")

    if args.dry_run:
        print(f"\n[dry run] {len(configs)} configs written to {out_dir}; nothing submitted.")
    else:
        print(f"\nSubmitted {len(submitted)} parallel evaluations.")
        print(f"Collect with: python {Path(__file__).relative_to(REPO_ROOT)} "
              f"--run {args.run} --collect")


if __name__ == "__main__":
    main()
