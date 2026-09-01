#!/usr/bin/env python3
"""Evaluate leave-one-out and add-one-in ablations for a PySR bundle.

For a bundle containing mutation, survival, selection, and loss operators, the
matrix is:

* the original four-operator bundle;
* four variants with one evolved operator replaced by PySR's default;
* four PySR-baseline variants with exactly one evolved operator enabled.

All configurations in a split share datasets and seeds and are submitted in a
single ``PySRSlurmEvaluator`` batch.  Multiple split batches are submitted
before waiting, matching the concurrency/caching/retry machinery used by
``srbench_full_eval.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_new_pysr import compute_per_run_avgs
from operator_types import (
    META_COMPONENTS,
    JuliaOperator,
    OperatorBundle,
    extract_function_name,
)
from parallel_eval_pysr import PySRSlurmEvaluator, get_default_pysr_kwargs
from utils import load_dataset_names_from_split


SECTION_RE = re.compile(
    r"^# === (mutation|survival|selection|loss): ([^=]+?) ===\s*$",
    re.MULTILINE,
)


def load_combined_bundle(path: str) -> OperatorBundle:
    """Load a multi-operator ``best_gen*.jl``/``best_final.jl`` artifact."""
    source_path = Path(path)
    text = source_path.read_text()
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        raise ValueError(
            f"{source_path} has no '# === <slot>: <name> ===' bundle sections"
        )

    operators: Dict[str, JuliaOperator] = {}
    for index, match in enumerate(matches):
        slot = match.group(1)
        declared_name = match.group(2).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        code = text[match.end():end].strip() + "\n"
        function_name = extract_function_name(code)
        if function_name != declared_name:
            raise ValueError(
                f"{source_path}: section {slot!r} declares {declared_name!r} "
                f"but defines {function_name!r}"
            )
        generation_match = re.search(r"(?:^|_)gen(\d+)(?:_|$)", declared_name)
        generation = int(generation_match.group(1)) if generation_match else 0
        operators[slot] = JuliaOperator(
            name=declared_name, code=code, generation=generation
        )

    missing = [slot for slot in META_COMPONENTS if slot not in operators]
    if missing:
        raise ValueError(f"{source_path} is missing operator sections: {missing}")

    score_match = re.search(r"^# Bundle score: ([0-9.eE+-]+)\s*$", text, re.MULTILINE)
    score = float(score_match.group(1)) if score_match else None
    return OperatorBundle(operators=operators, score=score)


def build_ablation_bundles(
    original: OperatorBundle,
) -> List[Tuple[str, OperatorBundle]]:
    """Return the original, four leave-one-out, and four add-one-in bundles."""
    missing = [
        slot for slot in META_COMPONENTS
        if original.operators.get(slot) is None
    ]
    if missing:
        raise ValueError(f"Ablation source bundle lacks evolved operators: {missing}")

    variants: List[Tuple[str, OperatorBundle]] = [
        ("original", copy.deepcopy(original)),
    ]
    for slot in META_COMPONENTS:
        operators = copy.deepcopy(original.operators)
        operators.pop(slot, None)
        variants.append((f"minus_{slot}", OperatorBundle(operators=operators)))
    for slot in META_COMPONENTS:
        variants.append((
            f"only_{slot}",
            OperatorBundle(operators={slot: copy.deepcopy(original.operators[slot])}),
        ))
    return variants


def summarize_result(result, n_runs: int) -> Dict:
    avg_r2, r2_vector, result_details = result
    per_run_r2 = compute_per_run_avgs(result_details, n_runs, "run_r2_scores")
    per_run_gt = compute_per_run_avgs(result_details, n_runs, "run_gt_scores")
    return {
        "avg_r2": avg_r2,
        "avg_gt": sum(per_run_gt) / len(per_run_gt) if per_run_gt else None,
        "per_run_r2_avgs": per_run_r2,
        "per_run_gt_avgs": per_run_gt,
        "r2_vector": r2_vector,
        "result_details": result_details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired PySR operator ablations on one or more splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bundle-jl",
        default="runs/709715/best_bundles/best_gen43.jl",
        help="Combined saved bundle containing four '# === slot ===' sections.",
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=[
            "splits/barely_unsolvable.txt",
            "splits/barely_unsolvable_val2.txt",
        ],
    )
    parser.add_argument("--output-dir", default="runs/709715/operator_ablation_gen43")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=192)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-evals", type=int, default=1_000_000)
    parser.add_argument("--timeout", type=int, default=500)
    parser.add_argument("--pysr-wall-limit", type=int, default=600)
    parser.add_argument("--partition", default="default_partition")
    parser.add_argument("--time-limit", default="00:15:00")
    parser.add_argument("--mem-per-cpu", default="8G")
    parser.add_argument("--job-timeout", type=int, default=14_400)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-concurrent-jobs", type=int, default=300)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print the matrix without creating output or SLURM jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be positive")
    if args.timeout > 0 and args.timeout >= args.pysr_wall_limit:
        raise ValueError("--timeout must be below --pysr-wall-limit")

    original = load_combined_bundle(args.bundle_jl)
    variants = build_ablation_bundles(original)
    split_names_requested = [Path(split).stem for split in args.splits]
    if len(set(split_names_requested)) != len(split_names_requested):
        raise ValueError("--splits must have distinct filename stems")
    split_datasets = {
        split_name: load_dataset_names_from_split(split)
        for split_name, split in zip(split_names_requested, args.splits)
    }

    pysr_kwargs = get_default_pysr_kwargs()
    pysr_kwargs["max_evals"] = args.max_evals
    if args.timeout > 0:
        pysr_kwargs["timeout_in_seconds"] = args.timeout
    configs = []
    for label, bundle in variants:
        config = bundle.to_pysr_config(pysr_kwargs)
        config.name = label
        configs.append(config)

    print(f"Bundle: {args.bundle_jl}")
    print(f"Matrix: {len(configs)} configurations x {args.n_runs} paired seeds")
    for label, bundle in variants:
        active = [slot for slot in META_COMPONENTS if bundle.operators.get(slot)]
        print(f"  {label:<16} active={','.join(active) if active else 'baseline'}")
    for split_name, datasets in split_datasets.items():
        print(f"  split={split_name}: {len(datasets)} datasets, "
              f"{len(datasets) * len(configs) * args.n_runs} fits")
    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bundle_jl": str(Path(args.bundle_jl).resolve()),
        "n_runs": args.n_runs,
        "seed": args.seed,
        "seeds": [args.seed + i for i in range(args.n_runs)],
        "max_samples": args.max_samples,
        "max_evals": args.max_evals,
        "timeout": args.timeout,
        "pysr_wall_limit": args.pysr_wall_limit,
        "fitness_metric": "gt",
        "variants": [
            {
                "name": label,
                "operators": {
                    slot: (bundle.operators[slot].name if bundle.operators.get(slot) else None)
                    for slot in META_COMPONENTS
                },
            }
            for label, bundle in variants
        ],
        "splits": args.splits,
        "batches": {},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    evaluator = PySRSlurmEvaluator(
        results_dir=str(output_dir),
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=args.max_samples,
        data_seed=args.seed,
        use_cache=not args.no_cache,
        job_timeout=args.job_timeout,
        pysr_wall_limit=args.pysr_wall_limit,
        max_retries=args.max_retries,
        max_concurrent_jobs=args.max_concurrent_jobs,
        domain="srbench",
    )

    handles = []
    split_names = []
    for split, (split_name, datasets) in zip(args.splits, split_datasets.items()):
        evaluator.split_label = split_name
        handle = evaluator.submit_configs(
            configs=configs,
            dataset_names=datasets,
            seed=args.seed,
            n_runs=args.n_runs,
            fitness_metric="gt",
        )
        handles.append(handle)
        split_names.append(split_name)
        manifest["batches"][split_name] = {
            "split": split,
            "datasets": datasets,
            "batch_dir": f"slurm_pysr/{handle.batch_dir.name}",
            "n_tasks": handle.n_tasks,
            "n_cached": handle.n_cached,
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    collected = evaluator.collect_batches(handles)
    summary = {
        "meta": manifest,
        "results": {
            split_name: {
                label: summarize_result(result, args.n_runs)
                for (label, _), result in zip(variants, split_results)
            }
            for split_name, split_results in zip(split_names, collected)
        },
    }
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nMean GT match rate:")
    header = "variant".ljust(18) + "".join(name.rjust(26) for name in split_names)
    print(header)
    for label, _ in variants:
        row = label.ljust(18)
        for split_name in split_names:
            row += f"{summary['results'][split_name][label]['avg_gt']:.4f}".rjust(26)
        print(row)
    print(f"\nSaved {output_dir / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
