#!/usr/bin/env python3
"""
Evaluate a single PySR method (baseline, evolve, openevolve, or HPO) on given splits.

Takes one method and evaluates it for n_runs seeds on each provided split,
reporting per-seed R² and GT symbolic solve rate. Logs results to W&B.

Usage:
    # Baseline (no method args)
    python evaluate_new_pysr.py --n-runs 5

    # Evolve result
    python evaluate_new_pysr.py \
        --evolve-results outputs/evolve_mutation_*/run_data.json \
        --n-runs 5

    # Unfinalized bundle run (uses best bundle seen so far)
    python evaluate_new_pysr.py \
        --evolve-results 399313 \
        --splits splits/val.txt \
        --n-runs 10

    # OpenEvolve result
    python evaluate_new_pysr.py \
        --openevolve-results outputs/openevolve_pysr_survival_*/ \
        --n-runs 5

    # HPO weights
    python evaluate_new_pysr.py \
        --best-weights outputs/hpo_pysr_*/best_weights.json \
        --n-runs 5

    # Custom splits
    python evaluate_new_pysr.py \
        --evolve-results outputs/evolve_mutation_*/run_data.json \
        --splits splits/train.txt splits/val.txt splits/test.txt \
        --n-runs 10
"""

import argparse
import atexit
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
)
from utils import load_dataset_names_from_split, copy_slurm_log, resolve_run_dir
from wandb_utils import init_wandb, log_wandb_summary, finish_wandb


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvalSummary:
    split_name: str
    avg_r2: float
    per_run_r2_avgs: List[float]
    per_run_gt_avgs: List[float]
    r2_vector: List[float]
    result_details: List[Dict]


# =============================================================================
# Loaders
# =============================================================================

def _bundle_label(bundle: "OperatorBundle", prefix: str) -> str:
    """Build a label like 'evolve_mutation+survival:nameA+nameB'.

    Returns just ``prefix`` when the bundle has no custom operators (baseline).
    """
    types, names = [], []
    for t, op in bundle.operators.items():
        if op is not None:
            types.append(t)
            names.append(op.name)
    if not types:
        return prefix
    return f"{prefix}_{'+'.join(types)}:{'+'.join(names)}"


def load_method(method_source, method_path):
    """Load a method as (OperatorBundle, label) via the single bundle_loader.

    method_source: 'evolve' | 'openevolve' | 'hpo' | 'baseline' | None.
    baseline/None returns the default (no custom operators) bundle; everything
    else is parsed by bundle_loader.load_bundle (run_data.json,
    best_params.json / best_weights.json, or best/best_program.py).
    """
    from operator_types import OperatorBundle
    if method_source in (None, "baseline"):
        return OperatorBundle.create_default(), "baseline"
    from bundle_loader import load_bundle
    bundle = load_bundle(method_path)
    if method_source == "hpo":
        return bundle, "hpo_best"
    prefix = "openevolve" if method_source == "openevolve" else "evolve"
    return bundle, _bundle_label(bundle, prefix)


def _saved_evolve_config(method_source: Optional[str], method_path: Optional[str]) -> Dict[str, Any]:
    """Read the run configuration needed to faithfully replay an evolve run.

    ``load_bundle`` intentionally returns only the winning bundle. Final
    evaluation also needs run-level settings (most importantly the domain and
    its base operators), which live alongside the bundle in ``run_data.json``.
    """
    if method_source != "evolve" or not method_path:
        return {}

    path = Path(method_path)
    if not path.exists() and len(path.parts) == 1 and path.name.isdigit():
        path = Path(__file__).resolve().parent / "runs" / path.name
    if path.is_dir():
        path = path / "run_data.json"
    if path.name != "run_data.json" or not path.exists():
        return {}

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  WARNING: could not read final-eval settings from {path}: {exc}")
        return {}
    config = data.get("config")
    return config if isinstance(config, dict) else {}


def _resolve_final_eval_context(
    method_source: Optional[str],
    method_path: Optional[str],
    domain: Optional[str],
    fitness_metric: Optional[str],
) -> Tuple[str, str, Dict[str, Any]]:
    """Resolve domain, metric, and base kwargs for a final evaluation."""
    saved = _saved_evolve_config(method_source, method_path)
    domain_name = domain or saved.get("domain") or "srbench"
    default_metric = "gt-acc" if domain_name in ("boolean", "boolformer") else (
        "gt" if domain_name == "neuron" else "r2"
    )
    metric_name = fitness_metric or saved.get("fitness_metric") or default_metric

    from domains import get_domain
    domain_obj = get_domain(domain_name)
    saved_kwargs = saved.get("pysr_kwargs")
    if isinstance(saved_kwargs, dict) and saved_kwargs:
        pysr_kwargs = copy.deepcopy(saved_kwargs)
    else:
        pysr_kwargs = domain_obj.base_pysr_kwargs()
    return domain_name, metric_name, pysr_kwargs


def _bundle_summary_fields(bundle) -> Dict[str, Any]:
    """Operator/training-score fields for the eval summary JSON (empty for baseline)."""
    ops = [(t, op) for t, op in bundle.operators.items() if op is not None]
    if not ops:
        return {}
    fields: Dict[str, Any] = {}
    if len(ops) == 1:
        fields["operator_type"] = ops[0][0]
        fields["operator_name"] = ops[0][1].name
        fields["generation"] = ops[0][1].generation
    else:
        fields["operators"] = [
            {"type": t, "name": op.name, "generation": op.generation}
            for t, op in ops
        ]
    fields["evolve_train_score"] = bundle.score
    return fields


def _read_autoresearch_results_tsv() -> List[Dict[str, str]]:
    tsv_path = Path("autoresearch_sr/results.tsv")
    if not tsv_path.exists():
        raise FileNotFoundError(f"{tsv_path} missing")
    rows: List[Dict[str, str]] = []
    with open(tsv_path) as f:
        lines = f.read().splitlines()
    if not lines:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append(dict(zip(header, parts)))
    return rows


def _best_autoresearch_row() -> Dict[str, str]:
    """Return the results.tsv row with the highest score (skipping crashes).

    Ties broken by highest experiment number (most recent).
    """
    rows = _read_autoresearch_results_tsv()
    candidates = [r for r in rows if r.get("status") != "crash"]
    if not candidates:
        raise ValueError("No non-crash rows in autoresearch_sr/results.tsv")
    best = max(candidates, key=lambda r: (float(r["score"]), int(r["exp"])))
    return best


def resolve_autoresearch_commit(target: str, submodule_path: Path) -> str:
    """Resolve an autoresearch target to a full commit hash in the SR.jl submodule.

    target can be 'best' (highest score in results.tsv), 'latest'/'HEAD',
    a commit hash, a branch, or an 'expN' row from autoresearch_sr/results.tsv.
    """
    if target == "best":
        best = _best_autoresearch_row()
        print(f"[autoresearch] best row: exp{best['exp']} "
              f"score={best['score']} status={best.get('status', '?')}")
        revspec = best["commit"]
    elif target in ("latest", "HEAD", None):
        revspec = "HEAD"
    elif target.lower().startswith("exp"):
        exp_num = target[3:]
        tsv_path = Path("autoresearch_sr/results.tsv")
        if not tsv_path.exists():
            raise FileNotFoundError(f"Cannot resolve {target!r}: {tsv_path} missing")
        revspec = None
        with open(tsv_path) as f:
            lines = f.read().splitlines()
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if parts[0] == exp_num:
                revspec = parts[1]
                break
        if revspec is None:
            raise ValueError(f"Experiment {target!r} not found in {tsv_path}")
    else:
        revspec = target

    result = subprocess.run(
        ["git", "-C", str(submodule_path), "rev-parse", revspec],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def create_sr_worktree(submodule_path: Path, commit: str) -> Path:
    """Create a detached git worktree of the SR.jl submodule at `commit`."""
    wt_dir = Path(tempfile.mkdtemp(prefix=f"srjl_{commit[:8]}_"))
    # `worktree add` insists the target directory not already exist.
    wt_dir.rmdir()
    subprocess.run(
        ["git", "-C", str(submodule_path), "worktree", "add", "--detach",
         str(wt_dir), commit],
        check=True,
    )
    return wt_dir


def cleanup_sr_worktree(submodule_path: Path, wt_dir: Path) -> None:
    subprocess.run(
        ["git", "-C", str(submodule_path), "worktree", "remove", "--force", str(wt_dir)],
        check=False,
    )


# =============================================================================
# Evaluation
# =============================================================================

def compute_per_run_avgs(result_details: List[Dict], n_runs: int,
                         score_key: str = "run_r2_scores") -> List[float]:
    """Compute per-run averages across datasets."""
    missing_fill = (0.0 if score_key in {
        "run_gt_scores", "run_acc_scores", "run_f1_scores"
    } else -1.0)
    per_run_avgs = []
    for run_idx in range(n_runs):
        run_scores = []
        for d in result_details:
            run_values = d.get(score_key, [])
            run_scores.append(run_values[run_idx] if len(run_values) > run_idx else missing_fill)
        per_run_avgs.append(float(np.mean(run_scores)) if run_scores else missing_fill)
    return per_run_avgs


def _classification_avgs(summary: EvalSummary, n_runs: int) -> Tuple[float, float]:
    """Mean selected-equation accuracy/F1, or NaN for nonclassification domains."""
    has_acc = any(d.get("run_acc_scores") for d in summary.result_details)
    has_f1 = any(d.get("run_f1_scores") for d in summary.result_details)
    acc = (float(np.mean(compute_per_run_avgs(
        summary.result_details, n_runs, "run_acc_scores"
    ))) if has_acc else float("nan"))
    f1 = (float(np.mean(compute_per_run_avgs(
        summary.result_details, n_runs, "run_f1_scores"
    ))) if has_f1 else float("nan"))
    return acc, f1


def evaluate_config(
    evaluator: PySRSlurmEvaluator,
    dataset_names: List[str],
    config: PySRConfig,
    seed: int,
    n_runs: int,
    name: str,
    target_noise_map: Optional[Dict[str, float]] = None,
    fitness_metric: str = "r2",
) -> EvalSummary:
    """Evaluate a PySRConfig and return summary. Cache stats tracked on evaluator."""
    config.name = name
    results = evaluator.evaluate_configs(
        [config], dataset_names, seed=seed, n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
    )

    avg_r2, r2_vector, result_details = results[0]
    per_run_r2_avgs = compute_per_run_avgs(result_details, n_runs, "run_r2_scores")
    per_run_gt_avgs = compute_per_run_avgs(result_details, n_runs, "run_gt_scores")
    return EvalSummary(
        split_name=name,
        avg_r2=avg_r2,
        per_run_r2_avgs=per_run_r2_avgs,
        per_run_gt_avgs=per_run_gt_avgs,
        r2_vector=r2_vector,
        result_details=result_details,
    )


def print_results(split_summaries: Dict[str, EvalSummary], n_runs: int, method_label: str) -> None:
    """Print per-seed results for each split."""
    for split_name, summary in split_summaries.items():
        print(f"\n{'=' * 70}")
        print(f"  {method_label} — {split_name} split  ({len(summary.r2_vector)} datasets)")
        print(f"{'=' * 70}")

        # Per-seed table
        print(f"  {'Seed':>6}  {'R²':>10}  {'GT':>10}")
        print(f"  {'-' * 30}")
        for seed_idx in range(n_runs):
            r2 = summary.per_run_r2_avgs[seed_idx] if seed_idx < len(summary.per_run_r2_avgs) else float("nan")
            gt = summary.per_run_gt_avgs[seed_idx] if seed_idx < len(summary.per_run_gt_avgs) else float("nan")
            print(f"  {seed_idx:>6}  {r2:>10.4f}  {gt:>10.4f}")

        print(f"  {'-' * 30}")
        avg_r2 = float(np.mean(summary.per_run_r2_avgs)) if summary.per_run_r2_avgs else float("nan")
        avg_gt = float(np.mean(summary.per_run_gt_avgs)) if summary.per_run_gt_avgs else float("nan")
        print(f"  {'Avg':>6}  {avg_r2:>10.4f}  {avg_gt:>10.4f}")
        avg_acc, avg_f1 = _classification_avgs(summary, n_runs)
        if np.isfinite(avg_acc) or np.isfinite(avg_f1):
            print(f"  Selected-equation accuracy={avg_acc:.4f}, F1={avg_f1:.4f}")

        # Per-dataset breakdown
        print(f"\n  Per-dataset breakdown:")
        for detail in summary.result_details:
            dataset = detail["dataset"]
            scores = detail.get("run_gt_scores", [])
            scores_str = ", ".join(f"{int(s)}" for s in scores)
            avg = detail.get("avg_gt", float("nan"))
            print(f"    {dataset}: GT=[{scores_str}] avg={avg:.2f}")


# =============================================================================
# Programmatic Final Evaluation
# =============================================================================

def _build_fixed_summary(
    split_name: str,
    datasets: List[str],
    per_level_summaries: Dict[float, "EvalSummary"],
    target_noise_map: Dict[str, float],
    n_runs: int,
) -> "EvalSummary":
    """Assemble an EvalSummary that, for each dataset, uses results from the noise
    level matching its training-time fixed assignment (target_noise_map).

    This reproduces what a single fixed-noise final eval would report, but pulls the
    per-dataset results out of the already-computed per-noise-level summaries (no
    extra compute, since each (dataset, noise) pair is cached).
    """
    detail_by_level_dataset = {
        (level, d["dataset"]): d
        for level, summ in per_level_summaries.items()
        for d in summ.result_details
    }
    selected: List[Dict] = []
    for ds in datasets:
        level = target_noise_map[ds]
        detail = detail_by_level_dataset.get((level, ds))
        if detail is not None:
            selected.append(detail)

    per_run_r2 = compute_per_run_avgs(selected, n_runs, "run_r2_scores")
    per_run_gt = compute_per_run_avgs(selected, n_runs, "run_gt_scores")
    avg_r2 = float(np.mean(per_run_r2)) if per_run_r2 else float("nan")
    return EvalSummary(
        split_name=f"{split_name} (fixed-noise)",
        avg_r2=avg_r2,
        per_run_r2_avgs=per_run_r2,
        per_run_gt_avgs=per_run_gt,
        r2_vector=[d.get("avg_r2", float("nan")) for d in selected],
        result_details=selected,
    )


def run_final_evaluation(
    output_dir: str,
    method_source: str,
    method_path: str,
    partition: str,
    splits: Optional[List[str]] = None,
    n_runs: int = 10,
    seed: int = 42,
    max_samples: int = 1000,
    max_evals: int = 1000000,
    timeout: int = 500,
    time_limit: str = "00:15:00",
    mem_per_cpu: str = "8G",
    job_timeout: float = 1800.0,
    max_concurrent_jobs: Optional[int] = None,
    use_cache: bool = True,
    wandb_run: Optional[Any] = None,
    target_noise_map: Optional[Dict[str, float]] = None,
    noise_levels: Optional[List[float]] = None,
    pysr_wall_limit: int = 600,
    black_box: bool = False,
    domain: Optional[str] = None,
    fitness_metric: Optional[str] = None,
) -> Dict[str, "EvalSummary"]:
    """Run final evaluation on requested splits after an evolution, OpenEvolve, or HPO run.

    Args:
        output_dir: Parent output directory; final eval results go in output_dir/final_eval/
        method_source: One of "evolve", "openevolve", "hpo"
        method_path: Path to the method result:
            - evolve: path to run_data.json
            - openevolve: path to output directory
            - hpo: path to best_params.json (or best_weights.json)
        partition: SLURM partition
        splits: List of split file paths (default: train.txt + val.txt)
        n_runs: Number of seeds per config per dataset
        seed: Base random seed
        max_samples/max_evals/timeout: PySR config
        time_limit/mem_per_cpu/job_timeout: SLURM config
        use_cache: Whether to use evaluation caching
        wandb_run: Existing wandb run to log to (or None)
        target_noise_map: Optional per-dataset noise map (the training-time fixed
            noise assignment). Used both for single-noise eval and, in multi-noise
            mode, to pick each dataset's fixed level for the avg_gt metric.
        noise_levels: Optional list of noise levels. When provided together with
            target_noise_map, each split is evaluated at every noise level (n_runs
            seeds per level per dataset), and the report includes both avg_gt (the
            per-dataset fixed-noise assignment, matching training/validation) and
            avg_gt_all_noise (averaged across all noise levels).
        black_box: Use SRBench's black-box split/scaling protocol and held-out
            R² scoring for every split.
        domain: Evaluation domain. For evolve runs this defaults to the domain
            persisted in run_data.json; otherwise it defaults to ``srbench``.
        fitness_metric: Aggregation metric. For evolve runs this defaults to
            the persisted metric; otherwise it uses the domain default.

    Returns:
        Dict mapping split name to EvalSummary
    """
    if splits is None:
        splits = ["splits/train.txt", "splits/val.txt"]

    eval_dir = str(Path(output_dir) / "final_eval")
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    # Load method as an OperatorBundle via the single loader (bundle_loader).
    bundle, method_label = load_method(method_source, method_path)

    domain, fitness_metric, pysr_kwargs = _resolve_final_eval_context(
        method_source, method_path, domain, fitness_metric,
    )
    from domains import get_domain
    domain_obj = get_domain(domain)
    # LogicBench is iteration-bounded and deliberately does not use the
    # SRBench max-evals/timeout budget. Preserve that distinction in final eval.
    if domain_obj.uses_run_budget:
        pysr_kwargs["max_evals"] = max_evals
        pysr_kwargs["timeout_in_seconds"] = timeout
    else:
        pysr_kwargs.pop("max_evals", None)
        pysr_kwargs.pop("timeout_in_seconds", None)

    print(f"\n{'=' * 60}")
    print(f"Final Evaluation: {method_label}")
    print(f"  Splits: {', '.join(Path(s).stem for s in splits)}")
    print(f"  Seeds: {n_runs}")
    print(f"  Domain: {domain} (fitness_metric={fitness_metric})")
    print(f"{'=' * 60}")

    evaluator = PySRSlurmEvaluator(
        results_dir=eval_dir,
        partition=partition,
        time_limit=time_limit,
        mem_per_cpu=mem_per_cpu,
        dataset_max_samples=max_samples,
        data_seed=seed,
        job_timeout=job_timeout,
        max_concurrent_jobs=max_concurrent_jobs,
        use_cache=use_cache,
        pysr_wall_limit=pysr_wall_limit,
        black_box=black_box,
        domain=domain,
    )

    # Single converter: bundle -> PySRConfig (merges custom code + HPO hparams).
    config = bundle.to_pysr_config(pysr_kwargs)
    for t, op in bundle.operators.items():
        if op is not None:
            print(f"  Loaded [{t}] {op.name}")
    if bundle.score is not None:
        print(f"  Training score: {bundle.score:.4f}")
    if bundle.best_hparams:
        print(f"  Applied {len(bundle.best_hparams)} HPO-tuned hparam(s)")

    multi_noise = bool(noise_levels) and target_noise_map is not None
    if multi_noise:
        print(f"  Multi-noise final eval: {len(noise_levels)} noise levels "
              f"× {n_runs} seeds = {len(noise_levels) * n_runs} runs per dataset/split")

    split_summaries: Dict[str, EvalSummary] = {}
    multi_noise_data: Dict[str, Any] = {}
    for split_path in splits:
        split_name = Path(split_path).stem
        datasets = load_dataset_names_from_split(split_path)

        if multi_noise:
            print(f"\nEvaluating on {split_name} ({len(datasets)} datasets, "
                  f"{n_runs} seeds × {len(noise_levels)} noise levels)...")
            per_level_summaries: Dict[float, EvalSummary] = {}
            for level in noise_levels:
                uniform_map = {d: level for d in datasets}
                evaluator.split_label = f"{split_name}@noise{level}"
                per_level_summaries[level] = evaluate_config(
                    evaluator, datasets, config,
                    seed, n_runs, f"final_{split_name}_noise{level}_{method_label}",
                    target_noise_map=uniform_map,
                    fitness_metric=fitness_metric,
                )

            # Fixed-noise summary: each dataset uses its training-time assigned level.
            summary = _build_fixed_summary(
                split_name, datasets, per_level_summaries, target_noise_map, n_runs,
            )
            split_summaries[split_name] = summary

            per_level_gt = {
                lvl: (float(np.mean(s.per_run_gt_avgs)) if s.per_run_gt_avgs else float("nan"))
                for lvl, s in per_level_summaries.items()
            }
            per_level_r2 = {
                lvl: (float(np.mean(s.per_run_r2_avgs)) if s.per_run_r2_avgs else float("nan"))
                for lvl, s in per_level_summaries.items()
            }
            avg_gt = float(np.mean(summary.per_run_gt_avgs)) if summary.per_run_gt_avgs else float("nan")
            avg_r2 = float(np.mean(summary.per_run_r2_avgs)) if summary.per_run_r2_avgs else float("nan")
            avg_gt_all_noise = float(np.mean(list(per_level_gt.values())))
            avg_r2_all_noise = float(np.mean(list(per_level_r2.values())))

            multi_noise_data[split_name] = {
                "avg_gt": avg_gt,
                "avg_gt_all_noise": avg_gt_all_noise,
                "avg_r2": avg_r2,
                "avg_r2_all_noise": avg_r2_all_noise,
                "per_noise_level": {
                    str(lvl): {"avg_gt": per_level_gt[lvl], "avg_r2": per_level_r2[lvl]}
                    for lvl in noise_levels
                },
            }

            print(f"  {split_name}: GT(fixed)={avg_gt:.4f}  GT(all-noise)={avg_gt_all_noise:.4f}  "
                  f"R²(fixed)={avg_r2:.4f}  R²(all-noise)={avg_r2_all_noise:.4f}")
            for lvl in noise_levels:
                print(f"      noise={lvl:<6}: GT={per_level_gt[lvl]:.4f}  R²={per_level_r2[lvl]:.4f}")

            if wandb_run is not None:
                import wandb
                log_dict = {
                    f"final_eval/{split_name}/avg_gt": avg_gt,
                    f"final_eval/{split_name}/avg_gt_all_noise": avg_gt_all_noise,
                    f"final_eval/{split_name}/avg_r2": avg_r2,
                    f"final_eval/{split_name}/avg_r2_all_noise": avg_r2_all_noise,
                }
                for lvl in noise_levels:
                    log_dict[f"final_eval/{split_name}/noise_{lvl}/avg_gt"] = per_level_gt[lvl]
                    log_dict[f"final_eval/{split_name}/noise_{lvl}/avg_r2"] = per_level_r2[lvl]
                wandb.log(log_dict)
        else:
            print(f"\nEvaluating on {split_name} ({len(datasets)} datasets, {n_runs} seeds)...")
            evaluator.split_label = split_name
            summary = evaluate_config(
                evaluator, datasets, config,
                seed, n_runs, f"final_{split_name}_{method_label}",
                target_noise_map=target_noise_map,
                fitness_metric=fitness_metric,
            )
            split_summaries[split_name] = summary

            avg_r2 = float(np.mean(summary.per_run_r2_avgs)) if summary.per_run_r2_avgs else float("nan")
            avg_gt = float(np.mean(summary.per_run_gt_avgs)) if summary.per_run_gt_avgs else float("nan")
            avg_acc, avg_f1 = _classification_avgs(summary, n_runs)
            class_suffix = (f", accuracy={avg_acc:.4f}, F1={avg_f1:.4f}"
                            if np.isfinite(avg_acc) or np.isfinite(avg_f1) else "")
            print(f"  {split_name}: R²={avg_r2:.4f}, GT={avg_gt:.4f}{class_suffix}")

            if wandb_run is not None:
                import wandb
                wandb.log({
                    f"final_eval/{split_name}/avg_r2": avg_r2,
                    f"final_eval/{split_name}/avg_gt": avg_gt,
                })

    # Print full results table (fixed-noise per-seed breakdown in multi-noise mode)
    print_results(split_summaries, n_runs, f"[Final Eval] {method_label}")

    # Save summary JSON
    summary_data = {
        "method": method_label,
        "splits": splits,
        "n_runs": n_runs,
        "seed": seed,
        "black_box": black_box,
        "domain": domain,
        "fitness_metric": fitness_metric,
    }
    summary_data.update(_bundle_summary_fields(bundle))
    for split_name, s in split_summaries.items():
        summary_data[split_name] = asdict(s)
        avg_acc, avg_f1 = _classification_avgs(s, n_runs)
        if np.isfinite(avg_acc):
            summary_data[split_name]["avg_accuracy"] = avg_acc
        if np.isfinite(avg_f1):
            summary_data[split_name]["avg_f1"] = avg_f1
    if multi_noise:
        summary_data["noise_levels"] = list(noise_levels)
        summary_data["multi_noise"] = multi_noise_data

    summary_path = Path(output_dir) / "final_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nFinal eval saved: {summary_path}")

    # Log summary metrics to wandb
    if wandb_run is not None:
        import wandb
        for split_name, s in split_summaries.items():
            avg_r2 = float(np.mean(s.per_run_r2_avgs)) if s.per_run_r2_avgs else float("nan")
            avg_gt = float(np.mean(s.per_run_gt_avgs)) if s.per_run_gt_avgs else float("nan")
            wandb.summary[f"final_eval_{split_name}_avg_r2"] = avg_r2
            wandb.summary[f"final_eval_{split_name}_avg_gt"] = avg_gt
            avg_acc, avg_f1 = _classification_avgs(s, n_runs)
            if np.isfinite(avg_acc):
                wandb.summary[f"final_eval_{split_name}_avg_accuracy"] = avg_acc
            if np.isfinite(avg_f1):
                wandb.summary[f"final_eval_{split_name}_avg_f1"] = avg_f1
            if multi_noise:
                wandb.summary[f"final_eval_{split_name}_avg_gt_all_noise"] = \
                    multi_noise_data[split_name]["avg_gt_all_noise"]
                wandb.summary[f"final_eval_{split_name}_avg_r2_all_noise"] = \
                    multi_noise_data[split_name]["avg_r2_all_noise"]

    return split_summaries


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a single PySR method on given splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Method source (at most one; none = baseline)
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument("--evolve-results", type=str,
                              help="Path to evolve_pysr run_data.json, a run "
                                   "directory, or a run id under runs/")
    method_group.add_argument("--openevolve-results", type=str,
                              help="Path to OpenEvolve output directory")
    method_group.add_argument("--best-weights", type=str,
                              help="Path to HPO best-weights JSON")
    method_group.add_argument("--autoresearch", type=str, nargs="?", const="best",
                              default=None, metavar="TARGET",
                              help="Evaluate an autoresearch_sr run. Pass a commit hash, "
                                   "branch, 'expN' (row in autoresearch_sr/results.tsv), "
                                   "'latest'/HEAD, or 'best' for the highest-scoring row "
                                   "(default when flag used without value).")
    parser.add_argument("--autoresearch-submodule", type=str,
                        default="SymbolicRegression.jl",
                        help="Path to the SR.jl submodule used for --autoresearch worktree")

    parser.add_argument("--splits", type=str, nargs="+",
                        default=["splits/train.txt", "splits/val.txt"],
                        help="Split files to evaluate on")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of seeds/runs per config per dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for evaluation")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum samples per dataset")
    parser.add_argument("--max-evals", type=int, default=1000000,
                        help="Maximum evaluations per PySR run (ignored if --wall-clock-only)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="PySR timeout in seconds (PySR's own timeout_in_seconds)")
    parser.add_argument("--wall-clock-only", action="store_true",
                        help="Stop PySR on wall-clock timeout only (drop max_evals)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Per-target Gaussian noise level applied uniformly to all datasets")
    parser.add_argument("--pysr-wall-limit", type=int, default=600,
                        help="Hard wall-clock guard inside the SLURM task (seconds); "
                             "must exceed --timeout with a buffer")
    parser.add_argument("--partition", type=str, default="default_partition",
                        help="SLURM partition")
    parser.add_argument("--time-limit", type=str, default="00:30:00",
                        help="SLURM time limit per job")
    parser.add_argument("--mem-per-cpu", type=str, default="8G",
                        help="SLURM memory per CPU")
    parser.add_argument("--job-timeout", type=float, default=3000.0,
                        help="Max wait for SLURM completion")
    parser.add_argument("--max-concurrent-jobs", type=int, default=None,
                        help="Max concurrent SLURM array tasks")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/eval_pysr_TIMESTAMP)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable evaluation caching")
    parser.add_argument("--domain", choices=["srbench", "boolean", "boolformer", "neuron"],
                        default=None,
                        help="Evaluation domain (inferred from evolve run_data when possible)")
    parser.add_argument("--fitness-metric",
                        choices=["r2", "gt", "gt-r2", "acc", "gt-acc"],
                        default=None,
                        help="Task aggregation metric (inferred from evolve run_data when possible)")

    args = parser.parse_args()

    # Autoresearch: resolve commit and create a worktree of SR.jl at that commit.
    autoresearch_commit: Optional[str] = None
    autoresearch_worktree: Optional[Path] = None
    autoresearch_submodule: Optional[Path] = None
    if args.autoresearch:
        autoresearch_submodule = Path(args.autoresearch_submodule).resolve()
        autoresearch_commit = resolve_autoresearch_commit(
            args.autoresearch, autoresearch_submodule
        )
        autoresearch_worktree = create_sr_worktree(
            autoresearch_submodule, autoresearch_commit
        )
        atexit.register(cleanup_sr_worktree, autoresearch_submodule, autoresearch_worktree)
        print(f"[autoresearch] submodule: {autoresearch_submodule}")
        print(f"[autoresearch] commit:    {autoresearch_commit}")
        print(f"[autoresearch] worktree:  {autoresearch_worktree}")
        # The evaluation cache key does not include the Julia source hash, so a
        # modified SR.jl source would otherwise collide with baseline cache entries.
        args.no_cache = True

    # Determine method (as an OperatorBundle) and label via the single loader.
    if args.evolve_results:
        method_source, method_path = "evolve", args.evolve_results
    elif args.openevolve_results:
        method_source, method_path = "openevolve", args.openevolve_results
    elif args.best_weights:
        method_source, method_path = "hpo", args.best_weights
    elif args.autoresearch:
        method_source, method_path = "baseline", None
    else:
        method_source, method_path = "baseline", None

    bundle, method_label = load_method(method_source, method_path)
    if args.autoresearch:
        method_label = f"autoresearch_{autoresearch_commit[:8]}"

    domain, fitness_metric, pysr_kwargs = _resolve_final_eval_context(
        method_source, method_path, args.domain, args.fitness_metric,
    )
    from domains import get_domain
    domain_obj = get_domain(domain)

    args.output_dir = resolve_run_dir(args.output_dir, label="eval_pysr")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wandb_config = {
        "method": method_label,
        "splits": args.splits,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "max_evals": args.max_evals,
        "timeout": args.timeout,
        "wall_clock_only": args.wall_clock_only,
        "noise": args.noise,
        "pysr_wall_limit": args.pysr_wall_limit,
        "partition": args.partition,
        "no_cache": args.no_cache,
        "domain": domain,
        "fitness_metric": fitness_metric,
    }
    _ops = [(t, op) for t, op in bundle.operators.items() if op is not None]
    if _ops:
        wandb_config["operator_type"] = "+".join(t for t, _ in _ops)
        wandb_config["operator_name"] = "+".join(op.name for _, op in _ops)
        wandb_config["generation"] = _ops[0][1].generation
        wandb_config["evolve_train_score"] = bundle.score

    # W&B tags must be 1-64 chars. Prefer the SLURM job id of the eval run when
    # available (so the tag links straight to out/<id>.out); otherwise fall back
    # to method_label, truncated to fit.
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if slurm_job_id:
        wandb_tag = slurm_job_id
    elif len(method_label) > 64:
        wandb_tag = method_label[:61] + "..."
    else:
        wandb_tag = method_label

    wandb_run = init_wandb(
        config=wandb_config,
        script_name="evaluate_new_pysr.py",
        output_dir=str(output_dir),
        extra_tags=[wandb_tag],
    )

    if domain_obj.uses_run_budget:
        pysr_kwargs["timeout_in_seconds"] = args.timeout
        if not args.wall_clock_only:
            pysr_kwargs["max_evals"] = args.max_evals
        else:
            pysr_kwargs.pop("max_evals", None)
    else:
        pysr_kwargs.pop("max_evals", None)
        pysr_kwargs.pop("timeout_in_seconds", None)

    evaluator_kwargs: Dict[str, Any] = {}
    if autoresearch_worktree is not None:
        evaluator_kwargs["julia_project"] = str(autoresearch_worktree)

    evaluator = PySRSlurmEvaluator(
        results_dir=str(output_dir),
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=args.max_samples,
        data_seed=args.seed,
        job_timeout=args.job_timeout,
        max_concurrent_jobs=args.max_concurrent_jobs,
        use_cache=not args.no_cache,
        pysr_wall_limit=args.pysr_wall_limit,
        domain=domain,
        **evaluator_kwargs,
    )

    # Single converter: bundle -> PySRConfig (merges custom code + HPO hparams).
    config = bundle.to_pysr_config(pysr_kwargs)
    if bundle.best_hparams:
        print(f"Applied {len(bundle.best_hparams)} HPO-tuned hparam(s) from bundle: "
              f"{sorted(bundle.best_hparams.keys())}")
    for t, op in bundle.operators.items():
        if op is not None:
            print(f"Loaded [{t}] {op.name}")
            print(f"  Generation: {op.generation}")
            code_file = output_dir / f"{op.name}.jl"
            code_file.write_text(op.code)
            print(f"  Saved code to: {code_file}")
    if bundle.score is not None:
        print(f"  Training score (from evolve): {bundle.score:.4f}")
    print()

    # Evaluate on each split
    split_summaries: Dict[str, EvalSummary] = {}
    for split_path in args.splits:
        split_name = Path(split_path).stem
        datasets = load_dataset_names_from_split(split_path)
        print(f"{'=' * 60}")
        print(f"Evaluating {method_label} on {split_name} split ({len(datasets)} datasets)...")
        print(f"{'=' * 60}")

        target_noise_map = (
            {d: args.noise for d in datasets} if args.noise > 0 else None
        )

        summary = evaluate_config(
            evaluator, datasets, config,
            args.seed, args.n_runs, f"{split_name}_{method_label}",
            target_noise_map=target_noise_map,
            fitness_metric=fitness_metric,
        )
        split_summaries[split_name] = summary

        # Log per-split metrics to W&B
        if wandb_run is not None:
            import wandb
            avg_r2 = float(np.mean(summary.per_run_r2_avgs)) if summary.per_run_r2_avgs else float("nan")
            avg_gt = float(np.mean(summary.per_run_gt_avgs)) if summary.per_run_gt_avgs else float("nan")
            wandb.log({
                f"{split_name}/avg_r2": avg_r2,
                f"{split_name}/avg_gt": avg_gt,
                f"{split_name}/n_datasets": len(datasets),
            })
            for seed_idx in range(args.n_runs):
                if seed_idx < len(summary.per_run_r2_avgs):
                    wandb.log({
                        f"{split_name}/seed_{seed_idx}_r2": summary.per_run_r2_avgs[seed_idx],
                        f"{split_name}/seed_{seed_idx}_gt": summary.per_run_gt_avgs[seed_idx],
                    })

    # Build per-split summary for wandb
    extra_summary = {}
    for split_name, s in split_summaries.items():
        avg_r2 = float(np.mean(s.per_run_r2_avgs)) if s.per_run_r2_avgs else float("nan")
        avg_gt = float(np.mean(s.per_run_gt_avgs)) if s.per_run_gt_avgs else float("nan")
        extra_summary[f"{split_name}_avg_r2"] = avg_r2
        extra_summary[f"{split_name}_avg_gt"] = avg_gt

    # Print results
    print_results(split_summaries, args.n_runs, method_label)

    # Log final summary to wandb (includes SR eval cache stats from evaluator)
    log_wandb_summary(wandb_run, evaluator=evaluator, extra_summary=extra_summary)
    finish_wandb(wandb_run)

    # Save summary JSON
    total_evals = evaluator.total_sr_evals
    total_cached = evaluator.total_sr_cached
    cache_fraction = total_cached / total_evals if total_evals > 0 else 0.0

    summary_data = {
        "method": method_label,
        "splits": args.splits,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "command": "python " + " ".join(sys.argv),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "output_dir": str(output_dir),
        "cache_fraction": cache_fraction,
        "total_evals": total_evals,
        "total_cached": total_cached,
        "domain": domain,
        "fitness_metric": fitness_metric,
    }
    summary_data.update(_bundle_summary_fields(bundle))
    for split_name, s in split_summaries.items():
        summary_data[split_name] = asdict(s)

    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved: {summary_path}")
    copy_slurm_log(output_dir)


if __name__ == "__main__":
    main()
