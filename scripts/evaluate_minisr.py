#!/usr/bin/env python3
"""
Evaluate MiniSR.jl (native Julia) commits on SRBench splits, with caching.

Mirrors scripts/evaluate.py but runs MiniSR.jl via mini_pysr.PyPySRRegressor.
Each commit is materialised as an isolated sandbox (git worktree of the
SymbolicRegression.jl submodule + a minimal parallel PySR/juliapkg overlay)
so baseline and improved commits can run concurrently on SLURM.

Two modes:
1. --srbench: full SRBench evaluation on a single commit with 3 seeds.
2. Default (compare): compare baseline vs improved commit on multiple splits
   (default: splits/barely_unsolvable.txt, splits/train.txt, splits/val.txt)
   with 10 seeds each. Both commits are submitted concurrently as separate
   SLURM arrays.

Usage:
    # Full SRBench eval on one commit (3 seeds)
    python scripts/evaluate_minisr.py --srbench --commit <sha>

    # Compare default-baseline vs improved on default splits (10 seeds each)
    python scripts/evaluate_minisr.py --commit <improved_sha>

    # Override baseline and splits
    python scripts/evaluate_minisr.py \\
        --commit <improved_sha> --baseline-commit <baseline_sha> \\
        --splits splits/train.txt splits/val.txt

Caching:
    Per-(commit, split, n_runs, seed, config) JSON files stored under
    outputs/eval_minisr_cache/. Disable with --no-cache.
"""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from parallel_eval_minisr import (  # noqa: E402
    MiniSRConfig,
    MiniSRSlurmEvaluator,
    get_default_minisr_kwargs,
    get_default_mutation_weights,
)
from utils import load_dataset_names_from_split  # noqa: E402

SUBMODULE_PATH = REPO_ROOT / "SymbolicRegression.jl"
SANDBOXES_ROOT = REPO_ROOT / "outputs" / "eval_minisr_sandboxes"

# Universal "original MiniSR" ref: the commit right before the apr22
# autoresearch experiments began tweaking MiniSR.jl (exp1 in the results
# table refers to this unmodified baseline). Update as needed.
DEFAULT_BASELINE_COMMIT = "5951a63cbac1cc6d9ea95bd85114b98bed63a21d"


@dataclass
class EvalSummary:
    split_name: str
    commit_sha: str
    commit_label: str
    avg_score: float
    avg_r2: float
    avg_gt: float
    per_dataset: List[Dict]
    per_run_r2_avgs: List[float]
    per_run_gt_avgs: List[float]
    fitness_metric: str
    n_runs: int


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _run_git(args: List[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _resolve_commit(ref: str, cwd: Path) -> str:
    return _run_git(["rev-parse", ref], cwd).stdout.strip()


# ---------------------------------------------------------------------------
# Sandbox builder
# ---------------------------------------------------------------------------

def build_sandbox(sha: str) -> Path:
    """Materialise a per-commit sandbox under outputs/eval_minisr_sandboxes/<short>.

    Layout:
        <sandbox>/SymbolicRegression.jl/   git worktree at <sha>
        <sandbox>/mini_pysr.py              *copy* (so Path(__file__).resolve() ≠ real repo)
        <sandbox>/PySR/pysr/juliapkg.json   rewritten to point at <sandbox>/SymbolicRegression.jl
        <sandbox>/PySR/...                  symlinks to real PySR siblings
        <sandbox>/.juliapkg_env/            empty; juliapkg resolves a fresh Manifest on first use
        <sandbox>/<other files/dirs>         symlinks to real repo

    Cached across runs via a `.ready` sentinel file.
    """
    short = sha[:12]
    sandbox = SANDBOXES_ROOT / short
    ready_marker = sandbox / ".ready"
    if ready_marker.exists():
        return sandbox

    sandbox.mkdir(parents=True, exist_ok=True)

    # Git worktree for the submodule at the target commit
    sr_worktree = sandbox / "SymbolicRegression.jl"
    if not (sr_worktree / ".git").exists():
        if sr_worktree.exists():
            shutil.rmtree(sr_worktree, ignore_errors=True)
        _run_git(
            ["worktree", "add", "--detach", str(sr_worktree), sha],
            SUBMODULE_PATH,
        )

    # Copy mini_pysr.py so its REPO_ROOT (via Path(__file__).resolve()) points here.
    shutil.copy(REPO_ROOT / "mini_pysr.py", sandbox / "mini_pysr.py")

    # Build a PySR overlay: symlink every child except pysr/juliapkg.json, which we rewrite.
    real_pysr = REPO_ROOT / "PySR"
    sandbox_pysr = sandbox / "PySR"
    sandbox_pysr.mkdir(exist_ok=True)
    for item in real_pysr.iterdir():
        if item.name == "pysr":
            continue
        link = sandbox_pysr / item.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(item)

    sandbox_pysr_pysr = sandbox_pysr / "pysr"
    sandbox_pysr_pysr.mkdir(exist_ok=True)
    for item in (real_pysr / "pysr").iterdir():
        if item.name == "juliapkg.json":
            continue
        link = sandbox_pysr_pysr / item.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(item)

    real_juliapkg = real_pysr / "pysr" / "juliapkg.json"
    juliapkg_data = json.loads(real_juliapkg.read_text())
    juliapkg_data.setdefault("packages", {}).setdefault("SymbolicRegression", {})[
        "path"
    ] = str(sr_worktree.resolve())
    (sandbox_pysr_pysr / "juliapkg.json").write_text(
        json.dumps(juliapkg_data, indent=2)
    )

    # Empty juliapkg env — juliapkg will resolve fresh on first use, writing a Manifest
    # that references our sandboxed SymbolicRegression.jl path.
    (sandbox / ".juliapkg_env").mkdir(exist_ok=True)

    # Symlink all other top-level repo items.
    overrides = {
        "SymbolicRegression.jl",
        "PySR",
        ".juliapkg_env",
        "mini_pysr.py",
        ".git",
        ".ready",
    }
    for item in REPO_ROOT.iterdir():
        if item.name in overrides:
            continue
        link = sandbox / item.name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(item)

    ready_marker.touch()
    return sandbox


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _make_cache_key(
    commit_sha: str,
    dataset_names: List[str],
    seed: int,
    n_runs: int,
    max_evals: int,
    max_samples: int,
    fitness_metric: str,
    mutation_weights: Dict[str, float],
    minisr_kwargs: Dict[str, Any],
) -> str:
    key_data = {
        "commit_sha": commit_sha,
        "datasets": sorted(dataset_names),
        "seed": seed,
        "n_runs": n_runs,
        "max_evals": max_evals,
        "max_samples": max_samples,
        "fitness_metric": fitness_metric,
        "mutation_weights": mutation_weights,
        "minisr_kwargs": minisr_kwargs,
    }
    key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _cache_load(cache_dir: Path, key: str) -> Optional[Dict]:
    p = cache_dir / f"{key}.json"
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _cache_save(cache_dir: Path, key: str, data: Dict) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{key}.json"
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(p)
    return p


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------

def _per_run_avgs(details: List[Dict], n_runs: int, key: str) -> List[float]:
    missing = 0.0 if key == "run_gt_scores" else -1.0
    out: List[float] = []
    for i in range(n_runs):
        scores = []
        for d in details:
            runs = d.get(key, []) or []
            scores.append(runs[i] if i < len(runs) else missing)
        out.append(float(np.mean(scores)) if scores else missing)
    return out


def _make_summary(
    *,
    split_name: str,
    commit_sha: str,
    commit_label: str,
    details: List[Dict],
    fitness_metric: str,
    n_runs: int,
) -> EvalSummary:
    if details:
        # Score vector = per-dataset score under the chosen metric.
        if fitness_metric == "gt":
            score_vector = [d.get("avg_gt", 0.0) for d in details]
        else:
            score_vector = [d.get("avg_r2", -1.0) for d in details]
        avg_score = float(np.mean(score_vector))
        avg_r2 = float(np.mean([d.get("avg_r2", -1.0) for d in details]))
        avg_gt = float(np.mean([d.get("avg_gt", 0.0) for d in details]))
    else:
        avg_score = 0.0
        avg_r2 = -1.0
        avg_gt = 0.0
    return EvalSummary(
        split_name=split_name,
        commit_sha=commit_sha,
        commit_label=commit_label,
        avg_score=avg_score,
        avg_r2=avg_r2,
        avg_gt=avg_gt,
        per_dataset=details,
        per_run_r2_avgs=_per_run_avgs(details, n_runs, "run_r2_scores"),
        per_run_gt_avgs=_per_run_avgs(details, n_runs, "run_gt_scores"),
        fitness_metric=fitness_metric,
        n_runs=n_runs,
    )


# ---------------------------------------------------------------------------
# Per-commit evaluation (thread-safe, runs one SLURM array for all uncached datasets)
# ---------------------------------------------------------------------------

def evaluate_commit(
    *,
    commit_sha: str,
    commit_label: str,
    split_datasets: List[Tuple[str, List[str]]],
    evaluator_kwargs: Dict[str, Any],
    config: MiniSRConfig,
    seed: int,
    n_runs: int,
    max_evals: int,
    max_samples: int,
    fitness_metric: str,
    cache_dir: Optional[Path],
    output_dir: Path,
) -> Dict[str, EvalSummary]:
    """Evaluate one commit on all provided splits. Returns {split_name: EvalSummary}.

    Caches per-split results. Datasets from uncached splits are de-duplicated
    and submitted as a single SLURM array (one per commit = one sandbox).
    """
    results: Dict[str, EvalSummary] = {}
    uncached: List[Tuple[str, List[str]]] = []

    for split_name, ds in split_datasets:
        key = _make_cache_key(
            commit_sha=commit_sha,
            dataset_names=ds,
            seed=seed,
            n_runs=n_runs,
            max_evals=max_evals,
            max_samples=max_samples,
            fitness_metric=fitness_metric,
            mutation_weights=config.mutation_weights,
            minisr_kwargs=config.minisr_kwargs,
        )
        if cache_dir is not None:
            cached = _cache_load(cache_dir, key)
            if cached is not None:
                print(
                    f"  [cache HIT ] {commit_label} / {split_name} "
                    f"({len(ds)} datasets × {n_runs} runs) -> {key[:12]}",
                    flush=True,
                )
                results[split_name] = EvalSummary(**cached)
                continue
            print(
                f"  [cache MISS] {commit_label} / {split_name} "
                f"({len(ds)} datasets × {n_runs} runs) -> {key[:12]}",
                flush=True,
            )
        uncached.append((split_name, ds))

    if not uncached:
        return results

    needed_datasets = sorted({d for _, ds in uncached for d in ds})
    run_dir = output_dir / commit_label
    run_dir.mkdir(parents=True, exist_ok=True)

    sandbox = build_sandbox(commit_sha)
    print(
        f"  [run      ] {commit_label}: {len(needed_datasets)} unique datasets × "
        f"{n_runs} runs across {len(uncached)} split(s); sandbox={sandbox}",
        flush=True,
    )

    evaluator = MiniSRSlurmEvaluator(
        results_dir=str(run_dir),
        use_cache=False,  # owned by this script's cache layer
        repo_root=str(sandbox),
        **evaluator_kwargs,
    )
    batch_results = evaluator.evaluate_configs(
        [config],
        needed_datasets,
        seed=seed,
        n_runs=n_runs,
        fitness_metric=fitness_metric,
    )
    _, _, all_details = batch_results[0]
    details_by_dataset = {d["dataset"]: d for d in all_details}

    for split_name, ds in uncached:
        split_details = [details_by_dataset[d] for d in ds if d in details_by_dataset]
        summary = _make_summary(
            split_name=split_name,
            commit_sha=commit_sha,
            commit_label=commit_label,
            details=split_details,
            fitness_metric=fitness_metric,
            n_runs=n_runs,
        )
        if cache_dir is not None:
            key = _make_cache_key(
                commit_sha=commit_sha,
                dataset_names=ds,
                seed=seed,
                n_runs=n_runs,
                max_evals=max_evals,
                max_samples=max_samples,
                fitness_metric=fitness_metric,
                mutation_weights=config.mutation_weights,
                minisr_kwargs=config.minisr_kwargs,
            )
            path = _cache_save(cache_dir, key, asdict(summary))
            print(
                f"  [cache SAVE] {commit_label} / {split_name} -> {key[:12]} "
                f"({path})",
                flush=True,
            )
        results[split_name] = summary

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_comparison(
    split_name: str,
    configs: List[Tuple[str, EvalSummary]],
    fitness_metric: str,
) -> None:
    print()
    print("=" * 80)
    print(f"  {split_name} (metric={fitness_metric})")
    print("=" * 80)

    if not configs:
        print("  (no results)")
        return

    n_runs = max(s.n_runs for _, s in configs)

    header = [f"{'Seed':>6}"]
    for label, _ in configs:
        header.append(f"{label+' R²':>20}")
        header.append(f"{label+' GT':>20}")
    print("  ".join(header))
    print("-" * (8 + len(configs) * 42))

    for i in range(n_runs):
        row = [f"{i:>6}"]
        for _, s in configs:
            r2 = s.per_run_r2_avgs[i] if i < len(s.per_run_r2_avgs) else float("nan")
            gt = s.per_run_gt_avgs[i] if i < len(s.per_run_gt_avgs) else float("nan")
            row.append(f"{r2:>20.4f}")
            row.append(f"{gt:>20.4f}")
        print("  ".join(row))
    print("-" * (8 + len(configs) * 42))

    row = [f"{'Avg':>6}"]
    for _, s in configs:
        row.append(f"{s.avg_r2:>20.4f}")
        row.append(f"{s.avg_gt:>20.4f}")
    print("  ".join(row))

    print()
    print(f"  Per-dataset breakdown — {split_name}")
    print("-" * 80)
    for label, s in configs:
        print(f"\n  [{label}]")
        for d in s.per_dataset:
            ds = d.get("dataset", "?")
            r2s = d.get("run_r2_scores", []) or []
            gts = d.get("run_gt_scores", []) or []
            r2_str = ",".join(f"{x:.2f}" for x in r2s)
            gt_str = ",".join(f"{x:.0f}" for x in gts)
            print(
                f"    {ds}: R2=[{r2_str}] avg_R2={d.get('avg_r2', float('nan')):.2f} "
                f"GT=[{gt_str}] avg_GT={d.get('avg_gt', float('nan')):.2f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate MiniSR.jl commits on SRBench splits, with caching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--commit", type=str, default="HEAD",
                   help="Improved (or sole, in --srbench mode) commit ref in the "
                        "SymbolicRegression.jl submodule")
    p.add_argument("--baseline-commit", type=str, default=DEFAULT_BASELINE_COMMIT,
                   help="Baseline commit ref (defaults to the pre-apr22 unmodified "
                        "MiniSR baseline, ignored in --srbench mode)")
    p.add_argument("--srbench", action="store_true",
                   help="Full SRBench eval on --commit only (3 seeds by default)")
    p.add_argument("--splits", nargs="+",
                   default=[
                       "splits/barely_unsolvable.txt",
                       "splits/train.txt",
                       "splits/val.txt",
                   ],
                   help="Splits to compare on (ignored in --srbench mode)")
    p.add_argument("--srbench-split", type=str, default="splits/srbench_all.txt",
                   help="Split to use in --srbench mode")
    p.add_argument("--n-runs", type=int, default=None,
                   help="Seeds per dataset (default: 3 if --srbench else 10)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-evals", type=int, default=1_000_000)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--fitness-metric", choices=["r2", "gt"], default="gt")
    p.add_argument("--partition", type=str, default="default_partition")
    p.add_argument("--time-limit", type=str, default="04:00:00")
    p.add_argument("--mem-per-cpu", type=str, default="8G")
    p.add_argument("--job-timeout", type=float, default=1500.0)
    p.add_argument("--max-concurrent-jobs", type=int, default=None,
                   help="Max concurrent SLURM array tasks (per SLURM array)")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--cache-dir", type=str,
                   default=str(REPO_ROOT / "outputs" / "eval_minisr_cache"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-warmstart", action="store_true")
    args = p.parse_args()

    if not SUBMODULE_PATH.exists():
        p.error(f"Submodule not found at {SUBMODULE_PATH}")

    improved_sha = _resolve_commit(args.commit, SUBMODULE_PATH)
    baseline_sha = None
    if not args.srbench:
        baseline_sha = _resolve_commit(args.baseline_commit, SUBMODULE_PATH)

    n_runs = args.n_runs
    if n_runs is None:
        n_runs = 3 if args.srbench else 10

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "srbench" if args.srbench else "compare"
        args.output_dir = str(REPO_ROOT / "outputs" / f"eval_minisr_{mode}_{ts}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = None if args.no_cache else Path(args.cache_dir)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cache dir: {cache_dir}")
    else:
        print("Cache: DISABLED (--no-cache)")

    minisr_kwargs = get_default_minisr_kwargs()
    minisr_kwargs["max_evals"] = args.max_evals
    config = MiniSRConfig(
        mutation_weights=get_default_mutation_weights(),
        minisr_kwargs=minisr_kwargs,
        name="minisr",
    )
    evaluator_kwargs = dict(
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=args.max_samples,
        data_seed=args.seed,
        max_retries=2,
        job_timeout=args.job_timeout,
        max_concurrent_jobs=args.max_concurrent_jobs,
        warm_start=not args.no_warmstart,
    )

    # Resolve splits
    if args.srbench:
        split_datasets = [
            (Path(args.srbench_split).stem, load_dataset_names_from_split(args.srbench_split))
        ]
        commits = [(improved_sha, f"commit_{improved_sha[:12]}")]
        print(f"\n[SRBench mode] commit={improved_sha[:12]}, split={split_datasets[0][0]}"
              f" ({len(split_datasets[0][1])} datasets) × {n_runs} runs")
    else:
        split_datasets = [
            (Path(sf).stem, load_dataset_names_from_split(sf)) for sf in args.splits
        ]
        commits = [
            (baseline_sha, f"baseline_{baseline_sha[:12]}"),
            (improved_sha, f"improved_{improved_sha[:12]}"),
        ]
        print(f"\n[Compare mode]")
        print(f"  baseline = {baseline_sha[:12]}")
        print(f"  improved = {improved_sha[:12]}")
        print(f"  splits   = {[s for s, _ in split_datasets]}")
        print(f"  n_runs   = {n_runs}")
    print(f"  output_dir = {output_dir}")

    # Pre-build sandboxes serially (git worktree add isn't thread-safe).
    print()
    print("Preparing sandboxes...")
    for sha, label in commits:
        sandbox = build_sandbox(sha)
        print(f"  {label}: {sandbox}")

    # Submit each commit's evaluation concurrently.
    print()
    print(f"Submitting {len(commits)} commit(s) concurrently...")
    results_by_commit: Dict[str, Dict[str, EvalSummary]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(commits)) as exe:
        futures = {
            exe.submit(
                evaluate_commit,
                commit_sha=sha,
                commit_label=label,
                split_datasets=split_datasets,
                evaluator_kwargs=evaluator_kwargs,
                config=config,
                seed=args.seed,
                n_runs=n_runs,
                max_evals=args.max_evals,
                max_samples=args.max_samples,
                fitness_metric=args.fitness_metric,
                cache_dir=cache_dir,
                output_dir=output_dir,
            ): label
            for sha, label in commits
        }
        for fut in concurrent.futures.as_completed(futures):
            label = futures[fut]
            try:
                results_by_commit[label] = fut.result()
                print(f"  [done] {label}", flush=True)
            except Exception as e:
                print(f"  [FAIL] {label}: {e}", flush=True)
                raise

    # Reshape: per split -> [(commit_label, summary), ...]
    results_by_split: Dict[str, List[Tuple[str, EvalSummary]]] = {}
    for _, label in commits:
        commit_results = results_by_commit.get(label, {})
        for split_name, summary in commit_results.items():
            results_by_split.setdefault(split_name, []).append((label, summary))

    for split_name, _ in split_datasets:
        if split_name in results_by_split:
            print_comparison(split_name, results_by_split[split_name], args.fitness_metric)

    summary_path = output_dir / "analysis_summary.json"
    out = {
        "mode": "srbench" if args.srbench else "compare",
        "improved_sha": improved_sha,
        "baseline_sha": baseline_sha,
        "n_runs": n_runs,
        "seed": args.seed,
        "max_evals": args.max_evals,
        "max_samples": args.max_samples,
        "fitness_metric": args.fitness_metric,
        "splits": {
            split: {label: asdict(s) for label, s in configs}
            for split, configs in results_by_split.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
