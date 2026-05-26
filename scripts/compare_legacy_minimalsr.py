#!/usr/bin/env python3
"""Compare legacy MinimalSR against the current SkeletonSR+PySR baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from parallel_eval_minisr import (  # noqa: E402
    MiniSRConfig,
    MiniSRSlurmEvaluator,
    get_default_minisr_kwargs,
    get_default_mutation_weights,
)
from scripts.evaluate_minisr import build_sandbox, _resolve_commit, SUBMODULE_PATH  # noqa: E402
from utils import load_dataset_names_from_split  # noqa: E402


DEFAULT_LEGACY_REF = "d00bfbcd^"
DEFAULT_SKELETON_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "skeleton_pysr_parity_v7_init_placeholder"
    / "pysrsr_summary.json"
)


def _per_seed_averages(result_details: List[Dict], n_runs: int) -> List[float]:
    per_seed = []
    for r in range(n_runs):
        vals = []
        for d in result_details:
            scores = d.get("run_gt_scores") or []
            vals.append(float(scores[r]) if r < len(scores) else 0.0)
        per_seed.append(float(np.mean(vals)) if vals else 0.0)
    return per_seed


def _summarize(per_seed: List[float]) -> Dict[str, object]:
    return {
        "mean": float(np.mean(per_seed)),
        "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        "per_seed": per_seed,
    }


def _install_legacy_wrapper(sandbox: Path) -> None:
    """Make sandbox/mini_pysr.py call legacy MinimalSR with the MiniSR API shape."""
    env_dir = sandbox / ".juliapkg_env"
    sandbox_sr = str((sandbox / "SymbolicRegression.jl").resolve())
    bad_sr = str((REPO_ROOT / "SymbolicRegression.jl").resolve())
    if env_dir.exists():
        tomls = [p for p in env_dir.glob("*.toml") if p.is_file()]
        combined = "\n".join(p.read_text(errors="ignore") for p in tomls)
        if bad_sr in combined and sandbox_sr not in combined:
            trash = Path.home() / "trash"
            trash.mkdir(parents=True, exist_ok=True)
            target = trash / f"{env_dir.name}.legacy_minimalsr_bad_{int(time.time())}"
            shutil.move(str(env_dir), str(target))
    env_dir.mkdir(exist_ok=True)

    (sandbox / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "repo = str(Path(__file__).resolve().parent)\n"
        "sys.path[:] = [p for p in sys.path if p != repo]\n"
        "sys.path.insert(0, repo)\n"
    )

    text = (REPO_ROOT / "mini_pysr.py").read_text()
    text = text.replace(
        '_MINISR = jl.seval("SymbolicRegression.MiniSR")',
        '_MINISR = jl.seval("SymbolicRegression.MinimalSR")',
    )
    text = text.replace("result = minisr.fit_mini_sr(", "result = minisr.fit_pysr_compat_sr(")
    text = text.replace(
        "            mutation_weights=dict(self.mutation_weights),\n"
        "            mutation_weight_names=list(self.mutation_weights.keys()),\n",
        "",
    )
    text = text.replace("            log_snapshots=int(self.log_snapshots),\n", "")
    (sandbox / "mini_pysr.py").write_text(text)


def _load_summary(path: Path) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("summary", data)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-ref", default=DEFAULT_LEGACY_REF)
    p.add_argument("--split", default="splits/barely_unsolvable.txt")
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-evals", type=int, default=500_000)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "legacy_minimalsr_barely_v1"))
    p.add_argument("--skeleton-summary", default=str(DEFAULT_SKELETON_SUMMARY))
    p.add_argument("--partition", default="default_partition")
    p.add_argument("--time-limit", default="04:00:00")
    p.add_argument("--mem-per-cpu", default="8G")
    p.add_argument("--job-timeout", type=float, default=None)
    p.add_argument("--max-concurrent-jobs", type=int, default=None)
    p.add_argument("--no-warmstart", action="store_true")
    args = p.parse_args()

    legacy_sha = _resolve_commit(args.legacy_ref, SUBMODULE_PATH)
    sandbox = build_sandbox(legacy_sha)
    _install_legacy_wrapper(sandbox)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = load_dataset_names_from_split(args.split)

    config_snapshot = {
        "legacy_ref": args.legacy_ref,
        "legacy_sha": legacy_sha,
        "sandbox": str(sandbox),
        "split": args.split,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "max_evals": args.max_evals,
        "max_samples": args.max_samples,
        "num_datasets": len(dataset_names),
        "dataset_names": dataset_names,
        "skeleton_summary": args.skeleton_summary,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    minisr_kwargs = get_default_minisr_kwargs()
    minisr_kwargs["max_evals"] = args.max_evals
    config = MiniSRConfig(
        mutation_weights=get_default_mutation_weights(),
        minisr_kwargs=minisr_kwargs,
        name=f"legacy_minimalsr_{legacy_sha[:12]}",
    )
    evaluator = MiniSRSlurmEvaluator(
        results_dir=str(out_dir / "legacy_minimalsr_results"),
        partition=args.partition,
        time_limit=args.time_limit,
        mem_per_cpu=args.mem_per_cpu,
        dataset_max_samples=args.max_samples,
        data_seed=args.seed,
        max_retries=2,
        job_timeout=args.job_timeout,
        max_concurrent_jobs=args.max_concurrent_jobs,
        use_cache=False,
        warm_start=not args.no_warmstart,
        repo_root=str(sandbox),
    )

    print(
        f"[legacy] ref={args.legacy_ref} sha={legacy_sha[:12]} "
        f"datasets={len(dataset_names)} n_runs={args.n_runs}"
    )
    print(f"[legacy] sandbox={sandbox}")
    print(f"[legacy] out_dir={out_dir}")
    t0 = time.time()
    avg, _vec, details = evaluator.evaluate_configs(
        [config],
        dataset_names,
        seed=args.seed,
        n_runs=args.n_runs,
        target_noise_map=None,
        fitness_metric="gt",
    )[0]
    per_seed = _per_seed_averages(details, args.n_runs)
    summary = _summarize(per_seed)
    summary["overall_avg"] = avg
    summary["wall_seconds"] = time.time() - t0
    summary["legacy_sha"] = legacy_sha

    payload = {"summary": summary, "details": details}
    with open(out_dir / "legacy_minimalsr_summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    comparison = {"legacy_minimalsr": summary}
    skeleton_path = Path(args.skeleton_summary)
    if skeleton_path.exists():
        comparison["skeleton_pysr"] = _load_summary(skeleton_path)
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"[legacy] per-seed avgs: {[f'{v:.4f}' for v in per_seed]}")
    print(f"[legacy] GT solve rate: {summary['mean']:.4f} ± {summary['std']:.4f}")
    if "skeleton_pysr" in comparison:
        s = comparison["skeleton_pysr"]
        print(f"[skeleton_pysr] GT solve rate: {s['mean']:.4f} ± {s['std']:.4f}")
    print(f"Artifacts saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
