#!/usr/bin/env python3
"""Shared IO + aggregation for full-SRBench evaluations.

This module is intentionally dependency-light (only stdlib): it reads the
on-disk artifacts produced by ``srbench_full_eval.py`` (a ``manifest.json``
plus per-batch ``tasks.json`` / ``results/task_NNNNNN.json`` files written by
the PySR or FullSR evaluator) and turns them into a single
results dict keyed by ``"<dataset>|<seed>|<noise>"``. Both
``srbench_full_eval.py`` and ``inspect_srbench_results.py`` import from here so
the two scripts never diverge on how results are joined or aggregated.

It does NOT import parallel_eval_pysr (and therefore not pysr/julia), so
``inspect_srbench_results.py`` stays fast to start.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The 3 SRBench ground-truth tasks whose formulas need inverse trig
# (arcsin / arccos), which is outside our operator set — a symbolic GT match is
# impossible, so they can never be "solved". Kept in the 133-task grid for a
# canonical denominator but flagged so stats can optionally exclude them.
UNSOLVABLE_TASKS = ("feynman_test_10", "feynman_I_26_2", "feynman_I_30_5")

FAMILIES = ("all", "feynman", "strogatz")


def family_of(dataset_name: str) -> str:
    """Return 'Feynman' or 'Strogatz' for a dataset (matches process_pysr_results.py)."""
    return "Feynman" if "feynman" in dataset_name.lower() else "Strogatz"


def fmt_noise(value: float) -> str:
    """Canonical string form of a noise level, used consistently for dict keys."""
    return f"{float(value):g}"


def result_key(dataset: str, seed: int, noise: float) -> str:
    return f"{dataset}|{int(seed)}|{fmt_noise(noise)}"


# ---------------------------------------------------------------------------
# Loading / joining raw batch artifacts
# ---------------------------------------------------------------------------

def load_manifest(run_dir: "str | Path") -> Dict[str, Any]:
    with open(Path(run_dir) / "manifest.json") as f:
        return json.load(f)


def build_keyed_results(run_dir: "str | Path", manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Join each batch's tasks.json with its results/task_NNNNNN.json files.

    Returns a dict keyed by ``"<dataset>|<seed>|<noise>"``. Every expected
    (task, seed, noise) combination gets an entry; tasks whose result file is
    missing are recorded with ``present=False``. The per-row ``seed`` is the
    actual run seed (base seed + run_index), matching run_pysr_srbench.py.
    """
    run_dir = Path(run_dir)
    if manifest is None:
        manifest = load_manifest(run_dir)

    results: Dict[str, Dict[str, Any]] = {}
    for batch in manifest["batches"]:
        batch_dir = run_dir / batch["batch_dir"]
        tasks_file = batch_dir / "tasks.json"
        if not tasks_file.exists():
            continue
        with open(tasks_file) as f:
            specs = json.load(f)
        results_subdir = batch_dir / "results"
        for idx, spec in enumerate(specs):
            dataset = spec["dataset_name"]
            seed = int(spec["seed"]) + int(spec.get("run_index", 0))
            noise = spec.get("target_noise", 0.0)
            key = result_key(dataset, seed, noise)

            res_path = results_subdir / f"task_{idx:06d}.json"
            entry: Dict[str, Any] = {
                "dataset": dataset,
                "family": family_of(dataset),
                "seed": seed,
                "run_index": int(spec.get("run_index", 0)),
                "noise": float(noise),
                "present": False,
                "solved": False,
                "gt_match_score": None,
                "test_r2": None,
                "runtime_seconds": None,
                "solve_time": None,
                "solve_time_source": None,
                "best_equation": None,
                "best_loss": None,
                "pareto_frontier": None,
                "config_id": int(spec.get("config_id", 0)),
                "error": None,
            }
            if res_path.exists():
                try:
                    with open(res_path) as f:
                        res = json.load(f)
                    gt = res.get("gt_match_score")
                    solve_time, solve_time_source = _solve_time_from_result(res)
                    entry.update(
                        present=True,
                        gt_match_score=gt,
                        solved=bool(gt is not None and gt >= 1.0),
                        test_r2=res.get("r2_score"),
                        runtime_seconds=res.get("runtime_seconds"),
                        solve_time=solve_time,
                        solve_time_source=solve_time_source,
                        best_equation=res.get("best_equation"),
                        best_loss=res.get("best_loss"),
                        pareto_frontier=res.get("pareto_frontier"),
                        error=res.get("error"),
                    )
                except Exception as e:  # corrupt/partial file
                    entry["error"] = f"unreadable result file: {e}"
            results[key] = entry
    return results


def merge_keyed_results(
    keyed: Dict[str, Dict[str, Any]],
    *,
    base_seed: int,
    bundle_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Collapse constituent searches into one result per dataset/noise."""
    from frontier_aggregation import merge_frontiers

    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for entry in keyed.values():
        grouped.setdefault((entry["dataset"], float(entry["noise"])), []).append(entry)

    merged: Dict[str, Dict[str, Any]] = {}
    for (dataset, noise), entries in grouped.items():
        entries.sort(key=lambda e: (e["run_index"], e.get("config_id", 0)))
        successful = [e for e in entries if e["present"] and e["error"] is None]
        sources = [{
            "source_run_index": e["run_index"],
            "source_seed": e["seed"],
            "source_config_id": e.get("config_id", 0),
            **({"source_bundle": bundle_names[e.get("config_id", 0)]}
               if bundle_names and e.get("config_id", 0) in bundle_names else {}),
        } for e in successful]
        frontier = merge_frontiers(
            [e.get("pareto_frontier") for e in successful], sources=sources,
        )
        best = frontier[-1] if frontier else None
        solved_rows = [row for row in frontier if row.get("solved")]
        solve_times = [_entry_solve_time(e) for e in successful]
        solve_times = [value for value in solve_times if value is not None]
        result = {
            "dataset": dataset,
            "family": family_of(dataset),
            "seed": int(base_seed),
            "run_index": 0,
            "noise": noise,
            "present": bool(successful),
            "solved": bool(solved_rows),
            "gt_match_score": 1.0 if solved_rows else 0.0,
            "test_r2": best.get("r2", best.get("test_r2")) if best else None,
            "runtime_seconds": sum(float(e.get("runtime_seconds") or 0.0) for e in entries),
            "solve_time": sum(solve_times) if solve_times else None,
            "solve_time_source": "sum_of_constituent_searches",
            "best_equation": best.get("equation") if best else None,
            "best_loss": best.get("train_mse") if best else None,
            "pareto_frontier": frontier,
            "error": None if successful else "No successful constituent searches",
            "n_searches": len(entries),
            "n_successful_searches": len(successful),
            "constituent_seeds": [e["seed"] for e in entries],
        }
        merged[result_key(dataset, base_seed, noise)] = result
    return merged


def _solve_time_from_result(res: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """(solve_time, source) for one raw task result dict.

    Prefers the execution-trace ``chunk_runtime`` sum (pure fit wall time,
    the methodology of scripts/analyze_pysr_solve_time.py); ``runtime_seconds``
    also includes worker startup / Julia compile overhead (~4.4x inflated on
    average) and is only used as a fallback for trace-less runs.
    """
    trace = res.get("execution_trace") or []
    chunks = [
        e.get("chunk_runtime") for e in trace
        if isinstance(e, dict) and e.get("chunk_runtime") is not None
    ]
    if chunks:
        return float(sum(chunks)), "trace"
    if res.get("runtime_seconds") is not None:
        return float(res["runtime_seconds"]), "runtime_seconds"
    return None, None


def _entry_solve_time(e: Dict[str, Any]) -> Optional[float]:
    """Solve time for an aggregated entry; falls back to runtime_seconds for
    entries persisted before the solve_time field existed."""
    if e.get("solve_time") is not None:
        return e["solve_time"]
    return e.get("runtime_seconds")


def expected_keys(manifest: Dict[str, Any]) -> List[str]:
    """All (dataset, seed, noise) keys the run is supposed to produce."""
    seeds = manifest["seeds"]
    keys = []
    for ds in manifest["datasets"]:
        for s in seeds:
            for noise in manifest["noise_levels"]:
                keys.append(result_key(ds, s, noise))
    return keys


# ---------------------------------------------------------------------------
# Persisting / loading the aggregated big JSON
# ---------------------------------------------------------------------------

def save_keyed_results(run_dir: "str | Path", keyed: Dict[str, Dict[str, Any]], meta: Dict[str, Any],
                       filename: str = "srbench_full_results.json") -> Path:
    out = Path(run_dir) / filename
    with open(out, "w") as f:
        json.dump({"meta": meta, "results": keyed}, f, indent=2)
    return out


def load_keyed_results(run_dir: "str | Path") -> Optional[Dict[str, Dict[str, Any]]]:
    path = Path(run_dir) / "srbench_full_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("results", {})


# ---------------------------------------------------------------------------
# Aggregation: solve rate + solve time per family x noise
# ---------------------------------------------------------------------------

def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _group_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute solve rate + solve time for one (family, noise) group of run entries."""
    present = [e for e in entries if e["present"] and e["error"] is None]
    solved = [e for e in present if e["solved"]]

    # per-task-any: a dataset counts as solved if ANY of its seeds solved
    by_dataset: Dict[str, bool] = {}
    for e in present:
        by_dataset[e["dataset"]] = by_dataset.get(e["dataset"], False) or e["solved"]
    n_datasets = len(by_dataset)
    n_solved_datasets = sum(1 for v in by_dataset.values() if v)

    return {
        "n_runs_present": len(present),
        "n_runs_solved": len(solved),
        "solve_rate_per_run": (len(solved) / len(present)) if present else None,
        "n_datasets": n_datasets,
        "n_datasets_solved": n_solved_datasets,
        "solve_rate_per_task_any": (n_solved_datasets / n_datasets) if n_datasets else None,
        # Trace-derived where available (see _solve_time_from_result); the
        # source-mix counters below say how many entries fell back to the
        # contaminated runtime_seconds.
        "solve_time_mean_solved": _mean([_entry_solve_time(e) for e in solved]),
        "solve_time_mean_all": _mean([_entry_solve_time(e) for e in present]),
        "solve_time_n_trace": sum(
            1 for e in present if e.get("solve_time_source") == "trace"
        ),
        "solve_time_n_runtime_fallback": sum(
            1 for e in present
            if _entry_solve_time(e) is not None
            and e.get("solve_time_source") != "trace"
        ),
        # Mean test R^2 over present runs, clamped at 0 so a handful of wildly
        # negative fits don't dominate the average.
        "test_r2_mean": _mean([max(0.0, e["test_r2"]) for e in present
                               if e["test_r2"] is not None]),
    }


def aggregate_metrics(
    keyed: Dict[str, Dict[str, Any]],
    noise_levels: List[float],
    exclude_unsolvable: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return metrics[family][noise_str | 'all'] = {solve rates, solve times, counts}.

    family in {'all', 'feynman', 'strogatz'}; the 'all' noise bucket rolls up
    across every noise level.
    """
    entries = list(keyed.values())
    if exclude_unsolvable:
        entries = [e for e in entries if e["dataset"] not in UNSOLVABLE_TASKS]

    noise_strs = [fmt_noise(n) for n in noise_levels]

    def family_filter(e, fam):
        if fam == "all":
            return True
        return e["family"].lower() == fam

    metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for fam in FAMILIES:
        fam_entries = [e for e in entries if family_filter(e, fam)]
        metrics[fam] = {}
        for ns in noise_strs:
            group = [e for e in fam_entries if fmt_noise(e["noise"]) == ns]
            metrics[fam][ns] = _group_metrics(group)
        metrics[fam]["all"] = _group_metrics(fam_entries)
    return metrics


def flatten_metrics_for_wandb(metrics: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, float]:
    """Flatten the nested metrics dict to ``srbench/<family>/noise_<n>/<metric>`` keys."""
    flat: Dict[str, float] = {}
    for fam, by_noise in metrics.items():
        for noise_str, m in by_noise.items():
            prefix = f"srbench/{fam}/noise_{noise_str}"
            for k, v in m.items():
                if v is not None:
                    flat[f"{prefix}/{k}"] = v
    return flat


# ---------------------------------------------------------------------------
# Console + wandb reporting (shared by both scripts)
# ---------------------------------------------------------------------------

def format_metrics_console(metrics: Dict[str, Dict[str, Dict[str, Any]]], noise_levels: List[float]) -> str:
    lines = []
    noise_strs = [fmt_noise(n) for n in noise_levels] + ["all"]
    for fam in FAMILIES:
        lines.append(f"\n=== {fam.upper()} ===")
        header = (f"  {'noise':>8}  {'solve%/run':>11}  {'solve%/task':>12}  "
                  f"{'t_solved(s)':>12}  {'t_all(s)':>10}  {'runs':>10}  {'tasks':>10}")
        lines.append(header)
        for ns in noise_strs:
            m = metrics[fam].get(ns)
            if not m:
                continue
            spr = m["solve_rate_per_run"]
            spt = m["solve_rate_per_task_any"]
            tsolved = m["solve_time_mean_solved"]
            tall = m["solve_time_mean_all"]
            lines.append(
                f"  {ns:>8}  "
                f"{(f'{spr*100:.1f}' if spr is not None else '-'):>11}  "
                f"{(f'{spt*100:.1f}' if spt is not None else '-'):>12}  "
                f"{(f'{tsolved:.1f}' if tsolved is not None else '-'):>12}  "
                f"{(f'{tall:.1f}' if tall is not None else '-'):>10}  "
                f"{m['n_runs_solved']}/{m['n_runs_present']:<8}  "
                f"{m['n_datasets_solved']}/{m['n_datasets']:<8}"
            )
    return "\n".join(lines)


def log_wandb_table_and_metrics(run, keyed: Dict[str, Dict[str, Any]], noise_levels: List[float]) -> None:
    """Log a per-run results Table plus flattened solve-rate/solve-time metrics."""
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return

    columns = ["dataset", "family", "seed", "noise", "solved",
               "gt_match_score", "test_r2", "runtime_seconds", "best_equation"]
    table = wandb.Table(columns=columns)
    for e in keyed.values():
        if not e["present"]:
            continue
        table.add_data(
            e["dataset"], e["family"], e["seed"], e["noise"], e["solved"],
            e["gt_match_score"], e["test_r2"], e["runtime_seconds"],
            (e["best_equation"] or "")[:500],
        )
    run.log({"srbench/results_table": table})

    metrics = aggregate_metrics(keyed, noise_levels)
    run.log(flatten_metrics_for_wandb(metrics))
    # Also log a variant that excludes the 3 inverse-trig unsolvables.
    metrics_excl = aggregate_metrics(keyed, noise_levels, exclude_unsolvable=True)
    excl_flat = {k.replace("srbench/", "srbench_excl_unsolvable/"): v
                 for k, v in flatten_metrics_for_wandb(metrics_excl).items()}
    run.log(excl_flat)
