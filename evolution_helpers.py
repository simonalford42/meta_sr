#!/usr/bin/env python3
"""
GA-specific helpers for evolve_pysr: racing merge, aggregation, parent/survivor
selection, task-aware unsolved-task selection, per-dataset noise maps, and the
evaluator-config wrapper.

Extracted from evolve_pysr.py during the refactor. Bodies are byte-identical
to the originals in evolve_pysr_old.py.
"""

import copy
import hashlib
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from parallel_eval_pysr import (
    PySRConfig,
    PySRSlurmEvaluator,
    run_scores_for_metric,
    metric_missing_fill,
)
from utils import PMLB_PATH, rhs_only

TARGET_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]

def _stable_target_noise(dataset_name: str, seed: int, noise_levels: List[float]) -> float:
    """Deterministically assign a target noise level based on dataset name + seed."""
    digest = hashlib.sha256(f"{seed}:{dataset_name}".encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "little") % len(noise_levels)
    return noise_levels[idx]

def _build_target_noise_map(
    dataset_names: List[str],
    seed: int,
    noise_levels: List[float],
) -> Dict[str, float]:
    """Map each dataset name to a deterministic target noise level."""
    return {name: _stable_target_noise(name, seed, noise_levels) for name in dataset_names}

def _evaluate_configs_with_noise_map(
    evaluator: PySRSlurmEvaluator,
    configs: List[PySRConfig],
    dataset_names: List[str],
    seed: int,
    n_runs: int,
    target_noise_map: Optional[Dict[str, float]] = None,
    fitness_metric: str = "r2",
    run_index_start_per_config: Optional[List[int]] = None,
) -> List[Tuple[float, List[float], List[Dict]]]:
    """Evaluate configs with optional per-dataset target noise mapping."""
    return evaluator.evaluate_configs(
        configs,
        dataset_names,
        seed=seed,
        n_runs=n_runs,
        target_noise_map=target_noise_map,
        fitness_metric=fitness_metric,
        run_index_start_per_config=run_index_start_per_config,
    )

def compute_per_run_avgs(
    result_details: List[Dict],
    n_runs: int,
    fitness_metric: str,
) -> List[float]:
    """Compute per-run averages using the same missing-score policy as aggregate scoring."""
    missing_fill = metric_missing_fill(fitness_metric)
    per_run_avgs: List[float] = []

    for run_idx in range(n_runs):
        run_scores: List[float] = []
        for detail in result_details:
            run_values = run_scores_for_metric(detail, fitness_metric)
            run_scores.append(run_values[run_idx] if len(run_values) > run_idx else missing_fill)
        per_run_avgs.append(float(np.mean(run_scores)) if run_scores else missing_fill)

    return per_run_avgs

def merge_result_details(
    old: Optional[List[Dict]],
    new: List[Dict],
) -> List[Dict]:
    """Append per-run scores from `new` onto `old` per dataset.

    Used by racing mode to accumulate fresh-seed evaluations into a member's
    existing result_details. Assumes both lists are aligned by dataset order
    (as produced by `_aggregate_pysr_results` using the same `dataset_names`).
    If `old` is None/empty this returns a deep copy of `new`.
    """
    if not old:
        return copy.deepcopy(new)
    if len(old) != len(new):
        # Dataset lists don't line up — fall back to the fresh evaluation
        # rather than corrupting the accumulated history.
        return copy.deepcopy(new)

    merged: List[Dict] = []
    for old_d, new_d in zip(old, new):
        old_r2 = list(old_d.get("run_r2_scores", []) or [])
        old_gt = list(old_d.get("run_gt_scores", []) or [])
        new_r2 = list(new_d.get("run_r2_scores", []) or [])
        new_gt = list(new_d.get("run_gt_scores", []) or [])
        run_r2 = old_r2 + new_r2
        run_gt = old_gt + new_gt
        # Frontier-avg R² per seed (fall back to best-eq R² when absent so the
        # accumulated history stays aligned for legacy details).
        old_r2c = list(old_d.get("run_r2c_scores", old_r2) or [])
        new_r2c = list(new_d.get("run_r2c_scores", new_r2) or [])
        run_r2c = old_r2c + new_r2c

        old_eqs = list(old_d.get("best_equations", []) or [])
        new_eqs = list(new_d.get("best_equations", []) or [])
        all_eqs = old_eqs + new_eqs

        old_errs = old_d.get("errors") or []
        new_errs = new_d.get("errors") or []
        all_errs = list(old_errs) + list(new_errs)

        old_evals = list(old_d.get("run_num_evaluations", []) or [])
        new_evals = list(new_d.get("run_num_evaluations", []) or [])
        all_evals = old_evals + new_evals
        valid_evals = [n for n in all_evals if n is not None]

        old_traces = list(old_d.get("execution_traces", []) or [])
        new_traces = list(new_d.get("execution_traces", []) or [])
        all_traces = old_traces + new_traces

        merged.append({
            "dataset": old_d.get("dataset") or new_d.get("dataset"),
            "avg_r2": float(np.mean(run_r2)) if run_r2 else -1.0,
            "avg_r2c": float(np.mean(run_r2c)) if run_r2c else -1.0,
            "avg_gt": float(np.mean(run_gt)) if run_gt else 0.0,
            "run_r2_scores": run_r2,
            "run_r2c_scores": run_r2c,
            "run_gt_scores": run_gt,
            "best_equations": all_eqs,
            "errors": all_errs if all_errs else None,
            "run_num_evaluations": all_evals,
            "avg_num_evaluations": float(np.mean(valid_evals)) if valid_evals else None,
            "n_successful_runs": len(run_r2),
            "n_total_runs": int(old_d.get("n_total_runs", len(old_r2) + len(old_errs)) or 0)
                            + int(new_d.get("n_total_runs", len(new_r2) + len(new_errs)) or 0),
            "execution_traces": all_traces,
        })
    return merged

def recompute_aggregate(
    result_details: List[Dict],
    fitness_metric: str,
) -> Tuple[float, List[float]]:
    """Recompute (avg_score, per_dataset_vector) from merged result_details.

    Uses the same missing-fill policy as `_aggregate_pysr_results`:
    an empty per-dataset run list maps to -1.0 (r2) or 0.0 (gt).
    """
    missing_fill = metric_missing_fill(fitness_metric)

    per_dataset: List[float] = []
    for detail in result_details:
        runs = run_scores_for_metric(detail, fitness_metric)
        per_dataset.append(float(np.mean(runs)) if runs else missing_fill)
    avg = float(np.mean(per_dataset)) if per_dataset else missing_fill
    return avg, per_dataset

def apply_racing_results(
    members: List[Any],
    results: List[Tuple[float, List[float], List[Dict]]],
    fitness_metric: str,
) -> None:
    """Merge fresh-seed evaluation results onto each member's state.

    For every member, appends the new per-run scores onto its existing
    `result_details`, bumps `seeds_evaluated` by however many seeds the new
    eval actually contained (max length of run_*_scores across tasks), and
    recomputes `score` / `score_vector` from the accumulated history.

    Per-bundle seed counts can differ across `members` since racing extra-runs
    may be capped per bundle.

    Members whose `new_details` is empty (e.g. SLURM submission failure, where
    `collect_bundle_futures` substitutes the placeholder `(-1.0, [], [])`) are
    skipped entirely so a failed re-eval does not erase the member's
    accumulated history. The caller is responsible for any retry/logging.
    """
    for m, (_, _, new_details) in zip(members, results):
        if not new_details:
            # Failed extras eval — preserve existing state unchanged.
            continue
        # Count newly added seeds from a metric-independent per-run array (all
        # of run_r2/gt/r2c have one entry per seed).
        n_added = max(
            (
                max(
                    len(d.get("run_r2_scores", []) or []),
                    len(d.get("run_gt_scores", []) or []),
                )
                for d in new_details
            ),
            default=0,
        )
        if n_added == 0:
            # Result list present but no per-seed scores — nothing to merge.
            continue
        merged = merge_result_details(m.result_details, new_details)
        m.result_details = merged
        m.seeds_evaluated = int(getattr(m, "seeds_evaluated", 0) or 0) + n_added
        avg, vec = recompute_aggregate(merged, fitness_metric)
        m.score = avg
        m.score_vector = vec


def lambda_for_gen(gen: int, n_generations: int, lambda_target: int) -> int:
    """Step schedule for the seed-count multiplier λ.

    λ starts at 1 and rises to `lambda_target` in equal-width steps over the
    run. With n_generations=50 and lambda_target=5: gens 0–9 → 1, 10–19 → 2,
    20–29 → 3, 30–39 → 4, 40+ → 5.

    Always clamped to [1, lambda_target]. Robust to lambda_target<=1 and to
    n_generations < lambda_target.
    """
    if lambda_target <= 1:
        return 1
    step = max(1, n_generations // lambda_target)
    return min(lambda_target, 1 + max(0, gen) // step)


def pooled_sigma(bundles: List[Any], fitness_metric: str) -> float:
    """Pooled within-bundle std of per-seed averages (Welch-style).

    For each bundle with N_i ≥ 2 seeds, compute the sample variance of its
    per-seed average scores. Pool with weights (N_i − 1). Returns 0.0 if no
    bundle has enough samples.
    """
    num = 0.0
    den = 0
    for b in bundles:
        N = int(getattr(b, "seeds_evaluated", 0) or 0)
        if N < 2:
            continue
        details = getattr(b, "result_details", None)
        if not details:
            continue
        per_seed = compute_per_run_avgs(details, n_runs=N, fitness_metric=fitness_metric)
        if len(per_seed) < 2:
            continue
        s2 = float(np.var(per_seed, ddof=1))
        num += (len(per_seed) - 1) * s2
        den += len(per_seed) - 1
    if den == 0:
        return 0.0
    return float(np.sqrt(num / den))


def qualification_prob(
    mu_i: float, N_i: int, mu_P: float, N_P: int, sigma: float,
) -> float:
    """P(true_i > true_P) under Gaussian per-seed noise with shared σ.

    Uses Phi((mu_i - mu_P) / sqrt(σ²/N_i + σ²/N_P)). When σ ≤ 0 or counts are
    invalid, treats the comparison as deterministic.
    """
    from scipy.stats import norm
    if sigma <= 0 or N_i <= 0 or N_P <= 0:
        return 1.0 if mu_i >= mu_P else 0.0
    se = float(np.sqrt(sigma**2 / N_i + sigma**2 / N_P))
    if se <= 0:
        return 1.0 if mu_i >= mu_P else 0.0
    return float(norm.cdf((mu_i - mu_P) / se))


def select_qualifying_bundles(
    archive: List[Any],
    population_size: int,
    n_extra_now: int,
    cap_now: int,
    sigma: float,
    fitness_metric: str,
    qual_threshold: float = 0.05,
    max_qualifiers_factor: int = 2,
) -> Tuple[List[Tuple[Any, int]], int]:
    """Pick archive bundles that should get extra-runs evaluation this generation.

    A bundle qualifies if P(true_i > true_P) ≥ `qual_threshold`, where the P-th
    member is taken from the top-`population_size` of `archive` by mean score.
    Bundles already at or beyond `cap_now` seeds are skipped. Among qualifiers,
    we keep at most `max_qualifiers_factor * population_size`, preferring the
    ones with the fewest accumulated seeds (variance-reduction first).

    Returns (qualifiers, n_eligible) where:
      - qualifiers is a list of (bundle, extra_runs_for_this_bundle), capped
        at `max_qualifiers_factor * population_size`. extra_runs is clamped
        so seeds_evaluated never exceeds `cap_now`.
      - n_eligible is the pre-cap count of bundles that passed the
        `qual_threshold` (or all candidates with cap headroom, when the
        archive is too small to define a P-th threshold).
    """
    if not archive or n_extra_now <= 0 or cap_now <= 0:
        return [], 0

    ranked = sorted(
        archive,
        key=lambda b: (b.score if b.score is not None else float("-inf")),
        reverse=True,
    )

    # Without a P-th member there's no threshold — every archive entry that
    # has headroom under the cap is a candidate. Still rank by fewest seeds.
    if len(ranked) < population_size:
        candidates: List[Tuple[Any, int]] = []
        for b in ranked:
            N_i = int(getattr(b, "seeds_evaluated", 0) or 0)
            extra = min(n_extra_now, cap_now - N_i)
            if extra > 0:
                candidates.append((b, extra))
        candidates.sort(key=lambda be: int(getattr(be[0], "seeds_evaluated", 0) or 0))
        return candidates[: max_qualifiers_factor * population_size], len(candidates)

    pth = ranked[population_size - 1]
    mu_P = float(pth.score) if pth.score is not None else 0.0
    N_P = int(getattr(pth, "seeds_evaluated", 0) or 0)

    qualifiers: List[Tuple[Any, int]] = []
    for b in ranked:
        N_i = int(getattr(b, "seeds_evaluated", 0) or 0)
        if N_i >= cap_now:
            continue
        if b.score is None:
            continue
        prob = qualification_prob(float(b.score), N_i, mu_P, N_P, sigma)
        if prob < qual_threshold:
            continue
        extra = min(n_extra_now, cap_now - N_i)
        if extra <= 0:
            continue
        qualifiers.append((b, extra))

    n_eligible = len(qualifiers)
    qualifiers.sort(key=lambda be: int(getattr(be[0], "seeds_evaluated", 0) or 0))
    return qualifiers[: max_qualifiers_factor * population_size], n_eligible

def select_parent(population: list, rng: random.Random):
    """Select a parent using tournament selection (size 2)."""
    candidates = rng.sample(population, min(2, len(population)))
    return max(candidates, key=lambda m: m.score if m.score is not None else -1)

def get_solved_tasks(result_details: Optional[List[Dict]]) -> List[int]:
    """Return indices of tasks where at least one run achieved gt_match >= 1.0."""
    if not result_details:
        return []
    solved = []
    for i, detail in enumerate(result_details):
        run_gt = detail.get("run_gt_scores", [])
        if any(g >= 1.0 for g in run_gt):
            solved.append(i)
    return solved

def format_solved_str(result_details: Optional[List[Dict]]) -> str:
    """Format solved task info for printing, e.g. 'solved 3/20 [0,4,12]'."""
    solved = get_solved_tasks(result_details)
    n_tasks = len(result_details) if result_details else 0
    if not solved:
        return f"solved 0/{n_tasks}"
    indices_str = ",".join(str(i) for i in solved)
    return f"solved {len(solved)}/{n_tasks} [{indices_str}]"


def format_errors_str(result_details: Optional[List[Dict]]) -> str:
    """Format errored-run info for printing, e.g. 'errs 12/100'. Empty when clean."""
    if not result_details:
        return ""
    n_total = sum(int(d.get("n_total_runs", 0) or 0) for d in result_details)
    n_good = sum(int(d.get("n_successful_runs", 0) or 0) for d in result_details)
    n_err = n_total - n_good
    if n_err <= 0:
        return ""
    return f"errs {n_err}/{n_total}"


def _compute_per_task_best(
    bundles: List, n_tasks: int,
) -> List[Optional[Tuple[Any, int, int]]]:
    """For each task, return the bundle with the most solves on that task.

    Entry is (bundle, n_solved, n_total) when at least one run solved the
    task, else None. Ties break by overall bundle score.
    """
    best: List[Optional[Tuple[Any, int, int]]] = [None] * n_tasks
    for task_idx in range(n_tasks):
        best_b = None
        best_solved = 0
        best_total = 0
        for c in bundles:
            rd = getattr(c, "result_details", None)
            if not rd or task_idx >= len(rd):
                continue
            detail = rd[task_idx] or {}
            run_gt = detail.get("run_gt_scores", []) or []
            if not run_gt:
                continue
            n_solved = sum(1 for g in run_gt if g >= 1.0)
            if n_solved == 0:
                continue
            if (
                n_solved > best_solved
                or (
                    n_solved == best_solved
                    and best_b is not None
                    and (c.score or -1) > (best_b.score or -1)
                )
            ):
                best_b = c
                best_solved = n_solved
                best_total = len(run_gt)
        if best_b is not None and best_solved > 0:
            best[task_idx] = (best_b, best_solved, best_total)
    return best


def compute_per_task_best_stats(
    bundles: List, dataset_names: List[str],
) -> Tuple[float, int, int]:
    """Return (avg_best_solve_rate, n_tasks_covered, n_tasks).

    Mirrors the `per-task best avg` reported by format_population_summary:
    for each task, take the winning bundle's solve rate, average across all
    tasks (uncovered tasks contribute 0).
    """
    n_tasks = len(dataset_names)
    if not bundles or n_tasks == 0:
        return 0.0, 0, n_tasks
    best_per_task = _compute_per_task_best(bundles, n_tasks)
    total_rate = 0.0
    for rec in best_per_task:
        if rec is None:
            continue
        _, ns, nt = rec
        if nt > 0:
            total_rate += ns / nt
    avg_best = total_rate / n_tasks
    n_covered = sum(1 for rec in best_per_task if rec is not None)
    return avg_best, n_covered, n_tasks


def format_population_summary(
    bundles: List,
    dataset_names: List[str],
    role_for: Optional[Any] = None,  # Callable[[bundle], str] or None
    header: Optional[str] = None,
) -> str:
    """Describe a population as score + the tasks each bundle is best at.

    For each task in `dataset_names`, find the bundle solving the most runs
    (tie-break by overall score). Invert that mapping so each bundle's line
    lists the task indices it leads on, with per-task solved/total counts.
    Appends `per-task best avg: ...`, the mean over tasks of the winning
    bundle's solve rate (tasks with no solver contribute 0).
    """
    if not bundles or not dataset_names:
        return ""
    n_tasks = len(dataset_names)

    best_per_task = _compute_per_task_best(bundles, n_tasks)

    tasks_per_bundle: Dict[int, List[Tuple[int, int, int]]] = {}
    for task_idx, rec in enumerate(best_per_task):
        if rec is None:
            continue
        b, ns, nt = rec
        tasks_per_bundle.setdefault(id(b), []).append((task_idx, ns, nt))

    lines: List[str] = []
    if header:
        lines.append(header)
    for c in bundles:
        role = role_for(c) if role_for else ""
        role_prefix = f"[{role}] " if role else ""
        score_str = f"{c.score:.4f}" if c.score is not None else "nan"
        best_list = tasks_per_bundle.get(id(c), [])
        if best_list:
            best_str = ", ".join(f"{t} ({ns}/{nt})" for (t, ns, nt) in best_list)
            lines.append(
                f"    {role_prefix}{c.display_name}: score={score_str} "
                f"best for tasks {best_str}"
            )
        else:
            lines.append(f"    {role_prefix}{c.display_name}: score={score_str}")

    total_rate = 0.0
    for rec in best_per_task:
        if rec is None:
            continue
        _, ns, nt = rec
        if nt > 0:
            total_rate += ns / nt
    avg_best = total_rate / n_tasks if n_tasks > 0 else 0.0
    n_tasks_covered = sum(1 for rec in best_per_task if rec is not None)
    lines.append(
        f"  per-task best avg: {avg_best:.4f} (covered {n_tasks_covered}/{n_tasks})"
    )
    return "\n".join(lines)

def _substitute_x_vars(formula: str, var_names: List[str]) -> str:
    """Rewrite `formula` with each variable name replaced by x0, x1, ...

    Uses placeholders so a name that contains another name (e.g. "rho_c_0"
    vs. "c") is not double-substituted.
    """
    if not formula or not var_names:
        return formula
    indexed = sorted(enumerate(var_names), key=lambda kv: -len(kv[1]))
    result = formula
    replacements: List[Tuple[str, str]] = []
    for i, name in indexed:
        if not name:
            continue
        placeholder = f"\x00X{i}\x00"
        result = re.sub(rf"\b{re.escape(name)}\b", placeholder, result)
        replacements.append((placeholder, f"x{i}"))
    for ph, repl in replacements:
        result = result.replace(ph, repl)
    return result


def load_task_formulas(dataset_names: List[str]) -> Dict[str, str]:
    """Load ground-truth formulas for each dataset by reading only metadata.yaml.

    Returns a dict mapping dataset name -> formula string. When feature names
    are listed in the metadata, the returned value is
    "<original> = <x-substituted>" so the trace's x0/x1/... variables are
    cross-referenced with the physics names. Empty string if unavailable.
    """
    from utils import PMLB_PATH, rhs_only
    formulas: Dict[str, str] = {}
    for name in dataset_names:
        formula = ""
        var_names: List[str] = []
        metadata_path = PMLB_PATH / name / "metadata.yaml"
        if metadata_path.exists():
            try:
                import yaml
                with open(metadata_path, "r") as f:
                    metadata = yaml.safe_load(f)
                if isinstance(metadata, dict):
                    desc = metadata.get("description", "") or ""
                    for line in desc.split("\n"):
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            if " in [" not in line and " in (" not in line:
                                formula = rhs_only(line)
                                break
                    for feat in metadata.get("features", []) or []:
                        if isinstance(feat, dict):
                            n = feat.get("name")
                            if isinstance(n, str) and n:
                                var_names.append(n)
            except Exception:
                pass
        if formula and var_names:
            x_form = _substitute_x_vars(formula, var_names)
            if x_form and x_form != formula:
                formula = f"{formula} = {x_form}"
        formulas[name] = formula
    return formulas

def format_bundle_equations_report(
    result_details: List[Dict],
    header_lines: List[str],
    baseline_solved_by_dataset: Optional[Dict[str, int]] = None,
    baseline_n_runs: Optional[int] = None,
) -> str:
    """Render per-(task, seed) GT-vs-predicted with per-task delta-vs-baseline.

    Output format:

        feynman_III_13_18 [Δ=+9; 10/10 vs baseline 1/10]  2*E_n*d**2*k/(h/(2*pi))
          GT:                2*x0*x1**2*x2/(x3/(2*pi))
          seed=0 [solved  ] (((x2 * x0) * 12.566382) / x3) * square(x1)
          seed=1 [unsolved] ...

    Tasks ranked by ``bundle solved - baseline solved`` desc (so the
    tasks-this-bundle-gained-over-baseline cluster at the top). When
    ``baseline_solved_by_dataset`` is None, ranks by bundle solved count only
    and omits the delta annotation.
    """
    dataset_names = [d.get("dataset", "?") for d in result_details]
    task_formulas = load_task_formulas(dataset_names)

    tasks = []
    for detail in result_details:
        dataset = detail.get("dataset", "?")
        run_gt_scores = detail.get("run_gt_scores") or []
        run_best_eqs = detail.get("run_best_equations") or []
        run_matched_eqs = detail.get("run_gt_matched_equations") or []
        legacy_best = detail.get("best_equations") or []
        n_succ = detail.get("n_successful_runs")
        n_total = detail.get("n_total_runs") or len(run_gt_scores)

        # Legacy alignment: only valid when every run succeeded and we have one
        # equation per seed. Otherwise we can't tell which equation belongs to
        # which seed.
        legacy_aligned = (
            not run_best_eqs
            and n_succ is not None
            and n_succ == n_total
            and len(legacy_best) == n_total
        )

        formula_pair = task_formulas.get(dataset, "")
        if "=" in formula_pair:
            original_gt, x_gt = (s.strip() for s in formula_pair.split("=", 1))
        else:
            original_gt = formula_pair
            x_gt = ""

        n_runs = max(len(run_gt_scores), n_total)
        seed_rows = []
        bundle_solved = 0
        for seed_idx in range(n_runs):
            gt_score = run_gt_scores[seed_idx] if seed_idx < len(run_gt_scores) else 0.0
            solved = bool(gt_score and gt_score >= 1.0)
            if solved:
                bundle_solved += 1

            matched_eq = (
                run_matched_eqs[seed_idx]
                if seed_idx < len(run_matched_eqs) else None
            )
            best_eq: Optional[str] = None
            if seed_idx < len(run_best_eqs):
                best_eq = run_best_eqs[seed_idx]
            elif legacy_aligned and seed_idx < len(legacy_best):
                best_eq = legacy_best[seed_idx]
            predicted = matched_eq if (solved and matched_eq) else best_eq
            if predicted is None:
                predicted = "(equation unavailable)"
            seed_rows.append((seed_idx, solved, predicted))

        baseline_solved = (
            baseline_solved_by_dataset.get(dataset)
            if baseline_solved_by_dataset is not None else None
        )
        if baseline_solved is None:
            delta = None
        else:
            delta = bundle_solved - baseline_solved

        tasks.append({
            "dataset": dataset,
            "original_gt": original_gt,
            "x_gt": x_gt,
            "seed_rows": seed_rows,
            "n_runs": n_runs,
            "bundle_solved": bundle_solved,
            "baseline_solved": baseline_solved,
            "delta": delta,
        })

    # Rank by delta desc when available; tiebreak by bundle solved desc, then
    # dataset name. Without baseline, just by bundle solved + name.
    def sort_key(t):
        primary = -(t["delta"] if t["delta"] is not None else t["bundle_solved"])
        return (primary, -t["bundle_solved"], t["dataset"])
    tasks.sort(key=sort_key)

    lines = list(header_lines)
    lines.append("")

    for t in tasks:
        dataset = t["dataset"]
        n_runs = t["n_runs"]
        bundle_solved = t["bundle_solved"]
        baseline_solved = t["baseline_solved"]
        delta = t["delta"]

        if delta is not None and baseline_n_runs is not None:
            sign = "+" if delta >= 0 else ""
            delta_str = (
                f" [Δ={sign}{delta}; {bundle_solved}/{n_runs} vs "
                f"baseline {baseline_solved}/{baseline_n_runs}]"
            )
        elif delta is not None:
            sign = "+" if delta >= 0 else ""
            delta_str = (
                f" [Δ={sign}{delta}; {bundle_solved} vs baseline {baseline_solved}]"
            )
        else:
            delta_str = f" [{bundle_solved}/{n_runs}]"

        header_line = f"{dataset}{delta_str}  {t['original_gt']}"
        lines.append(header_line)
        if t["x_gt"]:
            lines.append(f"  GT: {t['x_gt']}")
        for seed_idx, solved, predicted in t["seed_rows"]:
            status = "solved" if solved else "unsolved"
            lines.append(f"  seed={seed_idx} [{status}] {predicted}")
        lines.append("")

    return "\n".join(lines)


def _detail_has_trace(detail: Optional[Dict]) -> bool:
    if not detail:
        return False
    traces = detail.get("execution_traces") or []
    return any(t for t in traces)

def select_unsolved_task_with_trace(
    parent: Any,
    dataset_names: List[str],
    task_formulas: Dict[str, str],
    rng: random.Random,
) -> Optional[int]:
    """Return an unsolved task index (by `parent`) that has a recorded execution trace."""
    details = getattr(parent, "result_details", None) or []
    solved = set(get_solved_tasks(details))
    candidates = []
    for i, name in enumerate(dataset_names):
        if i in solved or not task_formulas.get(name, ""):
            continue
        detail = details[i] if i < len(details) else None
        if _detail_has_trace(detail):
            candidates.append(i)
    if not candidates:
        return None
    return rng.choice(candidates)

def format_pareto_trace_for_task(
    detail: Dict,
    dataset_name: str,
    formula: str,
) -> Optional[str]:
    """Render a single task's PySR execution trace as a Pareto-front-per-milestone block.

    Picks the first non-empty trace from `detail["execution_traces"]`. Returns
    None if no usable trace is present.
    """
    traces = (detail or {}).get("execution_traces") or []
    trace = next((t for t in traces if t), None)
    if not trace:
        return None

    lines: List[str] = []
    lines.append(f"=== Unsolved task: {dataset_name} ===")
    if formula:
        lines.append(f"Ground truth: {formula}")

    for milestone in trace:
        evals = milestone.get("milestone_evals")
        equations = milestone.get("equations") or []
        header = f"--- Pareto front after {evals:,} evals ---" if isinstance(evals, int) else "--- Pareto front ---"
        lines.append("")
        lines.append(header)
        for eq in equations:
            try:
                cplx = int(eq.get("complexity"))
                loss = float(eq.get("loss"))
            except (TypeError, ValueError):
                continue
            equation = str(eq.get("equation", "")).strip()
            if not equation:
                continue
            lines.append(f"  c={cplx:>3}  loss={loss:.4g}   {equation}")

    return "\n".join(lines)

def select_survivors(population: list, offspring: list, population_size: int) -> list:
    """Select best individuals from population + offspring."""
    combined = population + offspring
    scored = [m for m in combined if m.score is not None]
    scored.sort(key=lambda m: m.score, reverse=True)
    return scored[:population_size]

def _bundle_loc(bundle) -> int:
    """Total non-blank lines of code across a bundle's operators."""
    total = 0
    for op in (getattr(bundle, "operators", {}) or {}).values():
        if op is None or not getattr(op, "code", None):
            continue
        total += sum(1 for line in op.code.splitlines() if line.strip())
    return total


def select_survivors_complexity(
    population: list, offspring: list, population_size: int,
) -> list:
    """Complexity-aware Pareto survivor selection.

    Buckets candidates by total bundle LOC into `population_size` equal-width
    buckets spanning [min_loc, max_loc]. Picks the top-fitness candidate in
    each non-empty bucket, then drops anything Pareto-dominated by a
    smaller-LOC bucket-best with at-least-as-good fitness. Backfills with
    top-scored remaining candidates if the Pareto front is smaller than
    population_size (so the search doesn't collapse).
    """
    combined = population + offspring
    scored = [m for m in combined if m.score is not None]
    if not scored:
        return []
    if len(scored) <= population_size:
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored

    locs = [_bundle_loc(b) for b in scored]
    lo, hi = min(locs), max(locs)
    if hi == lo or population_size <= 1:
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:population_size]

    width = (hi - lo) / population_size
    buckets: List[List] = [[] for _ in range(population_size)]
    for b, l in zip(scored, locs):
        idx = int((l - lo) / width)
        if idx >= population_size:
            idx = population_size - 1
        buckets[idx].append(b)

    # One winner per non-empty bucket: highest fitness, tie-break by lower LOC.
    bucket_bests = []
    for bk in buckets:
        if bk:
            bk_sorted = sorted(bk, key=lambda b: (-b.score, _bundle_loc(b)))
            bucket_bests.append(bk_sorted[0])

    # Pareto sweep over bucket-bests sorted by LOC ascending: keep only those
    # whose fitness strictly exceeds the running max — equivalent to "if a
    # fewer-LOC bundle has at-least-as-good fitness, drop the higher-LOC one".
    bucket_bests.sort(key=lambda b: (_bundle_loc(b), -b.score))
    pareto = []
    pareto_ids = set()
    best_fit_so_far = float("-inf")
    for b in bucket_bests:
        if b.score > best_fit_so_far:
            pareto.append(b)
            pareto_ids.add(id(b))
            best_fit_so_far = b.score

    # Backfill with top-scored remaining candidates if the front is short of
    # population_size, so the population doesn't collapse on flat fitness.
    if len(pareto) < population_size:
        remaining = [c for c in scored if id(c) not in pareto_ids]
        remaining.sort(key=lambda m: m.score, reverse=True)
        for c in remaining:
            pareto.append(c)
            pareto_ids.add(id(c))
            if len(pareto) >= population_size:
                break

    pareto.sort(key=lambda m: m.score, reverse=True)

    print(
        f"  [complexity] Population: {len(pareto)} "
        f"(buckets filled: {sum(1 for bk in buckets if bk)}/{population_size}, "
        f"loc range: {lo}-{hi})"
    )
    for c in pareto:
        loc = _bundle_loc(c)
        print(f"    loc={loc:>3}  score={c.score:.4f}  {c.display_name}")

    return pareto


def select_survivors_diverse(
    population: list, offspring: list, min_population_size: int,
    dataset_names: List[str],
) -> list:
    """Task-diverse survivor selection.

    For each task, pick the candidate that solves the most runs on that task
    (count of run_gt_scores >= 1.0), breaking ties by highest overall score.
    Candidates with zero solved runs on a task are not selected for that task.
    Then backfill with top-scoring candidates until we reach min_population_size.

    Population can grow up to len(dataset_names) but never shrinks below
    min_population_size.
    """
    combined = population + offspring
    scored = [m for m in combined if m.score is not None]
    scored.sort(key=lambda m: m.score, reverse=True)

    # Step 1: For each task, find the candidate solving the most runs
    # (tie-break by overall score, already pre-sorted descending).
    frontier: Dict[int, Any] = {}  # candidate id -> candidate (use id to dedup)
    for task_idx in range(len(dataset_names)):
        best_solver = None
        best_n_solved = 0
        for candidate in scored:
            rd = getattr(candidate, "result_details", None)
            if not rd or task_idx >= len(rd):
                continue
            run_gt = rd[task_idx].get("run_gt_scores", [])
            n_solved = sum(1 for g in run_gt if g >= 1.0)
            if n_solved == 0:
                continue
            if n_solved > best_n_solved or (
                n_solved == best_n_solved
                and best_solver is not None
                and candidate.score > best_solver.score
            ):
                best_solver = candidate
                best_n_solved = n_solved
        if best_solver is not None:
            frontier[id(best_solver)] = best_solver

    frontier_list = sorted(frontier.values(), key=lambda m: m.score, reverse=True)

    # Step 2: Backfill with top-scoring candidates not in frontier
    if len(frontier_list) < min_population_size:
        frontier_ids = set(frontier.keys())
        for candidate in scored:
            if id(candidate) not in frontier_ids:
                frontier_list.append(candidate)
                frontier_ids.add(id(candidate))
            if len(frontier_list) >= min_population_size:
                break

    frontier_ids = set(frontier.keys())

    def _role_for(c):
        return "frontier" if id(c) in frontier_ids else "backfill"

    print(f"  [diverse] Population: {len(frontier_list)} "
          f"(frontier: {len(frontier)}, backfill: {len(frontier_list) - len(frontier)})")
    summary = format_population_summary(
        frontier_list, dataset_names, role_for=_role_for,
    )
    if summary:
        print(summary)

    # Report tasks no bundle in the population can solve — useful for spotting
    # systematic gaps in the meta-search.
    best_per_task = _compute_per_task_best(frontier_list, len(dataset_names))
    uncovered = [i for i, rec in enumerate(best_per_task) if rec is None]
    if uncovered:
        print(f"  [diverse] uncovered tasks: {uncovered}")

    return frontier_list
