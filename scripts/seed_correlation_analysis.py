"""
Test whether seeds (run_index 0..9) are systematically lucky/unlucky across bundles.

Experiment 1 — Are any seeds biased?
  Build matrix S[bundle, seed] of solve rates.
  Decompose S[i,j] = mu + alpha_i (bundle) + beta_j (seed) + eps.
  Estimate beta_j (seed "bonus") and test the null beta_j = 0 forall j.
    - Omnibus: two-way ANOVA F-test on the seed factor (blocked design with
      bundles as blocks). Friedman test reported alongside (rank-based, robust
      to bounded/non-normal solve rates).
    - Per-seed: t_j = beta_j_hat / SE(beta_j_hat), Holm-Bonferroni corrected
      across the 10 seeds.

Experiment 2 — Do parent-offspring pairs share seed luck?
  For each offspring bundle whose single evolved operator-slot points to a
  parent operator, find the parent bundle in the same run (the previous-gen
  bundle whose corresponding slot has that name). Compute:
    - Per-pair Pearson correlation of (parent_seed_scores, offspring_seed_scores).
    - Paired vs unpaired variance:
        V_paired   = Var_j(S_o[j] - S_p[j])
        V_unpaired = Var_j(S_o[j]) + Var_j(S_p[j])
      Ratio < 1 means using paired seeds reduces noise on (offspring - parent)
      comparisons.
  Stratify by mutation mode (refine/explore/simplify/crossover) and by which
  slot was evolved.

We run both experiments twice:
  - pre_apr29  (boundary 2026-04-29 16:58): runs whose data lived in the
    pre-truncate cache backup.
  - post_apr29: runs whose data lived in the current cache DB.

Outputs go to scripts/out/seed_correlation/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
OUT_DIR = REPO_ROOT / "scripts" / "out" / "seed_correlation"

# Apr 29 16:58 — when caches/pysr_evaluation_cache.db was truncated/migrated.
# Runs starting before this used the pre-truncate cache (now in .pre_truncate_backup/);
# runs starting after used the new cache file.
BOUNDARY = datetime(2026, 4, 29, 16, 58)

N_SEEDS = 10
SLOTS = ("mutation", "survival", "selection", "loss")


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class BundleRecord:
    """One bundle's per-seed solve rate vector and lineage info."""

    bundle_uid: str  # hash of (run_dir, gen, slot_in_gen) — unique per appearance
    bundle_code_hash: str  # hash of (mutation+survival+selection+loss code) — dedup id
    run_dir: str
    run_start: datetime
    generation: int
    is_offspring: bool
    pop_index: int
    evolved_type: Optional[str]
    fitness_metric: str
    seeds: np.ndarray  # shape (N_SEEDS,) — solve rate per seed, NaN if missing
    seeds_evaluated: int
    operator_names: Dict[str, str]  # slot -> operator name
    operator_parent_names: Dict[str, Optional[str]]  # slot -> parent op name
    operator_modes: Dict[str, Optional[str]]  # slot -> mode (refine/explore/...)


def _bundle_code_hash(operators: Dict) -> str:
    h = hashlib.sha256()
    for slot in SLOTS:
        op = operators.get(slot) or {}
        h.update((slot + "::" + (op.get("code") or "")).encode("utf-8"))
    return h.hexdigest()[:16]


def _per_seed_solve_rate(bundle: Dict, fitness_metric: str) -> np.ndarray:
    """Mean over datasets of per-seed score. NaN where seed wasn't evaluated."""
    score_key = "run_gt_scores" if fitness_metric == "gt" else "run_r2_scores"
    rd = bundle.get("result_details") or []
    if not rd:
        return np.full(N_SEEDS, np.nan)
    out = np.full(N_SEEDS, np.nan)
    counts = np.zeros(N_SEEDS, dtype=int)
    sums = np.zeros(N_SEEDS)
    for d in rd:
        vals = d.get(score_key) or []
        for j, v in enumerate(vals[:N_SEEDS]):
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            sums[j] += fv
            counts[j] += 1
    for j in range(N_SEEDS):
        if counts[j] > 0:
            out[j] = sums[j] / counts[j]
    return out


def _operator_meta(operators: Dict) -> Tuple[Dict[str, str], Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    names, parents, modes = {}, {}, {}
    for slot in SLOTS:
        op = operators.get(slot) or {}
        names[slot] = op.get("name") or ""
        parents[slot] = op.get("parent_name")
        modes[slot] = op.get("mode")
    return names, parents, modes


def load_run(run_data_path: Path) -> Optional[Tuple[datetime, str, List[BundleRecord]]]:
    try:
        d = json.load(open(run_data_path))
    except Exception:
        return None

    st = d.get("start_time")
    if not st:
        return None
    try:
        dt = datetime.fromisoformat(st.replace("Z", ""))
    except Exception:
        return None

    cfg = d.get("config") or {}
    fitness_metric = cfg.get("fitness_metric") or "gt"
    run_dir = str(run_data_path.parent.name)

    records: List[BundleRecord] = []
    for g in d.get("generations", []):
        gen = g.get("generation", 0)
        evolved_type = g.get("evolved_type")
        for is_off, label in [(False, "population"), (True, "offspring")]:
            for i, b in enumerate(g.get(label, []) or []):
                ops = b.get("operators") or {}
                seeds = _per_seed_solve_rate(b, fitness_metric)
                names, parents, modes = _operator_meta(ops)
                rec = BundleRecord(
                    bundle_uid=f"{run_dir}:g{gen}:{label}:{i}",
                    bundle_code_hash=_bundle_code_hash(ops),
                    run_dir=run_dir,
                    run_start=dt,
                    generation=gen,
                    is_offspring=is_off,
                    pop_index=i,
                    evolved_type=evolved_type,
                    fitness_metric=fitness_metric,
                    seeds=seeds,
                    seeds_evaluated=int(b.get("seeds_evaluated") or 0),
                    operator_names=names,
                    operator_parent_names=parents,
                    operator_modes=modes,
                )
                records.append(rec)
    return dt, fitness_metric, records


def load_all_runs(runs_dir: Path = RUNS_DIR) -> List[BundleRecord]:
    all_records: List[BundleRecord] = []
    for run_data_path in sorted(runs_dir.glob("*/run_data.json")):
        loaded = load_run(run_data_path)
        if loaded is None:
            continue
        _, _, recs = loaded
        all_records.extend(recs)
    return all_records


# --------------------------------------------------------------------------- #
# Experiment 1 — seed bonuses                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class Exp1Result:
    bonuses: np.ndarray  # shape (N_SEEDS,)
    bonus_se: np.ndarray
    t_stats: np.ndarray
    p_values: np.ndarray
    p_values_holm: np.ndarray
    f_stat: float
    f_p_value: float
    friedman_stat: float
    friedman_p_value: float
    n_bundles: int
    df_seed: int
    df_resid: int
    ms_resid: float
    label: str


def holm_bonferroni(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * p[idx]
        running = max(running, candidate)
        adj[idx] = min(1.0, running)
    return adj


def experiment_1(records: List[BundleRecord], label: str, min_seed_var: float = 1e-9) -> Optional[Exp1Result]:
    """Two-way ANOVA seed-effect test on the (bundle x seed) solve-rate matrix.

    Restricts to bundles with all N_SEEDS seed entries finite. Bundles that are
    constant across seeds (all 0 or all 1) contribute nothing — they're kept
    because the math still works (residual = 0, no effect), but we drop them to
    keep the residual variance estimate honest.
    """
    rows: List[np.ndarray] = []
    seen_bundles: set = set()
    for r in records:
        if r.bundle_code_hash in seen_bundles:
            continue  # dedupe identical bundles
        if r.seeds_evaluated < N_SEEDS:
            continue
        if not np.all(np.isfinite(r.seeds)):
            continue
        if float(np.var(r.seeds)) < min_seed_var:
            continue  # constant across seeds — no info
        rows.append(r.seeds)
        seen_bundles.add(r.bundle_code_hash)

    if len(rows) < 5:
        return None

    S = np.stack(rows, axis=0)  # (N_b, N_SEEDS)
    n_b = S.shape[0]
    bundle_mean = S.mean(axis=1, keepdims=True)
    grand_mean = S.mean()
    # Bundle-centered residuals; the *seed-bonus* estimate is the column mean.
    R = S - bundle_mean
    bonuses = R.mean(axis=0)  # shape (N_SEEDS,), sums to 0

    # Two-way ANOVA (randomized block) decomposition.
    # SS_seed = N_b * sum(beta_j^2). df_seed = N_SEEDS - 1.
    # Residual is the full additive-model residual:
    #     E[i,j] = S[i,j] - bundle_mean[i] - bonuses[j]   (grand_mean cancels)
    E = S - bundle_mean - bonuses[None, :]
    ss_seed = n_b * float(np.sum(bonuses ** 2))
    ss_resid = float(np.sum(E ** 2))
    df_seed = N_SEEDS - 1
    df_resid = (n_b - 1) * (N_SEEDS - 1)
    ms_seed = ss_seed / df_seed
    ms_resid = ss_resid / df_resid if df_resid > 0 else float("nan")
    f_stat = ms_seed / ms_resid if ms_resid > 0 else float("nan")
    f_p = float(stats.f.sf(f_stat, df_seed, df_resid)) if math.isfinite(f_stat) else float("nan")

    # Per-seed standard error: SE(beta_j) ~ sqrt(MS_resid * (N_SEEDS-1) / (N_b * N_SEEDS))
    # which is the standard expression for the SE of a marginal column-effect
    # estimate under the additive model. (Equivalent to sd_of_residuals/sqrt(N_b)
    # up to the (N_SEEDS-1)/N_SEEDS factor.)
    se = math.sqrt(ms_resid * (N_SEEDS - 1) / (n_b * N_SEEDS)) if ms_resid > 0 else float("nan")
    se_arr = np.full(N_SEEDS, se)
    t_stats = bonuses / se_arr
    # Two-sided p, df = df_resid.
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df_resid)
    p_holm = holm_bonferroni(p_values)

    # Friedman: rank within each bundle, test rank-mean homogeneity across seeds.
    friedman_stat, friedman_p = stats.friedmanchisquare(*[S[:, j] for j in range(N_SEEDS)])

    return Exp1Result(
        bonuses=bonuses,
        bonus_se=se_arr,
        t_stats=t_stats,
        p_values=p_values,
        p_values_holm=p_holm,
        f_stat=float(f_stat),
        f_p_value=float(f_p),
        friedman_stat=float(friedman_stat),
        friedman_p_value=float(friedman_p),
        n_bundles=n_b,
        df_seed=df_seed,
        df_resid=df_resid,
        ms_resid=float(ms_resid),
        label=label,
    )


def plot_exp1(res: Exp1Result, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    seeds = np.arange(N_SEEDS)
    ci_half = 1.96 * res.bonus_se
    sig_mask = res.p_values_holm < 0.05
    colors = ["#cc4444" if s else "#3b6ea5" for s in sig_mask]

    ax.bar(seeds, res.bonuses, color=colors, yerr=ci_half, capsize=4, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Seed index (run_index)")
    ax.set_ylabel("Seed bonus (mean solve-rate residual after removing bundle mean)")
    ax.set_xticks(seeds)
    title_lines = [
        f"Experiment 1 ({res.label}) — N bundles = {res.n_bundles}",
        f"Omnibus F({res.df_seed}, {res.df_resid}) = {res.f_stat:.2f}, p = {res.f_p_value:.2g}; "
        f"Friedman chi^2({res.df_seed}) = {res.friedman_stat:.2f}, p = {res.friedman_p_value:.2g}",
        "Red bars: Holm-Bonferroni p < 0.05",
    ]
    ax.set_title("\n".join(title_lines), fontsize=10)
    for j in seeds:
        ax.annotate(
            f"{res.bonuses[j]:+.3f}\np={res.p_values_holm[j]:.2g}",
            (j, res.bonuses[j]),
            textcoords="offset points",
            xytext=(0, 6 if res.bonuses[j] >= 0 else -22),
            ha="center",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Experiment 2 — parent / offspring correlation                               #
# --------------------------------------------------------------------------- #


@dataclass
class PairRecord:
    run_dir: str
    parent_uid: str
    offspring_uid: str
    slot: str
    mode: Optional[str]
    parent_seeds: np.ndarray
    offspring_seeds: np.ndarray


def find_pairs(records: List[BundleRecord]) -> List[PairRecord]:
    """Pair each offspring with its parent bundle by matching the evolved slot's parent_name.

    Restricted to runs where evolved_type names a single slot. In multi-slot
    generations the notion of 'the parent' is ambiguous so we skip them.
    """
    by_run: Dict[str, List[BundleRecord]] = defaultdict(list)
    for r in records:
        by_run[r.run_dir].append(r)

    pairs: List[PairRecord] = []
    for run_dir, recs in by_run.items():
        # name -> latest BundleRecord per slot (use most recent generation we've seen)
        by_slot_name: Dict[Tuple[str, str], BundleRecord] = {}
        # Iterate in generation order so "latest" is well-defined.
        recs_sorted = sorted(recs, key=lambda r: (r.generation, 0 if not r.is_offspring else 1))
        for r in recs_sorted:
            for slot in SLOTS:
                nm = r.operator_names.get(slot)
                if nm:
                    by_slot_name[(slot, nm)] = r

            # If this record is an offspring of a single-slot generation, find parent.
            if not r.is_offspring:
                continue
            et = r.evolved_type or ""
            if et not in SLOTS:  # only single-slot
                continue
            slot = et
            parent_name = r.operator_parent_names.get(slot)
            if not parent_name:
                continue
            parent_rec = by_slot_name.get((slot, parent_name))
            if parent_rec is None or parent_rec is r:
                continue
            if not (np.all(np.isfinite(parent_rec.seeds)) and np.all(np.isfinite(r.seeds))):
                continue
            if parent_rec.seeds_evaluated < N_SEEDS or r.seeds_evaluated < N_SEEDS:
                continue
            pairs.append(
                PairRecord(
                    run_dir=run_dir,
                    parent_uid=parent_rec.bundle_uid,
                    offspring_uid=r.bundle_uid,
                    slot=slot,
                    mode=r.operator_modes.get(slot),
                    parent_seeds=parent_rec.seeds,
                    offspring_seeds=r.seeds,
                )
            )
    return pairs


def _fisher_z(r: float) -> float:
    r = max(min(r, 0.999999), -0.999999)
    return 0.5 * math.log((1 + r) / (1 - r))


def _inv_fisher(z: float) -> float:
    e = math.exp(2 * z)
    return (e - 1) / (e + 1)


@dataclass
class Exp2Subgroup:
    label: str
    n_pairs: int
    correlations: np.ndarray  # raw r per pair
    mean_r: float
    mean_r_ci: Tuple[float, float]  # 95% CI via Fisher z
    fisher_t: float
    fisher_p: float
    var_ratio: np.ndarray  # V_paired / V_unpaired per pair (drop NaNs)
    mean_var_ratio: float


def experiment_2(pairs: List[PairRecord], label_prefix: str) -> Dict[str, Exp2Subgroup]:
    """Compute per-pair correlation + variance ratio, broken down by mode and slot."""
    subgroups: Dict[str, List[PairRecord]] = {"all": list(pairs)}
    by_mode: Dict[str, List[PairRecord]] = defaultdict(list)
    by_slot: Dict[str, List[PairRecord]] = defaultdict(list)
    for p in pairs:
        if p.mode:
            by_mode[p.mode].append(p)
        by_slot[p.slot].append(p)
    for k, v in by_mode.items():
        subgroups[f"mode={k}"] = v
    for k, v in by_slot.items():
        subgroups[f"slot={k}"] = v

    out: Dict[str, Exp2Subgroup] = {}
    for label, lst in subgroups.items():
        if len(lst) < 3:
            continue
        rs, vratios = [], []
        for p in lst:
            ps, os_ = p.parent_seeds, p.offspring_seeds
            # Pearson r — fall back to 0 if either side is constant.
            if np.std(ps) < 1e-12 or np.std(os_) < 1e-12:
                r = 0.0
            else:
                r = float(np.corrcoef(ps, os_)[0, 1])
            rs.append(r)
            vp = float(np.var(os_ - ps, ddof=1))
            vu = float(np.var(ps, ddof=1) + np.var(os_, ddof=1))
            if vu > 0:
                vratios.append(vp / vu)
        rs = np.array(rs)
        vratios = np.array(vratios)

        # Aggregate r via Fisher-z.
        zs = np.array([_fisher_z(r) for r in rs])
        z_mean = float(np.mean(zs))
        z_se = float(np.std(zs, ddof=1) / math.sqrt(len(zs))) if len(zs) > 1 else float("nan")
        t_stat = z_mean / z_se if z_se and math.isfinite(z_se) and z_se > 0 else float("nan")
        # df = n - 1 for one-sample t on transformed values.
        p_val = (
            float(2.0 * stats.t.sf(abs(t_stat), len(zs) - 1))
            if math.isfinite(t_stat)
            else float("nan")
        )
        ci_lo = _inv_fisher(z_mean - 1.96 * z_se) if math.isfinite(z_se) else float("nan")
        ci_hi = _inv_fisher(z_mean + 1.96 * z_se) if math.isfinite(z_se) else float("nan")

        out[f"{label_prefix}::{label}"] = Exp2Subgroup(
            label=label,
            n_pairs=len(lst),
            correlations=rs,
            mean_r=float(_inv_fisher(z_mean)),
            mean_r_ci=(float(ci_lo), float(ci_hi)),
            fisher_t=float(t_stat),
            fisher_p=float(p_val),
            var_ratio=vratios,
            mean_var_ratio=float(np.mean(vratios)) if len(vratios) else float("nan"),
        )
    return out


def plot_exp2(sub_results: Dict[str, Exp2Subgroup], label: str, path: Path) -> None:
    # Pick a stable ordering: 'all' first, then modes, then slots.
    def sort_key(k: str):
        s = k.split("::", 1)[-1]
        prefix = (0 if s == "all" else 1 if s.startswith("mode=") else 2)
        return (prefix, s)

    items = sorted(sub_results.items(), key=lambda kv: sort_key(kv[0]))
    n = len(items)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(max(8, n * 0.7), 8))
    labels = [k.split("::", 1)[-1] for k, _ in items]
    rs_means = [v.mean_r for _, v in items]
    rs_ci_lo = [v.mean_r_ci[0] for _, v in items]
    rs_ci_hi = [v.mean_r_ci[1] for _, v in items]
    rs_err = [
        [m - lo for m, lo in zip(rs_means, rs_ci_lo)],
        [hi - m for m, hi in zip(rs_means, rs_ci_hi)],
    ]
    ax0 = axes[0]
    ax0.errorbar(range(n), rs_means, yerr=rs_err, fmt="o", capsize=4, color="#3b6ea5")
    ax0.axhline(0, color="black", linewidth=0.5)
    ax0.set_xticks(range(n))
    ax0.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax0.set_ylabel("Mean Pearson r (parent vs offspring per-seed scores)")
    ax0.set_title(
        f"Experiment 2 ({label}) — mean correlation with 95% CI (Fisher-z)\n"
        f"each point is one subgroup; n_pairs shown below"
    )
    for x, (k, v) in enumerate(items):
        ax0.annotate(f"n={v.n_pairs}\np={v.fisher_p:.2g}", (x, v.mean_r),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7)

    ax1 = axes[1]
    var_means = [v.mean_var_ratio for _, v in items]
    ax1.bar(range(n), var_means, color="#7a3a8c")
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.6, label="no benefit from pairing")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("V_paired / V_unpaired (mean per pair)")
    ax1.set_title("Variance reduction from seed-paired comparison (lower = better)")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def split_by_era(records: List[BundleRecord]) -> Dict[str, List[BundleRecord]]:
    return {
        "pre_apr29": [r for r in records if r.run_start < BOUNDARY],
        "post_apr29": [r for r in records if r.run_start >= BOUNDARY],
    }


def write_summary(path: Path, exp1: Dict[str, Optional[Exp1Result]], exp2: Dict[str, Dict[str, Exp2Subgroup]]) -> None:
    lines: List[str] = []
    lines.append("Seed correlation analysis — summary\n")
    lines.append("=" * 60 + "\n\n")

    for era, res in exp1.items():
        lines.append(f"## Experiment 1 — era={era}\n")
        if res is None:
            lines.append("  (insufficient data)\n\n")
            continue
        lines.append(f"  n_bundles (deduped, all 10 seeds, non-constant): {res.n_bundles}\n")
        lines.append(
            f"  Omnibus ANOVA: F({res.df_seed}, {res.df_resid}) = {res.f_stat:.3f}, p = {res.f_p_value:.3g}\n"
        )
        lines.append(
            f"  Friedman test: chi^2({res.df_seed}) = {res.friedman_stat:.3f}, p = {res.friedman_p_value:.3g}\n"
        )
        lines.append(f"  Residual MS = {res.ms_resid:.4g}, sigma_hat = {math.sqrt(res.ms_resid):.4g}\n")
        lines.append("  Per-seed bonuses (estimate +/- 95% CI, raw p, Holm-adj p):\n")
        for j in range(N_SEEDS):
            ci = 1.96 * res.bonus_se[j]
            lines.append(
                f"    seed {j}: {res.bonuses[j]:+.4f} +/- {ci:.4f}  "
                f"t={res.t_stats[j]:+.2f}  p_raw={res.p_values[j]:.3g}  p_holm={res.p_values_holm[j]:.3g}\n"
            )
        lines.append("\n")

    for era, sub_results in exp2.items():
        lines.append(f"## Experiment 2 — era={era}\n")
        if not sub_results:
            lines.append("  (no pairs found)\n\n")
            continue
        for key, sub in sorted(sub_results.items()):
            lines.append(
                f"  [{sub.label}] n_pairs={sub.n_pairs}  "
                f"mean r = {sub.mean_r:+.3f}  CI=({sub.mean_r_ci[0]:+.3f},{sub.mean_r_ci[1]:+.3f})  "
                f"p={sub.fisher_p:.3g}  V_paired/V_unpaired = {sub.mean_var_ratio:.3f}\n"
            )
        lines.append("\n")

    path.write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading all run_data.json files...", file=sys.stderr)
    all_records = load_all_runs()
    print(f"  loaded {len(all_records)} bundle records", file=sys.stderr)

    by_era = split_by_era(all_records)
    for era, recs in by_era.items():
        print(f"  era={era}: {len(recs)} records", file=sys.stderr)

    exp1_results: Dict[str, Optional[Exp1Result]] = {}
    exp2_results: Dict[str, Dict[str, Exp2Subgroup]] = {}

    for era, recs in by_era.items():
        print(f"\n[era={era}] running experiment 1...", file=sys.stderr)
        res1 = experiment_1(recs, label=era)
        exp1_results[era] = res1
        if res1 is not None:
            plot_exp1(res1, args.out_dir / f"exp1_seed_bonuses_{era}.png")
            print(
                f"  exp1: n_bundles={res1.n_bundles}, "
                f"F p={res1.f_p_value:.3g}, Friedman p={res1.friedman_p_value:.3g}",
                file=sys.stderr,
            )

        print(f"[era={era}] running experiment 2...", file=sys.stderr)
        pairs = find_pairs(recs)
        sub = experiment_2(pairs, label_prefix=era)
        exp2_results[era] = sub
        if sub:
            plot_exp2(sub, label=era, path=args.out_dir / f"exp2_pair_corr_{era}.png")
        print(f"  exp2: {len(pairs)} parent-offspring pairs", file=sys.stderr)

    write_summary(args.out_dir / "summary.txt", exp1_results, exp2_results)
    print(f"\nWrote outputs to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
