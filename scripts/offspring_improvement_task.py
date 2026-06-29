"""Stacked old-vs-new offspring-EI comparison for a run.

Top panel:    old form (aggregate scores). Two offspring-EI lines:
              closed-form TS marginals, and topk-tourney(10, 2).
Bottom panel: new form (per-task Beta posteriors). Two offspring-EI lines:
              task-TS marginals, and task-topk-tourney(10, 2).

Both panels share the same scatter of realized per-offspring improvements
((O.final - D.final) / pop_size) and the same raw realized average line. The
realized series doesn't change with the selection rule — what changes is the
predicted EI.

Usage: python scripts/offspring_improvement_task.py [job]  (default 666286)
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monte_carlo_test import load_arms
from offspring_mc import (
    offspring_expected_improvement,
    thompson_sampling_closed_form_batch_selection_fn,
)
from monte_carlo import topk_tourney_batch_selection_fn
from offspring_mc_task import (
    extract_task_counts,
    pool_task_counts,
    offspring_expected_improvement_task,
)
from offspring_improvement import (
    analyze,
    load_mc_summary,
    offspring_empirical_for_gen,
    fit_ei_curve,
    smoothed_mei,
    indifference_B,
    MARGIN,
)


def load_task_summary(job):
    """Load plots/<job>/task_summary.json from monte_carlo_sweep_task."""
    p = Path("plots") / job / "task_summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    out = {}
    for r in s.get("records", []):
        out[int(r["gen"])] = {"curve": r["curve"]}
    return out


def offspring_empirical_task_for_gen(rows_by_gen_name, generations,
                                     gen, K=3, n_initial_evals=3):
    """For each past offspring with initial-eval state, return its initial-eval
    per-task (s, f) arrays. Returns (S_list [E, T], F_list [E, T])."""
    out_s, out_f = [], []
    for g in range(max(0, gen - K + 1), gen + 1):
        for o in generations[g].get("offspring", []):
            if o.get("seeds_evaluated") != n_initial_evals:
                continue
            s, f = extract_task_counts(o)
            out_s.append(s)
            out_f.append(f)
    if not out_s:
        return None, None
    return np.stack(out_s, axis=0), np.stack(out_f, axis=0)


def post_survival_pool(data, gen):
    """Bundles that survived into pop[gen] (i.e. pre-offspring-replacement)."""
    return list(data["generations"][gen]["population"])


def compute_old_offspring_ei(data, rows, mc_summary, rule, K=3, n_initial_evals=3):
    """rule in {'ts', 'topk_tourney'}. Uses offspring_mc.offspring_expected_improvement."""
    if rule == "ts":
        sel_fn = thompson_sampling_closed_form_batch_selection_fn()
    elif rule == "topk_tourney":
        sel_fn = topk_tourney_batch_selection_fn(topk=10, n=2)
    else:
        raise ValueError(rule)

    out = {}
    rng = np.random.default_rng(0)
    for gen, rec in mc_summary.items():
        sigma = rec["sigma"]
        empirical = offspring_empirical_for_gen(rows, gen, K, n_initial_evals)
        if empirical.size == 0:
            continue
        mu, N, _labels, _bundles = load_arms(data, gen)
        res = offspring_expected_improvement(
            pop_mu=mu, pop_N=N,
            offspring_empirical=empirical,
            sigma=sigma, n_initial_evals=n_initial_evals,
            batch_selection_fn=sel_fn,
            M_total=None, rng=rng,
        )
        if res is not None:
            out[gen] = float(res["improvement"])
    return out


def compute_new_offspring_ei(data, rule, K=3, n_initial_evals=3, M=4000):
    """rule in {'task_ts', 'task_topk_tourney'}."""
    gens = data["generations"]
    out = {}
    rng = np.random.default_rng(0)
    for gen in range(len(gens)):
        # Pool: pop + this gen's offspring (matches load_arms scope).
        pop = list(gens[gen]["population"])
        offspring = list(gens[gen]["offspring"])
        bundles = pop + offspring
        if not bundles:
            continue
        pop_S, pop_F = pool_task_counts(bundles)
        off_S, off_F = offspring_empirical_task_for_gen(None, gens, gen, K, n_initial_evals)
        if off_S is None:
            continue
        res = offspring_expected_improvement_task(
            pop_S, pop_F, off_S, off_F, M=M, rng=rng,
            selection_rule=rule, topk=10, n=2,
        )
        if res is not None:
            out[gen] = float(res["improvement"])
    return out


def compute_b_star_records(mc_curves, target_ei_by_gen, gens):
    """Per-gen B*(gen) records: (gen, status, B_star)."""
    records = []
    for g in gens:
        target = target_ei_by_gen.get(g)
        if g not in mc_curves or target is None or not np.isfinite(target):
            records.append((g, None, None))
            continue
        popt = fit_ei_curve(mc_curves[g])
        status, B = indifference_B(popt, float(target))
        records.append((g, status, B))
    return records


def plot_stacked(rows, pop_size, job, out_dir,
                 old_ts, old_tourney, new_ts, new_tourney,
                 old_curves, new_curves,
                 mei_B_values=(10, 40)):
    by_gen = defaultdict(list)
    for r in rows:
        by_gen[r["gen"]].append(r)
    gens = sorted(by_gen)
    gens_arr = np.array(gens)
    avgs = np.array([np.mean([r["improvement"] for r in by_gen[g]]) for g in gens])

    fig = plt.figure(figsize=(14, 20))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 1.6, 2.4, 1.6],
                          left=0.07, right=0.98, top=0.97, bottom=0.04,
                          hspace=0.22)
    ax_old_ei = fig.add_subplot(gs[0])
    ax_old_b  = fig.add_subplot(gs[1], sharex=ax_old_ei)
    ax_new_ei = fig.add_subplot(gs[2], sharex=ax_old_ei)
    ax_new_b  = fig.add_subplot(gs[3], sharex=ax_old_ei)

    def _scatter(ax):
        for g in gens:
            for r in by_gen[g]:
                color = "C3" if r["entered"] else "0.75"
                alpha = 0.55 if r["entered"] else 0.25
                ax.scatter([g], [r["improvement"]], color=color, alpha=alpha,
                           s=10, zorder=1, edgecolors="none")

    def _line(ax, d, label, color, lw=2.0):
        if not d:
            return
        gs_ = sorted(d)
        ax.plot(gs_, [d[g] for g in gs_], "o-", color=color,
                linewidth=lw, markersize=4, label=label, zorder=4)

    def _mei_overlay(ax, curves):
        if not curves:
            return
        mc_g = sorted(curves)
        cmap = plt.get_cmap("viridis")
        for i, B in enumerate(mei_B_values):
            color = cmap(0.1 + 0.7 * i / max(1, len(mei_B_values) - 1))
            sm_vals = []
            for g in mc_g:
                popt = fit_ei_curve(curves[g])
                sm_vals.append(smoothed_mei(popt, B))
            sm_gs = [g for g, v in zip(mc_g, sm_vals) if v is not None]
            sm_ys = [v for v in sm_vals if v is not None]
            ax.plot(sm_gs, sm_ys, "-", color=color, linewidth=1.0,
                    alpha=0.95, zorder=3,
                    label=f"smoothed MEI @ B={B}, Δ={MARGIN}")

    def _b_panel(ax, records, ylabel_extra="", y_clamp=300.0):
        """y_clamp caps the y-axis so a few B* outliers don't compress the rest.
        Finite B* values above the clamp are still plotted at the clamp with a
        small annotation."""
        finite = [(g, B) for g, st, B in records if st == "finite"]
        dom = [g for g, st, _ in records if st == "offspring-dominates"]
        noimp = [g for g, st, _ in records if st == "no-improvement"]
        if finite:
            gs_, bs_ = zip(*finite)
            bs_arr = np.array(bs_, dtype=float)
            plot_y = np.minimum(bs_arr, y_clamp)
            ax.plot(gs_, plot_y, "o-", color="C0", linewidth=1.6, markersize=5,
                    label=f"finite B*  (n={len(finite)} gens)")
            # Annotate clamped points with their true value.
            for g, B in zip(gs_, bs_):
                if B > y_clamp:
                    ax.annotate(f"{B:.0f}", (g, y_clamp),
                                textcoords="offset points", xytext=(2, -10),
                                fontsize=7, color="C0")
        ymax = y_clamp
        if dom:
            ax.scatter(dom, [0] * len(dom), marker="v", color="C2", s=45,
                       zorder=4,
                       label=f"offspring-dominates (B*≈0, n={len(dom)})")
        if noimp:
            ax.scatter(noimp, [ymax * 0.95] * len(noimp), marker="^",
                       color="C3", s=45, zorder=4,
                       label=f"target ≤ 0 (no crossover, n={len(noimp)})")
        ax.set_ylim(-y_clamp * 0.05, y_clamp * 1.02)
        ax.axhline(0, color="k", linewidth=0.7)
        ax.set_ylabel(f"B*  (reeval budget where\nMEI(B*; Δ={MARGIN}) = offspring EI)"
                      + ylabel_extra)
        ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper left")

    # === Row 1: OLD offspring EI ===
    ax = ax_old_ei
    _scatter(ax)
    ax.plot(gens_arr, avgs, "-", color="0.5", alpha=0.6, linewidth=0.8,
            label="realized raw avg")
    _line(ax, old_ts,      "old offspring EI — TS (closed-form α)",         "C0")
    _line(ax, old_tourney, "old offspring EI — topk-tourney(10, 2)",        "C2")
    _mei_overlay(ax, old_curves)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_ylabel("per-offspring Δ(mean pop μ)\n"
                  f"(O.final − D.final) / pop_size  [pop={pop_size}]")
    ax.set_title(f"{job} — OLD form (aggregate score per arm, "
                 "MEI from existing monte_carlo_sweep summary)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # === Row 2: OLD B* (vs topk-tourney offspring EI, matching existing pipeline) ===
    old_b_records = compute_b_star_records(old_curves or {}, old_tourney or {}, gens)
    _b_panel(ax_old_b, old_b_records,
             ylabel_extra="\n(target = old topk-tourney EI)")
    ax_old_b.set_title("OLD B*  —  how many reevals = one offspring "
                       "(topk-tourney(10, 2) parent select)")

    # === Row 3: NEW offspring EI ===
    ax = ax_new_ei
    _scatter(ax)
    ax.plot(gens_arr, avgs, "-", color="0.5", alpha=0.6, linewidth=0.8,
            label="realized raw avg")
    _line(ax, new_ts,      "new offspring EI — task-TS",                    "C0")
    _line(ax, new_tourney, "new offspring EI — task-topk-tourney(10, 2)",   "C2")
    _mei_overlay(ax, new_curves)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_ylabel("per-offspring Δ(mean pop μ)\n"
                  f"(O.final − D.final) / pop_size  [pop={pop_size}]")
    ax.set_title(f"{job} — NEW form (per-task Beta posteriors; MEI from "
                 "task-aware sweep with task-topk-tourney parent select)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # === Row 4: NEW B* (vs task-topk-tourney offspring EI) ===
    new_b_records = compute_b_star_records(new_curves or {}, new_tourney or {}, gens)
    _b_panel(ax_new_b, new_b_records,
             ylabel_extra="\n(target = task-topk-tourney EI;\n"
             "x-unit = task-reevals, 1 reeval = T seed-evals)")
    ax_new_b.set_title("NEW B*  —  how many task-reevals = one offspring "
                       "(task-topk-tourney(10, 2) parent select)")
    ax_new_b.set_xlabel("generation")

    out = out_dir / f"{job}_offspring_improvement_task_vs_old.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    return out, old_b_records, new_b_records


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "666286"
    print(f"loading runs/{job}/run_data.json ...")
    data = json.loads((Path("runs") / job / "run_data.json").read_text())
    rows, pop_size = analyze(data)
    mc_summary = load_mc_summary(job)
    if mc_summary is None:
        print(f"ERROR: plots/{job}/summary.json missing — run monte_carlo_sweep first.")
        sys.exit(1)

    t = time.time()
    print("computing old offspring EI (TS, closed-form α) ...")
    old_ts = compute_old_offspring_ei(data, rows, mc_summary, rule="ts")
    print(f"  done in {time.time() - t:.1f}s; {len(old_ts)} gens")

    t = time.time()
    print("computing old offspring EI (topk-tourney 10,2) ...")
    old_tourney = compute_old_offspring_ei(data, rows, mc_summary, rule="topk_tourney")
    print(f"  done in {time.time() - t:.1f}s; {len(old_tourney)} gens")

    t = time.time()
    print("computing new offspring EI (task-TS) ...")
    new_ts = compute_new_offspring_ei(data, rule="task_ts", M=4000)
    print(f"  done in {time.time() - t:.1f}s; {len(new_ts)} gens")

    t = time.time()
    print("computing new offspring EI (task-topk-tourney 10,2) ...")
    new_tourney = compute_new_offspring_ei(data, rule="task_topk_tourney", M=4000)
    print(f"  done in {time.time() - t:.1f}s; {len(new_tourney)} gens")

    # Quick print summary.
    print(f"\n{'gen':>4}  {'old-TS':>10}  {'old-tourney':>11}  "
          f"{'new-TS':>10}  {'new-tourney':>11}")
    all_g = sorted(set(old_ts) | set(old_tourney) | set(new_ts) | set(new_tourney))
    for g in all_g[::4]:
        vals = [old_ts.get(g), old_tourney.get(g), new_ts.get(g), new_tourney.get(g)]
        row = "  ".join(f"{'      —':>10}" if v is None
                        else f"{v:+10.5f}" for v in vals)
        print(f"{g:>4}  {row}")

    old_curves = {g: v["curve"] for g, v in mc_summary.items()}
    task_summary = load_task_summary(job)
    new_curves = ({g: v["curve"] for g, v in task_summary.items()}
                  if task_summary else None)
    if new_curves is None:
        print(f"WARNING: plots/{job}/task_summary.json not found. "
              "Run scripts/monte_carlo_sweep_task.py first to enable "
              "the new MEI overlay + B* panel.")

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    out, old_b_records, new_b_records = plot_stacked(
        rows, pop_size, job, out_dir,
        old_ts, old_tourney, new_ts, new_tourney,
        old_curves, new_curves,
    )

    # Compact B* summary so user can scan numbers.
    print("\nold B* (target = topk-tourney offspring EI):")
    for g, st, B in old_b_records[::4]:
        B_str = f"{B:.1f}" if B is not None else "—"
        print(f"  gen {g:>2}  {st or 'fit-failed':>22}  {B_str:>8}")
    if new_b_records:
        print("\nnew B* (target = task-topk-tourney offspring EI):")
        for g, st, B in new_b_records[::4]:
            B_str = f"{B:.1f}" if B is not None else "—"
            print(f"  gen {g:>2}  {st or 'fit-failed':>22}  {B_str:>8}")


if __name__ == "__main__":
    main()
