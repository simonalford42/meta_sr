"""Per-offspring contribution to mean-population score, tracked over time.

For each offspring O born in generation G:
  - find its LAST observed state across all generations (final_score, final_N)
  - if O entered pop[G] (replaced someone), find its paired dropped member D
    (pair entering offspring with dropped pop members in score-sorted order,
    matching select_survivors top-k semantics)
  - improvement(O) = (O.final_score − D.final_score) / pop_size
  - if O did not enter, improvement = 0

Per-gen avg = mean of improvement(O) over all N offspring that gen.

Plot mirrors 982249_offspring_inclusion.png style but on the improvement
axis: gray dots for individual offspring, raw + K=3 + K=10 trailing averages.
Optionally overlays monte_carlo EI[B=K] from a prior sweep summary.json so
"new offspring" and "re-evaluation" improvements can be eyeballed together.

Usage: python offspring_improvement.py [job]  (default 666286)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monte_carlo_test import load_arms_archive as load_arms
from offspring_mc import offspring_expected_improvement


# Per-seed cost of one offspring (used as the margin for MEI computation).
# Set at runtime from data["config"]["n_runs"] via configure_margin_from_data().
# Default 3 matches the original 666286 racing run.
MARGIN = 3


def configure_margin_from_data(data):
    """Read offspring seed cost (n_runs) from the run config and set MARGIN.

    Returns the resolved value. Falls back to the current MARGIN if absent.
    Modules that imported MARGIN by name won't see the rebind — use
    `offspring_improvement.MARGIN` from those callers.
    """
    global MARGIN
    n_runs = (data.get("config", {}) or {}).get("n_runs")
    if n_runs:
        MARGIN = int(n_runs)
    return MARGIN


def _smooth_ei(B, a1, tau1, a2, tau2):
    """Two-component saturating exponential:
        EI(B) = a₁·(1 − e^(−B/τ₁)) + a₂·(1 − e^(−B/τ₂))

    Monotonically increasing, concave, passes through (0, 0). Captures the
    fast initial-disambiguation phase (small τ) plus the slow tail of
    refining lower-confidence arms (large τ).
    """
    return (a1 * (1.0 - np.exp(-B / tau1))
            + a2 * (1.0 - np.exp(-B / tau2)))


# Single-exponential kept for backward compatibility (and as a fallback).
def _saturating(B, a, tau):
    return a * (1.0 - np.exp(-B / tau))


def fit_ei_curve(ei):
    """Fit the two-component saturating exponential to the EI curve.

    ei[B] is EI at budget B with ei[0] = 0. Returns popt tuple
    (a1, τ1, a2, τ2) or None on failure. Falls back to a single
    exponential (returned as (a1, τ1, 0.0, 1.0)) if the 4-parameter fit
    fails to converge.
    """
    xs = np.arange(0, len(ei), dtype=float)
    ys = np.asarray(ei, dtype=float)
    if ys.size == 0 or not np.all(np.isfinite(ys)):
        return None
    ymax = max(float(ys.max()), 1e-6)
    p0 = [ymax * 0.5, 5.0, ymax * 0.5, 60.0]
    try:
        popt, _ = curve_fit(_smooth_ei, xs, ys, p0=p0,
                            bounds=([0.0, 0.3, 0.0, 0.3],
                                    [np.inf, 200.0, np.inf, 2000.0]),
                            maxfev=10000)
    except Exception:
        return None
    return tuple(popt)


# margin defaults to None and resolves to the module-level MARGIN at call time,
# so a runtime rebind (configure_margin_from_data / oi.MARGIN = n_runs) is
# respected. A literal `margin=MARGIN` default would capture 3 at import time.
def raw_mei(ei, B, margin=None):
    """EI[B+margin] − EI[B] from the raw curve (absolute indexing).
    None if B+margin exceeds available budget."""
    if margin is None:
        margin = MARGIN
    if B + margin >= len(ei):
        return None
    return ei[B + margin] - ei[B]


def smoothed_mei(popt, B, margin=None):
    """Marginal smoothed EI at budget B for a margin of `margin` extra evals.

    For the two-component model MEI(B; Δ) reduces to:
        a₁·(1 − e^(−Δ/τ₁))·e^(−B/τ₁) + a₂·(1 − e^(−Δ/τ₂))·e^(−B/τ₂)
    which is monotonically decreasing in B → brentq for the inverse.
    """
    if margin is None:
        margin = MARGIN
    if popt is None:
        return None
    return _smooth_ei(B + margin, *popt) - _smooth_ei(B, *popt)


def indifference_B(popt, target, margin=None, B_upper=10000.0):
    """Numerical inverse: smoothed_MEI(B; margin) = target.

    Returns (status, B_star):
        ('finite', value) when a finite non-negative B solves it
        ('offspring-dominates', 0.0) when target ≥ MEI(0)
        ('no-improvement', None) when target ≤ 0
        (None, None) on fit failure or numerical issue.
    """
    if margin is None:
        margin = MARGIN
    if popt is None or target is None or not np.isfinite(target):
        return None, None
    if target <= 0:
        return "no-improvement", None
    mei0 = smoothed_mei(popt, 0.0, margin)
    if target >= mei0:
        return "offspring-dominates", 0.0
    # MEI is monotonically decreasing, so a unique root exists in [0, B_upper]
    # as long as MEI(B_upper) < target. Two-exp decays to 0 ⇒ this holds.
    try:
        B_star = brentq(lambda B: smoothed_mei(popt, B, margin) - target,
                        0.0, B_upper)
        return "finite", float(B_star)
    except Exception:
        return None, None


def bundle_key(m):
    # Identity by operator CODE, not name: distinct LLM outputs can collide on
    # the auto-generated name. Matches monte_carlo_test._bundle_key.
    ops = m["operators"]
    return tuple((s, ops[s].get("code")) for s in sorted(ops))


def build_history(data):
    """Map bundle_key -> sorted list of (gen, score, n_evals) sightings."""
    hist = defaultdict(list)
    for n, g in enumerate(data["generations"]):
        for m in list(g["population"]) + list(g["offspring"]):
            hist[bundle_key(m)].append((n, m["score"], m["seeds_evaluated"]))
    for k in hist:
        # Sort by gen, tiebreak by N so the latest most-evaluated state wins.
        hist[k].sort(key=lambda e: (e[0], e[2]))
    return hist


def final_state(hist, key):
    es = hist.get(key, [])
    return es[-1] if es else None


def analyze(data):
    gens = data["generations"]
    pop_size = data["config"]["population_size"]
    hist = build_history(data)

    rows = []
    for n in range(1, len(gens)):
        prev = gens[n - 1]
        cur = gens[n]
        offspring = cur["offspring"]
        prev_keys = {bundle_key(m): m for m in prev["population"]}
        new_keys = {bundle_key(m): m for m in cur["population"]}

        # Dropped members of prev_pop = in prev, not in new.
        dropped = [m for k, m in prev_keys.items() if k not in new_keys]
        # Entering offspring = in offspring list, in new_pop, NOT in prev_pop.
        entering = [o for o in offspring
                    if bundle_key(o) in new_keys
                    and bundle_key(o) not in prev_keys]

        # Pair entering with dropped by initial-score rank (top-k replacement
        # semantics: lowest dropped pop slot is filled by the lowest entering
        # offspring, etc.). Both sorted ascending so the lowest entering pairs
        # with the lowest dropped.
        dropped_sorted = sorted(dropped, key=lambda m: m["score"])
        entering_sorted = sorted(entering, key=lambda o: o["score"])
        pair = {bundle_key(o): d for o, d in zip(entering_sorted, dropped_sorted)}

        for o in offspring:
            k = bundle_key(o)
            entered = k in pair
            if not entered:
                rows.append({
                    "gen": n, "name": k, "entered": False,
                    "improvement": 0.0,
                    "o_init": o["score"], "o_init_N": o["seeds_evaluated"],
                    "o_final": None, "o_final_N": None,
                    "d_final": None, "d_final_N": None,
                })
                continue
            d = pair[k]
            fin = final_state(hist, k)
            o_final, o_final_N = (fin[1], fin[2]) if fin else (o["score"],
                                                                o["seeds_evaluated"])
            d_final = d["score"]
            d_final_N = d["seeds_evaluated"]
            improvement = (o_final - d_final) / pop_size
            rows.append({
                "gen": n, "name": k, "entered": True,
                "improvement": improvement,
                "o_init": o["score"], "o_init_N": o["seeds_evaluated"],
                "o_final": o_final, "o_final_N": o_final_N,
                "d_final": d_final, "d_final_N": d_final_N,
            })
    return rows, pop_size


def trailing_avg(xs, k):
    xs = np.asarray(xs, dtype=float)
    out = np.full_like(xs, np.nan)
    for i in range(len(xs)):
        out[i] = xs[max(0, i - k + 1): i + 1].mean()
    return out


def load_mc_summary(job):
    """Per-gen dict of {curve, sigma, ...} from a prior monte_carlo_sweep run."""
    p = Path("plots") / job / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    out = {}
    for r in s.get("records", []):
        out[int(r["gen"])] = {
            "curve": r["curve"],
            "sigma": r["sigma"],
        }
    return out


def load_mc_curve(job):
    """Per-gen EI curves only (back-compat shim)."""
    s = load_mc_summary(job)
    return None if s is None else {g: v["curve"] for g, v in s.items()}


def offspring_empirical_for_gen(rows, gen, K=3, n_initial_evals=None):
    """Collect offspring posterior-mean values from gens [gen-K+1 .. gen].

    Only includes offspring whose initial-eval state had N == n_initial_evals
    (so each empirical value has the same posterior width σ/√N when used
    downstream). Defaults n_initial_evals to MARGIN.
    """
    if n_initial_evals is None:
        n_initial_evals = MARGIN
    vals = []
    for r in rows:
        g = r["gen"]
        if gen - K + 1 <= g <= gen and r.get("o_init_N") == n_initial_evals:
            vals.append(r["o_init"])
    return np.array(vals, dtype=float)


def compute_per_offspring_ei(
    data, rows, mc_summary, n_initial_evals=None,
    batch_selection_fn=None, seed=0,
):
    if n_initial_evals is None:
        n_initial_evals = MARGIN
    """Per-offspring EI: for each offspring O born in gen g with N=n_initial_evals,
    compute (parent_dist(pool[g] ∪ {O}) · μ_ext) − baseline(pool[g]).

    Returns dict gen -> list of per-offspring EI values (one per offspring born
    in that gen that had the required initial-eval count).
    """
    rng = np.random.default_rng(seed)
    out = defaultdict(list)
    for gen in sorted(mc_summary):
        sigma = mc_summary[gen]["sigma"]
        vs = [r["o_init"] for r in rows
              if r["gen"] == gen and r.get("o_init_N") == n_initial_evals]
        if not vs:
            continue
        mu, N, _labels, _bundles = load_arms(data, gen)
        res = offspring_expected_improvement(
            pop_mu=mu, pop_N=N,
            offspring_empirical=np.asarray(vs, dtype=float),
            sigma=sigma, n_initial_evals=n_initial_evals,
            batch_selection_fn=batch_selection_fn,
            M_total=None, rng=rng,
        )
        if res is None:
            continue
        baseline = res["baseline"]
        out[gen] = [float(f - baseline) for f in res["per_value_fits"]]
    return dict(out)


def compute_offspring_ei_per_gen(
    data, rows, mc_summary,
    K=3, n_initial_evals=None,
    batch_selection_fn=None, M_total=None, seed=0,
):
    if n_initial_evals is None:
        n_initial_evals = MARGIN
    """For each gen in mc_summary, run offspring_expected_improvement.

    Pool = 20 arms from load_arms(data, gen) (post-survival pop + this gen's
    offspring) so the baseline matches monte_carlo_sweep's MEI baseline.
    Empirical distribution = posterior means of past offspring from gens
    [gen-K+1, gen]. Default batch_selection_fn is closed-form TS marginals
    (analytic, no MC). M_total=None ⇒ analytic offspring EI.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for gen in sorted(mc_summary):
        sigma = mc_summary[gen]["sigma"]
        empirical = offspring_empirical_for_gen(rows, gen, K, n_initial_evals)
        if empirical.size == 0:
            continue
        mu, N, _labels, _bundles = load_arms(data, gen)
        out[gen] = offspring_expected_improvement(
            pop_mu=mu, pop_N=N,
            offspring_empirical=empirical,
            sigma=sigma, n_initial_evals=n_initial_evals,
            batch_selection_fn=batch_selection_fn,
            M_total=M_total, rng=rng,
        )
    return out


def plot(rows, pop_size, job, out_dir, mc_curves=None,
         mei_B=40, mc_offspring_ei=None, per_offspring_ei=None):
    by_gen = defaultdict(list)
    for r in rows:
        by_gen[r["gen"]].append(r)
    gens = sorted(by_gen)

    # Per-gen offspring EI from the new principled MC framework (offspring_mc).
    # Each value is in the same units as the smoothed MEI: improvement in
    # E[truth[chosen-parent]] from adding one new offspring (MARGIN evals).
    offspring_ei_by_gen = {}
    if mc_offspring_ei is not None:
        for g, res in mc_offspring_ei.items():
            if res is not None:
                offspring_ei_by_gen[g] = float(res["improvement"])

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 3.0],
                          left=0.07, right=0.98, top=0.96, bottom=0.05,
                          hspace=0.18)
    ax = fig.add_subplot(gs[0])
    axes = [ax, fig.add_subplot(gs[1], sharex=ax)]

    # Per-offspring EI dots, plus a count of zeros (offspring that didn't change
    # the parent distribution, i.e. failed top-k truncation).
    zero_tol = 1e-12
    total_zero = 0
    total_dots = 0
    if per_offspring_ei:
        for g, vals in per_offspring_ei.items():
            for v in vals:
                ax.scatter([g], [v], color="C3", alpha=0.55, s=14,
                           zorder=2, edgecolors="none")
            n_zero = sum(1 for v in vals if abs(v) <= zero_tol)
            total_zero += n_zero
            total_dots += len(vals)
            if n_zero:
                ax.annotate(f"{n_zero}", (g, 0),
                            textcoords="offset points", xytext=(0, -10),
                            ha="center", fontsize=7, color="0.4")

    # MC-based offspring expected improvement — directly comparable to MEI.
    if offspring_ei_by_gen:
        ei_gs = sorted(offspring_ei_by_gen)
        ei_vs = [offspring_ei_by_gen[g] for g in ei_gs]
        ax.plot(ei_gs, ei_vs, "o-", color="C0", linewidth=2.2, markersize=5,
                label=f"offspring EI  (Δ={MARGIN}, topk-tourney parent select)",
                zorder=4)

    # Overlay: smoothed marginal EI at single budget B = mei_B.
    if mc_curves:
        mc_g = sorted(mc_curves)
        sm_vals = []
        for g in mc_g:
            popt = fit_ei_curve(mc_curves[g])
            sm_vals.append(smoothed_mei(popt, mei_B))
        sm_gs = [g for g, v in zip(mc_g, sm_vals) if v is not None]
        sm_ys = [v for v in sm_vals if v is not None]
        ax.plot(sm_gs, sm_ys, "-", color="C2", linewidth=1.6,
                alpha=0.95, zorder=3,
                label=f"smoothed MEI @ B={mei_B}, Δ={MARGIN}")

    # Cap y_max so red-dot outliers don't compress the EI/MEI comparison.
    line_max = 0.0
    if offspring_ei_by_gen:
        line_max = max(line_max, max(offspring_ei_by_gen.values()))
    if mc_curves and sm_ys:
        line_max = max(line_max, max(sm_ys))
    if line_max > 0:
        ax.set_ylim(top=2.0 * line_max)

    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_xlabel("generation")
    ax.set_ylabel("expected improvement (Δ E[truth of chosen parent])")
    zero_note = (f"  ({total_zero}/{total_dots} offspring have EI≈0, "
                 f"i.e. fail topk-truncation)" if total_dots else "")
    ax.set_title(f"{job} — per-offspring EI vs smoothed MEI @ B={mei_B} "
                 f"(Δ={MARGIN} evals){zero_note}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom panel: B* per gen where smoothed MEI(B*; Δ=MARGIN) = MC offspring EI.
    ax_b = axes[1]
    if mc_curves and offspring_ei_by_gen:
        b_star_records = []  # (gen, status, B_star) per gen
        for g in gens:
            target = offspring_ei_by_gen.get(g)
            if g not in mc_curves or target is None or not np.isfinite(target):
                b_star_records.append((g, None, None))
                continue
            popt = fit_ei_curve(mc_curves[g])
            status, B = indifference_B(popt, float(target))
            b_star_records.append((g, status, B))

        finite_g = [r[0] for r in b_star_records if r[1] == "finite"]
        finite_b = [r[2] for r in b_star_records if r[1] == "finite"]
        dom_g = [r[0] for r in b_star_records if r[1] == "offspring-dominates"]
        none_g = [r[0] for r in b_star_records if r[1] == "no-improvement"]

        if finite_g:
            ax_b.plot(finite_g, finite_b, "o-", color="C0", linewidth=1.6,
                      markersize=5,
                      label=f"finite B*  (n={len(finite_g)} gens)")
        if dom_g:
            ax_b.scatter(dom_g, [0] * len(dom_g), marker="v", color="C2",
                         s=45, zorder=4,
                         label=f"offspring dominates (B*≈0, n={len(dom_g)})")
        if none_g:
            # Plot at the top of the panel so the marker is visible.
            ymax = max(finite_b) * 1.1 if finite_b else 50.0
            ax_b.scatter(none_g, [ymax] * len(none_g), marker="^",
                         color="C3", s=45, zorder=4,
                         label=f"K=3 avg ≤ 0 (no crossover, n={len(none_g)})")
    ax_b.axhline(0, color="k", linewidth=0.7)
    ax_b.set_xlabel("generation")
    ax_b.set_ylabel(f"B*  (reeval budget where\nMEI(B*; Δ={MARGIN}) = MC offspring EI)")
    ax_b.set_title(f"Indifference point: how many reevals on the existing "
                   f"pool buy the same Δfitness as one new offspring "
                   f"(topk-tourney parent selection, k=10, n=2)")
    ax_b.grid(alpha=0.3); ax_b.legend(fontsize=9, loc="upper left")

    out = out_dir / f"{job}_offspring_improvement.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")

    if mc_curves and offspring_ei_by_gen:
        return b_star_records
    return None


def print_summary(rows, pop_size):
    total = len(rows)
    entered = [r for r in rows if r["entered"]]
    improvs = np.array([r["improvement"] for r in rows])
    improvs_entered = np.array([r["improvement"] for r in entered])
    print(f"Total offspring (gens 1+):    {total}")
    print(f"  entered pop:                {len(entered)} "
          f"({100*len(entered)/max(1,total):.1f}%)")
    print(f"Mean per-offspring Δ(mean pop μ):")
    print(f"  over all offspring:         {improvs.mean():+.5f}")
    print(f"  over entering offspring:    {improvs_entered.mean():+.5f}")
    print(f"  pop_size = {pop_size}, so total Δ(sum pop μ) per gen "
          f"≈ {improvs.mean() * 10 * pop_size:+.4f} (10 offspring × {pop_size})")


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "666286"

    data = json.loads((Path("runs") / job / "run_data.json").read_text())
    margin = configure_margin_from_data(data)
    print(f"Using MARGIN = n_initial_evals = {margin} (from config.n_runs)")
    rows, pop_size = analyze(data)
    mc_summary = load_mc_summary(job)
    mc_curves = ({g: v["curve"] for g, v in mc_summary.items()}
                 if mc_summary else None)
    if mc_curves:
        max_b = max(len(c) for c in mc_curves.values())
        print(f"Loaded MC sweep with EI curves for {len(mc_curves)} gens "
              f"(B_max={max_b}, margin={MARGIN}).")
        # max_b is the array length = B_max + 1; valid B values are 0..max_b-1.
        if max_b - 1 < 40 + MARGIN:
            print(f"  ⚠ raw MEI@B=40 needs B_max≥{40+MARGIN}; re-run sweep "
                  f"with python monte_carlo_sweep.py {job} 5000 50")
    else:
        print("No prior MC sweep summary.json — skipping MEI overlay. "
              "Run monte_carlo_sweep.py first to enable.")

    print(f"=== {job} (pop_size={pop_size}) ===")
    print_summary(rows, pop_size)

    # Print a few example MEI values so we can sanity-check before plotting.
    if mc_curves:
        print(f"\nSmoothed MEI examples (per gen, margin={MARGIN}):")
        for g in sorted(mc_curves)[::6]:
            ei = mc_curves[g]
            popt = fit_ei_curve(ei)
            if popt is None:
                continue
            a1, t1, a2, t2 = popt
            mei10 = smoothed_mei(popt, 10)
            mei25 = smoothed_mei(popt, 25)
            mei40 = smoothed_mei(popt, 40)
            print(f"  gen {g:2d}: a₁={a1:.3f} τ₁={t1:.1f} "
                  f"a₂={a2:.3f} τ₂={t2:.1f}  "
                  f"MEI[10]={mei10:.5f}  MEI[25]={mei25:.5f}  "
                  f"MEI[40]={mei40:.5f}")

    # Compute the MC-based offspring expected improvement for each gen.
    mc_offspring_ei = None
    if mc_summary is not None:
        print(f"\nComputing offspring EI per gen "
              f"(analytic, n_initial_evals={margin}, K=3 empirical window, "
              f"closed-form TS parent select)...")
        mc_offspring_ei = compute_offspring_ei_per_gen(
            data, rows, mc_summary, K=3, seed=0,
        )
        # Print a few examples.
        print(f"{'gen':>4}  {'E':>3}  {'baseline':>10}  {'new_fit':>10}  "
              f"{'improvement':>12}")
        for g in sorted(mc_offspring_ei)[::6]:
            r = mc_offspring_ei[g]
            if r is None:
                continue
            print(f"{g:>4}  {r['E']:>3}  {r['baseline']:>10.5f}  "
                  f"{r['new_fitness']:>10.5f}  {r['improvement']:>+12.5f}")

    per_offspring_ei = None
    if mc_summary is not None:
        per_offspring_ei = compute_per_offspring_ei(
            data, rows, mc_summary, seed=0,
        )

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    b_star_records = plot(rows, pop_size, job, out_dir,
                          mc_curves=mc_curves,
                          mc_offspring_ei=mc_offspring_ei,
                          per_offspring_ei=per_offspring_ei)

    if b_star_records:
        print(f"\nB* per generation (where smoothed MEI(B*; Δ={MARGIN}) = MC offspring EI):")
        print(f"  {'gen':>4} {'status':>22} {'B*':>10}")
        for g, status, B in b_star_records:
            B_str = f"{B:.2f}" if B is not None else "—"
            print(f"  {g:>4} {(status or 'fit-failed'):>22} {B_str:>10}")


if __name__ == "__main__":
    main()
