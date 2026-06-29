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

Usage: python scripts/offspring_improvement.py [job]  (default 666286)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq


# Per-seed cost of one offspring (used as the margin for MEI computation).
# Each offspring is evaluated on 3 seeds initially (matches the run config).
MARGIN = 3


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
    ymax = max(float(ys.max()), 1e-6)
    p0 = [ymax * 0.5, 5.0, ymax * 0.5, 60.0]
    popt, _ = curve_fit(_smooth_ei, xs, ys, p0=p0,
                        bounds=([0.0, 0.3, 0.0, 0.3],
                                [np.inf, 200.0, np.inf, 2000.0]),
                        maxfev=10000)
    return tuple(popt)


def raw_mei(ei, B, margin=MARGIN):
    """EI[B+margin] − EI[B] from the raw curve (absolute indexing).
    None if B+margin exceeds available budget."""
    if B + margin >= len(ei):
        return None
    return ei[B + margin] - ei[B]


def smoothed_mei(popt, B, margin=MARGIN):
    """Marginal smoothed EI at budget B for a margin of `margin` extra evals.

    For the two-component model MEI(B; Δ) reduces to:
        a₁·(1 − e^(−Δ/τ₁))·e^(−B/τ₁) + a₂·(1 − e^(−Δ/τ₂))·e^(−B/τ₂)
    which is monotonically decreasing in B → brentq for the inverse.
    """
    if popt is None:
        return None
    return _smooth_ei(B + margin, *popt) - _smooth_ei(B, *popt)


def indifference_B(popt, target, margin=MARGIN, B_upper=10000.0):
    """Numerical inverse: smoothed_MEI(B; margin) = target.

    Returns (status, B_star):
        ('finite', value) when a finite non-negative B solves it
        ('offspring-dominates', 0.0) when target ≥ MEI(0)
        ('no-improvement', None) when target ≤ 0
        (None, None) on fit failure or numerical issue.
    """
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
    ops = m["operators"]
    return tuple(ops[s]["name"] for s in sorted(ops))


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


def load_mc_curve(job):
    """Per-gen EI[B] dict from a prior monte_carlo_sweep run, if available."""
    p = Path("plots") / job / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    out = {}
    for r in s.get("records", []):
        out[int(r["gen"])] = r["curve"]
    return out


def plot(rows, pop_size, job, out_dir, mc_curves=None,
         mei_B_values=(10, 40)):
    by_gen = defaultdict(list)
    for r in rows:
        by_gen[r["gen"]].append(r)
    gens = sorted(by_gen)
    avgs = np.array([np.mean([r["improvement"] for r in by_gen[g]]) for g in gens])
    n_per_gen = np.array([len(by_gen[g]) for g in gens])
    gens_arr = np.array(gens)
    k3_avgs = trailing_avg(avgs, 3)

    fig = plt.figure(figsize=(14, 16))
    # Bottom panel is 3× its previous height so B* small variations are
    # readable even when the gen-34 spike (~575) anchors the y-range.
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 3.0],
                          left=0.07, right=0.98, top=0.96, bottom=0.05,
                          hspace=0.18)
    ax = fig.add_subplot(gs[0])
    axes = [ax, fig.add_subplot(gs[1], sharex=ax)]

    # Per-offspring dots — light gray; entered offspring slightly darker so
    # you can see which gens had which contributing.
    for g in gens:
        for r in by_gen[g]:
            color = "C3" if r["entered"] else "0.75"
            alpha = 0.55 if r["entered"] else 0.25
            ax.scatter([g], [r["improvement"]], color=color, alpha=alpha,
                       s=10, zorder=1, edgecolors="none")

    # Average line and trailing averages.
    ax.plot(gens_arr, avgs, "o-", color="C7", alpha=0.45, linewidth=0.7,
            markersize=4,
            label=f"raw avg (over {int(n_per_gen.mean())} offspring/gen)",
            zorder=2)
    ax.plot(gens_arr, k3_avgs, linewidth=2.2, color="C0",
            label="K=3 trailing avg of offspring Δμ-mean-pop", zorder=3)

    # Overlay: smoothed marginal EI from monte_carlo.
    # MEI(B; margin=3) = EI(B+3) − EI(B), per-3-eval marginal improvement
    # in top-arm posterior μ at budget B.
    if mc_curves:
        mc_g = sorted(mc_curves)
        cmap = plt.get_cmap("viridis")
        for i, B in enumerate(mei_B_values):
            color = cmap(0.1 + 0.7 * i / max(1, len(mei_B_values) - 1))
            sm_vals = []
            for g in mc_g:
                popt = fit_ei_curve(mc_curves[g])
                sm_vals.append(smoothed_mei(popt, B))
            sm_gs = [g for g, v in zip(mc_g, sm_vals) if v is not None]
            sm_ys = [v for v in sm_vals if v is not None]
            ax.plot(sm_gs, sm_ys, "-", color=color, linewidth=1.0,
                    alpha=0.95, zorder=4,
                    label=f"smoothed MEI @ B={B}, Δ=3")

    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_xlabel("generation")
    ax.set_ylabel(f"per-offspring Δ(mean pop μ)  =  "
                  f"(O.final − D.final) / pop_size   [pop_size={pop_size}]")
    ax.set_title(f"{job} — per-offspring contribution to mean-population score "
                 f"vs marginal MC EI (Δ={MARGIN} evals = one-offspring cost)\n"
                 "dots: each offspring's Δμ-mean-pop; "
                 "red = entered the pop, gray = didn't")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom panel: B* per gen where smoothed MEI(B*; Δ=3) = K=3 offspring avg.
    ax_b = axes[1]
    if mc_curves:
        b_star_records = []  # (gen, status, B_star) per gen
        for g, t in zip(gens, k3_avgs):
            if g not in mc_curves or not np.isfinite(t):
                b_star_records.append((g, None, None))
                continue
            popt = fit_ei_curve(mc_curves[g])
            status, B = indifference_B(popt, float(t))
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
    ax_b.set_ylabel(f"B*  (reeval budget where\nMEI(B*; Δ={MARGIN}) = K=3 avg)")
    ax_b.set_title(f"Indifference point: how many reevals on the existing "
                   f"pool buy the same Δμ-mean-pop as one offspring")
    ax_b.grid(alpha=0.3); ax_b.legend(fontsize=9, loc="upper left")

    out = out_dir / f"{job}_offspring_improvement.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")

    if mc_curves:
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
    rows, pop_size = analyze(data)
    mc_curves = load_mc_curve(job)
    if mc_curves:
        max_b = max(len(c) for c in mc_curves.values())
        print(f"Loaded MC sweep with EI curves for {len(mc_curves)} gens "
              f"(B_max={max_b}, margin={MARGIN}).")
        # max_b is the array length = B_max + 1; valid B values are 0..max_b-1.
        if max_b - 1 < 40 + MARGIN:
            print(f"  ⚠ raw MEI@B=40 needs B_max≥{40+MARGIN}; re-run sweep "
                  f"with python scripts/monte_carlo_sweep.py {job} 5000 50")
    else:
        print("No prior MC sweep summary.json — skipping MEI overlay. "
              "Run scripts/monte_carlo_sweep.py first to enable.")

    print(f"=== {job} (pop_size={pop_size}) ===")
    print_summary(rows, pop_size)

    # Print a few example MEI values so we can sanity-check before plotting.
    if mc_curves:
        print("\nSmoothed MEI examples (per gen, margin=3):")
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

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    b_star_records = plot(rows, pop_size, job, out_dir, mc_curves=mc_curves)

    if b_star_records:
        print("\nB* per generation (where smoothed MEI(B*; Δ=3) = K=3 avg):")
        print(f"  {'gen':>4} {'status':>22} {'B*':>10}")
        for g, status, B in b_star_records:
            B_str = f"{B:.2f}" if B is not None else "—"
            print(f"  {g:>4} {(status or 'fit-failed'):>22} {B_str:>10}")


if __name__ == "__main__":
    main()
