"""B* over generations under TTTS vs knowledge-gradient reeval allocation.

B* is the indifference point where the smoothed marginal EI of reevaluation,
MEI(B*; Δ=MARGIN), equals the MC offspring EI for that generation (the value of
one new offspring, offspring_mc.offspring_expected_improvement with the
topk-tourney parent-selection rule). Below B*, a reeval beats a new offspring;
above it, breed instead.

For each generation we compute the reeval EI curve two ways and derive B* from
each fit:
  - TTTS: batch psi allocation (monte_carlo_sweep's policy),
  - KG: sequential tournament-aware knowledge gradient.

Pass 1 extracts per-gen (mu, N, sigma, offspring-EI target) from
runs/<job>/run_data.json and caches to plots/.cache/<job>_bstar_inputs.* —
run_data.json can be huge, so this runs once. Pass 2 (curves) caches per-gen
curves to plots/.cache/<job>_bstar_curves.npz keyed by (gen, policy, M, B_max).

Usage: python scripts/plot_bstar_kg_vs_ttts.py [job] [max_gens] [M_kg] [B_max]
Writes plots/<job>_bstar_kg_vs_ttts.png
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monte_carlo import (
    simulate_reeval_expected_improvement,
    simulate_reeval_expected_improvement_policy,
    topk_tourney_batch_selection_fn,
    kg_reeval_policy,
)
from monte_carlo_test import load_arms_archive
from monte_carlo_sweep import cumulative_sigma_estimates
import offspring_improvement as oi
from offspring_improvement import (
    analyze, configure_margin_from_data, fit_ei_curve, indifference_B,
    offspring_empirical_for_gen,
)
from offspring_mc import offspring_expected_improvement

TOPK = 10
TOURNEY_N = 2
M_TTTS = 4000
KG_N_QUAD = 4
KG_PRUNE_Z = 2.33


def build_inputs(job):
    """Per-gen mu, N, sigma, offspring-EI target and MARGIN, via cache."""
    cache_dir = Path("plots/.cache")
    npz_path = cache_dir / f"{job}_bstar_inputs.npz"
    json_path = cache_dir / f"{job}_bstar_inputs.json"
    if npz_path.exists() and json_path.exists():
        z = np.load(npz_path)
        meta = json.loads(json_path.read_text())
        oi.MARGIN = meta["margin"]
        arms = {int(g): (z[f"mu_{g}"], z[f"N_{g}"]) for g in meta["gens"]}
        return arms, {int(g): v for g, v in meta["per_gen"].items()}, meta["margin"]

    print(f"cache miss — loading runs/{job}/run_data.json ...")
    t = time.perf_counter()
    data = json.loads((Path("runs") / job / "run_data.json").read_text())
    print(f"  loaded in {time.perf_counter() - t:.0f}s")
    margin = configure_margin_from_data(data)
    rows, pop_size = analyze(data)
    sigmas = cumulative_sigma_estimates(data)
    n_gens = len(data["generations"])
    print(f"  {n_gens} gens, pop_size={pop_size}, MARGIN={margin}")

    sel_fn = topk_tourney_batch_selection_fn(topk=TOPK, n=TOURNEY_N)
    arms, per_gen, payload = {}, {}, {}
    for g in range(n_gens):
        mu, N, _labels, _bundles = load_arms_archive(data, g)
        sigma = sigmas[g][0]
        target = None
        if sigma is not None and sigma > 0:
            empirical = offspring_empirical_for_gen(rows, g, K=3,
                                                    n_initial_evals=margin)
            if empirical.size:
                res = offspring_expected_improvement(
                    pop_mu=mu, pop_N=N, offspring_empirical=empirical,
                    sigma=sigma, n_initial_evals=margin,
                    batch_selection_fn=sel_fn,
                )
                if res is not None:
                    target = float(res["improvement"])
        arms[g] = (mu, N)
        per_gen[g] = {"sigma": sigma, "target": target}
        payload[f"mu_{g}"] = mu
        payload[f"N_{g}"] = N

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **payload)
    json_path.write_text(json.dumps(
        {"margin": margin, "gens": list(arms), "per_gen": per_gen}))
    print(f"  cached inputs -> {npz_path}")
    return arms, per_gen, margin


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "40319"
    max_gens = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    M_kg = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    B_max = int(sys.argv[4]) if len(sys.argv) > 4 else 50

    arms, per_gen, margin = build_inputs(job)
    usable = [g for g in sorted(arms)
              if per_gen[g]["sigma"] and per_gen[g]["target"] is not None]
    stride = max(1, int(np.ceil(len(usable) / max_gens)))
    gens = usable[::stride]
    print(f"{len(usable)} usable gens, stride={stride} -> {len(gens)} analyzed; "
          f"MARGIN={margin}, M_kg={M_kg}, B_max={B_max}")

    sel_fn = topk_tourney_batch_selection_fn(topk=TOPK, n=TOURNEY_N)
    curve_cache = Path("plots/.cache") / f"{job}_bstar_curves.npz"
    cached = dict(np.load(curve_cache)) if curve_cache.exists() else {}

    records = []
    for g in gens:
        mu, N = arms[g]
        sigma = per_gen[g]["sigma"]
        target = per_gen[g]["target"]

        kt = f"ttts_{g}_M{M_TTTS}_B{B_max}"
        kk = f"kg_{g}_M{M_kg}_B{B_max}_q{KG_N_QUAD}_z{KG_PRUNE_Z}"
        t0 = time.perf_counter()
        if kt not in cached:
            cached[kt] = simulate_reeval_expected_improvement(
                mu, sigma, N, sel_fn, M=M_TTTS, B_max=B_max,
                rng=np.random.default_rng(0))
        if kk not in cached:
            cached[kk] = simulate_reeval_expected_improvement_policy(
                mu, sigma, N, sel_fn,
                kg_reeval_policy(sel_fn, prune_topk=TOPK,
                                 n_quad=KG_N_QUAD, prune_z=KG_PRUNE_Z),
                M=M_kg, B_max=B_max, rng=np.random.default_rng(0))
            np.savez_compressed(curve_cache, **cached)

        rec = {"gen": g, "k": len(mu), "target": target}
        for name, key in (("ttts", kt), ("kg", kk)):
            popt = fit_ei_curve(cached[key])
            status, B_star = indifference_B(popt, target, margin=margin)
            rec[name] = (status, B_star, float(cached[key][-1]))
        records.append(rec)
        print(f"  gen {g:3d} k={len(mu):4d} target={target:+.5f}  "
              f"TTTS: {rec['ttts'][0]} B*={rec['ttts'][1]}  "
              f"KG: {rec['kg'][0]} B*={rec['kg'][1]}  "
              f"({time.perf_counter() - t0:.1f}s)")
    np.savez_compressed(curve_cache, **cached)

    fig, (ax_b, ax_ei) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    colors = {"ttts": "C0", "kg": "C2"}
    labels = {"ttts": "TTTS (batch ψ)", "kg": "KG (tournament-aware)"}
    finite_b_all = [r[p][1] for r in records for p in ("ttts", "kg")
                    if r[p][0] == "finite"]
    # Cap the axis so degenerate gens (offspring EI ≈ 0 ⇒ B* explodes) don't
    # flatten the readable range; off-scale points are clamped and annotated.
    in_scale = [v for v in finite_b_all if v < 200] or finite_b_all
    y_top = 1.15 * max(in_scale) if in_scale else 50.0
    for p in ("ttts", "kg"):
        fg = [r["gen"] for r in records if r[p][0] == "finite"]
        fb = [r[p][1] for r in records if r[p][0] == "finite"]
        dg = [r["gen"] for r in records if r[p][0] == "offspring-dominates"]
        ng = [r["gen"] for r in records if r[p][0] == "no-improvement"]
        for g_off, b_off in [(g, b) for g, b in zip(fg, fb) if b > y_top]:
            ax_b.annotate(f"↑{b_off:.0f}", (g_off, y_top * 0.97), ha="center",
                          va="top", fontsize=7, color=colors[p])
        fb = [min(b, y_top * 0.95) for b in fb]
        ax_b.plot(fg, fb, "o-", color=colors[p], linewidth=1.7, markersize=5,
                  label=f"{labels[p]} — finite B* (n={len(fg)})")
        if dg:
            ax_b.scatter(dg, [0] * len(dg), marker="v", color=colors[p], s=45,
                         zorder=4,
                         label=f"{labels[p]} — offspring dominates (n={len(dg)})")
        if ng:
            ax_b.scatter(ng, [y_top] * len(ng), marker="^", color=colors[p],
                         s=45, zorder=4,
                         label=f"{labels[p]} — target ≤ 0 (n={len(ng)})")
    ax_b.axhline(0, color="k", linewidth=0.7)
    ax_b.set_ylabel(f"B*  (MEI(B*; Δ={margin}) = offspring EI)")
    ax_b.set_title(f"{job} — indifference budget B* per generation: reevals "
                   f"allocated by TTTS vs KG\n(top-{TOPK} tournament n={TOURNEY_N}, "
                   f"M_ttts={M_TTTS}, M_kg={M_kg}, B_max={B_max})")
    ax_b.grid(alpha=0.3)
    ax_b.legend(fontsize=8, loc="upper left")

    g_axis = [r["gen"] for r in records]
    for p in ("ttts", "kg"):
        ax_ei.plot(g_axis, [r[p][2] for r in records], "o-", color=colors[p],
                   markersize=4, linewidth=1.4, label=f"{labels[p]} EI[{B_max}]")
    ax_ei.plot(g_axis, [r["target"] for r in records], "s--", color="C3",
               markersize=4, linewidth=1.2, label="offspring EI (target)")
    ax_ei.axhline(0, color="k", linewidth=0.7)
    ax_ei.set_xlabel("generation")
    ax_ei.set_ylabel("Δ E[truth of chosen parent]")
    ax_ei.set_title(f"Context: total reeval EI at B={B_max} vs one-offspring EI")
    ax_ei.grid(alpha=0.3)
    ax_ei.legend(fontsize=8)

    fig.tight_layout()
    out = Path("plots") / f"{job}_bstar_kg_vs_ttts.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
