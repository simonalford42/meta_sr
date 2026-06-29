"""Run monte_carlo_test analysis across every generation in a run.

Produces one per-gen plot under plots/<job>/<job>_genNN_monte_carlo.png plus a
summary plot plots/<job>_monte_carlo_sweep.png with: σ_per_seed, baseline μ,
top-arm concentration (max α), and EI[B] curves overlaid by generation.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monte_carlo import (
    thompson_sampling_select_probs as ts_select_probs,
    top_two_thompson_sampling_select_probs as ttts_select_probs,
    simulate_reeval_expected_improvement,
    thompson_sampling_batch_selection_fn,
    topk_tourney_batch_selection_fn,
)
from monte_carlo_test import load_arms_archive as load_arms, per_seed_score_std
import offspring_improvement as oi
from offspring_improvement import (
    analyze as analyze_offspring,
    fit_ei_curve, smoothed_mei, indifference_B, trailing_avg,
    _smooth_ei, configure_margin_from_data,
)


def bundle_key(member):
    """Identity = tuple of (slot, operator code) — code, not name, so distinct
    LLM outputs that collide on the auto-generated name stay separate."""
    ops = member["operators"]
    return tuple((s, ops[s].get("code")) for s in sorted(ops))


def cumulative_sigma_estimates(data):
    """For each generation, pool per-bundle per-seed-score std across all
    bundles observed in gens 0..n (deduped by operator-name signature; keep
    the max-N version of each bundle).

    Returns list of (sigma, sigma_se, dof) per generation (None when dof=0).
    Pooled variance: Σ((n_i − 1) s_i²) / Σ(n_i − 1). Approx SE for std:
    σ / √(2·dof).
    """
    gens = data["generations"]
    seen = {}  # key -> (n_seeds, per_seed_std)
    out = []
    for g in gens:
        for m in list(g["population"]) + list(g["offspring"]):
            n_seeds = m.get("seeds_evaluated") or 0
            if n_seeds < 2:
                continue
            s = per_seed_score_std(m)
            if s is None:
                continue
            key = bundle_key(m)
            if key not in seen or seen[key][0] < n_seeds:
                seen[key] = (n_seeds, s)
        if not seen:
            out.append((None, None, 0))
            continue
        ns = np.array([v[0] for v in seen.values()], dtype=float)
        ss = np.array([v[1] for v in seen.values()], dtype=float)
        dof = float((ns - 1).sum())
        if dof <= 0:
            out.append((None, None, 0))
            continue
        sigma = float(np.sqrt(((ns - 1) * ss ** 2).sum() / dof))
        se = sigma / np.sqrt(2 * dof) if dof > 0 else None
        out.append((sigma, se, int(dof)))
    return out


def analyze_gen(data, gen, sigma, M, B_max, rng):
    mu, N, labels, bundles = load_arms(data, gen)
    if sigma is None or sigma <= 0:
        return None

    alpha = ts_select_probs(mu, sigma, N)
    psi = ttts_select_probs(mu, sigma, N, beta=0.5)
    baseline = float(alpha @ mu)
    # Parent selection = top-k truncation + binary tournament, matching the
    # actual evolution loop (survival picks top pop_size by score, then
    # tournament selects a parent). topk is the post-survival pop size.
    batch_selection_fn = topk_tourney_batch_selection_fn(topk=10, n=2)
    curve = simulate_reeval_expected_improvement(
        mu, sigma, N, batch_selection_fn, M=M, B_max=B_max, rng=rng,
    )

    rec = {
        "gen": gen,
        "k": len(mu),
        "sigma": sigma,
        "mu_min": float(mu.min()),
        "mu_max": float(mu.max()),
        "n_min": int(N.min()),
        "n_max": int(N.max()),
        "baseline": baseline,
        "alpha_max": float(alpha.max()),
        "alpha_entropy": float(-(alpha * np.log(alpha + 1e-30)).sum()),
        "curve": curve.tolist(),
        # Per-arm data needed for the per-gen plot (kept out of summary.json
        # to keep that file small — stripped below before write).
        "_mu": mu.tolist(),
        "_N": N.tolist(),
        "_alpha": alpha.tolist(),
        "_psi": psi.tolist(),
        "_labels": labels,
    }
    return rec


def compute_offspring_k3(data):
    """Return dict mapping gen -> K=3 trailing avg of offspring Δμ-mean-pop."""
    rows, _pop_size = analyze_offspring(data)
    by_gen = defaultdict(list)
    for r in rows:
        by_gen[r["gen"]].append(r["improvement"])
    gens = sorted(by_gen)
    avgs = np.array([np.mean(by_gen[g]) for g in gens])
    k3 = trailing_avg(avgs, 3)
    return dict(zip(gens, k3))


def _plot_per_gen(job, gen, mu, N, labels, sigma, alpha, psi, baseline,
                  curve, B_max, save_plot_dir, limits, offspring_k3=None):
    # Sort arms by score descending.
    order = np.argsort(-mu)
    mu_s = mu[order]
    N_s = N[order]
    alpha_s = alpha[order]
    psi_s = psi[order]
    labels_s = [labels[i] for i in order]
    post_std = sigma / np.sqrt(N_s)
    x = np.arange(len(mu_s))

    # 2x2 layout. Left column shares x-axis: posterior on top, TS/TTTS bars
    # below. Right column: racing-style (score, N) scatter and EI curve.
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.35, 1.0],
                          hspace=0.42, wspace=0.22,
                          left=0.06, right=0.98, top=0.92, bottom=0.07)
    ax_post = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[1, 0], sharex=ax_post)
    ax_scat = fig.add_subplot(gs[0, 1])
    ax_curve = fig.add_subplot(gs[1, 1])

    kind_color = {"pop": "C0", "off": "C3", "arc": "0.6"}
    colors = [kind_color[k] for k, _ in labels_s]

    # Top-left: posterior dot + ±1 std error bars, with N annotated.
    ax_post.errorbar(x, mu_s, yerr=post_std, fmt="o", capsize=4,
                     markersize=7, linewidth=1.2,
                     color="gray", ecolor="gray", zorder=2)
    for xi, mi, c in zip(x, mu_s, colors):
        ax_post.scatter([xi], [mi], color=c, s=55, zorder=3,
                        edgecolor="black", linewidth=0.5)
    for xi, mi, ni in zip(x, mu_s, N_s):
        ax_post.annotate(f"N={int(ni)}", (xi, mi),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=7, color="black")
    ax_post.set_ylabel("μ ± σ/√N")
    ax_post.set_title(f"{job} gen {gen} — σ_per_seed={sigma:.3f}  "
                      f"k={len(mu)}  baseline={baseline:.3f}")
    ax_post.grid(alpha=0.3)
    ax_post.set_ylim(limits["score_y"])
    # Legend for pop/off
    from matplotlib.patches import Patch
    ax_post.legend(handles=[Patch(color="C0", label="pop"),
                             Patch(color="C3", label="offspring"),
                             Patch(color="0.6", label="archive")],
                    fontsize=8, loc="lower left")
    plt.setp(ax_post.get_xticklabels(), visible=False)

    # Bottom-left: TS / TTTS bars in the same order.
    ax_bars.bar(x - 0.2, alpha_s, width=0.4, color="C0", label="α (TS)")
    ax_bars.bar(x + 0.2, psi_s, width=0.4, color="C3", label="ψ (TTTS, β=0.5)")
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([f"{k}" for k, _ in labels_s], fontsize=7,
                            rotation=0, color="gray")
    ax_bars.set_xlabel("arm (sorted by μ ↓)")
    ax_bars.set_ylabel("selection prob")
    ax_bars.legend(fontsize=8, loc="upper right")
    ax_bars.grid(alpha=0.3, axis="y")
    ax_bars.set_ylim(limits["prob_y"])

    # Fit smoothed EI for both right-column panels.
    popt = fit_ei_curve(np.asarray(curve))
    B_axis = np.arange(0, len(curve))
    if popt is not None:
        smooth_ei = _smooth_ei(B_axis, *popt)
        # MEI(B; Δ=oi.MARGIN) = EI(B+oi.MARGIN) − EI(B), defined for B in [0, len-oi.MARGIN-1].
        raw_mei_vals = np.asarray(curve[oi.MARGIN:]) - np.asarray(curve[:-oi.MARGIN])
        smooth_mei_vals = (_smooth_ei(B_axis[:-oi.MARGIN] + oi.MARGIN, *popt)
                           - _smooth_ei(B_axis[:-oi.MARGIN], *popt))
    else:
        smooth_ei = None
        raw_mei_vals = np.asarray(curve[oi.MARGIN:]) - np.asarray(curve[:-oi.MARGIN])
        smooth_mei_vals = None

    # Top-right: marginal EI vs B, raw + smoothed, with offspring K=3 hline.
    B_mei = B_axis[:-oi.MARGIN]
    ax_scat.plot(B_mei, raw_mei_vals, color="0.55", linewidth=0.9, alpha=0.7,
                 label=f"raw MEI (Δ={oi.MARGIN})")
    if smooth_mei_vals is not None:
        ax_scat.plot(B_mei, smooth_mei_vals, color="C2", linewidth=2.0,
                     label=f"smoothed MEI (Δ={oi.MARGIN})")

    if offspring_k3 is not None and np.isfinite(offspring_k3):
        y_top = limits["mei_y"][1]
        off_axis_above = offspring_k3 > y_top
        if off_axis_above:
            # Draw an arrow at the top of the panel instead of an off-screen hline.
            # Placed top-left so it doesn't collide with the upper-right legend.
            ax_scat.annotate(f"↑ K=3 offspring = {offspring_k3:+.4f}  (above scale)",
                             xy=(0.02, 0.97), xycoords="axes fraction",
                             ha="left", va="top", color="C3", fontsize=9)
            # Add a phantom line entry for the legend so colors still match.
            ax_scat.plot([], [], color="C3", linestyle="--", linewidth=1.4,
                         label=f"K=3 offspring avg = {offspring_k3:+.4f} (off-axis)")
        else:
            ax_scat.axhline(offspring_k3, color="C3", linestyle="--",
                            linewidth=1.4,
                            label=f"K=3 offspring avg = {offspring_k3:+.4f}")
        # B* intersection: smoothed MEI(B*) = offspring_k3.
        if popt is not None:
            status, B_star = indifference_B(popt, float(offspring_k3))
            ann_y = (offspring_k3 if not off_axis_above
                     else y_top * 0.85)  # keep annotation inside the visible area
            if status == "finite" and B_star is not None and B_star <= B_max:
                ax_scat.axvline(B_star, color="C3", linestyle=":",
                                linewidth=1.2, alpha=0.8)
                ax_scat.annotate(f"B*={B_star:.1f}",
                                 (B_star, ann_y),
                                 textcoords="offset points",
                                 xytext=(8, 8), fontsize=9, color="C3")
            elif status == "offspring-dominates":
                ax_scat.annotate("offspring beats reeval at every B",
                                 (B_max * 0.5, ann_y),
                                 textcoords="offset points",
                                 xytext=(0, 8), fontsize=8, color="C3",
                                 ha="center")
            elif status == "no-improvement":
                ax_scat.annotate("K=3 avg ≤ 0",
                                 (B_max * 0.5, ann_y),
                                 textcoords="offset points",
                                 xytext=(0, -12), fontsize=8, color="C3",
                                 ha="center")

    ax_scat.axhline(0, color="k", linewidth=0.5)
    ax_scat.set_xlabel("reeval budget B")
    ax_scat.set_ylabel(f"MEI(B; Δ={oi.MARGIN}) = EI(B+{oi.MARGIN}) − EI(B)")
    ax_scat.set_title(f"Marginal EI per Δ={oi.MARGIN} evals\n"
                      f"intersection with K=3 offspring avg → B*")
    ax_scat.legend(fontsize=8, loc="upper right")
    ax_scat.grid(alpha=0.3)
    ax_scat.set_ylim(limits["mei_y"])
    ax_scat.set_xlim(0, B_max)

    # Bottom-right: EI curve with smoothed overlay.
    ax_curve.plot(B_axis, curve, "o-", color="C2",
                  markersize=2.5, linewidth=1.0, label="raw EI")
    if smooth_ei is not None:
        ax_curve.plot(B_axis, smooth_ei, color="0.45", linewidth=1.0,
                      linestyle="--", alpha=0.85,
                      label=(f"smoothed EI: 2-exp, τ₁={popt[1]:.1f}, "
                             f"τ₂={popt[3]:.1f}"))

    # Offspring tangent: a line with slope = offspring_EI / Δ (offspring's
    # per-seed value) is tangent to the smoothed EI curve at B* — same
    # indifference point as the MEI=offspring_EI intersection on the MEI panel,
    # viewed marginally on the cumulative curve.
    if (offspring_k3 is not None and np.isfinite(offspring_k3)
            and offspring_k3 > 0 and popt is not None and oi.MARGIN > 0):
        slope = float(offspring_k3) / float(oi.MARGIN)
        status_t, B_star_t = indifference_B(popt, float(offspring_k3))
        if status_t == "finite" and B_star_t is not None and B_star_t >= 0:
            y_at = _smooth_ei(B_star_t, *popt)
            xs = np.array([0.0, float(B_max)])
            ys = y_at + slope * (xs - B_star_t)
            off_axis = B_star_t > B_max
            anchor_note = " (anchor off-axis)" if off_axis else ""
            ax_curve.plot(xs, ys, color="C3", linewidth=1.2, alpha=0.85,
                          label=(f"offspring tangent: slope={slope:.5g} "
                                 f"= offspring_EI/Δ, B*={B_star_t:.1f}{anchor_note}"))
            if not off_axis:
                ax_curve.scatter([B_star_t], [y_at], color="C3", s=40, zorder=5,
                                 edgecolors="black", linewidth=0.6)
        elif status_t == "offspring-dominates":
            # No interior tangent: offspring's per-seed value exceeds the
            # steepest reeval slope (which is at B=0). Report both for context.
            steepest = smoothed_mei(popt, 0.0) / float(oi.MARGIN)
            xs = np.array([0.0, float(B_max)])
            ys = slope * xs
            ax_curve.plot(xs, ys, color="C3", linewidth=1.2, alpha=0.85,
                          linestyle="--",
                          label=(f"offspring slope={slope:.5g} > "
                                 f"steepest reeval slope (B=0)={steepest:.5g} "
                                 f"⇒ B*=0"))
    ax_curve.axhline(0, color="k", linewidth=0.7)
    ax_curve.set_xlabel("reeval budget B")
    ax_curve.set_ylabel("E[truth[chosen]] − baseline")
    ax_curve.set_title("Expected improvement (TTTS allocation, true-fitness)")
    ax_curve.legend(fontsize=8, loc="lower right")
    ax_curve.grid(alpha=0.3)
    ax_curve.set_ylim(limits["ei_y"])
    ax_curve.set_xlim(0, B_max)

    # No tight_layout — gridspec already supplies the margins; tight_layout
    # would override hspace/wspace and re-collapse the title-stacking issue.
    out = save_plot_dir / f"{job}_gen{gen:02d}_monte_carlo.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)


def compute_global_limits(records, sigma_per_gen, pad=0.05):
    """Common axis ranges across all per-gen plots so they're comparable.

    score_y  — μ ± σ_eff over all arms in all gens (σ_eff uses each gen's σ).
    prob_y   — 0..max(α, ψ) across gens.
    N_y      — observed N range, padded.
    ei_y     — 0..max EI value across gens.
    """
    score_lo, score_hi = np.inf, -np.inf
    N_lo, N_hi = np.inf, -np.inf
    prob_hi = 0.0
    ei_hi = 0.0
    mei_hi = 0.0
    for r, sig in zip(records, sigma_per_gen):
        mu = np.asarray(r["_mu"])
        N = np.asarray(r["_N"])
        post_std = sig / np.sqrt(N)
        score_lo = min(score_lo, float((mu - post_std).min()))
        score_hi = max(score_hi, float((mu + post_std).max()))
        N_lo = min(N_lo, float(N.min()))
        N_hi = max(N_hi, float(N.max()))
        prob_hi = max(prob_hi, float(max(max(r["_alpha"]), max(r["_psi"]))))
        ei_hi = max(ei_hi, float(max(r["curve"])))
        ei = np.asarray(r["curve"])
        if len(ei) > oi.MARGIN:
            raw_mei = ei[oi.MARGIN:] - ei[:-oi.MARGIN]
            mei_hi = max(mei_hi, float(raw_mei.max()))

    score_pad = (score_hi - score_lo) * pad
    N_pad = max(1.0, (N_hi - N_lo) * pad)
    return {
        "score_y": (score_lo - score_pad, score_hi + score_pad),
        "prob_y":  (0.0, prob_hi * (1 + pad)),
        "N_y":     (N_lo - N_pad, N_hi + N_pad),
        "ei_y":    (0.0, ei_hi * (1 + pad)),
        "mei_y":   (0.0, mei_hi * (1 + pad)),
    }


def plot_summary(records, job, out_dir, B_max):
    gens = np.array([r["gen"] for r in records])
    sigmas = np.array([r["sigma"] for r in records])
    baselines = np.array([r["baseline"] for r in records])
    alpha_max = np.array([r["alpha_max"] for r in records])
    alpha_ent = np.array([r["alpha_entropy"] for r in records])
    curves = np.array([r["curve"] for r in records])  # [G, B_max]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sigma_ses = np.array([r.get("sigma_se") or 0.0 for r in records])
    dofs = np.array([r.get("sigma_dof") or 0 for r in records])
    ax = axes[0, 0]
    ax.errorbar(gens, sigmas, yerr=sigma_ses, fmt="o-", color="C4",
                capsize=3, ecolor="C4", alpha=0.9)
    ax.set_xlabel("generation")
    ax.set_ylabel("σ_per_seed  (cumulative, pooled)")
    ax.set_title("Cumulative per-seed noise σ (errorbars: SE ≈ σ/√(2·dof))")
    ax.grid(alpha=0.3)
    # Twin axis showing accumulated dof so you can see how the estimate
    # tightens as more bundles contribute.
    ax2 = ax.twinx()
    ax2.plot(gens, dofs, "--", color="gray", alpha=0.55, linewidth=1)
    ax2.set_ylabel("cumulative dof  (gray, dashed)", color="gray")

    ax = axes[0, 1]
    ax.plot(gens, baselines, "o-", color="C0", label="baseline E[top-arm μ]")
    ax.fill_between(gens,
                    np.array([r["mu_min"] for r in records]),
                    np.array([r["mu_max"] for r in records]),
                    alpha=0.15, color="C0", label="μ range across arms")
    ax.set_xlabel("generation"); ax.set_ylabel("score")
    ax.set_title("Posterior baseline and μ-range per generation")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(gens, alpha_max, "o-", color="C1", label="max α (TS concentration)")
    ax2 = ax.twinx()
    ax2.plot(gens, alpha_ent, "s--", color="C3", label="entropy(α)")
    ax.set_xlabel("generation"); ax.set_ylabel("max α", color="C1")
    ax2.set_ylabel("entropy of α (nats)", color="C3")
    ax.set_title("Posterior concentration on the leading arm")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(records):
        c = cmap(i / max(1, len(records) - 1))
        ax.plot(np.arange(0, len(r["curve"])), r["curve"], color=c, alpha=0.8,
                linewidth=1.2)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=gens.min(),
                                                  vmax=gens.max()))
    fig.colorbar(sm, ax=ax, label="generation")
    ax.set_xlabel("reeval budget B")
    ax.set_ylabel("E[top-arm μ] − baseline")
    ax.set_title("Expected improvement curves per generation")
    ax.grid(alpha=0.3)

    fig.suptitle(f"{job} — monte-carlo sweep over all generations", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"{job}_monte_carlo_sweep.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "666286"
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    B_max = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    data = json.loads((Path("runs") / job / "run_data.json").read_text())
    margin = configure_margin_from_data(data)
    print(f"Using MARGIN = {margin} (from config.n_runs)")
    n_gens = len(data["generations"])
    out_dir = Path("plots")
    per_gen_dir = out_dir / job
    per_gen_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== sweep {job}: {n_gens} generations, M={M}, B_max={B_max} ===")
    cum_sigmas = cumulative_sigma_estimates(data)
    rng = np.random.default_rng(0)
    records = []
    sigma_per_gen = []
    t_start = time.perf_counter()
    print("--- pass 1: analysis ---")
    for gen in range(n_gens):
        sigma, sigma_se, dof = cum_sigmas[gen]
        if sigma is None:
            print(f"  gen {gen:2d}: skipped (no usable cumulative sigma)")
            continue
        t = time.perf_counter()
        rec = analyze_gen(data, gen, sigma, M, B_max, rng)
        elapsed = time.perf_counter() - t
        if rec is None:
            print(f"  gen {gen:2d}: skipped (no arms)")
            continue
        rec["sigma_se"] = sigma_se
        rec["sigma_dof"] = dof
        records.append(rec)
        sigma_per_gen.append(sigma)
        print(f"  gen {gen:2d}: σ_cum={sigma:.4f}±{sigma_se:.4f} (dof={dof}) "
              f"k={rec['k']} baseline={rec['baseline']:.3f} "
              f"α_max={rec['alpha_max']:.2f} "
              f"EI[1]={rec['curve'][1]:+.4f} "
              f"EI[10]={rec['curve'][min(10, B_max)]:+.4f} "
              f"EI[{B_max}]={rec['curve'][B_max]:+.4f}   ({elapsed:.2f}s)")
    analysis_t = time.perf_counter() - t_start

    # Pass 2: render per-gen plots with common axis limits.
    print("--- pass 2: render per-gen plots with shared limits ---")
    limits = compute_global_limits(records, sigma_per_gen)
    print(f"  score_y={limits['score_y']}  prob_y={limits['prob_y']}  "
          f"N_y={limits['N_y']}  ei_y={limits['ei_y']}  "
          f"mei_y={limits['mei_y']}")
    # Per-gen K=3 trailing average of offspring Δμ-mean-pop, used as the
    # indifference target in the marginal-EI panel.
    offspring_k3 = compute_offspring_k3(data)
    t = time.perf_counter()
    for rec, sig in zip(records, sigma_per_gen):
        _plot_per_gen(job, rec["gen"],
                      np.asarray(rec["_mu"]), np.asarray(rec["_N"]),
                      rec["_labels"], sig,
                      np.asarray(rec["_alpha"]), np.asarray(rec["_psi"]),
                      rec["baseline"], np.asarray(rec["curve"]),
                      B_max, per_gen_dir, limits,
                      offspring_k3=offspring_k3.get(rec["gen"]))
    render_t = time.perf_counter() - t

    total = time.perf_counter() - t_start
    print(f"\nTotal sweep: {total:.2f}s "
          f"(analysis {analysis_t:.1f}s, render {render_t:.1f}s) "
          f"for {len(records)}/{n_gens} gens")

    # Strip the per-arm payload from summary.json to keep it compact.
    slim = [{k: v for k, v in r.items() if not k.startswith("_")}
            for r in records]
    summary_path = per_gen_dir / "summary.json"
    summary_path.write_text(json.dumps(
        {"job": job, "M": M, "B_max": B_max, "records": slim}, indent=2))
    print(f"Wrote {summary_path}")

    if records:
        plot_summary(records, job, out_dir, B_max)


if __name__ == "__main__":
    main()
