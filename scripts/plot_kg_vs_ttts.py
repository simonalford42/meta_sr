"""Compare reeval-allocation policies on real-run arms: TTTS vs knowledge gradient.

For each selected generation of a run, simulates the expected parent-fitness
improvement EI(B) from B reevaluations under three allocation policies, all
scored against the actual parent-selection rule (top-k truncation + binary
tournament on means):

  - batch TTTS: psi computed once from the initial posterior (what
    monte_carlo_sweep.py uses),
  - sequential TTTS: TTTS re-drawn from each sample's current posterior,
  - sequential KG: one-step knowledge gradient matched to the tournament rule
    (argmax_a E[V(posterior after one eval of a)], V = parent_dist . mu).

Per-gen sigma is read from plots/<job>/summary.json (cumulative pooled
estimate). Arms (mu, N) are extracted from runs/<job>/run_data.json via
load_arms_archive and cached to plots/.cache/<job>_arms.npz since run_data.json
can be multiple GB.

Usage: python scripts/plot_kg_vs_ttts.py [job] [gens-csv] [M] [B_max]
Writes plots/<job>_kg_vs_ttts.png
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
    ttts_reeval_policy,
    kg_reeval_policy,
)
from monte_carlo_test import load_arms_archive

TOPK = 10
TOURNEY_N = 2


def load_arms_cached(job):
    """Return {gen: (mu, N)} for every generation, via npz cache."""
    cache = Path("plots/.cache") / f"{job}_arms.npz"
    if cache.exists():
        z = np.load(cache)
        gens = sorted({int(k.split("_")[1]) for k in z.files})
        return {g: (z[f"mu_{g}"], z[f"N_{g}"]) for g in gens}

    print(f"cache miss — loading runs/{job}/run_data.json (may take a while)...")
    t = time.perf_counter()
    data = json.loads((Path("runs") / job / "run_data.json").read_text())
    print(f"  loaded in {time.perf_counter() - t:.0f}s")
    out = {}
    payload = {}
    for g in range(len(data["generations"])):
        mu, N, _labels, _bundles = load_arms_archive(data, g)
        out[g] = (mu, N)
        payload[f"mu_{g}"] = mu
        payload[f"N_{g}"] = N
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    print(f"  cached arms for {len(out)} gens -> {cache}")
    return out


def load_sigmas(job):
    summary = json.loads((Path("plots") / job / "summary.json").read_text())
    return {r["gen"]: r["sigma"] for r in summary["records"]}


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "414990"
    gens = ([int(g) for g in sys.argv[2].split(",")] if len(sys.argv) > 2
            else [2, 10, 25])
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    B_max = int(sys.argv[4]) if len(sys.argv) > 4 else 50

    arms = load_arms_cached(job)
    sigmas = load_sigmas(job)
    sel_fn = topk_tourney_batch_selection_fn(topk=TOPK, n=TOURNEY_N)

    results = []
    for gen in gens:
        mu, N = arms[gen]
        sigma = sigmas[gen]
        print(f"=== gen {gen}: k={len(mu)} sigma={sigma:.4f} M={M} B_max={B_max} ===")

        # Same seed per policy -> identical true_mu draws (CRN on the truths).
        t = time.perf_counter()
        c_batch = simulate_reeval_expected_improvement(
            mu, sigma, N, sel_fn, M=M, B_max=B_max, rng=np.random.default_rng(0))
        print(f"  batch-TTTS {time.perf_counter() - t:.1f}s  EI[{B_max}]={c_batch[-1]:+.4f}")

        t = time.perf_counter()
        c_ttts, cnt_ttts = simulate_reeval_expected_improvement_policy(
            mu, sigma, N, sel_fn, ttts_reeval_policy(beta=0.5),
            M=M, B_max=B_max, rng=np.random.default_rng(0), return_counts=True)
        print(f"  seq-TTTS   {time.perf_counter() - t:.1f}s  EI[{B_max}]={c_ttts[-1]:+.4f}")

        t = time.perf_counter()
        c_kg, cnt_kg = simulate_reeval_expected_improvement_policy(
            mu, sigma, N, sel_fn, kg_reeval_policy(sel_fn, prune_topk=TOPK),
            M=M, B_max=B_max, rng=np.random.default_rng(0), return_counts=True)
        print(f"  KG         {time.perf_counter() - t:.1f}s  EI[{B_max}]={c_kg[-1]:+.4f}")

        results.append(dict(gen=gen, mu=mu, N=N, sigma=sigma,
                            c_batch=c_batch, c_ttts=c_ttts, c_kg=c_kg,
                            cnt_ttts=cnt_ttts, cnt_kg=cnt_kg))

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 9), squeeze=False)
    B_axis = np.arange(B_max + 1)

    for col, r in enumerate(results):
        ax = axes[0][col]
        ax.plot(B_axis, r["c_batch"], color="0.55", linewidth=1.3,
                label="TTTS (batch ψ)")
        ax.plot(B_axis, r["c_ttts"], color="C0", linewidth=1.6,
                label="TTTS (sequential)")
        ax.plot(B_axis, r["c_kg"], color="C2", linewidth=1.8,
                label="KG (tournament-aware)")
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_xlabel("reeval budget B")
        ax.set_ylabel("E[truth[parent]] − baseline")
        ax.set_title(f"gen {r['gen']} — k={len(r['mu'])}, σ={r['sigma']:.3f}")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

        # Allocation profile: avg reevals per arm rank (sorted by mu desc).
        ax = axes[1][col]
        order = np.argsort(-r["mu"])
        show = min(30, len(order))
        x = np.arange(show)
        ax.bar(x - 0.2, r["cnt_ttts"][order][:show], width=0.4, color="C0",
               label="TTTS (sequential)")
        ax.bar(x + 0.2, r["cnt_kg"][order][:show], width=0.4, color="C2",
               label="KG")
        ax.axvline(TOPK - 0.5, color="C3", linestyle="--", linewidth=1.0,
                   label=f"top-{TOPK} boundary")
        ax.set_xlabel(f"arm rank by μ (top {show} of {len(order)})")
        ax.set_ylabel(f"avg reevals over B={B_max}")
        ax.set_title("Where the budget goes")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"{job} — reeval allocation: TTTS vs knowledge gradient "
                 f"(top-{TOPK} tournament n={TOURNEY_N}, M={M})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path("plots") / f"{job}_kg_vs_ttts.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
