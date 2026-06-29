"""Smoke test monte_carlo.py on real-data arms from run 666286, generation 2.

Treats each population+offspring bundle as an arm with mu = bundle.score and
N = bundle.seeds_evaluated. Estimates per-seed sigma from the within-bundle
seed-score std (pooled across all bundles in this generation).
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monte_carlo import (
    thompson_sampling_select_probs, top_two_thompson_sampling_select_probs,
    simulate_reeval_expected_improvement,
    thompson_sampling_batch_selection_fn,
)


def per_seed_score_std(member):
    """Return std of one bundle's per-seed avg-across-tasks scores, or None."""
    rd = member.get("result_details") or []
    n_seeds = member["seeds_evaluated"]
    per_task = []
    for t in range(len(rd)):
        s = rd[t].get("run_gt_scores", [])
        if len(s) >= n_seeds:
            per_task.append(s[:n_seeds])
    if not per_task:
        return None
    arr = np.asarray(per_task)            # [tasks, seeds]
    per_seed = arr.mean(axis=0)           # [seeds]
    if len(per_seed) < 2:
        return None
    return float(np.std(per_seed, ddof=1))


def estimate_sigma(bundles):
    """Pooled per-seed sigma across all bundles (weighted by n_seeds)."""
    stds, ns = [], []
    for m in bundles:
        s = per_seed_score_std(m)
        if s is None:
            continue
        stds.append(s)
        ns.append(m["seeds_evaluated"])
    if not stds:
        return None
    return float(np.sqrt(np.average(np.array(stds) ** 2, weights=ns)))


def load_arms(run_data, gen):
    g = run_data["generations"][gen]
    bundles = list(g["population"]) + list(g["offspring"])
    mu = np.array([b["score"] for b in bundles], dtype=float)
    N = np.array([b["seeds_evaluated"] for b in bundles], dtype=float)
    labels = []
    for i, b in enumerate(bundles):
        kind = "pop" if i < len(g["population"]) else "off"
        ops = b["operators"]
        name = "|".join(ops[s]["name"] for s in sorted(ops))
        labels.append((kind, name))
    return mu, N, labels, bundles


def _bundle_key(b):
    # Identity by operator CODE, not name: the LLM can emit a new operator with
    # a coincidentally identical auto-generated name but different code, and
    # those are genuinely distinct bundles that must not be merged.
    ops = b["operators"]
    return tuple((s, ops[s].get("code")) for s in sorted(ops))


def load_arms_archive(run_data, gen):
    """All-time archive pool through `gen`: every distinct bundle ever seen,
    deduped by operator-name signature, keeping the max-N version.

    Labels are tagged "pop" if the bundle appears in gen's population, "off"
    if it's gen's freshly-born offspring, "arc" otherwise (older archive).
    """
    seen = {}  # key -> (n_seeds, bundle, latest_gen)
    for n in range(gen + 1):
        gdata = run_data["generations"][n]
        for b in list(gdata["population"]) + list(gdata["offspring"]):
            ns = int(b.get("seeds_evaluated") or 0)
            k = _bundle_key(b)
            cur = seen.get(k)
            if cur is None or cur[0] < ns:
                seen[k] = (ns, b, n)
    # Tag membership in the current gen's pop/offspring lists.
    cur_g = run_data["generations"][gen]
    pop_keys = {_bundle_key(b) for b in cur_g["population"]}
    off_keys = {_bundle_key(b) for b in cur_g["offspring"]}

    bundles = []
    mu_l, N_l, labels = [], [], []
    for key, (_ns, b, _gn) in seen.items():
        bundles.append(b)
        mu_l.append(float(b["score"]))
        N_l.append(float(b["seeds_evaluated"]))
        if key in off_keys:
            kind = "off"
        elif key in pop_keys:
            kind = "pop"
        else:
            kind = "arc"
        name = "|".join(b["operators"][s]["name"] for s in sorted(b["operators"]))
        labels.append((kind, name))
    return (np.asarray(mu_l, dtype=float),
            np.asarray(N_l, dtype=float),
            labels, bundles)


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "666286"
    gen = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    data = json.loads((Path("runs") / job / "run_data.json").read_text())
    mu, N, labels, bundles = load_arms(data, gen)
    sigma = estimate_sigma(bundles)
    print(f"=== {job} generation {gen} ===")
    print(f"k = {len(mu)} arms  (pop={sum(1 for k,_ in labels if k=='pop')}, "
          f"off={sum(1 for k,_ in labels if k=='off')})")
    print(f"estimated sigma_per_seed = {sigma:.4f}")
    print(f"N range: {int(N.min())}..{int(N.max())}   mu range: "
          f"{mu.min():.3f}..{mu.max():.3f}")

    alpha = thompson_sampling_select_probs(mu, sigma, N)
    psi = top_two_thompson_sampling_select_probs(mu, sigma, N, beta=0.5)
    baseline = float(alpha @ mu)
    print(f"\nbaseline E[top-arm mean] under current posterior = {baseline:.4f}")

    print(f"\n{'idx':>3} {'kind':>4} {'N':>3} {'mu':>7} {'alpha':>7} {'psi':>7}  name")
    order = np.argsort(-alpha)
    for rank, i in enumerate(order):
        kind, name = labels[i]
        marker = "*" if rank == 0 else " "
        print(f"{marker}{i:>3} {kind:>4} {int(N[i]):>3} {mu[i]:>7.3f} "
              f"{alpha[i]:>7.3f} {psi[i]:>7.3f}  {name[:60]}")

    # Reeval improvement curve: how much does the expected top-arm mean grow
    # if we spend B extra evaluations distributed via TTTS?
    B_max = 40
    M = 5000
    print(f"\nRunning simulate_reeval_expected_improvement "
          f"(M={M}, B_max={B_max})...")
    rng = np.random.default_rng(0)
    batch_selection_fn = thompson_sampling_batch_selection_fn(M=10_000, rng=rng)
    curve = simulate_reeval_expected_improvement(
        mu, sigma, N, batch_selection_fn, M=M, B_max=B_max, rng=rng,
    )
    print(f"  EI[B=1]  = {curve[1]:+.5f}")
    print(f"  EI[B=5]  = {curve[5]:+.5f}")
    print(f"  EI[B=10] = {curve[10]:+.5f}")
    print(f"  EI[B=20] = {curve[20]:+.5f}")
    print(f"  EI[B=40] = {curve[40]:+.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    idx = np.arange(len(mu))
    ax.bar(idx - 0.2, alpha, width=0.4, label="alpha (TS)", color="C0")
    ax.bar(idx + 0.2, psi, width=0.4, label="psi (TTTS, β=0.5)", color="C3")
    for i in idx:
        kind, _ = labels[i]
        ax.text(i, -0.02, kind, ha="center", va="top", fontsize=7, color="gray")
    ax.set_xlabel("arm index")
    ax.set_ylabel("selection probability")
    ax.set_title(f"{job} gen {gen} — TS vs TTTS over {len(mu)} arms\n"
                 f"σ_per_seed={sigma:.3f}")
    ax.set_xticks(idx)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(np.arange(0, len(curve)), curve, "o-", color="C2", markersize=4)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_xlabel("reeval budget B")
    ax.set_ylabel("E[top-arm-mean] − baseline")
    ax.set_title("Expected improvement from B re-evaluations (TTTS allocation)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path("plots") / f"{job}_gen{gen}_monte_carlo.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
