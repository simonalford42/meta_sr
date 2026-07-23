"""Is top-k churn a less noisy signal of offspring contribution than the
planner's offspring-EI estimate?

For each oracle run (568245/568246 n10 pair + 538190), replay the n1
observation stream and compute per generation:
  - offspring_EI: the smart-reeval planner's estimate (K=3 window of 1-seed
    offspring scores, analytic offspring_expected_improvement, pooled sigma)
  - churn: 1 - |observed top-10(g) ∩ observed top-10(g-1)| / 10
  - young_frac: fraction of observed top-10 born within the last 3 gens
    (the age-based retention signal, inverted)
  - true_delta: per-gen improvement of the TRUE frontier = Δ mean oracle score
    of the true top-10 among arms born <= g  (ground truth both signals proxy)

Outputs plots/oracle_replay/churn_vs_offspring_ei.png plus a metric table:
  - corr with true_delta (does the signal track reality?)
  - roughness = std(diff(x)) / std(x)  (lower = smoother)
  - cross-run reproducibility corr(568245, 568246) per signal
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path("/home/sca63/meta_sr")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from monte_carlo import topk_tourney_batch_selection_fn
from offspring_mc import offspring_expected_improvement
import oracle_replay as orp

SIGMA = orp.POOLED_SIGMA
TOPK = 10
K_WIN = 3


def signals_for_run(rid: int):
    rec = orp.cached_bundle_records(rid)["records"]
    orp._prep(rec)
    births = sorted({r["birth_gen"] for r in rec})
    by_birth = {}
    for i, r in enumerate(rec):
        by_birth.setdefault(r["birth_gen"], []).append(i)

    sel_fn = topk_tourney_batch_selection_fn(topk=TOPK, n=2)
    rng = np.random.default_rng(0)

    arch = []
    prev_top = None
    off_window = {}
    out = {"gen": [], "ei": [], "churn": [], "young": [], "true_delta": []}
    prev_frontier = None

    for g in births:
        arch.extend(by_birth[g])
        if g >= 1:
            off_window[g] = [rec[i]["seed_scores"][0] for i in by_birth[g]]

        mu_obs = np.array([rec[i]["seed_scores"][0] for i in arch])
        oracle = np.array([rec[i]["oracle_mean"] for i in arch])
        births_arr = np.array([rec[i]["birth_gen"] for i in arch])

        k = min(TOPK, mu_obs.size)
        top_idx = np.argsort(-mu_obs)[:k]
        top_set = {arch[j] for j in top_idx}
        churn = (1.0 - len(top_set & prev_top) / k) if prev_top is not None else np.nan
        prev_top = top_set
        young = float((g - births_arr[top_idx] < 3).mean())

        frontier = float(np.sort(oracle)[-k:].mean())
        true_delta = frontier - prev_frontier if prev_frontier is not None else np.nan
        prev_frontier = frontier

        emp = []
        for gg in range(g - K_WIN + 1, g + 1):
            emp.extend(off_window.get(gg, []))
        ei = np.nan
        if emp:
            res = offspring_expected_improvement(
                pop_mu=mu_obs, pop_N=np.ones_like(mu_obs),
                offspring_empirical=np.array(emp, dtype=float),
                sigma=SIGMA, n_initial_evals=1,
                batch_selection_fn=sel_fn, M_total=None, rng=rng,
            )
            if res is not None:
                ei = float(res["improvement"])

        out["gen"].append(g)
        out["ei"].append(ei)
        out["churn"].append(churn)
        out["young"].append(young)
        out["true_delta"].append(true_delta)

    return {k: np.array(v, dtype=float) for k, v in out.items()}


def _valid(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def corr(a, b):
    a, b = _valid(a, b)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def roughness(x):
    x = x[np.isfinite(x)]
    if x.size < 3 or x.std() == 0:
        return np.nan
    return float(np.std(np.diff(x)) / x.std())


def main():
    runs = [568245, 568246, 538190]
    sigs = {rid: signals_for_run(rid) for rid in runs}

    fig, axes = plt.subplots(2, 3, figsize=(19, 9), sharex="col")
    for c, rid in enumerate(runs):
        s = sigs[rid]
        ax = axes[0][c]
        ax.plot(s["gen"], s["ei"], "o-", color="C0", label="offspring EI (planner)")
        ax.plot(s["gen"], s["true_delta"], "s--", color="k", alpha=0.7,
                label="true frontier Δ/gen (oracle)")
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_title(f"{rid} — score-scale signals")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_ylabel("Δ score")

        ax = axes[1][c]
        ax.plot(s["gen"], s["churn"], "o-", color="C3", label="top-10 churn (1-gen)")
        ax.plot(s["gen"], s["young"], "^-", color="C2",
                label="young frac of top-10 (<3 gens)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("generation")
        ax.set_ylabel("fraction")
        ax.set_title(f"{rid} — churn signals (observed, n1)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Offspring-contribution signals: planner EI vs top-k churn "
                 "(all under n1 observation)", fontsize=13)
    fig.tight_layout()
    out = REPO / "plots" / "oracle_replay" / "churn_vs_offspring_ei.png"
    fig.savefig(out, dpi=130)
    print(f"saved: {out}\n")

    print(f"{'run':>8} {'signal':>12} {'corr(true Δ)':>13} {'roughness':>10}")
    for rid in runs:
        s = sigs[rid]
        for name in ("ei", "churn", "young"):
            print(f"{rid:>8} {name:>12} {corr(s[name], s['true_delta']):>13.3f} "
                  f"{roughness(s[name]):>10.3f}")

    a, b = sigs[568245], sigs[568246]
    n = min(len(a["gen"]), len(b["gen"]))
    print("\ncross-run reproducibility corr(568245, 568246) on shared gens:")
    for name in ("ei", "churn", "young", "true_delta"):
        print(f"  {name:>12}: {corr(a[name][:n], b[name][:n]):+.3f}")


if __name__ == "__main__":
    main()
