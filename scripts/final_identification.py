"""Final best-arm identification on the oracle archives.

Setup: take the FULL final archive of an n10 oracle run (282 bundles), but
observed at only n_base seeds/bundle (n1 or n3 start). Spend a one-shot budget
B revealing extra seeds (max 10/bundle), then pick argmax observed mean.
Regret = oracle_max - oracle(chosen). Pair-averaged over 568245/568246
(538190 excluded: most bundles have only 3 recorded seeds, caps bind).

Strategies:
  TTTS          - B draws from the top-two Thompson distribution (pooled sigma)
  tourney       - B draws from top-10 tournament(n=2) selection probs
  uniform-top{5,10,20} - even split over the observed top-k
  waterfill-10  - sequentially give a seed to the top-10 member w/ fewest seeds
  SH-{16,32}    - successive halving: start with observed top-n0, equal seeds
                  per round, keep top half each round (adaptive, multi-round)

Outputs plots/oracle_replay/final_identification.png + table.
Usage: python scripts/final_identification.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path("/home/sca63/meta_sr")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from monte_carlo import (
    top_two_thompson_sampling_select_probs,
    batch_topk_tourney_probs,
)
import oracle_replay as orp

SIGMA = orp.POOLED_SIGMA
PAIR = [568245, 568246]
B_LIST = [0, 10, 20, 40, 80, 120, 180, 270]
N_REPS = 20


class Arms:
    """Seed-reveal state over the full final archive."""

    def __init__(self, records, n_base):
        self.scores = [r["seed_scores"] for r in records]   # each [10]
        self.oracle = np.array([r["oracle_mean"] for r in records])
        self.nmax = np.array([r["nseeds"] for r in records])
        self.N = np.minimum(n_base, self.nmax).astype(int)
        self.spent = 0

    def mu(self):
        return np.array([s[:n].mean() for s, n in zip(self.scores, self.N)])

    def reveal(self, i, want):
        take = int(min(want, self.nmax[i] - self.N[i]))
        self.N[i] += take
        self.spent += take
        return take

    def regret(self, pick="raw"):
        mu = self.mu()
        if pick == "shrunk":
            # EB-shrunk argmax: lucky low-N estimates get discounted, so honest
            # verified scores aren't outbid by unverified 1-seed liars.
            mu = orp._eb_shrink(mu, self.N.astype(float), SIGMA)
        return float(self.oracle.max() - self.oracle[int(np.argmax(mu))])


def alloc_probs_counts(arms, B, probs, rng):
    k = probs.size
    p = probs / probs.sum()
    picks = rng.choice(k, size=B, p=p)
    for i in np.bincount(picks, minlength=k).nonzero()[0]:
        arms.reveal(i, int((picks == i).sum()))


def run_strategy(records, n_base, strat, B, rng):
    arms = Arms(records, n_base)
    if B == 0:
        return arms.regret("raw"), arms.regret("shrunk")
    mu = arms.mu()
    if strat == "TTTS":
        psi = top_two_thompson_sampling_select_probs(mu, SIGMA, arms.N.astype(float))
        alloc_probs_counts(arms, B, psi, rng)
    elif strat == "tourney":
        p = batch_topk_tourney_probs(mu[None], k=10, n=2)[0]
        alloc_probs_counts(arms, B, p, rng)
    elif strat.startswith("uniform-top"):
        k = int(strat.split("top")[1])
        top = np.argsort(-mu)[:k]
        base, rem = B // k, B % k
        for i in top:
            arms.reveal(i, base)
        for i in rng.choice(top, size=rem, replace=False):
            arms.reveal(i, 1)
    elif strat == "waterfill-10":
        for _ in range(B):
            mu = arms.mu()
            top = np.argsort(-mu)[:10]
            room = top[arms.N[top] < arms.nmax[top]]
            if room.size == 0:
                break
            arms.reveal(int(room[np.argmin(arms.N[room])]), 1)
    elif strat.startswith("SH-"):
        n0 = int(strat.split("-")[1])
        cand = list(np.argsort(-arms.mu())[:n0])
        rounds = max(1, int(np.ceil(np.log2(n0))))
        for r in range(rounds):
            if len(cand) <= 1:
                break
            per = (B - arms.spent) // (len(cand) * (rounds - r)) if rounds - r > 0 else 0
            if per > 0:
                for i in cand:
                    arms.reveal(i, per)
            else:
                # not enough budget for a full round: one seed each while it lasts
                for i in cand:
                    if arms.spent >= B:
                        break
                    arms.reveal(i, 1)
            mu = arms.mu()
            cand.sort(key=lambda i: -mu[i])
            cand = cand[: max(1, len(cand) // 2)]
    else:
        raise ValueError(strat)
    return arms.regret("raw"), arms.regret("shrunk")


STRATS = [
    ("TTTS",          "C4"),
    ("tourney",       "C3"),
    ("uniform-top5",  "C8"),
    ("uniform-top10", "C2"),
    ("uniform-top20", "C6"),
    ("waterfill-10",  "C5"),
    ("SH-16",         "C0"),
    ("SH-32",         "C1"),
]


def main():
    recs = {rid: orp.cached_bundle_records(rid)["records"] for rid in PAIR}
    for r in recs.values():
        orp._prep(r)

    results = {}  # (n_base, strat, pick) -> [regret per B]
    for n_base in (1, 3):
        for strat, _ in STRATS:
            raw_c, shr_c = [], []
            for B in B_LIST:
                raw_v, shr_v = [], []
                for rid in PAIR:
                    for rep in range(N_REPS):
                        rng = np.random.default_rng(31000 + 97 * rep)
                        r, s = run_strategy(recs[rid], n_base, strat, B, rng)
                        raw_v.append(r); shr_v.append(s)
                raw_c.append(float(np.mean(raw_v)))
                shr_c.append(float(np.mean(shr_v)))
            results[(n_base, strat, "raw")] = raw_c
            results[(n_base, strat, "shrunk")] = shr_c
            print(f"n{n_base} {strat:>13} raw   : " + " ".join(f"{v:.4f}" for v in raw_c), flush=True)
            print(f"n{n_base} {strat:>13} shrunk: " + " ".join(f"{v:.4f}" for v in shr_c), flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(17, 12), sharey="row", sharex=True)
    for col, n_base in enumerate((1, 3)):
        for row, pick in enumerate(("raw", "shrunk")):
            ax = axes[row][col]
            for strat, color in STRATS:
                ax.plot(B_LIST, results[(n_base, strat, pick)], "o-", color=color,
                        label=strat, markersize=4, linewidth=1.6)
            ax.set_title(f"start n{n_base}, pick = argmax {pick}")
            if row == 1:
                ax.set_xlabel("identification budget B (extra seeds)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    axes[0][0].set_ylabel("regret"); axes[1][0].set_ylabel("regret = oracle(best) − oracle(argmax observed)")
    fig.suptitle("Final best-arm identification on the full oracle archives "
                 "(pair avg 568245+568246, 20 reps)", fontsize=13)
    fig.tight_layout()
    out = REPO / "plots" / "oracle_replay" / "final_identification.png"
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
