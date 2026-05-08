"""Tournament selection: P(rank i chosen) for N=10, table + zoomed plot."""
import numpy as np
import matplotlib.pyplot as plt
from math import comb

N = 10
ranks = np.arange(1, N + 1)
ns = [2, 3, 4, 5]


def tournament_prob(N, n):
    return np.array([comb(N - i, n - 1) / comb(N, n) if (N - i) >= (n - 1) else 0.0
                     for i in ranks])


# --- Table ---
print(f"P(rank i chosen) under tournament selection (population N={N})")
print()
header = "  n  | " + " ".join(f"i={i:>5}" for i in ranks)
print(header)
print("-" * len(header))
probs = {}
for n in ns:
    p = tournament_prob(N, n)
    probs[n] = p
    row = f"  {n}  | " + " ".join(f"{v:7.4f}" for v in p)
    print(row)

# --- Plot, zoomed y-axis ---
fig, ax = plt.subplots(figsize=(7, 4.5))
for n in ns:
    ax.plot(ranks, probs[n], marker="o", label=f"n={n}")
ax.set_xticks(ranks)
ax.set_xlabel("rank (1 = best)")
ax.set_ylabel("P(selected)")
ax.set_title(f"Tournament selection probability by rank (N={N})")
ax.set_ylim(0, None)
ax.axhline(1.0 / N, ls="--", color="gray", lw=0.8, label=f"uniform (1/N={1/N:.2f})")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = "plots/tournament_N10.png"
fig.savefig(out, dpi=140)
print(f"\nsaved {out}")
