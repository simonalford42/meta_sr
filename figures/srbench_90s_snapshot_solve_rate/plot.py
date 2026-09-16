"""Compare cumulative symbolic recovery at scheduled SRBench snapshots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = [
    ("Baseline", "srbench_gt_baseline_9-15_1seed_90s_snap10", "#3975b7"),
    ("Evolved 709715", "709715-srbench_gt_9-14_1seed_90s_snap10", "#db7825"),
]

fig, ax = plt.subplots(figsize=(7.2, 4.5))
times = list(range(10, 91, 10))
for label, run, color in RUNS:
    records = json.loads((ROOT / "runs" / run / "snapshot_solve_times.json").read_text())["records"]
    rates = [100 * sum(any(
        o.get("matched_equation") and not o.get("final")
        and o.get("scheduled_seconds") is not None
        and o["scheduled_seconds"] <= t
        for o in r["observations"]
    ) for r in records) / len(records) for t in times]
    ax.plot(times, rates, marker="o", linewidth=2.3, markersize=5, color=color, label=label)
    ax.annotate(f"{rates[-1]:.1f}%", (90, rates[-1]), xytext=(7, 0),
                textcoords="offset points", va="center", color=color, weight="bold")

ax.set(title="SRBench: solve rate over time", xlabel="Scheduled snapshot time (s, from fit startup)",
       ylabel="Cumulative solve rate (%)", xlim=(7, 101), ylim=(0, 66))
ax.set_xticks(times)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False, loc="upper left")
fig.text(0.5, 0.025, "130 problems · 1 seed · no noise · symbolic equivalence · final frontiers excluded",
         ha="center", fontsize=9, color="#555555")
fig.tight_layout(rect=(0, 0.055, 1, 1))
for suffix in ("png", "pdf"):
    fig.savefig(OUT / f"solve_rate.{suffix}", dpi=200, bbox_inches="tight")
