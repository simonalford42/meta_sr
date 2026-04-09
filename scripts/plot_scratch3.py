import re
import matplotlib.pyplot as plt
import numpy as np

timestamps = []
avg_gt_values = []

with open("out/scratch3.txt") as f:
    for line in f:
        m = re.search(r"avg_gt=([\d.]+)", line)
        if m:
            avg_gt_values.append(float(m.group(1)))

indices = np.arange(1, len(avg_gt_values) + 1)
running_max = np.maximum.accumulate(avg_gt_values)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(indices, avg_gt_values, s=15, alpha=0.6, label="avg_gt")
ax.plot(indices, running_max, color="red", linewidth=1.5, label="running max")
ax.set_xlabel("Evaluation index")
ax.set_ylabel("avg_gt")
ax.set_title("avg_gt over time (scratch3.txt)")
ax.legend()
plt.tight_layout()
plt.savefig("out/scratch3_avg_gt.png", dpi=150)
print(f"Saved to out/scratch3_avg_gt.png ({len(avg_gt_values)} evals)")
