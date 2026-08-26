import matplotlib.pyplot as plt
import numpy as np

datasets = ["Antivirus", "Web Activity", "Middleware", "Storm"]

time_200 = [42.4, 405.3, 14.1, 33.5]
time_500 = [86.9, 766.1, 14.2, 69.9]

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars_200 = ax.bar(x - width / 2, time_200, width, label="200 clauses")
bars_500 = ax.bar(x + width / 2, time_500, width, label="500 clauses")

ax.bar_label(bars_200, fmt="%.1f", padding=3)
ax.bar_label(bars_500, fmt="%.1f", padding=3)

ax.set_xlabel("Dataset")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("Effect of Tsetlin Machine clause count on runtime")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()

fig.tight_layout()

plt.savefig(
    "images/clause_runtime_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()