import matplotlib.pyplot as plt
import numpy as np

datasets = ["Antivirus", "Web Activity", "Middleware", "Storm"]

f1_200 = [0.051, 0.286, 0.364, 0.222]
f1_500 = [0.103, 0.286, 0.364, 0.222]

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars_200 = ax.bar(x - width / 2, f1_200, width, label="200 clauses")
bars_500 = ax.bar(x + width / 2, f1_500, width, label="500 clauses")

ax.bar_label(bars_200, fmt="%.3f", padding=3)
ax.bar_label(bars_500, fmt="%.3f", padding=3)

ax.set_xlabel("Dataset")
ax.set_ylabel("Orientation F1-score")
ax.set_title("Effect of Tsetlin Machine clause count on orientation F1-score")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 0.45)
ax.legend()

fig.tight_layout()

plt.savefig(
    "images/clause_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()