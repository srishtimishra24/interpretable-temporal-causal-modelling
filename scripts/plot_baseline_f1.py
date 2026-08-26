import matplotlib.pyplot as plt
import numpy as np

datasets = ["Antivirus", "Web Activity", "Middleware", "Storm"]

methods = [
    "PCMCI",
    "CBNB-e",
    "NBCB-e",
    "GCMVL",
    "DYNOTEARS",
    "Proposed TM"
]

values = {
    "PCMCI": [0.160, 0.400, 0.000, 0.000],
    "CBNB-e": [0.391, 0.286, 0.125, 0.125],
    "NBCB-e": [0.327, 0.293, 0.235, 0.222],
    "GCMVL": [0.083, 0.100, 0.000, 0.143],
    "DYNOTEARS": [0.190, 0.261, 0.429, 0.143],
    "Proposed TM": [0.042, 0.390, 0.452, 0.222],
}

x = np.arange(len(datasets))
width = 0.13

fig, ax = plt.subplots(figsize=(11, 6))

for i, method in enumerate(methods):
    offset = (i - (len(methods) - 1) / 2) * width

    bars = ax.bar(
        x + offset,
        values[method],
        width,
        label=method
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=2,
        fontsize=7
    )

ax.set_ylabel("Orientation F1-score")
ax.set_xlabel("Dataset")
ax.set_title("Orientation F1-score comparison across IT monitoring datasets")

ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 0.5)

ax.legend(ncol=3)

fig.tight_layout()

plt.savefig(
    "images/baseline_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()