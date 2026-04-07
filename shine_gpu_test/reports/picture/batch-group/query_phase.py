import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [1, 4, 8, 16, 32, 64]

data = {
    "gpu_distance_ns":       [37.59, 39.50, 36.31, 35.60, 33.04, 33.20],
    "rabitq_fetch_ns":       [26.35, 24.31, 28.31, 29.06, 32.49, 31.85],
    "unattributed_ns":       [29.11, 29.43, 28.48, 28.88, 27.68, 27.52],
    "neighbor_fetch_ns":     [5.55,  5.29,  5.20,  4.79,  5.07,  5.64],
    "vector_fetch_ns":       [0.54,  0.49,  0.60,  0.54,  0.66,  0.61],
    "gpu_rerank_ns":         [0.34,  0.35,  0.33,  0.32,  0.27,  0.31],
    "gpu_prepare_ns":        [0.26,  0.30,  0.29,  0.30,  0.28,  0.32],
    "result_materialize_ns": [0.19,  0.26,  0.41,  0.45,  0.43,  0.46],
    "medoid_fetch_ns":       [0.03,  0.04,  0.05,  0.05,  0.06,  0.07],
    "cpu_merge_sort_ns":     [0.02,  0.03,  0.02,  0.02,  0.02,  0.02],
}

phase_order = [
    "gpu_distance_ns",
    "rabitq_fetch_ns",
    "unattributed_ns",
    "neighbor_fetch_ns",
    "vector_fetch_ns",
    "gpu_rerank_ns",
    "gpu_prepare_ns",
    "result_materialize_ns",
    "medoid_fetch_ns",
    "cpu_merge_sort_ns",
]

x = np.arange(len(batch_sizes))
width = 0.68

fig, ax = plt.subplots(figsize=(13, 7))
bottom = np.zeros(len(batch_sizes))

for phase in phase_order:
    values = np.array(data[phase])
    ax.bar(x, values, width, bottom=bottom, label=phase)

    for i, v in enumerate(values):
        if v >= 4.0:
            ax.text(
                x[i],
                bottom[i] + v / 2,
                f"{v:.2f}%",
                ha="center",
                va="center",
                fontsize=9
            )

    bottom += values

ax.set_title("Query Breakdown by Batch Size", fontsize=15)
ax.set_xlabel("Batch Size", fontsize=12)
ax.set_ylabel("Percentage of Phase Sum (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=11)
ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=9,
    frameon=True
)

plt.tight_layout()
plt.savefig("query_breakdown_stacked.png", dpi=300, bbox_inches="tight")
plt.show()