import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [1, 4, 8, 16, 32, 64]

data = {
    "neighbor_update_ns":         [66.31, 69.77, 69.43, 68.56, 68.02, 69.22],
    "gpu_distance_ns":            [8.51,  7.77,  7.85,  7.95,  8.05,  7.78],
    "candidate_vector_fetch_ns":  [7.55,  6.01,  6.44,  6.95,  7.16,  6.67],
    "candidate_search_ns":        [6.05,  4.81,  4.80,  4.87,  5.07,  4.64],
    "gpu_prune_ns":               [5.85,  6.17,  6.08,  5.93,  5.90,  5.97],
    "unattributed_ns":            [5.61,  5.33,  5.28,  5.62,  5.69,  5.60],
    "quantize_ns":                [0.10,  0.11,  0.10,  0.09,  0.09,  0.09],
    "medoid_fetch_ns":            [0.02,  0.01,  0.01,  0.01,  0.01,  0.01],
    "remote_alloc_ns":            [0.01,  0.01,  0.01,  0.01,  0.01,  0.01],
    "new_node_write_ns":          [0.01,  0.01,  0.01,  0.01,  0.01,  0.01],
    "medoid_update_ns":           [0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
}

phase_order = [
    "neighbor_update_ns",
    "gpu_distance_ns",
    "candidate_vector_fetch_ns",
    "candidate_search_ns",
    "gpu_prune_ns",
    "unattributed_ns",
    "quantize_ns",
    "medoid_fetch_ns",
    "remote_alloc_ns",
    "new_node_write_ns",
    "medoid_update_ns",
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

ax.set_title("Insert Breakdown by Batch Size", fontsize=15)
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
plt.savefig("insert_breakdown_stacked.png", dpi=300, bbox_inches="tight")
plt.show()