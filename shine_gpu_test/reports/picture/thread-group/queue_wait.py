import matplotlib.pyplot as plt
import numpy as np

thread_sizes = [1, 2, 4, 8, 16, 32, 64]

insert_queue_wait = [75.48, 89.16, 91.46, 97.28, 98.89, 99.51, 99.69]
insert_service    = [24.52, 10.84,  8.54,  2.72,  1.11,  0.59, 0.31]

query_queue_wait = [0.02, 0.01, 0.04, 0.06, 0.05, 5.54, 43.54]
query_service    = [99.98, 99.99, 99.96, 99.94, 99.95, 94.46, 56.46]


def draw_stacked_bar(ax, x_labels, queue_wait, service, title):
    x = np.arange(len(x_labels))
    width = 0.65

    queue_wait = np.array(queue_wait, dtype=float)
    service = np.array(service, dtype=float)

    ax.bar(x, queue_wait, width, label="queue_wait")
    ax.bar(x, service, width, bottom=queue_wait, label="service")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Client Threads", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_labels])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    for i, (qw, sv) in enumerate(zip(queue_wait, service)):
        if np.isnan(qw) or np.isnan(sv):
            ax.text(i, 50, "N/A", ha="center", va="center", fontsize=10)
            continue

        if qw >= 5:
            ax.text(i, qw / 2, f"{qw:.2f}%", ha="center", va="center", fontsize=9)
        else:
            ax.text(i, qw + 1.0, f"{qw:.2f}%", ha="center", va="bottom", fontsize=8)

        if sv >= 5:
            ax.text(i, qw + sv / 2, f"{sv:.2f}%", ha="center", va="center", fontsize=9)
        else:
            ax.text(i, qw + sv + 1.0, f"{sv:.2f}%", ha="center", va="bottom", fontsize=8)


fig, axes = plt.subplots(2, 1, figsize=(9, 10))

draw_stacked_bar(
    axes[0],
    thread_sizes,
    insert_queue_wait,
    insert_service,
    "Insert Breakdown by Client Threads"
)

draw_stacked_bar(
    axes[1],
    thread_sizes,
    query_queue_wait,
    query_service,
    "Query Breakdown by Client Threads"
)

plt.tight_layout()
plt.savefig("thread_breakdown_stacked.png", dpi=300, bbox_inches="tight")
plt.show()