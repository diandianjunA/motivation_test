import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [1, 4, 8, 16, 32, 64]

insert_queue_wait = [48.40, 81.45, 89.86, 94.85, 97.60, 98.90]
insert_service    = [51.60, 18.55, 10.14, 5.15, 2.40, 1.10]

query_queue_wait = [0.02, 0.03, 0.03, 0.03, 0.03, 0.04]
query_service    = [99.98, 99.97, 99.97, 99.97, 99.97, 99.96]


def draw_stacked_bar(ax, batches, queue_wait, service, title):
    x = np.arange(len(batches))
    width = 0.6

    ax.bar(x, queue_wait, width, label="queue_wait")
    ax.bar(x, service, width, bottom=queue_wait, label="service")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)

    for i, (qw, sv) in enumerate(zip(queue_wait, service)):
        # queue_wait
        if qw >= 5:
            ax.text(i, qw / 2, f"{qw:.2f}%", ha='center', va='center', fontsize=9)
        else:
            ax.text(i, qw + 1.0, f"{qw:.2f}%", ha='center', va='bottom', fontsize=8)

        # service
        if sv >= 5:
            ax.text(i, qw + sv / 2, f"{sv:.2f}%", ha='center', va='center', fontsize=9)
        else:
            ax.text(i, qw + sv + 1.0, f"{sv:.2f}%", ha='center', va='bottom', fontsize=8)


fig, axes = plt.subplots(2, 1, figsize=(9, 10))

draw_stacked_bar(
    axes[0],
    batch_sizes,
    insert_queue_wait,
    insert_service,
    "Insert Breakdown"
)

draw_stacked_bar(
    axes[1],
    batch_sizes,
    query_queue_wait,
    query_service,
    "Query Breakdown"
)

plt.tight_layout()
plt.savefig("breakdown_stacked_bar.png", dpi=300, bbox_inches="tight")
plt.show()