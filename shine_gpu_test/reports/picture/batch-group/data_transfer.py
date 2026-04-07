import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [1, 4, 8, 16, 32, 64]
ops = [16000, 4000, 4000, 4000, 2000, 1000]

raw_data = {
    "d2h_bytes": [
        2056134120, 1938664556, 4309558248, 9992470764, 9976772452, 10006240964
    ],
    "h2d_bytes": [
        1540732547008, 1522027783248, 3462599466520, 8287408221608, 8287138422064, 8315151754336
    ],
    "rdma_read_bytes": [
        1542608666258, 1523341643090, 3463551239562, 8286910871176, 8286378874810, 8314042684944
    ],
    "rdma_write_bytes": [
        1102092967, 1107277443, 2298209105, 4704298736, 4707433824, 4721871566
    ],
}

normalized_ops = 1000

# bytes/op
per_op = {}
for key, values in raw_data.items():
    per_op[key] = [v / op for v, op in zip(values, ops)]

# normalized bytes
normalized = {}
for key, values in per_op.items():
    normalized[key] = [v * normalized_ops for v in values]

# 转成 GB
per_op_gb = {k: [x / 1e9 for x in v] for k, v in per_op.items()}
normalized_gb = {k: [x / 1e9 for x in v] for k, v in normalized.items()}

x = np.arange(len(batch_sizes))

# -------------------------
# Fig 1: bytes/op line plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 6))

for key, values in per_op_gb.items():
    ax.plot(batch_sizes, values, marker='o', label=key)

ax.set_title("Data Transfer per Operation", fontsize=14)
ax.set_xlabel("Batch Size", fontsize=12)
ax.set_ylabel("Transfer Volume per Op (GB)", fontsize=12)
ax.set_yscale("log")
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("normalized_transfer_grouped.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------
# Fig 2: normalized grouped bar
# -------------------------
fig, ax = plt.subplots(figsize=(13, 6))

metrics = list(normalized_gb.keys())
width = 0.2

for i, metric in enumerate(metrics):
    ax.bar(
        x + (i - 1.5) * width,
        normalized_gb[metric],
        width,
        label=metric
    )

ax.set_title(f"Normalized Data Transfer per {normalized_ops} Ops", fontsize=14)
ax.set_xlabel("Batch Size", fontsize=12)
ax.set_ylabel("Normalized Transfer Volume (GB)", fontsize=12)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in batch_sizes])
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("normalized_transfer_stacked.png", dpi=300, bbox_inches="tight")
plt.show()