# -*- coding: utf-8 -*-
"""
从 CSV 绘制 3 类图：
1) 绝对内存占用图（上下两个子图，避免高值压扁低值）
2) 边际内存成本图（每增加 1M 向量带来的新增内存，单位 GB / 1M vectors）
3) 相对增长倍数图（相对于 1M 的倍数）

输入文件默认：
    索引内存占用2.csv

输出文件：
    memory_absolute_split.png
    memory_marginal_cost.png
    memory_growth_factor.png
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# 可按需要修改的配置
# =========================
CSV_PATH = "memory_usage.csv"

# 如果你想固定上图只显示这两个高内存方法，就保留默认；
# 如果实际方法名不同，可以改成你的名字。
TOP_GROUP = {"DiskANN", "HNSW"}

# 字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_size_label(col_name: str):
    """
    从列名中提取数值规模和显示标签
    例如:
        1024dim1M  -> (1.0, "1M")
        5M         -> (5.0, "5M")
        10M        -> (10.0, "10M")
    """
    m = re.search(r"(\d+(?:\.\d+)?)M$", str(col_name))
    if m:
        val = float(m.group(1))
        return val, f"{m.group(1)}M"
    return None, str(col_name)


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Index"})

    size_cols = [c for c in df.columns if c != "Index"]

    size_info = []
    for c in size_cols:
        x_val, x_label = parse_size_label(c)
        if x_val is not None:
            size_info.append((c, x_val, x_label))

    size_info = sorted(size_info, key=lambda x: x[1])

    ordered_cols = ["Index"] + [c for c, _, _ in size_info]
    x_vals = [x for _, x, _ in size_info]       # 真实横轴比例，例如 [1, 5, 10, 50]
    x_labels = [lab for _, _, lab in size_info] # 显示标签，例如 ["1M", "5M", "10M", "50M"]

    df = df[ordered_cols].copy()

    for c in ordered_cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, ordered_cols, x_vals, x_labels


def draw_absolute_split(df, ordered_cols, x_vals, x_labels, output_name="memory_absolute_split.png"):
    """
    图1：绝对内存占用图
    采用上下两个子图，避免高值方法压扁低值方法。
    """
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    for _, row in df.iterrows():
        idx = str(row["Index"])
        y_all = row[ordered_cols[1:]].tolist()
        valid_points = [(x, y) for x, y in zip(x_vals, y_all) if pd.notna(y)]
        if not valid_points:
            continue

        xs = [p[0] for p in valid_points]
        ys = [p[1] for p in valid_points]

        if idx in TOP_GROUP:
            ax_top.plot(xs, ys, marker="o", linewidth=2.2, label=idx)
        else:
            ax_bottom.plot(xs, ys, marker="o", linewidth=2.2, label=idx)

    ax_top.set_title("Absolute Memory Usage Across Data Scale")
    ax_top.set_ylabel("Memory Usage (GB)")
    ax_bottom.set_ylabel("Memory Usage (GB)")
    ax_bottom.set_xlabel("Data Scale")

    ax_top.grid(True, linestyle="--", alpha=0.3)
    ax_bottom.grid(True, linestyle="--", alpha=0.3)

    ax_top.legend(frameon=False)
    ax_bottom.legend(frameon=False)

    ax_bottom.set_xticks(x_vals)
    ax_bottom.set_xticklabels(x_labels)

    plt.tight_layout()
    plt.savefig(output_name, dpi=220, bbox_inches="tight")
    plt.close()


def compute_marginal_cost(df, ordered_cols, x_vals):
    """
    计算每个相邻区间的边际内存成本：
        (M2 - M1) / (N2 - N1)
    单位：GB / 1M vectors
    """
    interval_labels = []
    interval_centers = []
    for i in range(len(x_vals) - 1):
        left = x_vals[i]
        right = x_vals[i + 1]
        interval_labels.append(f"{int(left)}M→{int(right)}M" if left.is_integer() and right.is_integer()
                               else f"{left}M→{right}M")
        interval_centers.append((left + right) / 2)

    rows = []

    for _, row in df.iterrows():
        idx = str(row["Index"])
        vals = row[ordered_cols[1:]].tolist()

        marginals = []
        for i in range(len(vals) - 1):
            y1, y2 = vals[i], vals[i + 1]
            x1, x2 = x_vals[i], x_vals[i + 1]
            if pd.notna(y1) and pd.notna(y2):
                slope = (y2 - y1) / (x2 - x1)  # GB / 1M vectors
            else:
                slope = np.nan
            marginals.append(slope)

        rows.append([idx] + marginals)

    marginal_df = pd.DataFrame(rows, columns=["Index"] + interval_labels)
    return marginal_df, interval_labels


def draw_marginal_cost(marginal_df, interval_labels, output_name="memory_marginal_cost.png"):
    """
    图2：边际内存成本图
    分组柱状图，每组一个区间，每根柱代表一个索引。
    """
    methods = marginal_df["Index"].tolist()
    n_methods = len(methods)
    n_groups = len(interval_labels)

    x = np.arange(n_groups)
    bar_width = 0.8 / max(n_methods, 1)

    plt.figure(figsize=(11, 6))

    for i, method in enumerate(methods):
        ys = marginal_df.loc[marginal_df["Index"] == method, interval_labels].values.flatten().astype(float)
        offset = (i - (n_methods - 1) / 2) * bar_width
        plt.bar(x + offset, ys, width=bar_width, label=method, alpha=0.9)

    plt.xticks(x, interval_labels)
    plt.ylabel("Marginal Memory Cost (GB / 1M vectors)")
    plt.xlabel("Data Scale Interval")
    plt.title("Marginal Memory Cost Across Data Scale")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_name, dpi=220, bbox_inches="tight")
    plt.close()


def compute_growth_factor(df, ordered_cols):
    """
    计算相对于第一个规模（通常是 1M）的增长倍数：
        M(N) / M(N0)
    """
    growth_rows = []

    for _, row in df.iterrows():
        idx = str(row["Index"])
        vals = row[ordered_cols[1:]].tolist()

        base = vals[0]
        growth = []
        for v in vals:
            if pd.notna(base) and base != 0 and pd.notna(v):
                growth.append(v / base)
            else:
                growth.append(np.nan)

        growth_rows.append([idx] + growth)

    growth_df = pd.DataFrame(growth_rows, columns=["Index"] + ordered_cols[1:])
    return growth_df


def draw_growth_factor(growth_df, ordered_cols, x_vals, x_labels, output_name="memory_growth_factor.png"):
    """
    图3：相对增长倍数图
    """
    plt.figure(figsize=(10, 6))

    for _, row in growth_df.iterrows():
        idx = str(row["Index"])
        y_all = row[ordered_cols[1:]].tolist()
        valid_points = [(x, y) for x, y in zip(x_vals, y_all) if pd.notna(y)]
        if not valid_points:
            continue

        xs = [p[0] for p in valid_points]
        ys = [p[1] for p in valid_points]

        plt.plot(xs, ys, marker="o", linewidth=2.2, label=idx)

    plt.xticks(x_vals, x_labels)
    plt.xlabel("Data Scale")
    plt.ylabel("Growth Factor vs 1M")
    plt.title("Relative Memory Growth Factor Across Data Scale")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_name, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到文件: {csv_path.resolve()}")

    df, ordered_cols, x_vals, x_labels = load_data(str(csv_path))

    draw_absolute_split(df, ordered_cols, x_vals, x_labels, "memory_absolute_split.png")

    marginal_df, interval_labels = compute_marginal_cost(df, ordered_cols, x_vals)
    draw_marginal_cost(marginal_df, interval_labels, "memory_marginal_cost.png")

    growth_df = compute_growth_factor(df, ordered_cols)
    draw_growth_factor(growth_df, ordered_cols, x_vals, x_labels, "memory_growth_factor.png")

    print("完成，已生成：")
    print(" - memory_absolute_split.png")
    print(" - memory_marginal_cost.png")
    print(" - memory_growth_factor.png")


if __name__ == "__main__":
    main()