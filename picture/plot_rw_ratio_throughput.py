# -*- coding: utf-8 -*-
"""
绘制不同索引在不同读写比例下的吞吐变化图

输入文件:
    读写比例.csv

输出文件:
    throughput_vs_rw_ratio_linear.png
    throughput_vs_rw_ratio_log.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = Path("rw_ratio_throuput.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到文件: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    # 第一列默认为索引名称
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Index"})

    # 后续列是读比例，例如 1, 0.9, 0.8, ...
    ratio_cols = [c for c in df.columns if c != "Index"]
    ratio_vals = [float(c) for c in ratio_cols]

    # 统一数值化
    for c in ratio_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 字体设置（如果本机没中文字体，会自动回退）
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    def draw_normalized_chart(output_name: str):
        plt.figure(figsize=(10, 6))

        for _, row in df.iterrows():
            y_all = pd.to_numeric(row[ratio_cols], errors="coerce").tolist()
            valid_points = [(x, y) for x, y in zip(ratio_vals, y_all) if pd.notna(y)]
            if not valid_points:
                continue

            xs = [p[0] for p in valid_points]
            ys = [p[1] for p in valid_points]

            max_y = max(ys)
            ys_norm = [y / max_y for y in ys]

            plt.plot(
                xs, ys_norm,
                linewidth=2.2,
                marker="o",
                markersize=4,
                alpha=0.9,
                label=row["Index"]
            )

        plt.xlabel("Read Ratio (%)")
        plt.ylabel("Normalized Throughput")
        plt.title("Normalized Throughput Across Read Ratio")
        plt.xticks(
            ratio_vals,
            [f"{int(x*100)}:{int((1-x)*100)}" for x in ratio_vals],
            rotation=30
        )
        plt.grid(True, axis="y", linestyle="--", alpha=0.25)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        plt.tight_layout()
        plt.savefig(output_name, dpi=220, bbox_inches="tight")
        plt.close()

    def draw_chart(output_name: str, yscale: str = "linear"):
        plt.figure(figsize=(10, 6))

        for _, row in df.iterrows():
            y_all = row[ratio_cols].tolist()
            valid_points = [(x, y) for x, y in zip(ratio_vals, y_all) if pd.notna(y)]
            if not valid_points:
                continue

            xs = [p[0] for p in valid_points]
            ys = [p[1] for p in valid_points]

            plt.plot(
                xs, ys,
                linewidth=2.2,
                marker="o",
                markersize=4,
                alpha=0.9,
                label=row["Index"]
            )

        plt.xlabel("Read Ratio (%)")
        plt.ylabel("Throughput (ops/s)")
        plt.title("Throughput Across Read Ratio")
        plt.yscale(yscale)

        # 用更直观的比例标签
        plt.xticks(
            ratio_vals,
            [f"{int(x*100)}:{int((1-x)*100)}" for x in ratio_vals],
            rotation=30
        )

        # 网格更淡，只保留 y 方向主网格
        plt.grid(True, axis="y", linestyle="--", alpha=0.25)

        # 把图例放到右侧，避免和曲线打架
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        plt.tight_layout()
        plt.savefig(output_name, dpi=220, bbox_inches="tight")
        plt.close()

    draw_chart("throughput_vs_rw_ratio_linear.png", yscale="linear")
    draw_chart("throughput_vs_rw_ratio_log.png", yscale="log")
    draw_normalized_chart("throughput_vs_rw_ratio_normalized.png")

    print("绘图完成，已生成：")
    print(" - throughput_vs_rw_ratio_linear.png")
    print(" - throughput_vs_rw_ratio_log.png")
    print(" - throughput_vs_rw_ratio_normalized.png")



if __name__ == "__main__":
    main()
