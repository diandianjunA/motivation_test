import re
from pathlib import Path
import matplotlib.pyplot as plt


def parse_thread_from_filename(path: Path) -> int:
    m = re.search(r"(\d+)-thread", path.stem)
    if not m:
        raise ValueError(f"无法从文件名解析线程数: {path.name}")
    return int(m.group(1))


def extract_section(text: str, start_title: str, next_title_candidates=None) -> str:
    if next_title_candidates is None:
        next_title_candidates = []

    start = text.find(start_title)
    if start == -1:
        return ""

    sub = text[start:]
    end_positions = []

    for t in next_title_candidates:
        pos = sub.find(t, len(start_title))
        if pos != -1:
            end_positions.append(pos)

    if end_positions:
        return sub[:min(end_positions)]
    return sub


def extract_latency_metrics(text: str, op_name: str):
    """
    从报告中提取 INSERT / QUERY 的端到端延迟：
    平均、P50、P95、P99
    """
    if op_name.upper() == "INSERT":
        section = extract_section(
            text,
            "## INSERT 分析",
            next_title_candidates=["## QUERY 分析", "## Insert / Query 对比", "## System Counters"]
        )
    elif op_name.upper() == "QUERY":
        section = extract_section(
            text,
            "## QUERY 分析",
            next_title_candidates=["## Insert / Query 对比", "## System Counters"]
        )
    else:
        raise ValueError(f"不支持的操作名: {op_name}")

    if not section:
        raise ValueError(f"未找到 {op_name} 分析段落")

    patterns = {
        "mean": r"- 平均端到端延迟：\*\*([\d.]+)\s*ms\*\*",
        "p50": r"- P50 端到端延迟：\*\*([\d.]+)\s*ms\*\*",
        "p95": r"- P95 端到端延迟：\*\*([\d.]+)\s*ms\*\*",
        "p99": r"- P99 端到端延迟：\*\*([\d.]+)\s*ms\*\*",
    }

    result = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, section)
        if not m:
            raise ValueError(f"未找到 {op_name} 的 {key} 延迟")
        result[key] = float(m.group(1))

    return result


def load_all_reports(report_dir: Path):
    md_files = sorted(
        report_dir.glob("*-thread.md"),
        key=lambda p: parse_thread_from_filename(p)
    )

    if not md_files:
        raise FileNotFoundError(f"目录 {report_dir} 下未找到 *-thread.md 文件")

    data = {
        "insert": {},
        "query": {},
    }

    for path in md_files:
        thread = parse_thread_from_filename(path)
        text = path.read_text(encoding="utf-8")

        data["insert"][thread] = extract_latency_metrics(text, "INSERT")
        data["query"][thread] = extract_latency_metrics(text, "QUERY")

    return data


def annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=8
        )


def plot_grouped_latency(data_by_thread: dict, title: str, output_path: Path):
    threads = sorted(data_by_thread.keys())
    x = list(range(len(threads)))

    mean_vals = [data_by_thread[t]["mean"] for t in threads]
    p50_vals = [data_by_thread[t]["p50"] for t in threads]
    p95_vals = [data_by_thread[t]["p95"] for t in threads]
    p99_vals = [data_by_thread[t]["p99"] for t in threads]

    width = 0.2

    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    bars1 = ax.bar([i - 1.5 * width for i in x], mean_vals, width=width, label="mean")
    bars2 = ax.bar([i - 0.5 * width for i in x], p50_vals, width=width, label="p50")
    bars3 = ax.bar([i + 0.5 * width for i in x], p95_vals, width=width, label="p95")
    bars4 = ax.bar([i + 1.5 * width for i in x], p99_vals, width=width, label="p99")

    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    annotate_bars(ax, bars3)
    annotate_bars(ax, bars4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("Thread Count")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    report_dir = Path("/home/xjs/experiment/motivation_test/shine_gpu_test/reports/analyze")   # 默认当前目录
    data = load_all_reports(report_dir)

    plot_grouped_latency(
        data["insert"],
        "Insert End-to-End Latency by Thread Count",
        report_dir / "insert_end_to_end_latency.png"
    )

    plot_grouped_latency(
        data["query"],
        "Query End-to-End Latency by Thread Count",
        report_dir / "query_end_to_end_latency.png"
    )

    print("已生成:")
    print(report_dir / "insert_end_to_end_latency.png")
    print(report_dir / "query_end_to_end_latency.png")


if __name__ == "__main__":
    main()