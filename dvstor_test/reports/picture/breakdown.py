import re
from pathlib import Path
import matplotlib.pyplot as plt


CATEGORIES = ["cpu_ns", "gpu_ns", "rdma_ns", "transfer_ns"]


def parse_thread_from_filename(path: Path) -> int:
    m = re.search(r"(\d+)-thread", path.stem)
    if not m:
        raise ValueError(f"无法从文件名解析线程数: {path.name}")
    return int(m.group(1))


def extract_section(text: str, start_title: str, next_title_candidates=None) -> str:
    """
    提取从 start_title 开始，到下一个候选标题之前的文本
    """
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


def parse_markdown_table(table_text: str):
    """
    解析 markdown 表格，返回 [{'部分':..., '时间':..., '占比':...}, ...]
    只解析一级 breakdown 那种三列表格
    """
    rows = []
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]

    for line in lines:
        if not line.startswith("|"):
            continue
        if "---" in line:
            continue

        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) != 3:
            continue
        if cols[0] == "部分":
            continue

        rows.append({
            "part": cols[0],
            "time": cols[1],
            "pct": cols[2],
        })
    return rows


def extract_top_level_breakdown(text: str, op_name: str):
    """
    从报告中提取 INSERT / QUERY 的一级 Breakdown 占比
    返回 dict: {cpu_ns: xx, gpu_ns: xx, rdma_ns: xx, transfer_ns: xx}
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

    # 找“### 一级 Breakdown 占比”后面的第一张表
    m = re.search(
        r"### 一级 Breakdown 占比\s*(\|.*?(?=\n\n- |\n### |\n## |\Z))",
        section,
        flags=re.S
    )
    if not m:
        raise ValueError(f"未找到 {op_name} 的一级 Breakdown 表格")

    table_text = m.group(1)
    rows = parse_markdown_table(table_text)

    result = {}
    for row in rows:
        part = row["part"]
        pct_str = row["pct"].replace("%", "").strip()
        if part in CATEGORIES:
            result[part] = float(pct_str)

    # 缺失项补 0
    for cat in CATEGORIES:
        result.setdefault(cat, 0.0)

    return result


def load_all_reports(report_dir: Path):
    """
    扫描目录中的 *-thread.md 文件，提取 insert/query 一级 breakdown 百分比
    """
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

        insert_data = extract_top_level_breakdown(text, "INSERT")
        query_data = extract_top_level_breakdown(text, "QUERY")

        data["insert"][thread] = insert_data
        data["query"][thread] = query_data

    return data


def annotate_segments(ax, x_positions, bottoms, heights, threshold=6.0):
    """
    在堆叠柱每个分段内部标注百分比
    只给占比 >= threshold 的分段标，避免太挤
    """
    for x, bottom, h in zip(x_positions, bottoms, heights):
        if h >= threshold:
            ax.text(
                x,
                bottom + h / 2,
                f"{h:.1f}%",
                ha="center",
                va="center",
                fontsize=9
            )


def annotate_totals(ax, x_positions, totals):
    """
    在柱子顶部标 100%
    """
    for x, total in zip(x_positions, totals):
        ax.text(
            x,
            total + 1.0,
            f"{total:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )


def plot_stacked(data_by_thread: dict, title: str, output_path: Path):
    threads = sorted(data_by_thread.keys())
    x = list(range(len(threads)))

    bottoms = [0.0] * len(threads)
    totals = [0.0] * len(threads)

    plt.figure(figsize=(10, 6))

    for cat in CATEGORIES:
        values = [data_by_thread[t].get(cat, 0.0) for t in threads]
        plt.bar(x, values, bottom=bottoms, label=cat)

        annotate_segments(plt.gca(), x, bottoms, values, threshold=6.0)

        bottoms = [b + v for b, v in zip(bottoms, values)]

    totals = bottoms[:]
    annotate_totals(plt.gca(), x, totals)

    plt.xticks(x, [str(t) for t in threads])
    plt.xlabel("Thread Count")
    plt.ylabel("Percentage (%)")
    plt.title(title)
    plt.ylim(0, max(105, max(totals) + 5))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    report_dir = Path("/home/xjs/experiment/motivation_test/shine_gpu_test/reports/analyze")   # 默认当前目录
    data = load_all_reports(report_dir)

    plot_stacked(
        data["insert"],
        "Insert Breakdown (Top-level) by Thread Count",
        report_dir / "insert_breakdown_stacked.png"
    )

    plot_stacked(
        data["query"],
        "Query Breakdown (Top-level) by Thread Count",
        report_dir / "query_breakdown_stacked.png"
    )

    print("已生成:")
    print(report_dir / "insert_breakdown_stacked.png")
    print(report_dir / "query_breakdown_stacked.png")


if __name__ == "__main__":
    main()