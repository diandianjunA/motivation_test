#!/bin/bash
# =============================================================================
# SHINE GPU Offline Index Builder
# =============================================================================
# 使用 Jasper 离线构图，并导出为 GPU 分支可直接 load 的 shard 文件。
#
# 用法:
#   ./build_offline_index.sh [选项...]
#
# 选项:
#   -d, --data-path <path>         数据文件或数据目录路径
#   -o, --output-prefix <path>     输出前缀，不带 _nodeX_ofN.dat 后缀
#   -n, --memory-nodes <n>         输出 shard 数 / memory node 数（默认: 1）
#   -t, --threads <n>              构图线程数（默认: 0，表示硬件并发）
#       --m <n>                    平面图最大出度（默认: 32，必须 <= 128）
#       --ef-construction <n>      Jasper 构图 beam cap 目标值（默认: 200，会向上取整到支持档位）
#       --explore-per-iteration <n> Jasper 每轮构图扩展节点数（默认: 4）
#       --max-vectors <n>          最多读取多少条向量
#       --reserve-vectors <n>      导出索引的总容量，允许 load 后继续在线插入
#       --seed <n>                 随机种子（默认: 1234）
#   -h, --help                     显示帮助
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$PROJECT_DIR/bin/shine_offline_builder"

DATA_PATH="${DATA_PATH:-/data/xjs/random_dataset/1024dim1M}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/data/xjs/index/shine_gpu_index/1024dim1M}"
MEMORY_NODES="${MEMORY_NODES:-1}"
THREADS="${THREADS:-0}"
M="${M:-32}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-150}"
MAX_VECTORS="${MAX_VECTORS:-4294967295}"
RESERVE_VECTORS="${RESERVE_VECTORS:-0}"
SEED="${SEED:-1234}"
EXPLORE_PER_ITERATION="${EXPLORE_PER_ITERATION:-4}"

usage() {
    sed -n '/^# 用法:/,/^# =====/p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-path)         DATA_PATH="$2"; shift 2 ;;
        -o|--output-prefix)     OUTPUT_PREFIX="$2"; shift 2 ;;
        -n|--memory-nodes)      MEMORY_NODES="$2"; shift 2 ;;
        -t|--threads)           THREADS="$2"; shift 2 ;;
        --m)                    M="$2"; shift 2 ;;
        --ef-construction)      EF_CONSTRUCTION="$2"; shift 2 ;;
        --explore-per-iteration) EXPLORE_PER_ITERATION="$2"; shift 2 ;;
        --max-vectors)          MAX_VECTORS="$2"; shift 2 ;;
        --reserve-vectors)      RESERVE_VECTORS="$2"; shift 2 ;;
        --seed)                 SEED="$2"; shift 2 ;;
        -h|--help)              usage ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 查看帮助"
            exit 1
            ;;
    esac
done

if [[ -z "$DATA_PATH" ]]; then
    echo "错误: --data-path 是必填参数"
    exit 1
fi

if [[ ! -x "$BINARY" ]]; then
    echo "错误: 找不到可执行文件 $BINARY"
    echo "请先编译项目: cd $PROJECT_DIR && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

args=(
    --data-path "$DATA_PATH"
    --memory-nodes "$MEMORY_NODES"
    --threads "$THREADS"
    --m "$M"
    --ef-construction "$EF_CONSTRUCTION"
    --explore-per-iteration "$EXPLORE_PER_ITERATION"
    --max-vectors "$MAX_VECTORS"
    --reserve-vectors "$RESERVE_VECTORS"
    --seed "$SEED"
)

if [[ -n "$OUTPUT_PREFIX" ]]; then
    args+=(--output-prefix "$OUTPUT_PREFIX")
fi

echo "[SHINE GPU Offline Builder] 参数:"
echo "  数据路径:     $DATA_PATH"
if [[ -n "$OUTPUT_PREFIX" ]]; then
    echo "  输出前缀:     $OUTPUT_PREFIX"
fi
echo "  Memory 节点:  $MEMORY_NODES"
echo "  线程数:       $THREADS"
echo "  M:            $M"
echo "  efc:          $EF_CONSTRUCTION"
echo "  explore:      $EXPLORE_PER_ITERATION"
echo "  最大向量数:   $MAX_VECTORS"
echo "  预留容量:     $RESERVE_VECTORS"
echo ""

exec "$BINARY" "${args[@]}"
