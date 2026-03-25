#!/bin/bash
# =============================================================================
# SHINE GPU Compute Node Launcher
# =============================================================================
# 启动 GPU baseline 计算节点（in-process service + RDMA memory nodes）。
# 支持前台/后台运行，提供 start/stop/status/restart 操作。
#
# 用法:
#   ./start_compute_node.sh [命令] [选项...]
#
# 命令:
#   start       启动计算节点（默认）
#   stop        停止后台运行的计算节点
#   restart     重启计算节点
#   status      查看计算节点运行状态
#
# 选项:
#   -s, --servers <hosts>         Memory 节点地址，支持空格或逗号分隔（默认: 127.0.0.1）
#   -p, --port <port>             RDMA 通信端口（默认: 1234）
#   -d, --dim <dim>               向量维度（默认: 128）
#   -t, --threads <n>             工作线程数，GPU 服务要求 >= 2（默认: 4）
#       --data-path <path>        数据目录，用于派生默认 dump/gpu_index 前缀
#       --index-prefix <path>     索引 shard 前缀，不带 _nodeX_ofN.dat 后缀
#       --load-index              启动时从 --data-path 派生前缀或 --index-prefix 对应的 shard 文件加载离线索引
#       --store-index             启动时将当前索引持久化到 --data-path 派生前缀或 --index-prefix 对应的 shard 文件
#       --ef-search <n>           Beam search 宽度（默认: 64）
#   -k <n>                        Top-K 返回数量（默认: 10）
#       --m <n>                   Vamana R 参数（默认: 32）
#       --ef-construction <n>     插入/构建 beam 宽度（默认: 200）
#       --coroutines <n>          query micro-batch / coroutine 数（默认: 4）
#       --max-vectors <n>         最大向量数（默认: 1000000）
#       --cn-memory <GB>          计算节点内存（GB）（默认: 10）
#       --gpu-device <id>         CUDA 设备号（默认: 0）
#       --cache                   启用 GPU baseline 本地 record cache
#       --cache-ratio <n>         cache 大小占索引估算大小百分比（默认: 5）
#       --enable-thread-pinning   启用线程绑核（默认禁用）
#   -f, --foreground              前台运行（默认后台运行）
#   -h, --help                    显示此帮助信息
#
# 环境变量:
#   所有参数均可通过同名大写环境变量覆盖，例如:
#     SERVERS=192.168.6.201 DIM=256 ./start_compute_node.sh
#
# 示例:
#   ./start_compute_node.sh                          # 使用默认参数后台启动
#   ./start_compute_node.sh -s 192.168.6.201 -d 256  # 指定服务器和维度
#   ./start_compute_node.sh -f                       # 前台运行（调试用）
#   ./start_compute_node.sh stop                     # 停止节点
#   ./start_compute_node.sh status                   # 查看状态
#   ./start_compute_node.sh restart                  # 重启节点
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/build/shine"
PID_FILE="$PROJECT_DIR/.compute_node.pid"
ARGS_FILE="$PROJECT_DIR/.compute_node.args"
LOG_FILE="$PROJECT_DIR/logs/compute_node.log"

# ---- 默认参数（可通过环境变量覆盖） ----
SERVERS="${SERVERS:-127.0.0.1}"
DIM="${DIM:-1024}"
THREADS="${THREADS:-4}"
EF_SEARCH="${EF_SEARCH:-64}"
K="${K:-10}"
M="${M:-32}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-200}"
COROUTINES="${COROUTINES:-4}"
MAX_VECTORS="${MAX_VECTORS:-1000000}"
CN_MEMORY="${CN_MEMORY:-10}"
GPU_DEVICE="${GPU_DEVICE:-0}"
CACHE_RATIO="${CACHE_RATIO:-5}"
PORT="${PORT:-1234}"
DATA_PATH="${DATA_PATH:-}"
INDEX_PREFIX="${INDEX_PREFIX:-}"
FOREGROUND=false
DISABLE_THREAD_PINNING=true
USE_CACHE=false
LOAD_INDEX=false
STORE_INDEX=false

# ---- 帮助信息 ----
usage() {
    sed -n '/^# 用法:/,/^# =====/p' "$0" | sed 's/^# \?//'
    exit 0
}

# ---- 解析命令行参数 ----
COMMAND="start"
EXTRA_ARGS=()
HAS_EXPLICIT_ARGS=false

# 第一个非 - 开头的参数作为命令
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    COMMAND="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--servers)         HAS_EXPLICIT_ARGS=true; SERVERS="$2"; shift 2 ;;
        -p|--port)            HAS_EXPLICIT_ARGS=true; PORT="$2"; shift 2 ;;
        -d|--dim)             HAS_EXPLICIT_ARGS=true; DIM="$2"; shift 2 ;;
        -t|--threads)         HAS_EXPLICIT_ARGS=true; THREADS="$2"; shift 2 ;;
        --data-path)          HAS_EXPLICIT_ARGS=true; DATA_PATH="$2"; shift 2 ;;
        --index-prefix)       HAS_EXPLICIT_ARGS=true; INDEX_PREFIX="$2"; shift 2 ;;
        --load-index)         HAS_EXPLICIT_ARGS=true; LOAD_INDEX=true; shift ;;
        --store-index)        HAS_EXPLICIT_ARGS=true; STORE_INDEX=true; shift ;;
        --ef-search)          HAS_EXPLICIT_ARGS=true; EF_SEARCH="$2"; shift 2 ;;
        -k)                   HAS_EXPLICIT_ARGS=true; K="$2"; shift 2 ;;
        --m)                  HAS_EXPLICIT_ARGS=true; M="$2"; shift 2 ;;
        --ef-construction)    HAS_EXPLICIT_ARGS=true; EF_CONSTRUCTION="$2"; shift 2 ;;
        --coroutines)         HAS_EXPLICIT_ARGS=true; COROUTINES="$2"; shift 2 ;;
        --max-vectors)        HAS_EXPLICIT_ARGS=true; MAX_VECTORS="$2"; shift 2 ;;
        --cn-memory)          HAS_EXPLICIT_ARGS=true; CN_MEMORY="$2"; shift 2 ;;
        --gpu-device)         HAS_EXPLICIT_ARGS=true; GPU_DEVICE="$2"; shift 2 ;;
        --cache)              HAS_EXPLICIT_ARGS=true; USE_CACHE=true; shift ;;
        --cache-ratio)        HAS_EXPLICIT_ARGS=true; CACHE_RATIO="$2"; shift 2 ;;
        --enable-thread-pinning) HAS_EXPLICIT_ARGS=true; DISABLE_THREAD_PINNING=false; shift ;;
        -f|--foreground)      HAS_EXPLICIT_ARGS=true; FOREGROUND=true; shift ;;
        -h|--help)            usage ;;
        *)                    HAS_EXPLICIT_ARGS=true; EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---- 工具函数 ----
get_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        else
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

save_args() {
    local args=("$@")
    printf '%s\n' "${args[@]}" > "$ARGS_FILE"
}

load_saved_args() {
    [[ -f "$ARGS_FILE" ]] || return 1
    mapfile -t SAVED_ARGS < "$ARGS_FILE"
    [[ "${#SAVED_ARGS[@]}" -gt 0 ]] || return 1
}

launch_binary() {
    local args=("$@")

    if [[ "$FOREGROUND" == true ]]; then
        save_args "${args[@]}"
        exec "$BINARY" "${args[@]}"
    else
        mkdir -p "$(dirname "$LOG_FILE")"
        nohup "$BINARY" "${args[@]}" >> "$LOG_FILE" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        # 短暂等待，检查进程是否立即退出
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            save_args "${args[@]}"
            echo "[SHINE Compute Node] 已启动 (PID: $pid)"
            echo ""
            echo "常用操作:"
            echo "  查看状态:  $0 status"
            echo "  查看日志:  tail -f $LOG_FILE"
            echo "  停止节点:  $0 stop"
        else
            rm -f "$PID_FILE"
            echo "错误: 进程启动后立即退出，请检查日志:"
            echo "  tail -20 $LOG_FILE"
            exit 1
        fi
    fi
}

# ---- 命令实现 ----
do_status() {
    local pid
    if pid=$(get_pid); then
        echo "[SHINE Compute Node] 运行中 (PID: $pid)"
        echo "  日志文件: $LOG_FILE"
        echo "  PID 文件: $PID_FILE"
        return 0
    else
        echo "[SHINE Compute Node] 未运行"
        return 1
    fi
}

do_stop() {
    local pid
    if pid=$(get_pid); then
        echo "[SHINE Compute Node] 正在停止 (PID: $pid) ..."
        kill "$pid"
        # 等待进程退出，最多 10 秒
        for i in $(seq 1 10); do
            if ! kill -0 "$pid" 2>/dev/null; then
                rm -f "$PID_FILE"
                echo "[SHINE Compute Node] 已停止"
                return 0
            fi
            sleep 1
        done
        echo "[SHINE Compute Node] 进程未响应，强制终止 ..."
        kill -9 "$pid" 2>/dev/null
        rm -f "$PID_FILE"
        echo "[SHINE Compute Node] 已强制停止"
    else
        echo "[SHINE Compute Node] 未运行"
    fi
}

do_start() {
    # 检查是否已在运行
    if pid=$(get_pid); then
        echo "[SHINE Compute Node] 已在运行 (PID: $pid)，如需重启请使用 restart 命令"
        exit 1
    fi

    # 检查二进制文件
    if [[ ! -x "$BINARY" ]]; then
        echo "错误: 找不到可执行文件 $BINARY"
        echo "请先编译项目: cd $PROJECT_DIR && mkdir -p build && cd build && cmake .. && make"
        exit 1
    fi

    if [[ "$THREADS" -lt 2 ]]; then
        echo "错误: GPU baseline 至少需要 2 个线程（1 个 insert worker + >=1 个 query worker）"
        exit 1
    fi

    if [[ "$LOAD_INDEX" == true || "$STORE_INDEX" == true ]] && [[ -z "$DATA_PATH" && -z "$INDEX_PREFIX" ]]; then
        echo "错误: 使用 --load-index 或 --store-index 时必须提供 --data-path 或 --index-prefix"
        exit 1
    fi

    local normalized_servers="${SERVERS//,/ }"
    read -r -a server_list <<< "$normalized_servers"
    if [[ "${#server_list[@]}" -eq 0 ]]; then
        echo "错误: 至少需要一个 memory node 地址"
        exit 1
    fi

    # 构建参数列表
    local args=(
        --initiator
        --servers "${server_list[@]}"
        --port "$PORT"
        --dim "$DIM"
        --threads "$THREADS"
        --ef-search "$EF_SEARCH"
        --k "$K"
        --m "$M"
        --ef-construction "$EF_CONSTRUCTION"
        --coroutines "$COROUTINES"
        --max-vectors "$MAX_VECTORS"
        --cn-memory "$CN_MEMORY"
        --gpu-device "$GPU_DEVICE"
        --cache-ratio "$CACHE_RATIO"
    )

    if [[ -n "$DATA_PATH" ]]; then
        args+=(--data-path "$DATA_PATH")
    fi

    if [[ -n "$INDEX_PREFIX" ]]; then
        args+=(--index-prefix "$INDEX_PREFIX")
    fi

    if [[ "$LOAD_INDEX" == true ]]; then
        args+=(--load-index)
    fi

    if [[ "$STORE_INDEX" == true ]]; then
        args+=(--store-index)
    fi

    if [[ "$DISABLE_THREAD_PINNING" == true ]]; then
        args+=(--disable-thread-pinning)
    fi

    if [[ "$USE_CACHE" == true ]]; then
        args+=(--cache)
    fi

    args+=("${EXTRA_ARGS[@]}")

    echo "[SHINE Compute Node] 启动参数:"
    echo "  服务器:       ${server_list[*]}"
    echo "  RDMA 端口:    $PORT"
    echo "  向量维度:     $DIM"
    echo "  线程数:       $THREADS"
    echo "  协程数:       $COROUTINES"
    echo "  GPU 设备:     $GPU_DEVICE"
    echo "  最大向量数:   $MAX_VECTORS"
    echo "  Cache:        $USE_CACHE"
    echo "  Cache 比例:   $CACHE_RATIO%"
    echo "  内存(GB):     $CN_MEMORY"
    if [[ -n "$DATA_PATH" ]]; then
        echo "  数据路径:     $DATA_PATH"
    fi
    if [[ -n "$INDEX_PREFIX" ]]; then
        echo "  索引前缀:     $INDEX_PREFIX"
    fi
    echo "  启动加载:     $LOAD_INDEX"
    echo "  启动存储:     $STORE_INDEX"

    if [[ "$FOREGROUND" == true ]]; then
        echo "  模式:         前台运行"
        echo ""
    else
        mkdir -p "$(dirname "$LOG_FILE")"
        echo "  模式:         后台运行"
        echo "  日志文件:     $LOG_FILE"
        echo ""
    fi

    launch_binary "${args[@]}"
}

do_restart() {
    do_stop

    if [[ "$HAS_EXPLICIT_ARGS" == false ]] && load_saved_args; then
        echo "[SHINE Compute Node] 使用上次启动参数重启"
        printf '  命令参数:'
        printf ' %q' "${SAVED_ARGS[@]}"
        echo
        if [[ "$FOREGROUND" == true ]]; then
            echo "  模式:         前台运行"
            echo ""
        else
            mkdir -p "$(dirname "$LOG_FILE")"
            echo "  模式:         后台运行"
            echo "  日志文件:     $LOG_FILE"
            echo ""
        fi
        launch_binary "${SAVED_ARGS[@]}"
        return
    fi

    do_start
}

# ---- 执行命令 ----
case "$COMMAND" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_restart ;;
    status)  do_status ;;
    *)
        echo "未知命令: $COMMAND"
        echo "可用命令: start, stop, restart, status"
        echo "使用 -h 查看帮助"
        exit 1
        ;;
esac
