#!/bin/bash
# =============================================================================
# SHINE Compute Node Launcher
# =============================================================================
# 启动 SHINE 计算节点（含 HTTP 服务）。
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
#   -s, --servers <hosts>         Memory 节点地址（默认: cluster3）
#   -p, --port <port>             RDMA 通信端口（默认: 1234）
#   -d, --dim <dim>               向量维度（默认: 128）
#   -t, --threads <n>             工作线程数（默认: 4）
#       --ef-search <n>           HNSW 搜索 ef 参数（默认: 64）
#   -k <n>                        Top-K 返回数量（默认: 10）
#       --m <n>                   HNSW M 参数（默认: 32）
#       --ef-construction <n>     HNSW 构建 ef 参数（默认: 200）
#       --http-port <port>        HTTP 服务端口（默认: 8080）
#       --coroutines <n>          协程数量（默认: 4）
#       --max-vectors <n>         最大向量数（默认: 1000000）
#       --cn-memory <GB>          计算节点内存（GB）（默认: 10）
#       --enable-thread-pinning   启用线程绑核（默认禁用）
#   -f, --foreground              前台运行（默认后台运行）
#   -h, --help                    显示此帮助信息
#
# 环境变量:
#   所有参数均可通过同名大写环境变量覆盖，例如:
#     SERVERS=cluster3 DIM=256 ./start_compute_node.sh
#
# 示例:
#   ./start_compute_node.sh                          # 使用默认参数后台启动
#   ./start_compute_node.sh -s cluster3 -d 256       # 指定服务器和维度
#   ./start_compute_node.sh -f                       # 前台运行（调试用）
#   ./start_compute_node.sh stop                     # 停止节点
#   ./start_compute_node.sh status                   # 查看状态
#   ./start_compute_node.sh restart                  # 重启节点
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/bin/shine"
PID_FILE="$PROJECT_DIR/.compute_node.pid"
LOG_FILE="$PROJECT_DIR/logs/compute_node.log"

# ---- 默认参数（可通过环境变量覆盖） ----
SERVERS="${SERVERS:-cluster1}"
DIM="${DIM:-1024}"
THREADS="${THREADS:-4}"
EF_SEARCH="${EF_SEARCH:-64}"
K="${K:-10}"
M="${M:-32}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-200}"
HTTP_PORT="${HTTP_PORT:-8080}"
COROUTINES="${COROUTINES:-4}"
MAX_VECTORS="${MAX_VECTORS:-1000000}"
CN_MEMORY="${CN_MEMORY:-10}"
PORT="${PORT:-1234}"
FOREGROUND=false
DISABLE_THREAD_PINNING=true

# ---- 帮助信息 ----
usage() {
    sed -n '/^# 用法:/,/^# =====/p' "$0" | sed 's/^# \?//'
    exit 0
}

# ---- 解析命令行参数 ----
COMMAND="start"
EXTRA_ARGS=()

# 第一个非 - 开头的参数作为命令
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    COMMAND="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--servers)         SERVERS="$2"; shift 2 ;;
        -p|--port)            PORT="$2"; shift 2 ;;
        -d|--dim)             DIM="$2"; shift 2 ;;
        -t|--threads)         THREADS="$2"; shift 2 ;;
        --ef-search)          EF_SEARCH="$2"; shift 2 ;;
        -k)                   K="$2"; shift 2 ;;
        --m)                  M="$2"; shift 2 ;;
        --ef-construction)    EF_CONSTRUCTION="$2"; shift 2 ;;
        --http-port)          HTTP_PORT="$2"; shift 2 ;;
        --coroutines)         COROUTINES="$2"; shift 2 ;;
        --max-vectors)        MAX_VECTORS="$2"; shift 2 ;;
        --cn-memory)          CN_MEMORY="$2"; shift 2 ;;
        --enable-thread-pinning) DISABLE_THREAD_PINNING=false; shift ;;
        -f|--foreground)      FOREGROUND=true; shift ;;
        -h|--help)            usage ;;
        *)                    EXTRA_ARGS+=("$1"); shift ;;
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

    # 构建参数列表
    local args=(
        --initiator
        --servers "$SERVERS"
        --port "$PORT"
        --dim "$DIM"
        --threads "$THREADS"
        --ef-search "$EF_SEARCH"
        --k "$K"
        --m "$M"
        --ef-construction "$EF_CONSTRUCTION"
        --http-port "$HTTP_PORT"
        --coroutines "$COROUTINES"
        --max-vectors "$MAX_VECTORS"
        --cn-memory "$CN_MEMORY"
    )

    if [[ "$DISABLE_THREAD_PINNING" == true ]]; then
        args+=(--disable-thread-pinning)
    fi

    args+=("${EXTRA_ARGS[@]}")

    echo "[SHINE Compute Node] 启动参数:"
    echo "  服务器:       $SERVERS"
    echo "  RDMA 端口:    $PORT"
    echo "  HTTP 端口:    $HTTP_PORT"
    echo "  向量维度:     $DIM"
    echo "  线程数:       $THREADS"
    echo "  协程数:       $COROUTINES"
    echo "  最大向量数:   $MAX_VECTORS"
    echo "  内存(GB):     $CN_MEMORY"

    if [[ "$FOREGROUND" == true ]]; then
        echo "  模式:         前台运行"
        echo ""
        exec "$BINARY" "${args[@]}"
    else
        mkdir -p "$(dirname "$LOG_FILE")"
        echo "  模式:         后台运行"
        echo "  日志文件:     $LOG_FILE"
        echo ""
        nohup "$BINARY" "${args[@]}" >> "$LOG_FILE" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        # 短暂等待，检查进程是否立即退出
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
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

# ---- 执行命令 ----
case "$COMMAND" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)  do_status ;;
    *)
        echo "未知命令: $COMMAND"
        echo "可用命令: start, stop, restart, status"
        echo "使用 -h 查看帮助"
        exit 1
        ;;
esac
