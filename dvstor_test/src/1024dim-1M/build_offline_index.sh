#!/bin/bash
# =============================================================================
# Vamana Offline Index Builder
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$PROJECT_DIR/bin/vamana_offline_builder"

DATA_PATH="${DATA_PATH:-/data/xjs/random_dataset/1024dim1M}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/data/xjs/index/shine_gpu_index/1024dim1M_bits4}"
MEMORY_NODES="${MEMORY_NODES:-1}"
THREADS="${THREADS:-0}"
R="${R:-32}"
BEAM_WIDTH="${BEAM_WIDTH:-128}"
ALPHA="${ALPHA:-1.2}"
MAX_VECTORS="${MAX_VECTORS:-4294967295}"
SEED="${SEED:-1234}"
RABITQ_BITS="${RABITQ_BITS:-4}"
IP_DIST="${IP_DIST:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-path)         DATA_PATH="$2"; shift 2 ;;
        -o|--output-prefix)     OUTPUT_PREFIX="$2"; shift 2 ;;
        -n|--memory-nodes)      MEMORY_NODES="$2"; shift 2 ;;
        -t|--threads)           THREADS="$2"; shift 2 ;;
        --R)                    R="$2"; shift 2 ;;
        --beam-width|--beam-width-construction) BEAM_WIDTH="$2"; shift 2 ;;
        --alpha)                ALPHA="$2"; shift 2 ;;
        --max-vectors)          MAX_VECTORS="$2"; shift 2 ;;
        --seed)                 SEED="$2"; shift 2 ;;
        --rabitq-bits)          RABITQ_BITS="$2"; shift 2 ;;
        --ip-dist)              IP_DIST=true; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  -d, --data-path <path>"
            echo "  -o, --output-prefix <path>"
            echo "  -n, --memory-nodes <n>     (default: 1)"
            echo "  -t, --threads <n>          (default: auto)"
            echo "      --R <n>                (default: 32)"
            echo "      --beam-width <n>       Offline construction beam width (default: 128)"
            echo "      --alpha <f>            (default: 1.2)"
            echo "      --max-vectors <n>"
            echo "      --seed <n>             (default: 1234)"
            echo "      --rabitq-bits <n>      (default: 4)"
            echo "      --ip-dist"
            exit 0
            ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary not found: $BINARY"
    exit 1
fi

args=(
    --data-path "$DATA_PATH"
    --memory-nodes "$MEMORY_NODES"
    --threads "$THREADS"
    --R "$R"
    --beam-width "$BEAM_WIDTH"
    --alpha "$ALPHA"
    --max-vectors "$MAX_VECTORS"
    --seed "$SEED"
    --rabitq-bits "$RABITQ_BITS"
)

if [[ -n "$OUTPUT_PREFIX" ]]; then
    args+=(--output-prefix "$OUTPUT_PREFIX")
fi

if [[ "$IP_DIST" == true ]]; then
    args+=(--ip-dist)
fi

echo "[Vamana Offline Builder] parameters:"
echo "  data_path:     $DATA_PATH"
echo "  output_prefix: $OUTPUT_PREFIX"
echo "  memory_nodes:  $MEMORY_NODES"
echo "  threads:       $THREADS"
echo "  R:             $R"
echo "  build_beam:    $BEAM_WIDTH"
echo "  alpha:         $ALPHA"
echo "  rabitq_bits:   $RABITQ_BITS"
echo "  max_vectors:   $MAX_VECTORS"
echo ""

exec "$BINARY" "${args[@]}"
