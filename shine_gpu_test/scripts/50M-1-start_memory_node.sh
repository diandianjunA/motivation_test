#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/50M-start_memory_node.sh" "$@" -p 1234 --mn-memory 60
