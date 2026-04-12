#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CONFIG="${CONFIG:-$ROOT_DIR/config/1024dim1M.ini}"
RUNNER="${RUNNER:-$ROOT_DIR/build/DvstorVectorTest}"
HELPER="$ROOT_DIR/scripts/run_with_option.sh"

exec "$HELPER" "$BASE_CONFIG" storage "$RUNNER"
