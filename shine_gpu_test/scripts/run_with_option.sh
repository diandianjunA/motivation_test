#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <base_config> <option> <runner>" >&2
    exit 1
fi

BASE_CONFIG="$1"
OPTION_VALUE="$2"
RUNNER="$3"

TMP_CONFIG="$(mktemp)"
trap 'rm -f "$TMP_CONFIG"' EXIT

awk -v option_value="$OPTION_VALUE" '
BEGIN {
    replaced = 0
}
/^[[:space:]]*option[[:space:]]*=/ {
    if (!replaced) {
        print "option = " option_value
        replaced = 1
    }
    next
}
{
    print
}
END {
    if (!replaced) {
        print "option = " option_value
    }
}
' "$BASE_CONFIG" > "$TMP_CONFIG"

exec "$RUNNER" "$TMP_CONFIG"
