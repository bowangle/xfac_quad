#!/usr/bin/env bash
# Compatibility with the previous script name and its -slow option.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run-test.sh" "$@"
