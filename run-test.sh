#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/test/test_log.txt}"

# Preserve the old -slow switch: benchmarks are opt-in and use Catch2's [.] tag.
filter=(-LE slow)
if [[ "${1:-}" == "-slow" ]]; then
    filter=()
    shift
fi

mkdir -p -- "$(dirname -- "$LOG_FILE")"
echo "Running xfac_quad tests. Full output: $LOG_FILE"
status=0
ctest --test-dir "$BUILD_DIR" --verbose --no-tests=error --stop-on-failure \
    "${filter[@]}" "$@" > "$LOG_FILE" 2>&1 || status=$?

# Keep the terminal readable; stdout and stderr from every test remain in the log.
grep -E 'Test #[0-9]+:|tests passed|Total Test time|The following tests FAILED:' "$LOG_FILE" || true
if (( status != 0 )); then
    echo "Tests failed (exit $status). See $LOG_FILE for details." >&2
fi
exit "$status"
