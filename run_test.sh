#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_test_qtgrid.sh" "$@" || exit $?
bash "$SCRIPT_DIR/run_test_xfac_quad.sh" "$@" || exit $?
bash "$SCRIPT_DIR/run_test_xfac_quad_runner.sh" "$@" || exit $?

echo "==> All tests passed."
