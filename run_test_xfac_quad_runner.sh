#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory '$BUILD_DIR' not found."
    exit 1
fi

cd "$BUILD_DIR" || exit 1

echo "==> Running xfac_quad_runner tests..."

for name in param runner determinism
do
    tgt="xfac_quad_runner_test_$name"
    echo "--- $tgt ---"
    ./"$tgt" || exit 1
done

echo "==> All xfac_quad_runner tests passed."
