#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory '$BUILD_DIR' not found."
    exit 1
fi

cd "$BUILD_DIR" || exit 1

echo "==> Running QTGrid tests..."

for name in grid multigrid
do
    tgt="qtgrid_test_$name"
    echo "--- $tgt ---"
    ./"$tgt" || exit 1
done

echo "==> All QTGrid tests passed."
