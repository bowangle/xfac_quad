#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$PROJECT_ROOT/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"

# Lower layers must already be installed. Extra arguments are CMake options.
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native" \
    -DCMAKE_PREFIX_PATH="$REPO_ROOT/install" "$@"
cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-1}"
