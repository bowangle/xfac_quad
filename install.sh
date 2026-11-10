#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$PROJECT_ROOT/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$REPO_ROOT/install}"

BUILD_DIR="$BUILD_DIR" bash "$PROJECT_ROOT/compile.sh" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" "$@"
cmake --install "$BUILD_DIR"
