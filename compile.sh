#!/bin/bash
# Paths
PROJECT_ROOT=$(pwd)
BUILD_DIR="$PROJECT_ROOT/build"

# ccache on by default; pass -ncc to disable (forces a clean rebuild)
if [[ "$1" == "-ncc" ]]; then
    export CCACHE_DISABLE=1
    rm -rf "$BUILD_DIR"
fi

# Configure CMake (out-of-source build)
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"

# Build everything
cmake --build "$BUILD_DIR" -j 1
cat build/CMakeCache.txt | grep CMAKE_CXX_COMPILER