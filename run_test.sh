#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory '$BUILD_DIR' not found."
    exit 1
fi

cd "$BUILD_DIR" || exit 1

PREFIX=xfac_quad_test_

echo "==> Running xfac_quad tests..."

for name in \
SVDDecomp_double SVDDecomp_float128 SVDDecomp_dd128 \
matrix_ci_double matrix_ci_float128 matrix_ci_dd128 \
tensor_ci_double tensor_ci_float128 tensor_ci_dd128
do
    tgt="$PREFIX$name"
    echo "--- $tgt ---"
    ./"$tgt" || exit 1
done

if [ "$1" = "-slow" ]; then
    echo "==> Running benchmarks..."
    for name in benchmark_problem_double benchmark_problem_float128 benchmark_problem_dd128
    do
        tgt="$PREFIX$name"
        echo "--- $tgt ---"
        ./"$tgt" "[.]" || exit 1
    done
fi

echo "==> All tests passed."
