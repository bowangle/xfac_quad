#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory '$BUILD_DIR' not found."
    exit 1
fi

cd "$BUILD_DIR"

echo "==> Running xfac_quad tests..."

for tgt in \
    Test_SVDDecomp_double Test_SVDDecomp_float128 Test_SVDDecomp_dd128 \
    Test_matrix_ci_double Test_matrix_ci_float128 Test_matrix_ci_dd128 \
    Test_tensor_ci_double Test_tensor_ci_float128 Test_tensor_ci_dd128
do
    echo "--- $tgt ---"
    ./"$tgt" || exit 1
done

if [ "$1" = "-slow" ]; then
    echo "==> Running benchmarks..."
    ./Test_benchmark_double "[.]" || exit 1
    ./Test_benchmark_float128 "[.]" || exit 1
    ./Test_benchmark_dd128 "[.]" || exit 1
fi

echo "==> All tests passed."
