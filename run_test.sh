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
tree tensor_tree tensor_tree_ci tensor_tree_ci_process tensor_tree_io tensor_tree_sum \
cubemat_io \
SVDDecomp_double SVDDecomp_float128 SVDDecomp_dd128 \
matrix_ci_double matrix_ci_float128 matrix_ci_dd128 \
tensor_ci_double tensor_ci_float128 tensor_ci_dd128
do
    tgt="$PREFIX$name"
    echo "--- $tgt ---"
    ./"$tgt" || exit 1
done

if [ "$1" = "-slow" ]; then
    echo "==> Running 30-bit 1D exp(-r) OpenMP comparison..."
    ./"${PREFIX}tensor_tree_ci_parallel" "[ttci_parallel]" || exit 1

    echo "==> Running benchmarks..."
    for name in benchmark_problem_double benchmark_problem_float128 benchmark_problem_dd128
    do
        tgt="$PREFIX$name"
        echo "--- $tgt ---"
        ./"$tgt" "[.]" || exit 1
    done

    echo "==> Running MultQTGrid TensorCI2/TensorTreeCI comparisons..."
    for name in tensor_tree_ci_compare_double tensor_tree_ci_compare_float128 tensor_tree_ci_compare_dd128
    do
        tgt="$PREFIX$name"
        echo "--- $tgt ---"
        ./"$tgt" "[ttci_compare]" || exit 1
    done
fi

echo "==> All tests passed."
