#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <complex>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "xfac_quad/xfac_quad.hpp"

using namespace xfac_quad;

namespace {

template <typename Real>
std::vector<Tensor3D<std::complex<Real>>> make_test_cores()
{
    using Complex = std::complex<Real>;

    const Real pi_value = pi<Real>();
    std::vector<Tensor3D<Complex>> cores;
    cores.emplace_back(1, 2, 2);
    cores.emplace_back(2, 1, 1);

    std::size_t index = 0;
    for (auto& core : cores) {
        for (auto& value : core.data) {
            const Real re = pi_value / Real(3 + index);
            const Real im = -pi_value / Real(11 + index);
            value = Complex(re, im);
            ++index;
        }
    }

    return cores;
}

template <typename T, typename U>
void require_same_shapes(const std::vector<Tensor3D<T>>& lhs,
                         const std::vector<Tensor3D<U>>& rhs)
{
    REQUIRE(lhs.size() == rhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        REQUIRE(lhs[i].n_rows == rhs[i].n_rows);
        REQUIRE(lhs[i].n_cols == rhs[i].n_cols);
        REQUIRE(lhs[i].n_slices == rhs[i].n_slices);
        REQUIRE(lhs[i].data.size() == rhs[i].data.size());
    }
}

template <typename Real>
void require_exact_roundtrip(const char* type_name)
{
    using Complex = std::complex<Real>;
    CAPTURE(type_name);

    const auto original = make_test_cores<Real>();

    std::ostringstream output;
    save_Tensor3D_to_arma(output, original);
    const std::string serialized = output.str();

    if constexpr (std::is_same_v<Real, dd_128>) {
        REQUIRE(serialized.find("XFQD_DD128_EXACT_V1") != std::string::npos);

        bool has_nonzero_low_component = false;
        for (const auto& core : original) {
            for (const auto& value : core.data) {
                has_nonzero_low_component =
                    has_nonzero_low_component ||
                    value.real().x[1] != 0.0 || value.imag().x[1] != 0.0;
            }
        }
        REQUIRE(has_nonzero_low_component);
    } else {
        REQUIRE(serialized.find("XFQD_DD128_EXACT_V1") == std::string::npos);
    }

    std::istringstream input(serialized);
    const auto loaded = load_vector_tensor<Complex>(input);
    require_same_shapes(original, loaded);

    for (std::size_t c = 0; c < original.size(); ++c) {
        for (std::size_t i = 0; i < original[c].data.size(); ++i) {
            REQUIRE(loaded[c].data[i] == original[c].data[i]);
        }
    }
}

template<typename T>
T helper_test_value(int real, int imag)
{
    if constexpr (Eigen::NumTraits<T>::IsComplex)
        return T(real, imag);
    else
        return T(real);
}

template<typename T>
void check_cube_helpers(const char* type_name)
{
    CAPTURE(type_name);
    Tensor3D<T> cube(2, 3, 4);
    for (Eigen::Index k = 0; k < cube.n_slices; ++k)
        for (Eigen::Index j = 0; j < cube.n_cols; ++j)
            for (Eigen::Index i = 0; i < cube.n_rows; ++i)
                cube(i, j, k) = helper_test_value<T>(
                    static_cast<int>(1 + i + 2*j + 6*k), static_cast<int>(1 + i + j + k));
    const auto original_data = cube.data;

    for (int slice : {0, 3}) {
        auto evaluated = cube_eval(cube, slice);
        REQUIRE(evaluated.n_rows == 2);
        REQUIRE(evaluated.n_cols == 3);
        REQUIRE(evaluated.n_slices == 1);
        for (Eigen::Index j = 0; j < cube.n_cols; ++j)
            for (Eigen::Index i = 0; i < cube.n_rows; ++i)
                REQUIRE(evaluated(i, j, 0) == cube(i, j, slice));
        evaluated(0, 0, 0) = T(0);
        REQUIRE(cube.data == original_data);
    }
    REQUIRE_THROWS_AS(cube_eval(cube, -1), std::out_of_range);
    REQUIRE_THROWS_AS(cube_eval(cube, 4), std::out_of_range);

    using Col = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    const Eigen::Index dimensions[] = {2, 3, 4};
    const std::vector<Eigen::Index> shape{2, 3, 4};
    for (int axis = 0; axis < 3; ++axis) {
        CAPTURE(axis);
        const auto right = cubeToMat_R(cube, axis);
        const auto left = cubeToMat_L(cube, axis);
        REQUIRE(right.rows() == 24 / dimensions[axis]);
        REQUIRE(right.cols() == dimensions[axis]);
        REQUIRE(left.rows() == right.cols());
        REQUIRE(left.cols() == right.rows());
        for (Eigen::Index k = 0; k < 4; ++k)
            for (Eigen::Index j = 0; j < 3; ++j)
                for (Eigen::Index i = 0; i < 2; ++i) {
                    const Eigen::Index packed = axis == 0 ? j + 3*k
                                                     : axis == 1 ? i + 2*k : i + 2*j;
                    const Eigen::Index bond = axis == 0 ? i : axis == 1 ? j : k;
                    REQUIRE(right(packed, bond) == cube(i, j, k));
                    REQUIRE(left(bond, packed) == cube(i, j, k));
                }
        const auto restored_right = matToCube_R(right, shape, axis);
        const auto restored_left = matToCube_L(left, shape, axis);
        REQUIRE(restored_right.n_rows == 2);
        REQUIRE(restored_right.n_cols == 3);
        REQUIRE(restored_right.n_slices == 4);
        REQUIRE(restored_left.n_rows == 2);
        REQUIRE(restored_left.n_cols == 3);
        REQUIRE(restored_left.n_slices == 4);
        REQUIRE(restored_right.data == original_data);
        REQUIRE(restored_left.data == original_data);
        REQUIRE_THROWS_AS(matToCube_R(right, {1, 3, 4}, axis), std::invalid_argument);
        REQUIRE_THROWS_AS(matToCube_L(left, {1, 3, 4}, axis), std::invalid_argument);

        Col weights(dimensions[axis]);
        for (Eigen::Index i = 0; i < weights.size(); ++i)
            weights(i) = helper_test_value<T>(static_cast<int>(i + 1), static_cast<int>(i + 2));
        const auto contracted = cube_vec(cube, weights, axis);
        REQUIRE(contracted.n_rows == (axis == 0 ? 1 : 2));
        REQUIRE(contracted.n_cols == (axis == 1 ? 1 : 3));
        REQUIRE(contracted.n_slices == (axis == 2 ? 1 : 4));
        for (Eigen::Index k = 0; k < contracted.n_slices; ++k)
            for (Eigen::Index j = 0; j < contracted.n_cols; ++j)
                for (Eigen::Index i = 0; i < contracted.n_rows; ++i) {
                    T expected = T(0);
                    for (Eigen::Index n = 0; n < dimensions[axis]; ++n) {
                        Eigen::Index indices[] = {i, j, k};
                        indices[axis] = n;
                        expected += cube(indices[0], indices[1], indices[2]) * weights(n);
                    }
                    REQUIRE(contracted(i, j, k) == expected);
                }
        const Eigen::Map<const Col> mapped_weights(weights.data(), weights.size());
        REQUIRE(cube_vec(cube, mapped_weights, axis).data == contracted.data);
        REQUIRE(cube.data == original_data);

        const Col wrong_size = Col::Ones(dimensions[axis] + 1);
        REQUIRE_THROWS_AS(cube_vec(cube, wrong_size, axis), std::invalid_argument);
    }
    const Col weights = Col::Ones(2);
    REQUIRE_THROWS_AS(cube_vec(cube, weights, -1), std::invalid_argument);
    REQUIRE_THROWS_AS(cube_vec(cube, weights, 3), std::invalid_argument);
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Ones(2, 2);
    REQUIRE_THROWS_AS(cube_vec(cube, matrix, 0), std::invalid_argument);
    for (int axis : {-1, 3}) {
        REQUIRE_THROWS_AS(cubeToMat_R(cube, axis), std::invalid_argument);
        REQUIRE_THROWS_AS(cubeToMat_L(cube, axis), std::invalid_argument);
        REQUIRE_THROWS_AS(matToCube_R(matrix, shape, axis), std::invalid_argument);
        REQUIRE_THROWS_AS(matToCube_L(matrix, shape, axis), std::invalid_argument);
    }
    REQUIRE_THROWS_AS(matToCube_R(matrix, {2, 3}, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(matToCube_L(matrix, {2, 3}, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(matToCube_R(matrix, {-2, 3, 4}, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(matToCube_L(matrix, {-2, 3, 4}, 0), std::invalid_argument);
}

} // namespace

TEST_CASE("Tensor3D cube evaluation and vector contraction", "[cubemat][helpers]")
{
    check_cube_helpers<double>("double");
    check_cube_helpers<float128>("float128");
    check_cube_helpers<dd_128>("dd_128");
    check_cube_helpers<std::complex<double>>("complex<double>");
    check_cube_helpers<std::complex<float128>>("complex<float128>");
    check_cube_helpers<std::complex<dd_128>>("complex<dd_128>");
}

TEST_CASE("Tensor3D text I/O is exactly lossless for every supported real type",
          "[cubemat][io][roundtrip]")
{
    require_exact_roundtrip<double>("double");
    require_exact_roundtrip<float128>("float128");
    require_exact_roundtrip<dd_128>("dd_128");
}

TEST_CASE("dd_128 TT keeps the legacy decimal representation readable as double",
          "[cubemat][io][dd128][compatibility]")
{
    const auto original = make_test_cores<dd_128>();

    std::ostringstream output;
    save_Tensor3D_to_arma(output, original);

    std::istringstream input(output.str());
    const auto loaded_as_double =
        load_vector_tensor<std::complex<double>>(input);

    require_same_shapes(original, loaded_as_double);
}

TEST_CASE("dd_128 loader still accepts legacy TT files without an exact trailer",
          "[cubemat][io][dd128][legacy]")
{
    using Complex = Cdd_128;
    const auto original = make_test_cores<dd_128>();

    std::ostringstream output;
    save_Tensor3D_to_arma(output, original);
    std::string legacy = output.str();

    const std::size_t marker = legacy.find("XFQD_DD128_EXACT_V1");
    REQUIRE(marker != std::string::npos);
    legacy.erase(marker);

    std::istringstream input(legacy);
    const auto loaded = load_vector_tensor<Complex>(input);
    require_same_shapes(original, loaded);
}
