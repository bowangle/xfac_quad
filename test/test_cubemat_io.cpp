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

} // namespace

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
