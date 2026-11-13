#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <random>
#include <vector>

#include "xfac_quad/xfac_quad.hpp"
#include "xfac_quad/tree/tensor_tree_ci.hpp"

namespace {

/// A reproducible sum of separable products, independent of the CI implementation:
/// F(i_0,...,i_n) = sum_r coefficient[r] * product_p factor[r][p][i_p].
/// Its exact tree ranks are bounded by the number of products, and its full sum
/// can be evaluated analytically without reconstructing or traversing a tree.
template<typename T>
struct GeneratedTensor {
    using Real = typename Eigen::NumTraits<T>::Real;
    std::vector<int> dimensions;
    std::vector<T> coefficients;
    std::vector<std::vector<std::vector<T>>> factors;

    GeneratedTensor(std::vector<int> dims, int rank, unsigned int seed)
        : dimensions(std::move(dims)), coefficients(rank), factors(rank)
    {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> draw(-8, 8);
        for (int r = 0; r < rank; ++r) {
            if constexpr (Eigen::NumTraits<T>::IsComplex)
                coefficients[r] = T(Real(r + 1)/Real(r + 2), Real(draw(rng))/Real(23));
            else
                coefficients[r] = Real(r + 1)/Real(r + 2);
            factors[r].resize(dimensions.size());
            for (std::size_t p = 0; p < dimensions.size(); ++p)
                for (int s = 0; s < dimensions[p]; ++s) {
                    const Real re = Real(10 + draw(rng))/Real(16);
                    if constexpr (Eigen::NumTraits<T>::IsComplex)
                        factors[r][p].emplace_back(re, Real(draw(rng))/Real(32));
                    else
                        factors[r][p].push_back(re);
                }
        }
    }

    T operator()(std::vector<int> const& index) const {
        T result = T(0);
        for (std::size_t r = 0; r < coefficients.size(); ++r) {
            T term = coefficients[r];
            for (std::size_t p = 0; p < index.size(); ++p)
                term *= factors[r][p][index[p]];
            result += term;
        }
        return result;
    }

    T exact_sum() const {
        T result = T(0);
        for (std::size_t r = 0; r < coefficients.size(); ++r) {
            T term = coefficients[r];
            for (const auto& factor : factors[r]) {
                T local_sum = T(0);
                for (const T& value : factor) local_sum += value;
                term *= local_sum;
            }
            result += term;
        }
        return result;
    }
};

} // namespace

TEMPLATE_TEST_CASE("Generated tensors are reconstructed by the complete TensorTreeCI process",
                   "[tensor_tree_ci][process][generated]",
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    using namespace xfac_quad;
    using T = TestType;
    using Real = typename Eigen::NumTraits<T>::Real;
    using std::abs;
    // Linear topology, one artificial junction, and two artificial junctions.
    const int topology = GENERATE(0, 1, 2);
    const int rank = GENERATE(2, 5);
    const unsigned int seed = GENERATE(7u, 41u);
    const bool cache = GENERATE(false, true);
    const int dim = topology == 0 ? 1 : topology == 1 ? 3 : 4;
    const int nbit = topology == 0 ? 6 : 2;
    const auto tree = makeTuckerTree(dim, nbit);
    std::vector<int> dimensions(tree.nodes.size());
    std::size_t dense_size = 1;
    for (std::size_t p = 0; p < dimensions.size(); ++p) {
        dimensions[p] = 2 + static_cast<int>(p % 2);
        dense_size *= static_cast<std::size_t>(dimensions[p]);
    }
    CAPTURE(dim, nbit, rank, seed, cache, dense_size);
    const GeneratedTensor<T> model(dimensions, rank, seed);
    std::vector<std::vector<int>> indices;
    std::vector<T> dense;
    Real scale = Real(0);
    for (std::size_t code = 0; code < dense_size; ++code) {
        auto remaining = code;
        std::vector<int> index(dimensions.size());
        for (std::size_t p = 0; p < dimensions.size(); ++p) {
            index[p] = static_cast<int>(remaining % static_cast<std::size_t>(dimensions[p]));
            remaining /= static_cast<std::size_t>(dimensions[p]);
        }
        indices.push_back(index);
        dense.push_back(model(index));
        const Real magnitude = abs(dense.back());
        if (magnitude > scale) scale = magnitude;
    }
    REQUIRE(scale > Real(0));

    TensorTreeCIParam<T> param;
    param.bondDim = 8;
    param.reltol = Real(500) * Eigen::NumTraits<T>::epsilon();
    param.fullPiv = true;
    param.useCachedFunction = cache;
    param.pivot1 = indices.at(seed % dense_size);
    TensorTreeCI<T> ci([&model](std::vector<int> index) { return model(index); },
                       tree, dimensions, param);
    REQUIRE(ci.cIter == 1);

    // Exercise global pivot insertion and its DMRG-0 rebuild before rank learning.
    ci.addPivotsAllBonds({indices.at((seed + 17) % dense_size), indices.back()});
    REQUIRE(ci.cIter == 2);
    std::size_t initial_rank = 0;
    for (const auto& [edge, pivots] : ci.Iset) {
        (void)edge;
        if (pivots.size() > initial_rank) initial_rank = pivots.size();
    }
    ci.iterate(5); // DMRG-2: history, candidate enrichment, CUR, and tensor updates
    REQUIRE(ci.cIter == 7);

    Real max_error = Real(0);
    for (std::size_t p = 0; p < dense.size(); ++p) {
        const Real error = abs(ci.tt.eval(indices[p]) - dense[p]);
        if (error > max_error) max_error = error;
    }
    const Real accuracy = Real(100) * param.reltol;
    CAPTURE(initial_rank, max_error, scale, accuracy);
    REQUIRE(max_error <= accuracy * scale);

    const T exact_sum = model.exact_sum();
    const Real sum_scale = abs(exact_sum);
    REQUIRE(sum_scale > Real(0));
    REQUIRE(abs(ci.tt.sum() - exact_sum) <= accuracy * sum_scale);

    std::size_t final_rank = 0;
    for (const auto& [edge, pivots] : ci.Iset) {
        CAPTURE(edge.first, edge.second);
        REQUIRE(pivots.size() <= static_cast<std::size_t>(rank));
        REQUIRE(pivots.size() <= static_cast<std::size_t>(param.bondDim));
        if (pivots.size() > final_rank) final_rank = pivots.size();
    }
    // A chain starts with at most three global pivots; rank five must be learned by DMRG-2.
    if (topology == 0 && rank == 5) REQUIRE(final_rank > initial_rank);

    // A final DMRG-0 sweep must preserve the learned approximation without growing ranks.
    ci.iterate(1, 0);
    REQUIRE(ci.cIter == 8);
    for (std::size_t p = 0; p < dense.size(); ++p)
        REQUIRE(abs(ci.tt.eval(indices[p]) - dense[p]) <= accuracy * scale);
    REQUIRE(abs(ci.tt.sum() - exact_sum) <= accuracy * sum_scale);
}
