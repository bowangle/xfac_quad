#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <array>
#include <vector>

#include "xfac_quad/xfac_quad.hpp"
#include "xfac_quad/tree/tensor_tree.hpp"

TEMPLATE_TEST_CASE("Tree sum agrees with an explicit sum over every physical and bond index",
                   "[tensor_tree][sum][dense]",
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    using namespace xfac_quad;
    using T = TestType;
    using Real = typename TensorTree<T>::Real;
    using std::abs;

    auto tree = makeTuckerTree(3, 2);
    tree.neigh.at(3) = IndexSet<int>({6, 0});
    tree.neigh.at(6) = IndexSet<int>({5, 4, 3});
    TensorTree<T> tt(tree);
    std::vector<std::array<Eigen::Index, 3>> shapes(tree.size(), {1, 1, 1});
    std::size_t physical_configurations = 1;
    for (int node : tree.nodes) {
        shapes[node][2] = 2 + node % 2;
        physical_configurations *= static_cast<std::size_t>(shapes[node][2]);
    }
    struct Bond {
        int from, to, from_pos, to_pos;
        Eigen::Index dimension;
    };
    std::vector<Bond> bonds;
    std::size_t bond_configurations = 1;
    for (const auto& [from, to] : tree.leavesToRoot()) {
        const Eigen::Index dim = 2 + static_cast<Eigen::Index>(bonds.size() % 2);
        const int from_pos = tree.neigh.at(from).pos(to);
        const int to_pos = tree.neigh.at(to).pos(from);
        shapes[from][from_pos] = dim;
        shapes[to][to_pos] = dim;
        bonds.push_back({from, to, from_pos, to_pos, dim});
        bond_configurations *= static_cast<std::size_t>(dim);
    }
    for (std::size_t node = 0; node < tt.M.size(); ++node) {
        tt.M[node] = Tensor3D<T>(shapes[node][0], shapes[node][1], shapes[node][2]);
        for (std::size_t n = 0; n < tt.M[node].data.size(); ++n) {
            const Real re = Real(static_cast<int>((n*n + 17*n + 3*node) % 13) - 6)
                          / Real(7 + (n + node) % 5);
            if constexpr (Eigen::NumTraits<T>::IsComplex) {
                const Real im = Real(static_cast<int>((3*n + node) % 11) - 5)
                              / Real(9 + (n + node) % 3);
                tt.M[node].data[n] = T(re, im);
            } else {
                tt.M[node].data[n] = re;
            }
        }
    }
    const auto original = tt.M;

    // Independent dense oracle: no eval(), cube_vec(), or reshape helpers.
    // Enumerate 216 physical configurations and 216 bond configurations.
    REQUIRE(physical_configurations == 216);
    REQUIRE(bond_configurations == 216);
    T expected = T(0);
    Real absolute_sum = Real(0);
    std::vector<std::array<Eigen::Index, 3>> legs(tree.size(), {0, 0, 0});
    for (std::size_t physical = 0; physical < physical_configurations; ++physical) {
        auto remaining = physical;
        for (int node : tree.nodes) {
            const auto dim = static_cast<std::size_t>(shapes[node][2]);
            legs[node][2] = static_cast<Eigen::Index>(remaining % dim);
            remaining /= dim;
        }
        for (std::size_t bond = 0; bond < bond_configurations; ++bond) {
            remaining = bond;
            for (const auto& edge : bonds) {
                const auto dim = static_cast<std::size_t>(edge.dimension);
                const auto index = static_cast<Eigen::Index>(remaining % dim);
                remaining /= dim;
                legs[edge.from][edge.from_pos] = index;
                legs[edge.to][edge.to_pos] = index;
            }
            T term = T(1);
            for (std::size_t node = 0; node < tt.M.size(); ++node)
                term *= tt.M[node](legs[node][0], legs[node][1], legs[node][2]);
            expected += term;
            absolute_sum += abs(term);
        }
    }
    const Real scale = absolute_sum > Real(1) ? absolute_sum : Real(1);
    const Real tolerance = Real(1000) * Eigen::NumTraits<T>::epsilon() * scale;
    // Every root: physical leaves, physical internal nodes, and the artificial junction.
    for (int root = 0; root < static_cast<int>(tree.size()); ++root) {
        CAPTURE(root, expected, tolerance);
        tt.tree.root = root;
        const TTree_sum<T> accumulator(tt);
        REQUIRE(abs(accumulator.value() - expected) <= tolerance);
        REQUIRE(abs(tt.sum() - expected) <= tolerance);
        REQUIRE(tt.tree.root == root);
        REQUIRE(tt.tree.nodes == tree.nodes);
        REQUIRE(tt.tree.neigh == tree.neigh);
        for (std::size_t node = 0; node < tt.M.size(); ++node)
            REQUIRE(tt.M[node].data == original[node].data);
    }
}

TEMPLATE_TEST_CASE("Single-node sum preserves a precision-scale residual after cancellation",
                   "[tensor_tree][sum][precision]",
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    using namespace xfac_quad;
    using T = TestType;
    using Real = typename TensorTree<T>::Real;
    TopologyTree tree;
    tree.neigh[0];
    tree.nodes = {0};
    TensorTree<T> tt(tree);
    tt.M[0] = Tensor3D<T>(1, 1, 3);
    const Real delta = Real(16) * Eigen::NumTraits<T>::epsilon();
    T expected;
    if constexpr (Eigen::NumTraits<T>::IsComplex) {
        tt.M[0].data = {T(1, 2), T(delta, delta), T(-1, -2)};
        expected = T(delta, delta);
    } else {
        tt.M[0].data = {T(1), T(delta), T(-1)};
        expected = T(delta);
    }
    REQUIRE(tt.sum() == expected);
    REQUIRE(TTree_sum<T>(tt).value() == expected);
}
