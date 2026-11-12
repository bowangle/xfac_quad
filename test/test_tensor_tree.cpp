#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "xfac_quad/xfac_quad.hpp"
#include "xfac_quad/tree/tensor_tree.hpp"

namespace {

template<typename T>
T test_value(int real, int imag)
{
    if constexpr (Eigen::NumTraits<T>::IsComplex)
        return T(real, imag);
    else
        return T(real);
}

template<typename T>
void check_mps_eval(const char* type_name)
{
    using namespace xfac_quad;
    CAPTURE(type_name);

    SECTION("two physical nodes")
    {
        TopologyTree tree;
        tree.addEdge(0, 1);
        tree.nodes = {0, 1};
        TensorTree<T> tt(tree);
        tt.M[0] = Tensor3D<T>(2, 1, 2);
        tt.M[1] = Tensor3D<T>(2, 1, 3);
        for (int node = 0; node < 2; ++node)
            for (Eigen::Index k = 0; k < tt.M[node].n_slices; ++k)
                for (Eigen::Index i = 0; i < 2; ++i)
                    tt.M[node](i, 0, k) = test_value<T>(
                        static_cast<int>(1 + node + i + 2*k), static_cast<int>(1 + node + k));
        const auto original = tt.M;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j) {
                T expected = T(0);
                for (int bond = 0; bond < 2; ++bond)
                    expected += tt.M[0](bond, 0, i) * tt.M[1](bond, 0, j);
                REQUIRE(mps_eval<T>({i, j}, tt.M, tree, tree.leavesToRoot(), tree.nodes) == expected);
                REQUIRE(tt.eval({i, j}) == expected);
                REQUIRE(tt({i, j}) == expected);
            }
        REQUIRE_THROWS_AS(tt.eval({0}), std::invalid_argument);
        REQUIRE_THROWS_AS(tt.eval({0, 0, 0}), std::invalid_argument);
        REQUIRE_THROWS_AS(tt({0}), std::invalid_argument);
        for (std::size_t node = 0; node < tt.M.size(); ++node)
            REQUIRE(tt.M[node].data == original[node].data);
    }

    SECTION("branching tree with physical and artificial roots")
    {
        auto tree = makeTuckerTree(3, 1);
        TensorTree<T> tt(tree);
        for (int node = 0; node < 3; ++node) {
            tt.M[node] = Tensor3D<T>(2, 1, 2);
            for (int s = 0; s < 2; ++s)
                for (int bond = 0; bond < 2; ++bond)
                    tt.M[node](bond, 0, s) = test_value<T>(1 + node + bond + 2*s, 1 + node + s);
        }
        tt.M[3] = Tensor3D<T>(2, 2, 2);
        for (int k = 0; k < 2; ++k)
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i)
                    tt.M[3](i, j, k) = test_value<T>(1 + i + 2*j + 4*k, 1 + i + j + k);
        // The artificial node's neighbor order is [1, 0, 2].
        REQUIRE(tree.neigh.at(3).from_int() == std::vector<int>{1, 0, 2});
        const auto original = tt.M;
        for (int root : {0, 3}) {
            tree.root = root;
            tt.tree.root = root;
            for (int s0 = 0; s0 < 2; ++s0)
                for (int s1 = 0; s1 < 2; ++s1)
                    for (int s2 = 0; s2 < 2; ++s2) {
                        T expected = T(0);
                        for (int b0 = 0; b0 < 2; ++b0)
                            for (int b1 = 0; b1 < 2; ++b1)
                                for (int b2 = 0; b2 < 2; ++b2)
                                    expected += tt.M[0](b0, 0, s0) * tt.M[1](b1, 0, s1)
                                              * tt.M[2](b2, 0, s2) * tt.M[3](b1, b0, b2);
                        REQUIRE(mps_eval<T>({s0, s1, s2}, tt.M, tree,
                                            tree.leavesToRoot(), tree.nodes) == expected);
                        REQUIRE(tt.eval({s0, s1, s2}) == expected);
                        REQUIRE(tt({s0, s1, s2}) == expected);
                    }
        }
        for (std::size_t node = 0; node < tt.M.size(); ++node)
            REQUIRE(tt.M[node].data == original[node].data);
    }

    SECTION("single physical node")
    {
        TopologyTree tree;
        tree.neigh[0];
        tree.nodes = {0};
        TensorTree<T> tt(tree);
        tt.M[0] = Tensor3D<T>(1, 1, 2);
        tt.M[0](0, 0, 0) = test_value<T>(2, 3);
        tt.M[0](0, 0, 1) = test_value<T>(5, 7);
        for (int s = 0; s < 2; ++s) {
            REQUIRE(mps_eval<T>({s}, tt.M, tree, tree.leavesToRoot(), tree.nodes)
                    == tt.M[0](0, 0, s));
            REQUIRE(tt.eval({s}) == tt.M[0](0, 0, s));
            REQUIRE(tt({s}) == tt.M[0](0, 0, s));
        }
    }
}

template<typename T>
void check_compress_bond()
{
    using namespace xfac_quad;
    using Mat = typename Tensor3D<T>::MatrixX;
    using Real = typename Eigen::NumTraits<T>::Real;
    using std::abs;
    for (int from_pos = 0; from_pos < 3; ++from_pos)
        for (int to_pos = 0; to_pos < 3; ++to_pos) {
            CAPTURE(from_pos, to_pos);
            TopologyTree tree;
            int leaf = 2;
            for (int p = 0; p < from_pos; ++p) tree.addEdge(0, leaf++);
            for (int p = 0; p < to_pos; ++p) tree.addEdge(1, leaf++);
            tree.addEdge(0, 1);
            for (int p = from_pos; p < 2; ++p) tree.addEdge(0, leaf++);
            for (int p = to_pos; p < 2; ++p) tree.addEdge(1, leaf++);
            REQUIRE(isValid(tree));
            REQUIRE(tree.neigh.at(0).pos(1) == from_pos);
            REQUIRE(tree.neigh.at(1).pos(0) == to_pos);

            std::vector<Eigen::Index> from_shape{2, 2, 2};
            std::vector<Eigen::Index> to_shape{3, 3, 3};
            from_shape[from_pos] = 5;
            to_shape[to_pos] = 5;
            std::vector<Tensor3D<T>> tensors(tree.size(), Tensor3D<T>(1, 1, 1));
            tensors[0] = Tensor3D<T>(from_shape[0], from_shape[1], from_shape[2]);
            tensors[1] = Tensor3D<T>(to_shape[0], to_shape[1], to_shape[2]);
            for (int node = 0; node < 2; ++node)
                for (std::size_t n = 0; n < tensors[node].data.size(); ++n)
                    tensors[node].data[n] = test_value<T>(
                        static_cast<int>((n*n + 3*n + node) % 17) - 8,
                        static_cast<int>((2*n + node) % 7) - 3);
            const auto original = tensors;
            const Mat before = cubeToMat_R(tensors[0], from_pos)
                             * cubeToMat_L(tensors[1], to_pos);

            compress_bond(tensors, 0, 1, tree, MatQR<T>{});

            const Mat after = cubeToMat_R(tensors[0], from_pos)
                            * cubeToMat_L(tensors[1], to_pos);
            Real scale = before.cwiseAbs().maxCoeff();
            if (scale < Real(1)) scale = Real(1);
            const Real tolerance = Real(1000) * Eigen::NumTraits<T>::epsilon() * scale;
            REQUIRE((after - before).cwiseAbs().maxCoeff() <= tolerance);
            const Eigen::Index new_from_shape[] = {
                tensors[0].n_rows, tensors[0].n_cols, tensors[0].n_slices};
            const Eigen::Index new_to_shape[] = {
                tensors[1].n_rows, tensors[1].n_cols, tensors[1].n_slices};
            for (int p = 0; p < 3; ++p) {
                REQUIRE(new_from_shape[p] == (p == from_pos ? 4 : from_shape[p]));
                REQUIRE(new_to_shape[p] == (p == to_pos ? 4 : to_shape[p]));
            }
            for (std::size_t node = 2; node < tensors.size(); ++node)
                REQUIRE(tensors[node].data == original[node].data);
        }
}

} // namespace

TEST_CASE("TensorTree construction", "[tensor_tree][construction]")
{
    const xfac_quad::TensorTree<double> empty;
    REQUIRE(empty.M.empty());

    const auto tree = xfac_quad::makeTuckerTree(4, 2);
    const xfac_quad::TensorTree<double> tt(tree);
    REQUIRE(tt.M.size() == tree.size());
    REQUIRE(tt.tree.root == tree.root);
    REQUIRE(tt.tree.nodes == tree.nodes);
    REQUIRE(tt.tree.neigh == tree.neigh);
    REQUIRE_THROWS_AS(xfac_quad::TensorTree<double>(xfac_quad::TopologyTree{}), std::invalid_argument);
}

TEMPLATE_TEST_CASE("Tensor tree mps_eval", "[tensor_tree][eval]",
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    check_mps_eval<TestType>("scalar under test");
}

TEMPLATE_TEST_CASE("Bond compression preserves contraction for every axis pair",
                   "[tensor_tree][compression]",
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    check_compress_bond<TestType>();
}

TEMPLATE_TEST_CASE("Tree compression preserves all evaluations",
                   "[tensor_tree][compression]",
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    using namespace xfac_quad;
    using T = TestType;
    using Real = typename Eigen::NumTraits<T>::Real;
    using Mat = typename Tensor3D<T>::MatrixX;
    using std::abs;
    for (int root : {0, 6}) {
        CAPTURE(root);
        auto tree = makeTuckerTree(3, 2);
        tree.root = root;
        TensorTree<T> tt(tree);
        for (std::size_t node = 0; node < tt.M.size(); ++node) {
            const bool physical = tree.nodes.count(static_cast<int>(node)) != 0;
            tt.M[node] = Tensor3D<T>(3,
                physical && tree.neigh.at(static_cast<int>(node)).size() == 1 ? 1 : 3,
                physical ? 2 : 3);
            for (std::size_t n = 0; n < tt.M[node].data.size(); ++n)
                tt.M[node].data[n] = test_value<T>(
                    static_cast<int>((n*n + 3*n + node) % 11) - 5,
                    static_cast<int>((2*n + node) % 5) - 2);
        }
        std::vector<std::vector<int>> indices;
        std::vector<T> expected;
        Real scale = Real(1);
        for (int index = 0; index < 64; ++index) {
            std::vector<int> id(6);
            for (int node = 0; node < 6; ++node) id[node] = (index >> node) & 1;
            indices.push_back(id);
            expected.push_back(tt.eval(id));
            const Real magnitude = abs(expected.back());
            if (magnitude > scale) scale = magnitude;
        }
        std::size_t calls = 0;
        tt.compress_tree([&](Mat const& matrix, bool left_orthogonal) {
            ++calls;
            REQUIRE(left_orthogonal);
            return MatQR<T>{}(matrix, left_orthogonal);
        });
        REQUIRE(calls == 2 * (tree.size() - 1));
        REQUIRE(tt.tree.root == tree.root);
        REQUIRE(tt.tree.nodes == tree.nodes);
        REQUIRE(tt.tree.neigh == tree.neigh);
        const Real tolerance = Real(10000) * Eigen::NumTraits<T>::epsilon() * scale;
        for (std::size_t i = 0; i < indices.size(); ++i)
            REQUIRE(abs(tt.eval(indices[i]) - expected[i]) <= tolerance);
        for (int node : tree.nodes) REQUIRE(tt.M[node].n_slices == 2);
        for (const auto& [from, to] : tree.leavesToRoot()) {
            const Eigen::Index from_shape[] = {
                tt.M[from].n_rows, tt.M[from].n_cols, tt.M[from].n_slices};
            const Eigen::Index to_shape[] = {
                tt.M[to].n_rows, tt.M[to].n_cols, tt.M[to].n_slices};
            REQUIRE(from_shape[tree.neigh.at(from).pos(to)]
                    == to_shape[tree.neigh.at(to).pos(from)]);
        }
    }
}

TEST_CASE("Single-node tree compression has no bonds to update", "[tensor_tree][compression]")
{
    xfac_quad::TopologyTree tree;
    tree.neigh[0];
    tree.nodes = {0};
    xfac_quad::TensorTree<double> tt(tree);
    tt.M[0] = xfac_quad::Tensor3D<double>(1, 1, 2);
    tt.M[0].data = {2.0, 5.0};
    const auto original = tt.M[0].data;
    int calls = 0;
    tt.compress_tree([&](Eigen::MatrixXd const& matrix, bool left_orthogonal) {
        ++calls;
        return xfac_quad::MatQR<double>{}(matrix, left_orthogonal);
    });
    REQUIRE(calls == 0);
    REQUIRE(tt.M[0].data == original);
}

TEMPLATE_TEST_CASE("TensorTree compression wrappers", "[tensor_tree][compression][wrappers]",
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    using namespace xfac_quad;
    using T = TestType;
    using Tree = TensorTree<T>;
    using Real = typename Tree::Real;
    using Method = void (Tree::*)(Real, int);
    using std::abs;
    const Method methods[] = {&Tree::compressSVD, &Tree::compressLU, &Tree::compressCI};
    const char* names[] = {"SVD", "LU", "CI"};
    const int method = GENERATE(0, 1, 2);
    {
        CAPTURE(names[method]);
        const Real tolerance = Real(10000) * Eigen::NumTraits<T>::epsilon();

        SECTION("rank-one branching tree preserves evaluation and shrinks all bonds")
        {
            const auto tree = makeTuckerTree(3, 1);
            Tree tt(tree);
            for (int node = 0; node < 3; ++node) {
                tt.M[node] = Tensor3D<T>(3, 1, 2);
                for (int s = 0; s < 2; ++s)
                    for (int b = 0; b < 3; ++b)
                        tt.M[node](b, 0, s) = test_value<T>(1 + node + b + 2*s, 1 + node + s);
            }
            tt.M[3] = Tensor3D<T>(3, 3, 3);
            for (int k = 0; k < 3; ++k)
                for (int j = 0; j < 3; ++j)
                    for (int i = 0; i < 3; ++i)
                        tt.M[3](i, j, k) = T((i + 1)*(j + 1)*(k + 1));
            std::vector<T> before;
            Real scale = Real(1);
            for (int s = 0; s < 8; ++s) {
                before.push_back(tt.eval({s & 1, (s >> 1) & 1, (s >> 2) & 1}));
                const Real magnitude = abs(before.back());
                if (magnitude > scale) scale = magnitude;
            }
            (tt.*methods[method])(tolerance, 0);
            for (int s = 0; s < 8; ++s)
                REQUIRE(abs(tt.eval({s & 1, (s >> 1) & 1, (s >> 2) & 1}) - before[s])
                        <= tolerance * scale);
            for (int node = 0; node < 3; ++node) {
                REQUIRE(tt.M[node].n_rows == 1);
                REQUIRE(tt.M[node].n_cols == 1);
                REQUIRE(tt.M[node].n_slices == 2);
            }
            REQUIRE(tt.M[3].n_rows == 1);
            REQUIRE(tt.M[3].n_cols == 1);
            REQUIRE(tt.M[3].n_slices == 1);
        }

        SECTION("full-rank tensor is preserved without truncation")
        {
            TopologyTree tree;
            tree.addEdge(0, 1);
            tree.nodes = {0, 1};
            Tree tt(tree);
            for (auto& tensor : tt.M) {
                tensor = Tensor3D<T>(3, 1, 3);
                for (int i = 0; i < 3; ++i) tensor(i, 0, i) = T(1);
            }
            (tt.*methods[method])(tolerance, 0);
            REQUIRE(tt.M[0].n_rows == 3);
            REQUIRE(tt.M[1].n_rows == 3);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    REQUIRE(abs(tt.eval({i, j}) - T(i == j ? 1 : 0)) <= tolerance);

            (tt.*methods[method])(tolerance, 1);
            REQUIRE(tt.M[0].n_rows == 1);
            REQUIRE(tt.M[1].n_rows == 1);
        }

        SECTION("relative tolerance removes a small component")
        {
            TopologyTree tree;
            tree.addEdge(0, 1);
            tree.nodes = {0, 1};
            Tree tt(tree);
            for (auto& tensor : tt.M) {
                tensor = Tensor3D<T>(3, 1, 3);
                for (int i = 0; i < 3; ++i) tensor(i, 0, i) = T(1);
            }
            tt.M[0](1, 0, 1) = T(Real(1) / Real(1000));
            tt.M[0](2, 0, 2) = T(Real(1) / Real(1000000));
            (tt.*methods[method])(Real(1) / Real(10000), 0);
            REQUIRE(tt.M[0].n_rows == 2);
            REQUIRE(tt.M[1].n_rows == 2);
            REQUIRE(abs(tt.eval({0, 0}) - T(1)) <= tolerance);
            REQUIRE(abs(tt.eval({1, 1}) - T(Real(1) / Real(1000))) <= tolerance);
            REQUIRE(abs(tt.eval({2, 2})) <= tolerance);
        }
    }
}
