#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <array>
#include <set>

#include "xfac_quad/xfac_quad.hpp"
#include "xfac_quad/tree/tensor_tree_ci.hpp"

TEMPLATE_TEST_CASE("get_T3 and get_TP1 agree with independent entrywise references on every node",
                   "[tensor_tree_ci][get_T3][get_TP1]", 
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    using namespace xfac_quad;
    using T = TestType;
    using Real = typename TensorTreeCI<T>::Real;
    using Mat = typename TensorTreeCI<T>::Mat;
    using std::abs;
    const bool use_cache = GENERATE(false, true);
    CAPTURE(use_cache);
    const std::vector<int> local_dimensions{3, 4, 5, 3, 4, 5};
    const auto function = [](std::vector<int> const& id) -> T {
        Real re = Real(0), im = Real(0);
        for (std::size_t p = 0; p < id.size(); ++p) {
            re += Real((p + 1)*(id[p] + 1)*(id[(p + 1) % id.size()] + 2)) / Real(p + 3);
            im += Real((id[p] + 2)*(id[(p + 2) % id.size()] + 1)) / Real(p + 7);
        }
        if constexpr (Eigen::NumTraits<T>::IsComplex)
            return T(pi<Real>() * re, im / pi<Real>());
        else
            return pi<Real>() * re;
    };

    // Set up only the data needed by get_T3; the algorithm constructor is not implemented yet.
    TensorTreeCI<T> ci;
    REQUIRE(ci.len() == 0);
    ci.tree = makeTuckerTree(3, 2);
    ci.tree.neigh.at(3) = IndexSet<int>({6, 0});
    ci.tree.neigh.at(6) = IndexSet<int>({5, 4, 3});
    ci.f = TensorFunction<T>{function, use_cache, false};
    ci.localSet.resize(local_dimensions.size());
    for (std::size_t node = 0; node < local_dimensions.size(); ++node)
        for (int s = 0; s < local_dimensions[node]; ++s) {
            MultiIndex index(local_dimensions.size(), 0);
            index[node] = static_cast<char32_t>((s + 1) % local_dimensions[node]);
            ci.localSet[node].push_back(index);
        }

    // Incoming bond sets have different ranks and full-length, disjoint subtree supports.
    for (const auto& [from, to] : ci.tree.rootToLeaves()) {
        const auto physical_nodes = ci.tree.split(from, to).first;
        std::size_t available = 1;
        for (int node : physical_nodes) available *= static_cast<std::size_t>(local_dimensions[node]);
        const auto rank = std::min(available, static_cast<std::size_t>(2 + (2*from + to) % 3));
        for (std::size_t pivot = 0; pivot < rank; ++pivot) {
            MultiIndex index(local_dimensions.size(), 0);
            auto remaining = pivot;
            for (int node : physical_nodes) {
                const auto dimension = static_cast<std::size_t>(local_dimensions[node]);
                index[node] = static_cast<char32_t>(remaining % dimension);
                remaining /= dimension;
            }
            ci.Iset[{from, to}].push_back(index);
        }
    }
    REQUIRE(ci.len() == local_dimensions.size());
    REQUIRE(ci.len() != ci.tree.size());
    const auto original_indices = ci.Iset;
    const auto original_local = ci.localSet;
    for (int node = 0; node < static_cast<int>(ci.tree.size()); ++node) {
        CAPTURE(node);
        const bool physical = ci.tree.nodes.contains(node);
        const auto& neighbors = ci.tree.neigh.at(node).from_int();
        std::array<Eigen::Index, 3> expected_shape{1, 1, 1};
        for (std::size_t axis = 0; axis < neighbors.size(); ++axis)
            expected_shape[axis] = static_cast<Eigen::Index>(ci.Iset.at({neighbors[axis], node}).size());
        if (physical) expected_shape[2] = local_dimensions[node];
        const auto tensor = ci.get_T3(node);
        REQUIRE(tensor.n_rows == expected_shape[0]);
        REQUIRE(tensor.n_cols == expected_shape[1]);
        REQUIRE(tensor.n_slices == expected_shape[2]);
        for (Eigen::Index k = 0; k < tensor.n_slices; ++k)
            for (Eigen::Index j = 0; j < tensor.n_cols; ++j)
                for (Eigen::Index i = 0; i < tensor.n_rows; ++i) {
                    CAPTURE(i, j, k);
                    // Independent oracle: assemble coordinates explicitly, without add() or f.eval().
                    const Eigen::Index axis_indices[] = {i, j, k};
                    std::vector<int> input(local_dimensions.size(), 0);
                    for (std::size_t axis = 0; axis < neighbors.size(); ++axis) {
                        const auto& partial = ci.Iset.at({neighbors[axis], node}).at(
                            static_cast<int>(axis_indices[axis]));
                        for (std::size_t p = 0; p < input.size(); ++p)
                            input[p] += static_cast<int>(partial[p]);
                    }
                    if (physical) {
                        const auto& local = ci.localSet[node].at(static_cast<int>(k));
                        for (std::size_t p = 0; p < input.size(); ++p)
                            input[p] += static_cast<int>(local[p]);
                    }
                    REQUIRE(tensor(i, j, k) == function(input));
                }

        // Exercise every directed bond, including all three artificial-node axes.
        for (int neighbor : neighbors) {
            CAPTURE(neighbor);
            const int axis = ci.tree.neigh.at(node).pos(neighbor);
            const auto rank = expected_shape[axis];
            Mat lower = Mat::Identity(rank, rank);
            Mat upper = Mat::Zero(rank, rank);
            Mat packed = Mat::Zero(rank, rank);
            for (Eigen::Index row = 0; row < rank; ++row)
                for (Eigen::Index col = 0; col < rank; ++col) {
                    const Real re = row == col ? Real(2 + row)
                        : Real(1 + row + 2*col) / Real(13 + row + col);
                    T value;
                    if constexpr (Eigen::NumTraits<T>::IsComplex)
                        value = T(re, Real(1 + row + col) / Real(19 + row + col));
                    else
                        value = re;
                    if (row > col) lower(row, col) = value;
                    else upper(row, col) = value;
                    packed(row, col) = value;
                }
            const Mat dense_pivot = lower * upper;
            const Mat inverse = dense_pivot.fullPivLu().inverse();
            ci.P[{node, neighbor}] = packed;
            const auto normalized = ci.get_TP1(node, neighbor);
            REQUIRE(normalized.n_rows == tensor.n_rows);
            REQUIRE(normalized.n_cols == tensor.n_cols);
            REQUIRE(normalized.n_slices == tensor.n_slices);
            const Real epsilon = Real(1000) * Eigen::NumTraits<T>::epsilon();
            for (Eigen::Index k = 0; k < tensor.n_slices; ++k)
                for (Eigen::Index j = 0; j < tensor.n_cols; ++j)
                    for (Eigen::Index i = 0; i < tensor.n_rows; ++i) {
                        CAPTURE(i, j, k);
                        const Eigen::Index output[] = {i, j, k};
                        T expected = T(0), reconstructed = T(0);
                        Real expected_scale = Real(1), reconstruction_scale = Real(1);
                        for (Eigen::Index b = 0; b < rank; ++b) {
                            Eigen::Index source[] = {i, j, k};
                            source[axis] = b;
                            // Direct tensor-index oracle; no reshape or triangular-solve helpers.
                            const T term = tensor(source[0], source[1], source[2])
                                         * inverse(b, output[axis]);
                            expected += term;
                            expected_scale += abs(term);
                            const T reconstruction_term = normalized(source[0], source[1], source[2])
                                                        * dense_pivot(b, output[axis]);
                            reconstructed += reconstruction_term;
                            reconstruction_scale += abs(reconstruction_term);
                        }
                        REQUIRE(abs(normalized(i, j, k) - expected) <= epsilon * expected_scale);
                        REQUIRE(abs(reconstructed - tensor(i, j, k)) <= epsilon * reconstruction_scale);
                    }
            REQUIRE(ci.P.at({node, neighbor}) == packed);
        }
    }
    REQUIRE(ci.Iset == original_indices);
    REQUIRE(ci.localSet == original_local);
}

TEST_CASE("addPivotsAt projects all binary pivots correctly on every bond",
          "[tensor_tree_ci][pivots]")
{
    using namespace xfac_quad;
    const auto tree = makeTuckerTree(4, 3); // 12 physical nodes and 2 artificial nodes
    constexpr int physical_count = 12;
    std::vector<std::vector<int>> pivots;
    for (int code = 0; code < (1 << physical_count); ++code) {
        std::vector<int> pivot(physical_count);
        for (int p = 0; p < physical_count; ++p) pivot[p] = (code >> p) & 1;
        pivots.push_back(pivot);
    }
    const std::vector<std::vector<int>> first_batch(pivots.begin(), pivots.begin() + 2048);
    const std::vector<std::vector<int>> second_batch(pivots.begin() + 2048, pivots.end());
    std::vector<std::pair<int, int>> bonds;
    for (const auto& [node, neighbors] : tree.neigh)
        for (int neighbor : neighbors.from_int())
            if (node < neighbor) bonds.emplace_back(node, neighbor);
    REQUIRE(bonds.size() == 13);

    for (const auto& [from, to] : bonds) {
        CAPTURE(from, to);
        // Independent partition oracle: traverse while blocking this edge, without tree.split().
        std::set<int> reached{from};
        std::vector<int> frontier{from};
        for (std::size_t next = 0; next < frontier.size(); ++next) {
            const int node = frontier[next];
            for (int neighbor : tree.neigh.at(node).from_int()) {
                if ((node == from && neighbor == to) || (node == to && neighbor == from)) continue;
                if (reached.insert(neighbor).second) frontier.push_back(neighbor);
            }
        }
        const auto project = [&](std::vector<int> const& pivot) {
            std::array<MultiIndex, 2> result{
                MultiIndex(physical_count, 0), MultiIndex(physical_count, 0)};
            for (int p = 0; p < physical_count; ++p)
                result[reached.contains(p) ? 0 : 1][p] = static_cast<char32_t>(pivot[p]);
            return result;
        };
        const auto seed = project(std::vector<int>(physical_count, 1));
        std::array<std::vector<MultiIndex>, 2> expected;
        std::array<std::set<MultiIndex>, 2> unique;
        for (int side = 0; side < 2; ++side) {
            expected[side].push_back(seed[side]);
            unique[side].insert(seed[side]);
        }
        for (const auto& pivot : pivots) {
            const auto projected = project(pivot);
            for (int side = 0; side < 2; ++side)
                if (unique[side].insert(projected[side]).second)
                    expected[side].push_back(projected[side]);
        }

        TensorTreeCI<double> ci;
        ci.tree = tree;
        ci.tt = TensorTree<double>(tree);
        for (std::size_t node = 0; node < ci.tt.M.size(); ++node) {
            ci.tt.M[node] = Tensor3D<double>(1, 1, 1);
            ci.tt.M[node].data[0] = static_cast<double>(node + 1);
        }
        ci.localSet.resize(physical_count);
        ci.Iset[{from, to}].push_back(seed[0]);
        ci.Iset[{to, from}].push_back(seed[1]);
        const std::pair<int, int> unrelated{100, 101};
        const MultiIndex unrelated_pivot(physical_count, 7);
        ci.Iset[unrelated].push_back(unrelated_pivot);
        ci.P[{from, to}] = Eigen::MatrixXd::Identity(2, 2);
        ci.pivotError = {3.0, 1.0};
        ci.cIter = 7;
        const auto original_tensors = ci.tt.M;
        const auto original_pivots = ci.P;
        const auto original_local = ci.localSet;

        ci.addPivotsAt(first_batch, from, to);
        ci.addPivotsAt(second_batch, from, to);
        ci.addPivotsAt(pivots, to, from); // reverse direction and repeated pivots must not reorder sets
        ci.addPivotsAt({}, from, to);

        const auto& left = ci.Iset.at({from, to});
        const auto& right = ci.Iset.at({to, from});
        REQUIRE(left.from_int() == expected[0]);
        REQUIRE(right.from_int() == expected[1]);
        int left_physical = 0;
        for (int node : tree.nodes) left_physical += reached.contains(node) ? 1 : 0;
        REQUIRE(left.size() == (std::size_t(1) << left_physical));
        REQUIRE(right.size() == (std::size_t(1) << (physical_count - left_physical)));
        for (const auto* indices : {&left, &right})
            for (std::size_t pos = 0; pos < indices->size(); ++pos)
                REQUIRE(indices->pos(indices->at(static_cast<int>(pos))) == static_cast<int>(pos));
        REQUIRE(ci.Iset.size() == 3);
        REQUIRE(ci.Iset.at(unrelated).from_int() == std::vector<MultiIndex>{unrelated_pivot});
        REQUIRE(ci.P == original_pivots);
        REQUIRE(ci.localSet == original_local);
        REQUIRE(ci.pivotError == std::vector<double>{3.0, 1.0});
        REQUIRE(ci.cIter == 7);
        REQUIRE(ci.tree.root == tree.root);
        REQUIRE(ci.tree.nodes == tree.nodes);
        REQUIRE(ci.tree.neigh == tree.neigh);
        REQUIRE(ci.tt.tree.root == tree.root);
        REQUIRE(ci.tt.tree.nodes == tree.nodes);
        REQUIRE(ci.tt.tree.neigh == tree.neigh);
        for (std::size_t node = 0; node < ci.tt.M.size(); ++node) {
            REQUIRE(ci.tt.M[node].n_rows == original_tensors[node].n_rows);
            REQUIRE(ci.tt.M[node].n_cols == original_tensors[node].n_cols);
            REQUIRE(ci.tt.M[node].n_slices == original_tensors[node].n_slices);
            REQUIRE(ci.tt.M[node].data == original_tensors[node].data);
        }
    }
}

TEMPLATE_TEST_CASE("collectPivotError retains rank-wise historical maxima at scalar precision",
                   "[tensor_tree_ci][pivot_errors]",
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    using namespace xfac_quad;
    using T = TestType;
    using Real = typename TensorTreeCI<T>::Real;
    struct ExposedCI : TensorTreeCI<T> {
        using TensorTreeCI<T>::collectPivotError;
    };
    ExposedCI ci;
    const Real delta = Real(16) * Eigen::NumTraits<T>::epsilon();
    ci.collectPivotError(0, 1, {Real(1), delta});
    ci.collectPivotError(1, 0, {Real(1) + delta, delta / Real(2), delta * delta});
    REQUIRE(ci.pivotError == std::vector<Real>{Real(1) + delta, delta, delta * delta});
    // Replacing a bond's current errors does not erase historical maxima or higher ranks.
    ci.collectPivotError(1, 0, {Real(0), Real(0)});
    ci.collectPivotError(0, 1, {});
    REQUIRE(ci.pivotError == std::vector<Real>{Real(1) + delta, delta, delta * delta});

    // Compare varying-length updates on many directed bonds against a sorted-history oracle.
    std::vector<std::vector<Real>> history(3);
    history[0] = {Real(1) + delta};
    history[1] = {delta};
    history[2] = {delta * delta};
    const auto tree = makeTuckerTree(4, 3);
    for (int sweep = 0; sweep < 5; ++sweep)
        for (const auto& [from, to] : tree.rootToLeaves()) {
            CAPTURE(sweep, from, to);
            const auto ranks = static_cast<std::size_t>(1 + (7*sweep + 3*from + to) % 41);
            std::vector<Real> errors(ranks);
            if (history.size() < ranks) history.resize(ranks);
            for (std::size_t rank = 0; rank < ranks; ++rank) {
                errors[rank] = Real(1) / Real(rank + 2)
                    + delta * Real(1 + (sweep + from + 2*to + static_cast<int>(rank)) % 17);
                history[rank].push_back(errors[rank]);
            }
            ci.collectPivotError(from, to, errors);
            REQUIRE(ci.pivotError.size() == history.size());
            for (std::size_t rank = 0; rank < history.size(); ++rank) {
                auto ordered = history[rank];
                REQUIRE_FALSE(ordered.empty());
                std::sort(ordered.begin(), ordered.end());
                REQUIRE(ci.pivotError[rank] == ordered.back());
            }
        }
}
