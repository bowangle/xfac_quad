#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <chrono>
#include <filesystem>
#include <sstream>

#include "xfac_quad/xfac_quad.hpp"
#include "xfac_quad/tree/tensor_tree.hpp"

namespace {

struct TemporaryTreeFile {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("xfac_quad_tensor_tree_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");

    ~TemporaryTreeFile() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

} // namespace

TEMPLATE_TEST_CASE("TensorTree save/load preserves topology and tensor values exactly",
                   "[tensor_tree][io][roundtrip]",
                   double, float128, dd_128,
                   std::complex<double>, std::complex<float128>, std::complex<dd_128>)
{
    using namespace xfac_quad;
    using T = TestType;
    using Real = typename TensorTree<T>::Real;
    const bool use_filename = GENERATE(false, true);
    CAPTURE(use_filename);
    TemporaryTreeFile file;
    auto tree = makeTuckerTree(4, 1);
    tree.root = 4;
    tree.neigh.at(4) = IndexSet<int>({5, 1, 0});
    TensorTree<T> original(tree);
    for (std::size_t node = 0; node < original.M.size(); ++node) {
        const bool physical = tree.nodes.count(static_cast<int>(node)) != 0;
        auto& tensor = original.M[node];
        tensor = Tensor3D<T>(2, physical ? 1 : 2, 2);
        for (std::size_t n = 0; n < tensor.data.size(); ++n) {
            const Real re = pi<Real>() / Real(3 + node + n);
            if constexpr (Eigen::NumTraits<T>::IsComplex)
                tensor.data[n] = T(re, -pi<Real>() / Real(11 + node + n));
            else
                tensor.data[n] = re;
        }
    }
    const auto loaded = [&]() {
        if (use_filename) {
            original.save(file.path.string());
            return TensorTree<T>::load(file.path.string());
        }
        std::ostringstream output;
        original.save(output);
        {
            std::ofstream out(file.path);
            REQUIRE(out.good());
            out << output.str();
        }
        std::ifstream in(file.path);
        REQUIRE(in.good());
        return TensorTree<T>::load(in);
    }();
    REQUIRE(loaded.tree.root == original.tree.root);
    REQUIRE(loaded.tree.nodes == original.tree.nodes);
    REQUIRE(loaded.tree.neigh == original.tree.neigh);
    REQUIRE(loaded.M.size() == original.M.size());
    for (std::size_t node = 0; node < original.M.size(); ++node) {
        REQUIRE(loaded.M[node].n_rows == original.M[node].n_rows);
        REQUIRE(loaded.M[node].n_cols == original.M[node].n_cols);
        REQUIRE(loaded.M[node].n_slices == original.M[node].n_slices);
        REQUIRE(loaded.M[node].data == original.M[node].data);
    }
    for (int s = 0; s < 16; ++s) {
        const std::vector<int> id{s & 1, (s >> 1) & 1, (s >> 2) & 1, (s >> 3) & 1};
        REQUIRE(loaded.eval(id) == original.eval(id));
    }
}

TEST_CASE("TensorTree I/O rejects inconsistent tensor counts and missing files", "[tensor_tree][io]")
{
    using namespace xfac_quad;
    TemporaryTreeFile file;
    REQUIRE_THROWS_AS(TensorTree<std::complex<double>>::load(file.path.string()), std::runtime_error);
    {
        std::ofstream out(file.path);
        makeTuckerTree(3, 1).save(out);
        save_Tensor3D_to_arma(out, std::vector<Tensor3D<std::complex<double>>>{});
    }
    REQUIRE_THROWS_AS(TensorTree<std::complex<double>>::load(file.path.string()), std::runtime_error);

    const TensorTree<double> tt(makeTuckerTree(2, 1));
    REQUIRE_THROWS_AS(tt.save((file.path / "missing" / "tree.txt").string()), std::runtime_error);
}

TEST_CASE("Loading complex tensor data as real rejects nonzero imaginary parts", "[tensor_tree][io]")
{
    using namespace xfac_quad;
    TemporaryTreeFile file;
    TensorTree<std::complex<double>> tt(makeTuckerTree(2, 1));
    for (auto& tensor : tt.M) {
        tensor = Tensor3D<std::complex<double>>(1, 1, 1);
        tensor.data[0] = std::complex<double>(2, 1);
    }
    tt.save(file.path.string());
    REQUIRE_THROWS_AS(TensorTree<double>::load(file.path.string()), std::runtime_error);
}
