#pragma once

#include "xfac_quad/tree/tree.hpp"
#include "xfac_quad/cubemat_helper.h"
#include "xfac_quad/matrix/mat_decomp.h"

namespace xfac_quad {

/// Compress the bond from one node to its neighbor using a matrix decomposition.
template<typename Scalar, typename MatDecomp>
void compress_bond(std::vector<Tensor3D<Scalar>>& M, int from, int to,
                   TopologyTree const& tree, MatDecomp mat_decomp)
{
    const int from_pos = tree.neigh.at(from).pos(to);
    const int to_pos = tree.neigh.at(to).pos(from);
    auto ab = mat_decomp(cubeToMat_R(M.at(from), from_pos), true);
    const auto& M1 = ab[0];
    const typename Tensor3D<Scalar>::MatrixX M2 =
        ab[1] * cubeToMat_L(M.at(to), to_pos);

    std::vector<Eigen::Index> from_shape{
        M.at(from).n_rows, M.at(from).n_cols, M.at(from).n_slices};
    from_shape.at(from_pos) = M1.cols();
    std::vector<Eigen::Index> to_shape{
        M.at(to).n_rows, M.at(to).n_cols, M.at(to).n_slices};
    to_shape.at(to_pos) = M2.rows();
    M.at(from) = matToCube_R(M1, from_shape, from_pos);
    M.at(to) = matToCube_L(M2, to_shape, to_pos);
}

/// Evaluate a tensor tree over a leaves-to-root path, selecting the physical legs.
/// Physical node IDs index both M and id, as in the reference implementation.
template<typename Scalar>
Scalar mps_eval(std::vector<int> const& id,
                std::vector<Tensor3D<Scalar>> const& M,
                TopologyTree const& tree,
                std::vector<std::pair<int, int>> const& path,
                std::set<int> const& nodes)
{
    auto prod = M;
    for (int node : nodes)
        prod.at(node) = cube_eval(M.at(node), id.at(node));
    using Col = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    for (const auto& [from, to] : path) {
        const auto& source = prod.at(from);
        const Eigen::Map<const Col> v(source.data.data(),
                                      static_cast<Eigen::Index>(source.data.size()));
        const int pos = tree.neigh.at(to).pos(from);
        prod.at(to) = cube_vec(prod.at(to), v, pos);
    }
    // A single-node tree has no edges to traverse.
    const int root = path.empty() ? tree.root : path.back().second;
    return prod.at(root).data.at(0);
}

template<typename Scalar>
class TTree_sum;

template<typename Scalar>
struct TensorTree {
    using Real = typename Eigen::NumTraits<Scalar>::Real;

    TopologyTree tree;
    std::vector<Tensor3D<Scalar>> M;

    TensorTree()=default;
    TensorTree(TopologyTree const& tree_) : tree(tree_), M(tree_.size()) {
        if (!isValid(tree)) throw std::invalid_argument("TensorTree: tree is not valid");
    }

    /// Evaluate the full tensor tree at a given multi-index.
    Scalar eval(std::vector<int> const& id) const {
        if (id.size() != tree.nodes.size())
            throw std::invalid_argument("TensorTree.eval() id.size()!=tree.nodes.size()");
        return mps_eval(id, M, tree, tree.leavesToRoot(), tree.nodes);
    }

    Scalar operator()(std::vector<int> const& id) const { return eval(id); }

    /// Sum all physical-index combinations without forming the dense tensor.
    Scalar sum() const;

    /// Compress each directed bond in the full DFS sweep, including backtracking.
    template<typename MatDecomp>
    void compress_tree(MatDecomp mat_decomp) {
        for (const auto& [from, to] : tree.rootToLeaves())
            compress_bond(M, from, to, tree, mat_decomp);
    }

    /// Orthogonalize with QR, then truncate bonds using SVD.
    void compressSVD(Real reltol = Real(1e-12), int maxBondDim = 0) {
        compress_tree(MatQR<Scalar>{});
        compress_tree(MatSVDFixedTol<Scalar>{reltol, maxBondDim});
    }

    /// Prepare with rank-revealing LU, then apply the requested truncation.
    void compressLU(Real reltol = Real(1e-12), int maxBondDim = 0) {
        compress_tree(MatRRLUFixedTol<Scalar>{});
        compress_tree(MatRRLUFixedTol<Scalar>{reltol, maxBondDim});
    }

    /// Prepare with cross interpolation, then apply the requested truncation.
    void compressCI(Real reltol = Real(1e-12), int maxBondDim = 0) {
        compress_tree(MatCURFixedTol<Scalar>{});
        compress_tree(MatCURFixedTol<Scalar>{reltol, maxBondDim});
    }

    /// Save the topology, then the tensor-vector format used by TensorTrain.
    void save(std::ostream& out) const {
        tree.save(out);
        if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
            save_Tensor3D_to_arma(out, M);
        } else {
            // The existing I/O helpers expect complex scalars.
            std::vector<Tensor3D<std::complex<Real>>> stored;
            stored.reserve(M.size());
            for (const auto& tensor : M) {
                stored.emplace_back(tensor.n_rows, tensor.n_cols, tensor.n_slices);
                std::transform(tensor.data.begin(), tensor.data.end(), stored.back().data.begin(),
                               [](const Scalar& value) { return std::complex<Real>(value, Real(0)); });
            }
            save_Tensor3D_to_arma(out, stored);
        }
    }

    void save(std::string fileName) const {
        std::ofstream out(fileName);
        if (!out) throw std::runtime_error("TensorTree::save fails to open file: " + fileName);
        save(out);
        if (!out) throw std::runtime_error("TensorTree::save fails to write file: " + fileName);
    }

    static TensorTree load(std::ifstream& in) {
        TensorTree tt(TopologyTree::load(in));
        if constexpr (Eigen::NumTraits<Scalar>::IsComplex) {
            tt.M = load_vector_tensor<Scalar>(in);
        } else {
            const auto stored = load_vector_tensor<std::complex<Real>>(in);
            tt.M.clear();
            tt.M.reserve(stored.size());
            for (const auto& tensor : stored) {
                tt.M.emplace_back(tensor.n_rows, tensor.n_cols, tensor.n_slices);
                std::transform(tensor.data.begin(), tensor.data.end(), tt.M.back().data.begin(),
                               [](const std::complex<Real>& value) {
                                   if (value.imag() != Real(0))
                                       throw std::runtime_error("TensorTree::load: nonzero imaginary part for real scalar");
                                   return value.real();
                               });
            }
        }
        if (tt.M.size() != tt.tree.size())
            throw std::runtime_error("TensorTree::load: tensor count does not match topology");
        return tt;
    }

    static TensorTree load(std::string fileName) {
        std::ifstream in(fileName);
        if (!in) throw std::runtime_error("TensorTree::load fails to open file: " + fileName);
        return load(in);
    }
};

/// Sum physical legs with vectors of ones, then contract the tree toward its root.
template<typename Scalar>
class TTree_sum {
public:
    using Row = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Col = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    TopologyTree tree;
    int neigh = 0, root = 0;
    Row L;                              ///< accumulated root side of the final bond
    std::vector<Tensor3D<Scalar>> R;     ///< accumulated leaf-side tensors

    TTree_sum()=default;
    TTree_sum(TensorTree<Scalar> const& tt)
        : tree{tt.tree}, root{tt.tree.root}, R{tt.M}
    {
        auto path = tree.leavesToRoot();
        for (int node : tree.nodes)
            R.at(node) = sumPhysicalLeg(node, tt.M.at(node));

        if (path.empty()) {
            // A single physical node becomes a scalar after summing its physical leg.
            neigh = root;
            L = Row::Ones(1);
            return;
        }
        neigh = path.back().first;
        root = path.back().second;
        path.pop_back();
        for (const auto& [from, to] : path)
            R.at(to) = sumLeaveToRoot(from, to);
        // Form L after other root branches have been contracted, also for internal roots.
        L = sumRoot(root, neigh, R.at(root));
    }

    /// Sum the non-bond axes at from, leaving the bond toward to as a row vector.
    Row sumRoot(int from, int to, Tensor3D<Scalar> const& M) {
        const auto LM = cubeToMat_R(M, tree.neigh.at(from).pos(to));
        return Row::Ones(LM.rows()) * LM;
    }

    /// Sum the physical leg, always on axis 2, retaining a singleton axis.
    Tensor3D<Scalar> sumPhysicalLeg(int /*node*/, Tensor3D<Scalar> const& M) {
        return cube_vec(M, Col::Ones(M.n_slices), 2);
    }

    /// Contract the accumulated tensor at from into its neighbor to.
    Tensor3D<Scalar> sumLeaveToRoot(int from, int to) {
        const auto& source = R.at(from);
        const Eigen::Map<const Col> v(source.data.data(),
                                      static_cast<Eigen::Index>(source.data.size()));
        return cube_vec(R.at(to), v, tree.neigh.at(to).pos(from));
    }

    Scalar value() const {
        const auto& source = R.at(neigh);
        const Eigen::Map<const Col> v(source.data.data(),
                                      static_cast<Eigen::Index>(source.data.size()));
        // Ordinary matrix multiplication, not an inner product with conjugation.
        return (L * v)(0, 0);
    }
};

template<typename Scalar>
Scalar TensorTree<Scalar>::sum() const { return TTree_sum<Scalar>(*this).value(); }

}