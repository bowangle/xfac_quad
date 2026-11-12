#pragma once

#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "xfac_quad/index_set.h"
#include "xfac_quad/matrix/mat_decomp.h"
#include "xfac_quad/tensor/tensor_function.h"
#include "xfac_quad/tree/tensor_tree.hpp"

namespace xfac_quad {

/// Parameters of the TensorTreeCI algorithm.
template<typename Scalar>
struct TensorTreeCIParam {
    using Real = typename Eigen::NumTraits<Scalar>::Real;

    int bondDim = 30;                                ///< maximum bond dimension of the tensor tree
    Real reltol = Real(1e-12);                       ///< stop when pivotError < reltol*max(pivotError)
    std::vector<int> pivot1;                         ///< optional first pivot
    bool fullPiv = false;                           ///< search the full Pi matrix for a new pivot
    int nRookIter = 3;                               ///< rook pivoting iterations when fullPiv=false
    std::vector<std::vector<Real>> weight;           ///< TODO: weights for ENV learning
    std::function<bool(std::vector<int>)> cond;      ///< TODO: false when an index must not be a pivot
    bool useCachedFunction = true;                  ///< cache function evaluations at repeated points

    operator ARRLUParam<Scalar>() const {
        return {.reltol = reltol, .bondDim = bondDim,
                .fullPiv = fullPiv, .nRookIter = nRookIter};
    }
};

/// Build a cross interpolation of an input tensor on a tree topology.
template<typename Scalar>
struct TensorTreeCI {
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Cube = Tensor3D<Scalar>;
    using Real = typename Eigen::NumTraits<Scalar>::Real;

    TopologyTree tree;
    TensorFunction<Scalar> f;                        ///< input tensor function
    TensorFunction<unsigned int> fCond;              ///< condition function: 1 for allowed pivots, 0 otherwise
    TensorTreeCIParam<Scalar> param;                 ///< algorithm parameters
    std::vector<Real> pivotError;                    ///< maximum pivot error for each rank
    std::vector<IndexSet<MultiIndex>> localSet;       ///< physical multi-index set for each site
    std::map<std::pair<int, int>, IndexSet<MultiIndex>> Iset; ///< pivot indices for each directed bond
    TensorTree<Scalar> tt;                           ///< main output of the cross interpolation
    std::map<std::pair<int, int>, Mat> P;            ///< pivot matrix in LU form for each directed bond
    int cIter = 0;                                  ///< sweep iteration counter

    TensorTreeCI()=default;

    /// Initialize the cross from f and a first pivot, enriching around artificial nodes.
    /// Each physical index lies in [0,localDim[i]); the topology is validated by tt.
    TensorTreeCI(std::function<Scalar(std::vector<int>)> const& f_, TopologyTree tree_,
                 std::vector<int> localDim, TensorTreeCIParam<Scalar> param_ = {})
        : tree(tree_)
        , f{f_, param_.useCachedFunction, false}
        , param(param_)
        , pivotError(1, Real(0))
        , localSet(localDim.size())
        , tt{tree}
    {
        if (localDim.size() != tree.nodes.size())
            throw std::invalid_argument("tree and localDim are incompatible");
        if (param.pivot1.empty())
            param.pivot1.resize(localDim.size(), 0);
        const Scalar fpiv = f.f(param.pivot1);
        using std::abs;
        pivotError[0] = abs(fpiv);
        if (pivotError[0] == Real(0))
            throw std::invalid_argument("Not expecting f(pivot1)=0. Provide a better first pivot in the param");

        for (auto p = 0u; p < len(); ++p)
            for (int i = 0; i < localDim[p]; ++i) {
                MultiIndex lset(len(), 0);
                lset[p] = i;
                localSet[p].push_back(lset);
            }
        if (param.cond) {
            if (!param.cond(param.pivot1))
                throw std::invalid_argument("First pivot does not fulfill the condition function, expecting cond(f(pivot1))=true.");
            fCond = TensorFunction<unsigned int>{param.cond, false, false};
        }
        addPivotsAllBonds(enrich_pivot1());
    }

    /// Build the three-leg site tensor by evaluating f on combined multi-indices.
    /// Neighbor axes follow tree.neigh ordering; a physical leg is always axis 2.
    /// Physical nodes: T(i,j,sigma) = f(I_0[i] + I_1[j] + localSet[node][sigma]).
    /// Physical leaves have shape (rank,1,localDim), with a singleton second axis.
    /// Artificial nodes: T(i,j,k) = f(I_0[i] + I_1[j] + I_2[k]).
    /// Here I_a = Iset[{neighbor_a,node}], and + is component-wise addition.
    Cube get_T3(int node) const {
        std::vector<MultiIndex> Ip;
        std::vector<Eigen::Index> shape;
        for (int neighbor : tree.neigh.at(node).from_int()) {
            const auto& indices = Iset.at({neighbor, node});
            Ip = add(Ip, indices);
            shape.push_back(static_cast<Eigen::Index>(indices.size()));
        }
        if (shape.size() == 1) shape.push_back(1); // leaf: retain a dummy second leg

        std::vector<MultiIndex> Jp;
        if (tree.nodes.contains(node)) {
            Jp = add(Jp, localSet.at(node));
            shape.push_back(static_cast<Eigen::Index>(localSet.at(node).size()));
        } else {
            Jp.push_back(MultiIndex(tree.nodes.size(), 0));
        }
        if (shape.size() != 3)
            throw std::runtime_error("tensor degree not 3 at get_T3");

        const Mat data = f.eval(Ip, Jp);
        return Cube(data.data(), static_cast<std::size_t>(shape[0]),
                    static_cast<std::size_t>(shape[1]), static_cast<std::size_t>(shape[2]));
    }

    /// Build T * P^{-1} at from, contracting the bond toward to.
    /// P[{from,to}] stores the pivot matrix in packed LU form.
    Cube get_TP1(int from, int to) const {
        const auto M = get_T3(from);
        const int pos = tree.neigh.at(from).pos(to);
        const Mat t3mat = cubeToMat_R(M, pos);
        const auto& pivot = P.at({from, to});
        const Mat TP1 = compute_CU_on_rows(t3mat, pivot);
        std::vector<Eigen::Index> shape{M.n_rows, M.n_cols, M.n_slices};
        shape.at(pos) = pivot.cols();
        return matToCube_R(TP1, shape, pos);
    }

    /// Recompress the current cross on a bond without enriching its pivot sets.
    /// Update the two directed index sets, packed pivot matrix, adjacent tensors,
    /// and rank-wise pivot errors, optionally restricting pivot selection with fCond.
    void dmrg0_updatePivotAt(int from, int to) {
        auto ci = param.cond
            ? CURDecomp<Scalar>{f.eval(Iset[{from, to}], Iset[{to, from}]),
                                fCond.eval(Iset[{from, to}], Iset[{to, from}]),
                                true, param.reltol, param.bondDim}
            : CURDecomp<Scalar>{f.eval(Iset[{from, to}], Iset[{to, from}]),
                                true, param.reltol, param.bondDim};
        Iset[{from, to}] = Iset[{from, to}].at(ci.row_pivots());
        Iset[{to, from}] = Iset[{to, from}].at(ci.col_pivots());
        P[{from, to}] = ci.PivotMatrixTri();
        tt.M.at(from) = get_TP1(from, to);
        tt.M.at(to) = get_T3(to);
        collectPivotError(from, to, ci.pivotErrors());
    }

    /// Enrich the bond candidates with neighboring index sets and pivot history,
    /// then update the cross using adaptive CUR on the Pi matrix.
    void dmrg2_updatePivotAt(int from, int to) {
        IndexSet<MultiIndex> I = set_union(I0[{from, to}], kronecker(from, to));
        IndexSet<MultiIndex> J = set_union(I0[{to, from}], kronecker(to, from));
        auto ci = param.cond
            ? CURDecomp<Scalar>{f.matfun(I, J), fCond.matfun(I, J),
                                I.pos(Iset[{from, to}]), J.pos(Iset[{to, from}]),
                                true, param}
            : CURDecomp<Scalar>{f.matfun(I, J),
                                I.pos(Iset[{from, to}]), J.pos(Iset[{to, from}]),
                                true, param};
        Iset[{from, to}] = I.at(ci.row_pivots());
        Iset[{to, from}] = J.at(ci.col_pivots());
        P[{from, to}] = ci.PivotMatrixTri();
        tt.M.at(from) = get_TP1(from, to);
        tt.M.at(to) = get_T3(to);
        collectPivotError(from, to, ci.pivotErrors());
    }

    /// Number of physical legs, not including artificial nodes.
    std::size_t len() const { return localSet.size(); }

    /// Perform nIter full DFS sweeps, updating each directed bond with DMRG-0 or DMRG-2.
    void iterate(int nIter = 1, int dmrg_type = 2) {
        for (int i = 0; i < nIter; ++i) {
            for (const auto& [from, to] : tree.rootToLeaves())
                updatePivotAt(from, to, dmrg_type);
            ++cIter;
        }
    }

    /// Add global pivots at every bond, then rebuild tt with one DMRG-0 sweep.
    void addPivotsAllBonds(std::vector<std::vector<int>> const& pivots) {
        for (const auto& [from, to] : tree.leavesToRoot())
            addPivotsAt(pivots, from, to);
        iterate(1, 0);
    }

    /// Project global pivots onto the two physical subtrees separated by this bond.
    /// Each projected multi-index retains full length, with zeros outside its subtree.
    /// IndexSet preserves insertion order and ignores duplicates; tt is not touched.
    void addPivotsAt(std::vector<std::vector<int>> const& pivots, int from, int to) {
        const auto [nodes0, nodes1] = tree.split(from, to);
        MultiIndex pvec0(tree.nodes.size(), 0);
        MultiIndex pvec1(tree.nodes.size(), 0);
        for (const auto& pivot : pivots) {
            for (int node : nodes0) pvec0[node] = static_cast<char32_t>(pivot[node]);
            for (int node : nodes1) pvec1[node] = static_cast<char32_t>(pivot[node]);
            Iset[{from, to}].push_back(pvec0);
            Iset[{to, from}].push_back(pvec1);
        }
    }

protected:
    /// Enrich the first pivot by varying pairs of physical nodes neighboring artificial nodes.
    /// Keep all other coordinates fixed at param.pivot1 to avoid rank-one initialization problems.
    std::vector<std::vector<int>> enrich_pivot1() const {
        std::vector<int> nvnodes; // physical neighbors of artificial nodes
        for (auto i = 0u; i < tree.neigh.size(); ++i) {
            if (tree.nodes.contains(i)) continue;
            for (int j : tree.neigh.at(i).from_int())
                if (tree.nodes.contains(j)) nvnodes.push_back(j);
        }
        std::vector<std::vector<int>> pivots{param.pivot1};
        for (int i : nvnodes)
            for (int j : nvnodes)
                if (i != j)
                    for (const MultiIndex& xi : localSet[i].from_int())
                        for (const MultiIndex& xj : localSet[j].from_int()) {
                            MultiIndex mi{param.pivot1.begin(), param.pivot1.end()};
                            mi[i] = xi[i];
                            mi[j] = xj[j];
                            pivots.push_back({mi.begin(), mi.end()});
                        }
        return pivots;
    }

    /// Save the current pivots in the bond history, then perform a DMRG-0 or DMRG-2 update.
    void updatePivotAt(int from, int to, int dmrg = 2) {
        for (const auto& id : Iset[{from, to}].from_int()) I0[{from, to}].push_back(id);
        for (const auto& id : Iset[{to, from}].from_int()) I0[{to, from}].push_back(id);

        switch (dmrg) {
        case 0: dmrg0_updatePivotAt(from, to); break;
        //case 1: dmrg1_updatePivotAt(from, to); break;
        case 2: dmrg2_updatePivotAt(from, to); break;
        }
    }

    /// Build the candidate multi-indices on the from side of the bond (from,to).
    /// Combine incoming sets from all neighbors except to, in neighbor order,
    /// then include localSet[from] if from is physical. Combinations use
    /// component-wise addition of full-length multi-indices, not concatenation.
    IndexSet<MultiIndex> kronecker(int from, int to) const {
        IndexSet<MultiIndex> Is;
        for (int neighbor : tree.neigh.at(from).from_int())
            if (neighbor != to) Is = add(Is, Iset.at({neighbor, from}));
        if (tree.nodes.contains(from))
            return add(Is, localSet.at(from));
        return Is;
    }

    /// Record this bond's errors and retain the largest error seen at each rank.
    void collectPivotError(int from, int to, std::vector<Real> const& pe) {
        pivotErrorAll[{from, to}] = pe;
        if (pe.size() > pivotError.size()) pivotError.resize(pe.size(), Real(0));
        for (std::size_t i = 0; i < pe.size(); ++i)
            if (pe[i] > pivotError[i]) pivotError[i] = pe[i];
    }

private:
    std::map<std::pair<int, int>, std::vector<Real>> pivotErrorAll; ///< pivot errors for each directed bond
    std::map<std::pair<int, int>, IndexSet<MultiIndex>> I0; ///< historical accepted pivots for each directed bond
};

} // namespace xfac_quad
