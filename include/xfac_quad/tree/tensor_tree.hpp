#pragma once

#include "xfac_quad/tree/tree.hpp"
#include "xfac_quad/cubemat_helper.h"

namespace xfac_quad {

template<typename Scalar>
struct TensorTree {
    TopologyTree tree;
    std::vector<Tensor3D<Scalar>> M;

    TensorTree()=default;
    TensorTree(TopologyTree const& tree_) : tree(tree_), M(tree_.size()) {
        if (!isValid(tree)) throw std::invalid_argument("TensorTree: tree is not valid");
    }
};
}