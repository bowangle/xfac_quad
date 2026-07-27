#pragma once
// Boost math constants needed by type_float128_boost.h's pi<float128>()
#include <boost/math/constants/constants.hpp>
// Import quadruple-precision types from numeric-type-quad
#include "type_double_double.h"
#include "type_float128_boost.h"

// -------------------------------------------------------------------
// Fix boost bug: numeric_limits<float128>::min/max/lowest return
// raw __float128 instead of float128. Patch them.
// -------------------------------------------------------------------
#include <boost/multiprecision/float128.hpp>
namespace std {
template<>
inline BOOST_MP_CXX14_CONSTEXPR float128 numeric_limits<float128>::min() noexcept {
    return float128(BOOST_MP_QUAD_MIN);
}
template<>
inline BOOST_MP_CXX14_CONSTEXPR float128 numeric_limits<float128>::max() noexcept {
    return float128(BOOST_MP_QUAD_MAX);
}
template<>
inline BOOST_MP_CXX14_CONSTEXPR float128 numeric_limits<float128>::lowest() noexcept {
    return float128(-(max)());
}
}

// Include all xfac_quad headers
#include "xfac_quad/index_set.h"
#include "xfac_quad/grid.h"
#include "xfac_quad/cubemat_helper.h"
#include "xfac_quad/matrix/mat_decomp.h"
#include "xfac_quad/matrix/adaptive_lu.h"
#include "xfac_quad/matrix/cross_data.h"
#include "xfac_quad/matrix/matrix_interface.h"
#include "xfac_quad/matrix/pivot_finder.h"
#include "xfac_quad/tensor/tensor_train.h"
#include "xfac_quad/tensor/tensor_function.h"
#include "xfac_quad/tensor/tensor_ci.h"
#include "xfac_quad/tensor/tensor_ci_2.h"
#include "xfac_quad/tensor/tensor_ci_converter.h"
