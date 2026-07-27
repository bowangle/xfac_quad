#pragma once
// QTgrid-quad provides the core QTGrid<Scalar,Sint> class
#include <grid.h>

#include <cmath>
#include <vector>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <cassert>
#include "xfac_quad/index_set.h"

namespace xfac_quad {
namespace grid {

template <typename T>                                                                      
std::vector<T> linspace(T a, T b, size_t n) {                                              
    if (n == 0) return {};                                                                 
    if (n == 1) return {a};                                                                
    T h = (b - a) / static_cast<T>(n - 1);                                                 
    std::vector<T> xs(n);                                                                  
    for (size_t i = 0; i < n; i++)                                                         
        xs[i] = a + i * h;                                                                 
    return xs;                                                                             
}                                                                                          
                                                                                            
template <typename T>                                                                      
std::vector<T> logspace(T a, T b, size_t n) {                                              
    auto xs = linspace<T>(log(a), log(b), n);                                              
    for (auto& x : xs) x = exp(x);                                                         
    return xs;                                                                             
}            


// pi<Scalar>() — specialisations for build_dual_grid()
template<typename Real>
Real pi() {
    static_assert(sizeof(Real) == 0,
        "pi() not specialised for this type — add a specialisation or include the right type header");
    return Real(0);
}

template<> inline double     pi<double>()     { return M_PI; }
template<> inline dd_128     pi<dd_128>()     { return dd_real::_pi; }

// ============================================================
// QTGrid — thin wrapper around QTgrid-quad's ::QTGrid,
//          adding MultiIndex-based coord_to_id / id_to_coord
//          and keeping the same API as the original xfac_quad.
// ============================================================
template <typename Scalar, typename Sint>
class QTGrid : public ::QTGrid<Scalar, Sint> {
public:
    using ::QTGrid<Scalar, Sint>::QTGrid;

    // --- MultiIndex versions (convert to/from std::vector<int>) ---
    MultiIndex coord_to_id(Scalar x) const {
        auto bits = ::QTGrid<Scalar, Sint>::coord_to_id(x);
        MultiIndex mi(bits.size(), char32_t(0));
        for (size_t i = 0; i < bits.size(); ++i)
            mi[i] = char32_t(bits[i]);
        return mi;
    }

    Scalar id_to_coord(const MultiIndex& mi) const {
        std::vector<int> bits(mi.begin(), mi.end());
        return ::QTGrid<Scalar, Sint>::id_to_coord(bits);
    }

    // Needed because the inherited overload with vector<int> is hidden
    using ::QTGrid<Scalar, Sint>::coord_to_id;
    using ::QTGrid<Scalar, Sint>::id_to_coord;
};

} // namespace grid
} // namespace xfac_quad
