#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include "xfac_quad/xfac_quad.hpp"
#include "xfac_quad/grid.h"

using namespace xfac_quad;

TEST_CASE("Test benchmark problems - dd128", "[.][slow]")
{
    using T = dd_128;

    SECTION("exp-x integral")
    {
        int nBit = 30;
        int dim = 3;
        MultQTGrid<T, int64_t> mgrid({-40, -40, -40}, {40, 40, 40}, {nBit, nBit, nBit});
        auto tensorDims = std::vector<int>(dim * nBit, 2);

        auto func = [](T x, T y, T z) { return exp(-sqrt(x*x + y*y + z*z)); };
        auto tfunc = [&](std::vector<int> xi) {
            auto r = mgrid.id_to_coord(xi);
            return func(r[0], r[1], r[2]);
        };

        std::vector<std::vector<int>> pivots;
        for (auto x : {-1.0, 1.0})
            for (auto y : {-1.0, 1.0})
                for (auto z : {-1.0, 1.0})
                    pivots.push_back(mgrid.coord_to_id({T(x), T(y), T(z)}));

        T exact = T(8) * T(dd_real::_pi);
        T reltol = T(100) * std::numeric_limits<T>::epsilon();

        auto ci = TensorCI2<T>(TensorFunction<T>(tfunc), tensorDims, {.bondDim=150, .reltol=reltol, .pivot1={}, .fullPiv=true, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        ci.addPivotsAllBonds(pivots);

        std::cout << "# exp-x integral on ci2: iteration, integral, number of evaluations, last pivot error, abs(integral / exact - 1), rank\n";
        for (auto i = 0u; i < 10; i++) {
            ci.iterate();
            auto integ = ci.tt.sum1() * mgrid.delta_volume();
            std::cout << std::setprecision(14) << i << " " << integ << " " << ci.f.nEval() << " " << ci.pivotError.back() << " " << abs(integ / exact - 1) << " " << ci.pivotError.size() << std::endl;
        }
        std::cout << "# exp-x integral on ci2: rank, pivot error\n";
        for (auto i = 0u; i < ci.pivotError.size(); i++)
            std::cout << std::setprecision(14) << i << " " << ci.pivotError[i] << std::endl;
    }
}
