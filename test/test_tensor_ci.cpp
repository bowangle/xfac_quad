#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <complex>
#include <cstdlib>
#include <limits>
#include "xfac_quad/xfac_quad.hpp"

using namespace xfac_quad;
using std::abs;
using cmpx = std::complex<double>;

// --- debug helper: print actual vs tolerance with full precision ---
template<typename A, typename B>
bool check_le(std::string const& expr, A const& val, B const& tol) {
    if (!(val <= tol)) {
        std::cerr << std::scientific << std::setprecision(std::numeric_limits<decltype(val<=tol?val:tol)>::max_digits10)
                  << "  FAIL " << expr << ": val=" << val << " > tol=" << tol << "\n";
        return false;
    }
    return true;
}
template<typename A, typename B>
bool check_lt(std::string const& expr, A const& val, B const& tol) {
    if (!(val < tol)) {
        std::cerr << std::scientific << std::setprecision(std::numeric_limits<decltype(val<tol?val:tol)>::max_digits10)
                  << "  FAIL " << expr << ": val=" << val << " >= tol=" << tol << "\n";
        return false;
    }
    return true;
}

// ============================================================
// CI1 Tests - all types
// ============================================================
TEMPLATE_TEST_CASE("TensorCI1 basic", "[ci1]", double, cmpx, dd_128, Cdd_128, float128, Cfloat128)
{
    using T = TestType;
    using Real = typename Eigen::NumTraits<T>::Real;
    Real const reltol = Real(100) * std::numeric_limits<Real>::epsilon();
    Real const tol    = Real(1000) * std::numeric_limits<Real>::epsilon();

    SECTION("rank-1 separable function")
    {
        auto myf = [](std::vector<int> const& id) -> T {
            return (T(id[0]) + T(1)) * (T(id[1]) + T(2));
        };

        TensorCI1Param<T> p;
        p.pivot1 = {0, 0};
        p.reltol = reltol;
        auto ci = TensorCI1<T>(myf, {20, 20}, p);
        REQUIRE(ci.pivotError.size() >= 1);
        REQUIRE(check_lt("trueError", ci.trueError(), tol));
    }

    SECTION("sum + cos function")
    {
        int dim = 3, d = 8;
        long count = 0;
        auto myf = [&](std::vector<int> id) -> T {
            count++;
            double sum = std::accumulate(id.begin(), id.end(), 0.0);
            return T(sum + std::cos(sum));
        };

        TensorCI1Param<T> p;
        p.fullPiv = true;
        p.nIter = 10 * d;
        p.reltol = reltol;
        auto ci = TensorCI1<T>(myf, std::vector<int>(dim, d), p);
        REQUIRE(count < static_cast<long>(std::pow(d * ci.P[dim/2 - 1].rows(), 2) * (dim - 1)));
    }

    SECTION("polynomial function")
    {
        int n = 16;
        auto xi = std::vector<T>(n);
        for (int i = 0; i < n; i++) xi[i] = T(i) / T(n - 1);

        std::vector<T> data(n*n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                data[i*n + j] = T(1) + xi[i]*xi[i] + xi[j]*xi[j] + xi[i]*xi[j];

        auto myTf = [&](std::vector<int> const& id) { return data[id[0]*n + id[1]]; };

        TensorCI1Param<T> p;
        p.reltol = reltol;
        p.nIter = 20;
        auto ci = TensorCI1<T>(myTf, {n, n}, p);
        REQUIRE(check_le("pivotError", ci.pivotError.back(), tol));
        REQUIRE(check_le("trueError", ci.trueError(256), tol));
    }
}

// ============================================================
// CI2 Tests - all types
// ============================================================
TEMPLATE_TEST_CASE("TensorCI2 basic", "[ci2]", double, cmpx, dd_128, Cdd_128, float128, Cfloat128)
{
    using T = TestType;
    using Real = typename Eigen::NumTraits<T>::Real;
    Real const reltol = Real(100) * std::numeric_limits<Real>::epsilon();
    Real const tol    = Real(1000) * std::numeric_limits<Real>::epsilon();

    SECTION("rank-1 separable function")
    {
        auto myf = [](std::vector<int> const& id) -> T {
            return (T(id[0]) + T(1)) * (T(id[1]) + T(2));
        };

        auto ci = TensorCI2<T>(myf, {20, 20},
            {.bondDim=5, .reltol=reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        ci.iterate(3);
        REQUIRE(check_lt("trueError", ci.trueError(), tol));
    }

    SECTION("sum + cos function")
    {
        int dim = 3, d = 8;
        long count = 0;
        auto myf = [&](std::vector<int> id) -> T {
            count++;
            double sum = std::accumulate(id.begin(), id.end(), 0.0);
            return T(sum + std::cos(sum));
        };

        auto ci = TensorCI2<T>(myf, std::vector<int>(dim, d),
            {.bondDim=10*d, .reltol=reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        ci.iterate(3);
        REQUIRE(ci.pivotError.size() >= 2);
        REQUIRE(count < static_cast<long>(std::pow(d * ci.pivotError.size(), 2) * (dim - 2) * ci.cIter));
    }

    SECTION("polynomial function - check isDone")
    {
        int n = 16;
        auto myf = [n](std::vector<int> const& id) -> T {
            T x = T(id[0]) / T(n-1), y = T(id[1]) / T(n-1);
            return T(1) + x*x + y*y + x*y;
        };

        auto ci = TensorCI2<T>(myf, {n, n},
            {.bondDim=10, .reltol=reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        int maxIter = 20;
        while (!ci.isDone() && ci.cIter < maxIter) ci.iterate();
        REQUIRE(check_le("pivotError", ci.pivotError.back(), tol));
        REQUIRE(check_le("trueError", ci.trueError(256), tol));
    }

    SECTION("tt.eval matches function after isDone")
    {
        int n = 15;
        auto myf = [n](std::vector<int> const& id) -> T {
            T x = T(id[0]) / T(n-1), y = T(id[1]) / T(n-1);
            return T(1) + x*x + y*y + x*y;
        };

        auto ci = TensorCI2<T>(myf, {n, n},
            {.bondDim=8, .reltol=reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        int maxIter = 20;
        while (!ci.isDone() && ci.cIter < maxIter) ci.iterate();

        std::vector<int> ids = {3, 5};
        REQUIRE(check_le("tt.eval", abs(ci.tt.eval(ids) - myf(ids)), tol));
    }
}

// ============================================================
// CTensorCI1 Tests
// ============================================================
TEMPLATE_TEST_CASE("CTensorCI1 continuous", "[ci1][continuous]", double, cmpx, dd_128, Cdd_128, float128, Cfloat128)
{
    using T = TestType;
    using Real = typename Eigen::NumTraits<T>::Real;
    Real const reltol = Real(100) * std::numeric_limits<Real>::epsilon();
    Real const tol    = Real(1000) * std::numeric_limits<Real>::epsilon();

    SECTION("polynomial function directly")
    {
        auto f = [](const std::vector<T>& xs) -> T {
            return T(1) + xs[0]*xs[0] + xs[1]*xs[1] + xs[0]*xs[1];
        };

        auto xi = std::vector<T>(15);
        for (int i = 0; i < 15; i++) xi[i] = T(i) / T(14);

        auto ci = CTensorCI1<T, T>(f, std::vector(2, xi),
            {.nIter=15, .reltol=reltol, .pivot1={}, .fullPiv=false, .nRookIter=5, .weight={}, .cond={}, .useCachedFunction=true});

        REQUIRE(check_lt("pivotError", ci.pivotError.back(), tol));
        T x = T(3) / T(10), y = T(7) / T(10);
        REQUIRE(check_lt("ctt.eval", abs(ci.get_CTensorTrain().eval({x, y}) - f({x, y})), tol));
    }
}

// ============================================================
// CTensorCI2 Tests
// ============================================================
TEMPLATE_TEST_CASE("CTensorCI2 continuous", "[ci2][continuous]", double, cmpx, dd_128, Cdd_128, float128, Cfloat128)
{
    using T = TestType;
    using Real = typename Eigen::NumTraits<T>::Real;
    Real const reltol = Real(100) * std::numeric_limits<Real>::epsilon();
    Real const tol    = Real(1000) * std::numeric_limits<Real>::epsilon();

    SECTION("polynomial function directly")
    {
        auto f = [](const std::vector<T>& xs) -> T {
            return T(1) + xs[0]*xs[0] + xs[1]*xs[1] + xs[0]*xs[1];
        };

        auto xi = std::vector<T>(15);
        for (int i = 0; i < 15; i++) xi[i] = T(i) / T(14);

        auto ci = CTensorCI2<T, T>(f, std::vector(2, xi),
            {.bondDim=10, .reltol=reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});

        int maxIter = 15;
        while (!ci.isDone() && ci.cIter < maxIter) ci.iterate();

        T x = T(3) / T(10), y = T(7) / T(10);
        REQUIRE(check_lt("ctt.eval", abs(ci.get_CTensorTrain().eval({x, y}) - f({x, y})), tol));
    }
}
