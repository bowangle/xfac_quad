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
using T      = double;
using CT     = std::complex<double>;
using Real   = double;

constexpr Real ci_reltol = Real(10000) * std::numeric_limits<Real>::epsilon();
constexpr Real ci_tol    = Real(1000) * std::numeric_limits<Real>::epsilon();
constexpr Real ci_tol2   = Real(10000) * std::numeric_limits<Real>::epsilon();

// --- debug helper ---
template<typename A, typename B>
bool check_le(std::string const& expr, A const& val, B const& tol) {
    if (!(val <= tol)) {
        std::cerr << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10)
                  << "  FAIL " << expr << ": val=" << val << " > tol=" << tol << "\n";
        return false;
    }
    return true;
}
template<typename A, typename B>
bool check_lt(std::string const& expr, A const& val, B const& tol) {
    if (!(val < tol)) {
        std::cerr << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10)
                  << "  FAIL " << expr << ": val=" << val << " >= tol=" << tol << "\n";
        return false;
    }
    return true;
}

// ============================================================
// CI1 Tests
// ============================================================
TEST_CASE("TensorCI1 basic - double", "[ci1]")
{
    SECTION("rank-1 separable function")
    {
        auto myf = [](std::vector<int> const& id) -> T {
            return (T(id[0]) + T(1)) * (T(id[1]) + T(2));
        };

        TensorCI1Param<T> p;
        p.pivot1 = {0, 0};
        p.reltol = ci_reltol;
        auto ci = TensorCI1<T>(myf, {20, 20}, p);
        REQUIRE(ci.pivotError.size() >= 1);
        REQUIRE(check_lt("trueError", ci.trueError(), ci_tol2));
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
        p.reltol = ci_reltol;
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
        p.reltol = ci_reltol;
        p.nIter = 20;
        auto ci = TensorCI1<T>(myTf, {n, n}, p);
        REQUIRE(check_le("pivotError", ci.pivotError.back(), ci_tol2));
        REQUIRE(check_le("trueError", ci.trueError(256), ci_tol2));
    }
}

TEST_CASE("TensorCI1 basic - complex double", "[ci1]")
{
    SECTION("rank-1 separable function")
    {
        auto myf = [](std::vector<int> const& id) -> CT {
            return (CT(id[0]) + CT(1)) * (CT(id[1]) + CT(2));
        };

        TensorCI1Param<CT> p;
        p.pivot1 = {0, 0};
        p.reltol = ci_reltol;
        auto ci = TensorCI1<CT>(myf, {20, 20}, p);
        REQUIRE(ci.pivotError.size() >= 1);
        REQUIRE(check_lt("trueError", ci.trueError(), ci_tol2));
    }

    SECTION("sum + cos function")
    {
        int dim = 3, d = 8;
        long count = 0;
        auto myf = [&](std::vector<int> id) -> CT {
            count++;
            double sum = std::accumulate(id.begin(), id.end(), 0.0);
            return CT(sum + std::cos(sum));
        };

        TensorCI1Param<CT> p;
        p.fullPiv = true;
        p.nIter = 10 * d;
        p.reltol = ci_reltol;
        auto ci = TensorCI1<CT>(myf, std::vector<int>(dim, d), p);
        REQUIRE(count < static_cast<long>(std::pow(d * ci.P[dim/2 - 1].rows(), 2) * (dim - 1)));
    }

    SECTION("polynomial function")
    {
        int n = 16;
        auto xi = std::vector<CT>(n);
        for (int i = 0; i < n; i++) xi[i] = CT(i) / CT(n - 1);

        std::vector<CT> data(n*n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                data[i*n + j] = CT(1) + xi[i]*xi[i] + xi[j]*xi[j] + xi[i]*xi[j];

        auto myTf = [&](std::vector<int> const& id) { return data[id[0]*n + id[1]]; };

        TensorCI1Param<CT> p;
        p.reltol = ci_reltol;
        p.nIter = 20;
        auto ci = TensorCI1<CT>(myTf, {n, n}, p);
        REQUIRE(check_le("pivotError", ci.pivotError.back(), ci_tol2));
        REQUIRE(check_le("trueError", ci.trueError(256), ci_tol2));
    }
}

// ============================================================
// CI2 Tests
// ============================================================
TEST_CASE("TensorCI2 basic - double", "[ci2]")
{
    SECTION("rank-1 separable function")
    {
        auto myf = [](std::vector<int> const& id) -> T {
            return (T(id[0]) + T(1)) * (T(id[1]) + T(2));
        };

        auto ci = TensorCI2<T>(myf, {20, 20},
            {.bondDim=5, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        ci.iterate(3);
        REQUIRE(check_lt("trueError", ci.trueError(), ci_tol2));
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
            {.bondDim=10*d, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
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
            {.bondDim=10, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        int maxIter = 20;
        while (!ci.isDone() && ci.cIter < maxIter) ci.iterate();
        REQUIRE(check_le("pivotError", ci.pivotError.back(), ci_tol2));
        REQUIRE(check_le("trueError", ci.trueError(256), ci_tol2));
    }

    SECTION("tt.eval matches function after isDone")
    {
        int n = 15;
        auto myf = [n](std::vector<int> const& id) -> T {
            T x = T(id[0]) / T(n-1), y = T(id[1]) / T(n-1);
            return T(1) + x*x + y*y + x*y;
        };

        auto ci = TensorCI2<T>(myf, {n, n},
            {.bondDim=8, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        int maxIter = 20;
        while (!ci.isDone() && ci.cIter < maxIter) ci.iterate();

        std::vector<int> ids = {3, 5};
        REQUIRE(check_le("tt.eval", abs(ci.tt.eval(ids) - myf(ids)), ci_tol2));
    }
}

TEST_CASE("TensorCI2 basic - complex double", "[ci2]")
{
    SECTION("rank-1 separable function")
    {
        auto myf = [](std::vector<int> const& id) -> CT {
            return (CT(id[0]) + CT(1)) * (CT(id[1]) + CT(2));
        };

        auto ci = TensorCI2<CT>(myf, {20, 20},
            {.bondDim=5, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        ci.iterate(3);
        REQUIRE(check_lt("trueError", ci.trueError(), ci_tol2));
    }

    SECTION("sum + cos function")
    {
        int dim = 3, d = 8;
        long count = 0;
        auto myf = [&](std::vector<int> id) -> CT {
            count++;
            double sum = std::accumulate(id.begin(), id.end(), 0.0);
            return CT(sum + std::cos(sum));
        };

        auto ci = TensorCI2<CT>(myf, std::vector<int>(dim, d),
            {.bondDim=10*d, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        ci.iterate(3);
        REQUIRE(ci.pivotError.size() >= 2);
        REQUIRE(count < static_cast<long>(std::pow(d * ci.pivotError.size(), 2) * (dim - 2) * ci.cIter));
    }

    SECTION("polynomial function - check isDone")
    {
        int n = 16;
        auto myf = [n](std::vector<int> const& id) -> CT {
            CT x = CT(id[0]) / CT(n-1), y = CT(id[1]) / CT(n-1);
            return CT(1) + x*x + y*y + x*y;
        };

        auto ci = TensorCI2<CT>(myf, {n, n},
            {.bondDim=10, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        int maxIter = 20;
        while (!ci.isDone() && ci.cIter < maxIter) ci.iterate();
        REQUIRE(check_le("pivotError", ci.pivotError.back(), ci_tol2));
        REQUIRE(check_le("trueError", ci.trueError(256), ci_tol2));
    }

    SECTION("tt.eval matches function after isDone")
    {
        int n = 15;
        auto myf = [n](std::vector<int> const& id) -> CT {
            CT x = CT(id[0]) / CT(n-1), y = CT(id[1]) / CT(n-1);
            return CT(1) + x*x + y*y + x*y;
        };

        auto ci = TensorCI2<CT>(myf, {n, n},
            {.bondDim=8, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        int maxIter = 20;
        while (!ci.isDone() && ci.cIter < maxIter) ci.iterate();

        std::vector<int> ids = {3, 5};
        REQUIRE(check_le("tt.eval", abs(ci.tt.eval(ids) - myf(ids)), ci_tol2));
    }
}

// ============================================================
// CTensorCI1 Tests
// ============================================================
TEST_CASE("CTensorCI1 continuous - double", "[ci1][continuous]")
{
    SECTION("polynomial function directly")
    {
        auto f = [](const std::vector<T>& xs) -> T {
            return T(1) + xs[0]*xs[0] + xs[1]*xs[1] + xs[0]*xs[1];
        };

        auto xi = std::vector<T>(15);
        for (int i = 0; i < 15; i++) xi[i] = T(i) / T(14);

        auto ci = CTensorCI1<T, T>(f, std::vector(2, xi),
            {.nIter=15, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=5, .weight={}, .cond={}, .useCachedFunction=true});

        REQUIRE(check_lt("pivotError", ci.pivotError.back(), ci_tol2));
        T x = T(3) / T(10), y = T(7) / T(10);
        REQUIRE(check_lt("ctt.eval", abs(ci.get_CTensorTrain().eval({x, y}) - f({x, y})), ci_tol2));
    }
}

TEST_CASE("CTensorCI1 continuous - complex double", "[ci1][continuous]")
{
    SECTION("polynomial function directly")
    {
        auto f = [](const std::vector<CT>& xs) -> CT {
            return CT(1) + xs[0]*xs[0] + xs[1]*xs[1] + xs[0]*xs[1];
        };

        auto xi = std::vector<CT>(15);
        for (int i = 0; i < 15; i++) xi[i] = CT(i) / CT(14);

        auto ci = CTensorCI1<CT, CT>(f, std::vector(2, xi),
            {.nIter=15, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=5, .weight={}, .cond={}, .useCachedFunction=true});

        REQUIRE(check_lt("pivotError", ci.pivotError.back(), ci_tol2));
        CT x = CT(3) / CT(10), y = CT(7) / CT(10);
        REQUIRE(check_lt("ctt.eval", abs(ci.get_CTensorTrain().eval({x, y}) - f({x, y})), ci_tol2));
    }
}

// ============================================================
// CTensorCI2 Tests
// ============================================================
TEST_CASE("CTensorCI2 continuous - double", "[ci2][continuous]")
{
    SECTION("polynomial function directly")
    {
        auto f = [](const std::vector<T>& xs) -> T {
            return T(1) + xs[0]*xs[0] + xs[1]*xs[1] + xs[0]*xs[1];
        };

        auto xi = std::vector<T>(15);
        for (int i = 0; i < 15; i++) xi[i] = T(i) / T(14);

        auto ci = CTensorCI2<T, T>(f, std::vector(2, xi),
            {.bondDim=10, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});

        int maxIter = 15;
        while (!ci.isDone() && ci.cIter < maxIter) ci.iterate();

        T x = T(3) / T(10), y = T(7) / T(10);
        REQUIRE(check_lt("ctt.eval", abs(ci.get_CTensorTrain().eval({x, y}) - f({x, y})), ci_tol2));
    }
}

TEST_CASE("CTensorCI2 continuous - complex double", "[ci2][continuous]")
{
    SECTION("polynomial function directly")
    {
        auto f = [](const std::vector<CT>& xs) -> CT {
            return CT(1) + xs[0]*xs[0] + xs[1]*xs[1] + xs[0]*xs[1];
        };

        auto xi = std::vector<CT>(15);
        for (int i = 0; i < 15; i++) xi[i] = CT(i) / CT(14);

        auto ci = CTensorCI2<CT, CT>(f, std::vector(2, xi),
            {.bondDim=10, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});

        int maxIter = 15;
        while (!ci.isDone() && ci.cIter < maxIter) ci.iterate();

        CT x = CT(3) / CT(10), y = CT(7) / CT(10);
        REQUIRE(check_lt("ctt.eval", abs(ci.get_CTensorTrain().eval({x, y}) - f({x, y})), ci_tol2));
    }
}
