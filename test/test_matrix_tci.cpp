#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <complex>
#include <cstdlib>
#include <limits>
#include "xfac_quad/xfac_quad.hpp"
#include "xfac_quad/grid.h"
#include "xfac_quad/matrix/mat_decomp.h"

using namespace xfac_quad;
using cmpx = std::complex<double>;
using Mat = Eigen::MatrixXd;

// --- debug helper: print actual vs tolerance with full precision ---
template<typename A, typename B>
bool check_le(std::string const& expr, A const& val, B const& tol) {
    if (!(val <= tol)) {
        std::cerr << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << "  FAIL " << expr << ": val=" << val << " > tol=" << tol << "\n";
        return false;
    }
    return true;
}
template<typename A, typename B>
bool check_lt(std::string const& expr, A const& val, B const& tol) {
    if (!(val < tol)) {
        std::cerr << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << "  FAIL " << expr << ": val=" << val << " >= tol=" << tol << "\n";
        return false;
    }
    return true;
}

TEST_CASE("Test MatrixCI - double", "[matrix_ci]")
{
    constexpr double reltol = 100.0 * std::numeric_limits<double>::epsilon();
    constexpr double tol    = 1000.0 * std::numeric_limits<double>::epsilon();

    SECTION("fixed rank")
    {
        int n = 100, rank = 50;
        Mat A = Mat::Random(n, rank) * Mat::Random(rank, n);
        auto myf = [&](vector<int> const& id) -> double {
            return A(id[0], id[1]);
        };
        vector<int> dims = {(int)A.rows(), (int)A.cols()};
        TensorCI1Param<double> p;
        p.nIter = rank + 1;
        p.reltol = reltol;
        auto ci = TensorCI1<double>(myf, dims, p);
        REQUIRE(check_le("pivotError*norm", ci.pivotError.back(), tol * A.norm()));
        REQUIRE(check_le("trueError*norm", ci.trueError(), tol * A.norm()));
        int i = rand() % A.rows(), j = rand() % A.cols();
        REQUIRE(check_lt("tt.eval", std::abs(ci.get_TensorTrain().eval({i, j}) - A(i, j)), tol * A.norm()));
    }

    SECTION("rank-1 function")
    {
        auto myf = [](vector<int> const& id) -> double {
            return (id[0] + 1.0) * (id[1] + 2.0);
        };
        TensorCI1Param<double> p;
        p.pivot1 = {0, 0};
        p.reltol = reltol;
        auto ci = TensorCI1<double>(myf, {20, 20}, p);
        REQUIRE(check_le("trueError", ci.trueError(), tol));
    }
}

TEST_CASE("Test MatrixCI - complex", "[matrix_ci]")
{
    using Real = double;
    constexpr Real reltol = Real(100) * std::numeric_limits<Real>::epsilon();
    constexpr Real tol    = Real(1000) * std::numeric_limits<Real>::epsilon();

    SECTION("rank-2 separable function")
    {
        auto myf = [](vector<int> const& id) -> cmpx {
            return cmpx(1, 2) * cmpx(id[0] + 1.0, 0) * cmpx(id[1] + 2.0, 0)
                 + cmpx(3, -1) * cmpx(id[0] * 4.0, 0) * cmpx(id[1] + 5.0, 0);
        };
        TensorCI1Param<cmpx> p;
        p.nIter = 10;
        p.fullPiv = true;
        p.pivot1 = {0, 0};
        p.reltol = reltol;
        auto ci = TensorCI1<cmpx>(myf, {20, 20}, p);
        REQUIRE(check_le("trueError", ci.trueError(), tol));
    }

    SECTION("rank-1 complex function")
    {
        auto myf = [](vector<int> const& id) -> cmpx {
            return cmpx(1, 2) * cmpx(id[0] + 1.0, 0) * cmpx(id[1] + 2.0, 0);
        };
        TensorCI1Param<cmpx> p;
        p.pivot1 = {0, 0};
        p.reltol = reltol;
        auto ci = TensorCI1<cmpx>(myf, {20, 20}, p);
        REQUIRE(check_le("trueError", ci.trueError(), tol));
    }
}

TEST_CASE("Test RRLUDecomp - double", "[rrlu]")
{
    constexpr double tol = 1000.0 * std::numeric_limits<double>::epsilon();

    SECTION("2x3 matrix") {
        Mat A(2, 3);
        A << 1, 2, 3, 4, 5, 6;

        SECTION("horizontal matrix") {
            for (auto isLeft : {0, 1}) {
                RRLUDecomp<double> sol(A, isLeft, tol);
                REQUIRE(check_lt("RRLU left*right", (A - sol.left() * sol.right()).norm(), tol));
            }
        }

        SECTION("vertical matrix") {
            Mat At = A.transpose();
            for (auto isLeft : {0, 1}) {
                RRLUDecomp<double> sol(At, isLeft, tol);
                REQUIRE(check_lt("RRLU left*right", (At - sol.left() * sol.right()).norm(), tol));
            }
        }
    }

    SECTION("from MPO") {
        Mat A = Mat::Zero(4, 7);
        A.row(0).setOnes();
        A.row(3).setOnes();
        A(0, 0) = -1;
        RRLUDecomp<double> sol(A);
        REQUIRE(check_lt("RRLU left*right", (A - sol.left() * sol.right()).norm(), tol));
    }
}

TEST_CASE("Test CURDecomp - double", "[cur]")
{
    constexpr double tol = 1000.0 * std::numeric_limits<double>::epsilon();

    SECTION("2x3 matrix") {
        Mat A(2, 3);
        A << 1, 2, 3, 4, 5, 6;

        SECTION("horizontal") {
            for (auto isLeft : {0, 1}) {
                CURDecomp<double> sol(A, isLeft, tol);
                REQUIRE(check_lt("CUR left*right", (A - sol.left() * sol.right()).norm(), tol));
            }
        }

        SECTION("vertical") {
            Mat At = A.transpose();
            CURDecomp<double> sol(At, false, tol);
            REQUIRE(check_lt("CUR left*right", (At - sol.left() * sol.right()).norm(), tol));
        }
    }

    SECTION("from MPO") {
        Mat A = Mat::Zero(4, 7);
        A.row(0).setOnes();
        A.row(3).setOnes();
        A(0, 0) = -1;
        CURDecomp<double> sol(A);
        REQUIRE(check_lt("CUR left*right", (A - sol.left() * sol.right()).norm(), tol));
    }
}

TEST_CASE("Test SVDDecomp - double", "[svd]")
{
    constexpr double tol = 1000.0 * std::numeric_limits<double>::epsilon();

    SECTION("2x3 matrix") {
        Mat A(2, 3);
        A << 1, 2, 3, 4, 5, 6;

        SECTION("horizontal matrix") {
            for (auto isLeft : {0, 1}) {
                SVDDecomp<double> sol(A, isLeft, tol);
                REQUIRE(check_lt("RRLU left*right", (A - sol.left() * sol.right()).norm(), tol));
            }
        }

        SECTION("vertical matrix") {
            Mat At = A.transpose();
            for (auto isLeft : {0, 1}) {
                SVDDecomp<double> sol(At, isLeft, tol);
                REQUIRE(check_lt("RRLU left*right", (At - sol.left() * sol.right()).norm(), tol));
            }
        }
    }
}
