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
#include "xfac_quad/matrix/matrix_interface.h"
#include "xfac_quad/matrix/cross_data.h"

using namespace xfac_quad;
using std::abs;
using T      = dd_128;
using CT     = Cdd_128;
using Real   = dd_128;
using Mat    = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using MatCT  = Eigen::Matrix<CT, Eigen::Dynamic, Eigen::Dynamic>;
using MatUint= Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic>;

const Real ci_reltol = Real(10000) * std::numeric_limits<Real>::epsilon();
const Real ci_tol    = Real(1000) * std::numeric_limits<Real>::epsilon();
const Real ci_tol2   = Real(100000) * std::numeric_limits<Real>::epsilon();
static const T Pi = dd_real::_pi;

// --- debug helper ---
template<typename A, typename B>
bool check_le(std::string const& expr, A const& val, B const& tol_) {
    if (!(val <= tol_)) {
        std::cerr << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << "  FAIL " << expr << ": val=" << val << " > ci_tol=" << tol_ << "\n";
        return false;
    }
    return true;
}
template<typename A, typename B>
bool check_lt(std::string const& expr, A const& val, B const& tol_) {
    if (!(val < tol_)) {
        std::cerr << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << "  FAIL " << expr << ": val=" << val << " >= ci_tol=" << tol_ << "\n";
        return false;
    }
    return true;
}

// ============================================================
// TensorCI1 Tests
// ============================================================

TEST_CASE("Test MatrixCI1 - dd128", "[matrix_ci1]")
{
    SECTION("fixed rank")
    {
        int n = 100, rank = 50;
        Mat A = Mat::Random(n, rank) * Mat::Random(rank, n);
        auto myf = [&](vector<int> const& id) -> T {
            return A(id[0], id[1]);
        };
        vector<int> dims = {(int)A.rows(), (int)A.cols()};
        TensorCI1Param<T> p;
        p.nIter = rank + 1;
        p.reltol = ci_reltol;
        auto ci = TensorCI1<T>(myf, dims, p);
        REQUIRE(check_le("pivotError*norm", ci.pivotError.back(), ci_tol2 * A.norm()));
        REQUIRE(check_le("trueError*norm", ci.trueError(), ci_tol2 * A.norm()));
        int i = rand() % A.rows(), j = rand() % A.cols();
        REQUIRE(check_lt("tt.eval", abs(ci.get_TensorTrain().eval({i, j}) - A(i, j)), ci_tol2 * A.norm()));
    }

    SECTION("rank-1 function")
    {
        auto myf = [](vector<int> const& id) -> T {
            return (T(id[0]) + T(1)) * (T(id[1]) + T(2));
        };
        TensorCI1Param<T> p;
        p.pivot1 = {0, 0};
        p.reltol = ci_reltol;
        auto ci = TensorCI1<T>(myf, {20, 20}, p);
        REQUIRE(check_le("trueError", ci.trueError(), ci_tol2));
    }
}

TEST_CASE("Test MatrixCI1 complex - dd128", "[matrix_ci1]")
{
    SECTION("rank-2 separable function")
    {
        auto myf = [](vector<int> const& id) -> CT {
            return CT(1, 2) * CT(T(id[0]) + T(1), T(0)) * CT(T(id[1]) + T(2), T(0))
                 + CT(3, -1) * CT(T(id[0]) * T(4), T(0)) * CT(T(id[1]) + T(5), T(0));
        };
        TensorCI1Param<CT> p;
        p.nIter = 10;
        p.fullPiv = true;
        p.pivot1 = {0, 0};
        p.reltol = ci_reltol;
        auto ci = TensorCI1<CT>(myf, {20, 20}, p);
        REQUIRE(check_le("trueError", ci.trueError(), ci_tol2));
    }

    SECTION("rank-1 complex function")
    {
        auto myf = [](vector<int> const& id) -> CT {
            return CT(1, 2) * CT(T(id[0]) + T(1), T(0)) * CT(T(id[1]) + T(2), T(0));
        };
        TensorCI1Param<CT> p;
        p.pivot1 = {0, 0};
        p.reltol = ci_reltol;
        auto ci = TensorCI1<CT>(myf, {20, 20}, p);
        REQUIRE(check_le("trueError", ci.trueError(), ci_tol2));
    }

    SECTION("continuous function - discretized")
    {
        auto myf = [](T x, T y) -> CT {
            T arg = T(1) + (x + T(2)*y + x*y) * Pi;
            return CT(T(1)+x+cos(arg), x*x+T(0.5)*sin(arg));
        };
        auto xi = grid::linspace<T>(T(0), T(1), 100u);
        int rank = 15;

        MatCT A(static_cast<Eigen::Index>(xi.size()), static_cast<Eigen::Index>(xi.size()));
        for (size_t i = 0; i < xi.size(); i++)
            for (size_t j = 0; j < xi.size(); j++)
                A(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = myf(xi[i], xi[j]);

        auto myTf = [&](vector<int> const& id) -> CT { return A(id[0], id[1]); };
        vector<int> dims = {(int)A.rows(), (int)A.cols()};
        TensorCI1Param<CT> p;
        p.fullPiv = true;
        p.reltol = ci_reltol;
        auto ci = TensorCI1<CT>(myTf, dims, p);
        for (int r = 1; r <= rank; r++) ci.iterate();

        REQUIRE(check_le("trueError", ci.trueError(), ci_tol2 * A.norm()));
        REQUIRE(check_le("pivotError", ci.pivotError.back(), ci_tol2 * A.norm()));
        int i = rand() % (int)A.rows(), j = rand() % (int)A.cols();
        REQUIRE(check_lt("tt.eval", abs(ci.get_TensorTrain().eval({i, j}) - A(i, j)), ci_tol2 * A.norm()));
    }

    SECTION("continuous function - directly")
    {
        auto myf = [](T x, T y) -> CT {
            T arg = T(1) + (x + T(2)*y + x*y) * Pi;
            return CT(T(1)+x+cos(arg), x*x+T(0.5)*sin(arg));
        };
        auto xi = grid::linspace<T>(T(0), T(1), 100u);
        int rank = 15;

        auto myTf = [&](vector<T> xs) -> CT { return myf(xs[0], xs[1]); };
        auto ci = CTensorCI1<CT,T>(myTf, vector(2, xi), {.nIter=rank, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=5, .weight={}, .cond={}, .useCachedFunction=true});
        REQUIRE(check_le("pivotError", ci.pivotError.back(), ci_tol2));
        T x = T(1.0 * rand() / RAND_MAX), y = T(1.0 * rand() / RAND_MAX);
        REQUIRE(check_lt("ctt.eval", abs(ci.get_CTensorTrain().eval({x, y}) - myTf({x, y})), ci_tol2));
    }
}

// ============================================================
// TensorCI2 Tests
// ============================================================

TEST_CASE("Test MatrixCI2 - dd128", "[matrix_ci2]")
{
    SECTION("fixed rank")
    {
        int n = 100, rank = 50;
        Mat A = Mat::Random(n, rank) * Mat::Random(rank, n);
        auto myf = [&](vector<int> const& id) -> T { return A(id[0], id[1]); };
        vector<int> dims = {(int)A.rows(), (int)A.cols()};
        auto ci = TensorCI2<T>(TensorFunction<T>(myf), dims, {.bondDim=rank+10, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        ci.iterate(6);

        REQUIRE(ci.pivotError.size() - 1 == static_cast<size_t>(rank));
        REQUIRE(check_le("trueError", ci.trueError(), ci_tol2 * A.norm()));
        int i = rand() % A.rows(), j = rand() % A.cols();
        REQUIRE(check_lt("tt.eval", abs(ci.tt.eval({i, j}) - A(i, j)), ci_tol2 * A.norm()));
    }

    SECTION("continuous function - discretized")
    {
        auto myf = [](T x, T y) -> CT {
            T arg = T(1) + (x + T(2)*y + x*y) * Pi;
            return CT(T(1)+x+cos(arg), x*x+T(0.5)*sin(arg));
        };
        auto xi = grid::linspace<T>(T(0), T(1), 100u);
        int rank = 15;

        MatCT A(static_cast<Eigen::Index>(xi.size()), static_cast<Eigen::Index>(xi.size()));
        for (size_t i = 0; i < xi.size(); i++)
            for (size_t j = 0; j < xi.size(); j++)
                A(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = myf(xi[i], xi[j]);

        auto myTf = [&](vector<int> const& id) -> CT { return A(id[0], id[1]); };
        vector<int> dims = {(int)A.rows(), (int)A.cols()};
        auto ci = TensorCI2<CT>(TensorFunction<CT>(myTf), dims, {.bondDim=rank, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        ci.iterate(5);

        REQUIRE(ci.pivotError.size() - 1 <= static_cast<size_t>(rank));
        REQUIRE(check_le("trueError", ci.trueError(), ci_tol2 * A.norm()));
        int i = rand() % (int)A.rows(), j = rand() % (int)A.cols();
        REQUIRE(check_lt("tt.eval", abs(ci.tt.eval({i, j}) - A(i, j)), ci_tol2 * A.norm()));
    }

    SECTION("continuous function - directly")
    {
        auto myf = [](T x, T y) -> CT {
            T arg = T(1) + (x + T(2)*y + x*y) * Pi;
            return CT(T(1)+x+cos(arg), x*x+T(0.5)*sin(arg));
        };
        auto xi = grid::linspace<T>(T(0), T(1), 100u);
        int rank = 15;

        auto myTf = [&](vector<T> xs) -> CT { return myf(xs[0], xs[1]); };
        auto ci = CTensorCI2<CT,T>(myTf, vector(2, xi), {.bondDim=2*rank, .reltol=ci_reltol, .pivot1={}, .fullPiv=false, .nRookIter=3, .weight={}, .cond={}, .useCachedFunction=true});
        ci.iterate(5);
        REQUIRE(check_le("trueError", ci.trueError(), ci_tol2));
        T x = T(1.0 * rand() / RAND_MAX), y = T(1.0 * rand() / RAND_MAX);
        REQUIRE(check_lt("ctt.eval", abs(ci.get_CTensorTrain().eval({x, y}) - myTf({x, y})), ci_tol2));
        ci.makeCanonical();
        REQUIRE(check_lt("ctt.eval canonical", abs(ci.get_CTensorTrain().eval({x, y}) - myTf({x, y})), ci_tol2));
    }
}

// ============================================================
// CrossData Tests
// ============================================================

TEST_CASE("Test CrossData - dd128", "[cross]")
{
    Mat A = Mat::Random(100, 120).array().abs();
    A /= A.norm();

    // select random pivots
    vector<int> I, J;
    int maxPivots = static_cast<int>(0.2 * A.rows());
    while (static_cast<int>(I.size()) < maxPivots) {
        int i = rand() % static_cast<int>(A.rows()), j = rand() % static_cast<int>(A.cols());
        if (I.empty()) {
            I.push_back(i); J.push_back(j);
            continue;
        }
        Mat AIJ(I.size(), J.size());
        for (size_t ri = 0; ri < I.size(); ri++)
            for (size_t rj = 0; rj < J.size(); rj++)
                AIJ(static_cast<Eigen::Index>(ri), static_cast<Eigen::Index>(rj)) = A(I[ri], J[rj]);

        Eigen::Matrix<T,1,Eigen::Dynamic> AiJ(static_cast<Eigen::Index>(J.size()));
        for (size_t rj = 0; rj < J.size(); rj++)
            AiJ(static_cast<Eigen::Index>(rj)) = A(i, J[rj]);

        Eigen::Matrix<T,Eigen::Dynamic,1> AIj(static_cast<Eigen::Index>(I.size()));
        for (size_t ri = 0; ri < I.size(); ri++)
            AIj(static_cast<Eigen::Index>(ri)) = A(I[ri], j);

        Real err = abs(A(i,j) - static_cast<Real>((AiJ * AIJ.inverse() * AIj)(0)));
        if (err > ci_tol) { I.push_back(i); J.push_back(j); }
    }

    Mat C(A.rows(), static_cast<Eigen::Index>(J.size()));
    for (Eigen::Index c = 0; c < C.cols(); c++)
        C.col(c) = A.col(J[static_cast<size_t>(c)]);
    Mat RR(static_cast<Eigen::Index>(I.size()), A.cols());
    for (Eigen::Index r = 0; r < RR.rows(); r++)
        RR.row(r) = A.row(I[static_cast<size_t>(r)]);

    SECTION("interpolation") {
        auto cross = CrossData<T>(I, J, C, RR);
        SECTION("direct formula") {
            Mat AIJ_full(I.size(), J.size());
            for (size_t ri = 0; ri < I.size(); ri++)
                for (size_t rj = 0; rj < J.size(); rj++)
                    AIJ_full(static_cast<Eigen::Index>(ri), static_cast<Eigen::Index>(rj)) = A(I[ri], J[rj]);
            Mat Aci = C * AIJ_full.inverse() * RR;
            Mat Aci_I(I.size(), A.cols()), A_I(I.size(), A.cols());
            for (Eigen::Index r = 0; r < Aci_I.rows(); r++) {
                Aci_I.row(r) = Aci.row(I[static_cast<size_t>(r)]);
                A_I.row(r)   = A.row(I[static_cast<size_t>(r)]);
            }
            REQUIRE(check_lt("Aci_I-A_I", (Aci_I - A_I).norm(), ci_tol));
            Mat Aci_J(A.rows(), J.size()), A_J(A.rows(), J.size());
            for (Eigen::Index c = 0; c < Aci_J.cols(); c++) {
                Aci_J.col(c) = Aci.col(J[static_cast<size_t>(c)]);
                A_J.col(c)   = A.col(J[static_cast<size_t>(c)]);
            }
            REQUIRE(check_lt("Aci_J-A_J", (Aci_J - A_J).norm(), ci_tol));
        }
        SECTION("at cross") {
            Mat Across = cross.mat();
            Mat Across_I(I.size(), A.cols()), A_I2(I.size(), A.cols());
            for (Eigen::Index r = 0; r < Across_I.rows(); r++) {
                Across_I.row(r) = Across.row(I[static_cast<size_t>(r)]);
                A_I2.row(r)     = A.row(I[static_cast<size_t>(r)]);
            }
            REQUIRE(check_lt("Across_I-A_I", (Across_I - A_I2).norm(), ci_tol));
            Mat Across_J(A.rows(), J.size()), A_J2(A.rows(), J.size());
            for (Eigen::Index c = 0; c < Across_J.cols(); c++) {
                Across_J.col(c) = Across.col(J[static_cast<size_t>(c)]);
                A_J2.col(c)     = A.col(J[static_cast<size_t>(c)]);
            }
            REQUIRE(check_lt("Across_J-A_J", (Across_J - A_J2).norm(), ci_tol));
        }
        SECTION("each row/col") {
            for (int i : I)
                REQUIRE(abs(cross.eval(i, 0) - A(i, 0)) < ci_tol * Real(100));
        }
    }
    SECTION("insert") {
        auto cross = CrossData<T>(A.rows(), A.cols());
        MatDense<T> Adense(A);
        for (size_t c = 0; c < I.size(); c++)
            cross.addPivot(I[c], J[c], Adense);
        SECTION("well copied") {
            Mat Rcopy = cross.R, Ccopy = cross.C;
            Mat A_I3(I.size(), A.cols()), A_J3(A.rows(), J.size());
            for (Eigen::Index r = 0; r < A_I3.rows(); r++)
                A_I3.row(r) = A.row(I[static_cast<size_t>(r)]);
            for (Eigen::Index c = 0; c < A_J3.cols(); c++)
                A_J3.col(c) = A.col(J[static_cast<size_t>(c)]);
            REQUIRE(check_lt("cross.R - A.rows(I)", (Rcopy - A_I3).norm(), ci_tol));
            REQUIRE(check_lt("cross.C - A.cols(J)", (Ccopy - A_J3).norm(), ci_tol));
        }
        SECTION("interpolation at cross") {
            Mat Across = cross.mat();
            Mat Across_I(I.size(), A.cols()), A_I4(I.size(), A.cols());
            for (Eigen::Index r = 0; r < Across_I.rows(); r++) {
                Across_I.row(r) = Across.row(I[static_cast<size_t>(r)]);
                A_I4.row(r)     = A.row(I[static_cast<size_t>(r)]);
            }
            REQUIRE(check_lt("insert cross rows", (Across_I - A_I4).norm(), ci_tol));
            Mat Across_J(A.rows(), J.size()), A_J4(A.rows(), J.size());
            for (Eigen::Index c = 0; c < Across_J.cols(); c++) {
                Across_J.col(c) = Across.col(J[static_cast<size_t>(c)]);
                A_J4.col(c)     = A.col(J[static_cast<size_t>(c)]);
            }
            REQUIRE(check_lt("insert cross cols", (Across_J - A_J4).norm(), ci_tol));
        }
    }
}

// ============================================================
// RRLUDecomp Tests
// ============================================================

TEST_CASE("Test RRLUDecomp - dd128", "[rrlu]")
{
    SECTION("2x3 matrix") {
        Mat A(2, 3);
        A << 1, 2, 3, 4, 5, 6;

        SECTION("horizontal matrix") {
            for (auto isLeft : {0, 1}) {
                RRLUDecomp<T> sol(A, isLeft, ci_tol);
                REQUIRE(check_lt("RRLU left*right", (A - sol.left() * sol.right()).norm(), ci_tol));
            }
        }
        SECTION("vertical matrix") {
            Mat At = A.transpose();
            for (auto isLeft : {0, 1}) {
                RRLUDecomp<T> sol(At, isLeft, ci_tol);
                REQUIRE(check_lt("RRLU left*right", (At - sol.left() * sol.right()).norm(), ci_tol));
            }
        }
        SECTION("horizontal matrix cond") {
            MatUint cond(2, 3);
            cond << 0, 1, 1, 1, 1, 0;
            for (auto isLeft : {0, 1}) {
                RRLUDecomp<T> sol(A, cond, isLeft, ci_tol);
                REQUIRE(check_lt("RRLU cond left*right", (A - sol.left() * sol.right()).norm(), ci_tol));
                auto rp = sol.row_pivots();
                auto cp = sol.col_pivots();
                for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(rp.size()); k++)
                    REQUIRE(cond(rp[static_cast<size_t>(k)], cp[static_cast<size_t>(k)]) == 1);
            }
        }
        SECTION("vertical matrix cond") {
            Mat At = A.transpose();
            MatUint condT_t(3, 2);
            condT_t << 0, 1, 1, 1, 1, 0;
            for (auto isLeft : {0, 1}) {
                RRLUDecomp<T> sol(At, condT_t, isLeft, ci_tol);
                REQUIRE(check_lt("RRLU condT left*right", (At - sol.left() * sol.right()).norm(), ci_tol));
                auto rp = sol.row_pivots();
                auto cp = sol.col_pivots();
                for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(rp.size()); k++)
                    REQUIRE(condT_t(rp[static_cast<size_t>(k)], cp[static_cast<size_t>(k)]) == 1);
            }
        }
    }

    SECTION("from MPO") {
        Mat A = Mat::Zero(4, 7);
        A.row(0).setOnes();
        A.row(3).setOnes();
        A(0, 0) = -1;
        RRLUDecomp<T> sol(A);
        REQUIRE(check_lt("RRLU MPO left*right", (A - sol.left() * sol.right()).norm(), ci_tol));
    }
}

// ============================================================
// CURDecomp Tests
// ============================================================

TEST_CASE("Test CURDecomp - dd128", "[cur]")
{
    SECTION("2x3 matrix") {
        Mat A(2, 3);
        A << 1, 2, 3, 4, 5, 6;

        SECTION("horizontal") {
            for (auto isLeft : {0, 1}) {
                CURDecomp<T> sol(A, isLeft, ci_tol);
                REQUIRE(check_lt("CUR left*right", (A - sol.left() * sol.right()).norm(), ci_tol));
            }
        }
        SECTION("vertical") {
            Mat At = A.transpose();
            CURDecomp<T> sol(At, false, ci_tol);
            REQUIRE(check_lt("CUR left*right", (At - sol.left() * sol.right()).norm(), ci_tol));
        }
    }

    SECTION("from MPO") {
        Mat A = Mat::Zero(4, 7);
        A.row(0).setOnes();
        A.row(3).setOnes();
        A(0, 0) = -1;
        CURDecomp<T> sol(A);
        REQUIRE(check_lt("CUR MPO left*right", (A - sol.left() * sol.right()).norm(), ci_tol));
    }
}

// ============================================================
// ARRLUDecomp Tests
// ============================================================

TEST_CASE("Test ARRLUDecomp - dd128", "[arrlu]")
{
    auto to_mat_fun = [](Mat A) {
        auto submat = [A](vector<int> I, vector<int> J) -> Mat {
            Mat result(static_cast<Eigen::Index>(I.size()), static_cast<Eigen::Index>(J.size()));
            for (Eigen::Index ri = 0; ri < result.rows(); ri++)
                for (Eigen::Index rj = 0; rj < result.cols(); rj++)
                    result(ri, rj) = A(I[static_cast<size_t>(ri)], J[static_cast<size_t>(rj)]);
            return result;
        };
        return MatFun<T> {static_cast<size_t>(A.rows()), static_cast<size_t>(A.cols()), submat};
    };

    SECTION("2x3 matrix") {
        Mat A(2, 3);
        A << 1, 2, 3, 4, 5, 6;

        SECTION("horizontal matrix") {
            for (auto isLeft : {0, 1}) {
                ARRLUDecomp<T> sol(to_mat_fun(A), {}, {}, isLeft, {.reltol=ci_tol, .bondDim=0, .fullPiv=false, .nRookIter=3});
                REQUIRE(check_lt("ARRLU left*right", (A - sol.left() * sol.right()).norm(), ci_tol));
            }
        }
        SECTION("vertical matrix") {
            for (auto isLeft : {0, 1}) {
                ARRLUDecomp<T> sol(to_mat_fun(A.transpose()), {}, {}, isLeft, {.reltol=ci_tol, .bondDim=0, .fullPiv=false, .nRookIter=3});
                REQUIRE(check_lt("ARRLU left*right", (A.transpose() - sol.left() * sol.right()).norm(), ci_tol));
            }
        }
    }

    SECTION("from MPO") {
        Mat A = Mat::Zero(4, 7);
        A.row(0).setOnes();
        A.row(3).setOnes();
        A(0, 0) = -1;
        ARRLUDecomp<T> sol(to_mat_fun(A), {}, {}, true, {.reltol=ci_tol, .bondDim=0, .fullPiv=false, .nRookIter=3});
        REQUIRE(check_lt("ARRLU MPO left*right", (A - sol.left() * sol.right()).norm(), ci_tol));
    }
}

// ============================================================
// SVDDecomp Tests
// ============================================================

TEST_CASE("Test SVDDecomp - dd128", "[svd]")
{
    SECTION("2x3 matrix") {
        Mat A(2, 3);
        A << 1, 2, 3, 4, 5, 6;

        SECTION("horizontal matrix") {
            for (auto isLeft : {0, 1}) {
                SVDDecomp<T> sol(A, isLeft, ci_tol);
                REQUIRE(check_lt("SVD left*right", (A - sol.left() * sol.right()).norm(), ci_tol));
            }
        }
        SECTION("vertical matrix") {
            Mat At = A.transpose();
            for (auto isLeft : {0, 1}) {
                SVDDecomp<T> sol(At, isLeft, ci_tol);
                REQUIRE(check_lt("SVD left*right", (At - sol.left() * sol.right()).norm(), ci_tol));
            }
        }
    }
}
