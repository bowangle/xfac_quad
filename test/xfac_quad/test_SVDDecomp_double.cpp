#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <limits>
#include "xfac_quad/xfac_quad.hpp"

using namespace xfac_quad;
using T = double;
using Real = double;
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

constexpr Real ci_reltol = Real(10000) * std::numeric_limits<Real>::epsilon();
constexpr Real ci_tol    = Real(1000) * std::numeric_limits<Real>::epsilon();
constexpr Real ci_tol2   = Real(10000) * std::numeric_limits<Real>::epsilon();

TEST_CASE("SVDDecomp - double", "[svd]")
{
    SECTION("no ci_reltol: full rank random matrix reconstructed exactly")
    {
        Mat A = Mat::Random(10, 8);
        SVDDecomp<double> sol(A, true, 0.0);
        REQUIRE((int)sol.s.size() == 8);
        REQUIRE((A - sol.left() * sol.right()).norm() < ci_tol);
    }

    SECTION("no ci_reltol: left orthogonal")
    {
        Mat A = Mat::Random(10, 8);
        SVDDecomp<double> sol(A, true, 0.0);
        auto should_be_eye = sol.U.transpose() * sol.U;
        Mat eye = Mat::Identity(sol.U.cols(), sol.U.cols());
        REQUIRE((should_be_eye - eye).norm() < ci_tol);
    }

    SECTION("no ci_reltol: right orthogonal")
    {
        Mat A = Mat::Random(10, 8);
        SVDDecomp<double> sol(A, false, 0.0);
        auto should_be_eye = sol.V.transpose() * sol.V;
        Mat eye = Mat::Identity(sol.V.cols(), sol.V.cols());
        REQUIRE((should_be_eye - eye).norm() < ci_tol);
    }

    SECTION("ci_reltol: low rank matrix truncated to correct rank")
    {
        int n = 20, rank = 5;
        Mat A = Mat::Random(n, rank) * Mat::Random(rank, n);
        SVDDecomp<double> sol(A, true, ci_reltol);
        REQUIRE((int)sol.s.size() == rank);
        REQUIRE((A - sol.left() * sol.right()).norm() < ci_tol * A.norm());
    }

    SECTION("ci_reltol: full rank matrix")
    {
        int rank = 15;
        Mat A = Mat::Random(20, rank);
        SVDDecomp<double> sol(A, true, ci_reltol);
        REQUIRE((int)sol.s.size() == rank);
        REQUIRE((A - sol.left() * sol.right()).norm() < ci_tol);
    }

    SECTION("ci_reltol: reconstruction error within tolerance")
    {
        int n = 20, rank = 5;
        Mat A = Mat::Random(n, rank) * Mat::Random(rank, n);
        double bigRel = 1e-6;
        SVDDecomp<double> sol(A, true, bigRel);
        REQUIRE((A - sol.left() * sol.right()).norm() < bigRel * A.norm());
    }

    SECTION("rankMax: caps the rank even if ci_reltol would keep more")
    {
        int n = 20, rank = 10;
        Mat A = Mat::Random(n, rank) * Mat::Random(rank, n);
        int rankMax = 4;
        SVDDecomp<double> sol(A, true, 0.0, rankMax);
        REQUIRE((int)sol.s.size() == rankMax);
    }

    SECTION("exact zeros: low rank matrix with trailing zero singular values cut")
    {
        int n = 10, rank = 3;
        Mat A = Mat::Random(n, rank) * Mat::Random(rank, n);
        SVDDecomp<double> sol(A, true, ci_reltol);
        REQUIRE((int)sol.s.size() == rank);
        for (int i = 0; i < (int)sol.s.size(); i++)
            REQUIRE(sol.s[i] > 0.0);
    }

    SECTION("exact zeros: zero matrix returns rank 1")
    {
        Mat A = Mat::Zero(10, 10);
        SVDDecomp<double> sol(A, true, ci_reltol);
        REQUIRE((int)sol.s.size() == 1);
    }

    SECTION("exact zeros: reconstruction still correct after cutting zeros")
    {
        int n = 10, rank = 3;
        Mat A = Mat::Random(n, rank) * Mat::Random(rank, n);
        SVDDecomp<double> sol(A, true, 0.0);
        REQUIRE((A - sol.left() * sol.right()).norm() < ci_tol * A.norm());
    }
}

TEST_CASE("MatQR - double", "[qr]")
{

    SECTION("left orthogonal: Q^T Q = I")
    {
        Mat A = Mat::Random(10, 6);
        auto [Q, R] = MatQR<T>{}(A, true);
        Mat should_be_eye = Q.transpose() * Q;
        Mat eye = Mat::Identity(Q.cols(), Q.cols());
        REQUIRE((should_be_eye - eye).norm() < ci_tol);
    }

    SECTION("right orthogonal: Q Q^T = I")
    {
        Mat A = Mat::Random(6, 10);
        auto [L, Q] = MatQR<T>{}(A, false);
        Mat should_be_eye = Q * Q.transpose();
        Mat eye = Mat::Identity(Q.rows(), Q.rows());
        REQUIRE((should_be_eye - eye).norm() < ci_tol);
    }

    SECTION("reconstruction: Q*R = A")
    {
        Mat A = Mat::Random(10, 6);
        auto [Q, R] = MatQR<T>{}(A, true);
        REQUIRE((A - Q * R).norm() < ci_tol * A.norm());
    }

    SECTION("reconstruction (right): L*Q = A")
    {
        Mat A = Mat::Random(6, 10);
        auto [L, Q] = MatQR<T>{}(A, false);
        REQUIRE((A - L * Q).norm() < ci_tol * A.norm());
    }
}
