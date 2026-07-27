#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <limits>
#include "xfac_quad/xfac_quad.hpp"

using namespace xfac_quad;
using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

TEST_CASE("SVDDecomp - double", "[svd]")
{
    constexpr double reltol = 100.0 * std::numeric_limits<double>::epsilon();
    constexpr double tol    = 1000.0 * std::numeric_limits<double>::epsilon();

    SECTION("no reltol: full rank random matrix reconstructed exactly")
    {
        MatD A = MatD::Random(10, 8);
        SVDDecomp<double> sol(A, true, 0.0);
        REQUIRE((int)sol.s.size() == 8);
        REQUIRE((A - sol.left() * sol.right()).norm() < tol);
    }

    SECTION("no reltol: left orthogonal")
    {
        MatD A = MatD::Random(10, 8);
        SVDDecomp<double> sol(A, true, 0.0);
        auto should_be_eye = sol.U.transpose() * sol.U;
        MatD eye = MatD::Identity(sol.U.cols(), sol.U.cols());
        REQUIRE((should_be_eye - eye).norm() < tol);
    }

    SECTION("no reltol: right orthogonal")
    {
        MatD A = MatD::Random(10, 8);
        SVDDecomp<double> sol(A, false, 0.0);
        auto should_be_eye = sol.V.transpose() * sol.V;
        MatD eye = MatD::Identity(sol.V.cols(), sol.V.cols());
        REQUIRE((should_be_eye - eye).norm() < tol);
    }

    SECTION("reltol: low rank matrix truncated to correct rank")
    {
        int n = 20, rank = 5;
        MatD A = MatD::Random(n, rank) * MatD::Random(rank, n);
        SVDDecomp<double> sol(A, true, reltol);
        REQUIRE((int)sol.s.size() == rank);
        REQUIRE((A - sol.left() * sol.right()).norm() < tol * A.norm());
    }

    SECTION("reltol: full rank matrix")
    {
        int rank = 15;
        MatD A = MatD::Random(20, rank);
        SVDDecomp<double> sol(A, true, reltol);
        REQUIRE((int)sol.s.size() == rank);
        REQUIRE((A - sol.left() * sol.right()).norm() < tol);
    }

    SECTION("reltol: reconstruction error within tolerance")
    {
        int n = 20, rank = 5;
        MatD A = MatD::Random(n, rank) * MatD::Random(rank, n);
        double bigRel = 1e-6;
        SVDDecomp<double> sol(A, true, bigRel);
        REQUIRE((A - sol.left() * sol.right()).norm() < bigRel * A.norm());
    }

    SECTION("rankMax: caps the rank even if reltol would keep more")
    {
        int n = 20, rank = 10;
        MatD A = MatD::Random(n, rank) * MatD::Random(rank, n);
        int rankMax = 4;
        SVDDecomp<double> sol(A, true, 0.0, rankMax);
        REQUIRE((int)sol.s.size() == rankMax);
    }

    SECTION("exact zeros: low rank matrix with trailing zero singular values cut")
    {
        int n = 10, rank = 3;
        MatD A = MatD::Random(n, rank) * MatD::Random(rank, n);
        SVDDecomp<double> sol(A, true, reltol);
        REQUIRE((int)sol.s.size() == rank);
        for (int i = 0; i < (int)sol.s.size(); i++)
            REQUIRE(sol.s[i] > 0.0);
    }

    SECTION("exact zeros: zero matrix returns rank 1")
    {
        MatD A = MatD::Zero(10, 10);
        SVDDecomp<double> sol(A, true, reltol);
        REQUIRE((int)sol.s.size() == 1);
    }

    SECTION("exact zeros: reconstruction still correct after cutting zeros")
    {
        int n = 10, rank = 3;
        MatD A = MatD::Random(n, rank) * MatD::Random(rank, n);
        SVDDecomp<double> sol(A, true, 0.0);
        REQUIRE((A - sol.left() * sol.right()).norm() < tol * A.norm());
    }
}

TEST_CASE("MatQR - double", "[qr]")
{
    using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    constexpr double tol = 1000.0 * std::numeric_limits<double>::epsilon();

    SECTION("left orthogonal: Q^T Q = I")
    {
        Mat A = Mat::Random(10, 6);
        auto [Q, R] = MatQR<double>{}(A, true);
        Mat should_be_eye = Q.transpose() * Q;
        Mat eye = Mat::Identity(Q.cols(), Q.cols());
        REQUIRE((should_be_eye - eye).norm() < tol);
    }

    SECTION("right orthogonal: Q Q^T = I")
    {
        Mat A = Mat::Random(6, 10);
        auto [L, Q] = MatQR<double>{}(A, false);
        Mat should_be_eye = Q * Q.transpose();
        Mat eye = Mat::Identity(Q.rows(), Q.rows());
        REQUIRE((should_be_eye - eye).norm() < tol);
    }

    SECTION("reconstruction: Q*R = A")
    {
        Mat A = Mat::Random(10, 6);
        auto [Q, R] = MatQR<double>{}(A, true);
        REQUIRE((A - Q * R).norm() < tol * A.norm());
    }

    SECTION("reconstruction (right): L*Q = A")
    {
        Mat A = Mat::Random(6, 10);
        auto [L, Q] = MatQR<double>{}(A, false);
        REQUIRE((A - L * Q).norm() < tol * A.norm());
    }
}
