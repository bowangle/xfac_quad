#pragma once
#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <complex>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include "xfac_quad/index_set.h"
#include "type_float128_boost.h"
#include "type_double_double.h"

namespace xfac_quad {

using std::vector;

// -------------------------------------------------------------------
// Type trait: does T benefit from LAPACK-backed Eigen decompositions?
// -------------------------------------------------------------------
template<typename T>
struct is_standard_blas_type : std::false_type {};
template<> struct is_standard_blas_type<float>                : std::true_type {};
template<> struct is_standard_blas_type<double>               : std::true_type {};
template<> struct is_standard_blas_type<std::complex<float>>  : std::true_type {};
template<> struct is_standard_blas_type<std::complex<double>> : std::true_type {};

//--------------------------------------- QR decomposition -----------------------

template<class T>
struct MatQR {
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    std::array<Mat, 2> operator()(Mat const& A, bool leftOrthogonal) const {
        return leftOrthogonal ? mat_qr(A) : mat_qr_t(A);
    }

private:
    static std::array<Mat, 2> mat_qr(Mat const& A)
    {
        Eigen::HouseholderQR<Mat> qr(A);
        Eigen::Index rows = A.rows();
        Eigen::Index cols = A.cols();
        Eigen::Index k = std::min(rows, cols);
        Mat Q = qr.householderQ() * Mat::Identity(rows, k);
        Mat R = qr.matrixQR().topRows(k).template triangularView<Eigen::Upper>();
        return {std::move(Q), std::move(R)};
    }

    static std::array<Mat, 2> mat_qr_t(Mat const& A)
    {
        auto [Q, R] = mat_qr(Mat(A.adjoint()));
        return {Mat(R.adjoint()), Mat(Q.adjoint())};
    }
};

//--------------------------------------- SVD decomposition -----------------------

template<class T>
struct SVDDecomp {
    using value_type = T;
    using RealScalar = typename Eigen::NumTraits<T>::Real;
    using Mat        = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec        = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

    Mat U, V;
    Vec s;
    bool leftOrthogonal = true;

    SVDDecomp(Mat const& M, bool leftOrthogonal_ = true,
              RealScalar reltol = RealScalar(1e-12), int rankMax = 0)
        : leftOrthogonal(leftOrthogonal_)
    {
        if constexpr (is_standard_blas_type<T>::value) {
            Eigen::BDCSVD<Mat, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(M);
            U = svd.matrixU();
            V = svd.matrixV();
            s = svd.singularValues();
        } else {
            Eigen::JacobiSVD<Mat> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU();
            V = svd.matrixV();
            s = svd.singularValues();
        }

        Eigen::Index n = findnValues(s, reltol);
        if (rankMax > 0 && rankMax < n) n = rankMax;
        s.conservativeResize(n);
        U.conservativeResize(U.rows(), n);
        V.conservativeResize(V.rows(), n);
    }

    Mat left()  const { if (leftOrthogonal) return U; return U * s.template cast<T>().asDiagonal(); }
    Mat right() const { if (leftOrthogonal) return s.template cast<T>().asDiagonal() * V.adjoint(); return V.adjoint(); }

private:
    static Eigen::Index findnValues(Vec const& s, RealScalar reltol)
    {
        if (reltol < RealScalar(0))
            return s.size();
        RealScalar tol2  = reltol * reltol;
        RealScalar norm2 = s.squaredNorm();
        if (norm2 == RealScalar(0)) return 1;
        RealScalar sum = 0;
        Eigen::Index n = s.size();
        for (Eigen::Index i = s.size() - 1; i >= 0; --i) {
            sum += s(i) * s(i);
            if (sum > tol2 * norm2) { n = i + 1; break; }
        }
        return n;
    }
};

//--------------------------------------- Fixed-tolerance wrapper -----------------------

template<class Decomp>
struct MatDecompFixedTol
{
    using T          = typename Decomp::value_type;
    using RealScalar = typename Eigen::NumTraits<T>::Real;
    using Mat        = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    RealScalar tol;
    int rankMax;

    MatDecompFixedTol(RealScalar tol_ = RealScalar(1e-14), int rankMax_ = 0)
        : tol(tol_), rankMax(rankMax_) {}

    std::array<Mat, 2> operator()(Mat const& M, bool leftOrthogonal)
    {
        Decomp s{M, leftOrthogonal, tol, rankMax};
        return { s.left(), s.right() };
    }
};

template<class T>
struct MatSVDFixedTol : public MatDecompFixedTol<SVDDecomp<T>> {
    using MatDecompFixedTol<SVDDecomp<T>>::MatDecompFixedTol;
};

//--------------------------------------- rank-revealing LU decomposition -----------------------

template<class T>
struct RRLUDecomp {
    using value_type = T;
    using RealScalar = typename Eigen::NumTraits<T>::Real;
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using MatUint = Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic>;

    vector<int> Iset, Jset; ///< permutation of rows and columns
    Mat L, U;
    bool leftOrthogonal = true;
    int npivot = 0;
    RealScalar error = 0;

    RRLUDecomp() = default;

    RRLUDecomp(Mat const& A, bool leftOrthogonal_ = true, RealScalar reltol = 0, int rankMax = 0)
        : RRLUDecomp(A, MatUint(), leftOrthogonal_, reltol, rankMax) {}

    RRLUDecomp(Mat const& A, MatUint const& cond_, bool leftOrthogonal_ = true, RealScalar reltol = 0, int rankMax = 0)
        : Iset(iota(static_cast<int>(A.rows())))
        , Jset(iota(static_cast<int>(A.cols())))
        , leftOrthogonal(leftOrthogonal_)
    { calculate(A, reltol, rankMax, cond_); }

    void calculate(Mat A, RealScalar reltol, int rankMax, MatUint cond = MatUint())
    {
        using std::abs;
        npivot = std::min(static_cast<int>(A.rows()), static_cast<int>(A.cols()));
        if (reltol == 0) reltol = RealScalar(npivot) * Eigen::NumTraits<T>::epsilon();
        RealScalar max_error = 0;
        if (rankMax > 0 && rankMax < npivot) npivot = rankMax;

        for (int k = 0; k < npivot; k++) {
            auto subBlock = A.block(k, k, A.rows()-k, A.cols()-k);
            Eigen::Index maxRow, maxCol;
            if (cond.size() == 0) {
                subBlock.cwiseAbs().maxCoeff(&maxRow, &maxCol);
            } else {
                auto condBlock = cond.block(k, k, A.rows()-k, A.cols()-k);
                (subBlock.cwiseAbs().template cast<RealScalar>().cwiseProduct(
                    condBlock.template cast<RealScalar>())).maxCoeff(&maxRow, &maxCol);
            }
            auto i0 = static_cast<int>(maxRow) + k;
            auto j0 = static_cast<int>(maxCol) + k;
            if (cond.size() != 0)
                if (cond(i0, j0) == 0) { npivot = k; break; }
            if (RealScalar err = abs(A(i0, j0));
                k > 0 && err < reltol * max_error) { npivot = k; break; }
            else max_error = std::max(max_error, err);

            std::swap(Iset[k], Iset[i0]);
            std::swap(Jset[k], Jset[j0]);
            A.row(k).swap(A.row(i0));
            A.col(k).swap(A.col(j0));
            if (cond.size() != 0) {
                cond.row(k).swap(cond.row(i0));
                cond.col(k).swap(cond.col(j0));
            }

            if (k+1 < A.rows() && leftOrthogonal)
                A.col(k).tail(A.rows()-k-1) *= T(1) / A(k,k);
            else if (k+1 < A.cols() && !leftOrthogonal)
                A.row(k).tail(A.cols()-k-1) *= T(1) / A(k,k);
            if (k+1 < npivot)
                A.bottomRightCorner(A.rows()-k-1, A.cols()-k-1) -=
                    A.col(k).tail(A.rows()-k-1) * A.row(k).tail(A.cols()-k-1);
        }
        if (npivot < std::min(static_cast<int>(A.rows()), static_cast<int>(A.cols()))) {
            auto remBlock = A.block(npivot, npivot, A.rows()-npivot, A.cols()-npivot);
            error = (cond.size() == 0) ? remBlock.cwiseAbs().maxCoeff()
                     : remBlock.cwiseAbs().template cast<RealScalar>().cwiseProduct(
                         cond.block(npivot, npivot, A.rows()-npivot, A.cols()-npivot).template cast<RealScalar>()).maxCoeff();
        }
        readLU(A);
    }

    vector<int> row_pivots() const { return {Iset.begin(), Iset.begin()+npivot}; }
    vector<int> col_pivots() const { return {Jset.begin(), Jset.begin()+npivot}; }
    Mat PivotMatrixTri() const
    {
        Mat P = U.topLeftCorner(npivot, npivot);
        for (int j = 0; j < P.cols(); j++)
            for (int i = j + static_cast<int>(leftOrthogonal); i < P.rows(); i++)
                P(i, j) = L(i, j);
        return P;
    }

    vector<RealScalar> pivotErrors() const
    {
        using std::abs;
        vector<RealScalar> out(npivot);
        auto diag = leftOrthogonal ? U.diagonal() : L.diagonal();
        for (int i = 0; i < npivot; i++)
            out[i] = abs(diag(i));
        if (npivot < std::min(static_cast<int>(L.rows()), static_cast<int>(U.cols())))
            out.push_back(error);
        return out;
    }

    /// return the matrix L with permuted rows. left()*right() gives the rank=npivot reconstructed matrix.
    Mat left() const {
        auto invP = inversePermutation(Iset);
        Mat result(invP.size(), L.cols());
        for (size_t i = 0; i < invP.size(); i++)
            result.row(i) = L.row(invP[i]);
        return result;
    }

    /// return the matrix U with permuted columns. left()*right() gives the rank=npivot reconstructed matrix.
    Mat right() const {
        auto invP = inversePermutation(Jset);
        Mat result(U.rows(), invP.size());
        for (size_t j = 0; j < invP.size(); j++)
            result.col(j) = U.col(invP[j]);
        return result;
    }

protected:
    void readLU(Mat const& lu) {
        L = Mat::Identity(lu.rows(), npivot);
        for (int j = 0; j < L.cols(); j++)
            for (int i = j + static_cast<int>(leftOrthogonal); i < L.rows(); i++)
                L(i, j) = lu(i, j);

        U = Mat::Identity(npivot, lu.cols());
        for (int j = 0; j < U.cols(); j++) {
            int m = std::min(j + static_cast<int>(leftOrthogonal), static_cast<int>(U.rows()));
            for (int i = 0; i < m; i++)
                U(i, j) = lu(i, j);
        }
    }
};

/// Given the pivot matrix P already in LU form, update the C columns according to P.
template<class T>
void apply_LU_on_cols(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& C,
                      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> const& P, bool leftOrthogonal)
{
    if (P.rows() != C.rows())
        throw std::invalid_argument("apply_LU_on_Cols: incompatible matrices");
    for (int k = 0; k < P.rows(); k++) {
        if (!leftOrthogonal)
            C.row(k) *= T(1) / P(k,k);
        if (k+1 < P.rows())
            C.bottomRows(C.rows()-k-1) -= P.col(k).segment(k+1, C.rows()-k-1) * C.row(k);
    }
}

/// Given the pivot matrix P already in LU form, update the R rows according to P.
template<class T>
void apply_LU_on_rows(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& R,
                      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> const& P, bool leftOrthogonal)
{
    if (P.cols() != R.cols())
        throw std::invalid_argument("apply_LU_on_Rows: incompatible matrices");
    for (int k = 0; k < P.rows(); k++) {
        if (leftOrthogonal)
            R.col(k) *= T(1) / P(k,k);
        if (k+1 < P.rows())
            R.rightCols(R.cols()-k-1) -= R.col(k) * P.row(k).segment(k+1, P.cols()-k-1);
    }
}

template<class T>
struct MatRRLUFixedTol : public MatDecompFixedTol<RRLUDecomp<T>> {
    using MatDecompFixedTol<RRLUDecomp<T>>::MatDecompFixedTol;
};


//--------------------------------------- alternate rank-revealing LU decomposition -----------------------

template<class T>
struct MatFun {
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    size_t n_rows = 0;
    size_t n_cols = 0;
    std::function<Mat(vector<int> const&, vector<int> const&)> submat;
};

template<class T>
struct ARRLUParam {
    using RealScalar = typename Eigen::NumTraits<T>::Real;
    RealScalar reltol = 0;
    int bondDim = 0;
    bool fullPiv = false;
    int nRookIter = 3;
};

template<class T>
struct ARRLUDecomp : public RRLUDecomp<T> {
    using typename RRLUDecomp<T>::Mat;
    using typename RRLUDecomp<T>::MatUint;
    using RRLUDecomp<T>::Iset;
    using RRLUDecomp<T>::Jset;
    using RRLUDecomp<T>::npivot;
    using RRLUDecomp<T>::L;
    using RRLUDecomp<T>::U;
    using RRLUDecomp<T>::RRLUDecomp;
    // MatFun<unsigned int> fCond; it's getting shadow and never used. This is a one method struct.
    // Nothing need to be store by the class. In xfac it even getting shadow and left uninitialised

    ARRLUDecomp(MatFun<T> fA, MatFun<unsigned int> fCond_, vector<int> I0, vector<int> J0,
                bool leftOrthogonal_ = true, ARRLUParam<T> param = {})
    {
        using std::abs; // force ADL
        RRLUDecomp<T>::leftOrthogonal=leftOrthogonal_;
        RRLUDecomp<T>::Iset=iota(fA.n_rows);
        RRLUDecomp<T>::Jset=iota(fA.n_cols);

        if (param.fullPiv) {
            if (fCond_.n_rows != 0 && fCond_.n_cols != 0) {
                this->calculate(fA.submat(Iset, Jset), param.reltol, param.bondDim, fCond_.submat(Iset, Jset));
            } else {
                this->calculate(fA.submat(Iset, Jset), param.reltol, param.bondDim);
            }
            return;
        }

        int rankMax = std::min(static_cast<int>(fA.n_rows), static_cast<int>(fA.n_cols));
        if (param.bondDim > 0 && param.bondDim < rankMax) rankMax = param.bondDim;
        bool is_low_rank = false;
        do {
            // take new random rows or cols trying to duplicate the rank
            if (!leftOrthogonal_)
                for (auto x : take_n_random(set_diff(static_cast<int>(fA.n_rows), I0), std::max(size_t(1), I0.size())))
                    I0.push_back(x);
            else
                for (auto x : take_n_random(set_diff(static_cast<int>(fA.n_cols), J0), std::max(size_t(1), J0.size())))
                    J0.push_back(x);

            // iterate
            for (int k = 0; k < param.nRookIter; k++)
            {
                Mat A = (k % 2 == static_cast<int>(leftOrthogonal_)) ? fA.submat(I0, Jset) : fA.submat(Iset, J0);
                if (fCond_.n_rows != 0 && fCond_.n_cols != 0) {
                    MatUint cond = (k % 2 == static_cast<int>(leftOrthogonal_)) ? fCond_.submat(I0, Jset) : fCond_.submat(Iset, J0);
                    this->calculate(A, param.reltol, param.bondDim, cond);
                } else {
                    this->calculate(A, param.reltol, param.bondDim);
                }
                auto I1 = this->row_pivots();
                auto J1 = this->col_pivots();
                if (I1.size() < std::min(static_cast<size_t>(A.rows()), static_cast<size_t>(A.cols())))
                    is_low_rank = true;
                if (I0 == I1 && J0 == J1) break;
                I0 = I1;
                J0 = J1;
            }
        } while (I0.size() < static_cast<size_t>(rankMax) && !is_low_rank);

        if (!is_low_rank) this->error = abs((leftOrthogonal_ ? U.diagonal() : L.diagonal())(npivot-1));

        if (L.rows() < static_cast<Eigen::Index>(fA.n_rows)) {
            vector<int> rowsRemaining(Iset.begin() + npivot, Iset.end());
            vector<int> colsPivots(Jset.begin(), Jset.begin() + npivot);
            Mat L2 = fA.submat(rowsRemaining, colsPivots);
            apply_LU_on_rows(L2, this->PivotMatrixTri(), leftOrthogonal_);
            L.conservativeResize(L.rows() + L2.rows(), Eigen::NoChange);
            L.bottomRows(L2.rows()) = L2;
        }
        if (U.cols() < static_cast<Eigen::Index>(fA.n_cols)) {
            vector<int> rowsPivots(Iset.begin(), Iset.begin() + npivot);
            vector<int> colsRemaining(Jset.begin() + npivot, Jset.end());
            Mat U2 = fA.submat(rowsPivots, colsRemaining);
            apply_LU_on_cols(U2, this->PivotMatrixTri(), leftOrthogonal_);
            U.conservativeResize(Eigen::NoChange, U.cols() + U2.cols());
            U.rightCols(U2.cols()) = U2;
        }
    }

    ARRLUDecomp(MatFun<T> fA, vector<int> I0, vector<int> J0,
                bool leftOrthogonal_ = true, ARRLUParam<T> param = {})
        : ARRLUDecomp(fA, {}, I0, J0, leftOrthogonal_, param) {}
};



//--------------------------------------- Cross interpolation or CUR decomposition -----------------------

template<class T>
struct CURDecomp : public ARRLUDecomp<T> {
    using value_type = T;
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using MatUint = Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic>;
    using RealScalar = typename Eigen::NumTraits<T>::Real;

    using ARRLUDecomp<T>::Iset;
    using ARRLUDecomp<T>::Jset;
    using ARRLUDecomp<T>::L;
    using ARRLUDecomp<T>::U;
    using ARRLUDecomp<T>::npivot;
    using ARRLUDecomp<T>::leftOrthogonal;
    using ARRLUDecomp<T>::row_pivots;
    using ARRLUDecomp<T>::col_pivots;

    const Mat C, R;

    CURDecomp(Mat const& A_, bool leftOrthogonal_ = true, RealScalar reltol = RealScalar(1e-14), int rankMax = 0)
        : CURDecomp(A_, MatUint(), leftOrthogonal_, reltol, rankMax) {}

    CURDecomp(MatFun<T> fA, vector<int> const& I0, vector<int> const& J0,
              bool leftOrthogonal_ = true, ARRLUParam<T> param = {})
        : CURDecomp(fA, {}, I0, J0, leftOrthogonal_, param) {}

    CURDecomp(Mat const& A_, MatUint const& fCond_, bool leftOrthogonal_ = true, RealScalar reltol = RealScalar(1e-14), int rankMax = 0)
        : ARRLUDecomp<T>(A_, fCond_, leftOrthogonal_, reltol, rankMax)
        , C([&]() {
            auto piv = this->col_pivots();
            Mat result(A_.rows(), piv.size());
            for (size_t j = 0; j < piv.size(); j++)
                result.col(j) = A_.col(piv[j]);
            return result;
          }())
        , R([&]() {
            auto piv = this->row_pivots();
            Mat result(piv.size(), A_.cols());
            for (size_t i = 0; i < piv.size(); i++)
                result.row(i) = A_.row(piv[i]);
            return result;
          }())
    {}

    CURDecomp(MatFun<T> fA, MatFun<unsigned int> const& fCond_, vector<int> const& I0, vector<int> const& J0, bool leftOrthogonal_ = true, ARRLUParam<T> param = {})
        : ARRLUDecomp<T>(fA, fCond_, I0, J0, leftOrthogonal_, param)
        , C(fA.submat(iota(static_cast<int>(fA.n_rows)), col_pivots()))
        , R(fA.submat(row_pivots(), iota(static_cast<int>(fA.n_cols))))
    {}

    Mat left()  const { return leftOrthogonal ? CU() : C; }

    Mat right() const { return leftOrthogonal ? R  : UR(); }

    Mat CU() const
    {
        Mat cu = Mat::Identity(C.rows(), npivot);
        if (npivot < C.rows()) {
            auto L1 = L.topRows(npivot);
            auto L2 = L.bottomRows(C.rows() - npivot);
            cu.bottomRows(C.rows() - npivot) =
                L1.transpose().template triangularView<Eigen::Upper>()
                    .solve(L2.transpose()).transpose();
        }
        auto invP = inversePermutation(Iset);
        Mat result(invP.size(), cu.cols());
        for (size_t i = 0; i < invP.size(); i++)
            result.row(i) = cu.row(invP[i]);
        return result;
    }

    Mat UR() const
    {
        Mat ur = Mat::Identity(npivot, R.cols());
        if (npivot < R.cols()) {
            auto U1 = U.leftCols(npivot);
            auto U2 = U.rightCols(R.cols() - npivot);
            ur.rightCols(R.cols() - npivot) =
                U1.template triangularView<Eigen::Upper>().solve(U2);
        }
        auto invP = inversePermutation(Jset);
        Mat result(ur.rows(), invP.size());
        for (size_t j = 0; j < invP.size(); j++)
            result.col(j) = ur.col(invP[j]);
        return result;
    }
};

/// Given the pivot matrix P already in LU form, compute UR according to P.
template<class DerivedC, class DerivedP>
auto compute_UR_on_cols(Eigen::MatrixBase<DerivedC> const& Cexpr,
                        Eigen::MatrixBase<DerivedP> const& Pexpr)
{
    using T   = typename DerivedC::Scalar;              // <-- scalar, not Derived
    static_assert(std::is_same_v<T, typename DerivedP::Scalar>,
                  "compute_UR_on_cols: mismatched scalar types");
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    Mat C = Cexpr;
    Mat P = Pexpr;
    apply_LU_on_cols(C, P, false);
    Mat Pu = P;
    Pu.diagonal().setOnes();
    return Pu.template triangularView<Eigen::Upper>().solve(C).eval();
}

/// Given the pivot matrix P already in LU form, update the R rows according to P.
template<class DerivedR, class DerivedP>
auto compute_CU_on_rows(Eigen::MatrixBase<DerivedR> const& Rexpr,
                        Eigen::MatrixBase<DerivedP> const& Pexpr)
{
    using T   = typename DerivedR::Scalar;
    static_assert(std::is_same_v<T, typename DerivedP::Scalar>,
                  "compute_CU_on_rows: mismatched scalar types");
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    Mat R = Rexpr;
    Mat P = Pexpr;
    apply_LU_on_rows(R, P, true);
    Mat Pu = P.transpose();
    Pu.diagonal().setOnes();
    return Pu.template triangularView<Eigen::Upper>().solve(R.transpose()).transpose().eval();
}

template<class T>
struct MatCURFixedTol : public MatDecompFixedTol<CURDecomp<T>> {
    using MatDecompFixedTol<CURDecomp<T>>::MatDecompFixedTol;
};

} // namespace xfac_quad
