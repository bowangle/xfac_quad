#pragma once
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "matrix_interface.h"
#include "adaptive_lu.h"
#include "xfac_quad/index_set.h"

namespace xfac_quad {

using std::vector;

//------------------------------------------------------- stable product by inverse matrix -------

/// compute matrix A*B^-1 in a stable way
/// WARNING not exactly similar to xfac
template<class DerivedA, class DerivedB>
auto mat_AB1(Eigen::MatrixBase<DerivedA> const& A,
             Eigen::MatrixBase<DerivedB> const& B)
{
    using T = typename DerivedA::Scalar;
    static_assert(std::is_same_v<T, typename DerivedB::Scalar>);
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    const Eigen::Index rA = A.rows(), rB = B.rows(), k = A.cols();
    eigen_assert(B.cols() == k && rB == k);   // Qb must be square

    Mat AB(rA + rB, k);
    AB.topRows(rA)    = A;
    AB.bottomRows(rB) = B;

    Eigen::HouseholderQR<Mat> qr(AB);
    Mat Q = qr.householderQ() * Mat::Identity(rA + rB, k);   // thin Q, == qr_econ

    // Qa * Qb^-1  ⟺  solve  Qb^T X^T = Qa^T
    Mat Qa = Q.topRows(rA), Qb = Q.bottomRows(rB);
    return Mat(Qb.transpose().colPivHouseholderQr().solve(Qa.transpose()).transpose());
}

/// compute matrix A^-1*B in a stable way
/// WARNING not exactly similar to xfac
template<class DerivedA, class DerivedB>
auto mat_A1B(Eigen::MatrixBase<DerivedA> const& A,
             Eigen::MatrixBase<DerivedB> const& B)
{
    using T = typename DerivedA::Scalar;
    static_assert(std::is_same_v<T, typename DerivedB::Scalar>);
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    const Eigen::Index rowsA = A.rows(), colsA = A.cols(), colsB = B.cols();
    eigen_assert(B.rows() == rowsA);

    Mat BA(colsB + colsA, rowsA);
    BA.topRows(colsB)    = B.transpose();
    BA.bottomRows(colsA) = A.transpose();

    const Eigen::Index k = std::min(BA.rows(), BA.cols());
    eigen_assert(colsA == k);          // Qa must be square, as inv() requires

    Eigen::HouseholderQR<Mat> qr(BA);
    Mat Q = qr.householderQ() * Mat::Identity(BA.rows(), k);

    Mat Qb = Q.topRows(colsB), Qa = Q.bottomRows(colsA);

    // Qa^-T * Qb^T  ⟺  solve  Qa^T Y = Qb^T
    return Mat(Qa.transpose().colPivHouseholderQr().solve(Qb.transpose()));
}

//-------------------------------------------------------- CrossData class ----------------------------

/// This class store a cross data from a big matrix
/// and can compute the cross interpolation associated to this data:
/// Aapprox=C*P^-1*R
template<class T>
class CrossData {
public:
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    Mat C, R;
    AdaptiveLU<T> lu;

    CrossData(size_t n_rows_, size_t n_cols_): lu(n_rows_,n_cols_) {}

    template<class Container>
    CrossData(Container const& I, Container const& J,
              Mat const& C_, Mat const& R_)
        : C(C_), R(R_), lu(C_.rows(), R.cols())
    {
        for(auto k=0u; k<I.size(); k++) {
            lu.addPivotRow(I[k], R_.row(k));
            lu.addPivotCol(J[k], C_.col(k));
        }
    }

    Mat pivotMat() const {
        Mat result(static_cast<Eigen::Index>(lu.Iset.size()), C.cols());
        for (Eigen::Index i=0; i<result.rows(); i++)
            result.row(i)=C.row(lu.Iset[i]);
        return result;
    }

    const Mat& leftMat() const {
        if (cache_.LD.size()==0) cache_.LD=lu.L*lu.D.asDiagonal();
        return cache_.LD;
    }
    const Mat& rightMat() const { return lu.U; }
    const vector<int>& availRows() const {
        if (cache_.I_avail.empty()) cache_.I_avail=set_diff(static_cast<int>(lu.n_rows), lu.Iset);
        return cache_.I_avail;
    }
    const vector<int>& availCols() const {
        if (cache_.J_avail.empty()) cache_.J_avail=set_diff(static_cast<int>(lu.n_cols), lu.Jset);
        return cache_.J_avail;
    }

    size_t rank() const { return lu.npivot(); }
    T firstPivotValue() const { return C.size()==0 ? T(1) : C(lu.Iset.at(0),0); }

    ///@{compute the cross formula at given rows/columns
    T eval(int i,int j) const {
        if (C.size()==0) return T(0);
        return (leftMat().row(i) * rightMat().col(j)).value();
    }

    vector<T> eval(vector<pair<int,int>> const& ids) const
    {
        vector<T> values(ids.size());
        if (C.size()==0) return values;
        for(auto c=0u;c<values.size();c++) {
            auto [i,j]=ids[c];
            values[c]=eval(i,j);
        }
        return values;
    }

    vector<T> row(int i) const {
        auto rv=leftMat().row(i)*rightMat();
        return vector<T>(rv.data(), rv.data()+rv.size());
    }

    vector<T> col(int j) const {
        auto cv=leftMat()*rightMat().col(j);
        return vector<T>(cv.data(), cv.data()+cv.size());
    }

    vector<T> submat(vector<int> const& rows, vector<int> const& cols) const                   
    {                
        using namespace Eigen::indexing;        // works on 3.4 and on master

        if (C.size() == 0)
            return vector<T>(rows.size() * cols.size(), T(0));
        Eigen::Map<const Eigen::VectorXi> I0(rows.data(),
        static_cast<Eigen::Index>(rows.size()));
        Eigen::Map<const Eigen::VectorXi> J0(cols.data(),
        static_cast<Eigen::Index>(cols.size()));
        Mat M = leftMat()(I0, all) * rightMat()(all, J0);
        // Eigen default is column-major → matches Armadillo's conv_to ordering
        return vector<T>(M.data(), M.data() + M.size());
    }

    Mat mat() const { return leftMat()*rightMat(); }

    /// return the update after adding last pivot (rank-1 update)
    Mat matDiff() const { return lu.L.rightCols(1)*lu.U.bottomRows(1)*lu.D.tail(1)(0); }

    /// Update the cross data with a new pivot at row i, column j of the matrix A.
    void addPivot(int i, int j, IMatrix<T> const& A)
    {
        addPivotRow(i,A);
        addPivotCol(j,A);
    }

    /// Update the cross by adding the row i of the matrix A. This is for developers only.
    void addPivotRow(int i, IMatrix<T> const& A)
    {
        using RowT = Eigen::Matrix<T,1,Eigen::Dynamic>;
        RowT row(static_cast<Eigen::Index>(A.n_cols));
        for(size_t ji=0; ji<lu.Jset.size(); ji++)
            row[lu.Jset[ji]]=C(i,static_cast<Eigen::Index>(ji));
        auto Ri=A.submat({i}, availCols());
        for(size_t jj=0; jj<availCols().size(); jj++)
            row[availCols()[jj]]=Ri[jj];
        auto k = lu.npivot();
        R.conservativeResize(k+1, row.size());
        R.row(static_cast<Eigen::Index>(k)) = row;
        lu.addPivotRow(i, row);
        cache_={};
    }

    void addPivotCol(int j, IMatrix<T> const& A)
    {
        using ColT = Eigen::Matrix<T,Eigen::Dynamic,1>;
        ColT col(static_cast<Eigen::Index>(A.n_rows));
        for(size_t ii=0; ii<lu.Iset.size(); ii++)
            col[lu.Iset[ii]]=R(static_cast<Eigen::Index>(ii),j);
        auto Cj=A.submat(availRows(),{j});
        for(size_t ii=0; ii<availRows().size(); ii++)
            col[availRows()[ii]]=Cj[ii];
        auto k = lu.npivot();
        C.conservativeResize(col.size(), k+1);
        C.col(static_cast<Eigen::Index>(k)) = col;
        lu.addPivotCol(j, col);
        cache_={};
    }

    void setRows(Mat const& C_, vector<int> const& P)
    {
        C=C_;
        lu.setRows(C_,P);
        cache_={};
    }

    void setCols(Mat const& R_, vector<int> const& Q)
    {
        R=R_;
        lu.setCols(R_,Q);
        cache_={};
    }

private:
    struct Cache {
        vector<int> I_avail, J_avail;
        Mat LD;
    };
    mutable Cache cache_;
};

} // namespace xfac_quad
