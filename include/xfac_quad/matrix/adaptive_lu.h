#pragma once
#include <Eigen/Dense>
#include <vector>
#include "xfac_quad/index_set.h"

namespace xfac_quad {

using std::vector;

template<class T>
struct AdaptiveLU {
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Col = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Row = Eigen::Matrix<T, 1, Eigen::Dynamic>;

    size_t n_rows, n_cols;
    vector<int> Iset, Jset;

    Mat L, U;
    Col D;

    AdaptiveLU(size_t n_rows_, size_t n_cols_): n_rows(n_rows_), n_cols(n_cols_){}

    size_t npivot() const { return static_cast<size_t>(D.size()); }

    /// Gaussian elimination with pivoting on rows
    void addPivotRow(int i, Row const& row)
    {
        Iset.push_back(i);
        Eigen::Index k = static_cast<Eigen::Index>(npivot());
        // add a new row
        U.conservativeResize(k+1, row.size());
        U.row(k) = row;
        // gaussian pivoting on row
        for(Eigen::Index l=0; l<k; l++)
            U.row(k) -= U.row(l)
                * (L(Iset[k],l) * D(l));
    }

    /// Gaussian elimination with pivoting on columns
    void addPivotCol(int j, Col const& col)
    {
        Jset.push_back(j);
        Eigen::Index k = static_cast<Eigen::Index>(npivot());
        // add a new column
        L.conservativeResize(col.size(), k+1);
        L.col(k) = col;
        for(Eigen::Index l=0u;l<k;l++)
            L.col(k) -= L.col(l)
                * (U(l,Jset[k]) * D(l));
        D.conservativeResize(k+1);
        D(k)=T(1)/L(Iset[k],k);
    }

    /// Increase the rows of the matrix according to C_,
    /// while reordering its old rows according to P: row i -> row P[i]
    void setRows(Mat const& C, vector<int> const& P)
    {
        n_rows=static_cast<size_t>(C.rows());
        for(auto& i : Iset) i=P.at(i);
        auto r=static_cast<Eigen::Index>(L.cols());
        L=[&](){
            Mat newL(C.rows(), r);
            for(Eigen::Index col=0; col<r; col++)
                for(size_t p_i=0; p_i<P.size(); p_i++)
                    newL(P[p_i],col)=L(static_cast<Eigen::Index>(p_i),col);
            return newL;
        }();
        auto Pc=set_diff(static_cast<int>(C.rows()),P);
        if (!Pc.empty()) {
            Eigen::Map<Eigen::VectorXi> PcMap(Pc.data(), static_cast<Eigen::Index>(Pc.size()));
            for(Eigen::Index k=0; k<r; k++) {
                L(PcMap, k) = C(PcMap, k);
                for(Eigen::Index l=0; l<k; l++)
                    L(PcMap, k) -= L(PcMap, l) * (U(l,Jset[static_cast<size_t>(k)]) * D(l));
            }
        }
    }

    /// Increase the cols of the matrix  according to R_,
    /// while reordering its old cols according to Q: col j -> col Q[j].
    void setCols(Mat const& R, vector<int> const& Q)
    {
        n_cols=static_cast<size_t>(R.cols());
        for(auto& j : Jset) j=Q.at(j);
        auto r=static_cast<Eigen::Index>(U.rows());
        U=[&](){
            Mat newU(r, R.cols());
            for(Eigen::Index row=0; row<r; row++)
                for(size_t qj=0; qj<Q.size(); qj++)
                    newU(row,Q[qj])=U(row,static_cast<Eigen::Index>(qj));
            return newU;
        }();
        auto Qc=set_diff(static_cast<int>(R.cols()),Q);
        if (!Qc.empty()) {
            Eigen::Map<Eigen::VectorXi> QcMap(Qc.data(), static_cast<Eigen::Index>(Qc.size()));
            for(Eigen::Index k=0; k<r; k++) {
                U(k, QcMap) = R(k, QcMap);
                for(Eigen::Index l=0; l<k; l++)
                    U(k, QcMap) -= U(l, QcMap) * (L(Iset[static_cast<size_t>(k)],l) * D(l));
            }
        }
    }
};

} // namespace xfac_quad
