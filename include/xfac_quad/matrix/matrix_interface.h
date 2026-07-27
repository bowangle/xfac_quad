#pragma once
#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <map>
#include <memory>
#include <functional>
#include "xfac_quad/index_set.h"

namespace xfac_quad {

using std::pair;
using std::function;
using std::vector;
using std::map;


// ===============================================

// WARNING use of #pragma omp parallel for. 

// ===============================================

//---------------------------------------------------- IMatrix ---------------------------

template<class T,class... Indices> class IMatrix;

/// Interface for a matrix that we need in our implementation.
template<class T>
class IMatrix<T>
{
public:
    size_t n_rows, n_cols;
    IMatrix()=default;
    IMatrix(size_t nr, size_t nc): n_rows(nr), n_cols(nc) {}
    virtual ~IMatrix() = default;

    /// return the values ordered by columns
    virtual vector<T> submat(vector<int> const& rows, vector<int> const& cols) const = 0;

    /// return the values of the matrix at the given list of pairs (row,column).
    virtual vector<T> eval(vector<pair<int,int>> const& ids) const = 0;

    /// Useful in case of using cache
    virtual void forgetRow(int) const {}

    /// Useful in case of using cache
    virtual void forgetCol(int) const {}
};

template<class T, class Index>
class IMatrix<T,Index>: virtual public IMatrix<T> {
public:
    function<T(Index,Index)> A;
    IndexSet<Index> Iset, Jset;
    IMatrix()=default;
    IMatrix(function<T(Index,Index)> A_, vector<Index> const& Iset_, vector<Index> const& Jset_)
        : IMatrix<T>{Iset_.size(), Jset_.size()}
        , A{A_}, Iset{Iset_}, Jset{Jset_}
    {}

    function<T(int,int)> matFun() const
    {
        return [A=A, I=Iset.from_int(), J=Jset.from_int()](int i, int j) {
            return A(I.at(i), J.at(j));
        };
    }

    /// returns the values A(x, Jset[J])
    vector<T> evalHyb(Index x, vector<int> const& J) const
    {
        vector<T> values(J.size());
        #pragma omp parallel for
        for(auto j=0u; j<J.size(); j++)
            values[j]=A(x, Jset[J[j]]);
        return values;
    }

    /// returns the values A(Iset[I], y)
    vector<T> evalHyb(vector<int> const& I, Index y) const
    {
        vector<T> values(I.size());
        #pragma omp parallel for
        for(auto i=0u; i<I.size(); i++)
            values[i]=A(Iset[I[i]], y);
        return values;
    }

    /// and return the permutation that transforms the rows:  i -> P[i]
    virtual vector<int> setRows(vector<Index> const& i_set)=0;
    /// and return the permutation that transforms the columns:  j -> Q[j]
    virtual vector<int> setCols(vector<Index> const& j_set)=0;
};

//---------------------------------------------------- Dense matrix ---------------------------

template<class T,class... Indices> class MatDense;

/// A dense matrix implementing an IMatrix
template<class T>
class MatDense<T>: virtual public IMatrix<T> {
public:
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    MatDense()=default;
    MatDense(Mat A): IMatrix<T>(A.rows(),A.cols()), data(std::move(A)) {}

    MatDense(function<T(int,int)> f_, int nr, int nc)
        : IMatrix<T>(nr,nc), data(nr,nc)
    {
        #pragma omp parallel for collapse(2)
        for(int j=0;j<nc;j++)
            for(int i=0;i<nr;i++)
                data(i,j)=f_(i,j);
    }

    const T& operator()(int i,int j) const {return data(i,j);}

    vector<T> submat(vector<int> const& rows, vector<int> const& cols) const override
   {
       Eigen::Map<const Eigen::VectorXi> I0(rows.data(), static_cast<Eigen::Index>(rows.size()));
       Eigen::Map<const Eigen::VectorXi> J0(cols.data(), static_cast<Eigen::Index>(cols.size()));
       Mat sm = data(I0, J0);
       return vector<T>(sm.data(), sm.data() + sm.size());
   }

    vector<T> eval(vector<pair<int,int>> const& ids) const override
    {
        vector<T> values;
        values.reserve(ids.size());
        for(auto [i,j]:ids) values.push_back(data(i,j));
        return values;
    }

    void setRows(int nr, vector<int> const& P, function<T(int,int)> fnew)
    {
        Mat data2(nr, static_cast<Eigen::Index>(this->n_cols));
        for(size_t pi=0; pi<P.size(); pi++)
            data2.row(P[pi])=data.row(static_cast<Eigen::Index>(pi));
        vector<int> Pc=set_diff(nr,P);
        #pragma omp parallel for collapse(2)
        for(int j=0; j<static_cast<int>(this->n_cols); j++)
            for(int i:Pc) data2(i,j)=fnew(i,j);
        data=data2;
        this->n_rows=nr;
    }

    void setCols(int nc, vector<int> const& Q, function<T(int,int)> fnew)
    {
        Mat data2(static_cast<Eigen::Index>(this->n_rows), nc);
        for(size_t qj=0; qj<Q.size(); qj++)
            data2.col(Q[qj])=data.col(static_cast<Eigen::Index>(qj));
        vector<int> Qc=set_diff(nc,Q);
        #pragma omp parallel for collapse(2)
        for(int j:Qc)
            for(int i=0; i<static_cast<int>(this->n_rows); i++)
                data2(i,j)=fnew(i,j);
        data=data2;
        this->n_cols=nc;
    }

private:
    Mat data;
};

/// similar to MatDense but the Index can be anything: i.e double, or MultiIndex
/// whenever the possible values are provided in Iset, Jset
template<class T, class Index>
class MatDense<T,Index>: public IMatrix<T,Index>, public MatDense<T> {
public:
    MatDense()=default;
    MatDense(function<T(Index,Index)> A_, vector<Index> const& i_set, vector<Index> const& j_set)
        : IMatrix<T>(i_set.size(), j_set.size())
        , IMatrix<T,Index>(A_, i_set, j_set)
        , MatDense<T>(IMatrix<T,Index>::matFun(), static_cast<int>(IMatrix<T>::n_rows), static_cast<int>(IMatrix<T>::n_cols)) {}

    vector<int> setRows(vector<Index> const& i_set) override
    {
        IndexSet<Index> I(i_set);
        vector<int> pos=I.pos(this->Iset.from_int());
        IMatrix<T,Index>::Iset=I;
        MatDense<T>::setRows(static_cast<int>(i_set.size()), pos, IMatrix<T,Index>::matFun());
        return pos;
    }

    vector<int> setCols(vector<Index> const& j_set) override
    {
        IndexSet<Index> J(j_set);
        vector<int> pos=J.pos(this->Jset.from_int());
        IMatrix<T,Index>::Jset=J;
        MatDense<T>::setCols(static_cast<int>(j_set.size()), pos, IMatrix<T,Index>::matFun());
        return pos;
    }
};

//---------------------------------------------------- Lazy matrix ---------------------------

template<class T,class... Indices> class MatLazy;

/// A lazy matrix is defined through a function f:(int,int)->T
/// It only computes/stores the requested values.
template<class T>
class MatLazy<T>: virtual public IMatrix<T> {
public:
    function<T(int,int)> f;

    MatLazy()=default;
    MatLazy(function<T(int,int)> f_, int nr, int nc)
        : IMatrix<T>(nr,nc), f(f_) {}

    vector<T> eval(vector<pair<int,int>> const& ids) const override
    {
        vector<T> values(ids.size());
        vector<size_t> pos_eval;
        for(size_t c=0; c<ids.size(); c++)
            if (auto it=data.find(ids[c]); it!=data.end())
                values[c]=it->second;
            else pos_eval.push_back(c);
        #pragma omp parallel for
        for(auto c:pos_eval) values[c]=f(ids[c].first, ids[c].second);
        for(auto c:pos_eval) data[ids[c]]=values[c];
        return values;
    }

    vector<T> submat(vector<int> const& rows, vector<int> const& cols) const override
    {
        vector<pair<int,int>> ids;
        ids.reserve(rows.size()*cols.size());
        for(int j:cols)
            for(int i:rows)
                ids.emplace_back(i,j);
        return eval(ids);
    }

    void forgetRow(int i0) const override
    {
        for(size_t j=0; j < this->n_cols; j++)
            data.erase({i0,static_cast<int>(j)});
    }

    void forgetCol(int j0) const override
    {
        for(size_t i=0; i < this->n_rows; i++)
            data.erase({static_cast<int>(i),j0});
    }

    /// Increase the rows of the matrix to n_rows, while reordering its old rows according to P: row i -> row P[i].
    /// The new matrix function is required
    void setRows(int nr, vector<int> const& P, function<T(int,int)> fnew)
    {
        map<pair<int,int>,T> data2;
        for(auto [id,value]:data) { 
            auto [i,j]=id;
            data2[{P[i],j}]=value;
        }
        data=data2;
        this->n_rows=nr;
        f=fnew;
    }

    /// Increase the cols of the matrix to n_cols, while reordering its old cols according to Q: col j -> col Q[j].
    /// The new matrix function is required
    void setCols(int nc, vector<int> const& Q, function<T(int,int)> fnew)
    {
        map<pair<int,int>,T> data2;
        for(auto [id,value]:data) {
            auto [i,j]=id;
            data2[{i,Q[j]}]=value;
        }
        data=data2;
        this->n_cols=nc;
        f=fnew;
    }

private:
    mutable map<pair<int,int>,T> data;
};

/// similar to MatLazy but the Index can be anything: i.e double, or MultiIndex
/// whenever the possible values are provided in Iset, Jset
template<class T, class Index>
class MatLazy<T,Index>: public IMatrix<T,Index>, public MatLazy<T> {
public:
    MatLazy()=default;

    MatLazy(function<T(Index,Index)> A_, vector<Index> const& Iset_, vector<Index> const& Jset_)
        : IMatrix<T>(Iset_.size(), Jset_.size())
        , IMatrix<T,Index>(A_, Iset_, Jset_)
        , MatLazy<T>(IMatrix<T,Index>::matFun(), static_cast<int>(IMatrix<T>::n_rows), static_cast<int>(IMatrix<T>::n_cols)) {}

    vector<int> setRows(vector<Index> const& i_set) override
    {
        IndexSet<Index> I(i_set);
        vector<int> pos=I.pos(this->Iset.from_int());
        IMatrix<T,Index>::Iset=I;
        MatLazy<T>::setRows(static_cast<int>(i_set.size()), pos, IMatrix<T,Index>::matFun());
        return pos;
    }

    vector<int> setCols(vector<Index> const& j_set) override
    {
        IndexSet<Index> J(j_set);
        vector<int> pos=J.pos(this->Jset.from_int());
        IMatrix<T,Index>::Jset=J;
        MatLazy<T>::setCols(static_cast<int>(j_set.size()), pos, IMatrix<T,Index>::matFun());
        return pos;
    }
};

//--------------------------------------------------------- Factory functions for IMatrix ------------------

///@{ Factory functions for the matrix
template<class T>
std::unique_ptr<IMatrix<T>> make_IMatrix(function<T(int,int)> f, int n_rows, int n_cols, bool is_full)
{
    if (is_full) return std::make_unique<MatDense<T>>(f,n_rows,n_cols);
    else return std::make_unique<MatLazy<T>>(f,n_rows,n_cols);
}

template<class T, class Index>
std::unique_ptr<IMatrix<T,Index>> make_IMatrix(function<T(Index,Index)> f, vector<Index> const& Iset, vector<Index> const& Jset, bool is_full)
{
    if (is_full) return std::make_unique<MatDense<T,Index>>(f,Iset,Jset);
    else return std::make_unique<MatLazy<T,Index>>(f,Iset,Jset);
}

///@}

} // namespace xfac_quad
