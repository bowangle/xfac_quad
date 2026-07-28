#pragma once
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <functional>
#include <fstream>
#include <stdexcept>
#include "xfac_quad/grid.h"
#include "xfac_quad/index_set.h"
#include "xfac_quad/cubemat_helper.h"
#include "xfac_quad/matrix/mat_decomp.h"

namespace xfac_quad {

using std::vector;
using std::function;
using std::array;

/// stores a tensor train, i.e., a list of cubes.
template<class T>
struct TensorTrain {
    using Real = typename Eigen::NumTraits<T>::Real;
    vector<Tensor3D<T>> M;  ///< list of 3-leg tensors

    TensorTrain()=default;
    TensorTrain(size_t len) : M(len) {}

    /// evaluate the tensor train at a given multi index.
    T eval(vector<int> const& id) const
    {
        if (id.size()!=M.size()) throw std::invalid_argument("TensorTrain::() id.size()!=size()");
        using Mat=Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;
        Mat prod=Mat::Identity(1,1);
        for(auto k=0u; k<M.size(); k++)
            prod=prod* M[k].col_as_mat_const(id[k]);
        return prod(0,0);
    }

    /// evaluate the tensor train at a given multi index. Same as eval()
    T operator()(vector<int> const& id) const { return eval(id); }

    /// compute the weighted sum of the tensor train
    T sum(const vector<vector<Real>>& weight) const;

    /// compute the plane sum of the tensor train
    T sum1() const
    {
        vector<vector<Real>> weight;
        for(auto const& Mi : M)
            weight.push_back(vector<Real>(Mi.n_cols,Real(1)));
        return sum(weight);
    }

    /// compute the overlap with another tensor train
    T overlap(const TensorTrain<T>& tt) const
    {
        if (M.empty() || tt.M.empty()) return 0;
        if (M.size() != tt.M.size())
            throw std::invalid_argument("tt1.overlap(tt2) with different lengths");
        using Mat=Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;
        Mat L=Mat::Identity(1,1);
        for(auto p=0u; p<M.size(); p++) {  // L(A,B) = L(a,b)*N(a,s,A)*M(b,s,B)
            Mat LN=L.transpose()*cube_as_matrix1(tt.M.at(p));
            auto LNm=Eigen::Map<Mat>(LN.data(), LN.rows()*static_cast<Eigen::Index>(tt.M[p].n_cols), tt.M[p].n_slices);
            L=LNm.transpose()*cube_as_matrix2(M[p]);
        }
        return L(0,0);
    }

    T norm2() const { return overlap(*this); }

    void compressSVD(Real reltol=Real(1e-12), int maxBondDim=0) { right_to_left(MatQR<T>{}); sweep(MatSVDFixedTol<T>{reltol,maxBondDim}); }
    void compressLU(Real reltol=Real(1e-12), int maxBondDim=0)  { right_to_left(MatRRLUFixedTol<T>{}); sweep(MatRRLUFixedTol<T>{reltol,maxBondDim}); }
    void compressCI(Real reltol=Real(1e-12), int maxBondDim=0)  { right_to_left(MatCURFixedTol<T>{}); sweep(MatCURFixedTol<T>{reltol,maxBondDim}); }

    /// computes the max error |f-tt| if the tensor is smaller than max_n_Eval.
    typename Eigen::NumTraits<T>::Real trueError(function<T(vector<int>)> f, size_t max_n_eval=1e6) const
    {
        size_t prod=1;
        vector<int> dims(M.size());
        for(size_t k=0; k<dims.size(); k++) {
            dims[k]=static_cast<int>(M[k].n_cols);
            prod*=dims[k];
            if (prod>max_n_eval) return Real(-1);
        }
        Real e=Real(0);
        for(size_t i=0;i<prod;i++) {
            auto idv=to_tensorIndex(i,dims);
            T diff = eval(idv)-f({idv.begin(),idv.end()});
            using std::abs;
            Real error=abs(diff);
            if (error>e) e=error;
        }
        return e;
    }

    void save(std::ostream &out) const
    {
        save_Tensor3D_to_arma(out, M);
    }
    void save(std::string fileName) const { std::ofstream out(fileName); save(out); }

    static TensorTrain<T> load(std::ifstream& in)
    {
        return TensorTrain<T>(load_vector_tensor<T>(in));
    }
    static TensorTrain<T> load(std::string fileName)
    {
        std::ifstream in(fileName);
        if (in.fail()) throw std::runtime_error("TensorTrain::load fails to load file: "+fileName);
        return load(in);
    }

    /// for developers
    void sweep(function<array<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>,2>(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>,bool)> mat_decomp)
    {
        left_to_right(mat_decomp);
        right_to_left(mat_decomp);
    }

    /// for developers
    void left_to_right(function<array<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>,2>(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>,bool)> mat_decomp)
    {
        for(auto i=0u; i+1<M.size(); i++) {
            auto ab=mat_decomp(cube_as_matrix2(M[i]), true);
            auto &M1=ab[0];
            auto M2=(ab[1]*cube_as_matrix1(M[i+1])).eval();
            M[i]=Tensor3D<T>(M[i].n_rows, M[i].n_cols, M1.cols());
            std::copy(M1.data(), M1.data()+M1.size(), M[i].data.begin());
            M[i+1]=Tensor3D<T>(M2.rows(), M[i+1].n_cols, M[i+1].n_slices);
            std::copy(M2.data(), M2.data()+M2.size(), M[i+1].data.begin());
        }
    }

    void right_to_left(function<array<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>,2>(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>,bool)> mat_decomp)
    {
        for(int i=static_cast<int>(M.size())-1; i>0; i--) {
            auto ab=mat_decomp(cube_as_matrix1(M[i]), false);
            auto M1=(cube_as_matrix2(M[i-1])*ab[0]).eval();
            auto &M2=ab[1];
            M[i-1]=Tensor3D<T>(M[i-1].n_rows, M[i-1].n_cols, M1.cols());
            std::copy(M1.data(), M1.data()+M1.size(), M[i-1].data.begin());
            M[i]=Tensor3D<T>(M2.rows(), M[i].n_cols, M[i].n_slices);
            std::copy(M2.data(), M2.data()+M2.size(), M[i].data.begin());
        }
    }
};

template<class T>
TensorTrain<T> operator+(TensorTrain<T> const& tt1, TensorTrain<T> const& tt2)
{
    if (tt1.M.empty()) return tt2;
    if (tt2.M.empty()) return tt1;
    if (tt1.M.size() != tt2.M.size())
        throw std::invalid_argument("tt1+tt2 with different lengths");
    auto kron_add=[&](Tensor3D<T> const& A, Tensor3D<T> const& B, int i)
    {
        if (A.n_cols != B.n_cols)
            throw std::invalid_argument("kron_add(A,B) with A.n_cols != B.n_cols");
        Tensor3D<T> C;
        if (i==0) {
            C=Tensor3D<T>(A.n_rows, A.n_cols, A.n_slices+B.n_slices);
            std::fill(C.data.begin(), C.data.end(), T(0));
            for(Eigen::Index s=0; s<A.n_slices; s++)
                for(Eigen::Index r=0; r<A.n_rows; r++)
                    for(Eigen::Index c=0; c<A.n_cols; c++)
                        C(r,c,s)=A(r,c,s);
            for(Eigen::Index s=0; s<B.n_slices; s++)
                for(Eigen::Index r=0; r<B.n_rows; r++)
                    for(Eigen::Index c=0; c<B.n_cols; c++)
                        C(r,c,A.n_slices+s)=B(r,c,s);
        }
        else if (i==static_cast<int>(tt1.M.size())-1) {
            C=Tensor3D<T>(A.n_rows+B.n_rows, A.n_cols, B.n_slices);
            std::fill(C.data.begin(), C.data.end(), T(0));
            for(Eigen::Index s=0; s<A.n_slices; s++)
                for(Eigen::Index r=0; r<A.n_rows; r++)
                    for(Eigen::Index c=0; c<A.n_cols; c++)
                        C(r,c,s)=A(r,c,s);
            for(Eigen::Index s=0; s<B.n_slices; s++)
                for(Eigen::Index r=0; r<B.n_rows; r++)
                    for(Eigen::Index c=0; c<B.n_cols; c++)
                        C(A.n_rows+r,c,s)=B(r,c,s);
        }
        else {
            C=Tensor3D<T>(A.n_rows+B.n_rows, A.n_cols, A.n_slices+B.n_slices);
            std::fill(C.data.begin(), C.data.end(), T(0));
            for(Eigen::Index s=0; s<A.n_slices; s++)
                for(Eigen::Index r=0; r<A.n_rows; r++)
                    for(Eigen::Index c=0; c<A.n_cols; c++)
                        C(r,c,s)=A(r,c,s);
            for(Eigen::Index s=0; s<B.n_slices; s++)
                for(Eigen::Index r=0; r<B.n_rows; r++)
                    for(Eigen::Index c=0; c<B.n_cols; c++)
                        C(A.n_rows+r,c,A.n_slices+s)=B(r,c,s);
        }
        return C;
    };

    TensorTrain<T> tt;
    for(auto i=0u; i<tt1.M.size(); i++) {
        tt.M.push_back( kron_add(tt1.M[i], tt2.M[i], static_cast<int>(i)) );
    }
    return tt;
}

/// compute the sum of many tensor trains, while compressing along the tree.
template<class T>
TensorTrain<T> sum(vector<TensorTrain<T>> v, typename Eigen::NumTraits<T>::Real reltol=typename Eigen::NumTraits<T>::Real(1e-12), int maxBondDim=0, bool use_svd=false)
{
    if (v.empty()) return {};
    int step=1;
    while (static_cast<size_t>(v.size())>static_cast<size_t>(step))
    {
        #pragma omp parallel for
        for(size_t i=0; i<v.size(); i+=static_cast<size_t>(2*step))
            if (i+static_cast<size_t>(step)<v.size()) {
                v[i]=v[i]+v[i+static_cast<size_t>(step)];
                if (v[i].M.empty()) continue;
                if (use_svd) v[i].compressSVD(reltol, maxBondDim);
                else v[i].compressCI(reltol, maxBondDim);
            }
        step+=step;
    }
    return v.at(0);
}

/// stores a continuous tensor train
template<class T, class Index>
struct CTensorTrain {
    vector<std::function<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>(Index)>> M;

    T eval(vector<Index> const& xs) const
    {
        if (xs.size()!=M.size()) throw std::invalid_argument("CTensorTrain::() xs.size()!=size()");
        using Mat=Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;
        Mat prod=Mat::Identity(1,1);
        for(auto k=0u; k<M.size(); k++)
            prod=prod* M[k](xs[k]);
        return prod(0,0);
    }
};

/// Manage the weighted sum of a tensor train.
///             w0     w1     w2
///             |      |      |
///             M0 --- M1 --- M2
template<class T>
class TT_sum {
    using Real = typename Eigen::NumTraits<T>::Real;
    vector<vector<Real>> w; ///< the weights at each site
public:
    using RowT=Eigen::Matrix<T,1,Eigen::Dynamic>;   
    using ColT=Eigen::Matrix<T,Eigen::Dynamic,1>;   

    vector<RowT> L; ///< accumulated left product up to before a given site
    vector<ColT> R; ///< accumulated right product up to after a given site

    TT_sum(){}
    TT_sum(TensorTrain<T> const& tt, vector<vector<Real>> const& weight)
        : w(weight), L(tt.M.size()), R(tt.M.size())
    {
        R.back()=ColT::Ones(1);
        L.front()=RowT::Ones(1);
        for(auto s=0u; s<tt.M.size()-1; s++)
            updateSite(static_cast<int>(s), tt.M[s], true);
        for(int s=static_cast<int>(tt.M.size())-1; s>0; s--)
            updateSite(s, tt.M[s], false);
    }

    TT_sum(TensorTrain<T> const& tt, vector<Real> const& weight)
        : TT_sum(tt, vector<vector<Real>>(tt.M.size(), weight)) {}

    T value() const { return (L[1] * R[0]).value(); }

    /// update the left or right product given the new cube M at site s.
    void updateSite(int s, Tensor3D<T> const& M, bool updateLeft)
    {
        if (!updateLeft && s>0) {   // R[s-1]=M(i,j,k)*R[s](k)*w[s](j)
            auto MRv=cube_as_matrix2(M)*R[s];
            auto MRvEval=MRv.eval();
            Eigen::Map<const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>> MR(MRvEval.data(), M.n_rows, M.n_cols);
            Eigen::Map<const Eigen::Matrix<Real,Eigen::Dynamic,1>> ws(
                const_cast<Real*>(w[s].data()), static_cast<Eigen::Index>(w[s].size()));
            R[s-1]=MR*ws.template cast<T>();
        }
        if (updateLeft && s<static_cast<int>(L.size())-1) { // L[s+1]=w[s](j)*L[s](i)*M(i,j,k)
            auto LMv=L[s]*cube_as_matrix1(M);
            auto LMvEval=LMv.eval();
            Eigen::Map<const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>> LM(LMvEval.data(), M.n_cols, M.n_slices);
            Eigen::Map<const Eigen::Matrix<Real,1,Eigen::Dynamic>> ws(
                const_cast<Real*>(w[s].data()), static_cast<Eigen::Index>(w[s].size()));
            L[s+1]=ws.template cast<T>()*LM;
        }
    }
};

template<class T>
T TensorTrain<T>::sum(const vector<vector<Real>>& weight) const { return TT_sum<T>(*this,weight).value(); }

} // namespace xfac_quad
