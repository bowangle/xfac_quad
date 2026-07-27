#pragma once
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "xfac_quad/index_set.h"
#include "xfac_quad/matrix/mat_decomp.h"

namespace xfac_quad {

using std::vector;
using std::function;

///< To store the tensor function f:(a1,a2,...,an)->T
template<class T>
struct TensorFunction {
    function<T(vector<int>)> f;
    bool useCache=false;
    bool concatenate=true; // use concatenation of the function argument (tensor train) or not (tensor tree)

    TensorFunction() = default;
    TensorFunction(function<T(vector<int>)> f_, bool useCache_=false, bool concatenate_=true) : f(f_), useCache(useCache_), concatenate(concatenate_) {}

    T operator()(MultiIndex const& id) const { cEval+=1; return f({id.begin(),id.end()}); }

    /// add two multiindices I and J
    MultiIndex addIJ(MultiIndex const& I, MultiIndex const& J) const {
        return concatenate ? I+J : add(I,J);
    }

    using Mat = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

    Mat evalCache(vector<MultiIndex> const& I, vector<MultiIndex> const& J) const
   {
       Mat values(I.size(), J.size());
       vector<std::tuple<Eigen::Index, Eigen::Index, decltype(dat.begin())>> pos_eval;
       for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(I.size()); ++i)
           for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(J.size()); ++j) {
               auto [it, isNew] = dat.try_emplace(addIJ(I[static_cast<size_t>(i)], J[static_cast<size_t>(j)]));
               if (!isNew)
                   values(i, j) = it->second;
               else
                   pos_eval.push_back({i, j, it});
           }
   #pragma omp parallel for
       for (auto const& [i, j, it] : pos_eval)
           values(i, j) = it->second = f({it->first.begin(), it->first.end()});
       return values;
   }

    Mat eval(vector<MultiIndex> const& I, vector<MultiIndex> const& J) const
    {
        if (useCache) return evalCache(I,J);
        Mat values(I.size(), J.size());
        #pragma omp parallel for collapse(2)
        for(auto i=0u; i<I.size(); i++)
            for(auto j=0u; j<J.size(); j++) {
                MultiIndex ij=addIJ(I[i],J[j]);
                values(i,j)=f({ij.begin(), ij.end()});
            }
        cEval += values.size();
        return values;
    }

    MatFun<T> matfun(vector<MultiIndex> const& I, vector<MultiIndex> const& J) const
    {
        auto submat=[this,I,J](vector<int> const& I0, vector<int> const& J0) {
            vector<MultiIndex> Is(I0.size()), Js(J0.size());
            for(auto i=0u; i<Is.size(); i++) Is[i]=I[I0[i]];
            for(auto j=0u; j<Js.size(); j++) Js[j]=J[J0[j]];
            return eval(Is,Js);
        };
        return {I.size(), J.size(), submat};
    }

    void clearCache() { cEval+=dat.size(); dat.clear(); }
    size_t nEval() const { return dat.size()+cEval; }

private:
    mutable size_t cEval=0;
    mutable std::unordered_map<MultiIndex,T,MultiIndexHash> dat;
};

} // namespace xfac_quad
