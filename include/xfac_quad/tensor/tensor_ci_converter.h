#pragma once
#include "xfac_quad/tensor/tensor_ci.h"
#include "xfac_quad/tensor/tensor_ci_2.h"

namespace xfac_quad {

template<class T>
TensorCI1Param<T> to_tci1Param(TensorCI2Param<T> const& x)
{
    TensorCI1Param<T> y;
    y.reltol=x.reltol;
    y.pivot1=x.pivot1;
    y.fullPiv=x.fullPiv;
    y.nRookIter=x.nRookIter;
    y.weight=x.weight;
    y.cond=x.cond;
    y.useCachedFunction=x.useCachedFunction;
    return y;
}

template<class T>
TensorCI2Param<T> to_tci2Param(TensorCI1Param<T> const& x)
{
    TensorCI2Param<T> y;
    y.reltol=x.reltol;
    y.pivot1=x.pivot1;
    y.fullPiv=x.fullPiv;
    y.nRookIter=x.nRookIter;
    y.weight=x.weight;
    y.cond=x.cond;
    y.useCachedFunction=x.useCachedFunction;
    return y;
}

namespace impl {
template<class Tci>
vector<int> readDims(Tci const& ci)
{
    vector<int> dims(ci.localSet.size());
    for(auto i=0u; i<dims.size(); i++)
        dims[i]=static_cast<int>(ci.localSet[i].size());
    return dims;
}
}

template<class T>
TensorCI1<T> to_tci1(TensorCI2<T> const& tci2)
{
    return TensorCI1<T>(tci2, to_tci1Param(tci2.param));
}

template<class T>
TensorCI2<T> to_tci2(TensorCI1<T> const& tci1, function<T(vector<int>)> g, TensorCI2Param<T> par)
{
    TensorCI2<T> tci2(g, impl::readDims(tci1), par);
    tci2.addPivots(tci1);
    return tci2;
}

template<class T>
TensorCI2<T> to_tci2(TensorCI1<T> const& tci1, function<T(vector<int>)> g)
{
    TensorCI2Param<T> par=to_tci2Param(tci1.param);
    par.bondDim=static_cast<int>(tci1.pivotError.size());
    return to_tci2(tci1,g,par);
}

template<class T>
TensorCI2<T> to_tci2(TensorCI1<T> const& tci1, TensorCI2Param<T> par)
{
    TensorCI2<T> tci2(tci1.f, impl::readDims(tci1), par);
    tci2.addPivots(tci1);
    return tci2;
}

template<class T>
TensorCI2<T> to_tci2(TensorCI1<T> const& tci1)
{
    TensorCI2Param<T> par=to_tci2Param(tci1.param);
    par.bondDim=static_cast<int>(tci1.pivotError.size());
    return to_tci2(tci1,par);
}

} // namespace xfac_quad
