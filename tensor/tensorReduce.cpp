#ifndef TENSOR_REDUCE_
#define TENSOR_REDUCE_

#include "../include/tensor.h"


#ifdef __CUDACC__
  #include <thrust/device_vector.h>
#endif

template <typename XPU, typename DT>
DT Tensor<XPU, DT>::reduce_sum () const
{ DT val = 0.;
#ifdef __CUDACC__
  thrust::device_ptr<DT> dev_ptr(dptr);
  val = thrust::reduce (dev_ptr, dev_ptr+size(), val, thrust::plus<DT>());
#else
#pragma omp parallel for reduction(+: val)
  for (int i = 0; i < size(); ++i)
    val += dptr[i];
#endif
  return val;
};
#ifdef __CUDACC__
template float  TensorGPUf::reduce_sum () const;
template double TensorGPUd::reduce_sum () const;
#else
template float  TensorCPUf::reduce_sum () const;
template double TensorCPUd::reduce_sum () const;
#endif


template <typename XPU, typename DT>
DT Tensor<XPU, DT>::reduce_max () const
{ DT val = -FLT_MAX;
#ifdef __CUDACC__
  thrust::device_ptr<DT> dev_ptr(dptr);
  val = thrust::reduce (dev_ptr, dev_ptr+size(), val, thrust::maximum<DT>());
#else
#pragma omp parallel for reduction(max: val)
  for (int i = 0; i < size(); ++i)
    if (val < dptr[i])
      val = dptr[i];
#endif
  return val;
};
#ifdef __CUDACC__
template float  TensorGPUf::reduce_max () const;
template double TensorGPUd::reduce_max () const;
#else
template float  TensorCPUf::reduce_max () const;
template double TensorCPUd::reduce_max () const;
#endif

#endif
