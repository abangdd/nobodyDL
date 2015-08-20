#ifndef TENSOR_RANDOM_
#define TENSOR_RANDOM_

#include "../include/tensor.h"

#ifndef __CUDACC__
void rand_check (const int status)
{ CHECK_EQ (status, VSL_STATUS_OK);
}
#endif

#ifdef __CUDACC__
template <>
Random<GPU>::Random (const int did) : did_(did) { }
#else
template <>
Random<CPU>::Random (const int did) : did_(did)
{ rand_check (vslNewStream (&vStream_, VSL_BRNG_MT19937, 1));
}
#endif

#ifdef __CUDACC__
template <>
Random<GPU>::~Random () { }
#else
template <>
Random<CPU>::~Random ()
{ rand_check (vslDeleteStream (&vStream_));
}
#endif

#ifdef __CUDACC__
template <>
void Random<GPU>::set_seed (int seed)
{ cuda_check (curandSetPseudoRandomGeneratorSeed (dnnctx[did_]->curand_, seed));
}
#else
template <>
void Random<CPU>::set_seed (int seed)
{ rand_check (vslDeleteStream (&vStream_));
  rand_check (vslNewStream (&vStream_, VSL_BRNG_MT19937, seed));
}
#endif



XPU_KERNEL(tensor_scale) (
  const int num_kernels, float *data, const float a,  const float b)
{ kernel_for (index, num_kernels)
    data[index] = data[index] * (b - a) + a;
}

template <typename DT>
XPU_KERNEL(tensor_constant) (
  const int num_kernels, DT *data, const DT a)
{ kernel_for (index, num_kernels)
    data[index] = a;
}



#ifdef __CUDACC__
template <>
void Random<GPU>::gaussian (float *data, int size, const float mu, const float sigma) const
{ CHECK (sigma > 0.f);
  cuda_check (curandGenerateNormal (dnnctx[did_]->curand_, data, size, mu, sigma));
}
#else
template <>
void Random<CPU>::gaussian (float *data, int size, const float mu, const float sigma) const
{ CHECK (sigma > 0.f);
  rand_check (vsRngGaussian (0, vStream_, size, data, mu, sigma)); // TODO
}
#endif

#ifdef __CUDACC__
template <>
void Random<GPU>::uniform  (float *data, int size, const float  a, const float b) const
{ const int N = size;
  cuda_check (curandGenerateUniform (dnnctx[did_]->curand_, data, N));
  if (a != 0.f || b != 1.f)
  XPU_KERNEL_LAUNCH (tensor_scale, cuda_get_blocks(N), CUDA_NUM_THREADS, 0, dnnctx[did_]->stream_,
    N, data, a, b);
}
#else
template <>
void Random<CPU>::uniform  (float *data, int size, const float  a, const float b) const
{ const int N = size;
  rand_check (vsRngUniform (0, vStream_, N, data, a, b)); // TODO
}
#endif



template <typename XPU, typename DT>
void Tensor<XPU, DT>::init (const Random<XPU> &random, const int method, const DT a,  const DT b)
{ if      (method == GAUSSIAN)
    random.gaussian (dptr, size(), a, b);
  else if (method == UNIFORM)
    random.uniform  (dptr, size(), a, b);
}
#ifdef __CUDACC__
template void TensorGPUf::init (const Random<GPU> &random, const int method, const float a,  const float b);
#else
template void TensorCPUf::init (const Random<CPU> &random, const int method, const float a,  const float b);
#endif

template <typename XPU, typename DT>
void Tensor<XPU, DT>::init (const DT a)
{ const int N = size();
  XPU_KERNEL_LAUNCH (tensor_constant, cuda_get_blocks(N), CUDA_NUM_THREADS, 0, dnnctx[did_]->stream_,
    N, dptr, a);
}
#ifdef __CUDACC__
template void TensorGPUf::init (const float  a);
template void TensorGPUd::init (const double a);
#else
template void TensorCPUf::init (const float  a);
template void TensorCPUd::init (const double a);
#endif

#endif
