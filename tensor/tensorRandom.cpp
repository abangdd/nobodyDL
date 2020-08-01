#ifndef TENSOR_RANDOM_
#define TENSOR_RANDOM_

#include "../include/tensor.h"

#ifndef __CUDACC__
void rand_check (const int status) {
    CHECK_EQ (status, VSL_STATUS_OK);
}
#endif

#ifdef __CUDACC__
template <>
RandomGPUf::Random (const int did) : did_(did) { }
template <>
RandomGPUd::Random (const int did) : did_(did) { }
#else
template <>
RandomCPUf::Random (const int did) : did_(did) { rand_check (vslNewStream (&vStream_, VSL_BRNG_MT19937, 1)); }
template <>
RandomCPUd::Random (const int did) : did_(did) { rand_check (vslNewStream (&vStream_, VSL_BRNG_MT19937, 1)); }
#endif

#ifdef __CUDACC__
template <>
RandomGPUf::~Random () { }
template <>
RandomGPUd::~Random () { }
#else
template <>
RandomCPUf::~Random () { rand_check (vslDeleteStream (&vStream_)); }
template <>
RandomCPUd::~Random () { rand_check (vslDeleteStream (&vStream_)); }
#endif

#ifdef __CUDACC__
template <>
void RandomGPUf::set_seed (int seed) {
    cuda_check (curandSetPseudoRandomGeneratorSeed (dnnCtx[did_].curand_, seed));
}
template <>
void RandomGPUd::set_seed (int seed) {
    cuda_check (curandSetPseudoRandomGeneratorSeed (dnnCtx[did_].curand_, seed));
}
#else
template <>
void RandomCPUf::set_seed (int seed) {
    rand_check (vslDeleteStream (&vStream_));
    rand_check (vslNewStream (&vStream_, VSL_BRNG_MT19937, seed));
}
template <>
void RandomCPUd::set_seed (int seed) {
    rand_check (vslDeleteStream (&vStream_));
    rand_check (vslNewStream (&vStream_, VSL_BRNG_MT19937, seed));
}
#endif



template <typename DT>
XPU_KERNEL(kscale) (const int knum, DT* data, const DT a, const DT b) {
    kernel_for (index, knum)
        data[index] = data[index] * (b - a) + a;
}

template <typename DT>
XPU_KERNEL(kconstant) (const int knum, DT* data, const DT a) {
    kernel_for (index, knum)
        data[index] = a;
}



#ifdef __CUDACC__
template <>
void RandomGPUf::gaussian (float* data, int size, const float mu, const float sigma) const {
    CHECK_GT (sigma, 0);
    cuda_check (curandGenerateNormal (dnnCtx[did_].curand_, data, size, mu, sigma));
}
template <>
void RandomGPUd::gaussian (double* data, int size, const double mu, const double sigma) const {
    CHECK_GT (sigma, 0);
    cuda_check (curandGenerateNormalDouble (dnnCtx[did_].curand_, data, size, mu, sigma));
}
#else
template <>
void RandomCPUf::gaussian (float* data, int size, const float mu, const float sigma) const {
    CHECK_GT (sigma, 0);
    rand_check (vsRngGaussian (0, vStream_, size, data, mu, sigma)); // TODO
}
template <>
void RandomCPUd::gaussian (double* data, int size, const double mu, const double sigma) const {
    CHECK_GT (sigma, 0);
    rand_check (vdRngGaussian (0, vStream_, size, data, mu, sigma)); // TODO
}
#endif

#ifdef __CUDACC__
template <>
void RandomGPUf::uniform (float* data, int size, const float a, const float b) const {
    const int N = size;
    cuda_check (curandGenerateUniform (dnnCtx[did_].curand_, data, N));
    if (a != 0.f || b != 1.f)
    XPU_KERNEL_LAUNCH (kscale, GPU::get_blocks(N), CUDA_NUM_THREADS, 0, dnnCtx[did_].stream_,
        N, data, a, b);
    GPU::check_sync ("kscale");
}
template <>
void RandomGPUd::uniform (double* data, int size, const double a, const double b) const {
    const int N = size;
    cuda_check (curandGenerateUniformDouble (dnnCtx[did_].curand_, data, N));
    if (a != 0.f || b != 1.f)
    XPU_KERNEL_LAUNCH (kscale, GPU::get_blocks(N), CUDA_NUM_THREADS, 0, dnnCtx[did_].stream_,
        N, data, a, b);
    GPU::check_sync ("kscale");
}
#else
template <>
void RandomCPUf::uniform (float* data, int size, const float a, const float b) const {
    const int N = size;
    rand_check (vsRngUniform (0, vStream_, N, data, a, b)); // TODO
}
template <>
void RandomCPUd::uniform (double* data, int size, const double a, const double b) const {
    const int N = size;
    rand_check (vdRngUniform (0, vStream_, N, data, a, b)); // TODO
}
#endif



template <typename XPU, typename DT>
void Tensor<XPU, DT>::init (const Random<XPU, DT>& random, const int method, const DT a, const DT b) {
    if (method == GAUSSIAN)
        random.gaussian (dptr, size(), a, b);
    else if (method == UNIFORM)
        random.uniform (dptr, size(), a, b);
}
#ifdef __CUDACC__
template void TensorGPUf::init (const RandomGPUf& random, const int method, const float a, const float b);
template void TensorGPUd::init (const RandomGPUd& random, const int method, const double a, const double b);
#else
template void TensorCPUf::init (const RandomCPUf& random, const int method, const float a, const float b);
template void TensorCPUd::init (const RandomCPUd& random, const int method, const double a, const double b);
#endif

template <typename XPU, typename DT>
void Tensor<XPU, DT>::init (const DT a) {
    const int N = size();
    XPU_KERNEL_LAUNCH (kconstant, XPU::get_blocks(N), CUDA_NUM_THREADS, 0, dnnCtx[did_].stream_,
        N, dptr, a);
    XPU::check_sync ("kconstant");
}
#ifdef __CUDACC__
template void TensorGPUf::init (const float a);
template void TensorGPUd::init (const double a);
#else
template void TensorCPUf::init (const float a);
template void TensorCPUd::init (const double a);
#endif

#endif
