#ifndef TENSOR_BASE_
#define TENSOR_BASE_

#include <queue>
#include "../include/tensor.h"

template <typename XPU, typename DT>
Tensor<XPU, DT>::Tensor (const Tensor<XPU, DT>& t) {
    shape = t.shape;
    dptr = t.dptr;
    did_ = t.did_;
    cherry = false;
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::create (const Shape& s, const int did) {
    shape = s;
    did_ = did;
    mem_free ();
    mem_alloc();
    cherry = true;
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::create (const Shape& s, DT* sptr, const int did) {
    shape = s;
    dptr = sptr;
    did_ = did;
    cherry = false;
}
#ifdef __CUDACC__
template TensorGPUi::~Tensor();
template TensorGPUf::~Tensor();
template TensorGPUd::~Tensor();
template TensorGPUi::Tensor (const TensorGPUi& t);
template TensorGPUf::Tensor (const TensorGPUf& t);
template TensorGPUd::Tensor (const TensorGPUd& t);
template void TensorGPUi::create (const Shape& s, const int did);
template void TensorGPUf::create (const Shape& s, const int did);
template void TensorGPUd::create (const Shape& s, const int did);
template void TensorGPUi::create (const Shape& s, int* sptr, const int did);
template void TensorGPUf::create (const Shape& s, float* sptr, const int did);
template void TensorGPUd::create (const Shape& s, double* sptr, const int did);
#else
template TensorCPUi::~Tensor();
template TensorCPUf::~Tensor();
template TensorCPUd::~Tensor();
template TensorCPUi::Tensor (const TensorCPUi& t);
template TensorCPUf::Tensor (const TensorCPUf& t);
template TensorCPUd::Tensor (const TensorCPUd& t);
template void TensorCPUi::create (const Shape& s, const int did);
template void TensorCPUf::create (const Shape& s, const int did);
template void TensorCPUd::create (const Shape& s, const int did);
template void TensorCPUi::create (const Shape& s, int* sptr, const int did);
template void TensorCPUf::create (const Shape& s, float* sptr, const int did);
template void TensorCPUd::create (const Shape& s, double* sptr, const int did);
#endif



template <typename XPU, typename DT>
Tensor<XPU, DT> Tensor<XPU, DT>::mat_view (const int cols) const {
    CHECK_EQ (size() % cols, 0);
    Tensor<XPU, DT> t;
    t.create (Shape (size()/cols, cols, 1, 1), dptr, did_);
    return t;
}

template <typename XPU, typename DT>
Tensor<XPU, DT> Tensor<XPU, DT>::section (const int begin, const int end) const {
    Tensor<XPU, DT> t;
    t.shape = shape.section (begin, end);
    t.dptr = dptr + t.size() / (end + 1 - begin) * begin;  // 127 + 1 - 0
    t.did_ = did_;
    t.cherry = false;
    return t;
}

template <typename XPU, typename DT>
Tensor<XPU, DT>& Tensor<XPU, DT>::operator= (const Tensor<XPU, DT>& t) {
    shape = t.shape;
    dptr = t.dptr;
    did_ = t.did_;
    cherry = false;
    return *this;
}
#ifdef __CUDACC__
template TensorGPUf TensorGPUf::mat_view (const int cols) const;
template TensorGPUd TensorGPUd::mat_view (const int cols) const;
template TensorGPUi TensorGPUi::section (const int begin, const int end) const;
template TensorGPUf TensorGPUf::section (const int begin, const int end) const;
template TensorGPUd TensorGPUd::section (const int begin, const int end) const;
template TensorGPUi& TensorGPUi::operator= (const TensorGPUi& t);
template TensorGPUf& TensorGPUf::operator= (const TensorGPUf& t);
template TensorGPUd& TensorGPUd::operator= (const TensorGPUd& t);
#else
template TensorCPUf TensorCPUf::mat_view (const int cols) const;
template TensorCPUd TensorCPUd::mat_view (const int cols) const;
template TensorCPUi TensorCPUi::section (const int begin, const int end) const;
template TensorCPUf TensorCPUf::section (const int begin, const int end) const;
template TensorCPUd TensorCPUd::section (const int begin, const int end) const;
template TensorCPUi& TensorCPUi::operator= (const TensorCPUi& t);
template TensorCPUf& TensorCPUf::operator= (const TensorCPUf& t);
template TensorCPUd& TensorCPUd::operator= (const TensorCPUd& t);
#endif



template <typename XPU, typename DT>
void Tensor<XPU, DT>::mem_alloc () {
    char size_m[16]; sprintf (size_m, "%.2f MB", size_d() / 1048576.f);
#ifdef __CUDACC__
    LOG_IF (INFO, size_d() >= 1048576) << "\tGPU  " << did_ << "  memory required for Tensor\t" << size_m;
    cuda_check (cudaMallocManaged (&dptr, size_d(), cudaMemAttachGlobal));
#else
    LOG_IF (INFO, size_d() >= 1073741824) << "\tCPU memory required for Tensor\t" << size_m;
    dptr = (DT*) malloc (size_d());
#endif
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::mem_free () { 
#ifdef __CUDACC__
    if (cherry)  cuda_check (cudaFree (dptr));
#else
    if (cherry)  free (dptr);
#endif
    dptr = nullptr;
    cherry = false;
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::mem_set (const unsigned char a) {
#ifdef __CUDACC__
    cuda_check (cudaMemset ((void*)dptr, a, size_d()));
#else
    memset ((void*)dptr, a, size_d());
#endif
}

#ifdef __CUDACC__
template void TensorGPUi::mem_free ();
template void TensorGPUf::mem_free ();
template void TensorGPUd::mem_free ();
template void TensorGPUf::mem_set (const unsigned char a);
template void TensorGPUd::mem_set (const unsigned char a);
#else
template void TensorCPUi::mem_free ();
template void TensorCPUf::mem_free ();
template void TensorCPUd::mem_free ();
template void TensorCPUf::mem_set (const unsigned char a);
template void TensorCPUd::mem_set (const unsigned char a);
#endif



#define CPU2GPU cudaMemcpyHostToDevice
#define GPU2CPU cudaMemcpyDeviceToHost
#define CPU2CPU cudaMemcpyHostToHost
#define GPU2GPU cudaMemcpyDeviceToDevice

#ifdef __CUDACC__
template <>
void TensorGPUf::memcpy_from_gpu (void* ptr) {
    cuda_check (cudaMemcpy (dptr, ptr, size_d(), GPU2GPU));
}
template <>
void TensorGPUd::memcpy_from_gpu (void* ptr) {
    cuda_check (cudaMemcpy (dptr, ptr, size_d(), GPU2GPU));
}
template <>
void TensorGPUf::memcpy_from_cpu (void* ptr) {
    cuda_check (cudaMemcpy (dptr, ptr, size_d(), CPU2GPU));
}
template <>
void TensorGPUd::memcpy_from_cpu (void* ptr) {
    cuda_check (cudaMemcpy (dptr, ptr, size_d(), CPU2GPU));
}
template <>
void TensorCPUi::memcpy_from_gpu (void* ptr) {
    cuda_check (cudaMemcpy (dptr, ptr, size_d(), GPU2CPU));
}
template <>
void TensorCPUf::memcpy_from_gpu (void* ptr) {
    cuda_check (cudaMemcpy (dptr, ptr, size_d(), GPU2CPU));
}
template <>
void TensorCPUd::memcpy_from_gpu (void* ptr) {
    cuda_check (cudaMemcpy (dptr, ptr, size_d(), GPU2CPU));
}
#else
template <>
void TensorCPUi::memcpy_from_cpu (void* ptr) {
    memcpy ((void*)dptr, ptr, size_d());
}
template <>
void TensorCPUf::memcpy_from_cpu (void* ptr) {
    memcpy ((void*)dptr, ptr, size_d());
}
template <>
void TensorCPUd::memcpy_from_cpu (void* ptr) {
    memcpy ((void*)dptr, ptr, size_d());
}
#endif

#ifdef __CUDACC__
template <>
void TensorGPUf::memcpy_to_gpu (void* ptr) const {
    cuda_check (cudaMemcpy (ptr, dptr, size_d(), GPU2GPU));
}
template <>
void TensorGPUd::memcpy_to_gpu (void* ptr) const {
    cuda_check (cudaMemcpy (ptr, dptr, size_d(), GPU2GPU));
}
template <>
void TensorGPUf::memcpy_to_cpu (void* ptr) const {
    cuda_check (cudaMemcpy (ptr, dptr, size_d(), GPU2CPU));
}
template <>
void TensorGPUd::memcpy_to_cpu (void* ptr) const {
    cuda_check (cudaMemcpy (ptr, dptr, size_d(), GPU2CPU));
}
template <>
void TensorCPUi::memcpy_to_gpu (void* ptr) const {
    cuda_check (cudaMemcpy (ptr, dptr, size_d(), CPU2GPU));
}
template <>
void TensorCPUf::memcpy_to_gpu (void* ptr) const {
    cuda_check (cudaMemcpy (ptr, dptr, size_d(), CPU2GPU));
}
template <>
void TensorCPUd::memcpy_to_gpu (void* ptr) const {
    cuda_check (cudaMemcpy (ptr, dptr, size_d(), CPU2GPU));
}
#else
template <>
void TensorCPUi::memcpy_to_cpu (void* ptr) const {
    memcpy (ptr, (void*)dptr, size_d());
}
template <>
void TensorCPUf::memcpy_to_cpu (void* ptr) const {
    memcpy (ptr, (void*)dptr, size_d());
}
template <>
void TensorCPUd::memcpy_to_cpu (void* ptr) const {
    memcpy (ptr, (void*)dptr, size_d());
}
#endif



template <typename XPU, typename DT>
void Tensor<XPU, DT>::copy (const Tensor<GPU, DT>& in) {
    CHECK_LE (size(), in.size());
    memcpy_from_gpu (in.dptr);
}
template <typename XPU, typename DT>
void Tensor<XPU, DT>::copy (const Tensor<CPU, DT>& in) {
    CHECK_LE (size(), in.size());
    memcpy_from_cpu (in.dptr);
}
#ifdef __CUDACC__
template void TensorGPUf::copy (const TensorGPUf& in);
template void TensorGPUd::copy (const TensorGPUd& in);
template void TensorGPUf::copy (const TensorCPUf& in);
template void TensorGPUd::copy (const TensorCPUd& in);
template void TensorCPUi::copy (const TensorGPUi& in);
template void TensorCPUf::copy (const TensorGPUf& in);
template void TensorCPUd::copy (const TensorGPUd& in);
#else
template void TensorCPUi::copy (const TensorCPUi& in);
template void TensorCPUf::copy (const TensorCPUf& in);
template void TensorCPUd::copy (const TensorCPUd& in);
#endif



template <typename XPU, typename DT>
void Tensor<XPU, DT>::print (const int cnt) const {
    const int N = std::min (int(size()), cnt);
    XPU_KERNEL_LAUNCH (kprint, XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream(),
        N, dptr);
    XPU::check_async ("kprint");
    printf ("\n");
}
#ifdef __CUDACC__
template void TensorGPUi::print (const int cnt) const;
template void TensorGPUf::print (const int cnt) const;
template void TensorGPUd::print (const int cnt) const;
#else
template void TensorCPUi::print (const int cnt) const;
template void TensorCPUf::print (const int cnt) const;
template void TensorCPUd::print (const int cnt) const;
#endif

#endif
