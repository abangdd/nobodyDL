#ifndef TENSOR_VML_
#define TENSOR_VML_

#include <cmath>
#include "../include/tensor.h"

using std::max;
using std::min;

template <typename DT>
XPU_KERNEL(kernel_add) (const int knum, const DT val, DT* y) {
    kernel_for (i, knum)
        y[i] += val;
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::add (const DT val) {
    const int N = size();
    XPU_KERNEL_LAUNCH (kernel_add, XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream(),
        N, val, dptr);
};
#ifdef __CUDACC__
template void TensorGPUf::add (const float val);
template void TensorGPUd::add (const double val);
#else
template void TensorCPUf::add (const float val);
template void TensorCPUd::add (const double val);
#endif



template <typename DT>
XPU_KERNEL(kernel_binary_loss) (const int knum, const DT* pred, const DT* anno, DT* loss) {
    kernel_for (i, knum)
        loss[i] = round(pred[i]) != anno[i];
};

template <typename XPU, typename DT>
void Tensor<XPU, DT>::binary_loss (const Tensor<XPU, DT>& pred, const Tensor<XPU, DT>& anno) {
    const int N = size();
    XPU_KERNEL_LAUNCH (kernel_binary_loss, XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream(),
        N, pred.dptr, anno.dptr, dptr);
    XPU::check_sync ("kernel_binary_loss");
};
#ifdef __CUDACC__
template void TensorGPUf::binary_loss (const TensorGPUf& pred, const TensorGPUf& anno);
template void TensorGPUd::binary_loss (const TensorGPUd& pred, const TensorGPUd& anno);
#else
template void TensorCPUf::binary_loss (const TensorCPUf& pred, const TensorCPUf& anno);
template void TensorCPUd::binary_loss (const TensorCPUd& pred, const TensorCPUd& anno);
#endif



template <typename DT>
XPU_KERNEL(kernel_drop_chls) (const int knum, const int area, const DT drop, const DT* mask, DT* data) {
    kernel_for (i, knum)
        data[i] *= (mask[i/area] > drop) / (DT(1) - drop);
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::drop_chls (const Tensor<XPU, DT>& mask, const DT drop) {
    const int N = size();
    XPU_KERNEL_LAUNCH (kernel_drop_chls, XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream(),
        N, area(), drop, mask.dptr, dptr);
    XPU::check_sync ("kernel_drop_chls");
};
#ifdef __CUDACC__
template void TensorGPUf::drop_chls (const TensorGPUf& mask, const float drop);
template void TensorGPUd::drop_chls (const TensorGPUd& mask, const double drop);
#else
template void TensorCPUf::drop_chls (const TensorCPUf& mask, const float drop);
template void TensorCPUd::drop_chls (const TensorCPUd& mask, const double drop);
#endif



template <typename DT>
XPU_KERNEL(kernel_sigmoid) (const int knum, const DT* src_data, DT* dst_data) {
    kernel_for (i, knum)
        dst_data[i] = DT(1) / (exp(-src_data[i]) + DT(1));
};

template <typename XPU, typename DT>
void Tensor<XPU, DT>::sigmoid () {
    const int N = size();
    XPU_KERNEL_LAUNCH (kernel_sigmoid, XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream(),
        N, dptr, dptr);
    XPU::check_sync ("kernel_sigmoid");
};

template <typename XPU, typename DT>
void Tensor<XPU, DT>::softmax () {
#ifdef __CUDACC__
    const DT alpha = 1, beta = 0;
    cudnnTensorDescriptor_t tensorDesc;
    cuda_check (cudnnCreateTensorDescriptor (&tensorDesc));
    cuda_check (cudnnSetTensor4dDescriptor (tensorDesc, CUDNN_TENSOR_NCHW, cudnn_type<DT>(), nums(), chls(), rows(), cols()));
    cuda_check (cudnnSoftmaxForward (get_cunn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha, tensorDesc, dptr,
        &beta,  tensorDesc, dptr));
    cuda_check (cudnnDestroyTensorDescriptor (tensorDesc));
#else
#pragma omp parallel for
    for (int i = 0; i < nums(); ++i) {
        DT* ptr = dptr + dims() * i;
        DT sumval = 0;
        for (int j = 0; j < dims(); ++j)
            sumval += ptr[j] = std::exp (ptr[j]);
        for (int j = 0; j < dims(); ++j)
            ptr[j] /= sumval;
    }
#endif
}
#ifdef __CUDACC__
template void TensorGPUf::sigmoid ();
template void TensorGPUd::sigmoid ();
template void TensorGPUf::softmax ();
template void TensorGPUd::softmax ();
#else
template void TensorCPUf::sigmoid ();
template void TensorCPUd::sigmoid ();
template void TensorCPUf::softmax ();
template void TensorCPUd::softmax ();
#endif



template <class Oper, typename DT>
XPU_KERNEL(binary_vexpr) (const int knum, const DT* a, const DT* b, DT* y) {
    Oper op;
    kernel_for (i, knum)
        y[i] = op (a[i], b[i]);
}

template <class Oper, typename DT>
XPU_KERNEL(unary_vexpr) (const int knum, const DT* a, DT* y) {
    Oper op;
    kernel_for (i, knum)
        y[i] = op (a[i]);
}

#ifdef __CUDACC__
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vadd (const Tensor<XPU, DT>& A, const Tensor<XPU, DT>& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    binary_vexpr_kernel<opplus<DT>><<<XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
        (N, A.dptr, B.dptr, dptr);
    XPU::check_sync ("cublas_vadd");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vsub (const Tensor<XPU, DT>& A, const Tensor<XPU, DT>& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    binary_vexpr_kernel<opsub <DT>><<<XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
        (N, A.dptr, B.dptr, dptr);
    XPU::check_sync ("cublas_vsub");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vmul (const Tensor<XPU, DT>& A, const Tensor<XPU, DT>& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    binary_vexpr_kernel<opmul <DT>><<<XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
        (N, A.dptr, B.dptr, dptr);
    XPU::check_sync ("cublas_vmul");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vdiv (const Tensor<XPU, DT>& A, const Tensor<XPU, DT>& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    binary_vexpr_kernel<opdiv <DT>><<<XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
        (N, A.dptr, B.dptr, dptr);
    XPU::check_sync ("cublas_vdiv");
};
template void TensorGPUf::blas_vadd (const TensorGPUf& A, const TensorGPUf& B);
template void TensorGPUd::blas_vadd (const TensorGPUd& A, const TensorGPUd& B);
template void TensorGPUf::blas_vsub (const TensorGPUf& A, const TensorGPUf& B);
template void TensorGPUd::blas_vsub (const TensorGPUd& A, const TensorGPUd& B);
template void TensorGPUf::blas_vmul (const TensorGPUf& A, const TensorGPUf& B);
template void TensorGPUd::blas_vmul (const TensorGPUd& A, const TensorGPUd& B);
template void TensorGPUf::blas_vdiv (const TensorGPUf& A, const TensorGPUf& B);
template void TensorGPUd::blas_vdiv (const TensorGPUd& A, const TensorGPUd& B);
#else
template <>
void TensorCPUf::blas_vadd (const TensorCPUf& A, const TensorCPUf& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    vsAdd (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUd::blas_vadd (const TensorCPUd& A, const TensorCPUd& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    vdAdd (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUf::blas_vsub (const TensorCPUf& A, const TensorCPUf& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    vsSub (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUd::blas_vsub (const TensorCPUd& A, const TensorCPUd& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    vdSub (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUf::blas_vmul (const TensorCPUf& A, const TensorCPUf& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    vsMul (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUd::blas_vmul (const TensorCPUd& A, const TensorCPUd& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    vdMul (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUf::blas_vdiv (const TensorCPUf& A, const TensorCPUf& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    vsDiv (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUd::blas_vdiv (const TensorCPUd& A, const TensorCPUd& B) {
    const int N = size();    CHECK_EQ (A.size(), B.size());
    vdDiv (N, A.dptr, B.dptr, dptr);
};
#endif



#ifdef __CUDACC__
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vabs (const Tensor<XPU, DT>& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    unary_vexpr_kernel<opabs<DT>><<<XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
        (N, in.dptr, dptr);
    XPU::check_sync ("cublas_vabs");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vexp (const Tensor<XPU, DT>& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    unary_vexpr_kernel<opexp<DT>><<<XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
        (N, in.dptr, dptr);
    XPU::check_sync ("cublas_vexp");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vinv (const Tensor<XPU, DT>& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    unary_vexpr_kernel<opinv<DT>><<<XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
        (N, in.dptr, dptr);
    XPU::check_sync ("cublas_vinv");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vsqr (const Tensor<XPU, DT>& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    unary_vexpr_kernel<opsquare<DT>><<<XPU::get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
        (N, in.dptr, dptr);
    XPU::check_sync ("cublas_vsqr");
};
template void TensorGPUf::blas_vabs (const TensorGPUf& in);
template void TensorGPUd::blas_vabs (const TensorGPUd& in);
template void TensorGPUf::blas_vexp (const TensorGPUf& in);
template void TensorGPUd::blas_vexp (const TensorGPUd& in);
template void TensorGPUf::blas_vinv (const TensorGPUf& in);
template void TensorGPUd::blas_vinv (const TensorGPUd& in);
template void TensorGPUf::blas_vsqr (const TensorGPUf& in);
template void TensorGPUd::blas_vsqr (const TensorGPUd& in);
#else
template <>
void TensorCPUf::blas_vabs (const TensorCPUf& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    vsAbs (N, in.dptr, dptr);
};
template <>
void TensorCPUd::blas_vabs (const TensorCPUd& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    vdAbs (N, in.dptr, dptr);
};
template <>
void TensorCPUf::blas_vexp (const TensorCPUf& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    vsExp (N, in.dptr, dptr);
};
template <>
void TensorCPUd::blas_vexp (const TensorCPUd& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    vdExp (N, in.dptr, dptr);
};
template <>
void TensorCPUf::blas_vinv (const TensorCPUf& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    vsInv (N, in.dptr, dptr);
};
template <>
void TensorCPUd::blas_vinv (const TensorCPUd& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    vdInv (N, in.dptr, dptr);
};
template <>
void TensorCPUf::blas_vsqr (const TensorCPUf& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    vsSqr (N, in.dptr, dptr);
};
template <>
void TensorCPUd::blas_vsqr (const TensorCPUd& in) {
    const int N = size();    CHECK_EQ (N, in.size());
    vdSqr (N, in.dptr, dptr);
};
#endif

#endif
