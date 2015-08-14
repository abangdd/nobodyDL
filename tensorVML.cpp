#ifndef TENSOR_VML_
#define TENSOR_VML_

#include "../include/tensor.h"

template <typename DT>
XPU_KERNEL(kernel_add) (const int num_kernels, const DT val, DT *y)
{ kernel_for (index, num_kernels)
    y[index] += val;
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::add (const DT val)
{ const int N = size();
  XPU_KERNEL_LAUNCH (kernel_add, cuda_get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream(),
    N, val, dptr);
};
#ifdef __CUDACC__
template void TensorGPUf::add (const float  val);
template void TensorGPUd::add (const double val);
#else
template void TensorCPUf::add (const float  val);
template void TensorCPUd::add (const double val);
#endif



template <class Oper, typename DT>
XPU_KERNEL(binary_vexpr) (const int num_kernels, const DT *a, const DT *b, DT *y)
{ Oper op;
  kernel_for (i, num_kernels)
    y[i] = op (a[i], b[i]);
}

template <class Oper, typename DT>
XPU_KERNEL(unary_vexpr)  (const int num_kernels, const DT *a, DT *y)
{ Oper op;
  kernel_for (i, num_kernels)
    y[i] = op (a[i]);
}



#ifdef __CUDACC__
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vadd (const Tensor<XPU, DT> &A, const Tensor<XPU, DT> &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  binary_vexpr_kernel<opplus <DT>><<<cuda_get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
    (N, A.dptr, B.dptr, dptr);
  cuda_sync_check ("cublas_vadd");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vsub (const Tensor<XPU, DT> &A, const Tensor<XPU, DT> &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  binary_vexpr_kernel<opminus<DT>><<<cuda_get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
    (N, A.dptr, B.dptr, dptr);
  cuda_sync_check ("cublas_vsub");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vmul (const Tensor<XPU, DT> &A, const Tensor<XPU, DT> &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  binary_vexpr_kernel<opmul  <DT>><<<cuda_get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
    (N, A.dptr, B.dptr, dptr);
  cuda_sync_check ("cublas_vmul");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vdiv (const Tensor<XPU, DT> &A, const Tensor<XPU, DT> &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  binary_vexpr_kernel<opdiv  <DT>><<<cuda_get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
    (N, A.dptr, B.dptr, dptr);
  cuda_sync_check ("cublas_vdiv");
};
template void TensorGPUf::blas_vadd (const TensorGPUf &A, const TensorGPUf &B);
template void TensorGPUd::blas_vadd (const TensorGPUd &A, const TensorGPUd &B);
template void TensorGPUf::blas_vsub (const TensorGPUf &A, const TensorGPUf &B);
template void TensorGPUd::blas_vsub (const TensorGPUd &A, const TensorGPUd &B);
template void TensorGPUf::blas_vmul (const TensorGPUf &A, const TensorGPUf &B);
template void TensorGPUd::blas_vmul (const TensorGPUd &A, const TensorGPUd &B);
template void TensorGPUf::blas_vdiv (const TensorGPUf &A, const TensorGPUf &B);
template void TensorGPUd::blas_vdiv (const TensorGPUd &A, const TensorGPUd &B);
#else
template <>
void TensorCPUf::blas_vadd (const TensorCPUf &A, const TensorCPUf &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  vsAdd (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUd::blas_vadd (const TensorCPUd &A, const TensorCPUd &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  vdAdd (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUf::blas_vsub (const TensorCPUf &A, const TensorCPUf &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  vsSub (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUd::blas_vsub (const TensorCPUd &A, const TensorCPUd &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  vdSub (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUf::blas_vmul (const TensorCPUf &A, const TensorCPUf &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  vsMul (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUd::blas_vmul (const TensorCPUd &A, const TensorCPUd &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  vdMul (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUf::blas_vdiv (const TensorCPUf &A, const TensorCPUf &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  vsDiv (N, A.dptr, B.dptr, dptr);
};
template <>
void TensorCPUd::blas_vdiv (const TensorCPUd &A, const TensorCPUd &B)
{ const int N = size();  CHECK_EQ (A.size(), B.size());
  vdDiv (N, A.dptr, B.dptr, dptr);
};
#endif



#ifdef __CUDACC__
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vabs (const Tensor<XPU, DT> &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  unary_vexpr_kernel<opabs<DT>><<<cuda_get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
    (N, in.dptr, dptr);
  cuda_sync_check ("cublas_vabs");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vexp (const Tensor<XPU, DT> &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  unary_vexpr_kernel<opexp<DT>><<<cuda_get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
    (N, in.dptr, dptr);
  cuda_sync_check ("cublas_vexp");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vinv (const Tensor<XPU, DT> &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  unary_vexpr_kernel<opinv<DT>><<<cuda_get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
    (N, in.dptr, dptr);
  cuda_sync_check ("cublas_vinv");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vsqr (const Tensor<XPU, DT> &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  unary_vexpr_kernel<opsquare<DT>><<<cuda_get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
    (N, in.dptr, dptr);
  cuda_sync_check ("cublas_vsqr");
};
template <typename XPU, typename DT>
void Tensor<XPU, DT>::blas_vsqrt(const Tensor<XPU, DT> &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  unary_vexpr_kernel<opsqrt<DT>><<<cuda_get_blocks(N), CUDA_NUM_THREADS, 0, get_calc_stream()>>>
    (N, in.dptr, dptr);
  cuda_sync_check ("cublas_vsqrt");
};
template void TensorGPUf::blas_vabs (const TensorGPUf &in);
template void TensorGPUd::blas_vabs (const TensorGPUd &in);
template void TensorGPUf::blas_vexp (const TensorGPUf &in);
template void TensorGPUd::blas_vexp (const TensorGPUd &in);
template void TensorGPUf::blas_vinv (const TensorGPUf &in);
template void TensorGPUd::blas_vinv (const TensorGPUd &in);
template void TensorGPUf::blas_vsqr (const TensorGPUf &in);
template void TensorGPUd::blas_vsqr (const TensorGPUd &in);
template void TensorGPUf::blas_vsqrt(const TensorGPUf &in);
template void TensorGPUd::blas_vsqrt(const TensorGPUd &in);
#else
template <>
void TensorCPUf::blas_vabs (const TensorCPUf &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  vsAbs (N, in.dptr, dptr);
};
template <>
void TensorCPUd::blas_vabs (const TensorCPUd &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  vdAbs (N, in.dptr, dptr);
};
template <>
void TensorCPUf::blas_vexp (const TensorCPUf &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  vsExp (N, in.dptr, dptr);
};
template <>
void TensorCPUd::blas_vexp (const TensorCPUd &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  vdExp (N, in.dptr, dptr);
};
template <>
void TensorCPUf::blas_vinv (const TensorCPUf &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  vsInv (N, in.dptr, dptr);
};
template <>
void TensorCPUd::blas_vinv (const TensorCPUd &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  vdInv (N, in.dptr, dptr);
};
template <>
void TensorCPUf::blas_vsqr (const TensorCPUf &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  vsSqr (N, in.dptr, dptr);
};
template <>
void TensorCPUd::blas_vsqr (const TensorCPUd &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  vdSqr (N, in.dptr, dptr);
};
template <>
void TensorCPUf::blas_vsqrt(const TensorCPUf &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  vsSqrt(N, in.dptr, dptr);
};
template <>
void TensorCPUd::blas_vsqrt(const TensorCPUd &in)
{ const int N = size();  CHECK_EQ (N, in.size());
  vdSqrt(N, in.dptr, dptr);
};
#endif

#endif
