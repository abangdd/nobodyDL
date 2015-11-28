#ifndef TENSOR_BLAS_
#define TENSOR_BLAS_

#include "../include/tensor.h"

#ifdef __CUDACC__
inline cublasOperation_t cublas_get_trans (bool t)  { return t ? CUBLAS_OP_T : CUBLAS_OP_N;  }
#else
inline CBLAS_TRANSPOSE    cblas_get_trans (bool t)  { return t ? CblasTrans : CblasNoTrans;  }
#endif


#ifdef __CUDACC__
template <>
void TensorGPUf::blas_amax (int &idx, float &val) const
{ const int N = size();
  cuda_check (
  cublasIsamax (get_blas_handle(), N, dptr, 1, &idx));
  idx = idx - 1;
  val = dptr[idx];
};
template <>
void TensorGPUd::blas_amax (int &idx, double &val) const
{ const int N = size();
  cuda_check (
  cublasIdamax (get_blas_handle(), N, dptr, 1, &idx));
  idx = idx - 1;
  val = dptr[idx];
};
#else
template <>
void TensorCPUf::blas_amax (int &idx, float  &val) const
{ const int N = size();
  idx = cblas_isamax (N, dptr, 1);
  val = dptr[idx];
};
template <>
void TensorCPUd::blas_amax (int &idx, double &val) const
{ const int N = size();
  idx = cblas_idamax (N, dptr, 1);
  val = dptr[idx];
};
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_amin (int &idx, float &val) const
{ const int N = size();
  cuda_check (
  cublasIsamin (get_blas_handle(), N, dptr, 1, &idx));
  idx = idx - 1;
  val = dptr[idx];
};
template <>
void TensorGPUd::blas_amin (int &idx, double &val) const
{ const int N = size();
  cuda_check (
  cublasIdamin (get_blas_handle(), N, dptr, 1, &idx));
  idx = idx - 1;
  val = dptr[idx];
};
#else
template <>
void TensorCPUf::blas_amin (int &idx, float  &val) const
{ const int N = size();
  idx = cblas_isamin (N, dptr, 1);
  val = dptr[idx];
};
template <>
void TensorCPUd::blas_amin (int &idx, double &val) const
{ const int N = size();
  idx = cblas_idamin (N, dptr, 1);
  val = dptr[idx];
};
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_asum (float  &val) const
{ const int N = size();
  cuda_check (
  cublasSasum (get_blas_handle(), N, dptr, 1, &val));
};
template <>
void TensorGPUd::blas_asum (double &val) const
{ const int N = size();
  cuda_check (
  cublasDasum (get_blas_handle(), N, dptr, 1, &val));
};
#else
template <>
void TensorCPUf::blas_asum (float  &val) const
{ const int N = size();
  val = cblas_sasum (N, dptr, 1);
};
template <>
void TensorCPUd::blas_asum (double &val) const
{ const int N = size();
  val = cblas_dasum (N, dptr, 1);
};
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_axpy (const TensorGPUf &in, float  alpha)
{ const int N = size();  CHECK_EQ (N, in.size());
  cuda_check (
  cublasSaxpy (get_blas_handle(), N, &alpha, in.dptr, 1, dptr, 1));
};
template <>
void TensorGPUd::blas_axpy (const TensorGPUd &in, double alpha)
{ const int N = size();  CHECK_EQ (N, in.size());
  cuda_check (
  cublasDaxpy (get_blas_handle(), N, &alpha, in.dptr, 1, dptr, 1));
};
#else
template <>
void TensorCPUf::blas_axpy (const TensorCPUf &in, float  alpha)
{ const int N = size();  CHECK_EQ (N, in.size());
  cblas_saxpy (N, alpha, in.dptr, 1, dptr, 1);
};
template <>
void TensorCPUd::blas_axpy (const TensorCPUd &in, double alpha)
{ const int N = size();  CHECK_EQ (N, in.size());
  cblas_daxpy (N, alpha, in.dptr, 1, dptr, 1);
};
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_copy_from (const float *x, const int incx, const int incy)
{ const int N = size();
  cuda_check (
  cublasScopy (get_blas_handle(), N, x, incx, dptr, incy));
};
template <>
void TensorGPUf::blas_copy_to   (float *x, const int incx, const int incy) const
{ const int N = size();
  cuda_check (
  cublasScopy (get_blas_handle(), N, dptr, incy, x, incx));
};
#else
template <>
void TensorCPUf::blas_copy_from (const float *x, const int incx, const int incy)
{ const int N = size();
  cblas_scopy (N, x, incx, dptr, incy);
};
template <>
void TensorCPUf::blas_copy_to   (float *x, const int incx, const int incy) const
{ const int N = size();
  cblas_scopy (N, dptr, incy, x, incx);
};
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_sdot  (const TensorGPUf &in, float  &val) const
{ const int N = size();  CHECK_EQ (N, in.size());
  cuda_check (
  cublasSdot (get_blas_handle(), N, in.dptr, 1, dptr, 1, &val));
};
template <>
void TensorGPUd::blas_sdot  (const TensorGPUd &in, double &val) const
{ const int N = size();  CHECK_EQ (N, in.size());
  cuda_check (
  cublasDdot (get_blas_handle(), N, in.dptr, 1, dptr, 1, &val));
};
#else
template <>
void TensorCPUf::blas_sdot  (const TensorCPUf &in, float  &val) const
{ const int N = size();  CHECK_EQ (N, in.size());
  val = cblas_sdot (N, in.dptr, 1, dptr, 1);
};
template <>
void TensorCPUd::blas_sdot  (const TensorCPUd &in, double &val) const
{ const int N = size();  CHECK_EQ (N, in.size());
  val = cblas_ddot (N, in.dptr, 1, dptr, 1);
};
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_nrm2 (float  &val) const
{ const int N = size();
  cuda_check (
  cublasSnrm2 (get_blas_handle(), N, dptr, 1, &val));
};
template <>
void TensorGPUd::blas_nrm2 (double &val) const
{ const int N = size();
  cuda_check (
  cublasDnrm2 (get_blas_handle(), N, dptr, 1, &val));
};
#else
template <>
void TensorCPUf::blas_nrm2 (float  &val) const
{ const int N = size();
  val = cblas_snrm2 (N, dptr, 1);
};
template <>
void TensorCPUd::blas_nrm2 (double &val) const
{ const int N = size();
  val = cblas_dnrm2 (N, dptr, 1);
};
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_scal (float  alpha)
{ const int N = size();
  cuda_check (
  cublasSscal (get_blas_handle(), N, &alpha, dptr, 1));
};
template <>
void TensorGPUd::blas_scal (double alpha)
{ const int N = size();
  cuda_check (
  cublasDscal (get_blas_handle(), N, &alpha, dptr, 1));
};
#else
template <>
void TensorCPUf::blas_scal (float  alpha)
{ const int N = size();
  cblas_sscal (N, alpha, dptr, 1);
};
template <>
void TensorCPUd::blas_scal (double alpha)
{ const int N = size();
  cblas_dscal (N, alpha, dptr, 1);
};
#endif

#endif
