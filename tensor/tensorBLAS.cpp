#ifndef TENSOR_BLAS_
#define TENSOR_BLAS_

#include "../include/tensor.h"

#ifdef __CUDACC__
inline cublasOperation_t cublas_get_trans (bool t) { return t ? CUBLAS_OP_T : CUBLAS_OP_N; }
#else
inline CBLAS_TRANSPOSE    cblas_get_trans (bool t) { return t ? CblasTrans : CblasNoTrans; }
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_gemm (const bool transA, const bool transB, const TensorGPUf& A, const TensorGPUf& B, float alpha, float beta) {
    const int M = (!transA) ? A.rows() : A.cols();    CHECK_EQ (M, rows());
    const int N = (!transB) ? B.cols() : B.rows();    CHECK_EQ (N, cols());
    const int K = (!transA) ? A.cols() : A.rows();
    const int lda = (!transA) ? K : M;    // A.cols() : A.rows();
    const int ldb = (!transB) ? N : K;    // B.cols() : B.rows();
    cuda_check (cublasSgemm (get_blas_handle(), cublas_get_trans (transB), cublas_get_trans (transA), N, M, K,
        &alpha, B.dptr, ldb, A.dptr, lda, &beta, dptr, N));
};
template <>
void TensorGPUd::blas_gemm (const bool transA, const bool transB, const TensorGPUd& A, const TensorGPUd& B, double alpha, double beta) {
    const int M = (!transA) ? A.rows() : A.cols();    CHECK_EQ (M, rows());
    const int N = (!transB) ? B.cols() : B.rows();    CHECK_EQ (N, cols());
    const int K = (!transA) ? A.cols() : A.rows();
    const int lda = (!transA) ? K : M;
    const int ldb = (!transB) ? N : K;
    cuda_check (cublasDgemm (get_blas_handle(), cublas_get_trans (transB), cublas_get_trans (transA), N, M, K,
        &alpha, B.dptr, ldb, A.dptr, lda, &beta, dptr, N));
};
#else
template <>
void TensorCPUf::blas_gemm (const bool transA, const bool transB, const TensorCPUf& A, const TensorCPUf& B, float alpha, float beta) {
    const int M = (!transA) ? A.rows() : A.cols();    CHECK_EQ (M, rows());
    const int N = (!transB) ? B.cols() : B.rows();    CHECK_EQ (N, cols());
    const int K = (!transA) ? A.cols() : A.rows();
    const int lda = (!transA) ? K : M;
    const int ldb = (!transB) ? N : K;
    cblas_sgemm (CblasRowMajor, cblas_get_trans (transA), cblas_get_trans (transB), M, N, K,
        alpha, A.dptr, lda, B.dptr, ldb, beta, dptr, N);
};
template <>
void TensorCPUd::blas_gemm (const bool transA, const bool transB, const TensorCPUd& A, const TensorCPUd& B, double alpha, double beta) {
    const int M = (!transA) ? A.rows() : A.cols();    CHECK_EQ (M, rows());
    const int N = (!transB) ? B.cols() : B.rows();    CHECK_EQ (N, cols());
    const int K = (!transA) ? A.cols() : A.rows();
    const int lda = (!transA) ? K : M;
    const int ldb = (!transB) ? N : K;
    cblas_dgemm (CblasRowMajor, cblas_get_trans (transA), cblas_get_trans (transB), M, N, K,
        alpha, A.dptr, lda, B.dptr, ldb, beta, dptr, N);
};
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_gemv (const bool transA, const TensorGPUf& A, const TensorGPUf& X, float alpha, float beta) {
    const int M = A.rows();
    const int N = A.cols();
    const int lda = N;
    CHECK_EQ (transA ? N:M, rows());
    CHECK_EQ (size(), rows());
    cuda_check (cublasSgemv (get_blas_handle(), cublas_get_trans (!transA), N, M, &alpha, A.dptr, lda, X.dptr, 1, &beta, dptr, 1));
};
template <>
void TensorGPUd::blas_gemv (const bool transA, const TensorGPUd& A, const TensorGPUd& X, double alpha, double beta) {
    const int M = A.rows();
    const int N = A.cols();
    const int lda = N;
    CHECK_EQ (transA ? N:M, rows());
    CHECK_EQ (size(), rows());
    cuda_check (cublasDgemv (get_blas_handle(), cublas_get_trans (!transA), N, M, &alpha, A.dptr, lda, X.dptr, 1, &beta, dptr, 1));
};
#else
template <>
void TensorCPUf::blas_gemv (const bool transA, const TensorCPUf& A, const TensorCPUf& X, float alpha, float beta) {
    const int M = A.rows();
    const int N = A.cols();
    const int lda = N;
    CHECK_EQ (transA ? N:M, rows());
    CHECK_EQ (size(), rows());
    cblas_sgemv (CblasRowMajor, cblas_get_trans ( transA), M, N, alpha, A.dptr, lda, X.dptr, 1, beta, dptr, 1);
};
template <>
void TensorCPUd::blas_gemv (const bool transA, const TensorCPUd& A, const TensorCPUd& X, double alpha, double beta) {
    const int M = A.rows();
    const int N = A.cols();
    const int lda = N;
    CHECK_EQ (transA ? N:M, rows());
    CHECK_EQ (size(), rows());
    cblas_dgemv (CblasRowMajor, cblas_get_trans ( transA), M, N, alpha, A.dptr, lda, X.dptr, 1, beta, dptr, 1);
};
#endif



#ifdef __CUDACC__
template <>
float  TensorGPUf::blas_asum () const {
    float val;
    cuda_check (cublasSasum (get_blas_handle(), size(), dptr, 1, &val));
    return val;
};
template <>
double TensorGPUd::blas_asum () const {
    double val;
    cuda_check (cublasDasum (get_blas_handle(), size(), dptr, 1, &val));
    return val;
};
template <>
float  TensorGPUf::blas_nrm2 () const {
    float val;
    cuda_check (cublasSnrm2 (get_blas_handle(), size(), dptr, 1, &val));
    return val;
};
template <>
double TensorGPUd::blas_nrm2 () const {
    double val;
    cuda_check (cublasDnrm2 (get_blas_handle(), size(), dptr, 1, &val));
    return val;
};
#else
template <>
float  TensorCPUf::blas_asum () const {
    return cblas_sasum (size(), dptr, 1);
};
template <>
double TensorCPUd::blas_asum () const {
    return cblas_dasum (size(), dptr, 1);
};
template <>
float  TensorCPUf::blas_nrm2 () const {
    return cblas_snrm2 (size(), dptr, 1);
};
template <>
double TensorCPUd::blas_nrm2 () const {
    return cblas_dnrm2 (size(), dptr, 1);
};
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_amax (int& idx, float& val) const {
    const int N = size();
    cuda_check (cublasIsamax (get_blas_handle(), N, dptr, 1, &idx));
    idx = idx - 1;
    val = dptr[idx];
};
template <>
void TensorGPUd::blas_amax (int& idx, double& val) const {
    const int N = size();
    cuda_check (cublasIdamax (get_blas_handle(), N, dptr, 1, &idx));
    idx = idx - 1;
    val = dptr[idx];
};
template <>
void TensorGPUf::blas_amin (int& idx, float& val) const {
    const int N = size();
    cuda_check (cublasIsamin (get_blas_handle(), N, dptr, 1, &idx));
    idx = idx - 1;
    val = dptr[idx];
};
template <>
void TensorGPUd::blas_amin (int& idx, double& val) const {
    const int N = size();
    cuda_check (cublasIdamin (get_blas_handle(), N, dptr, 1, &idx));
    idx = idx - 1;
    val = dptr[idx];
};
#else
template <>
void TensorCPUf::blas_amax (int& idx, float& val) const {
    const int N = size();
    idx = cblas_isamax (N, dptr, 1);
    val = dptr[idx];
};
template <>
void TensorCPUd::blas_amax (int& idx, double& val) const {
    const int N = size();
    idx = cblas_idamax (N, dptr, 1);
    val = dptr[idx];
};
template <>
void TensorCPUf::blas_amin (int& idx, float& val) const {
    const int N = size();
    idx = cblas_isamin (N, dptr, 1);
    val = dptr[idx];
};
template <>
void TensorCPUd::blas_amin (int& idx, double& val) const {
    const int N = size();
    idx = cblas_idamin (N, dptr, 1);
    val = dptr[idx];
};
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_axpy (const TensorGPUf& in, float alpha) {
    CHECK_EQ (size(), in.size());
    cuda_check (cublasSaxpy (get_blas_handle(), size(), &alpha, in.dptr, 1, dptr, 1));
};
template <>
void TensorGPUd::blas_axpy (const TensorGPUd& in, double alpha) {
    CHECK_EQ (size(), in.size());
    cuda_check (cublasDaxpy (get_blas_handle(), size(), &alpha, in.dptr, 1, dptr, 1));
};
template <>
void TensorGPUf::blas_sdot (const TensorGPUf& in, float& val) const {
    CHECK_EQ (size(), in.size());
    cuda_check (cublasSdot (get_blas_handle(), size(), in.dptr, 1, dptr, 1, &val));
};
template <>
void TensorGPUd::blas_sdot (const TensorGPUd& in, double& val) const {
    CHECK_EQ (size(), in.size());
    cuda_check (cublasDdot (get_blas_handle(), size(), in.dptr, 1, dptr, 1, &val));
};
#else
template <>
void TensorCPUf::blas_axpy (const TensorCPUf& in, float alpha) {
    CHECK_EQ (size(), in.size());
    cblas_saxpy (size(), alpha, in.dptr, 1, dptr, 1);
};
template <>
void TensorCPUd::blas_axpy (const TensorCPUd& in, double alpha) {
    CHECK_EQ (size(), in.size());
    cblas_daxpy (size(), alpha, in.dptr, 1, dptr, 1);
};
template <>
void TensorCPUf::blas_sdot (const TensorCPUf& in, float& val) const {
    CHECK_EQ (size(), in.size());
    val = cblas_sdot (size(), in.dptr, 1, dptr, 1);
};
template <>
void TensorCPUd::blas_sdot (const TensorCPUd& in, double& val) const {
    CHECK_EQ (size(), in.size());
    val = cblas_ddot (size(), in.dptr, 1, dptr, 1);
};
#endif



#ifdef __CUDACC__
template <>
void TensorGPUf::blas_scal (float alpha) {
    cuda_check (cublasSscal (get_blas_handle(), size(), &alpha, dptr, 1));
};
template <>
void TensorGPUd::blas_scal (double alpha) {
    cuda_check (cublasDscal (get_blas_handle(), size(), &alpha, dptr, 1));
};
#else
template <>
void TensorCPUf::blas_scal (float alpha) {
    cblas_sscal (size(), alpha, dptr, 1);
};
template <>
void TensorCPUd::blas_scal (double alpha) {
    cblas_dscal (size(), alpha, dptr, 1);
};
#endif

#endif
