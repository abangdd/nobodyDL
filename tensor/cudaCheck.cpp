#ifndef CUDA_CHECK_
#define CUDA_CHECK_

#include <glog/logging.h>
#include "../include/xpu.h"

#ifdef __CUDACC__
void GPU::check_sync (const char* msg) {
    cudaError_t err = cudaGetLastError();
    CHECK_EQ (err, cudaSuccess) << "\tkernel error " << msg << "\t" << cudaGetErrorString(err);
}

void GPU::check_async (const char* msg) {
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_EQ (err, cudaSuccess) << "\tkernel error " << msg << "\t" << cudaGetErrorString(cudaGetLastError());
}

template <typename T>
void cuda_check (const T& status) {
    // static_cast<unsigned int>(status)
    if (status) {
        LOG (FATAL) << cuda_get_status (status);
//    DEVICE_RESET
    }
}
template void cuda_check (const CUresult& status);
template void cuda_check (const cudaError_t& status);
template void cuda_check (const cublasStatus_t& status);
template void cuda_check (const curandStatus_t& status);
template void cuda_check (const cusparseStatus_t& status);
template void cuda_check (const cudnnStatus_t& status);
template void cuda_check (const ncclResult_t& status);

inline const char* cuda_get_status (const CUresult& status) {
    const char** pStr = (const char**)calloc(128, sizeof(char*));
    cuGetErrorName (status, pStr);
    return *pStr;
}

inline const char* cuda_get_status (const cudaError_t& status) {
    return cudaGetErrorString (status);
}

const char* cuda_get_status (const cublasStatus_t& status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "cublas_status_success";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "cublas_status_not_initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "cublas_status_alloc_failed";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "cublas_status_invalid_value";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "cublas_status_arch_mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "cublas_status_mapping_error";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "cublas_status_execution_failed";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "cublas_status_internal_error";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "cublas_status_not_supported";
        default:
            return "<unknown>";
    }
}

const char* cuda_get_status (const curandStatus_t& status) {
    switch (status) {
        case CURAND_STATUS_SUCCESS:
            return "curand_status_success";
        case CURAND_STATUS_VERSION_MISMATCH:
            return "curand_status_version_mismatch";
        case CURAND_STATUS_NOT_INITIALIZED:
            return "curand_status_not_initialized";
        case CURAND_STATUS_ALLOCATION_FAILED:
            return "curand_status_allocation_failed";
        case CURAND_STATUS_TYPE_ERROR:
            return "curand_status_type_error";
        case CURAND_STATUS_OUT_OF_RANGE:
            return "curand_status_out_of_range";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "curand_status_length_not_multiple";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "curand_status_double_precision_required";
        case CURAND_STATUS_LAUNCH_FAILURE:
            return "curand_status_launch_failure";
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "curand_status_preexisting_failure";
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "curand_status_initialization_failed";
        case CURAND_STATUS_ARCH_MISMATCH:
            return "curand_status_arch_mismatch";
        case CURAND_STATUS_INTERNAL_ERROR:
            return "curand_status_internal_error";
        default:
            return "<unknown>";
    }
}

const char* cuda_get_status (const cusparseStatus_t& status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS:
            return "cusparse_status_success";
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "cusparse_status_not_initialized";
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "cusparse_status_alloc_failed";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "cusparse_status_invalid_value";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "cusparse_status_arch_mismatch";
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "cusparse_status_mapping_error";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "cusparse_status_execution_failed";
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "cusparse_status_internal_error";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "cusparse_status_matrix_type_not_supported";
        default:
            return "<unknown>";
    }
}

const char* cuda_get_status (const cudnnStatus_t& status) {
    return cudnnGetErrorString (status);
}

const char* cuda_get_status (const ncclResult_t& status) {
    return ncclGetErrorString (status);
}

#else

template <typename T>
void mkl_check (const T& status) {
    if (status)
        LOG (FATAL) << mkl_get_status (status);
}
template void mkl_check (const sparse_status_t& status);

const char* mkl_get_status (const sparse_status_t& status) {
    switch (status) {
        case SPARSE_STATUS_NOT_INITIALIZED:
            return "sparse_status_not_initialized";
        case SPARSE_STATUS_ALLOC_FAILED:
            return "sparse_status_alloc_failed";
        case SPARSE_STATUS_INVALID_VALUE:
            return "sparse_status_invalid_value";
        case SPARSE_STATUS_EXECUTION_FAILED:
            return "sparse_status_execution_failed";
        case SPARSE_STATUS_INTERNAL_ERROR:
            return "sparse_status_internal_error";
        case SPARSE_STATUS_NOT_SUPPORTED:
            return "sparse_status_not_supported";
        default:
            return "<unknown>";
    }
}
#endif

#endif
