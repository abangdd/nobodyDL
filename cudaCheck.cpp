#ifndef CUDA_CHECK_
#define CUDA_CHECK_

#include <glog/logging.h>
#include "../include/xpu.h"

#ifdef __CUDACC__
void cuda_sync_check  (const char *msg)
{ cudaError_t err = cudaGetLastError();
  CHECK_EQ (err, cudaSuccess) << "\tkernel error " << msg << "\t" << cudaGetErrorString(err);
}

void cuda_async_check (const char *msg)
{ cudaError_t err = cudaDeviceSynchronize();
  CHECK_EQ (err, cudaSuccess) << "\tkernel error " << msg << "\t" << cudaGetErrorString(cudaGetLastError());
}

template <typename T>
void cuda_check (const T &status)
{ // static_cast<unsigned int>(status)
  if (status)
  { LOG (FATAL) << cuda_get_status (status);
//  DEVICE_RESET
  }
}
template void cuda_check (const CUresult         &status);
template void cuda_check (const cudaError_t      &status);
template void cuda_check (const cublasStatus_t   &status);
template void cuda_check (const curandStatus_t   &status);
template void cuda_check (const cusparseStatus_t &status);
template void cuda_check (const cudnnStatus_t    &status);
template void cuda_check (const NppStatus        &status);

inline const char *cuda_get_status (const CUresult    &status)
{ const char** pStr = (const char**)calloc(128, sizeof(char*));  // TODO
  cuGetErrorName (status, pStr);
  return *pStr;
}

inline const char *cuda_get_status (const cudaError_t &status)
{ return cudaGetErrorString (status);
}

const char *cuda_get_status (const cublasStatus_t &status)
{ switch (status)
  { case CUBLAS_STATUS_SUCCESS:
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

const char *cuda_get_status (const curandStatus_t &status)
{ switch (status)
  { case CURAND_STATUS_SUCCESS:
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

const char *cuda_get_status (const cusparseStatus_t &status)
{ switch (status)
  { case CUSPARSE_STATUS_SUCCESS:
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

const char *cuda_get_status (const cudnnStatus_t &status)
{ return cudnnGetErrorString (status);
}

const char *cuda_get_status (const NppStatus &status)
{ switch (status)
  { case NPP_NOT_SUPPORTED_MODE_ERROR:
      return "NPP_NOT_SUPPORTED_MODE_ERROR";
    case NPP_INVALID_HOST_POINTER_ERROR:
      return "NPP_INVALID_HOST_POINTER_ERROR";
    case NPP_INVALID_DEVICE_POINTER_ERROR:
      return "NPP_INVALID_DEVICE_POINTER_ERROR";
    case NPP_LUT_PALETTE_BITSIZE_ERROR:
      return "NPP_LUT_PALETTE_BITSIZE_ERROR";
    case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";
    case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
      return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";
    case NPP_TEXTURE_BIND_ERROR:
      return "NPP_TEXTURE_BIND_ERROR";
    case NPP_WRONG_INTERSECTION_ROI_ERROR:
      return "NPP_WRONG_INTERSECTION_ROI_ERROR";
    case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
      return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";
    case NPP_MEMFREE_ERROR:
      return "NPP_MEMFREE_ERROR";
    case NPP_MEMSET_ERROR:
      return "NPP_MEMSET_ERROR";
    case NPP_MEMCPY_ERROR:
      return "NPP_MEMCPY_ERROR";
    case NPP_ALIGNMENT_ERROR:
      return "NPP_ALIGNMENT_ERROR";
    case NPP_CUDA_KERNEL_EXECUTION_ERROR:
      return "NPP_CUDA_KERNEL_EXECUTION_ERROR";
    case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
      return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";
    case NPP_QUALITY_INDEX_ERROR:
      return "NPP_QUALITY_INDEX_ERROR";
    case NPP_RESIZE_NO_OPERATION_ERROR:
      return "NPP_RESIZE_NO_OPERATION_ERROR";
    case NPP_OVERFLOW_ERROR:
      return "NPP_OVERFLOW_ERROR";
    case NPP_NOT_EVEN_STEP_ERROR:
      return "NPP_NOT_EVEN_STEP_ERROR";
    case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
      return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";
    case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
      return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";
    case NPP_CHANNEL_ORDER_ERROR:
      return "NPP_CHANNEL_ORDER_ERROR";
    case NPP_ZERO_MASK_VALUE_ERROR:
      return "NPP_ZERO_MASK_VALUE_ERROR";
    case NPP_QUADRANGLE_ERROR:
      return "NPP_QUADRANGLE_ERROR";
    case NPP_RECTANGLE_ERROR:
      return "NPP_RECTANGLE_ERROR";
    case NPP_COEFFICIENT_ERROR:
      return "NPP_COEFFICIENT_ERROR";
    case NPP_NUMBER_OF_CHANNELS_ERROR:
      return "NPP_NUMBER_OF_CHANNELS_ERROR";
    case NPP_COI_ERROR:
      return "NPP_COI_ERROR";
    case NPP_DIVISOR_ERROR:
      return "NPP_DIVISOR_ERROR";
    case NPP_CHANNEL_ERROR:
      return "NPP_CHANNEL_ERROR";
    case NPP_STRIDE_ERROR:
      return "NPP_STRIDE_ERROR";
    case NPP_ANCHOR_ERROR:
      return "NPP_ANCHOR_ERROR";
    case NPP_MASK_SIZE_ERROR:
      return "NPP_MASK_SIZE_ERROR";
    case NPP_RESIZE_FACTOR_ERROR:
      return "NPP_RESIZE_FACTOR_ERROR";
    case NPP_INTERPOLATION_ERROR:
      return "NPP_INTERPOLATION_ERROR";
    case NPP_MIRROR_FLIP_ERROR:
      return "NPP_MIRROR_FLIP_ERROR";
    case NPP_MOMENT_00_ZERO_ERROR:
      return "NPP_MOMENT_00_ZERO_ERROR";
    case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
      return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";
    case NPP_THRESHOLD_ERROR:
      return "NPP_THRESHOLD_ERROR";
    case NPP_CONTEXT_MATCH_ERROR:
      return "NPP_CONTEXT_MATCH_ERROR";
    case NPP_FFT_FLAG_ERROR:
      return "NPP_FFT_FLAG_ERROR";
    case NPP_FFT_ORDER_ERROR:
      return "NPP_FFT_ORDER_ERROR";
    case NPP_STEP_ERROR:
      return "NPP_STEP_ERROR";
    case NPP_SCALE_RANGE_ERROR:
      return "NPP_SCALE_RANGE_ERROR";
    case NPP_DATA_TYPE_ERROR:
      return "NPP_DATA_TYPE_ERROR";
    case NPP_OUT_OFF_RANGE_ERROR:
      return "NPP_OUT_OFF_RANGE_ERROR";
    case NPP_DIVIDE_BY_ZERO_ERROR:
      return "NPP_DIVIDE_BY_ZERO_ERROR";
    case NPP_MEMORY_ALLOCATION_ERR:
      return "NPP_MEMORY_ALLOCATION_ERR";
    case NPP_NULL_POINTER_ERROR:
      return "NPP_NULL_POINTER_ERROR";
    case NPP_RANGE_ERROR:
      return "NPP_RANGE_ERROR";
    case NPP_SIZE_ERROR:
      return "NPP_SIZE_ERROR";
    case NPP_BAD_ARGUMENT_ERROR:
      return "NPP_BAD_ARGUMENT_ERROR";
    case NPP_NO_MEMORY_ERROR:
      return "NPP_NO_MEMORY_ERROR";
    case NPP_NOT_IMPLEMENTED_ERROR:
      return "NPP_NOT_IMPLEMENTED_ERROR";
    case NPP_ERROR:
      return "NPP_ERROR";
    case NPP_ERROR_RESERVED:
      return "NPP_ERROR_RESERVED";
    case NPP_NO_ERROR:
      return "NPP_NO_ERROR";
    case NPP_NO_OPERATION_WARNING:
      return "NPP_NO_OPERATION_WARNING";
    case NPP_DIVIDE_BY_ZERO_WARNING:
      return "NPP_DIVIDE_BY_ZERO_WARNING";
    case NPP_AFFINE_QUAD_INCORRECT_WARNING:
      return "NPP_AFFINE_QUAD_INCORRECT_WARNING";
    case NPP_WRONG_INTERSECTION_ROI_WARNING:
      return "NPP_WRONG_INTERSECTION_ROI_WARNING";
    case NPP_WRONG_INTERSECTION_QUAD_WARNING:
      return "NPP_WRONG_INTERSECTION_QUAD_WARNING";
    case NPP_DOUBLE_SIZE_WARNING:
      return "NPP_DOUBLE_SIZE_WARNING";
    case NPP_MISALIGNED_DST_ROI_WARNING:
      return "NPP_MISALIGNED_DST_ROI_WARNING";
    default:
      return "<unknown>";
  }
}
#endif

#endif
