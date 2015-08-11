#ifndef XPU_H_
#define XPU_H_

#include <omp.h>
#include <stdio.h>
#include <sys/types.h>
#include <vector>
  #include <cuda.h>
  #include <driver_types.h>
  #include <cublas_v2.h>
  #include <cusparse.h>
  #include <curand.h>
  #include <cudnn.h>
  #include <npp.h>
  #include <nppdefs.h>

  #include <math.h>
  #include <mkl.h>
  #include <mkl_cblas.h>
  #include <mkl_vsl.h>
  #include <mkl_vsl_functions.h>

#define CUDA_NUM_THREADS 1024
#define CUDA_NUM_DEVICES 2

#ifdef __CUDACC__

  #ifdef __CUDA_ARCH__
    #define HEMI_DEV_CODE
  #endif

  #define CUDA_MANAGED true

  #define XPU_KERNEL(name)		__global__ void name ## _kernel

  #if defined(DEBUG) || defined(_DEBUG)
    #define XPU_KERNEL_LAUNCH(name, gridDim, blockDim, sharedBytes, streamId, ...) \
      do { \
        name ## _kernel<<< (gridDim), (blockDim), (sharedBytes), (streamId) >>>(__VA_ARGS__); \
        cudaAsyncCheck (name); \
      } while(0)
  #else
    #define XPU_KERNEL_LAUNCH(name, gridDim, blockDim, sharedBytes, streamId, ...) \
      name ## _kernel<<< (gridDim) , (blockDim), (sharedBytes), (streamId) >>>(__VA_ARGS__)
  #endif

  #define XPU_CALLABLE			__host__ __device__
  #define XPU_CALLABLE_INLINE		__host__ __device__ inline

  #if !defined(HEMI_ALIGN)
    #define HEMI_ALIGN(n)		__align__(n)
  #endif

  #define XPU_GET_ELEMENT_OFFSET	blockDim.x * blockIdx.x + threadIdx.x
  #define XPU_GET_ELEMENT_STRIDE	blockDim.x * gridDim.x

#else
  #define CUDA_MANAGED false
  #define XPU_KERNEL(name)		void name
  #define XPU_KERNEL_LAUNCH(name, gridDim, blockDim, sharedBytes, streamId, ...) name(__VA_ARGS__)

  #define XPU_CALLABLE
  #define XPU_CALLABLE_INLINE		inline

  #define XPU_GET_ELEMENT_OFFSET	0
  #define XPU_GET_ELEMENT_STRIDE	1

#endif

#define kernel_for(i, n) \
  _Pragma ("omp parallel for") \
  for (int i = XPU_GET_ELEMENT_OFFSET; i < n; i += XPU_GET_ELEMENT_STRIDE)

void cuda_nervana_load (const char* const base_path);
void cuda_nervana_unload ();
void cuda_set_p2p (const int num_device);
void cuda_del_p2p (const int num_device);
void cuda_set_device (const int did);
void cuda_stream_sync(const int did);
int  cuda_get_blocks (const int N);

class XPUCtx {
public:
  explicit XPUCtx (const int did) : did_(did) { }
  void reset ();
  void release ();
  int  did_;
  cudaEvent_t       accept_ = nullptr;
  cudaStream_t      stream_ = nullptr;
  cublasHandle_t    cublas_ = nullptr;
  curandGenerator_t curand_ = nullptr;
  cudnnHandle_t     cudnn_  = nullptr;
  int cup2p_[CUDA_NUM_DEVICES];
};

template <typename T>
void cuda_check (const T &status);
void cuda_sync_check  (const char *msg);
void cuda_async_check (const char *msg);

void cuda_malloc (void **ptr, const size_t len);
void cuda_memcpy       (void *dst, const void *src, const size_t size, cudaMemcpyKind kind);
void cuda_memcpy_async (void *dst, const void *src, const size_t size, cudaMemcpyKind kind, cudaStream_t stream);
void cuda_memcpy_peer       (void *dst, const void *src, const size_t size, const int dst_id, const int src_id);
void cuda_memcpy_peer_async (void *dst, const void *src, const size_t size, const int dst_id, const int src_id);

template <typename DT>
struct SharedMemory
{ __device__ DT*     getPointer() { return (DT*)0;  }
};
template <>
struct SharedMemory <float>
{ __device__ float * getPointer() { extern __shared__ float  s_float [];  return s_float;  }
};
template <>
struct SharedMemory <double>
{ __device__ double* getPointer() { extern __shared__ double s_double[];  return s_double;  }
};

class GPU {
public:
  void *operator new (size_t len);
  void operator delete(void *ptr);
};

class CPU {
};

extern std::vector<XPUCtx*> dnnctx;

#endif
