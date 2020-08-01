#ifndef XPU_H_
#define XPU_H_

#include <omp.h>
#include <stdio.h>
#include <sys/types.h>

#include <cmath>
#include <string>
#include <vector>

using std::string;
using std::vector;

#include <glog/logging.h>
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_vsl.h>
#include <mkl_vsl_functions.h>

#define CUDA_NUM_THREADS 1024
#define CUDA_NUM_STREAMS 8
#define CUDA_NUM_DEVICES 4

#ifdef __CUDACC__
    #ifdef __CUDA_ARCH__
        #define HEMI_DEV_CODE
    #endif

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

    #if !defined(HEMI_ALIGN)
        #define HEMI_ALIGN(n)		__align__(n)
    #endif

    #define XPU_INLINE			__host__ __device__ inline
    #define XPU_GET_ELEMENT_OFFSET	blockDim.x * blockIdx.x + threadIdx.x
    #define XPU_GET_ELEMENT_STRIDE	blockDim.x * gridDim.x
#else
    #define XPU_KERNEL(name)		void name
    #define XPU_KERNEL_LAUNCH(name, gridDim, blockDim, sharedBytes, streamId, ...) name(__VA_ARGS__)

    #define XPU_INLINE			inline
    #define XPU_GET_ELEMENT_OFFSET	0
    #define XPU_GET_ELEMENT_STRIDE	1
#endif

#define kernel_for(i, n) \
    _Pragma ("omp parallel for") \
    for (int i = XPU_GET_ELEMENT_OFFSET; i < n; i += XPU_GET_ELEMENT_STRIDE)

#ifdef __CUDACC__
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse.h>
#include <driver_types.h>
#include <nccl.h>

template <typename DT>
struct SharedMemory {
    __device__ DT* getPointer() {
        return (DT*)0;
    }
};
template <>
struct SharedMemory <float> {
    __device__ float* getPointer() {
        extern __shared__ float s_float[];
        return s_float;
    }
};
template <>
struct SharedMemory <double> {
    __device__ double* getPointer() {
        extern __shared__ double s_double[];
        return s_double;
    }
};

template <typename DT>
ncclDataType_t nccl_type ();
template <typename DT>
cudnnDataType_t cudnn_type ();
#endif

template <typename DT>
XPU_KERNEL(kprint) (const int knum, const DT* x);

template <typename T>
void mkl_check (const T& status);
template <typename T>
void cuda_check (const T& status);
void cuda_set_p2p (const int num_device);
void cuda_del_p2p (const int num_device);

class GPU {
public:
    void* operator new (size_t len);
    void operator delete (void* ptr);
    static int get_blocks (const int N) { return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS; }
    static void set_device (const int did);
    static void check_sync (const char* msg);
    static void check_async (const char* msg);
    static void sync_stream (const int did);
};

class CPU {
public:
    static int get_blocks (const int N) { return 1; }
    static void set_device (const int did) { }
    static void check_sync (const char* msg) { }
    static void check_async (const char* msg) { }
    static void sync_stream (const int did) { }
};

template <typename XPU>
class Device {
public:
    explicit Device (const int did) : did_(did) { }
    void init ();
    void release ();  // 不直接析构
    void* algoWorkAddr = nullptr;
    void* convWorkAddr = nullptr;
    void* normWorkAddr = nullptr;
    void* reluWorkAddr = nullptr;
    size_t algoWorkSize = 0;
    size_t convWorkSize = 0;
    size_t normWorkSize = 0;
    size_t reluWorkSize = 0;
    int cup2p_[CUDA_NUM_DEVICES];
    int did_;
#ifdef __CUDACC__
    cudaEvent_t accept_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    curandGenerator_t curand_ = nullptr;
    cusparseHandle_t cusparse_ = nullptr;
    cudnnHandle_t cudnn_ = nullptr;
    ncclComm_t cucomm_ = nullptr;
#endif
};

template <typename XPU>
class DNNCtx {
public:
    void init (const int min_device, const int max_device);
    void release ();  // 不直接析构
    int dnums () { return dnum_; }
    Device<XPU>& operator[] (const int idx) { return ctx_.at(idx-dmin_); }
private:
    int dmin_, dmax_, dnum_;
    vector<Device<XPU>> ctx_;
};

extern DNNCtx<GPU> dnnCtx;

#endif
