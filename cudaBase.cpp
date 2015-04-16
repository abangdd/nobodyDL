#ifndef CUDA_BASE_
#define CUDA_BASE_

#include <glog/logging.h>
#include "../include/xpu.h"

#ifdef __CUDACC__
std::vector<XPUCtx*> dnnctx;

XPUCtx::~XPUCtx()
{ cuda_check (cudaStreamDestroy (stream_));
  cuda_check (cublasDestroy (cublas_));
  cuda_check (curandDestroyGenerator (curand_));
  cuda_check (cudnnDestroy  (cudnn_));
}

void XPUCtx::reset (const int did)
{ LOG (INFO) << "\tDevice\t" << did << "\tinitializing";
  cuda_check (cudaSetDevice (did));
  cuda_check (cudaDeviceReset ());

  cuda_check (cudaStreamCreate (&stream_));
  cuda_check (cublasCreate (&cublas_));
  cuda_check (curandCreateGenerator (&curand_, CURAND_RNG_PSEUDO_DEFAULT));
  cuda_check (curandSetPseudoRandomGeneratorSeed (curand_, rand()));
  cuda_check (cudnnCreate  (&cudnn_));
}

void cuda_set_device (const int did)
{ int curr_device;
  cuda_check (cudaGetDevice (&curr_device));
  if (curr_device == did)
    return;
  cuda_check (cudaSetDevice (did));
}

int cuda_get_blocks (const int N)
{ return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

enum cudaMemcpyKind get_memcpy_type (enum memcpy_t kind)
{ switch (kind)
  { case CPU2GPU:
      return cudaMemcpyHostToDevice;
    case GPU2CPU:
      return cudaMemcpyDeviceToHost;
    case CPU2CPU:
      return cudaMemcpyHostToHost;
    case GPU2GPU:
      return cudaMemcpyDeviceToDevice;
    default:
      return cudaMemcpyHostToDevice;
  }
}

void cuda_malloc (void **ptr, const size_t len)
{
#if CUDA_MANAGED
  cuda_check (cudaMallocManaged (ptr, len));
#else
  cuda_check (cudaMalloc (ptr, len));
#endif
}

void cuda_memcpy       (void *dst, const void *src, const size_t size, enum memcpy_t kind)
{ cuda_check (cudaMemcpy      ((void*)dst, (const void*)src, size, get_memcpy_type(kind)));
}

void cuda_memcpy_async (void *dst, const void *src, const size_t size, enum memcpy_t kind)
{ cudaStream_t stream;
  cuda_check (cudaStreamCreate (&stream));
  cuda_check (cudaMemcpyAsync ((void*)dst, (const void*)src, size, get_memcpy_type(kind), stream));
  cuda_check (cudaStreamDestroy (stream));
}



void *GPU::operator new(size_t len)
{ void *ptr = NULL;
  cuda_malloc (&ptr, len);
  return ptr;
}

void GPU::operator delete(void *ptr)
{ cuda_check (cudaFree ((void*)ptr));
}
#endif

#endif
