#ifndef CUDA_BASE_
#define CUDA_BASE_

#include <map>
#include <vector>
#include "../include/xpu.h"

#ifndef __CUDACC__
std::vector<XPUCtx*> dnnctx;

void XPUCtx::release ()
{ printf ("Device\t%d\treleasing\n",    did_);
  cuda_check (cudaSetDevice (did_));
//cuda_check (cublasShutdown ());
  cuda_check (cublasDestroy (cublas_));
  cuda_check (cudnnDestroy  (cudnn_));
  cuda_check (curandDestroyGenerator (curand_));
  cuda_check (cudaStreamDestroy (stream_));
  cuda_check (cudaEventDestroy  (accept_));
}

void XPUCtx::reset ()
{ printf ("Device\t%d\tinitializing\n", did_);
  cuda_check (cudaSetDevice (did_));
  cuda_check (cudaDeviceReset ());

//cuda_check (cublasInit ());
  cuda_check (cublasCreate (&cublas_));
  cuda_check (cudnnCreate  (&cudnn_));
  cuda_check (curandCreateGenerator (&curand_, CURAND_RNG_PSEUDO_DEFAULT));
  cuda_check (cudaStreamCreate (&stream_));
  cuda_check (cudaEventCreateWithFlags (&accept_, cudaEventDisableTiming));

//cuda_check (cublasSetKernelStream (stream_));
  cuda_check (cublasSetStream (cublas_, stream_));
  cuda_check (curandSetStream (curand_, stream_));
  cuda_check (cudnnSetStream  (cudnn_,  stream_));
  cuda_check (curandSetPseudoRandomGeneratorSeed (curand_, rand()));
}


void cuda_set_p2p (const int num_device)
{ for (int did = 0; did < num_device; ++did)
  { cuda_set_device (did);
    for (int pid = 0; pid < num_device; ++pid)
      if (pid != did)
      { dnnctx[did]->cup2p_[pid] = 0;
        cuda_check (cudaDeviceCanAccessPeer (&dnnctx[did]->cup2p_[pid], did, pid));
        if (dnnctx[did]->cup2p_[pid])
          cuda_check (cudaDeviceEnablePeerAccess (pid, 0));
      } else
        dnnctx[did]->cup2p_[pid] = 1;
  }
}

void cuda_del_p2p (const int num_device)
{ for (int did = 0; did < num_device; ++did)
  { cuda_set_device (did);
    for (int pid = 0; pid < num_device; ++pid)
      if (pid != did)
      { cuda_check (cudaDeviceCanAccessPeer (&dnnctx[did]->cup2p_[pid], did, pid));
        if (dnnctx[did]->cup2p_[pid])
          cuda_check (cudaDeviceDisablePeerAccess (pid));
      }
  }
}

void cuda_set_device (const int did)
{ int curr_device;
  cuda_check (cudaGetDevice (&curr_device));
  if (curr_device == did)
    return;
  cuda_check (cudaSetDevice (did));
}

void cuda_stream_sync(const int did)
{ cuda_check (cudaStreamSynchronize (dnnctx[did]->stream_));
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

void cuda_memcpy       (void *dst, const void *src, const size_t size, enum memcpy_t kind)
{ cuda_check (cudaMemcpy      ((void*)dst, (const void*)src, size, get_memcpy_type(kind)));
}
void cuda_memcpy_async (void *dst, const void *src, const size_t size, enum memcpy_t kind, cudaStream_t stream)
{ cuda_check (cudaMemcpyAsync ((void*)dst, (const void*)src, size, get_memcpy_type(kind), stream));
}
void cuda_memcpy_peer       (void *dst, const void *src, const size_t size, const int dst_id, const int src_id)
{ cuda_check (cudaMemcpyPeer      ((void*)dst, dst_id, (const void*)src, src_id, size));
}
void cuda_memcpy_peer_async (void *dst, const void *src, const size_t size, const int dst_id, const int src_id)
{ cuda_check (cudaMemcpyPeerAsync ((void*)dst, dst_id, (const void*)src, src_id, size, dnnctx[dst_id]->stream_));
}

#else
void cuda_malloc (void **ptr, const size_t len)
{
#if CUDA_MANAGED
  cuda_check (cudaMallocManaged (ptr, len, cudaMemAttachGlobal));
#else
  cuda_check (cudaMalloc (ptr, len));
#endif
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
