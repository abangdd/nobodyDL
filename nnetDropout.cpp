#ifndef NNET_DROPOUT_
#define NNET_DROPOUT_

#include "../include/nnet.h"

#ifdef __CUDACC__
template LayerDropout<GPU>::LayerDropout (ParaLayer &pl, const int did, TensorGPUf &src, TensorGPUf &dst);
#else
template LayerDropout<CPU>::LayerDropout (ParaLayer &pl, const int did, TensorCPUf &src, TensorCPUf &dst);
#endif

template <typename DT>
XPU_KERNEL(DropoutForward)  (const int num_kernels, const DT *src, const DT *mask, DT threshold, const DT scale, DT *dst)
{ kernel_for (i, num_kernels)
    dst[i] = src[i] * (mask[i] > threshold) * scale;
}

template <typename DT>
XPU_KERNEL(DropoutBackward) (const int num_kernels, const DT *dst, const DT *mask, DT threshold, const DT scale, DT *src)
{ kernel_for (i, num_kernels)
    src[i] = dst[i] * (mask[i] > threshold) * scale;
}

LAYER_FORWARD (LayerDropout)
{ const int N = dst_.size();
  if (is_train && drop_ >= 0.1f)
  { mask.init (rand_, UNIFORM, 0.f, 1.f);
    XPU_KERNEL_LAUNCH (DropoutForward,  cuda_get_blocks(N), CUDA_NUM_THREADS, 0, CUDNN_STREAM,
      N, src_.dptr, mask.dptr, drop_, scal_, dst_.dptr);
    cuda_sync_check ("DropoutForward");
  } else
    dst_.copy (src_);
}

LAYER_BACKPROP (LayerDropout)
{ const int N = dst_.size();
  if (is_prop_grad && drop_ >= 0.1f)
  { XPU_KERNEL_LAUNCH (DropoutBackward, cuda_get_blocks(N), CUDA_NUM_THREADS, 0, CUDNN_STREAM,
      N, dst_.dptr, mask.dptr, drop_, scal_, src_.dptr);
    cuda_sync_check ("DropoutBackward");
  } else
    src_.copy (dst_);
}

LAYER_INIT (LayerDropout)
{ dst_.create (src_.shape, did_);
  mask.create (src_.shape, did_);
  drop_ = pl_.dropout;  CHECK (drop_ > 0 && drop_ < 1);
  scal_ = 1. / (1. - drop_);
}

#endif
