#ifndef NNET_DROPOUT_
#define NNET_DROPOUT_

#include "../include/nnet.h"

#ifdef __CUDACC__
template LayerDropout<GPU>::LayerDropout (ParaLayer &pl, const int did, TensorGPUf &src, TensorGPUf &dst);
#else
template LayerDropout<CPU>::LayerDropout (ParaLayer &pl, const int did, TensorCPUf &src, TensorCPUf &dst);
#endif

template <typename DT>
XPU_KERNEL(kernel_dropout) (const int num_kernels, const DT *mask, DT threshold, const DT scale, DT *data)
{ kernel_for (i, num_kernels)
    data[i] *= (mask[i] > threshold) * scale;
}

LAYER_FORWARD (LayerDropout)
{ scal_ = 1 / (1 - pl_.drop);
  const int N = dst_.size();
  if (is_train && pl_.drop > 0.01)
  { mask.init (rand_, UNIFORM, 0.f, 1.f);
    XPU_KERNEL_LAUNCH (kernel_dropout, cuda_get_blocks(N), CUDA_NUM_THREADS, 0, CUDNN_STREAM,
      N, mask.dptr, pl_.drop, scal_, dst_.dptr);
    cuda_sync_check ("DropoutForward");
  }
}

LAYER_BACKPROP (LayerDropout)
{ scal_ = 1 / (1 - pl_.drop);
  const int N = src_.size();
  if (is_prop_grad && pl_.drop > 0.01)
  { XPU_KERNEL_LAUNCH (kernel_dropout, cuda_get_blocks(N), CUDA_NUM_THREADS, 0, CUDNN_STREAM,
      N, mask.dptr, pl_.drop, scal_, src_.dptr);
    cuda_sync_check ("DropoutBackward");
  }
}

LAYER_INIT (LayerDropout)
{ dst_ = src_;
  mask.create (src_.shape, did_);
}

#endif
