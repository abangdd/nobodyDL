#ifndef NNET_LOSS_
#define NNET_LOSS_

#include "../include/nnet.h"
using std::max;
using std::min;

enum loss_t
{ ENTROPY	= 1,
  EUCLIDEAN	= 2,
  LOGISTIC	= 3
};

#ifdef __CUDACC__
template LayerLoss<GPU>::LayerLoss (ParaLayer &pl, const int did, TensorGPUf &src, TensorGPUf &dst);
#else
template LayerLoss<CPU>::LayerLoss (ParaLayer &pl, const int did, TensorCPUf &src, TensorCPUf &dst);
#endif

template <typename DT>
XPU_KERNEL(SigmoidForward) (
  const int num_kernels, const DT* src_data, DT* dst_data)
{ kernel_for (i, num_kernels)
    dst_data[i] = (DT)1. / ((DT)1. + exp(min(-src_data[i], (DT)32.)));
};

LAYER_FORWARD (LayerLoss)
{ const int N = dst_.size();
  float  sasum = 0;
  switch (pl_.loss)
  { case ENTROPY:  // p(c|x)
#ifdef __CUDACC__
      cuda_check (cudnnSoftmaxForward (CUDNN_HANDLE, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, srcDesc_, src_.dptr,
       	&beta,  dstDesc_, src_.dptr));
#else
      for (int i = 0; i < src_.nums(); ++i)
        src_[i].softmax ();
#endif
      break;
    case EUCLIDEAN:
      break;
    case LOGISTIC:  // p(c|x)
      XPU_KERNEL_LAUNCH (SigmoidForward,  cuda_get_blocks(N), CUDA_NUM_THREADS, 0, CUDNN_STREAM,
      N, src_.dptr, src_.dptr);
      cuda_sync_check ("SigmoidForward");
      break;
    default:
      LOG (FATAL) << "not implemented loss method";
  }
  src_.blas_asum (sasum);
  if (isnan (sasum))
    LOG (FATAL) << "\tXPU  " << did_ << "\tsasum is\tnan";
  if (fabs (sasum) < 1e-8)
    LOG (FATAL) << "\tXPU  " << did_ << "\tsasum is\t0";
}

LAYER_BACKPROP (LayerLoss)
{ float  gasum = 0;
  switch (pl_.loss)
  { case ENTROPY:  // p(c|x) - 1(y == c)
      src_.blas_axpy (dst_, -1);
      break;
    case EUCLIDEAN:
      src_.blas_axpy (dst_, -1);
      break;
    case LOGISTIC:  // dp/dx = p*(1-p)
      src_.blas_axpy (dst_, -1);
      break;
    default:
      LOG (FATAL) << "not implemented loss method";
  }
  src_.blas_scal (1.f/nums_);
  src_.blas_asum (gasum);
  if (isnan (gasum))
    LOG (FATAL) << "\tXPU  " << did_ << "\tgasum is\tnan";
  if (fabs (gasum) < 1e-8)
    LOG (FATAL) << "\tXPU  " << did_ << "\tgasum is\t0";
}

LAYER_INIT (LayerLoss)
{ nums_ = src_.nums();
  dims_ = src_.size() / nums_;
#ifdef __CUDACC__
  cuda_check (cudnnCreateTensorDescriptor (&srcDesc_));
  cuda_check (cudnnCreateTensorDescriptor (&dstDesc_));
  src_.setTensor4dDesc (srcDesc_);
  dst_.setTensor4dDesc (dstDesc_);
#endif
}

#endif
