#ifndef NNET_POOLING_
#define NNET_POOLING_

#include "../include/nnet.h"
using std::max;
using std::min;

enum pool_t
{ MAX	= 1,
  AVE	= 2
};

#ifdef __CUDACC__
void ParaLayer::setPoolingDesc (cudnnPoolingDescriptor_t &desc)
{ cuda_check (cudnnSetPooling2dDescriptor (desc, pool == AVE ?
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_MAX, ksize, ksize, pad, pad, stride, stride));
}

template LayerPooling<GPU>::LayerPooling (ParaLayer &pl, const int did, TensorGPUf &src, TensorGPUf &dst);
#else
template LayerPooling<CPU>::LayerPooling (ParaLayer &pl, const int did, TensorCPUf &src, TensorCPUf &dst);
#endif

LAYER_FORWARD (LayerPooling)
{ 
#ifdef __CUDACC__
  cuda_check (cudnnPoolingForward  (CUDNN_HANDLE, poolDesc_, 
    &alpha, srcDesc_, src_.dptr,
    &beta,  dstDesc_, dst_.dptr));
#endif
  if (is_train)
    tdst_.copy (dst_);
}

LAYER_BACKPROP (LayerPooling)
{ 
#ifdef __CUDACC__
  if (is_prop_grad)
  for (int i = 0; i < secs_; ++i)
  { const int s1 = secn_ * i, s2 = secn_ * (i+1);
    if (pl_.pool == MAX)
      tsrc_.copy (src_.section(s1, s2));
    cuda_check (cudnnPoolingBackward (CUDNN_HANDLE, poolDesc_,
    &alpha, sdstDesc_, tdst_.section(s1, s2).dptr,
            sdstDesc_,  dst_.section(s1, s2).dptr,
            ssrcDesc_, tsrc_                .dptr,
    &beta,  ssrcDesc_,  src_.section(s1, s2).dptr));
  }
#endif
}

LAYER_INIT (LayerPooling)
{ secs_ = pl_.pool == MAX ? 4 : 1;  // TODO
  secn_ = src_.nums() / secs_;

  kernal_ = Kernal (pl_.ksize, pl_.pad, pl_.stride);
  Shape dst_shape = kernal_.get_pool_size (src_.shape);
  Shape sec_shape = src_.section(0, secn_).shape;

  if (pl_.pool == MAX)
    tsrc_.create (sec_shape, did_);
  else
    tsrc_ = src_;
  tdst_.create (dst_shape, did_);
   dst_.create (dst_shape, did_);
#ifdef __CUDACC__
  cuda_check (cudnnCreateTensorDescriptor  (& srcDesc_));
  cuda_check (cudnnCreateTensorDescriptor  (&ssrcDesc_));
  cuda_check (cudnnCreateTensorDescriptor  (& dstDesc_));
  cuda_check (cudnnCreateTensorDescriptor  (&sdstDesc_));
  cuda_check (cudnnCreatePoolingDescriptor (&poolDesc_));

  src_.setTensor4dDesc (srcDesc_);
  dst_.setTensor4dDesc (dstDesc_);
   pl_.setPoolingDesc (poolDesc_);
  src_.section(0, secn_).setTensor4dDesc (ssrcDesc_);
  dst_.section(0, secn_).setTensor4dDesc (sdstDesc_);
#endif
}

#endif
