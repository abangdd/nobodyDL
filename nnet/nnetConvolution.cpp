#ifndef NNET_CONVOLUTION_
#define NNET_CONVOLUTION_

#include "../include/nnet.h"

#ifdef __CUDACC__
template LayerConvolution<GPU>::LayerConvolution (ParaLayer &pl, const int did, TensorGPUf &src, TensorGPUf &dst);
#else
template LayerConvolution<CPU>::LayerConvolution (ParaLayer &pl, const int did, TensorCPUf &src, TensorCPUf &dst);
#endif

LAYER_FORWARD (LayerConvolution)
{ 
#if USE_CUDNN
  cuda_check (cudnnConvolutionForward (CUDNN_HANDLE,
    &alpha,  srcDesc_,  src_.dptr, wmatDesc_, wmat_.dptr, convDesc_, fwdDataAlgo, fwdDataAddr, fwdDataSize,
    &beta,   dstDesc_,  dst_.dptr));
  cuda_check (cudnnAddTensor (CUDNN_HANDLE, CUDNN_ADD_SAME_C,
    &alpha, biasDesc_, bias_.dptr,
    &alpha,  dstDesc_,  dst_.dptr));
#endif
}

LAYER_BACKPROP (LayerConvolution)
{ gwmat_.mem_set(0);
  gbias_.mem_set(0);
#if USE_CUDNN
  cuda_check (cudnnConvolutionBackwardBias   (CUDNN_HANDLE,
    &alpha,  dstDesc_,   dst_.dptr, 
    &beta,  biasDesc_, gbias_.dptr));
  cuda_check (cudnnConvolutionBackwardFilter_v3 (CUDNN_HANDLE,
    &alpha,  srcDesc_,   src_.dptr, dstDesc_, dst_.dptr, convDesc_, bwdFltrAlgo, bwdFltrAddr, bwdFltrSize,
    &beta,  wmatDesc_, gwmat_.dptr));
  if (is_prop_grad)
  cuda_check (cudnnConvolutionBackwardData_v3   (CUDNN_HANDLE,
    &alpha, wmatDesc_,  wmat_.dptr, dstDesc_, dst_.dptr, convDesc_, bwdDataAlgo, bwdDataAddr, bwdDataSize,
    &beta,   srcDesc_,   src_.dptr));
#endif
}

LAYER_INIT (LayerConvolution)
{ nums_ = src_.nums();
  chls_ = src_.chls();
  flts_ = pl_.flts;
  dims_ = pl_.ksize * pl_.ksize * chls_;

  kernal_ = Kernal (pl_.ksize, pl_.pad, pl_.stride);
  Shape col_shape = kernal_.get_pack_size (src_[0].shape);
  Shape dst_shape (kernal_.h_col, kernal_.w_col, flts_, nums_);
  Shape dim_shape (kernal_.h_col* kernal_.w_col, 1, 1, 1);

   dst_.create (dst_shape, did_);
  drep_.create (dim_shape, did_);
  drep_.init (1.f);

#if USE_CUDNN
  cuda_check (cudnnCreateTensorDescriptor  (&srcDesc_));
  cuda_check (cudnnCreateTensorDescriptor  (&dstDesc_));
  cuda_check (cudnnCreateTensorDescriptor (&biasDesc_));
  cuda_check (cudnnCreateFilterDescriptor (&wmatDesc_));

  src_.setTensor4dDesc (srcDesc_);
  dst_.setTensor4dDesc (dstDesc_);
#else
  tcol_.create (col_shape, did_);
#endif
}

template <typename XPU>
void LayerConvolution<XPU>::init_model ()
{ Shape wmat_shape (pl_.ksize, pl_.ksize, chls_, flts_);
  Shape bias_shape (1, 1, flts_, 1);
//Shape wmat_shape (flts_, dims_, 1, 1);
//Shape bias_shape (flts_,     1, 1, 1);

   wmat_.create (wmat_shape, did_);
  gwmat_.create (wmat_shape, did_);
   bias_.create (bias_shape, did_);
  gbias_.create (bias_shape, did_);

  if (pl_.sigma == 0.f)
    pl_.sigma = std::min (std::max (sqrt (2./dims_), 0.02), 0.06);
  wmat_.init (rand_, GAUSSIAN, 0.f, pl_.sigma);
  bias_.init (0.01);
#if USE_CUDNN
  bias_.setTensor4dDesc (biasDesc_);
  wmat_.setFilter4dDesc (wmatDesc_);
  cuda_check (cudnnCreateConvolutionDescriptor (&convDesc_));
  cuda_check (cudnnSetConvolution2dDescriptor  ( convDesc_, pl_.pad, pl_.pad, pl_.stride, pl_.stride, 1, 1,
    CUDNN_CROSS_CORRELATION));

  fwdDataAlgo = cudnnConvolutionFwdAlgo_t      (1);
  bwdDataAlgo = cudnnConvolutionBwdDataAlgo_t  (1);
  bwdFltrAlgo = cudnnConvolutionBwdFilterAlgo_t(1);
  cuda_check (cudnnGetConvolutionForwardWorkspaceSize        (CUDNN_HANDLE, srcDesc_, wmatDesc_, convDesc_, dstDesc_,
    fwdDataAlgo, &fwdDataSize));
  cuda_check (cudnnGetConvolutionBackwardDataWorkspaceSize   (CUDNN_HANDLE, wmatDesc_, dstDesc_, convDesc_, srcDesc_,
    bwdDataAlgo, &bwdDataSize));
  cuda_check (cudnnGetConvolutionBackwardFilterWorkspaceSize (CUDNN_HANDLE, srcDesc_, dstDesc_, convDesc_, wmatDesc_,
    bwdFltrAlgo, &bwdFltrSize));
  cuda_malloc ((void**)&fwdDataAddr, fwdDataSize);
  cuda_malloc ((void**)&bwdDataAddr, bwdDataSize);
  cuda_malloc ((void**)&bwdFltrAddr, bwdFltrSize);
#endif
}

template <typename XPU>
void LayerConvolution<XPU>::save_model (const string file)
{ Tensor<CPU, float> bt;
  bt.create (wmat_.shape);  bt.copy (wmat_);  bt.save (file+"_wmat");
  bt.create (bias_.shape);  bt.copy (bias_);  bt.save (file+"_bias");
}

template <typename XPU>
void LayerConvolution<XPU>::load_model (const string file)
{ if (pl_.isLoad)
  { wmat_.load (file+"_wmat", did_);
    bias_.load (file+"_bias", did_);
  }
}

template <typename XPU>
void LayerConvolution<XPU>::set_optimization (ParaOptim &paraWmat, ParaOptim &paraBias, vector<OptimBase<XPU, float>*> &optims)
{ 
  optims.push_back (create_optim (paraWmat, did_, wmat_, gwmat_));
  optims.push_back (create_optim (paraBias, did_, bias_, gbias_));
  pl_.get_model_info ();
}
#endif
