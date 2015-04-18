#ifndef NNET_CONVOLUTION_
#define NNET_CONVOLUTION_

#include "../include/nnet.h"

#ifdef __CUDACC__
template LayerConvolution<GPU>::LayerConvolution (ParaLayer &pl, const int did, TensorGPUf &src, TensorGPUf &dst);
#else
template LayerConvolution<CPU>::LayerConvolution (ParaLayer &pl, const int did, TensorCPUf &src, TensorCPUf &dst);
#endif

LAYER_FORWARD (LayerConvolution)
{ if (is_train && !pl_.isFixed)
  { mwmat_.reduce_mean     (wmat_, 4);
    wmat_.broadcast_minus (mwmat_, 4);
  }
  if (is_train && !pl_.isFixed && pl_.isVarN)
  { nwmat_.reduce_var      (wmat_, 4);  nwmat_.add (1e-8f);  nwmat_.blas_vsqrt (nwmat_);
    wmat_.broadcast_div   (nwmat_, 4);
    wmat_.broadcast_mul    (scal_, 4);
  }
#if USE_CUDNN
  cuda_check (cudnnConvolutionForward (dnnctx[did_]->cudnn_, 
    &alpha,  srcDesc_,  src_.dptr, wmatDesc_, wmat_.dptr, convDesc_, algo_, workspace, worksize,
    &beta,   dstDesc_,  dst_.dptr));
  cuda_check (cudnnAddTensor (dnnctx[did_]->cudnn_, CUDNN_ADD_SAME_C,
    &alpha, biasDesc_, bias_.dptr,
    &alpha,  dstDesc_,  dst_.dptr));
#else
  for (int i = 0; i < nums_; ++i)
  { tsrc_ = src_[i];
    tdst_ = dst_[i];
    tsrc_.im2col_fprop (patch_, tcol_);

    tdst_.shape.set_cols (drep_.rows());
    tdst_.shape.set_chls (1);
    tdst_.blas_gemm (false, true,  bias_, drep_, 1, 0);  // Y = B * 1.t()
    tdst_.blas_gemm (false, false, wmat_, tcol_, 1, 1);  // Y += W * X
  }
#endif
}

LAYER_BACKPROP (LayerConvolution)
{ gwmat_.mem_set(0);
  gbias_.mem_set(0);
#if USE_CUDNN
  cuda_check (cudnnConvolutionBackwardBias   (dnnctx[did_]->cudnn_,
    &alpha,  dstDesc_,   dst_.dptr, 
    &beta,  biasDesc_, gbias_.dptr));
  cuda_check (cudnnConvolutionBackwardFilter (dnnctx[did_]->cudnn_,
    &alpha,  srcDesc_,   src_.dptr, dstDesc_, dst_.dptr, convDesc_,
    &beta,  wmatDesc_, gwmat_.dptr));
  if (is_prop_grad)
  cuda_check (cudnnConvolutionBackwardData   (dnnctx[did_]->cudnn_,
    &alpha, wmatDesc_,  wmat_.dptr, dstDesc_, dst_.dptr, convDesc_,
    &beta,   srcDesc_,   src_.dptr));
#else
  for (int i = 0; i < nums_; ++i)
  { tsrc_ = src_[i];
    tdst_ = dst_[i];
    tsrc_.im2col_fprop (patch_, tcol_);

    tdst_.shape.set_cols (drep_.rows());
    tdst_.shape.set_chls (1);
    gbias_.blas_gemv (false, tdst_, drep_, 1, 1);  // dB += dY * 1
    gwmat_.blas_gemm (false, true, tdst_, tcol_, 1, 1);  // dW += dY * X.t()

    if (is_prop_grad)
    { tcol_.blas_gemm (true, false, wmat_, tdst_, 1, 0);  // dX = W.t() * dY
      tsrc_.col2im_bprop (patch_, tcol_);
    }
  }
#endif
  if (pl_.isVarN)
  { iwmat_.reduce_sum_product (wmat_, gwmat_, 4);
    iwmat_.broadcast_div   ( scal_, 4);
  }
  { mwmat_.reduce_mean     (gwmat_, 4);
    gwmat_.broadcast_minus (mwmat_, 4);
  }
  if (pl_.isVarN)
  { gwmat_.broadcast_mul   ( scal_, 4);
    gwmat_.broadcast_minus_product (wmat_, iwmat_, 4);
    gwmat_.broadcast_div   (nwmat_, 4);
    gscal_.copy (iwmat_);
  }
}

LAYER_INIT (LayerConvolution)
{ nums_ = src_.nums();
  chls_ = src_.chls();
  flts_ = pl_.flts;
  grps_ = pl_.grps;

  patch_ = Patch (pl_.ksize, pl_.pad, pl_.stride);
  Shape col_shape = patch_.get_pack_size (src_[0].shape);
  Shape dst_shape (patch_.h_col, patch_.w_col, flts_, nums_);
  Shape dim_shape (patch_.h_col* patch_.w_col, 1, 1, 1);

   dst_.create (dst_shape, did_);
  drep_.create (dim_shape, did_);
  drep_.constant (1);

#if USE_CUDNN
  cuda_check (cudnnCreateTensorDescriptor  (&srcDesc_));
  cuda_check (cudnnCreateTensorDescriptor  (&dstDesc_));
  cuda_check (cudnnCreateTensorDescriptor (&biasDesc_));
  cuda_check (cudnnCreateFilterDescriptor (&wmatDesc_));

  src_.setTensor4dDescriptor (srcDesc_);
  dst_.setTensor4dDescriptor (dstDesc_);
#else
  tcol_.create (col_shape, did_);
#endif
}

template <typename XPU>
void LayerConvolution<XPU>::init_model ()
{ Shape wmat_shape (pl_.ksize, pl_.ksize, chls_, flts_);
  Shape bias_shape (1, 1, flts_, 1);
//Shape wmat_shape (flts_, pl_.ksize * pl_.ksize * chls_, 1, 1);
//Shape bias_shape (flts_, 1, 1, 1);

   wmat_.create (wmat_shape, did_);
  gwmat_.create (wmat_shape, did_);
   bias_.create (bias_shape, did_);
  gbias_.create (bias_shape, did_);
   scal_.create (bias_shape, did_);
  gscal_.create (bias_shape, did_);

  mwmat_.create (bias_shape, did_);
  nwmat_.create (bias_shape, did_);
  iwmat_.create (bias_shape, did_);

  wmat_.random (rand_, pl_.random, 0.f, pl_.sigma);
  bias_.constant (pl_.bias);
  scal_.constant (pl_.scale);
#if USE_CUDNN
  bias_.setTensor4dDescriptor (biasDesc_);
  wmat_.setFilter4dDescriptor (wmatDesc_);
  cuda_check (cudnnCreateConvolutionDescriptor (&convDesc_));
  cuda_check (cudnnSetConvolution2dDescriptor  ( convDesc_, pl_.pad, pl_.pad, pl_.stride, pl_.stride, 1, 1,
    CUDNN_CROSS_CORRELATION));
  cuda_check (cudnnGetConvolutionForwardAlgorithm     (dnnctx[did_]->cudnn_, srcDesc_, wmatDesc_, convDesc_, dstDesc_,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 10e6, &algo_));
  cuda_check (cudnnGetConvolutionForwardWorkspaceSize (dnnctx[did_]->cudnn_, srcDesc_, wmatDesc_, convDesc_, dstDesc_,
    algo_, &worksize));
  cuda_malloc ((void**)&workspace, worksize);
#endif
}

template <typename XPU>
void LayerConvolution<XPU>::save_model (const string file)
{ wmat_.save (file+"_wmat");
  bias_.save (file+"_bias");
}

template <typename XPU>
void LayerConvolution<XPU>::load_model (const string file)
{ if (pl_.isLoad)
  { wmat_.load (file+"_wmat", did_);
    bias_.load (file+"_bias", did_);
  }
}

template <typename XPU>
void LayerConvolution<XPU>::show_model ()
{ wmat_.show_image ();
}

template <typename XPU>
void LayerConvolution<XPU>::set_optimization (ParaOptim &paraWmat, ParaOptim &paraBias, vector<OptimBase<XPU, float>*> &optims)
{ optims.push_back (create_optim (paraWmat, did_, wmat_, gwmat_));
  optims.push_back (create_optim (paraBias, did_, bias_, gbias_));
//optims.push_back (create_optim (paraBias, did_, scal_, gscal_));
}

#endif
