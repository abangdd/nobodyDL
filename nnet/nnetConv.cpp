#ifndef NNET_CONV_
#define NNET_CONV_

#include "../include/nnet.h"

#ifdef __CUDACC__
template LayerConv<GPU>::LayerConv (ParaLayer& pl, const int did, TensorGPUf& src, TensorGPUf& dst);
#else
template LayerConv<CPU>::LayerConv (ParaLayer& pl, const int did, TensorCPUf& src, TensorCPUf& dst);
#endif

template <typename XPU>
void LayerConv<XPU>::fprop (const bool is_train, const bool is_fixed) {
#ifdef __CUDACC__
    for (int g = 0; g < grps_; g++) {
        if (g < grid_) {
            cuda_check (cudnnConvolutionForward (CUDNN_HANDLE,
                &alpha, srcDesc_,  tsrc_[chlg_*g].dptr,
                        wmatDesc_, wmat_[fltg_*g].dptr, convDesc_[g], fwdDataAlgo[g], dnnCtx[did_].algoWorkAddr, dnnCtx[did_].algoWorkSize,
                &beta,  dstDesc_,  tdst_[fltg_*g].dptr));
            if (!pl_.isNorm)
            cuda_check (cudnnAddTensor (CUDNN_HANDLE,
                &alpha, biasDesc_, bias_[fltg_*g].dptr,
                &alpha, dstDesc_,  tdst_[fltg_*g].dptr));
        }
        else
            cuda_check (cudnnAddTensor (CUDNN_HANDLE,
                &alpha, srcDesc_,  tsrc_[chlg_*g].dptr,
                &beta,  dstDesc_,  tdst_[fltg_*g].dptr));
    }
#else
    for (int i = 0; i < nums_; ++i) {
        tsrc_ = src_[i];
        tdst_ = dst_[i];
        tsrc_.im2col_fprop (kernal_, tcol_);

        tdst_.shape.set_cols (drep_.rows());
        tdst_.shape.set_chls (1);
        tdst_.blas_gemm (false, true,  bias_, drep_, 1, 0);  // Y = B * 1.t()
        tdst_.blas_gemm (false, false, wmat_, tcol_, 1, 1);  // Y += W * X
    }
#endif
}

template <typename XPU>
void LayerConv<XPU>::bprop (const bool is_prop_grad) {
    gmat_.mem_set(0);
    gias_.mem_set(0);
#ifdef __CUDACC__
    for (int g = 0; g < grps_; g++) {
        if (g < grid_) {
            if (!pl_.isNorm)
            cuda_check (cudnnConvolutionBackwardBias (CUDNN_HANDLE,
                &alpha, dstDesc_,  tdst_[fltg_*g].dptr, 
                &beta,  biasDesc_, gias_[fltg_*g].dptr));
            cuda_check (cudnnConvolutionBackwardFilter (CUDNN_HANDLE,
                &alpha, srcDesc_,  tsrc_[chlg_*g].dptr,
                        dstDesc_,  tdst_[fltg_*g].dptr, convDesc_[g], bwdFltrAlgo[g], dnnCtx[did_].algoWorkAddr, dnnCtx[did_].algoWorkSize,
                &beta,  wmatDesc_, gmat_[fltg_*g].dptr));
            if (is_prop_grad)
            cuda_check (cudnnConvolutionBackwardData (CUDNN_HANDLE,
                &alpha, wmatDesc_, wmat_[fltg_*g].dptr,
                        dstDesc_,  tdst_[fltg_*g].dptr, convDesc_[g], bwdDataAlgo[g], dnnCtx[did_].algoWorkAddr, dnnCtx[did_].algoWorkSize,
                &beta,  srcDesc_,  tsrc_[chlg_*g].dptr));
        }
        else
            cuda_check (cudnnAddTensor (CUDNN_HANDLE,
                &alpha, dstDesc_,  tdst_[fltg_*g].dptr,
                &beta,  srcDesc_,  tsrc_[chlg_*g].dptr));
    }
#else
    for (int i = 0; i < nums_; ++i) {
        tsrc_ = src_[i];
        tdst_ = dst_[i];
        tsrc_.im2col_fprop (kernal_, tcol_);

        tdst_.shape.set_cols (drep_.rows());
        tdst_.shape.set_chls (1);
        gias_.blas_gemv (false, tdst_, drep_, 1, 1);  // dB += dY * 1
        gmat_.blas_gemm (false, true, tdst_, tcol_, 1, 1);  // dW += dY * X.t()

        if (is_prop_grad) {
            tcol_.blas_gemm (true, false, wmat_, tdst_, 1, 0);  // dX = W.t() * dY
            tsrc_.col2im_bprop (kernal_, tcol_);
        }
    }
#endif
}

template <typename XPU>
void LayerConv<XPU>::init_layer () {
    nums_ = src_.nums();
    chls_ = src_.chls();
    flts_ = pl_.flts;
    chlg_ = chls_ / pl_.grps;
    fltg_ = flts_ / pl_.grps;

    grps_ = pl_.grps;
    grid_ = std::max (grps_ / 2, 1);  // TODO
    dims_ = pl_.krow * pl_.kcol * chls_ / grps_;

    CHECK_EQ (chls_ % grps_, 0);
    CHECK_EQ (flts_ % grps_, 0);

    kernal_ = Kernal (pl_.krow, pl_.kcol, pl_.strd);
    Shape col_shape = kernal_.get_pack_size (src_[0].shape);
    Shape dst_shape (kernal_.h_col, kernal_.w_col, flts_, nums_);
    Shape dim_shape (dims_, 1, 1, 1);

    if (pl_.isShared) {
        dst_.create (dst_shape, (float*)dnnCtx[did_].convWorkAddr, did_);
        CHECK_LE (dst_.size_d(), dnnCtx[did_].convWorkSize);
    }
    else
        dst_.create (dst_shape, did_);

    tsrc_ = src_[0];
    tdst_ = dst_[0];

    drep_.create (dim_shape, did_);
    drep_.init (1.f);
#ifdef __CUDACC__
    cuda_check (cudnnCreateTensorDescriptor (&srcDesc_));
    cuda_check (cudnnCreateTensorDescriptor (&dstDesc_));
    cuda_check (cudnnCreateTensorDescriptor (&biasDesc_));
    cuda_check (cudnnCreateFilterDescriptor (&wmatDesc_));

    src_.setTensor4dDesc (srcDesc_, grps_);
    dst_.setTensor4dDesc (dstDesc_, grps_);
#else
    tcol_.create (col_shape, did_);
#endif
}

template <typename XPU>
void LayerConv<XPU>::init_model () {
    pl_.get_model_info ();

    Shape wmat_shape (pl_.krow, pl_.kcol, chls_/grps_, flts_/grps_*grid_);
    Shape bias_shape (1, 1, flts_/grps_*grid_, 1);

    wmat_.create (wmat_shape, did_);
    gmat_.create (wmat_shape, did_);
    bias_.create (bias_shape, did_);
    gias_.create (bias_shape, did_);

    pl_.sigma = sqrt (1./dims_);
    wmat_.init (rand_, GAUSSIAN, 0.f, pl_.sigma);
    bias_.init (pl_.bias);
#ifdef __CUDACC__
    cuda_check (cudnnSetTensor4dDescriptor (biasDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, flts_/grps_, 1, 1));
    cuda_check (cudnnSetFilter4dDescriptor (wmatDesc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, flts_/grps_, chls_/grps_, pl_.krow, pl_.kcol));
    convDesc_.resize(grid_);
    fwdDataAlgo.resize(grid_);
    bwdDataAlgo.resize(grid_);
    bwdFltrAlgo.resize(grid_);
    fwdDataSize.resize(grid_);
    bwdDataSize.resize(grid_);
    bwdFltrSize.resize(grid_);
    for (int g = 0; g < grid_; g++) {
        const int hole = pow (pl_.hole, g);
        const int pad_h = hole * (pl_.krow-1) / 2;
        const int pad_w = hole * (pl_.kcol-1) / 2;
        cuda_check (cudnnCreateConvolutionDescriptor (&convDesc_[g]));
        cuda_check (cudnnSetConvolution2dDescriptor (convDesc_[g], pad_h, pad_w, pl_.strd, pl_.strd, hole, hole,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        int retAlgoCount = 10;
        cudnnConvolutionFwdAlgoPerf_t fwdDataPerf[10];
        cudnnConvolutionBwdDataAlgoPerf_t bwdDataPerf[10];
        cudnnConvolutionBwdFilterAlgoPerf_t bwdFltrPerf[10];
        cuda_check (cudnnFindConvolutionForwardAlgorithm        (CUDNN_HANDLE, srcDesc_, wmatDesc_, convDesc_[g], dstDesc_, 10, &retAlgoCount, fwdDataPerf));
        cuda_check (cudnnFindConvolutionBackwardDataAlgorithm   (CUDNN_HANDLE, wmatDesc_, dstDesc_, convDesc_[g], srcDesc_, 10, &retAlgoCount, bwdDataPerf));
        cuda_check (cudnnFindConvolutionBackwardFilterAlgorithm (CUDNN_HANDLE, srcDesc_, dstDesc_, convDesc_[g], wmatDesc_, 10, &retAlgoCount, bwdFltrPerf));

        for (int i = 9; i >= 0; --i) {
            if (fwdDataPerf[i].memory < 1073741824)
                fwdDataAlgo[g] = fwdDataPerf[i].algo;
            if (bwdDataPerf[i].memory < 1073741824)
                bwdDataAlgo[g] = bwdDataPerf[i].algo;
            if (bwdFltrPerf[i].memory < 1073741824 && bwdFltrPerf[i].algo != CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 && bwdFltrPerf[i].algo != CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)
                bwdFltrAlgo[g] = bwdFltrPerf[i].algo;
        }

        cuda_check (cudnnGetConvolutionForwardWorkspaceSize (CUDNN_HANDLE, srcDesc_, wmatDesc_, convDesc_[g], dstDesc_,
            fwdDataAlgo[g], &fwdDataSize[g]));
        cuda_check (cudnnGetConvolutionBackwardDataWorkspaceSize (CUDNN_HANDLE, wmatDesc_, dstDesc_, convDesc_[g], srcDesc_,
            bwdDataAlgo[g], &bwdDataSize[g]));
        cuda_check (cudnnGetConvolutionBackwardFilterWorkspaceSize (CUDNN_HANDLE, srcDesc_, dstDesc_, convDesc_[g], wmatDesc_,
            bwdFltrAlgo[g], &bwdFltrSize[g]));

        const int workSize = std::max (std::max (fwdDataSize[g], bwdDataSize[g]), bwdFltrSize[g]);
        if (workSize > dnnCtx[did_].algoWorkSize) {
            dnnCtx[did_].algoWorkSize = workSize;
            cuda_check (cudaFree (dnnCtx[did_].algoWorkAddr));
            cuda_check (cudaMallocManaged (&dnnCtx[did_].algoWorkAddr, dnnCtx[did_].algoWorkSize, cudaMemAttachGlobal));
        }
    }
#endif
}

template <typename XPU>
void LayerConv<XPU>::save_model (const string file) {
    if (!pl_.isFixed) {
        wmat_.save_bin (file+"_wmat");
        bias_.save_bin (file+"_bias");
    }
}

template <typename XPU>
void LayerConv<XPU>::load_model (const string file) {
    if (pl_.isLoad) {
        wmat_.load_bin (file+"_wmat", did_);
        bias_.load_bin (file+"_bias", did_);
    }
}

template <typename XPU>
void LayerConv<XPU>::set_optimization (ParaOptim& paraWmat, ParaOptim& paraBias, vector<std::shared_ptr<OptimBase<XPU, float>>>& optims) {
    if (!pl_.isFixed) {
        optims.push_back (create_optim (paraWmat, did_, wmat_, gmat_));
        optims.push_back (create_optim (paraBias, did_, bias_, gias_));
    }
}
#endif
