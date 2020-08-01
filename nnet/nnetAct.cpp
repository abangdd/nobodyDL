#ifndef NNET_ACT_
#define NNET_ACT_

#include "../include/nnet.h"

#ifdef __CUDACC__
template LayerAct<GPU>::LayerAct (ParaLayer& pl, const int did, TensorGPUf& src, TensorGPUf& dst);
#else
template LayerAct<CPU>::LayerAct (ParaLayer& pl, const int did, TensorCPUf& src, TensorCPUf& dst);
#endif

template <typename XPU>
void LayerAct<XPU>::fprop (const bool is_train, const bool is_fixed) {
#ifdef __CUDACC__
    if (pl_.isNorm && is_train && !is_fixed) {
        cuda_check (cudnnBatchNormalizationForwardTraining (CUDNN_HANDLE, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            srcDesc_,  src_.dptr,
            srcDesc_,  mid_.dptr,
            nrmDesc_, wmat_.dptr, bias_.dptr, 1e-1, mavg_.dptr, mvar_.dptr, 1e-5, savg_.dptr, idev_.dptr));
        svar_.blas_vinv (idev_);
        svar_.blas_vsqr (svar_);
    }
    else if (pl_.isNorm && is_train)
        cuda_check (cudnnBatchNormalizationForwardInference (CUDNN_HANDLE, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            srcDesc_,  src_.dptr,
            srcDesc_,  mid_.dptr,
            nrmDesc_, wmat_.dptr, bias_.dptr, savg_.dptr, svar_.dptr, 1e-5));
    else if (pl_.isNorm)  // inference
        cuda_check (cudnnBatchNormalizationForwardInference (CUDNN_HANDLE, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            srcDesc_,  src_.dptr,
            srcDesc_,  mid_.dptr,
            nrmDesc_, wmat_.dptr, bias_.dptr, mavg_.dptr, mvar_.dptr, 1e-5));
#endif
    if (pl_.act == 1)
        dst_.relu_fprop (mid_);
    if (pl_.drop > 0 && is_train)
        dst_.drop_chls (mask_, pl_.drop);
}

template <typename XPU>
void LayerAct<XPU>::bprop (const bool is_prop_grad) {
    if (pl_.drop > 0 && is_prop_grad)
        dst_.drop_chls (mask_, pl_.drop);
    if (pl_.act == 1)
        mid_.relu_bprop (dst_);
#ifdef __CUDACC__
    if (pl_.isNorm) {
        cuda_check (cudnnBatchNormalizationBackward (CUDNN_HANDLE, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta, &alpha, &beta,
            srcDesc_,  src_.dptr,  // x
            srcDesc_,  mid_.dptr,  // dy
            srcDesc_,  mid_.dptr,  // dx
            nrmDesc_, wmat_.dptr, gmat_.dptr, gias_.dptr, 1e-5, savg_.dptr, idev_.dptr));
        cuda_check (cudnnAddTensor (CUDNN_HANDLE,
            &alpha, srcDesc_,  mid_.dptr,
            &beta,  srcDesc_,  src_.dptr));
    }
#endif
    if (pl_.drop > 0)
        mask_.init (rand_, UNIFORM, 0.f, 1.f);
}

template <typename XPU>
void LayerAct<XPU>::init_layer () {
    nums_ = src_.nums();
    chls_ = src_.chls();

    if (pl_.drop > 0) {
        mask_.create (Shape (1, 1, chls_, nums_), did_);
        mask_.init (rand_, UNIFORM, 0.f, 1.f);
    }
    if (pl_.act >= 1) {
        dst_.create (src_.shape, (float*)dnnCtx[did_].reluWorkAddr, did_);
        CHECK_LE (dst_.size_d(), dnnCtx[did_].reluWorkSize);
    }
    if (pl_.isNorm) {
        mid_.create (src_.shape, (float*)dnnCtx[did_].normWorkAddr, did_);
        CHECK_LE (mid_.size_d(), dnnCtx[did_].normWorkSize);
    }
    else
        mid_ = src_;
#ifdef __CUDACC__
    cuda_check (cudnnCreateTensorDescriptor (&srcDesc_));
    cuda_check (cudnnCreateTensorDescriptor (&dstDesc_));
    cuda_check (cudnnCreateTensorDescriptor (&nrmDesc_));

    src_.setTensor4dDesc (srcDesc_);
    dst_.setTensor4dDesc (dstDesc_);
#endif
}

template <typename XPU>
void LayerAct<XPU>::init_model () {
    Shape scal_shape (1, 1, chls_, 1);
    
    wmat_.create (scal_shape, did_);
    gmat_.create (scal_shape, did_);
    bias_.create (scal_shape, did_);
    gias_.create (scal_shape, did_);

    mavg_.create (scal_shape, did_);
    savg_.create (scal_shape, did_);
    mvar_.create (scal_shape, did_);
    svar_.create (scal_shape, did_);
    idev_.create (scal_shape, did_);

    wmat_.init (1.);
    bias_.init (0.01);
    mvar_.init (0.01);
    mavg_.init (0.);
#ifdef __CUDACC__
    wmat_.setTensor4dDesc (nrmDesc_);
#endif
}

template <typename XPU>
void LayerAct<XPU>::save_model (const string file) {
    if (pl_.isNorm) {
        wmat_.save_bin (file+"_scal");
        bias_.save_bin (file+"_lift");
    }
}

template <typename XPU>
void LayerAct<XPU>::load_model (const string file) {
    if (pl_.isNorm && pl_.isLoad) {
        wmat_.load_bin (file+"_scal", did_);
        bias_.load_bin (file+"_lift", did_);
    }
}

template <typename XPU>
void LayerAct<XPU>::set_optimization (ParaOptim& paraNorm, ParaOptim& paraBias, vector<std::shared_ptr<OptimBase<XPU, float>>>& optims) {
    if (pl_.isNorm && !pl_.isFixed) {
        optims.push_back (create_optim (paraNorm, did_, wmat_, gmat_));
        optims.push_back (create_optim (paraBias, did_, bias_, gias_));
    }
}

#endif
