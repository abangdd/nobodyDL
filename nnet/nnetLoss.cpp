#ifndef NNET_LOSS_
#define NNET_LOSS_

#include "../include/nnet.h"

enum loss_t {
    CLASSIFICATION = 1,
    SEGMENT = 2,
    EUCLIDEAN = 3
};

#ifdef __CUDACC__
template LayerLoss<GPU>::LayerLoss (ParaLayer& pl, const int did, TensorGPUf& src, TensorGPUf& dst);
#else
template LayerLoss<CPU>::LayerLoss (ParaLayer& pl, const int did, TensorCPUf& src, TensorCPUf& dst);
#endif

template <typename XPU>
void LayerLoss<XPU>::fprop (const bool is_train, const bool is_fixed) {
    float sasum = src_.blas_asum();
    if (sasum >= 1e10)
        LOG (FATAL) << "\tXPU\t" << did_ << "\tsasum is too large";
    if (sasum <= 1e-8)
        LOG (FATAL) << "\tXPU\t" << did_ << "\tsasum is too small";
    if (std::isnan (sasum))
        LOG (FATAL) << "\tXPU\t" << did_ << "\tsasum is nan";
    if (std::isinf (sasum))
        LOG (FATAL) << "\tXPU\t" << did_ << "\tsasum is inf";
    switch (pl_.type) {
        case CLASSIFICATION:  // p(c|x)
            src_.softmax();
            break;
        case SEGMENT:  // p(c|x)
            src_.sigmoid();
            aux_.binary_loss (src_, dst_);
            mid_.reduce_sum (aux_);
            mid_.blas_scal (1.f/area_);
            break;
        case EUCLIDEAN:
            break;
        default:
            LOG (FATAL) << "not implemented loss method";
    }
}

template <typename XPU>
void LayerLoss<XPU>::bprop (const bool is_prop_grad) {
    switch (pl_.type) {
        case CLASSIFICATION:  // p(c|x) - 1(y == c)
            src_.blas_axpy (aux_, -pl_.drop);
            src_.blas_axpy (dst_, +pl_.drop-1);
            break;
        case SEGMENT:
            src_.blas_axpy (dst_, -1);
            break;
        case EUCLIDEAN:
            src_.blas_axpy (dst_, -1);
            break;
        default:
            LOG (FATAL) << "not implemented loss method";
    }
    src_.blas_scal (1.f/nums_/area_);
    float gasum = src_.blas_asum();
    if (gasum >= 1e10)
        LOG (FATAL) << "\tXPU\t" << did_ << "\tgasum is too large";
    if (gasum <= 1e-8)
        LOG (FATAL) << "\tXPU\t" << did_ << "\tgasum is too small";
    if (std::isnan (gasum))
        LOG (FATAL) << "\tXPU\t" << did_ << "\tgasum is nan";
    if (std::isinf (gasum))
        LOG (FATAL) << "\tXPU\t" << did_ << "\tgasum is inf";
}

template <typename XPU>
void LayerLoss<XPU>::init_layer () {
    chls_ = src_.chls();
    dims_ = src_.dims();
    nums_ = src_.nums();
    area_ = src_.area();

    Shape eval_shape (1, 1, chls_, nums_);
    mid_.create (eval_shape, did_);  // TODO
    dst_.create (src_.shape, did_);
    aux_.create (src_.shape, did_);
    aux_.init (1.f/dims_);
#ifdef __CUDACC__
    cuda_check (cudnnCreateTensorDescriptor (&srcDesc_));
    cuda_check (cudnnCreateTensorDescriptor (&dstDesc_));
    src_.setTensor4dDesc (srcDesc_);
    dst_.setTensor4dDesc (dstDesc_);
#endif
}

#endif
