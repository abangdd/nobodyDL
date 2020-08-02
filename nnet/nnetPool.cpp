#ifndef NNET_POOL_
#define NNET_POOL_

#include "../include/nnet.h"

#ifdef __CUDACC__
template LayerPool<GPU>::LayerPool (ParaLayer& pl, const int did, TensorGPUf& src, TensorGPUf& dst);
#else
template LayerPool<CPU>::LayerPool (ParaLayer& pl, const int did, TensorCPUf& src, TensorCPUf& dst);
#endif

template <typename XPU>
void LayerPool<XPU>::fprop (const bool is_train, const bool is_fixed) {
    switch (pl_.type) {
        case 1:  // upsample
            dst_.upsample_fprop (src_, pl_.strd);
            break;
        case 3:  // chl2im
            dst_.expand_area (src_, pl_.strd);
            break;
        case 4:  // im2chl
            dst_.expand_chls (src_, pl_.strd);
            break;
        default: // avg pool
            LOG (FATAL) << "not implemented pool method";
    }

    if (pl_.drop > 0 && is_train)
        dst_.drop_chls (mask_, pl_.drop);
}

template <typename XPU>
void LayerPool<XPU>::bprop (const bool is_prop_grad) {
    if (pl_.drop > 0 && is_prop_grad) {
        dst_.drop_chls (mask_, pl_.drop);
        mask_.init (rand_, UNIFORM, 0.f, 1.f);
    }

    switch (pl_.type) {
        case 1:  // upsample
            src_.upsample_bprop (dst_, pl_.strd);
            break;
        case 3:  // im2chl
            src_.expand_chls (dst_, pl_.strd);
            break;
        case 4:  // chl2im
            src_.expand_area (dst_, pl_.strd);
            break;
        default: // avg pool
            LOG (FATAL) << "not implemented pool method";
    }
}

template <typename XPU>
void LayerPool<XPU>::init_layer () {
    nums_ = src_.nums();
    chls_ = src_.chls();
    area_ = src_.area();

    switch (pl_.type) {
        case 1:  // upsample
            dst_.create (src_.shape.upsample(pl_.strd), did_);
            break;
        case 3:  // chl2im
            dst_.create (src_.shape.expand_area(pl_.strd), did_);
            break;
        case 4:  // im2chl
            dst_.create (src_.shape.expand_chls(pl_.strd), did_);
            break;
        default: // avg pool
            dst_.create (Shape (1, 1, chls_, nums_), did_);
    }

    if (pl_.drop > 0) {
        mask_.create (dst_.shape, did_);
        mask_.init (rand_, UNIFORM, 0.f, 1.f);
    }
}

#endif
