#ifndef OPTIM_BSGD_
#define OPTIM_BSGD_

#include "../include/optimization.h"

template <typename XPU, typename DT>
OptimBSGD<XPU, DT>::OptimBSGD (ParaOptim& po, const int did, Tensor<XPU, DT>& wmat, Tensor<XPU, DT>& gmat) :
    OptimBase<XPU, DT> (po, did, wmat, gmat), po_(po), did_(did), wmat_(wmat), gmat_(gmat) {
    mmat_.create (wmat_.shape, did_);  mmat_.mem_set (0);
    hmat_.create (wmat_.shape, did_);  hmat_.mem_set (0);
}
#ifdef __CUDACC__
template OptimBSGD<GPU, float >::OptimBSGD (ParaOptim& po, const int did, TensorGPUf& wmat, TensorGPUf& gmat);
template OptimBSGD<GPU, double>::OptimBSGD (ParaOptim& po, const int did, TensorGPUd& wmat, TensorGPUd& gmat);
#else
template OptimBSGD<CPU, float >::OptimBSGD (ParaOptim& po, const int did, TensorCPUf& wmat, TensorCPUf& gmat);
template OptimBSGD<CPU, double>::OptimBSGD (ParaOptim& po, const int did, TensorCPUd& wmat, TensorCPUd& gmat);
#endif



template <typename XPU, typename DT>
void OptimBSGD<XPU, DT>::update () {
    if (po_.algo == 0)
        update_sgd ();
    if (po_.algo == 1)
        update_nag ();
}

template <typename XPU, typename DT>
void OptimBSGD<XPU, DT>::update_sgd () {
    // mmat_ = momentum * mmat_ - lr * (gmat_ + decay * wmat_)
    gmat_.blas_axpy (wmat_, po_.decay);
    mmat_.blas_scal (po_.momentum);
    mmat_.blas_axpy (gmat_, - po_.lrate);

    wmat_.blas_vadd (wmat_, mmat_);
}

template <typename XPU, typename DT>
void OptimBSGD<XPU, DT>::update_nag () {
    gmat_.blas_axpy (wmat_, po_.decay);
    wmat_.blas_axpy (mmat_, - po_.momentum);

    mmat_.blas_scal (po_.momentum);
    mmat_.blas_axpy (gmat_, - po_.lrate);
    wmat_.blas_axpy (mmat_, 1+po_.momentum);
}

#endif
