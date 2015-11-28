#ifndef OPTIM_VSGD_
#define OPTIM_VSGD_

#include "../include/optimization.h"

template <typename XPU, typename DT>
OptimVSGD<XPU, DT>::OptimVSGD (ParaOptim &po, const int did, Tensor<XPU, DT> &weight, Tensor<XPU, DT> &wgrad) :
  OptimBase<XPU, DT> (po, did, weight, wgrad), po_(po), did_(did), wmat_(weight), gmat_(wgrad), epoch(0)
{ mmat_.create (wmat_.shape, did_);  mmat_.mem_set (0);
  hmat_.create (wmat_.shape, did_);  hmat_.mem_set (0);
}
#ifdef __CUDACC__
template OptimVSGD<GPU, float >::OptimVSGD (ParaOptim &po, const int did, TensorGPUf &weight, TensorGPUf &wgrad);
template OptimVSGD<GPU, double>::OptimVSGD (ParaOptim &po, const int did, TensorGPUd &weight, TensorGPUd &wgrad);
#else
template OptimVSGD<CPU, float >::OptimVSGD (ParaOptim &po, const int did, TensorCPUf &weight, TensorCPUf &wgrad);
template OptimVSGD<CPU, double>::OptimVSGD (ParaOptim &po, const int did, TensorCPUd &weight, TensorCPUd &wgrad);
#endif

template <typename XPU, typename DT>
void OptimVSGD<XPU, DT>::update ()
{ if (po_.algo == 0)
    update_sgd ();
  if (po_.algo == 1)
    update_nag ();
}

template <typename XPU, typename DT>
void OptimVSGD<XPU, DT>::update_sgd ()
{ // mmat_ = momentum * mmat_ - lr * (gmat_ + decay * wmat_)
  gmat_.blas_axpy (wmat_, po_.decay);
  mmat_.blas_scal (po_.momentum);
  mmat_.blas_axpy (gmat_, - po_.lrate);

  wmat_.blas_vadd (wmat_, mmat_);
}

template <typename XPU, typename DT>
void OptimVSGD<XPU, DT>::update_nag ()
{ hmat_.copy (mmat_);

  gmat_.blas_axpy (wmat_, po_.decay);
  mmat_.blas_scal (po_.momentum);
  mmat_.blas_axpy (gmat_, - po_.lrate);

  wmat_.blas_axpy (hmat_, - po_.momentum);
  wmat_.blas_axpy (mmat_, 1+po_.momentum);
}

#endif
