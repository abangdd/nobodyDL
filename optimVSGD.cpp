#ifndef OPTIM_VSGD_
#define OPTIM_VSGD_

#include "../include/optimization.h"

template <typename XPU, typename DT>
OptimVSGD<XPU, DT>::OptimVSGD (ParaOptim &po, Tensor<XPU, DT> &weight, Tensor<XPU, DT> &wgrad) :
  OptimBase<XPU, DT> (po, weight, wgrad), para_(po), wmat_(weight), gmat_(wgrad), epoch(0)
{ mmat_.create (wmat_.shape);  mmat_.mem_set (0);
  hmat_.create (wmat_.shape);  hmat_.mem_set (0);
if (para_.algo == 2)
  vmat_.create (wmat_.shape);  vmat_.mem_set (0);
}
#ifdef __CUDACC__
template OptimVSGD<GPU, float >::OptimVSGD (ParaOptim &po, TensorGPUf &weight, TensorGPUf &wgrad);
template OptimVSGD<GPU, double>::OptimVSGD (ParaOptim &po, TensorGPUd &weight, TensorGPUd &wgrad);
#else
template OptimVSGD<CPU, float >::OptimVSGD (ParaOptim &po, TensorCPUf &weight, TensorCPUf &wgrad);
template OptimVSGD<CPU, double>::OptimVSGD (ParaOptim &po, TensorCPUd &weight, TensorCPUd &wgrad);
#endif

template <typename XPU, typename DT>
void OptimVSGD<XPU, DT>::update ()
{ if (para_.algo == 0)
    update_sgd ();
  if (para_.algo == 1)
    update_nag ();
  if (para_.algo == 2)
    update_adam ();
}

template <typename XPU, typename DT>
void OptimVSGD<XPU, DT>::update_sgd ()
{ // mmat_ = momentum * mmat_ - lr * (gmat_ + wd * wmat_)
  gmat_.blas_axpy (wmat_, para_.wd);
  mmat_.blas_scal (para_.momentum);
  mmat_.blas_axpy (gmat_, - para_.lrate);

  wmat_.blas_vadd (wmat_, mmat_);
}

template <typename XPU, typename DT>
void OptimVSGD<XPU, DT>::update_nag ()
{ hmat_.copy (mmat_);

  gmat_.blas_axpy (wmat_, para_.wd);
  mmat_.blas_scal (para_.momentum);
  mmat_.blas_axpy (gmat_, - para_.lrate);

  wmat_.blas_axpy (hmat_, - para_.momentum);
  wmat_.blas_axpy (mmat_, 1+para_.momentum);
}

template <typename XPU, typename DT>
void OptimVSGD<XPU, DT>::update_adam ()
{ gmat_.blas_axpy (wmat_, para_.wd);
  hmat_.blas_vsqr (gmat_);

  mmat_.blas_scal (0.9f);
  mmat_.blas_axpy (gmat_, 0.1f);
  vmat_.blas_scal (0.999f);
  vmat_.blas_axpy (hmat_, 0.001f);

  hmat_.blas_vsqrt(vmat_);  hmat_.add (1e-8f);
  hmat_.blas_vdiv (mmat_, hmat_);

  DT fix1 = 1.f - powf (0.9f,   epoch + 1);
  DT fix2 = 1.f - powf (0.999f, epoch + 1);
  DT lr_t = para_.lr_alpha * para_.lrate * sqrt(fix2) / fix1;
  epoch++;

  wmat_.blas_axpy (hmat_, - lr_t);
}

template <typename XPU, typename DT>
void OptimVSGD<XPU, DT>::optimize (SparseBuffer<XPU, DT> &buffer)
{ this->set_cache (buffer.data_, buffer.label_);

  int epoch = 0;
  while (epoch++ < 30)
  { this->get_grad (buffer);
    this->get_eval (buffer, this->loss_k);
    update ();
  }
}

#endif
