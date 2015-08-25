#ifndef OPTIM_LBFGS_
#define OPTIM_LBFGS_

#include "../include/optimization.h"

using std::min;

template <typename XPU, typename DT>
OptimLBFGS<XPU, DT>::OptimLBFGS (ParaOptim &po, const int did, Tensor<XPU, DT> &weight, Tensor<XPU, DT> &wgrad) :
  OptimBase<XPU, DT> (po, did, weight, wgrad), did_(did), H0(1.f), hsize(8)
{ const int numFeat = this->wmat_.rows();  // TODO

  wvec_j.create (this->wmat_.shape, did_);
  gvec_j.create (this->wmat_.shape, did_);
  dir   .create (this->wmat_.shape, did_);

  // treat arrays as ring buffers!
  Shape s_shape (hsize, numFeat, 1, 1);
  smat_.create (s_shape, did_);
  ymat_.create (s_shape, did_);

  alpha.assign (hsize, 0);
  beta .assign (hsize, 0);
  rho  .assign (hsize, 0);
}
#ifdef __CUDACC__
template OptimLBFGS<GPU, float >::OptimLBFGS (ParaOptim &po, const int did, TensorGPUf &weight, TensorGPUf &wgrad);
template OptimLBFGS<GPU, double>::OptimLBFGS (ParaOptim &po, const int did, TensorGPUd &weight, TensorGPUd &wgrad);
#else
template OptimLBFGS<CPU, float >::OptimLBFGS (ParaOptim &po, const int did, TensorCPUf &weight, TensorCPUf &wgrad);
template OptimLBFGS<CPU, double>::OptimLBFGS (ParaOptim &po, const int did, TensorCPUd &weight, TensorCPUd &wgrad);
#endif

template <typename XPU, typename DT>
void OptimLBFGS<XPU, DT>::get_direction (const int k)
{ dir.mem_set (0);
  dir.blas_vsub (dir, gvec_k);

  for (int m = 1; m <= min (k, hsize); ++m)
  { const int i = (k - m) % hsize;
    dir.blas_sdot (smat_[i],   alpha[i]);
    alpha[i] *= rho[i];
    dir.blas_axpy (ymat_[i], - alpha[i]);

  }

  dir.blas_scal (H0);

  for (int m = min (k, hsize); m > 0; --m)
  { const int i = (k - m) % hsize;
    dir.blas_sdot (ymat_[i],  beta[i]);
    beta[i] *= rho[i];
    dir.blas_axpy (smat_[i], alpha[i] - beta[i]);
  }
}

template <typename XPU, typename DT>
void OptimLBFGS<XPU, DT>::set_s_y_rho_h (const int k)
{ Tensor<XPU, DT> s_k = smat_[k % hsize];
  Tensor<XPU, DT> y_k = ymat_[k % hsize];

  s_k.blas_vsub (wvec_k, wvec_j);  // = x_k - x_{k-1}
  y_k.blas_vsub (gvec_k, gvec_j);  // = g_k - g_{k-1}

  DT yts;  y_k.blas_sdot (s_k, yts);  yts += 1e-8;

  rho[k % hsize] = 1.f / yts;

  DT yty;  y_k.blas_sdot (y_k, yty);  yty += 1e-8;

  H0 = yts / yty;
}

template <typename XPU, typename DT>
void OptimLBFGS<XPU, DT>::optimize (SparseBuffer<XPU, DT> &buffer)
{ this->set_cache (buffer.data_, buffer.label_);

  this->get_grad (buffer);
  this->get_eval (buffer, this->loss_k);
  wvec_k = this->wmat_;  // x_k,  current solution
  gvec_k = this->gmat_;  // g_k,  current gradient

  DT gnorm = 0.f;  gvec_k.blas_nrm2 (gnorm);
  this->step_length = 1.f / gnorm;

  for (int k = 0; ; ++k)
  { if (terminate ())
      break;

    get_direction (k);

    wvec_j.copy (wvec_k);
    gvec_j.copy (gvec_k);
    if (!this->line_search_backtracking (buffer, dir, wvec_j, 8))
    { wvec_k.copy (wvec_j);
      break;
    }
    set_s_y_rho_h (k);

    this->step_length = 1.f;
  }
}

template <typename XPU, typename DT>
bool OptimLBFGS<XPU, DT>::terminate ()
{ DT xnorm, gnorm;
  wvec_k.blas_sdot (wvec_k, xnorm);
  gvec_k.blas_sdot (gvec_k, gnorm);
  if (xnorm < 1.f) xnorm = 1.f;
  return (gnorm / xnorm <= 0.00001);
}

#endif
