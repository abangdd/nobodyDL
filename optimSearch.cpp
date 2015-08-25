#ifndef OPTIM_SEARCH_
#define OPTIM_SEARCH_

#include "../include/optimization.h"

template <typename XPU, typename DT>
bool OptimBase<XPU, DT>::line_search_backtracking (SparseBuffer<XPU, DT> &buffer, const Tensor<XPU, DT> &dir,
  const Tensor<XPU, DT> &wvec_b, int maxEvals)
{ int evals = 0;
  DT rho = 0;
  const DT dec = 0.5;
  const DT inc = 2.1;
  const DT c1 = 1e-4;
  const DT c2 = 0.9f;

  f_phi_0 = loss_k;  LOG (INFO) << "\tloss_k\t" << f_phi_0;
  gmat_.blas_sdot (dir, d_phi_0);

  if (d_phi_0 > 0.f)
  { LOG (WARNING) << "\tOPTIM_LINE_SEARCH_FAILED";
    return false;
  }

  for (;;)
  { 
    wmat_.copy (wvec_b);
    wmat_.blas_axpy (dir, step_length);

    get_grad (buffer);
    get_eval (buffer, f_phi_alpha);
    gmat_.blas_sdot (dir, d_phi_alpha);

    ++evals;

    const bool armijo_violated = f_phi_alpha > f_phi_0 + c1 * step_length * d_phi_0;

    if (armijo_violated)
      rho = dec;
    else if (d_phi_alpha <   c2 * d_phi_0)  // Wolfe condition
      rho = inc;
    else if (d_phi_alpha > - c2 * d_phi_0)  // strong Wolfe condition
      rho = dec;
    else
    { loss_k = f_phi_alpha;
      return true;
    }

    if (evals >= maxEvals)
    { LOG (WARNING) << "\tOPTIM_REACHED_MAX_EVALS";
      return false;
    }

    step_length *= rho;
  }
}
#ifdef __CUDACC__
template bool OptimBaseGPUf::line_search_backtracking (SBufferGPUf &buffer, const TensorGPUf &dir, const TensorGPUf &wvec_b, int maxEvals);
template bool OptimBaseGPUd::line_search_backtracking (SBufferGPUd &buffer, const TensorGPUd &dir, const TensorGPUd &wvec_b, int maxEvals);
#else
template bool OptimBaseCPUf::line_search_backtracking (SBufferCPUf &buffer, const TensorCPUf &dir, const TensorCPUf &wvec_b, int maxEvals);
template bool OptimBaseCPUd::line_search_backtracking (SBufferCPUd &buffer, const TensorCPUd &dir, const TensorCPUd &wvec_b, int maxEvals);
#endif

#endif
