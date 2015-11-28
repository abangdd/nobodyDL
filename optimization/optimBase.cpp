#ifndef OPTIM_BASE_
#define OPTIM_BASE_

#include "../include/optimization.h"

#ifndef __CUDACC__
ParaOptim::ParaOptim () : algo(1), schedule(0), lossType(0), isFixed(false)
{ momentum = 0.9;
  lr_base = 1e-2;
  lr_last = 1e-4;
  wd_base = 2e-4;
}

int  ParaOptim::get_optim_type (const char *t)
{ if (!strcmp (t, "lbfgs")) return kLBFGS;
  if (!strcmp (t, "sgd"	 )) return kVSGD;
  LOG (FATAL) << "unknown optim type";
  return 0;
}

void ParaOptim::get_optim_info ()
{ LOG (INFO) << "\tOptim initialized\t" << lr_base << "\t" << lr_last << "\t" << momentum << "\t" << wd_base;
}

void ParaOptim::set_para (const int now_round, const int max_round)
{ const float a = max_round;
  const float b = now_round;
  const float r = lr_last / lr_base;
  const float x = log (r) / log (1.f/a);
  lrate = lr_base * pow (1.f - b/a, x);
  decay = wd_base;
//std::cout << epoch << "\t" << lrate << std::endl;
}
#endif

template <typename XPU, typename DT>
OptimBase<XPU, DT>* create_optim (ParaOptim &po, const int did, Tensor<XPU, DT> &weight, Tensor<XPU, DT> &wgrad)
{ switch (po.type)
  { case kLBFGS	: return new OptimLBFGS<XPU, DT>(po, did, weight, wgrad);
    case kVSGD	: return new OptimVSGD <XPU, DT>(po, did, weight, wgrad);
    default	: LOG (FATAL) << "not implemented optim type";
  }
  return NULL;
}
#ifdef __CUDACC__
template OptimBaseGPUf* create_optim (ParaOptim &po, const int did, TensorGPUf &weight, TensorGPUf &wgrad);
template OptimBaseGPUd* create_optim (ParaOptim &po, const int did, TensorGPUd &weight, TensorGPUd &wgrad);
#else
template OptimBaseCPUf* create_optim (ParaOptim &po, const int did, TensorCPUf &weight, TensorCPUf &wgrad);
template OptimBaseCPUd* create_optim (ParaOptim &po, const int did, TensorCPUd &weight, TensorCPUd &wgrad);
#endif

#endif
