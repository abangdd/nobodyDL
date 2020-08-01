#ifndef OPTIM_BASE_
#define OPTIM_BASE_

#include "../include/optimization.h"

#ifndef __CUDACC__
ParaOptim::ParaOptim () : algo(1) {
    momentum = 0.9;
    lr_base = 0.04;
    lr_last = 1e-4;
    decay = 1e-4;
}

int ParaOptim::get_optim_type (const char *t) {
    if (!strcmp (t, "sgd")) return kBSGD;
    if (!strcmp (t, "lbfgs")) return kLBFGS;
    LOG (FATAL) << "unknown optim type";
    return 0;
}

void ParaOptim::get_optim_info () {
    LOG (INFO) << "\tOptim initialized\t" << lr_base << "\t" << lr_last << "\t" << momentum << "\t" << decay;
}

void ParaOptim::set_para (const int now_round, const int max_round) {
    const float a = max_round;
    const float b = now_round;
    const float r = lr_last / lr_base;
    const float x = log (r) / log (1.f/a);
    lrate = lr_base * pow (1.f - b/a, x);
//printf ("%d\t%.5f\n", now_round, lrate);
}
#endif

template <typename XPU, typename DT>
void OptimBase<XPU, DT>::ccl_update () {
#ifdef __CUDACC__
    cuda_check (ncclReduce (gmat_.dptr, gmat_.dptr, gmat_.size(), nccl_type<DT>(), ncclSum, 0, dnnCtx[did_].cucomm_, dnnCtx[did_].stream_));
    XPU::sync_stream (did_);
#endif
    gmat_.blas_scal (1.f/dnnCtx.dnums());
    update ();
#ifdef __CUDACC__
    cuda_check (ncclBroadcast (wmat_.dptr, wmat_.dptr, wmat_.size(), nccl_type<DT>(), 0, dnnCtx[did_].cucomm_, dnnCtx[did_].stream_));
    XPU::sync_stream (did_);
#endif
}

template <typename XPU, typename DT>
std::shared_ptr<OptimBase<XPU, DT>> create_optim (ParaOptim& po, const int did, Tensor<XPU, DT>& wmat, Tensor<XPU, DT>& gmat) {
    switch (po.type) {
        case kBSGD  : return std::make_shared<OptimBSGD <XPU, DT>>(po, did, wmat, gmat);
        case kLBFGS : return std::make_shared<OptimLBFGS<XPU, DT>>(po, did, wmat, gmat);
        default : LOG (FATAL) << "not implemented optim type";
    }
    return nullptr;
}
#ifdef __CUDACC__
template void OptimBaseGPUf::ccl_update ();
template void OptimBaseGPUd::ccl_update ();
template std::shared_ptr<OptimBaseGPUf> create_optim (ParaOptim& po, const int did, TensorGPUf& wmat, TensorGPUf& gmat);
template std::shared_ptr<OptimBaseGPUd> create_optim (ParaOptim& po, const int did, TensorGPUd& wmat, TensorGPUd& gmat);
#else
template void OptimBaseCPUf::ccl_update ();
template void OptimBaseCPUd::ccl_update ();
template std::shared_ptr<OptimBaseCPUf> create_optim (ParaOptim& po, const int did, TensorCPUf& wmat, TensorCPUf& gmat);
template std::shared_ptr<OptimBaseCPUd> create_optim (ParaOptim& po, const int did, TensorCPUd& wmat, TensorCPUd& gmat);
#endif

#endif
