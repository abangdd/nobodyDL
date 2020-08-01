#ifndef OPTIMIZATION_H_
#define OPTIMIZATION_H_

#include "tensor.h"

enum optim_t {
    kLBFGS = 1,
    kBSGD = 2
};

class ParaOptim {
public:
    ParaOptim ();
    int  get_optim_type (const char* t);
    void get_optim_info ();
    void set_para (const int now_round, const int max_round);
public:
    int type;
    int algo;
    float momentum, decay;
    float lrate = 0.1, multi;
    float lr_base, c1 = 1e-4;
    float lr_last, c2 = 0.9;
};



template <typename XPU, typename DT>
class OptimBase {
public:
    using opt_cb = std::function<void(DT&)>;
    OptimBase (ParaOptim& po, const int did, Tensor<XPU, DT>& wmat, Tensor<XPU, DT>& gmat) : po_(po), did_(did), wmat_(wmat), gmat_(gmat) { }
    virtual ~OptimBase () { }
    virtual void init_model () { }
    virtual void swap_model () { }
    virtual void ccl_update ();
    virtual void update () { };
    virtual void optimize (opt_cb estimate, opt_cb validate) { }
    virtual void reduce_notify () { reduce_.notify (); }
    virtual void accept_notify () { accept_.notify (); }
    virtual void reduce_wait (OptimBase<XPU, DT>& in) { in.reduce_.wait (); }
    virtual void accept_wait (OptimBase<XPU, DT>& in) { in.accept_.wait (); }
    virtual void reduce_gmat (OptimBase<XPU, DT>& in) { gmat_.blas_axpy (in.gmat_, 1);  XPU::sync_stream (did_); }
    virtual void accept_wmat (OptimBase<XPU, DT>& in) { wmat_.copy (in.wmat_);  XPU::sync_stream (did_); }
public:
    ParaOptim& po_;
    int did_ = 0;
private:
    Tensor<XPU, DT> &wmat_;
    Tensor<XPU, DT> &gmat_;
    SyncCV reduce_;
    SyncCV accept_;
};

using OptimBaseGPUf = OptimBase<GPU, float>;
using OptimBaseCPUf = OptimBase<CPU, float>;
using OptimBaseGPUd = OptimBase<GPU, double>;
using OptimBaseCPUd = OptimBase<CPU, double>;

template <typename XPU, typename DT>
class OptimBSGD : public OptimBase<XPU, DT> {
public:
    using opt_cb = std::function<void(DT&)>;
    OptimBSGD (ParaOptim& po, const int did, Tensor<XPU, DT>& wmat, Tensor<XPU, DT>& gmat);
    void swap_model ();
    void update ();
private:
    void update_sgd ();
    void update_nag ();
    ParaOptim& po_;
    int did_ = 0, epoch = 0;
    Tensor<XPU, DT> &wmat_, mmat_;
    Tensor<XPU, DT> &gmat_, hmat_;
};

template <typename XPU, typename DT>
class OptimLBFGS : public OptimBase<XPU, DT> {
public:
    using opt_cb = std::function<void(DT&)>;
    OptimLBFGS (ParaOptim& po, const int did, Tensor<XPU, DT>& wmat, Tensor<XPU, DT>& gmat);
    void init_model ();
    void optimize (opt_cb estimate, opt_cb validate);
private:
    bool terminate (const int k);
    void set_direction (const int k);
    void set_smat_ymat (const int k);
    void back_tracking (opt_cb estimate, opt_cb validate);
    ParaOptim& po_;
    int did_ = 0, hsize = 8;
    Tensor<XPU, DT> &wmat_, wold_, smat_, hmat_;
    Tensor<XPU, DT> &gmat_, gold_, ymat_, dmat_;
    vector<DT> alpha, beta, yts, yty;
    DT H0 = 1, loss_old, loss_new, loss_val;
    vector<DT> loss_his;
};

template <typename XPU, typename DT>
std::shared_ptr<OptimBase<XPU, DT>> create_optim (ParaOptim& po, const int did, Tensor<XPU, DT>& wmat, Tensor<XPU, DT>& gmat);

#endif
