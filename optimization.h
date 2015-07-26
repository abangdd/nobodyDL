#ifndef OPTIMIZATION_H_
#define OPTIMIZATION_H_

#include "tensor.h"
#include "sparse.h"

enum optim_t
{ kLBFGS = 1,
  kVSGD	 = 2
};

class ParaOptim {
public:
  explicit ParaOptim ();
  int get_optim_type (const char *t);
  void get_optim_info ();
  void set_lrate (const int epoch, const int max_round);
public:
  int type;
  int algo;
  int lossType;
  bool isFixed;
  float wd;
  float momentum;
  float lrate;
  float lr_alpha;
  float lr_base;
  float lr_last;
};

template <typename XPU, typename DT>
class OptimBase {
public:
  explicit OptimBase (ParaOptim &po, const int did, Tensor<XPU, DT> &weight, Tensor<XPU, DT> &wgrad) :
    para_(po), did_(did), wmat_(weight), gmat_(wgrad) { }
  virtual ~OptimBase () { }
  virtual void update () = 0;
  virtual void get_direction (const int k) = 0;
  virtual void optimize (SparseBuffer<XPU, DT> &buffer) = 0;
  virtual void reduce_notify () { reduce_.notify ();  }
  virtual void accept_notify () { accept_.notify ();  }
  virtual void reduce_wait (OptimBase<XPU, DT> &in)  { in.reduce_.wait ();  }
  virtual void accept_wait (OptimBase<XPU, DT> &in)  { in.accept_.wait ();  }
  virtual void reduce_gmat (OptimBase<XPU, DT> &in)  { gmat_.blas_axpy (in.gmat_, (DT)1.);  cuda_stream_sync (did_);  }
  virtual void accept_wmat (OptimBase<XPU, DT> &in)  { wmat_.copy      (in.wmat_);          cuda_stream_sync (did_);  }
  virtual void reduce_scal (const DT alpha)          { gmat_.blas_scal (alpha);  }
  void set_cache(SparseTensor<XPU, DT> &data, Tensor<XPU, DT> &label);
  void get_pred (SparseTensor<XPU, DT> &data, Tensor<XPU, DT> &pred);
  void get_grad (SparseTensor<XPU, DT> &data, Tensor<XPU, DT> &label);
  void get_grad (SparseBuffer<XPU, DT> &buffer);
  void get_eval (SparseBuffer<XPU, DT> &buffer, DT &loss);
  bool line_search_backtracking (SparseBuffer<XPU, DT> &buffer, const Tensor<XPU, DT> &dir,
    Tensor<XPU, DT> &wvec, Tensor<XPU, DT> &gvec, int &evals, int maxEvals);
  bool line_search (SparseBuffer<XPU, DT> &buffer, const Tensor<XPU, DT> &dir,
    Tensor<XPU, DT> &wvec, Tensor<XPU, DT> &gvec, int &evals, int maxEvals);
public:
  ParaOptim &para_;
  int did_;
  Tensor<XPU, DT> &wmat_, &gmat_;
  Tensor<XPU, DT> dloss_;
  Tensor<XPU, DT> invc_;
  DT loss_k, step_length;
  SyncCV reduce_;
  SyncCV accept_;
private:
  DT alpha_0, alpha_j, alpha_low, alpha_high;
  DT f_phi_0, f_phi_j, f_phi_low, f_phi_high, f_phi_alpha;
  DT d_phi_0, d_phi_j, d_phi_low, d_phi_high, d_phi_alpha;  // directional derivative
};

typedef OptimBase<GPU, float>  OptimBaseGPUf;
typedef OptimBase<CPU, float>  OptimBaseCPUf;
typedef OptimBase<GPU, double> OptimBaseGPUd;
typedef OptimBase<CPU, double> OptimBaseCPUd;

template <typename XPU, typename DT>
class OptimVSGD : public OptimBase<XPU, DT> {
public:
  OptimVSGD (ParaOptim &po, const int did, Tensor<XPU, DT> &weight, Tensor<XPU, DT> &wgrad);
  void get_direction (const int k) { };
  void update ();
  void update_sgd ();
  void update_nag ();
  void update_adam ();
  void optimize (SparseBuffer<XPU, DT> &buffer);
private:
  ParaOptim &para_;
  int did_;
  Tensor<XPU, DT> &wmat_, &gmat_;
  Tensor<XPU, DT> mmat_, vmat_, hmat_;
  int epoch;
};

template <typename XPU, typename DT>
class OptimLBFGS : public OptimBase<XPU, DT> {
public:
  OptimLBFGS (ParaOptim &po, const int did, Tensor<XPU, DT> &weight, Tensor<XPU, DT> &wgrad);
  void get_direction (const int k);
  void set_s_y_rho_h (const int k);
  void update () { }
  void optimize (SparseBuffer<XPU, DT> &buffer);
  bool terminate ();
private:
  int did_;
  Tensor<XPU, DT> dir;
  Tensor<XPU, DT> smat_,  ymat_;
  Tensor<XPU, DT> wvec_k, wvec_j;
  Tensor<XPU, DT> gvec_k, gvec_j;
  std::vector<DT> alpha, beta, rho;
  DT H0;
  int hsize;
};

template <typename XPU, typename DT>
OptimBase<XPU, DT>* create_optim (ParaOptim &po, const int did, Tensor<XPU, DT> &weight, Tensor<XPU, DT> &wgrad);

#endif
