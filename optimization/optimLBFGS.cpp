#ifndef OPTIM_LBFGS_
#define OPTIM_LBFGS_

#include <algorithm>
#include "../include/optimization.h"

template <typename XPU, typename DT>
OptimLBFGS<XPU, DT>::OptimLBFGS (ParaOptim& po, const int did, Tensor<XPU, DT>& wmat, Tensor<XPU, DT>& hmat) :
    OptimBase<XPU, DT> (po, did, wmat, hmat), po_(po), did_(did), wmat_(wmat), gmat_(hmat) {
    wold_.create (wmat_.shape, did_);
    gold_.create (wmat_.shape, did_);
    dmat_.create (wmat_.shape, did_);

    Shape hshape (hsize, wmat_.size(), 1, 1);  // TODO
    smat_.create (hshape, did_);
    ymat_.create (hshape, did_);
    hmat_.create (hshape, did_);

    loss_his.assign (hsize, 0);
    alpha.assign (hsize, 0);
    beta.assign (hsize, 0);
    yts.assign (hsize, 0);
    yty.assign (hsize, 0);
}
#ifdef __CUDACC__
template OptimLBFGS<GPU, float >::OptimLBFGS (ParaOptim& po, const int did, TensorGPUf& wmat, TensorGPUf& hmat);
template OptimLBFGS<GPU, double>::OptimLBFGS (ParaOptim& po, const int did, TensorGPUd& wmat, TensorGPUd& hmat);
#else
template OptimLBFGS<CPU, float >::OptimLBFGS (ParaOptim& po, const int did, TensorCPUf& wmat, TensorCPUf& hmat);
template OptimLBFGS<CPU, double>::OptimLBFGS (ParaOptim& po, const int did, TensorCPUd& wmat, TensorCPUd& hmat);
#endif

template <typename XPU, typename DT>
void OptimLBFGS<XPU, DT>::init_model () {
    wmat_.mem_set(0);
    gmat_.mem_set(0);
    dmat_.mem_set(0);
    smat_.mem_set(0);
    ymat_.mem_set(0);
    hmat_.mem_set(0);

    loss_his.assign (hsize, 0);
    alpha.assign (hsize, 0);
    beta.assign (hsize, 0);
    yts.assign (hsize, 0);
    yty.assign (hsize, 0);
    H0 = 1;
}

template <typename XPU, typename DT>
bool OptimLBFGS<XPU, DT>::terminate (const int k) {
    const int i = k % hsize;
    if (k >= hsize && loss_new/loss_his[i] >= 1+po_.c1) {
        const int idx = std::min_element (loss_his.begin(), loss_his.end()) - loss_his.begin();
        LOG (INFO) << "\tterminated with minimal loss\t" << loss_his[idx];
        wmat_.copy (hmat_[idx]);
        return true;
    }
    loss_his[i] = loss_new;
    hmat_[i].copy (wmat_);

    DT wnorm = wmat_.blas_nrm2();
    DT gnorm = gmat_.blas_nrm2();
    wnorm = std::max(wnorm, DT(1));
    return (gnorm / wnorm <= 1e-3);
}

template <typename XPU, typename DT>
void OptimLBFGS<XPU, DT>::set_direction (const int k) {
    dmat_.mem_set (0);
    dmat_.blas_vsub (dmat_, gmat_);
    for (int n = k-1; n >= k-std::min(k,hsize); --n) {
        const int i = n % hsize;
        dmat_.blas_sdot (smat_[i], alpha[i]);
        alpha[i] /= yts[i];
        dmat_.blas_axpy (ymat_[i], -alpha[i]);
    }
    dmat_.blas_scal (H0);
    for (int n = k-std::min(k,hsize); n <= k-1; ++n) {
        const int i = n % hsize;
        dmat_.blas_sdot (ymat_[i], beta[i]);
        beta[i] /= yts[i];
        dmat_.blas_axpy (smat_[i], alpha[i]-beta[i]);
    }
    wold_.copy (wmat_);
    gold_.copy (gmat_);
}

template <typename XPU, typename DT>
void OptimLBFGS<XPU, DT>::set_smat_ymat (const int k) {
    const int i = k % hsize;
    smat_[i].blas_vsub (wmat_, wold_);  // = x_k - x_{k-1}
    ymat_[i].blas_vsub (gmat_, gold_);  // = g_k - g_{k-1}
    ymat_[i].blas_sdot (smat_[i], yts[i]);
    ymat_[i].blas_sdot (ymat_[i], yty[i]);
    H0 = yts[i] / yty[i];
}

template <typename XPU, typename DT>
void OptimLBFGS<XPU, DT>::back_tracking (std::function<void(DT&)> estimate, std::function<void(DT&)> validate) {
    for (int i = 0; i < 8; ++i) {
        wmat_.copy (wold_);
        wmat_.blas_axpy (dmat_, po_.lrate);
        estimate (loss_new);
        if (loss_new > loss_old * 1.01)  // Armijo condition modified
            po_.multi = 0.1;
        else if (loss_new > loss_old * (1+po_.c1))  // Armijo condition modified
            po_.multi = 0.5;
        else {
            validate (loss_val);
            LOG_IF (WARNING, loss_new > loss_old) << "\tobjective function increased";
            loss_old = loss_new;
            break;
        }
        po_.lrate *= po_.multi;
    }
    validate (loss_val);
    LOG_IF (WARNING, loss_new > loss_old) << "\tobjective function increased";
}

template <typename XPU, typename DT>
void OptimLBFGS<XPU, DT>::optimize (std::function<void(DT&)> estimate, std::function<void(DT&)> validate) {
    init_model ();
    estimate (loss_old);
    po_.lrate = 1 / gmat_.blas_nrm2();
    for (int k = 0; k < 50; ++k) {
        set_direction (k);
        back_tracking (estimate, validate);
        set_smat_ymat (k);
        po_.lrate = 1;
        char errstr[64];  sprintf (errstr, "\ttrain\t%.6f\tvalid\t%.6f", loss_new, loss_val);
        LOG (INFO) << "\tround  " << k << errstr;
        if (terminate (k))
            break;
    }
}
#endif
