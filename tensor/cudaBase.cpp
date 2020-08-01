#ifndef CUDA_BASE_
#define CUDA_BASE_

#include "../include/xpu.h"

template <>
XPU_KERNEL(kprint) (const int knum, const int* x) {
    kernel_for (i, knum)
        printf ("%d\t", x[i]);
}
template <>
XPU_KERNEL(kprint) (const int knum, const float* x) {
    kernel_for (i, knum)
        printf ("%.4f\t", x[i]);
}
template <>
XPU_KERNEL(kprint) (const int knum, const double* x) {
    kernel_for (i, knum)
        printf ("%.4f\t", x[i]);
}

#ifndef __CUDACC__
DNNCtx<GPU> dnnCtx;
#else
template <>
void Device<GPU>::init() {
    LOG (INFO) << "\tGPU  " << did_ << "  initializing";
    cuda_check (cudaSetDevice (did_));
    cuda_check (cudaDeviceReset ());

    cuda_check (cudnnCreate (&cudnn_));
    cuda_check (cublasCreate (&cublas_));
    cuda_check (cusparseCreate (&cusparse_));
    cuda_check (cudaStreamCreate (&stream_));
    cuda_check (curandCreateGenerator (&curand_, CURAND_RNG_PSEUDO_DEFAULT));
    cuda_check (cudaEventCreateWithFlags (&accept_, cudaEventDisableTiming));

    cuda_check (cublasSetStream (cublas_, stream_));
    cuda_check (curandSetStream (curand_, stream_));
    cuda_check (cudnnSetStream (cudnn_, stream_));
    cuda_check (curandSetPseudoRandomGeneratorSeed (curand_, 0));

    convWorkSize = 1024 * 1024 * 1024;
    normWorkSize = 16 * 1024 * 1024;
    reluWorkSize = 1024 * 1024 * 1024;
    if (convWorkSize > 0) {
        char size_m[16]; sprintf (size_m, "%.2f MB", convWorkSize / 1048576.f);
        LOG (INFO) << "\tGPU  " << did_ << "  memory required for convWorkAddr\t" << size_m;
        cuda_check (cudaMallocManaged (&convWorkAddr, convWorkSize, cudaMemAttachGlobal));
    }
    if (normWorkSize > 0) {
        char size_m[16]; sprintf (size_m, "%.2f MB", normWorkSize / 1048576.f);
        LOG (INFO) << "\tGPU  " << did_ << "  memory required for normWorkAddr\t" << size_m;
        cuda_check (cudaMallocManaged (&normWorkAddr, normWorkSize, cudaMemAttachGlobal));
    }
    if (reluWorkSize > 0) {
        char size_m[16]; sprintf (size_m, "%.2f MB", reluWorkSize / 1048576.f);
        LOG (INFO) << "\tGPU  " << did_ << "  memory required for reluWorkAddr\t" << size_m;
        cuda_check (cudaMallocManaged (&reluWorkAddr, reluWorkSize, cudaMemAttachGlobal));
    }
}

template <>
void Device<GPU>::release () {
    LOG (INFO) << "\tGPU  " << did_ << "  releasing";
    cuda_check (cudaSetDevice (did_));

    cuda_check (cudnnDestroy (cudnn_));
    cuda_check (cublasDestroy (cublas_));
    cuda_check (cusparseDestroy (cusparse_));
    cuda_check (cudaStreamDestroy (stream_));
    cuda_check (curandDestroyGenerator (curand_));
    cuda_check (cudaEventDestroy (accept_));
    cuda_check (ncclCommDestroy (cucomm_));

    if (algoWorkSize > 0)  cuda_check (cudaFree (algoWorkAddr));
    if (convWorkSize > 0)  cuda_check (cudaFree (convWorkAddr));
    if (normWorkSize > 0)  cuda_check (cudaFree (normWorkAddr));
    if (reluWorkSize > 0)  cuda_check (cudaFree (reluWorkAddr));
}



template <>
void DNNCtx<GPU>::init (const int dmin, const int dmax) {
    dmin_ = dmin;
    dmax_ = dmax;
    dnum_ = dmax - dmin + 1;

    for (int i = 0; i < dnum_; ++i) {
        ctx_.emplace_back(i+dmin_);
        ctx_[i].init();
    }
    ncclUniqueId id;
    cuda_check (ncclGetUniqueId (&id));
    cuda_check (ncclGroupStart());
    for (int i = 0; i < dnum_; ++i) {
        GPU::set_device (i+dmin_);
        cuda_check (ncclCommInitRank (&ctx_[i].cucomm_, dnum_, id, i));
    }
    cuda_check (ncclGroupEnd());
}

template <>
void DNNCtx<GPU>::release () {
    for (auto& ctx : ctx_)
        ctx.release();
}



template <> ncclDataType_t nccl_type<float> () { return ncclFloat; }
template <> ncclDataType_t nccl_type<double>() { return ncclDouble; }
template <> cudnnDataType_t cudnn_type<float> () { return CUDNN_DATA_FLOAT; }
template <> cudnnDataType_t cudnn_type<double>() { return CUDNN_DATA_DOUBLE; }

void cuda_set_p2p (const int num_device) {
    for (int did = 0; did < num_device; ++did) {
        GPU::set_device (did);
        for (int pid = 0; pid < num_device; ++pid)
            if (pid != did) {
                dnnCtx[did].cup2p_[pid] = 0;
                cuda_check (cudaDeviceCanAccessPeer (&dnnCtx[did].cup2p_[pid], did, pid));
                if (dnnCtx[did].cup2p_[pid])
                    cuda_check (cudaDeviceEnablePeerAccess (pid, 0));
            } else
                dnnCtx[did].cup2p_[pid] = 1;
    }
}

void cuda_del_p2p (const int num_device) {
    for (int did = 0; did < num_device; ++did) {
        GPU::set_device (did);
        for (int pid = 0; pid < num_device; ++pid)
            if (pid != did) {
                cuda_check (cudaDeviceCanAccessPeer (&dnnCtx[did].cup2p_[pid], did, pid));
                if (dnnCtx[did].cup2p_[pid])
                    cuda_check (cudaDeviceDisablePeerAccess (pid));
            }
    }
}



void* GPU::operator new(size_t len) {
    void* ptr = nullptr;
    cuda_check (cudaMallocManaged (&ptr, len, cudaMemAttachGlobal));
    return ptr;
}

void GPU::operator delete(void* ptr) {
    cuda_check (cudaFree (ptr));
}

void GPU::set_device (const int did) {
    int curr_device;
    cuda_check (cudaGetDevice (&curr_device));
    if (curr_device == did)
        return;
    cuda_check (cudaSetDevice (did));
}

void GPU::sync_stream (const int did) {
    cuda_check (cudaStreamSynchronize (dnnCtx[did].stream_));
}
#endif

#endif
