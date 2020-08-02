#ifndef NNET_H_
#define NNET_H_

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse.h>
#include <driver_types.h>
#include <nccl.h>

#include "tensor.h"
#include "optimization.h"

enum layer_t {
    kAct  = 1,
    kConv = 2,
    kLoss = 4,
    kPool = 5
};

class ParaLayer {
public:
    explicit ParaLayer () : isLoad(false), isFixed(false), isNorm(false), isShared(false), sigma(0.05), bias(0.02), drop(0.f) { }
    string get_layer_name ();
    void get_layer_info ();
    void get_model_info ();
    void set_para (const int epoch, const int fix_round, const int max_round);
    int name, type;
    int idxs, idxd;
    int krow, kcol, strd;
    int flts, grps, grid, hole, act;
    bool isLoad, isFixed;
    bool isNorm, isShared;
    float sigma, bias, drop;
};



template <typename XPU>
class LayerBase {
public:
    explicit LayerBase (ParaLayer& pl, const int did, Tensor<XPU, float>& src, Tensor<XPU, float>& dst) :
        pl_(pl), did_(did), src_(src), dst_(dst) { }
    virtual ~LayerBase () { }
    virtual void fprop (const bool is_train, const bool is_fixed) = 0;
    virtual void bprop (const bool is_prop_grad) = 0;
public:
    virtual void init_layer () { }
    virtual void init_model () { }
    virtual void save_model (const string file) { }
    virtual void load_model (const string file) { }
    virtual void set_data (Tensor<XPU, float>& data) { data = src_; }
    virtual void set_anno (Tensor<XPU, float>& anno) { anno = dst_; }
    virtual void set_pred (Tensor<XPU, float>& pred) { pred = dst_; }
    virtual void set_eval (Tensor<XPU, float>& eval) { }
    virtual void set_optimization (ParaOptim& paraWmat, ParaOptim& paraNorm, vector<std::shared_ptr<OptimBase<XPU, float>>>& optims) { }
#ifdef __CUDACC__
    virtual cudaStream_t  get_calc_stream () const { return dnnCtx[did_].stream_; }
    virtual cudnnHandle_t get_cunn_handle () const { return dnnCtx[did_].cudnn_; }
#endif
    ParaLayer& pl_;
    int did_;
    Tensor<XPU, float> &src_, tsrc_;
    Tensor<XPU, float> &dst_, tdst_;
};

#define LAYER_CONSTRUCTOR(layername) \
    explicit layername (ParaLayer& pl, const int did, Tensor<XPU, float>& src, Tensor<XPU, float>& dst) \
    : LayerBase<XPU> (pl, did, src, dst), pl_(pl), did_(did), src_(src), dst_(dst), rand_(did), alpha(1.), beta(0.) \
    { init_layer (); }

#define LAYER_FUNC() \
    void fprop (const bool is_train, const bool is_fixed); \
    void bprop (const bool is_prop_grad); \
    void init_layer ()

#define MODEL_FUNC() \
    void init_model (); \
    void save_model (const string file); \
    void load_model (const string file); \
    void set_optimization (ParaOptim& paraWmat, ParaOptim& paraBias, vector<std::shared_ptr<OptimBase<XPU, float>>>& optims)

#define CUDNN_HANDLE LayerBase<XPU>::get_cunn_handle()
#define CUDNN_STREAM LayerBase<XPU>::get_calc_stream()

#define LAYER_MEMBER \
    ParaLayer& pl_; \
    int did_; \
    Tensor<XPU, float> &src_, tsrc_; \
    Tensor<XPU, float> &dst_, tdst_; \
    Tensor<XPU, float> drep_, mask_; \
    Tensor<XPU, float> aux_, mid_, mul_; \
    Random<XPU, float> rand_; \
    float alpha, beta; \
    int rows_, cols_, chls_; \
    int area_, dims_, nums_; \
    int flts_, grps_, grid_; \
    int chlg_, fltg_; \
    cudnnTensorDescriptor_t srcDesc_, dstDesc_

#define MODEL_MEMBER \
    Tensor<XPU, float> rmat_, mmat_; \
    Tensor<XPU, float> wmat_, gmat_; \
    Tensor<XPU, float> bias_, gias_

template <typename XPU>
class LayerAct : public LayerBase<XPU> {
public:
    LAYER_CONSTRUCTOR (LayerAct);
    LAYER_FUNC ();
    MODEL_FUNC ();
private:
    LAYER_MEMBER;
    MODEL_MEMBER;
    Tensor<XPU, float> mavg_, mvar_;
    Tensor<XPU, float> savg_, svar_, idev_;
    cudnnTensorDescriptor_t nrmDesc_;
};

template <typename XPU>
class LayerConv : public LayerBase<XPU> {
public:
    LAYER_CONSTRUCTOR (LayerConv);
    LAYER_FUNC ();
    MODEL_FUNC ();
private:
    LAYER_MEMBER;
    MODEL_MEMBER;
    Kernal kernal_;
    Tensor<XPU, float> tcol_;
    cudnnFilterDescriptor_t wmatDesc_;
    cudnnTensorDescriptor_t biasDesc_;
    vector<cudnnConvolutionDescriptor_t> convDesc_;
    vector<cudnnConvolutionFwdAlgo_t> fwdDataAlgo;
    vector<cudnnConvolutionBwdDataAlgo_t> bwdDataAlgo;
    vector<cudnnConvolutionBwdFilterAlgo_t> bwdFltrAlgo;
    vector<size_t> fwdDataSize;
    vector<size_t> bwdDataSize;
    vector<size_t> bwdFltrSize;
};

template <typename XPU>
class LayerPool : public LayerBase<XPU> {
public:
    LAYER_CONSTRUCTOR (LayerPool);
    LAYER_FUNC ();
private:
    LAYER_MEMBER;
    cudnnPoolingDescriptor_t poolDesc_;
};

template <typename XPU>
class LayerLoss : public LayerBase<XPU> {
public:
    LAYER_CONSTRUCTOR (LayerLoss);
    LAYER_FUNC ();
    void set_pred (Tensor<XPU, float>& pred) { pred = src_; }
    void set_eval (Tensor<XPU, float>& eval) { eval = mid_; }
private:
    LAYER_MEMBER;
};



template<typename XPU>
std::shared_ptr<LayerBase<XPU>> create_layer (ParaLayer& pl, const int did, Tensor<XPU, float>& src, Tensor<XPU, float>& dst);

class ParaNNet {
public:
    explicit ParaNNet () { }
    void config (const libconfig::Config& cfg);
    int get_layer_type (const char* type);
public:
    vector<ParaLayer> paraLayer_;
    vector<ParaOptim> paraWmat_;
    vector<ParaOptim> paraBias_;
    vector<ParaOptim> paraNorm_;
    BufferFormat format_;
    ParaFileData train_;
    ParaFileData infer_;
    ParaModel model_;
    int num_nnets;
    int num_nodes;
    int num_evals;
    int min_device;
    int max_device;
    int num_device;
    int num_layers;
    int stt_round;
    int end_round;
    int max_round;
    int now_round;
};

class NNetResult {
public:
    NNetResult () {}
    NNetResult (const float prob, const int cat) : prob_(prob), cat_(cat) { }
    bool operator< (const NNetResult& other) const { return prob_ > other.prob_; }
    float prob_;
    int cat_;
};

template <typename XPU>
class NNetModel {
public:
    void init_model();
    void init_data ();
    void terminate ();
    void train ();
    void infer ();
    void infer_serve ();
    void infer_uchar (const vector<unsigned char>& bytes, const int topK, vector<NNetResult>& result);
    void save_model (const int nid);
    void load_model (const int nid);
private:
    void fprop (const int nid, const bool is_train);
    void bprop (const int nid);
    void train_epoch (TensorBuffer<float>& buffer, TensorBatch<XPU, float>& batch, const int nid);
    void valid_epoch (TensorBuffer<float>& buffer, TensorBatch<XPU, float>& batch, const int nid);
    void infer_epoch (TensorBuffer<float>& buffer, TensorBatch<XPU, float>& batch, const int nid);
public:
    ParaNNet para_;
    vector<float> terr_, perr_;
    vector<vector<std::shared_ptr<LayerBase<XPU>>>> layers_;
    vector<vector<std::shared_ptr<OptimBase<XPU, float>>>> optims_;
    vector<vector<Tensor<XPU, float>>> nodes_;
    vector<TensorBatch<XPU, float>> batch_;
    std::deque<TensorBuffer<float>> train_;
    std::deque<TensorBuffer<float>> infer_;
};

#endif
